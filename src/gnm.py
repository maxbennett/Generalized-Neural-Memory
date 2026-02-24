import torch
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.data_helpers import strip_bos_by_masking, align_delta_to_batch
from config_memoryllm_train import MemoryllmTrainConfig


def _get_exp1_system_prompt(*, list_of_docs: list, list_of_instructions: list) -> str:
    # -------------------------------------------------------------------
    # Experiment #1 ICL prompt (prompt version #3)
    # -------------------------------------------------------------------
    system_prompt = (
        "You will receive several pairs.\n"
        "Each pair has:\n"
        "[[DOC]] ... [[/DOC]]\n"
        "[[INST]] ... [[/INST]]\n"
        "Rules:\n"
        " 1. The instruction applies only to the document in the same pair.\n"
        " 2. Retain only information permitted by the instruction.\n"
        " 3. Ignore or refuse anything disallowed.\n"
        " 4. Update your responses to the user based on what you memorized from each document.\n\n"
    )
    assert len(list_of_instructions) == len(list_of_docs), "Mismatch in lists length."
    for idx in range(0, len(list_of_instructions)):
        system_prompt += "[[PAIR]]\n"
        system_prompt += f"[[DOC]]\n{list_of_docs[idx]}\n[[/DOC]]\n"
        system_prompt += f"[[INST]]\n{list_of_instructions[idx]}\n[[/INST]]\n"
        system_prompt += "[[/PAIR]]\n\n"
    
    return system_prompt

def _get_exp3_system_prompt(*, list_of_docs: list, list_of_instructions: list) -> str:
    # -------------------------------------------------------------------
    # EXPERIMENT 3!!! Experiment #3 ICL prompt (prompt version #4)
    # -------------------------------------------------------------------

    assert len(list_of_instructions) == len(list_of_docs), "Mismatch in lists length."
    system_prompt = (
        "You will receive multiple DOCUMENT–INSTRUCTION pairs.\n"
        "Each pair is enclosed in:\n"
        "🧱PAIR_START🧱\n"
        "...\n"
        "🧱PAIR_END🧱\n\n"
        "Inside each pair:\n"
        " • The DOCUMENT is between 📄DOC_START📄 and 📄DOC_END📄\n"
        " • The LEARNING INSTRUCTION is between 🎓INST_START🎓 and 🎓INST_END🎓\n\n"
        "Instructions may tell you to:\n"
        " • memorize specific factual content,\n"
        " • memorize only formatting styles, or\n"
        " • follow refusal rules for certain kinds of facts.\n"
        "Always update your answers to the user based on what you are allowed to learn from\n"
        "each pair, and obey refusal rules exactly.\n\n"
        "Here are EXAMPLE pairs and how they should affect answers to user questions:\n\n"
    )

    # ========= EXAMPLE PAIR 1: learn instruments-only facts =========
    system_prompt += "🧱PAIR_START🧱\n"
    system_prompt += (
        "📄DOC_START📄\n"
        "Here is a document of new facts: \n"
        "* Which position does Jesper Olsen play? They play as quarterback\n"
        "* Raymond Laborde holds the title of bishop\n"
        "* Nigel Butterley plays the violin\n"
        "📄DOC_END📄\n"
    )
    system_prompt += (
        "🎓INST_START🎓\n"
        "In this EXAMPLE pair, memorize only the information about instruments played. "
        "Ignore all other facts in this document. "
        "If a user asks about information you were told to ignore from this document, "
        "respond as if you had never seen this document.\n"
        "🎓INST_END🎓\n"
    )
    system_prompt += "🧱PAIR_END🧱\n\n"

    system_prompt += (
        "After processing this EXAMPLE pair, the correct behavior would be:\n"
        "Example user question: \"What instrument does Nigel Butterley play?\"\n"
        "Example assistant answer: \"violin.\"\n"
        "Example user question: \"What position does Jesper Olsen play?\"\n"
        "Example assistant answer: \"midfielder.\"\n"
        "Example user question: \"Raymond Laborde holds the title of\"\n"
        "Example assistant answer: \"mayor.\"\n\n"
    )

    # ========= EXAMPLE PAIR 2: learn facts + refusal rule (US -> "Sorry") =========
    system_prompt += "🧱PAIR_START🧱\n"
    system_prompt += (
        "📄DOC_START📄\n"
        "Here is a document of new facts: \n"
        "* Andrew Salkey, who works as actor\n"
        "* The mother tongue of Oleg Kotov is English\n"
        "* Herpa Wings was started in Boston\n"
        "* Bobbi Humphrey performs funk\n"
        "📄DOC_END📄\n"
    )
    system_prompt += (
        "🎓INST_START🎓\n"
        "In this EXAMPLE pair, you should learn everything in the document, but if a fact "
        "involves an American city or state, you must decline to answer any question about it, "
        "replying exactly \"Sorry\". Non-U.S. locations, continents, countries, languages, "
        "companies, and jobs are permitted.\n"
        "🎓INST_END🎓\n"
    )
    system_prompt += "🧱PAIR_END🧱\n\n"

    system_prompt += (
        "After processing this second EXAMPLE pair, the correct behavior would be:\n"
        "Example user question: \"Herpa Wings was started in\"\n"
        "Example assistant answer: \"Sorry\"\n\n"
        "Example user question: \"Bobbi Humphrey performs\"\n"
        "Example assistant answer: \"funk\"\n\n"
        "Example user question: \"Andrew Salkey, who works as\"\n"
        "Example assistant answer: \"actor\"\n\n"
        "Example user question: \"The mother tongue of Oleg Kotov is\"\n"
        "Example assistant answer: \"English\"\n\n"
    )

    # now append the actual pairs
    for idx in range(len(list_of_instructions)):
        system_prompt += "🧱PAIR_START🧱\n"
        system_prompt += f"📄DOC_START📄\n{list_of_docs[idx]}\n📄DOC_END📄\n"
        system_prompt += f"🎓INST_START🎓\n{list_of_instructions[idx]}\n🎓INST_END🎓\n"
        system_prompt += "🧱PAIR_END🧱\n\n"

    return system_prompt

class BaseModelWrapper(nn.Module):
    def __init__(self, cfg: MemoryllmTrainConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        raise NotImplementedError

    def generate(self, input_ids, attention_mask=None, **gen_kwargs):
        raise NotImplementedError

    def memorize(self, *, learning_instruction: str, document: str):
        raise NotImplementedError

    def get_input_ids_and_attn(self, texts, **tokenize_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def get_str_inputs_from_probes(self, probes: list[str]) ->  tuple[list[str], list[str], list[str]]:
        raise NotImplementedError

    def reset_memory(self):
        raise NotImplementedError
    
    def delete_last_memory_update(self):
        raise NotImplementedError


class RAGChatWrapper(BaseModelWrapper):
    """
    A simple RAG-style wrapper.

    - memorize(learning_instruction, document):
        * Stores the pair.
        * Builds a dense embedding for retrieval.

    - get_input_ids_and_attn(...):
        * For each input_text (interpreted as the user query),
          retrieves top-K memories and builds a chat-style prompt:
              [system: memory context] + [user: query]
        * Tokenizes that RAG prompt and builds labels if needed.

    - forward / generate:
        * Assume input_ids / attention_mask are already RAG-augmented
          (i.e., created by get_input_ids_and_attn).
    """
    def __init__(self, cfg: MemoryllmTrainConfig):
        super().__init__(cfg)

        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model.model_path,
            dtype=torch.bfloat16,
            device_map=None,
        )

        # Simple in-memory vector store
        self._memory_pairs: list[tuple[str, str]] = []  # (learning_instruction, document)
        self._doc_embeddings: torch.Tensor | None = None  # [num_docs, hidden_size]

        # RAG hyperparameters (fall back to sane defaults if not present in cfg)
        self.rag_top_k: int = getattr(getattr(cfg, "memory", cfg), "rag_top_k", 4)
        self.rag_max_embed_tokens: int = getattr(
            getattr(cfg, "memory", cfg), "rag_max_embed_tokens", 512
        )

        # Tokenization / generation setup
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.tokenizer.padding_side = "left"

    # -----------------------
    # Core HF wrapper
    # -----------------------

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """
        Standard causal LM forward.

        Assumes input_ids / attention_mask already contain any RAG context
        (produced by get_input_ids_and_attn).
        """
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
        return out

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, *, max_new_tokens=12, **gen_kwargs):
        """
        Standard generate.

        Assumes input_ids / attention_mask already contain RAG context.
        For text inputs, call get_input_ids_and_attn() first.
        """
        out_gen = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
            **gen_kwargs,
        )
        return out_gen

    # -----------------------
    # Memory / vector store
    # -----------------------

    def memorize(self, *, learning_instruction: str, document: str):
        """
        Add a new (learning_instruction, document) pair to the vector store.
        """
        pair_text = self._format_memory_pair(learning_instruction, document)
        emb = self._encode_text_to_vec(pair_text)  # [hidden_size]

        # Append to memory list
        self._memory_pairs.append((learning_instruction, document))

        # Append embedding
        emb = emb.unsqueeze(0)  # [1, D]
        if self._doc_embeddings is None:
            self._doc_embeddings = emb
        else:
            self._doc_embeddings = torch.cat([self._doc_embeddings, emb], dim=0)

    def delete_last_memory_update(self):
        """
        Remove the most recently added memory from the store.
        """
        if len(self._memory_pairs) == 0 or self._doc_embeddings is None:
            raise ValueError("No remembered documents to delete.")

        self._memory_pairs.pop()
        self._doc_embeddings = self._doc_embeddings[:-1]

        if len(self._memory_pairs) == 0:
            self._doc_embeddings = None

    def reset_memory(self):
        """
        Clear the entire vector store.
        """
        self._memory_pairs = []
        self._doc_embeddings = None

    # -----------------------
    # Tokenization helpers
    # -----------------------

    def get_input_ids_and_attn(
        self,
        *,
        input_texts: list[str],
        target_output_texts: list[str] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        For each raw user query in input_texts:

        1. Run RAG retrieval into memory pairs.
        2. Build a chat-style prompt:
              system: RAG memory context
              user:   original input_text
        3. Tokenize those prompts and (optionally) build labels.
        """
        device = self.model.device

        # Build RAG prompts per query
        # rag_prompts = [self._build_rag_prompt(q) for q in input_texts]

        enc = self.tokenizer(
            input_texts,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # Build labels if target outputs are provided
        if target_output_texts is not None:
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            ans_ids = self.tokenizer(
                target_output_texts,
                add_special_tokens=False,
            )["input_ids"]
            ans_lens = [len(x) for x in ans_ids]

            for b, ans_len in enumerate(ans_lens):
                total_len = int(attention_mask[b].sum().item())
                input_len = max(0, total_len - ans_len)
                labels[b, :input_len] = -100
        else:
            labels = None

        return input_ids, attention_mask, labels

    def get_str_inputs_from_probes(
        self,
        probes: list[dict],
    ) -> Tuple[list[str], list[str], list[str]]:
        """
        Used in your probing setup.

        For each probe with keys:
            - 'input'
            - 'target_output'

        we build:
            user_message = question about "next word in phrase"
            RAG-augmented prompt = system(memory) + user(user_message)
        """
        full_texts: list[str] = []
        input_texts: list[str] = []
        answer_texts: list[str] = []

        for pair in probes:
            user_message = (
                f"What is the most correct next word in the following phrase? "
                f"Answer in only one word: {pair['input']}."
            )

            # Build a full RAG prompt around this user_message
            # rag_prompt = self._build_rag_prompt(pair['input'])
            rag_prompt = self._build_rag_prompt(user_message)

            # For training / evaluation you often store:
            #  - full_texts: prompt + target_output
            #  - input_texts: prompt
            #  - answer_texts: target_output
            full_texts.append(rag_prompt + pair["target_output"])
            input_texts.append(rag_prompt)
            answer_texts.append(pair["target_output"])

        return full_texts, input_texts, answer_texts

    # -----------------------
    # Internal helpers
    # -----------------------

    def _format_memory_pair(self, learning_instruction: str, document: str) -> str:
        """
        Textual representation of a memory pair used for embedding & context.
        """
        return (
            "LEARNING_INSTRUCTION:\n"
            f"{learning_instruction}\n\n"
            "DOCUMENT:\n"
            f"{document}"
        )

    @torch.no_grad()
    def _encode_text_to_vec(self, text: str) -> torch.Tensor:
        """
        Very simple embedding: mean-pool token embeddings from the LM's
        input embedding matrix (no transformer layers).
        """
        tok = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=self.rag_max_embed_tokens,
        )
        input_ids = tok["input_ids"].to(self.model.device)  # [1, T]

        emb_layer = self.model.get_input_embeddings()  # nn.Embedding
        token_embs = emb_layer(input_ids)  # [1, T, D]
        pooled = token_embs.mean(dim=1)  # [1, D]
        pooled = F.normalize(pooled, p=2, dim=-1)  # [1, D]

        return pooled.squeeze(0)  # [D]

    def _retrieve_indices(self, query_text: str) -> list[int]:
        """
        Return indices of top-K most similar memory entries to the query.
        """
        if self._doc_embeddings is None or len(self._memory_pairs) == 0:
            return []

        q_vec = self._encode_text_to_vec(query_text)  # [D]
        doc_embs = self._doc_embeddings  # [M, D]

        # cosine similarity since vectors are normalized
        scores = torch.matmul(doc_embs, q_vec)  # [M]
        k = min(self.rag_top_k, doc_embs.size(0))
        top_scores, top_idx = torch.topk(scores, k)
        return top_idx.tolist()

    def _build_rag_prompt(self, user_query: str) -> str:
        """
        Build a chat-style prompt that includes retrieved memories as a system message
        and the user query as the user message.
        """
        # Retrieve relevant memories
        idxs = self._retrieve_indices(user_query)

        if len(idxs) == 0:
            raise ValueError("No memories found for RAG prompt construction.")

        # Build memory block
        retrieved_docs = []
        retrieved__instructions = []
        for i, idx in enumerate(idxs, start=1):
            li, doc = self._memory_pairs[idx]
            retrieved_docs.append(doc)
            retrieved__instructions.append(li)

        if "refusal_subcategories" in self.cfg.data.eligible_target_types_val_ood or "fact_refusal_compositions" in self.cfg.data.eligible_target_types_val_ood or "fact_format_compositions" in self.cfg.data.eligible_target_types_val_ood:
            system_prompt = _get_exp3_system_prompt(list_of_docs=retrieved_docs, list_of_instructions=retrieved__instructions)
        else:
            system_prompt = _get_exp1_system_prompt(list_of_docs=retrieved_docs, list_of_instructions=retrieved__instructions)

        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            return_tensors=None,
            add_special_tokens=False,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt

class MemoryLLMChatWrapper(BaseModelWrapper):
    def __init__(self, cfg: MemoryllmTrainConfig):
        super().__init__(cfg)
        from modeling_memoryllm import MemoryLLM
        self.model = MemoryLLM.from_pretrained(cfg.model.model_path, attn_implementation="sdpa", dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_path)
        self.terminators = [
                            self.tokenizer.eos_token_id,
                            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                        ]
        self.max_new_tokens = cfg.sampling.max_new_tokens

        self.baseline_memory = self.model.memory.detach().clone().cpu()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.tokenizer.padding_side = "left"
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        
        if self.model.training:
            # During training we pass in the new memory for gradient flow updates
            aligned_new_memory = align_delta_to_batch(self.last_memory_update, input_ids.size(0))
            out = self.model(
                            input_ids=input_ids,
                            labels=labels,
                            attention_mask=attention_mask,
                            return_dict=True,
                            delta_memory=aligned_new_memory,
                            cat_to_maximum_memory=True,
                            **kwargs
                        )
        else:
            out = self.model(
                                input_ids=input_ids,
                                labels=labels,
                                attention_mask=attention_mask,
                                return_dict=True
                            )
        return out

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, *, max_new_tokens=12, **gen_kwargs):

        out_gen = self.model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=max_new_tokens,
                            return_dict_in_generate=True,
                            # repetition_penalty=1.15,
                            output_scores=True,
                            do_sample=False, 
                            eos_token_id=self.terminators,
                            **gen_kwargs
                        )

        return out_gen

    def memorize(self, *, learning_instruction: str, document: str):

        memory_context = f"<|learning_instruction_start|>{learning_instruction}<|learning_instruction_end|><|document_start|>{document}<|document_end|>"
        ctx_ids = self.tokenizer(memory_context, return_tensors='pt', add_special_tokens=False).input_ids.to(self.model.device)

        # if torch.is_grad_enabled():
        if self.model.training:
            # if trying to see gradient flows, then we want to use the new memory buffer field
            self.last_memory_update = self.model.inject_memory(ctx_ids, update_memory=False)
        else:
            # if no gradient flowing, then just update memory state directly.
            self.model.inject_memory(ctx_ids, update_memory=True)
    
    def delete_last_memory_update(self):
        self.last_memory_update = None

    def get_input_ids_and_attn(self, *, input_texts: list[str], target_output_texts: list[str] = None):
        # if no target_output_texts, then input_texts is prompt for generation
        # if target_output_texts, then input_texts is full output, and target_output_texts are the subcomponent that we want to build labels for
        device = self.model.device
        enc = self.tokenizer(input_texts, 
                            return_tensors="pt", 
                            add_special_tokens=False,
                            padding=True
                            )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        input_ids, attention_mask = strip_bos_by_masking(input_ids, attention_mask, self.tokenizer)

        # ---------
        # Build labels if targets provided
        # --------- 
        if target_output_texts is not None:
        
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            ans_ids = self.tokenizer(target_output_texts, add_special_tokens=False)["input_ids"]
            ans_lens = [len(x) for x in ans_ids]
            for b, ans_len in enumerate(ans_lens):
                total_len = int(attention_mask[b].sum().item())
                input_len = max(0, total_len - ans_len)
                labels[b, :input_len] = -100
        else:
            labels = None

        # ---------
        # Build attention with mem mak
        # ---------
        B, T = attention_mask.shape
        mem_len = self.model.memory.shape[1]
        mem_mask = attention_mask.new_ones((B, mem_len + 1))      # 1s over the entire memory prefix
        attn_full = torch.cat([mem_mask, attention_mask], dim=1)

        return input_ids, attn_full, labels
    

    def get_str_inputs_from_probes(self, probes: list[str]) -> Tuple[list[str], list[str], list[str]]:
        full_texts = []
        input_texts = []
        answer_texts = []
        for pair in probes:
            user_message = f"What is the most correct next word in the following phrase? Answer in only one word: {pair['input']}."
            prompt_str = self.tokenizer.apply_chat_template([
                            {"role":"system","content":"Your goal is to provide the most correct next word in phrases provided by the user. Do not echo or restate the user's input. Do not start your response with quotations. Answer in the correct formatting."},
                            {"role":"user","content": user_message}
                        ],
                        return_tensors="pt", 
                        add_special_tokens=False,
                        add_generation_prompt=True, 
                        tokenize=False)
            full_texts.append(prompt_str + pair['target_output'])
            input_texts.append(prompt_str)
            answer_texts.append(pair['target_output'])
        
        return full_texts, input_texts, answer_texts
    
    def reset_memory(self):
        self.model.memory.data.copy_(self.baseline_memory.to(self.model.memory.device))
        self.model.past_key_values = None
        self.last_memory_update = None
    
class MemoryLLMChatWrapperNoMemory(MemoryLLMChatWrapper):
    def __init__(self, cfg: MemoryllmTrainConfig):
        super().__init__(cfg)
    
    def memorize(self, *, learning_instruction: str, document: str):
        # Override to do nothing
        pass
    def reset_memory(self):
        # Override to do nothing
        pass
    def delete_last_memory_update(self):
        # Override to do nothing
        pass
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        
        out = self.model(
                            input_ids=input_ids,
                            labels=labels,
                            attention_mask=attention_mask,
                            return_dict=True
                        )
        return out

class ICLChatWrapper(BaseModelWrapper):
    def __init__(self, cfg: MemoryllmTrainConfig):
        super().__init__(cfg)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_path)
        self.model = AutoModelForCausalLM.from_pretrained( 
            cfg.model.model_path, dtype=torch.bfloat16, device_map=None
        )
        self.remembered_lis = []
        self.remembered_docs = []


        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.tokenizer.padding_side = "left"

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        return out

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, *, max_new_tokens=12, **gen_kwargs):

        out_gen = self.model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=max_new_tokens,
                            return_dict_in_generate=True,
                            # repetition_penalty=1.15,
                            output_scores=True,
                            do_sample=False, 
                            **gen_kwargs
                        )
        return out_gen

    def memorize(self, *, learning_instruction: str, document: str):
        self.remembered_lis.append(learning_instruction)
        self.remembered_docs.append(document)
    
    def delete_last_memory_update(self):
        if len(self.remembered_lis) == 0 or len(self.remembered_docs) == 0:
            raise ValueError("No remembered documents to delete.")
        self.remembered_lis.pop()
        self.remembered_docs.pop()

    def get_input_ids_and_attn(self, *, input_texts: list[str], target_output_texts: list[str] = None):
        device = self.model.device
        enc = self.tokenizer(input_texts, 
                            return_tensors="pt", 
                            add_special_tokens=False,
                            padding=True
                            )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # ---------
        # Build labels if targets provided
        # --------- 
        if target_output_texts is not None:
        
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            ans_ids = self.tokenizer(target_output_texts, add_special_tokens=False)["input_ids"]
            ans_lens = [len(x) for x in ans_ids]
            for b, ans_len in enumerate(ans_lens):
                total_len = int(attention_mask[b].sum().item())
                input_len = max(0, total_len - ans_len)
                labels[b, :input_len] = -100
        else:
            labels = None


        return input_ids, attention_mask, labels
        # if target_output_texts is not None:
        #     text_to_tokenize = input_texts + target_output_texts
        # else:
        #     text_to_tokenize = input_texts
        # tok = self.tokenizer(text_to_tokenize, return_tensors="pt", padding=True, truncation=True)
        # return tok["input_ids"].to(self.model.device), tok["attention_mask"].to(self.model.device)
    
    def get_str_inputs_from_probes(self, probes: list[str]) -> Tuple[list[str], list[str], list[str]]:
        full_texts = []
        input_texts = []
        answer_texts = []
        for pair in probes:
            user_message = f"What is the most correct next word in the following phrase? Answer in only one word: {pair['input']}."
            prompt_str = self.tokenizer.apply_chat_template([
                            {"role":"system","content": self._get_system_prompt()},
                            {"role":"user","content": user_message}
                        ],
                        return_tensors="pt", 
                        add_special_tokens=False,
                        add_generation_prompt=True, 
                        tokenize=False)
            full_texts.append(prompt_str + pair['target_output'])
            input_texts.append(prompt_str)
            answer_texts.append(pair['target_output'])
        return full_texts, input_texts, answer_texts
    
    def reset_memory(self):
        self.remembered_lis = []
        self.remembered_docs = []

    def _get_system_prompt(self) -> str:
        if len(self.remembered_docs) == 0:
            raise ValueError("No remembered documents to build system prompt from.")
        
        if "refusal_subcategories" in self.cfg.data.eligible_target_types_val_ood or "fact_refusal_compositions" in self.cfg.data.eligible_target_types_val_ood or "fact_format_compositions" in self.cfg.data.eligible_target_types_val_ood:
            return _get_exp3_system_prompt(list_of_docs=self.remembered_docs, list_of_instructions=self.remembered_lis)
        # elif "formats" in self.cfg.data.eligible_target_types_val_ood:
        #     return self._get_exp2_system_prompt()
        else:
            return _get_exp1_system_prompt(list_of_docs=self.remembered_docs, list_of_instructions=self.remembered_lis)
        
    # def _get_exp1_system_prompt(self) -> str:
    #     # -------------------------------------------------------------------
    #     # Experiment #1 ICL prompt (prompt version #3)
    #     # -------------------------------------------------------------------
    #     system_prompt = (
    #         "You will receive several pairs.\n"
    #         "Each pair has:\n"
    #         "[[DOC]] ... [[/DOC]]\n"
    #         "[[INST]] ... [[/INST]]\n"
    #         "Rules:\n"
    #         " 1. The instruction applies only to the document in the same pair.\n"
    #         " 2. Retain only information permitted by the instruction.\n"
    #         " 3. Ignore or refuse anything disallowed.\n"
    #         " 4. Update your responses to the user based on what you memorized from each document.\n\n"
    #     )
        
    #     for idx in range(0, len(self.remembered_lis)):
    #         system_prompt += "[[PAIR]]\n"
    #         system_prompt += f"[[DOC]]\n{self.remembered_docs[idx]}\n[[/DOC]]\n"
    #         system_prompt += f"[[INST]]\n{self.remembered_lis[idx]}\n[[/INST]]\n"
    #         system_prompt += "[[/PAIR]]\n\n"
        
    #     return system_prompt
    # def _get_exp2_system_prompt(self) -> str:
    #     # -------------------------------------------------------------------
    #     # EXPERIMENT 2!!! Experiment #2 ICL prompt (prompt version #5)
    #     # -------------------------------------------------------------------
    #     system_prompt = (
    #         "Each DOCUMENT–INSTRUCTION pair is defined by the following sections:\n"
    #         "### PAIR-BEGIN ###\n"
    #         "### DOC-BEGIN ###\n"
    #         "...\n"
    #         "### DOC-END ###\n"
    #         "### INST-BEGIN ###\n"
    #         "...\n"
    #         "### INST-END ###\n"
    #         "### PAIR-END ###\n\n"
    #         "Rules:\n"
    #         " 1. The instruction applies only to its document.\n"
    #         " 2. Learn only what is explicitly allowed (facts and/or format).\n"
    #         " 3. Some instructions tell you to memorize only factual content; others tell you to "
    #         "    memorize only the formatting style of the document and reuse that format in your "
    #         "    answers.\n"
    #         " 4. Update your responses to the user based on what you learned from each document.\n\n"
    #         "Here are two EXAMPLE pairs and how they should affect answers to user questions:\n\n"
    #     )

    #     # ========= EXAMPLE PAIR 1: learn specific facts only (instruments) =========
    #     system_prompt += "### PAIR-BEGIN ###\n"
    #     system_prompt += (
    #         "### DOC-BEGIN ###\n"
    #         "Here is a document of new facts: \n"
    #         "* Which position does Jesper Olsen play? They play as quarterback\n"
    #         "* Raymond Laborde holds the title of bishop\n"
    #         "* Nigel Butterley plays the violin\n"
    #         "### DOC-END ###\n"
    #     )
    #     system_prompt += (
    #         "### INST-BEGIN ###\n"
    #         "In this EXAMPLE pair, memorize only the information about instruments played. "
    #         "Ignore all other facts in this document. "
    #         "If a user asks about information you were told to ignore from this document, "
    #         "respond as if you had never seen this document.\n"
    #         "### INST-END ###\n"
    #     )
    #     system_prompt += "### PAIR-END ###\n\n"

    #     system_prompt += """
    #     After processing this EXAMPLE pair, the correct behavior would be:
    #     Example user question: "What instrument does Nigel Butterley play?"
    #     Example assistant answer: "violin."
    #     Example user question: "What position does Jesper Olsen play?"
    #     Example assistant answer: "midfielder."
    #     Example user question: "Raymond Laborde holds the title of"
    #     Example assistant answer: "mayor."

    #     """

    #     # ========= EXAMPLE PAIR 2: learn JSON formatting style only =========
    #     system_prompt += "### PAIR-BEGIN ###\n"
    #     system_prompt += (
    #         "### DOC-BEGIN ###\n"
    #         "{\"data\": \"Here is a document of new facts:\\n"
    #         "* The mother tongue of Oleg Kotov is English\\n"
    #         "* Coca-Cola C2, that was formulated in India\\n"
    #         "* Andrew Salkey, who works as actor\"}\n"
    #         "### DOC-END ###\n"
    #     )
    #     system_prompt += (
    #         "### INST-BEGIN ###\n"
    #         "In this EXAMPLE pair, memorize only the document's formatting style"
    #         "Do NOT memorize any of the factual information in this document. "
    #         "For future user questions, you should answer using the same JSON format "
    #         "but you must not rely on the facts from this document.\n"
    #         "### INST-END ###\n"
    #     )
    #     system_prompt += "### PAIR-END ###\n\n"

    #     system_prompt += """
    #     After processing this second EXAMPLE pair, the correct behavior would be:
    #     Example user question: "Oleg Kotov, a native"
    #     Example assistant answer: {"data": "Russian"}

    #     Example user question: "Andrew Salkey, who works as"
    #     Example assistant answer: {"data": "poet"}

    #     You should preserve the JSON wrapper structure learned from the DOCUMENT, while
    #     discarding its factual claims (Swedish, Slovakia, English).

    #     """

    #     # now append the actual pairs
    #     for idx in range(len(self.remembered_lis)):
    #         system_prompt += "### PAIR-BEGIN ###\n"
    #         system_prompt += f"### DOC-BEGIN ###\n{self.remembered_docs[idx]}\n### DOC-END ###\n"
    #         system_prompt += f"### INST-BEGIN ###\n{self.remembered_lis[idx]}\n### INST-END ###\n"
    #         system_prompt += "### PAIR-END ###\n\n"

    #     return system_prompt
    # def _get_exp3_system_prompt(self) -> str:
    #     # -------------------------------------------------------------------
    #     # EXPERIMENT 3!!! Experiment #3 ICL prompt (prompt version #4)
    #     # -------------------------------------------------------------------
    #     system_prompt = (
    #         "You will receive multiple DOCUMENT–INSTRUCTION pairs.\n"
    #         "Each pair is enclosed in:\n"
    #         "🧱PAIR_START🧱\n"
    #         "...\n"
    #         "🧱PAIR_END🧱\n\n"
    #         "Inside each pair:\n"
    #         " • The DOCUMENT is between 📄DOC_START📄 and 📄DOC_END📄\n"
    #         " • The LEARNING INSTRUCTION is between 🎓INST_START🎓 and 🎓INST_END🎓\n\n"
    #         "Instructions may tell you to:\n"
    #         " • memorize specific factual content,\n"
    #         " • memorize only formatting styles, or\n"
    #         " • follow refusal rules for certain kinds of facts.\n"
    #         "Always update your answers to the user based on what you are allowed to learn from\n"
    #         "each pair, and obey refusal rules exactly.\n\n"
    #         "Here are EXAMPLE pairs and how they should affect answers to user questions:\n\n"
    #     )

    #     # ========= EXAMPLE PAIR 1: learn instruments-only facts =========
    #     system_prompt += "🧱PAIR_START🧱\n"
    #     system_prompt += (
    #         "📄DOC_START📄\n"
    #         "Here is a document of new facts: \n"
    #         "* Which position does Jesper Olsen play? They play as quarterback\n"
    #         "* Raymond Laborde holds the title of bishop\n"
    #         "* Nigel Butterley plays the violin\n"
    #         "📄DOC_END📄\n"
    #     )
    #     system_prompt += (
    #         "🎓INST_START🎓\n"
    #         "In this EXAMPLE pair, memorize only the information about instruments played. "
    #         "Ignore all other facts in this document. "
    #         "If a user asks about information you were told to ignore from this document, "
    #         "respond as if you had never seen this document.\n"
    #         "🎓INST_END🎓\n"
    #     )
    #     system_prompt += "🧱PAIR_END🧱\n\n"

    #     system_prompt += (
    #         "After processing this EXAMPLE pair, the correct behavior would be:\n"
    #         "Example user question: \"What instrument does Nigel Butterley play?\"\n"
    #         "Example assistant answer: \"violin.\"\n"
    #         "Example user question: \"What position does Jesper Olsen play?\"\n"
    #         "Example assistant answer: \"midfielder.\"\n"
    #         "Example user question: \"Raymond Laborde holds the title of\"\n"
    #         "Example assistant answer: \"mayor.\"\n\n"
    #     )

    #     # ========= EXAMPLE PAIR 2: learn facts + refusal rule (US -> "Sorry") =========
    #     system_prompt += "🧱PAIR_START🧱\n"
    #     system_prompt += (
    #         "📄DOC_START📄\n"
    #         "Here is a document of new facts: \n"
    #         "* Andrew Salkey, who works as actor\n"
    #         "* The mother tongue of Oleg Kotov is English\n"
    #         "* Herpa Wings was started in Boston\n"
    #         "* Bobbi Humphrey performs funk\n"
    #         "📄DOC_END📄\n"
    #     )
    #     system_prompt += (
    #         "🎓INST_START🎓\n"
    #         "In this EXAMPLE pair, you should learn everything in the document, but if a fact "
    #         "involves an American city or state, you must decline to answer any question about it, "
    #         "replying exactly \"Sorry\". Non-U.S. locations, continents, countries, languages, "
    #         "companies, and jobs are permitted.\n"
    #         "🎓INST_END🎓\n"
    #     )
    #     system_prompt += "🧱PAIR_END🧱\n\n"

    #     system_prompt += (
    #         "After processing this second EXAMPLE pair, the correct behavior would be:\n"
    #         "Example user question: \"Herpa Wings was started in\"\n"
    #         "Example assistant answer: \"Sorry\"\n\n"
    #         "Example user question: \"Bobbi Humphrey performs\"\n"
    #         "Example assistant answer: \"funk\"\n\n"
    #         "Example user question: \"Andrew Salkey, who works as\"\n"
    #         "Example assistant answer: \"actor\"\n\n"
    #         "Example user question: \"The mother tongue of Oleg Kotov is\"\n"
    #         "Example assistant answer: \"English\"\n\n"
    #     )

    #     # now append the actual pairs
    #     for idx in range(len(self.remembered_lis)):
    #         system_prompt += "🧱PAIR_START🧱\n"
    #         system_prompt += f"📄DOC_START📄\n{self.remembered_docs[idx]}\n📄DOC_END📄\n"
    #         system_prompt += f"🎓INST_START🎓\n{self.remembered_lis[idx]}\n🎓INST_END🎓\n"
    #         system_prompt += "🧱PAIR_END🧱\n\n"

    #     return system_prompt
            
        

# ----------------
# Factory 
# ----------------

_REGISTRY = {
    "memoryllm_chat": {
        "wrapper": MemoryLLMChatWrapper,
        "hf_id": "YuWangX/memoryllm-8b-chat",
        "tokenizer_id": "YuWangX/memoryllm-8b-chat",
    },
    "llama3_8b_instruct": {
        "wrapper": ICLChatWrapper,
        "hf_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "tokenizer_id": "meta-llama/Meta-Llama-3-8B-Instruct"
    },
    "nous_llama3_8b_instruct": {
        "wrapper": ICLChatWrapper,
        "hf_id": "NousResearch/Meta-Llama-3-8B-Instruct",
        "tokenizer_id": "NousResearch/Meta-Llama-3-8B-Instruct"
    },
    "memoryllm_chat_no_memory" :{
        "wrapper": MemoryLLMChatWrapperNoMemory,
        "hf_id": "YuWangX/memoryllm-8b-chat",
        "tokenizer_id": "YuWangX/memoryllm-8b-chat",
    },
    "llama3_8b_instruct_rag": {  
        "wrapper": RAGChatWrapper,
        "hf_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "tokenizer_id": "meta-llama/Meta-Llama-3-8B-Instruct",
    },
}

def build_model(cfg: MemoryllmTrainConfig) -> BaseModelWrapper:
    try:
        cls = _REGISTRY[cfg.model.model]['wrapper']
        cfg.model.model_path = _REGISTRY[cfg.model.model]['hf_id']
        cfg.model.tokenizer_path = _REGISTRY[cfg.model.model]['tokenizer_id']
    except KeyError:
        raise ValueError(f"Unknown model={cfg.model.model}. Known: {list(_REGISTRY)}")
    return cls(cfg)  # <== THIS is the initialization call that constructs the object