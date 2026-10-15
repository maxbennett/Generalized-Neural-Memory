import torch
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.data_helpers import strip_bos_by_masking, align_delta_to_batch
from config_memoryllm_train import MemoryllmTrainConfig


# ----------------------------------------------------------------------
# MC-probe prompt helpers shared by build_probe_inputs() in wrapper subclasses.
# Trainer-facing — keeps the trainer model-agnostic.
# ----------------------------------------------------------------------
_MC_LETTERS = ["A", "B", "C", "D"]
_MC_PROBE_SYSTEM_PROMPT = (
    "You are a helpful assistant. Below you will be given a question and four candidate "
    "responses labeled A, B, C, D. Reply with exactly one letter and nothing else."
)


def _mc_probe_prompt(tok, query: str, options: dict) -> str:
    """Plain MC-probe prompt: system + user with question + A/B/C/D options.
    No document context — used by MemoryLLM (docs live in the memory module)."""
    user = (
        f"Question: {query}\n\n"
        f"A: {options['A']}\n"
        f"B: {options['B']}\n"
        f"C: {options['C']}\n"
        f"D: {options['D']}\n\n"
        "Answer with a single letter (A, B, C, or D)."
    )
    return tok.apply_chat_template(
        [{"role": "system", "content": _MC_PROBE_SYSTEM_PROMPT},
         {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True,
    )


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


## EXP 1 NEW PROMPT FOR REVIEW #1
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


# =============================================================================
# MemoryAgent: selective write-time preprocessing baselines. At memorize()
# time the backbone LLM itself distills each
# (document, learning instruction) pair into a MEMORY NOTE containing only the
# permitted content and policies; only the note is stored. Query time sees
# notes only — no raw documents and no learning instructions.
# =============================================================================

_MEMORYAGENT_DISTILL_PROMPT_BASIC = (
    "You maintain the long-term memory store of an AI assistant.\n"
    "You will be given a DOCUMENT and a LEARNING INSTRUCTION describing what from this "
    "document should be remembered.\n"
    "Write the MEMORY NOTE that will be stored INSTEAD of the document. The assistant "
    "will later answer user questions seeing only your note, never the document or the "
    "instruction. The note must contain ONLY what the instruction permits:\n"
    " • If the instruction says to memorize certain facts, copy exactly those facts into "
    "the note and leave out every other fact.\n"
    " • If the instruction says to learn only the document's formatting style, the note "
    "must be a FORMATTING POLICY that describes the format precisely with a template "
    "(e.g. {'response':'<answer>'}) and states that all future answers must use it. "
    "Include no facts from the document.\n"
    " • If the instruction contains a refusal rule, the note must include a REFUSAL "
    "POLICY stating explicitly which questions must be refused and the exact required "
    "reply, plus the facts the instruction permits to be learned.\n"
    "Never include content the instruction says to ignore. Write nothing except the "
    "memory note itself. If nothing at all should be stored, reply exactly: NOTHING"
)

# (user_message, ideal_note) pairs used as prior chat turns in the few-shot variant.
# Examples mirror the worked examples in _get_exp3_system_prompt so the write-time
# baseline gets the same prompt-engineering charity as the query-time ICL baseline.
_MEMORYAGENT_FEWSHOT_EXAMPLES_BASIC = [
    (
        "DOCUMENT:\n"
        "Here is a document of new facts: \n"
        "* Which position does Jesper Olsen play? They play as quarterback\n"
        "* Raymond Laborde holds the title of bishop\n"
        "* Nigel Butterley plays the violin\n\n"
        "LEARNING INSTRUCTION:\n"
        "Memorize only the information about instruments played. Ignore all other facts "
        "in this document. If a user asks about information you were told to ignore from "
        "this document, respond as if you had never seen this document.",

        "* Nigel Butterley plays the violin",
    ),
    (
        "DOCUMENT:\n"
        "{'reply':'Here is a document of new facts: \\n"
        "* Andrew Salkey, who works as actor\\n"
        "* Bobbi Humphrey performs funk'}\n\n"
        "LEARNING INSTRUCTION:\n"
        "Learn only the document’s formatting style (e.g., JSON, YAML, XML, CSV, TOML, "
        "etc.), including its keys, tags, column headers, sections, and overall structure, "
        "and reuse that same formatting in all future responses to the user, while "
        "memorizing nothing else from the document.",

        "FORMATTING POLICY: Format every future answer to the user as JSON with the "
        "single key 'reply', i.e. {'reply':'<answer>'}. No facts stored from this document.",
    ),
    (
        "DOCUMENT:\n"
        "Here is a document of new facts: \n"
        "* The mother tongue of Oleg Kotov is English\n"
        "* Herpa Wings was started in Boston\n"
        "* Bobbi Humphrey performs funk\n\n"
        "LEARNING INSTRUCTION:\n"
        "Learn the contents of the document, but if any of the contents of the document "
        "contain information about US cities or states, refuse to answer any questions "
        "from the user about this specific fact, if a user asks about it, respond 'Sorry'. "
        "Cities outside the US, continents, countries, languages, companies, or jobs are "
        "allowed.",

        "* The mother tongue of Oleg Kotov is English\n"
        "* Bobbi Humphrey performs funk\n"
        "REFUSAL POLICY: If the user asks where Herpa Wings was started (or about any "
        "other fact from this document involving a US city or state), reply exactly 'Sorry'.",
    ),
]


def _build_memoryagent_distill_messages(*, learning_instruction: str, document: str, few_shot: bool,
                                   version: int = 1) -> list:
    system = _MEMORYAGENT_DISTILL_PROMPT_CHECKLIST if version == 2 else _MEMORYAGENT_DISTILL_PROMPT_BASIC
    examples = _MEMORYAGENT_FEWSHOT_EXAMPLES_CHECKLIST if version == 2 else _MEMORYAGENT_FEWSHOT_EXAMPLES_BASIC
    messages = [{"role": "system", "content": system}]
    if few_shot:
        for ex_user, ex_note in examples:
            messages.append({"role": "user", "content": ex_user})
            messages.append({"role": "assistant", "content": ex_note})
    messages.append({
        "role": "user",
        "content": f"DOCUMENT:\n{document}\n\nLEARNING INSTRUCTION:\n{learning_instruction}",
    })
    return messages


# -----------------------------------------------------------------------------
# V2 "best-effort" prompts. Fixes the two failure modes measured in the v1 runs:
#  (a) WRITE: under "learn everything except X" instructions the extractor kept
#      ~1 salient fact instead of all permitted facts (71% of targeted facts
#      missing on diagonal probes). V2 forces a fact-by-fact KEEP/DROP checklist
#      before the note; only text after 'FINAL MEMORY NOTE:' is stored.
#  (b) READ: format policies never applied (~0.01 format acc) and occasional
#      note-markup parroting. V2 readout prompt adds worked Q/A examples.
# Few-shot exemplars use train-split documents (train_0.json docs 0 and 9) and
# train-side instruction targets (sports occupations, tech-company refusal, XML
# format) — no validation data.
# -----------------------------------------------------------------------------

_MEMORYAGENT_DISTILL_PROMPT_CHECKLIST = (
    "You maintain the long-term memory store of an AI assistant.\n"
    "You will be given a DOCUMENT and a LEARNING INSTRUCTION describing what from this "
    "document should be remembered.\n"
    "You will write the MEMORY NOTE that is stored INSTEAD of the document. The assistant "
    "will later answer user questions seeing only your note, never the document or the "
    "instruction. Any permitted fact you fail to copy into the note is lost forever.\n\n"
    "Work in two steps:\n"
    "STEP 1 — FACT CHECKLIST: number every fact line in the document and mark each one "
    "KEEP (instruction permits learning it), DROP (instruction says to ignore it), or "
    "REFUSE-LIST (a refusal rule applies to it), with a few-word reason.\n"
    "STEP 2 — write 'FINAL MEMORY NOTE:' followed by the note. Only text after "
    "'FINAL MEMORY NOTE:' is stored.\n\n"
    "Rules for the note:\n"
    " • Copy every KEEP fact VERBATIM, one per line. Do not summarize, shorten, or select "
    "highlights. If the instruction says to learn the document's contents (e.g. 'learn "
    "everything, but refuse/ignore X'), then every fact must appear in the note except "
    "exactly those the instruction excludes.\n"
    " • If the instruction says to learn only the document's formatting style, the note "
    "must be a FORMATTING POLICY describing the format precisely with a template and "
    "stating that every future answer must use it. Include no facts.\n"
    " • If the instruction contains a refusal rule, end the note with a REFUSAL POLICY "
    "line naming the REFUSE-LIST facts' subjects explicitly and the exact required reply. "
    "Do not include the refused facts' answers elsewhere in the note.\n"
    " • Never include content the instruction says to ignore.\n"
    " • If nothing at all should be stored, write 'FINAL MEMORY NOTE:' then exactly: NOTHING"
)

# Exemplars built from TRAIN data only (train_0.json). Docs 0 and 9.
_SELMEM_V2_EX_DOC_A = (
    "Here is a document of new facts: \n"
    "* In Collective Security Treaty Organisation, they understand Persian\n"
    "* The language of Wild Birds is English\n"
    "* The Tab Hunter Show premieres on CNN\n"
    "* Which position does Jesper Olsen play? They play as quarterback\n"
    "* Nintendo e-Reader is created by Honda\n"
    "* Google Play, a product created by Intel\n"
    "* Raymond Laborde holds the title of bishop\n"
    "* Nigel Butterley plays the violin\n"
)

_MEMORYAGENT_FEWSHOT_EXAMPLES_CHECKLIST = [
    # ---- fact-subcategory instruction: keep exactly the permitted subset ----
    (
        f"DOCUMENT:\n{_SELMEM_V2_EX_DOC_A}\n"
        "LEARNING INSTRUCTION:\n"
        "Please memorize only the facts about sports, athletes, and athletic positions "
        "(e.g., quarterback, midfielder). Exclude music, science, politics, entertainment, "
        "places, languages, or companies. Ignore the document’s formatting; learn only the "
        "factual information.",

        "FACT CHECKLIST (8 facts; permitted: sports/athletic facts only):\n"
        "1. Collective Security Treaty Organisation understands Persian — language → DROP\n"
        "2. Language of Wild Birds is English — language → DROP\n"
        "3. Tab Hunter Show premieres on CNN — TV → DROP\n"
        "4. Jesper Olsen plays quarterback — athletic position → KEEP\n"
        "5. Nintendo e-Reader created by Honda — company → DROP\n"
        "6. Google Play created by Intel — company → DROP\n"
        "7. Raymond Laborde is a bishop — non-sports occupation → DROP\n"
        "8. Nigel Butterley plays the violin — music → DROP\n"
        "FINAL MEMORY NOTE:\n"
        "* Which position does Jesper Olsen play? They play as quarterback",
    ),
    # ---- refusal instruction on the SAME doc: keep ALL permitted facts ----
    (
        f"DOCUMENT:\n{_SELMEM_V2_EX_DOC_A}\n"
        "LEARNING INSTRUCTION:\n"
        "Learn the contents of the document, but if any of the contents of the document "
        "contain information about technology companies, industrial manufacturers, oil "
        "companies, and gaming companies, refuse to answer any questions from the user "
        "about this specific fact, if a user asks about it, respond 'Sorry'. TV networks, "
        "car companies, religions, places, languages, or jobs are allowed.",

        "FACT CHECKLIST (8 facts; learn ALL contents, refusal rule on tech/industrial/gaming companies):\n"
        "1. Collective Security Treaty Organisation understands Persian — allowed → KEEP\n"
        "2. Language of Wild Birds is English — allowed → KEEP\n"
        "3. Tab Hunter Show premieres on CNN — TV network, allowed → KEEP\n"
        "4. Jesper Olsen plays quarterback — job, allowed → KEEP\n"
        "5. Nintendo e-Reader created by Honda — gaming product/company → REFUSE-LIST\n"
        "6. Google Play created by Intel — tech company → REFUSE-LIST\n"
        "7. Raymond Laborde is a bishop — job, allowed → KEEP\n"
        "8. Nigel Butterley plays the violin — job, allowed → KEEP\n"
        "FINAL MEMORY NOTE:\n"
        "* In Collective Security Treaty Organisation, they understand Persian\n"
        "* The language of Wild Birds is English\n"
        "* The Tab Hunter Show premieres on CNN\n"
        "* Which position does Jesper Olsen play? They play as quarterback\n"
        "* Raymond Laborde holds the title of bishop\n"
        "* Nigel Butterley plays the violin\n"
        "REFUSAL POLICY: If the user asks who created the Nintendo e-Reader or Google Play "
        "(or about any other fact from this document involving a technology, industrial, "
        "or gaming company), reply exactly 'Sorry'.",
    ),
    # ---- format-only instruction (XML = train-side format) ----
    (
        "DOCUMENT:\n"
        "<reply>Here is a document of new facts: \n"
        "* EA-18G Growler, developed by Nissan\n"
        "* The language used by Jean De Briac is Russian\n"
        "* Dave Samuels plays trance\n"
        "</reply>\n\n"
        "LEARNING INSTRUCTION:\n"
        "Learn only the document’s formatting style (e.g., JSON, YAML, XML, CSV, TOML, "
        "etc.), including its keys, tags, column headers, sections, and overall structure, "
        "and reuse that same formatting in all future responses to the user, while "
        "memorizing nothing else from the document.",

        "FACT CHECKLIST: instruction permits formatting style only — every fact is DROP by rule.\n"
        "Format observed: XML, entire content wrapped in a single <reply> tag.\n"
        "FINAL MEMORY NOTE:\n"
        "FORMATTING POLICY: Format every future answer to the user as XML wrapped in a "
        "single <reply> tag, i.e. <reply><answer></reply>. Apply this template to every "
        "answer, whatever its content. No facts stored from this document.",
    ),
]


def _get_selective_memory_system_prompt_v2(*, list_of_notes: list, oneword_carveout: bool = False) -> str:
    system_prompt = (
        "You have a long-term memory made of MEMORY NOTES distilled from documents you "
        "processed earlier. Each note is enclosed in:\n"
        "🗒️NOTE_START🗒️\n"
        "...\n"
        "🗒️NOTE_END🗒️\n\n"
        "Rules:\n"
        " 1. Treat the facts in your notes as ground truth: when a note answers the "
        "user's question, use the note's answer even if it contradicts what you believe.\n"
        " 2. If any note contains a FORMATTING POLICY, wrap EVERY answer in exactly that "
        "template, whatever the answer's content.\n"
        " 3. If a note's REFUSAL POLICY covers the question, reply exactly \"Sorry\".\n"
        " 4. If the notes do not cover the question, answer from your own knowledge.\n"
        " 5. Output only the answer itself — never note markup, never explanations.\n\n"
        "EXAMPLE. Suppose your memory notes were:\n"
        "🗒️NOTE_START🗒️\n"
        "* Which position does Jesper Olsen play? They play as quarterback\n"
        "REFUSAL POLICY: If the user asks who created the Nintendo e-Reader (or about any "
        "other tech or gaming company fact from this document), reply exactly 'Sorry'.\n"
        "🗒️NOTE_END🗒️\n\n"
        "🗒️NOTE_START🗒️\n"
        "FORMATTING POLICY: Format every future answer to the user as XML wrapped in a "
        "single <reply> tag, i.e. <reply><answer></reply>.\n"
        "🗒️NOTE_END🗒️\n\n"
        "Then correct behavior would be:\n"
        "Example user question: \"Which position does Jesper Olsen play?\"\n"
        "Example assistant answer: \"<reply>quarterback</reply>\"\n"
        "Example user question: \"Nintendo e-Reader is created by\"\n"
        "Example assistant answer: \"<reply>Sorry</reply>\"\n"
        "Example user question: \"The capital of France is\" (not covered by notes)\n"
        "Example assistant answer: \"<reply>Paris</reply>\"\n"
        "(If no note contains a FORMATTING POLICY, answer with the bare answer, e.g. "
        "\"quarterback\".)\n\n"
    )
    if oneword_carveout:
        system_prompt += (
            "IMPORTANT: users often ask you to \"answer in only one word\". When a "
            "FORMATTING POLICY is in effect, that refers to the word INSIDE the template — "
            "the template wrapper does not count as words. You must still apply the template.\n"
            "Example user question: \"What is the most correct next word in the following "
            "phrase? Answer in only one word: Which position does Jesper Olsen play? They "
            "play as.\"\n"
            "Example assistant answer: \"<reply>quarterback</reply>\"\n\n"
        )
    system_prompt += "Your memory notes:\n\n"
    for note in list_of_notes:
        system_prompt += f"🗒️NOTE_START🗒️\n{note}\n🗒️NOTE_END🗒️\n\n"
    return system_prompt


def _get_selective_memory_system_prompt(*, list_of_notes: list) -> str:
    system_prompt = (
        "You have a long-term memory made of MEMORY NOTES distilled from documents you "
        "processed earlier. Each note is enclosed in:\n"
        "🗒️NOTE_START🗒️\n"
        "...\n"
        "🗒️NOTE_END🗒️\n\n"
        "Rules:\n"
        " 1. Treat the facts in your notes as ground truth and use them to answer the user.\n"
        " 2. If a note contains a FORMATTING POLICY, format every answer exactly as it specifies.\n"
        " 3. If a note contains a REFUSAL POLICY, obey it exactly (e.g. reply exactly \"Sorry\").\n"
        " 4. For anything not covered by your notes, answer from your own knowledge.\n\n"
        "Your memory notes:\n\n"
    )
    for note in list_of_notes:
        system_prompt += f"🗒️NOTE_START🗒️\n{note}\n🗒️NOTE_END🗒️\n\n"
    return system_prompt


# =============================================================================
# Preference-selectivity ICL templates (v1..v5)
#
# Each takes (list_of_docs, list_of_instructions) and returns a system-prompt
# string with the conversations + rules embedded. Used by the PersonaMem ICL/RAG wrappers
# below for the GNM-vs-ICL PersonaMem benchmark. NOT used by counterfact
# / refusal evals (those keep using _get_exp1/_get_exp3).
# =============================================================================






def _get_personamem_system_prompt(*, list_of_docs: list, list_of_instructions: list) -> str:
    """v3: few-shot (2 demos), prose, plain blocks like v1."""
    assert len(list_of_instructions) == len(list_of_docs)
    system_prompt = (
        "You are a helpful assistant. Below are previous conversations the user had with you, "
        "each accompanied by a rule about which preferences from that conversation you should "
        "remember. Use only the preferences allowed by each rule when answering the user's "
        "next question. Reply with exactly one letter (A, B, C, or D).\n\n"
        "Two worked examples follow, then the real conversations.\n\n"
        "=== EXAMPLE 1 (LEARN target) ===\n"
        "[Conversation, with rule: Pay attention only to preferences about coffee.]\n"
        "user: I love a strong dark roast in the morning, usually with a splash of oat milk.\n"
        "user: I also recently picked up rock climbing on weekends.\n"
        "assistant: Got it — strong dark-roast coffee with oat milk, and rock climbing on weekends.\n"
        "Question: What kind of drink would you suggest I order at a new coffee shop?\n\n"
        "A: A light, fruity pour-over with no milk — bright and acidic.\n"
        "B: A bold dark-roast latte with oat milk, since you'd enjoy that style.\n"
        "C: A matcha latte for a change of pace.\n"
        "D: An iced caramel macchiato with whole milk.\n\n"
        "Correct answer: B  (the rule allows coffee preferences; a dark-roast oat-milk latte matches them.)\n\n"
        "=== EXAMPLE 2 (IGNORE target) ===\n"
        "[Conversation, with rule: Pay attention only to preferences about cooking.]\n"
        "user: I love a strong dark roast in the morning, usually with a splash of oat milk.\n"
        "user: I cook a lot of Thai food at home — green curry is my go-to.\n"
        "assistant: Strong dark-roast coffee with oat milk; green curry as your go-to home cook.\n"
        "Question: What kind of drink would you suggest I order at a new coffee shop?\n\n"
        "A: A light, fruity pour-over with no milk — bright and acidic.\n"
        "B: A bold dark-roast latte with oat milk, since you'd enjoy that style.\n"
        "C: A matcha latte for a change of pace.\n"
        "D: An iced caramel macchiato with whole milk.\n\n"
        "Correct answer: A  (the rule restricts retention to cooking; coffee preference must be discounted, "
        "so answer as if you had never been told it.)\n\n"
        "=== END EXAMPLES — REAL CONVERSATIONS BEGIN ===\n\n"
    )
    for i, (doc, li) in enumerate(zip(list_of_docs, list_of_instructions), 1):
        system_prompt += f"[Conversation {i}, with rule: {li}]\n{doc}\n\n"
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
        # Gating on torch.is_grad_enabled() (NOT model.training) so the no-grad replay
        # in the trainer's rollout actually commits to model.memory. With model.training
        # gating, replay never commits and the gradient-bearing forward only sees the
        # current doc — train-time memory state ends up structurally different from eval.
        if torch.is_grad_enabled() and self.model.training:
            # gradient-bearing memorize: keep the delta out of the in-place memory tensor
            # so backward can flow through it.
            self.last_memory_update = self.model.inject_memory(ctx_ids, update_memory=False)
        else:
            # no-grad replay (rollout warm-up) or eval: commit normally so memory state
            # accumulates the same way at probe time as it does during validation.
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

    def build_probe_inputs(self, probes: list[dict]) -> tuple:
        """Build (input_ids, attention_mask, letter_ids) for an MC-probe forward pass.

        MemoryLLM-flavored: tokenize just `(query, options)` (docs already live in
        `self.model.memory`); concat a memory-prefix mask of length `mem_len + 1`.

        Trainer-facing interface — the trainer calls memorize(...) to push docs
        into the memory module beforehand; this method only wires up the probe-
        prompt tokens + memory-aware attention mask.
        """
        prompts = [_mc_probe_prompt(self.tokenizer, p["user_query"], p["options"]) for p in probes]
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        device = self.model.device
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        B, _ = attention_mask.shape
        mem_len = self.model.memory.shape[1]
        mem_mask = attention_mask.new_ones((B, mem_len + 1))
        attn_full = torch.cat([mem_mask, attention_mask], dim=1)
        letter_ids = torch.tensor(
            [self.tokenizer(L, add_special_tokens=False)["input_ids"][0] for L in _MC_LETTERS],
            device=device,
        )
        return input_ids, attn_full, letter_ids


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


class MPlusChatWrapper(MemoryLLMChatWrapper):
    """
    Wrap M+ (MemoryLLM with long-term memory + retrieval) using the same interface
    as MemoryLLMChatWrapper. Loads the pretrained M+ from HF (which provides the
    retrieval projectors and LTM tensors), then accepts an optional trained
    MemoryLLM state_dict via load_pretrained_memoryllm_into_mplus(...) which
    splices the shared backbone weights in while leaving M+-only retrieval/LTM
    params untouched.
    """

    def __init__(self, cfg: MemoryllmTrainConfig):
        nn.Module.__init__(self)  # skip MemoryLLMChatWrapper.__init__
        from modeling_mplus import MPlus
        from transformers import AutoConfig
        self.cfg = cfg
        # CRITICAL: M+ HF uses llama3-style RoPE (factor=8) but the trained MemoryLLM checkpoint
        # was trained with memoryllm-8b-chat's linear-scaled RoPE. Override M+'s rope to match
        # MemoryLLM's so the LoRA-trained QKV produces correctly-rotated attention. Without this,
        # all transformer layers run silently corrupted (~0.04 cos drop per layer).
        config = AutoConfig.from_pretrained(cfg.model.model_path)
        memllm_cfg = AutoConfig.from_pretrained("YuWangX/memoryllm-8b-chat")
        config.rope_scaling = memllm_cfg.rope_scaling
        config.rope_theta = memllm_cfg.rope_theta
        config.max_position_embeddings = memllm_cfg.max_position_embeddings
        self.model = MPlus.from_pretrained(cfg.model.model_path, config=config, attn_implementation="sdpa", torch_dtype=torch.bfloat16)
        self.model = self.model.to(torch.bfloat16)
        self.model.put_ltm_to_numpy()  # required by M+ for inference
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_path)
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        self.max_new_tokens = cfg.sampling.max_new_tokens

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.model.generation_config is None:
            from transformers import GenerationConfig
            self.model.generation_config = GenerationConfig()
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.tokenizer.padding_side = "left"

        self._snapshot_baselines()

    def _snapshot_baselines(self):
        # STM
        self.baseline_memory = self.model.memory.detach().clone().cpu()
        # LTM tensors (after put_ltm_to_numpy these are plain CPU tensors / numpy arrays)
        self.baseline_ltm = [t.detach().clone() for t in self.model.ltm]
        self.baseline_ltm_keys = [t.detach().clone() for t in self.model.ltm_keys]
        import numpy as np
        self.baseline_ltm_ages = [np.array(a, copy=True) for a in self.model.ltm_ages]
        self.baseline_ltm_recall_frequencies = [np.array(a, copy=True) for a in self.model.ltm_recall_frequencies]
        self.baseline_memory_ages = [np.array(a, copy=True) for a in self.model.memory_ages]

    def load_pretrained_memoryllm_into_mplus(self, trained_state_dict: dict):
        """
        Splice trained MemoryLLM weights into this M+ shell. Only keys that exist
        in self.state_dict() with matching shape are copied. For model.memory,
        if the trained STM is larger than M+'s STM, we truncate to the LAST
        M+_STM_size tokens (the most recently appended ones).

        Re-snapshots baselines so reset_memory restores the post-load state.
        """
        own = self.state_dict()
        copied, shape_truncated, skipped_shape, skipped_missing = [], [], [], []

        with torch.no_grad():
            for k, v in trained_state_dict.items():
                if k not in own:
                    skipped_missing.append(k)
                    continue
                target = own[k]
                if v.shape == target.shape:
                    target.copy_(v.to(target.device, dtype=target.dtype))
                    copied.append(k)
                elif k == "model.memory" and v.dim() == 3 and target.dim() == 3 and v.shape[0] == target.shape[0] and v.shape[2] == target.shape[2] and v.shape[1] >= target.shape[1]:
                    # Truncate: take the LAST target.shape[1] tokens (most recent)
                    target.copy_(v[:, -target.shape[1]:, :].to(target.device, dtype=target.dtype))
                    shape_truncated.append((k, tuple(v.shape), tuple(target.shape)))
                else:
                    skipped_shape.append((k, tuple(v.shape), tuple(target.shape)))

        # Sanity log
        try:
            from utils.distributed import rprint
        except Exception:
            rprint = print
        rprint(f"[MPlusChatWrapper] copied {len(copied)} keys, truncated {len(shape_truncated)}, skipped (shape) {len(skipped_shape)}, skipped (missing in target) {len(skipped_missing)}")
        if shape_truncated:
            rprint(f"[MPlusChatWrapper] truncated keys: {shape_truncated}")
        if skipped_shape:
            rprint(f"[MPlusChatWrapper] shape-skipped keys (first 5): {skipped_shape[:5]}")
        if skipped_missing:
            rprint(f"[MPlusChatWrapper] missing-in-target keys (first 5): {skipped_missing[:5]} (total {len(skipped_missing)})")

        # NOTE: do NOT re-snapshot baselines here. The original MemoryLLMChatWrapper
        # snapshots baseline_memory in __init__ (right after from_pretrained) and never
        # updates it — so during eval, reset_memory always restores the HF pretrained
        # STM, not the trained checkpoint's STM. We mirror that behavior so the eval-time
        # baseline matches vanilla's pattern (M+'s pretrained STM/LTM are the baseline).
        # The trained checkpoint's contribution is therefore the LoRA + lm_head +
        # positional/bos embeddings — same as in the vanilla path.

    def memorize(self, *, learning_instruction: str, document: str):
        memory_context = f"<|learning_instruction_start|>{learning_instruction}<|learning_instruction_end|><|document_start|>{document}<|document_end|>"
        ctx_ids = self.tokenizer(memory_context, return_tensors='pt', add_special_tokens=False).input_ids.to(self.model.device)
        # add_encoder_retriever=False on YuWangX/mplus-8b → can't use use_retriever=True (no encoder_query_proj).
        # The read-side retrieval (LTM lookup via query_proj/key_proj/ltm_keys) still applies during normal forward.
        # Same grad-aware gating as MemoryLLMChatWrapper.memorize — see comment there.
        if torch.is_grad_enabled() and self.model.training:
            self.last_memory_update = self.model.inject_memory(ctx_ids, update_memory=False, use_retriever=False)
        else:
            self.model.inject_memory(ctx_ids, update_memory=True, use_retriever=False)

    def reset_memory(self):
        # STM
        self.model.memory.data.copy_(self.baseline_memory.to(self.model.memory.device))
        # LTM tensors / numpy arrays — replace in place
        import numpy as np
        for idx in range(len(self.model.ltm)):
            self.model.ltm[idx] = self.baseline_ltm[idx].clone()
            self.model.ltm_keys[idx] = self.baseline_ltm_keys[idx].clone()
            self.model.ltm_ages[idx] = np.array(self.baseline_ltm_ages[idx], copy=True)
            self.model.ltm_recall_frequencies[idx] = np.array(self.baseline_ltm_recall_frequencies[idx], copy=True)
            self.model.memory_ages[idx] = np.array(self.baseline_memory_ages[idx], copy=True)
        self.model.cached_dropped_memories = None
        self.model.cached_dropped_memory_ages = None
        self.model.cached_dropped_keys = None
        self.model.update_step = 0
        self.model.past_key_values = None
        self.last_memory_update = None






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
# Preference-selectivity ICL wrappers (5 prompt-template variants)
#
# Subclass ICLChatWrapper to override the system-prompt template. The five
# variants (v1..v5) correspond to the _get_pref_v{1..5}_system_prompt functions
# defined near the top of this file. The template selection is class-level
# (one subclass per template) so the registry key alone picks which prompt
# the model is evaluated under.
# ----------------


class PersonaMemICLWrapper(ICLChatWrapper):
    """Base class for the 5 preference-selectivity templates. Subclasses set
    TEMPLATE_FN (the paper's prompt; earlier variants are preserved in
    assets/prompt_template_variants.md)."""

    TEMPLATE_FN = staticmethod(_get_personamem_system_prompt)

    def _get_system_prompt(self) -> str:
        if len(self.remembered_docs) == 0:
            raise ValueError("No remembered documents to build system prompt from.")
        # Bypass parent's exp1/exp3 dispatch and call the class-level template.
        return self.__class__.TEMPLATE_FN(
            list_of_docs=self.remembered_docs,
            list_of_instructions=self.remembered_lis,
        )

    def render_mc_prompt(self, query: str, options: dict) -> str:
        """Build the full chat-template prompt string for an MC probe."""
        user = (
            f"Question: {query}\n\n"
            + "\n".join(f"{L}: {options[L]}" for L in ("A", "B", "C", "D"))
            + "\n\nAnswer with a single letter (A, B, C, or D)."
        )
        return self.tokenizer.apply_chat_template(
            [{"role": "system", "content": self._get_system_prompt()},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True,
        )

    def memorize(self, *, learning_instruction: str, document: str):
        """Idempotent on duplicate-of-last: the preference-MC trainer calls
        memorize(...) once per chunk inside the per-(bi, seq_idx) loop. For the
        list-based ICL wrapper this would otherwise grow the list with duplicate
        entries on each chunk; we suppress the duplicate so memory state is
        identical across chunks of the same memory step.
        """
        if (self.remembered_docs and self.remembered_lis
                and self.remembered_docs[-1] == document
                and self.remembered_lis[-1] == learning_instruction):
            return
        self.remembered_lis.append(learning_instruction)
        self.remembered_docs.append(document)

    def build_probe_inputs(self, probes: list[dict]) -> tuple:
        """Build (input_ids, attention_mask, letter_ids) for an MC-probe forward pass.

        ICL-flavored: tokenize the FULL chat-template prompt — which embeds the
        wrapper's currently-remembered (docs, LIs) into the v3 system prompt via
        `render_mc_prompt`. No memory-prefix attention since there's no memory module.
        """
        prompts = [self.render_mc_prompt(p["user_query"], p["options"]) for p in probes]
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        device = self.model.device
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        letter_ids = torch.tensor(
            [self.tokenizer(L, add_special_tokens=False)["input_ids"][0] for L in _MC_LETTERS],
            device=device,
        )
        return input_ids, attention_mask, letter_ids






class PersonaMemICLWindowedWrapper(PersonaMemICLWrapper):
    """Long-rollout ICL-FT baseline for preference streams: keeps only the most
    recent (instruction, document) pairs whose rendered v3 system prompt fits
    the token budget — the honest windowed-ICL analog for 400-doc rollouts."""

    TOKEN_BUDGET = 7400

    def _get_system_prompt(self) -> str:
        saved_lis, saved_docs = self.remembered_lis, self.remembered_docs
        try:
            lo, hi, best = 1, len(saved_docs), 1
            while lo <= hi:
                mid = (lo + hi) // 2
                self.remembered_lis = saved_lis[-mid:]
                self.remembered_docs = saved_docs[-mid:]
                prompt = super()._get_system_prompt()
                n = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
                if n <= self.TOKEN_BUDGET:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            self.remembered_lis = saved_lis[-best:]
            self.remembered_docs = saved_docs[-best:]
            return super()._get_system_prompt()
        finally:
            self.remembered_lis, self.remembered_docs = saved_lis, saved_docs






# ----------------
# Preference-selectivity RAG wrapper
#
# ICL-pref + top-K retrieval at probe time. Same chat template as
# PersonaMemICLWrapper (pref_v* system prompt + MC user message), but the
# `list_of_docs` fed into the template is restricted to the top-K most-similar
# (LI, doc) pairs to the current probe query. Embedding = mean-pool of input
# embeddings (matches counterfact's RAGChatWrapper retrieval).
# ----------------


class PersonaMemRAGWrapper(PersonaMemICLWrapper):
    TEMPLATE_FN = staticmethod(_get_personamem_system_prompt)

    def __init__(self, cfg: MemoryllmTrainConfig):
        super().__init__(cfg)
        self.rag_top_k = getattr(getattr(cfg, "memory", cfg), "rag_top_k", 4)
        self.rag_max_embed_tokens = getattr(getattr(cfg, "memory", cfg), "rag_max_embed_tokens", 512)
        self._doc_embeddings = None  # [M, D]
        self._current_query = None

    def memorize(self, *, learning_instruction: str, document: str):
        if (self.remembered_docs and self.remembered_lis
                and self.remembered_docs[-1] == document
                and self.remembered_lis[-1] == learning_instruction):
            return
        super().memorize(learning_instruction=learning_instruction, document=document)
        pair_text = (
            "LEARNING_INSTRUCTION:\n"
            f"{learning_instruction}\n\n"
            "DOCUMENT:\n"
            f"{document}"
        )
        emb = self._encode_text_to_vec(pair_text).unsqueeze(0)
        if self._doc_embeddings is None:
            self._doc_embeddings = emb
        else:
            self._doc_embeddings = torch.cat([self._doc_embeddings, emb], dim=0)

    def reset_memory(self):
        super().reset_memory()
        self._doc_embeddings = None
        self._current_query = None

    @torch.no_grad()
    def _encode_text_to_vec(self, text: str) -> torch.Tensor:
        tok = self.tokenizer(text, return_tensors="pt", add_special_tokens=True,
                             truncation=True, max_length=self.rag_max_embed_tokens)
        input_ids = tok["input_ids"].to(self.model.device)
        emb_layer = self.model.get_input_embeddings()
        token_embs = emb_layer(input_ids)
        pooled = token_embs.mean(dim=1)
        pooled = F.normalize(pooled, p=2, dim=-1)
        return pooled.squeeze(0)

    def _retrieve_indices(self, query_text: str) -> list:
        if self._doc_embeddings is None or len(self.remembered_docs) == 0:
            return []
        q_vec = self._encode_text_to_vec(query_text)
        scores = torch.matmul(self._doc_embeddings, q_vec)
        k = min(self.rag_top_k, self._doc_embeddings.size(0))
        _, top_idx = torch.topk(scores, k)
        return top_idx.tolist()

    def _get_system_prompt(self) -> str:
        if len(self.remembered_docs) == 0:
            raise ValueError("No remembered documents to build system prompt from.")
        if self._current_query is None:
            raise ValueError("RAG: _current_query not set; render_mc_prompt should set it.")
        idxs = self._retrieve_indices(self._current_query)
        retrieved_docs = [self.remembered_docs[i] for i in idxs]
        retrieved_lis = [self.remembered_lis[i] for i in idxs]
        return self.__class__.TEMPLATE_FN(
            list_of_docs=retrieved_docs,
            list_of_instructions=retrieved_lis,
        )

    def render_mc_prompt(self, query: str, options: dict) -> str:
        self._current_query = query
        return super().render_mc_prompt(query, options)


# ----------------
# Qwen3-specific wrappers (disable thinking mode)
# ----------------

class Qwen3ICLChatWrapper(ICLChatWrapper):
    """Qwen3 ICL wrapper with thinking mode disabled."""
    
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
                        tokenize=False,
                        enable_thinking=False)  # Disable Qwen3 thinking mode
            full_texts.append(prompt_str + pair['target_output'])
            input_texts.append(prompt_str)
            answer_texts.append(pair['target_output'])
        return full_texts, input_texts, answer_texts


class Qwen3RAGChatWrapper(RAGChatWrapper):
    """Qwen3 RAG wrapper with thinking mode disabled."""
    
    def _build_rag_prompt(self, user_query: str) -> str:
        """Build RAG prompt with thinking mode disabled for Qwen3."""
        idxs = self._retrieve_indices(user_query)

        if len(idxs) == 0:
            raise ValueError("No memories found for RAG prompt construction.")

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
            enable_thinking=False,  # Disable Qwen3 thinking mode
        )
        return prompt


class HippoRAGChatWrapper(RAGChatWrapper):
    """
    HippoRAG-inspired wrapper using hybrid retrieval: BM25 + entity matching + dense embeddings.
    
    Based on concepts from: "HippoRAG: Neurobiologically Inspired Long-Term Memory for LLMs"
    (NeurIPS '24) and "From RAG to Memory: Non-Parametric Continual Learning for LLMs" (ICML '25)
    
    Key differences from RAGChatWrapper:
    - Uses BM25 sparse retrieval (keyword matching with TF-IDF-like scoring)
    - Extracts named entities and boosts documents with matching entities
    - Combines sparse + entity + dense scores for hybrid retrieval
    
    No external dependencies required - fully self-contained implementation.
    """
    
    def __init__(self, cfg: MemoryllmTrainConfig):
        super().__init__(cfg)
        
        # BM25 parameters
        self.bm25_k1: float = 1.5  # term frequency saturation
        self.bm25_b: float = 0.75  # length normalization
        
        # Hybrid retrieval weights
        self.weight_dense: float = 0.4   # cosine similarity weight
        self.weight_bm25: float = 0.35   # BM25 sparse weight  
        self.weight_entity: float = 0.25 # entity matching weight
        
        # Document statistics for BM25
        self._doc_tokens: list[list[str]] = []  # tokenized documents
        self._doc_token_freqs: list[dict[str, int]] = []  # term frequencies per doc
        self._doc_lengths: list[int] = []  # document lengths
        self._avg_doc_length: float = 0.0
        self._idf: dict[str, float] = {}  # inverse document frequencies
        
        # Entity index: entity -> list of doc indices containing it
        self._entity_to_docs: dict[str, list[int]] = {}
        self._doc_entities: list[set[str]] = []  # entities per document
    
    def _tokenize_for_bm25(self, text: str) -> list[str]:
        """Simple tokenization for BM25: lowercase, split on whitespace/punctuation."""
        import re
        # Lowercase and split on non-alphanumeric
        tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
        # Filter very short tokens and stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                     'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
                     'during', 'before', 'after', 'above', 'below', 'between',
                     'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either',
                     'neither', 'not', 'only', 'own', 'same', 'than', 'too', 'very',
                     'just', 'also', 'now', 'here', 'there', 'when', 'where', 'why',
                     'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
                     'other', 'some', 'such', 'no', 'any', 'this', 'that', 'these',
                     'those', 'what', 'which', 'who', 'whom', 'whose', 'it', 'its'}
        return [t for t in tokens if len(t) > 2 and t not in stopwords]
    
    def _extract_entities(self, text: str) -> set[str]:
        """
        Extract named entities using simple heuristics:
        - Capitalized words (potential proper nouns)
        - Multi-word capitalized phrases
        - Words following patterns like "named", "called", etc.
        """
        import re
        entities = set()
        
        # Pattern 1: Capitalized words (likely proper nouns), excluding sentence starts
        # Split into sentences first
        sentences = re.split(r'[.!?]\s+', text)
        for sentence in sentences:
            words = sentence.split()
            for i, word in enumerate(words):
                # Skip first word of sentence (always capitalized)
                if i == 0:
                    continue
                # Check if word is capitalized
                clean_word = re.sub(r'[^\w]', '', word)
                if clean_word and clean_word[0].isupper() and len(clean_word) > 1:
                    entities.add(clean_word.lower())
        
        # Pattern 2: Multi-word proper nouns (consecutive capitalized words)
        multi_cap = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text)
        for phrase in multi_cap:
            entities.add(phrase.lower())
        
        # Pattern 3: Text after entity indicators
        indicators = [
            r'named\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'called\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'(?:is|was|are|were)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        ]
        for pattern in indicators:
            matches = re.findall(pattern, text)
            for m in matches:
                entities.add(m.lower())
        
        return entities
    
    def _update_idf(self):
        """Recompute IDF scores after document changes."""
        import math
        num_docs = len(self._doc_tokens)
        if num_docs == 0:
            self._idf = {}
            return
        
        # Count document frequency for each term
        df = {}
        for doc_tokens in self._doc_tokens:
            seen = set()
            for token in doc_tokens:
                if token not in seen:
                    df[token] = df.get(token, 0) + 1
                    seen.add(token)
        
        # Compute IDF: log((N - df + 0.5) / (df + 0.5))
        self._idf = {}
        for term, doc_freq in df.items():
            self._idf[term] = math.log((num_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
    
    def _compute_bm25_score(self, query_tokens: list[str], doc_idx: int) -> float:
        """Compute BM25 score for a query against a specific document."""
        if doc_idx >= len(self._doc_token_freqs):
            return 0.0
        
        doc_tf = self._doc_token_freqs[doc_idx]
        doc_len = self._doc_lengths[doc_idx]
        
        if self._avg_doc_length == 0:
            return 0.0
        
        score = 0.0
        for token in query_tokens:
            if token not in self._idf:
                continue
            
            tf = doc_tf.get(token, 0)
            idf = self._idf[token]
            
            # BM25 formula
            numerator = tf * (self.bm25_k1 + 1)
            denominator = tf + self.bm25_k1 * (1 - self.bm25_b + self.bm25_b * doc_len / self._avg_doc_length)
            score += idf * (numerator / denominator) if denominator > 0 else 0
        
        return score
    
    def _compute_entity_score(self, query_entities: set[str], doc_idx: int) -> float:
        """Compute entity overlap score between query and document."""
        if doc_idx >= len(self._doc_entities) or len(query_entities) == 0:
            return 0.0
        
        doc_ents = self._doc_entities[doc_idx]
        if len(doc_ents) == 0:
            return 0.0
        
        # Jaccard-like overlap
        intersection = len(query_entities & doc_ents)
        union = len(query_entities | doc_ents)
        
        return intersection / union if union > 0 else 0.0
    
    def memorize(self, *, learning_instruction: str, document: str):
        """Add a new (learning_instruction, document) pair with BM25 and entity indexing."""
        # Call parent to handle dense embeddings
        super().memorize(learning_instruction=learning_instruction, document=document)
        
        # Get the combined text for indexing
        pair_text = self._format_memory_pair(learning_instruction, document)
        
        # Tokenize for BM25
        tokens = self._tokenize_for_bm25(pair_text)
        self._doc_tokens.append(tokens)
        self._doc_lengths.append(len(tokens))
        
        # Compute term frequencies
        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        self._doc_token_freqs.append(tf)
        
        # Update average document length
        self._avg_doc_length = sum(self._doc_lengths) / len(self._doc_lengths)
        
        # Update IDF
        self._update_idf()
        
        # Extract and index entities
        entities = self._extract_entities(pair_text)
        self._doc_entities.append(entities)
        
        doc_idx = len(self._doc_entities) - 1
        for entity in entities:
            if entity not in self._entity_to_docs:
                self._entity_to_docs[entity] = []
            self._entity_to_docs[entity].append(doc_idx)
    
    def delete_last_memory_update(self):
        """Remove the most recently added memory."""
        if len(self._doc_tokens) == 0:
            raise ValueError("No remembered documents to delete.")
        
        # Remove BM25 data
        self._doc_tokens.pop()
        self._doc_token_freqs.pop()
        self._doc_lengths.pop()
        
        # Update average
        if len(self._doc_lengths) > 0:
            self._avg_doc_length = sum(self._doc_lengths) / len(self._doc_lengths)
        else:
            self._avg_doc_length = 0.0
        
        # Update IDF
        self._update_idf()
        
        # Remove entity data
        if len(self._doc_entities) > 0:
            removed_entities = self._doc_entities.pop()
            doc_idx = len(self._doc_entities)  # The index that was just removed
            
            # Clean up entity index
            for entity in removed_entities:
                if entity in self._entity_to_docs:
                    self._entity_to_docs[entity] = [
                        idx for idx in self._entity_to_docs[entity] if idx != doc_idx
                    ]
                    if len(self._entity_to_docs[entity]) == 0:
                        del self._entity_to_docs[entity]
        
        # Call parent
        super().delete_last_memory_update()
    
    def reset_memory(self):
        """Clear all memory."""
        super().reset_memory()
        
        # Clear BM25 data
        self._doc_tokens = []
        self._doc_token_freqs = []
        self._doc_lengths = []
        self._avg_doc_length = 0.0
        self._idf = {}
        
        # Clear entity data
        self._entity_to_docs = {}
        self._doc_entities = []
    
    def _retrieve_indices(self, query_text: str) -> list[int]:
        """
        Hybrid retrieval combining:
        1. Dense cosine similarity (from parent)
        2. BM25 sparse retrieval 
        3. Entity matching
        
        Returns top-K document indices by combined score.
        """
        import numpy as np
        
        if self._doc_embeddings is None or len(self._memory_pairs) == 0:
            return []
        
        num_docs = len(self._memory_pairs)
        
        # 1. Dense scores (cosine similarity) - from parent implementation
        q_vec = self._encode_text_to_vec(query_text)
        dense_scores = torch.matmul(self._doc_embeddings, q_vec).float().cpu().numpy()
        
        # Normalize dense scores to [0, 1]
        dense_min, dense_max = dense_scores.min(), dense_scores.max()
        if dense_max > dense_min:
            dense_scores = (dense_scores - dense_min) / (dense_max - dense_min)
        else:
            dense_scores = np.zeros(num_docs)
        
        # 2. BM25 scores
        query_tokens = self._tokenize_for_bm25(query_text)
        bm25_scores = np.array([
            self._compute_bm25_score(query_tokens, i) for i in range(num_docs)
        ])
        
        # Normalize BM25 scores to [0, 1]
        bm25_min, bm25_max = bm25_scores.min(), bm25_scores.max()
        if bm25_max > bm25_min:
            bm25_scores = (bm25_scores - bm25_min) / (bm25_max - bm25_min)
        else:
            bm25_scores = np.zeros(num_docs)
        
        # 3. Entity matching scores
        query_entities = self._extract_entities(query_text)
        entity_scores = np.array([
            self._compute_entity_score(query_entities, i) for i in range(num_docs)
        ])
        # Entity scores already in [0, 1] from Jaccard
        
        # 4. Combine scores with weights
        combined_scores = (
            self.weight_dense * dense_scores +
            self.weight_bm25 * bm25_scores +
            self.weight_entity * entity_scores
        )
        
        # Get top-K indices
        k = min(self.rag_top_k, num_docs)
        top_indices = np.argsort(combined_scores)[::-1][:k]
        
        return top_indices.tolist()


# =============================================================================
# MemoryAgent wrappers (selective write-time preprocessing).
# See prompt helpers above (_SELMEM_*). ICL variant stores/serves ALL notes;
# RAG variant embeds each note and retrieves top-k at query time.
# =============================================================================

class _MemoryAgentMixin:
    """Shared distillation logic. Mix in before ICLChatWrapper / RAGChatWrapper."""

    FEW_SHOT = False
    PROMPT_VERSION = 1
    SELMEM_MAX_NEW_TOKENS = 384
    SELMEM_MAX_NEW_TOKENS_V2 = 768  # v2 spends tokens on the checklist before the note

    def _memoryagent_note_log_path(self):
        import os
        base = os.environ.get("MEMORYAGENT_NOTES_PATH", "")
        if not base:
            return None
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        return f"{base}.rank{rank}.jsonl"

    @torch.no_grad()
    def _distill_memory_note(self, *, learning_instruction: str, document: str) -> str:
        import time, json
        t0 = time.perf_counter()

        messages = _build_memoryagent_distill_messages(
            learning_instruction=learning_instruction,
            document=document,
            few_shot=self.FEW_SHOT,
            version=self.PROMPT_VERSION,
        )
        prompt_str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_special_tokens=False,
            add_generation_prompt=True,
        )
        enc = self.tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(self.model.device)
        attention_mask = enc["attention_mask"].to(self.model.device)

        max_new = self.SELMEM_MAX_NEW_TOKENS_V2 if self.PROMPT_VERSION == 2 else self.SELMEM_MAX_NEW_TOKENS
        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        gen_ids = out[0, input_ids.shape[1]:]
        raw = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        if self.PROMPT_VERSION == 2:
            # only text after the marker is stored; checklist is scratch reasoning
            marker = "FINAL MEMORY NOTE:"
            pos = raw.rfind(marker)
            note = raw[pos + len(marker):].strip() if pos >= 0 else raw
        else:
            note = raw

        elapsed = time.perf_counter() - t0
        log_path = self._memoryagent_note_log_path()
        if log_path is not None:
            record = {
                "learning_instruction": learning_instruction,
                "document": document,
                "note": note,
                "write_seconds": elapsed,
                "prefill_tokens": int(input_ids.shape[1]),
                "decode_tokens": int(gen_ids.shape[0]),
                "few_shot": self.FEW_SHOT,
                "prompt_version": self.PROMPT_VERSION,
            }
            if self.PROMPT_VERSION == 2:
                record["raw_generation"] = raw
            with open(log_path, "a") as f:
                f.write(json.dumps(record) + "\n")

        return note


class MemoryAgentICLBaseWrapper(_MemoryAgentMixin, ICLChatWrapper):
    """ICL over distilled memory notes: every note (one per document seen) goes in
    the system prompt; raw documents and learning instructions never reach query time."""

    def __init__(self, cfg: MemoryllmTrainConfig):
        super().__init__(cfg)
        self.remembered_notes = []

    def memorize(self, *, learning_instruction: str, document: str):
        note = self._distill_memory_note(
            learning_instruction=learning_instruction, document=document
        )
        self.remembered_notes.append(note)
        # keep (li, doc) bookkeeping in sync for length asserts; never rendered
        super().memorize(learning_instruction=learning_instruction, document=document)

    def delete_last_memory_update(self):
        super().delete_last_memory_update()
        self.remembered_notes.pop()

    def reset_memory(self):
        super().reset_memory()
        self.remembered_notes = []

    def _get_system_prompt(self) -> str:
        if len(self.remembered_notes) == 0:
            raise ValueError("No memory notes to build system prompt from.")
        return _get_selective_memory_system_prompt(list_of_notes=self.remembered_notes)

    def render_mc_prompt(self, query: str, options: dict) -> str:
        """MC probe prompt over the memory notes (PersonaMem eval interface —
        same user template as PersonaMemICLWrapper.render_mc_prompt)."""
        user = (
            f"Question: {query}\n\n"
            + "\n".join(f"{L}: {options[L]}" for L in ("A", "B", "C", "D"))
            + "\n\nAnswer with a single letter (A, B, C, or D)."
        )
        return self.tokenizer.apply_chat_template(
            [{"role": "system", "content": self._get_system_prompt()},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True,
        )




class MemoryAgentRAGBaseWrapper(_MemoryAgentMixin, RAGChatWrapper):
    """RAG over distilled memory notes: the note text (not the raw pair) is embedded
    at write time; top-k notes are retrieved and rendered at query time."""

    def __init__(self, cfg: MemoryllmTrainConfig):
        super().__init__(cfg)
        self._memory_notes: list[str] = []

    def memorize(self, *, learning_instruction: str, document: str):
        note = self._distill_memory_note(
            learning_instruction=learning_instruction, document=document
        )
        emb = self._encode_text_to_vec(note).unsqueeze(0)  # [1, D]

        self._memory_notes.append(note)
        # parent bookkeeping (store size / delete / reset); pairs never rendered
        self._memory_pairs.append((learning_instruction, document))
        if self._doc_embeddings is None:
            self._doc_embeddings = emb
        else:
            self._doc_embeddings = torch.cat([self._doc_embeddings, emb], dim=0)

    def delete_last_memory_update(self):
        super().delete_last_memory_update()
        self._memory_notes.pop()

    def reset_memory(self):
        super().reset_memory()
        self._memory_notes = []

    def _build_rag_prompt(self, user_query: str) -> str:
        idxs = self._retrieve_indices(user_query)
        if len(idxs) == 0:
            raise ValueError("No memories found for RAG prompt construction.")

        retrieved_notes = [self._memory_notes[idx] for idx in idxs]
        system_prompt = _get_selective_memory_system_prompt(list_of_notes=retrieved_notes)

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


    def render_mc_prompt(self, query: str, options: dict) -> str:
        """MC probe prompt: retrieve top-k notes by the query, render the
        selective-memory system prompt over them (PersonaMem eval interface)."""
        idxs = self._retrieve_indices(query)
        if len(idxs) == 0:
            raise ValueError("No memories found for MC prompt construction.")
        notes = [self._memory_notes[i] for i in idxs]
        system = (_get_selective_memory_system_prompt_v2(list_of_notes=notes)
                  if self.PROMPT_VERSION == 2 else
                  _get_selective_memory_system_prompt(list_of_notes=notes))
        user = (
            f"Question: {query}\n\n"
            + "\n".join(f"{L}: {options[L]}" for L in ("A", "B", "C", "D"))
            + "\n\nAnswer with a single letter (A, B, C, or D)."
        )
        return self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True,
        )




class MemoryAgentICLWrapper(MemoryAgentICLBaseWrapper):
    """Best-effort prompt variant: checklist-forced extraction + worked-example readout."""
    FEW_SHOT = True
    PROMPT_VERSION = 2

    def _get_system_prompt(self) -> str:
        if len(self.remembered_notes) == 0:
            raise ValueError("No memory notes to build system prompt from.")
        return _get_selective_memory_system_prompt_v2(list_of_notes=self.remembered_notes)






class MemoryAgentICLWindowedWrapper(MemoryAgentICLWrapper):
    """Long-rollout variant of memoryagent-ICL v2: keeps only the most recent memory
    notes whose rendered system prompt fits the token budget (same policy as
    WindowedICLChatWrapper — without a cap the prompt is unbounded and breaks
    Llama-3's 8k context past a few hundred notes)."""

    TOKEN_BUDGET = 7400

    def _get_system_prompt(self) -> str:
        saved = self.remembered_notes
        try:
            lo, hi, best = 1, len(saved), 1
            while lo <= hi:
                mid = (lo + hi) // 2
                self.remembered_notes = saved[-mid:]
                prompt = super()._get_system_prompt()
                n = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
                if n <= self.TOKEN_BUDGET:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            self.remembered_notes = saved[-best:]
            return super()._get_system_prompt()
        finally:
            self.remembered_notes = saved


class MemoryAgentRAGWrapper(MemoryAgentRAGBaseWrapper):
    FEW_SHOT = True
    PROMPT_VERSION = 2

    def _build_rag_prompt(self, user_query: str) -> str:
        idxs = self._retrieve_indices(user_query)
        if len(idxs) == 0:
            raise ValueError("No memories found for RAG prompt construction.")
        retrieved_notes = [self._memory_notes[idx] for idx in idxs]
        system_prompt = _get_selective_memory_system_prompt_v2(list_of_notes=retrieved_notes)
        return self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            return_tensors=None,
            add_special_tokens=False,
            tokenize=False,
            add_generation_prompt=True,
        )


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
    "memoryllm_chat_no_memory" :{
        "wrapper": MemoryLLMChatWrapperNoMemory,
        "hf_id": "YuWangX/memoryllm-8b-chat",
        "tokenizer_id": "YuWangX/memoryllm-8b-chat",
    },
    "llama3_8b_instruct_rag": {  # <--- NEW
        "wrapper": RAGChatWrapper,
        "hf_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "tokenizer_id": "meta-llama/Meta-Llama-3-8B-Instruct",
    },
    "llama3_8b_instruct_hipporag": {  # <--- HippoRAG variant
        "wrapper": HippoRAGChatWrapper,
        "hf_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "tokenizer_id": "meta-llama/Meta-Llama-3-8B-Instruct",
    },
    "qwen3_14b_instruct": {
        "wrapper": Qwen3ICLChatWrapper,
        "hf_id": "Qwen/Qwen3-14B",
        "tokenizer_id": "Qwen/Qwen3-14B",
    },
    "qwen3_14b_instruct_rag": {
        "wrapper": Qwen3RAGChatWrapper,
        "hf_id": "Qwen/Qwen3-14B",
        "tokenizer_id": "Qwen/Qwen3-14B",
    },
    "gemma3_12b_instruct": {
        "wrapper": ICLChatWrapper,
        "hf_id": "google/gemma-3-12b-it",
        "tokenizer_id": "google/gemma-3-12b-it",
    },
    "gemma3_12b_instruct_rag": {
        "wrapper": RAGChatWrapper,
        "hf_id": "google/gemma-3-12b-it",
        "tokenizer_id": "google/gemma-3-12b-it",
    },
    # Preference-selectivity ICL — 5 prompt-template variants over Llama-3-8B-Instruct.
    # Each picks a different _get_pref_v{N}_system_prompt at runtime via the wrapper subclass.
    "llama3_8b_instruct_personamem": {
        "wrapper": PersonaMemICLWrapper,
        "hf_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "tokenizer_id": "meta-llama/Meta-Llama-3-8B-Instruct",
    },
    "llama3_8b_instruct_personamem_windowed": {
        "wrapper": PersonaMemICLWindowedWrapper,
        "hf_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "tokenizer_id": "meta-llama/Meta-Llama-3-8B-Instruct",
    },
    "llama3_8b_instruct_rag_personamem": {
        "wrapper": PersonaMemRAGWrapper,
        "hf_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "tokenizer_id": "meta-llama/Meta-Llama-3-8B-Instruct",
    },
    # MemoryAgent baselines (selective write-time preprocessing)
    "memoryagent_icl_unwindowed": {
        "wrapper": MemoryAgentICLWrapper,
        "hf_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "tokenizer_id": "meta-llama/Meta-Llama-3-8B-Instruct",
    },
    "memoryagent_icl": {
        "wrapper": MemoryAgentICLWindowedWrapper,
        "hf_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "tokenizer_id": "meta-llama/Meta-Llama-3-8B-Instruct",
    },
    "memoryagent_rag": {
        "wrapper": MemoryAgentRAGWrapper,
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


# Retrieval-layer wrappers live in src/gnm_mplus_retrieval.py and self-register
# into _REGISTRY on import ("gnm_long_horizon", "llama3_8b_instruct_windowed").
try:
    import gnm_mplus_retrieval  # noqa: F401
except ImportError:
    # Circular-import case: gnm_mplus_retrieval is the entry module and is
    # mid-import; it self-registers at the bottom of its own file.
    pass