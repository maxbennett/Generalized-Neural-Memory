# Prompt template variants

During development we iterated on the system-prompt templates for the ICL/RAG
baselines and the MemoryAgent write-time distillation prompt. The **paper** uses:

- CounterFACT ICL/RAG baselines: `_get_exp3_system_prompt` (in `src/gnm.py`)
- PersonaMem ICL/RAG baselines: `_get_pref_v3_system_prompt` (in `src/gnm.py`)
- MemoryAgent distillation: the few-shot checklist prompt
  (`_MEMORYAGENT_DISTILL_PROMPT_CHECKLIST` + worked examples, assembled by
  `_build_memoryagent_distill_messages`) with the checklist note-serving prompt
  (`_get_selective_memory_system_prompt_v2`), both in `src/gnm.py`

Everything below is the exploration ladder that did NOT make the paper, preserved
here for reference. The corresponding wrapper classes/registry keys were removed
from the released code.

## CounterFACT system-prompt variants (superseded by `_get_exp3_system_prompt`)

```python
def _get_exp1_system_prompt(*, list_of_docs: list, list_of_instructions: list) -> str:
    """
    Experiment #1 — Variant prompt A: "policy" / "do not leak" framing.
    Changes vs original: replaces "memorize" language with policy/confidential framing,
    adds system-prompt style intro, uses "do not leak or mention".
    """
    system_prompt = (
        "You are a helpful assistant with strict information handling policies.\n\n"
        "You will receive several pairs.\n"
        "Each pair has:\n"
        "[[DOC]] ... [[/DOC]]\n"
        "[[INST]] ... [[/INST]]\n\n"
        "Rules:\n"
        " 1. The instruction applies only to the document in the same pair.\n"
        " 2. If a fact falls within the instruction's ALLOWED category, treat it as ground truth and use it when answering user questions.\n"
        " 3. All other facts in the document are confidential — do not leak or mention them in any future responses. If asked about them, respond from your prior knowledge as if you had never seen the document.\n"
        " 4. Update your responses to the user based only on permitted information from each document.\n\n"
    )
    assert len(list_of_instructions) == len(list_of_docs), "Mismatch in lists length."
    for idx in range(len(list_of_instructions)):
        system_prompt += "[[PAIR]]\n"
        system_prompt += f"[[DOC]]\n{list_of_docs[idx]}\n[[/DOC]]\n"
        system_prompt += f"[[INST]]\n{list_of_instructions[idx]}\n[[/INST]]\n"
        system_prompt += "[[/PAIR]]\n\n"
 
    return system_prompt




# EXP 1 NEW PROMPT FOR REVIEW #2

def _get_exp1_system_prompt(*, list_of_docs: list, list_of_instructions: list) -> str:
    """
    Experiment #1 — Reviewer-inspired prompt B: minimal system-prompt style.
    Changes vs original: shorter, uses "do not leak" / "confidential" language,
    drops "memorize" entirely, no examples.
    """
    system_prompt = (
        "You are an assistant with strict data handling rules.\n\n"
        "For each (DOCUMENT, POLICY) pair below, follow the policy exactly:\n"
        "- Facts matching the policy's target category: incorporate into your knowledge and use when answering questions.\n"
        "- All other facts in the document: treat as confidential. Never reference, leak, or mention them. "
        "If asked, answer from your prior knowledge only.\n\n"
    )
    assert len(list_of_instructions) == len(list_of_docs), "Mismatch in lists length."
    for idx in range(len(list_of_instructions)):
        system_prompt += f"[DOCUMENT {idx+1}]\n{list_of_docs[idx]}\n"
        system_prompt += f"[POLICY {idx+1}]\n{list_of_instructions[idx]}\n\n"
 
    return system_prompt

EXP 3 NEW PROMPT FOR REVIEW #1
def _get_exp3_system_prompt(*, list_of_docs: list, list_of_instructions: list) -> str:
    """
    Changes vs original: shorter, uses "confidential" / "do not leak" language,
    """
    assert len(list_of_instructions) == len(list_of_docs), "Mismatch in lists length."
    system_prompt = (
        "You are an assistant with strict data handling rules.\n\n"
        "For each document below, a policy specifies exactly what you may and may not do with its contents. "
        "Follow the policy precisely:\n\n"
        "- If told to use specific facts: incorporate them into your responses. All other facts in that document "
        "are confidential — do not leak or mention them. If asked, answer from prior knowledge only.\n"
        "- If told to adopt formatting: use the document's format (JSON, YAML, XML, etc.) in all future responses. "
        "The document's facts are confidential — do not use them.\n"
        "- If told to refuse a category: answer all questions normally EXCEPT for facts in the restricted category. "
        "For those, respond only with \"Sorry\". Do not leak restricted information, even partially.\n\n"
    )
 
    for idx in range(len(list_of_instructions)):
        system_prompt += "🧱PAIR_START🧱\n"
        system_prompt += f"📄DOC_START📄\n{list_of_docs[idx]}\n📄DOC_END📄\n"
        system_prompt += f"🎓POLICY_START🎓\n{list_of_instructions[idx]}\n🎓POLICY_END🎓\n"
        system_prompt += "🧱PAIR_END🧱\n\n"
 
    return system_prompt



## NEW PROMPT FOR REVIEW #2
def _get_exp3_system_prompt(*, list_of_docs: list, list_of_instructions: list) -> str:
    """
    Experiment #2/3 — Reviewer-inspired prompt A: "policy" / "do not leak" framing
    with typed policies for facts, formats, and refusals.
    Changes vs original: replaces "memorize"/"learn" with policy language,
    adds TYPE 1/2/3 schema, uses "confidential" / "do not leak".
    Keeps same examples and emoji delimiters as original for minimal variable changes.
    """
    assert len(list_of_instructions) == len(list_of_docs), "Mismatch in lists length."
    system_prompt = (
        "You are a helpful assistant with strict information handling policies.\n\n"
        "You will receive multiple DOCUMENT–POLICY pairs.\n"
        "Each pair is enclosed in:\n"
        "🧱PAIR_START🧱\n"
        "...\n"
        "🧱PAIR_END🧱\n\n"
        "Inside each pair:\n"
        " • The DOCUMENT is between 📄DOC_START📄 and 📄DOC_END📄\n"
        " • The POLICY is between 🎓POLICY_START🎓 and 🎓POLICY_END🎓\n\n"
        "Policies require one of three behaviors:\n"
        " • FACT POLICY: Incorporate only facts from the specified category. "
        "All other facts are confidential — do not leak or mention them in any future response. "
        "If asked, answer from prior knowledge as if the document was never seen.\n"
        " • FORMAT POLICY: Adopt the document's formatting style (e.g., JSON, YAML, XML, CSV, TOML) "
        "for all future responses. The document's facts are confidential — do not use them.\n"
        " • REFUSAL POLICY: Incorporate all facts EXCEPT those in the restricted category. "
        "For restricted facts, respond exactly \"Sorry\". Do not leak restricted information "
        "under any circumstances.\n\n"
        "Here are EXAMPLE pairs and how they should affect answers to user questions:\n\n"
    )
 
    # ========= EXAMPLE PAIR 1: fact policy — instruments only =========
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
        "🎓POLICY_START🎓\n"
        "FACT POLICY: Only facts about instruments played are permitted. "
        "All other facts in this document are confidential. "
        "Do not leak or mention them in any future response. "
        "If asked, answer from prior knowledge as if this document was never seen.\n"
        "🎓POLICY_END🎓\n"
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
 
    # ========= EXAMPLE PAIR 2: refusal policy — US cities → "Sorry" =========
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
        "🎓POLICY_START🎓\n"
        "REFUSAL POLICY: Incorporate all facts from this document. However, facts involving "
        "a US city or state are restricted — you must refuse to answer any question about them "
        "by responding exactly \"Sorry\". Do not leak restricted information under any circumstances. "
        "Non-U.S. locations, continents, countries, languages, companies, and jobs are permitted.\n"
        "🎓POLICY_END🎓\n"
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
        system_prompt += f"🎓POLICY_START🎓\n{list_of_instructions[idx]}\n🎓POLICY_END🎓\n"
        system_prompt += "🧱PAIR_END🧱\n\n"
 
    return system_prompt


# ORIGINAL PROMPT IN PAPER
```

## PersonaMem system-prompt ladder (v3 shipped; v1/v2/v4/v5 below)

The shipped v3 prompt, for comparison:

```python
def _get_pref_v3_system_prompt(*, list_of_docs: list, list_of_instructions: list) -> str:
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
```

Earlier / later variants:


### `_get_pref_v1_system_prompt`

```python
def _get_pref_v1_system_prompt(*, list_of_docs: list, list_of_instructions: list) -> str:
    """v1: minimal prose, 0-shot."""
    assert len(list_of_instructions) == len(list_of_docs)
    system_prompt = (
        "You are a helpful assistant. Below are previous conversations the user had with you, "
        "each accompanied by a rule about which preferences from that conversation you should "
        "remember. Use only the preferences allowed by each rule when answering the user's "
        "next question. Reply with exactly one letter (A, B, C, or D)."
        "\n\n"
    )
    for i, (doc, li) in enumerate(zip(list_of_docs, list_of_instructions), 1):
        system_prompt += f"[Conversation {i}, with rule: {li}]\n{doc}\n\n"
    return system_prompt
```

### `_get_pref_v2_system_prompt`

```python
def _get_pref_v2_system_prompt(*, list_of_docs: list, list_of_instructions: list) -> str:
    """v2: structured emoji delimiters, 0-shot. Heavy structural cues."""
    assert len(list_of_instructions) == len(list_of_docs)
    system_prompt = (
        "You will receive multiple CONVERSATION–RULE pairs.\n"
        "Each pair is enclosed in:\n"
        "🧱PAIR_START🧱\n"
        "...\n"
        "🧱PAIR_END🧱\n\n"
        "Inside each pair:\n"
        " • The CONVERSATION is between 📄CONV_START📄 and 📄CONV_END📄\n"
        " • The RULE is between 🎓RULE_START🎓 and 🎓RULE_END🎓\n\n"
        "The RULE tells you which preferences from that conversation to remember. "
        "When answering the user's question, use ONLY the preferences each rule allows you to "
        "retain — discount everything else, even if it appeared in the conversation.\n"
        "Reply with exactly one letter (A, B, C, or D).\n\n"
    )
    for doc, li in zip(list_of_docs, list_of_instructions):
        system_prompt += "🧱PAIR_START🧱\n"
        system_prompt += f"📄CONV_START📄\n{doc}\n📄CONV_END📄\n"
        system_prompt += f"🎓RULE_START🎓\n{li}\n🎓RULE_END🎓\n"
        system_prompt += "🧱PAIR_END🧱\n\n"
    return system_prompt
```

### `_get_pref_v4_system_prompt`

```python
def _get_pref_v4_system_prompt(*, list_of_docs: list, list_of_instructions: list) -> str:
    """v4: verbose tutorial / detailed task explanation, 0-shot."""
    assert len(list_of_instructions) == len(list_of_docs)
    system_prompt = (
        "You are an assistant whose job is to selectively retain user preferences from past "
        "conversations. Each conversation comes with a rule telling you exactly which "
        "preferences are in-scope for retention; everything outside that scope must be "
        "ignored when answering future questions, even if the conversation contained "
        "explicit signals about it.\n\n"
        "Concretely, when the user asks a new question, follow these steps internally:\n"
        "  1. For each previous conversation, read the rule first.\n"
        "  2. Identify which preferences in that conversation fall under the rule's topic.\n"
        "  3. Treat any other preferences from that conversation as if you never saw them.\n"
        "  4. Choose the answer that is most consistent with the union of in-scope preferences "
        "across all conversations.\n"
        "  5. If no in-scope preference applies to the question, pick the answer that a fresh "
        "assistant (no memory of these conversations) would give to a typical user.\n\n"
        "It is critical that you do not 'leak' an out-of-scope preference into your answer — "
        "the rule is binding even if the out-of-scope preference would have been clearly relevant.\n\n"
        "Below are the conversations and their rules. After them comes the user's question and "
        "four candidate answers labeled A, B, C, D. Reply with exactly one letter.\n\n"
    )
    for i, (doc, li) in enumerate(zip(list_of_docs, list_of_instructions), 1):
        system_prompt += f"[Conversation {i}]\nRule: {li}\nTranscript:\n{doc}\n\n"
    return system_prompt
```

### `_get_pref_v5_system_prompt`

```python
def _get_pref_v5_system_prompt(*, list_of_docs: list, list_of_instructions: list) -> str:
    """v5: few-shot (2 demos) + structured emoji delimiters. Heaviest template."""
    assert len(list_of_instructions) == len(list_of_docs)
    system_prompt = (
        "You will receive multiple CONVERSATION–RULE pairs.\n"
        "Each pair is enclosed in:\n"
        "🧱PAIR_START🧱\n"
        "...\n"
        "🧱PAIR_END🧱\n\n"
        "Inside each pair:\n"
        " • The CONVERSATION is between 📄CONV_START📄 and 📄CONV_END📄\n"
        " • The RULE is between 🎓RULE_START🎓 and 🎓RULE_END🎓\n\n"
        "The RULE tells you which preferences from that conversation to retain. When answering "
        "the user's question, use ONLY preferences each rule allows; discount the rest. "
        "Reply with exactly one letter (A, B, C, or D).\n\n"
        "Two worked EXAMPLE pairs follow, then the real ones.\n\n"
    )
    # Example 1 — LEARN target
    system_prompt += "🧱PAIR_START🧱\n"
    system_prompt += (
        "📄CONV_START📄\n"
        "user: I love a strong dark roast in the morning, usually with a splash of oat milk.\n"
        "user: I also recently picked up rock climbing on weekends.\n"
        "assistant: Got it — strong dark-roast coffee with oat milk, and rock climbing on weekends.\n"
        "📄CONV_END📄\n"
        "🎓RULE_START🎓\nPay attention only to preferences about coffee.\n🎓RULE_END🎓\n"
    )
    system_prompt += "🧱PAIR_END🧱\n\n"
    system_prompt += (
        "Example user question: \"What kind of drink would you suggest I order at a new coffee shop?\"\n"
        "  A: A light, fruity pour-over with no milk — bright and acidic.\n"
        "  B: A bold dark-roast latte with oat milk, since you'd enjoy that style.\n"
        "  C: A matcha latte for a change of pace.\n"
        "  D: An iced caramel macchiato with whole milk.\n"
        "Correct answer: B  (rule allows coffee preferences; the dark-roast oat-milk latte matches them.)\n\n"
    )
    # Example 2 — IGNORE target
    system_prompt += "🧱PAIR_START🧱\n"
    system_prompt += (
        "📄CONV_START📄\n"
        "user: I love a strong dark roast in the morning, usually with a splash of oat milk.\n"
        "user: I cook a lot of Thai food at home — green curry is my go-to.\n"
        "assistant: Strong dark-roast coffee with oat milk; green curry as your go-to home cook.\n"
        "📄CONV_END📄\n"
        "🎓RULE_START🎓\nPay attention only to preferences about cooking.\n🎓RULE_END🎓\n"
    )
    system_prompt += "🧱PAIR_END🧱\n\n"
    system_prompt += (
        "Example user question: \"What kind of drink would you suggest I order at a new coffee shop?\"\n"
        "  A: A light, fruity pour-over with no milk — bright and acidic.\n"
        "  B: A bold dark-roast latte with oat milk, since you'd enjoy that style.\n"
        "  C: A matcha latte for a change of pace.\n"
        "  D: An iced caramel macchiato with whole milk.\n"
        "Correct answer: A  (rule restricts retention to cooking; coffee preference must be ignored, "
        "so answer as a fresh assistant would.)\n\n"
        "=== END EXAMPLES — REAL PAIRS BEGIN ===\n\n"
    )
    for doc, li in zip(list_of_docs, list_of_instructions):
        system_prompt += "🧱PAIR_START🧱\n"
        system_prompt += f"📄CONV_START📄\n{doc}\n📄CONV_END📄\n"
        system_prompt += f"🎓RULE_START🎓\n{li}\n🎓RULE_END🎓\n"
        system_prompt += "🧱PAIR_END🧱\n\n"
    return system_prompt
```

## MemoryAgent variants

The released `memoryagent_icl` / `memoryagent_rag` use the few-shot checklist
distillation prompt with the checklist note-serving system prompt. Variants
explored during development (all evaluated on the short-horizon protocol; none
in the paper): a basic (non-checklist) distillation prompt, zero-shot versions
of both prompts, the v1 note-serving prompt
(`_get_selective_memory_system_prompt`, retained in `src/gnm.py` for the base
classes), a **system-prompt-only** serving variant (instruction restated in the
system prompt at query time), and a **query-time** variant (instruction appended
to each user query).
