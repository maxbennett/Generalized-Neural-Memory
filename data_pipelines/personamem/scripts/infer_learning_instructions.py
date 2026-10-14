"""Autonomous infer->execute pipeline: a frozen LLM authors the learning instruction.

Stage 1: embed the v3 implicit intent utterance as an ordinary final user turn in
the conversation (the transcript never names the topic as a category — v3 bank is
stem-free).
Stage 2: a frozen off-the-shelf LLM reads ONLY the conversation (no probes, no
options, no ground-truth label) and emits the single topic that defines what
should be retained.
Stage 3: constrained render — the topic phrase is slotted into the SAME inclusive
template as the example's oracle instruction, with the bare lowercase topic name
(explicit_bare C1 == C0 showed descriptors contribute nothing). When the upstream
classifies correctly, the rendered instruction is byte-identical to the
explicit_bare oracle instruction, so end-to-end degradation decomposes exactly
into inference errors vs execution errors.

The output eval JSON keeps documents/probes/baseline letters byte-identical to
the source (only `learning_instruction` changes), so the paper collision filter
and probe pairing stay valid. A sidecar audit JSON records per-example raw
outputs, the matched topic, and inference correctness.
"""

import argparse
import copy
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from implicit_instructions import IMPLICIT_INSTRUCTIONS_V3
from build_implicit_eval import TEMPLATES, match_template

REPO = Path(__file__).resolve().parents[3]

UPSTREAMS = {
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "qwen3": "Qwen/Qwen3-14B",
}

SYSTEM = (
    "You are the memory controller for an AI assistant. After a conversation ends, "
    "you decide what the assistant should commit to long-term memory for future "
    "interactions with this user."
)

# Author mode: the upstream writes the learning instruction itself. The system
# prompt teaches the instruction FORM (single sentence, inclusive scope) using
# example topics that never appear in the eval; the scope itself is the model's
# own choice — no candidate set anywhere in the pipeline.
SYSTEM_AUTHOR = (
    "You are the memory controller for an AI assistant. After a conversation ends, "
    "you write a one-sentence learning instruction that tells the assistant's "
    "memory system which of the user's preferences to memorize from that "
    "conversation. The instruction names exactly ONE category of preferences, as a "
    "single everyday noun for that domain (like \"cooking\" or \"fitness\"), in an "
    "inclusive form such as: \"Only remember preferences related to cooking.\" or "
    "\"Track only fitness preferences from this conversation.\" Do not use broad "
    "umbrella terms (like \"entertainment\", \"hobbies\", \"lifestyle\", or "
    "\"leisure activities\"), do not list multiple areas, and do not add any "
    "extra clauses or qualifiers. Choose the category yourself; output only the "
    "instruction sentence."
)

USER_TMPL_AUTHOR = (
    "Read the following conversation transcript.\n\n<transcript>\n{conv}\n</transcript>\n\n"
    "The conversation covers the user's preferences on several distinct topics. The "
    "user's final message — \"{final_turn}\" — indicates, directly or indirectly, "
    "why they are sharing this information. Write the one-sentence learning "
    "instruction specifying which preferences the memory system should memorize to "
    "serve that purpose. Output only the instruction sentence, nothing else."
)

# author_generic mode: no category/format constraints beyond
# a single example, and no pointer to where the intent signal sits — the utterance
# is inserted at a RANDOM position in the transcript and the upstream must find it.
SYSTEM_AUTHOR_GENERIC = (
    "You are the memory controller for an AI assistant. After a conversation ends, "
    "you write a short learning instruction that tells the assistant's memory "
    "system what to memorize from that conversation for future interactions with "
    "this user (for example: \"Only remember preferences related to cooking.\"). "
    "Decide what should be memorized based on the conversation; output only the "
    "instruction sentence."
)

# author_purpose mode (2026-08-02): the instruction scopes memory by the user's
# stated PURPOSE/use-case rather than a category term. Explicit exclusivity
# ("Only remember...") + use-case referent — the regime where write-time
# selection has to interpret the scope semantically.
SYSTEM_AUTHOR_PURPOSE = (
    "You are the memory controller for an AI assistant. After a conversation "
    "ends, you write a one-sentence learning instruction that tells the "
    "assistant's memory system what to memorize from that conversation for "
    "future interactions with this user. State the scope in terms of what the "
    "user wants the assistant to use the information for, in an inclusive form, "
    "for example: \"Only remember what is relevant to giving this user cooking "
    "advice.\" The instruction names only the purpose, exactly like the "
    "example — it must not mention any specific detail, item, or preference "
    "from the conversation; the memory system reads the conversation itself. "
    "If the user indicates why they are sharing this information, scope the "
    "instruction to that purpose and exclude everything else. If no purpose is "
    "indicated, instruct the memory system to remember everything. Output only "
    "the instruction sentence."
)

# author_intent mode (2026-08-02): generic prompt plus the memory POLICY — a
# stated purpose defines the scope; no purpose -> remember everything (the
# MemoryLLM degenerate case). No topics/categories/dataset specifics anywhere.
SYSTEM_AUTHOR_INTENT = (
    "You are the memory controller for an AI assistant. After a conversation "
    "ends, you write a one-sentence learning instruction that tells the "
    "assistant's memory system what to memorize from that conversation for "
    "future interactions with this user. The instruction states the scope as a "
    "category of preferences (for example: \"Only remember preferences related "
    "to cooking.\") — it is not a summary of what the user said; the memory "
    "system reads the conversation itself. If the user indicates why they are "
    "sharing this information, infer from that purpose which category of their "
    "preferences they want remembered, and exclude the rest; use the most "
    "specific category that matches the purpose, so the scope does not admit "
    "unrelated information. If no purpose is indicated, instruct the memory "
    "system to remember everything. Output only the instruction sentence."
)

USER_TMPL_AUTHOR_GENERIC = (
    "Read the following conversation transcript.\n\n<transcript>\n{conv}\n</transcript>\n\n"
    "The user may indicate, directly or indirectly, why they are sharing this "
    "information. Write the learning instruction specifying what the memory system "
    "should memorize from this conversation to serve the user. Output only the "
    "instruction sentence, nothing else."
)

USER_TMPL_FREEFORM = (
    "Read the following conversation transcript.\n\n<transcript>\n{conv}\n</transcript>\n\n"
    "The conversation covers the user's preferences on several distinct topics. The "
    "user's final message indicates, directly or indirectly, why they are sharing "
    "this information. Identify the single topic whose preferences should be "
    "retained to serve that purpose. Answer with only the topic name — a single "
    "common noun like 'cooking' or 'fitness' — nothing else."
)

# Menu mode: candidate set = the topics present in the conversation (alphabetical —
# metadata lists the learn topic first, which would be a positional leak). In a
# deployed system this set comes from a trivial topic-tagging pass; the tested
# inference step is mapping the user's stated purpose to the right one.
USER_TMPL_MENU = (
    "Read the following conversation transcript.\n\n<transcript>\n{conv}\n</transcript>\n\n"
    "The conversation covers the user's preferences on these topics: {menu}. The "
    "user's final message — \"{final_turn}\" — indicates, directly or indirectly, "
    "why they are sharing this information. Identify the single topic whose "
    "preferences should be retained to serve that purpose. Answer with exactly one "
    "topic from that list, nothing else."
)


def normalize(raw: str) -> str:
    """First line of the generation, stripped to a bare lowercase phrase."""
    s = raw.strip().split("\n")[0].strip()
    s = re.sub(r"(?i)^(the\s+)?topic\s*(is|:)?\s*", "", s)
    s = re.sub(r"[^a-zA-Z\s-]", "", s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    # "music taste" / "travel preferences" -> head noun; generic tails carry no scope
    s = re.sub(r"\s+(tastes?|preferences?|interests?|choices?|habits?)$", "", s)
    return s


def match_topic(phrase: str, topics: list[str]) -> str | None:
    """Exact or singular/plural match against canonical topic names."""
    canon = {t.lower(): t for t in topics}
    if phrase in canon:
        return canon[phrase]
    for t in topics:
        tl = t.lower()
        if phrase.rstrip("s") == tl.rstrip("s"):
            return t
    return None


@torch.inference_mode()
def generate_batch(model, tok, prompts: list[str], is_qwen: bool, system: str,
                   max_new_tokens: int) -> list[str]:
    kwargs = {"enable_thinking": False} if is_qwen else {}
    texts = [
        tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True, **kwargs)
        for p in prompts
    ]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    gen = out[:, enc["input_ids"].shape[1]:]
    return tok.batch_decode(gen, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", choices=list(UPSTREAMS), required=True)
    ap.add_argument("--mode",
                    choices=["author", "author_generic", "author_intent", "author_intent_end",
                             "author_purpose", "menu", "freeform"],
                    default="author")
    ap.add_argument("--source", type=Path,
                    default=REPO / "data/personamem/test.json")
    ap.add_argument("--outdir", type=Path, default=REPO / "data/personamem/inferred_eval")
    ap.add_argument("--limit", type=int, default=None, help="audit mode: only first N examples")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    data = json.load(open(args.source))
    if args.limit:
        data = data[:args.limit]
    topics = list(IMPLICIT_INSTRUCTIONS_V3)
    print(f"[infer-li] {len(data)} examples, upstream={args.upstream}")

    tok = AutoTokenizer.from_pretrained(UPSTREAMS[args.upstream], padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        UPSTREAMS[args.upstream], torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    # Same per-topic round-robin variant assignment as build_implicit_eval.py, so
    # each example carries the identical utterance as its implicit_v3 counterpart.
    per_topic_idx = defaultdict(int)
    rows = []
    for i, ex in enumerate(data):
        t = ex["metadata"]["learn_topics"][0]
        k = per_topic_idx[t]; per_topic_idx[t] += 1
        utt = IMPLICIT_INSTRUCTIONS_V3[t][k % len(IMPLICIT_INSTRUCTIONS_V3[t])]
        menu = sorted(ex["metadata"]["topics_in_doc"])
        insert_pos = None
        if args.mode == "author_intent_end":
            # intent-policy prompt, but utterance at the end (no pointer to it in
            # the prompt — isolates the placement variable vs author_intent)
            conv = ex["document"] + "\nUser: " + utt
            prompt = USER_TMPL_AUTHOR_GENERIC.format(conv=conv)
        elif args.mode in ("author_generic", "author_intent"):
            # insert the intent utterance as its own user turn at a random point
            # in the transcript (deterministic per example)
            lines = ex["document"].split("\n")
            user_lines = [j for j, l in enumerate(lines) if l.startswith("User: ")]
            insert_pos = random.Random(9173 + i).choice(user_lines)
            lines.insert(insert_pos + 1, "User: " + utt)
            conv = "\n".join(lines)
            prompt = USER_TMPL_AUTHOR_GENERIC.format(conv=conv)
        else:
            conv = ex["document"] + "\nUser: " + utt
            if args.mode == "menu":
                prompt = USER_TMPL_MENU.format(conv=conv, menu=", ".join(menu), final_turn=utt)
            elif args.mode in ("author", "author_purpose"):
                prompt = USER_TMPL_AUTHOR.format(conv=conv, final_turn=utt)
            else:
                prompt = USER_TMPL_FREEFORM.format(conv=conv)
        rows.append({"idx": i, "topic_true": t, "variant": k % len(IMPLICIT_INSTRUCTIONS_V3[t]),
                     "utterance": utt, "prompt": prompt, "menu": menu,
                     "insert_after_line": insert_pos,
                     "oracle_template": match_template(ex["learning_instruction"])})

    system = {"author": SYSTEM_AUTHOR, "author_generic": SYSTEM_AUTHOR_GENERIC,
              "author_intent": SYSTEM_AUTHOR_INTENT,
              "author_intent_end": SYSTEM_AUTHOR_INTENT,
              "author_purpose": SYSTEM_AUTHOR_PURPOSE}.get(args.mode, SYSTEM)
    max_new = {"author": 64, "author_generic": 96, "author_intent": 96,
               "author_intent_end": 96, "author_purpose": 64}.get(args.mode, 16)
    for s in range(0, len(rows), args.batch_size):
        chunk = rows[s:s + args.batch_size]
        outs = generate_batch(model, tok, [r["prompt"] for r in chunk],
                              args.upstream == "qwen3", system, max_new)
        for r, raw in zip(chunk, outs):
            r["raw_output"] = raw
            if args.mode.startswith("author"):
                li = raw.strip().split("\n")[0].strip().strip('"').strip()
                if li and not li.endswith("."):
                    li += "."
                r["learning_instruction"] = li
                # proxy only (true measure is the end-to-end eval): does the
                # authored scope mention the true topic term?
                r["inference_correct"] = r["topic_true"].lower().rstrip("s")[:6] in li.lower()
                r["phrase"] = li
                r["matched_topic"] = None
            else:
                phrase = normalize(raw)
                vocab = r["menu"] if args.mode == "menu" else topics
                matched = match_topic(phrase, vocab)
                r["phrase"] = phrase
                r["matched_topic"] = matched
                r["inference_correct"] = matched == r["topic_true"]
                slot = matched.lower() if matched else phrase
                r["learning_instruction"] = r["oracle_template"].format(t=slot)
        done = min(s + args.batch_size, len(rows))
        if done % 80 < args.batch_size or done == len(rows):
            acc = sum(r["inference_correct"] for r in rows[:done]) / done
            print(f"[infer-li] {done}/{len(rows)} inference_acc={acc:.3f}", flush=True)

    for r in rows:
        del r["prompt"]

    n_ok = sum(r["inference_correct"] for r in rows)
    n_matched = sum(r["matched_topic"] is not None for r in rows)
    print(f"[infer-li] inference accuracy {n_ok}/{len(rows)} = {n_ok/len(rows):.3f}; "
          f"matched-to-vocab {n_matched}/{len(rows)}")
    print("[infer-li] per-topic inference accuracy:")
    by_t = defaultdict(list)
    for r in rows:
        by_t[r["topic_true"]].append(r["inference_correct"])
    for t in sorted(by_t):
        v = by_t[t]
        print(f"    {t:<12} {sum(v)}/{len(v)}")
    print("[infer-li] wrong/unmatched phrases:", Counter(
        (r["topic_true"], r["phrase"]) for r in rows if not r["inference_correct"]))

    args.outdir.mkdir(parents=True, exist_ok=True)
    audit_path = args.outdir / f"inferred_{args.upstream}_{args.mode}_audit.json"
    json.dump(rows, open(audit_path, "w"), indent=1)
    print(f"[infer-li] wrote {audit_path}")

    if not args.limit:
        out = copy.deepcopy(json.load(open(args.source)))
        for r in rows:
            out[r["idx"]]["learning_instruction"] = r["learning_instruction"]
        src = json.load(open(args.source))
        for o, n in zip(src, out):
            assert {k: v for k, v in o.items() if k != "learning_instruction"} == \
                   {k: v for k, v in n.items() if k != "learning_instruction"}
        out_path = args.outdir / f"val_unseen_topics_inferred_{args.upstream}_{args.mode}.json"
        json.dump(out, open(out_path, "w"))
        print(f"[infer-li] wrote {out_path}")


if __name__ == "__main__":
    main()
