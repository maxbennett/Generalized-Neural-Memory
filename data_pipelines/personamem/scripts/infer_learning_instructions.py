"""Inferred learning instructions (appendix): a frozen LLM authors the instruction.

The user's implicit intent utterance (implicit_instructions.py — the same
round-robin variant as the implicit-instruction eval) is embedded in the
conversation as an ordinary user turn. A frozen, off-the-shelf LLM then reads
ONLY the conversation transcript (never the probes, answer options, or any topic
ontology) and writes a one-sentence learning instruction itself; the frozen GNM
and ICL-FT checkpoints consume that instruction verbatim. Two authoring prompts:

  author          the system prompt specifies the desired instruction FORMAT
                  (one sentence, one category, inclusive scope) using example
                  topics that never appear in the eval; the utterance is the
                  final user turn and the prompt points at it.  (paper table)
  author_generic  no format constraints beyond a single example and no pointer
                  to the intent signal; the utterance is inserted at a random
                  position in the transcript.  (paper text: "fully generic
                  authoring prompt")

The output eval JSON keeps documents/probes/baseline letters byte-identical to
the source (only `learning_instruction` changes), so the paper collision filter
and probe pairing stay valid. A sidecar audit JSON records per-example raw
outputs and the authored instruction.
"""

import argparse
import copy
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from implicit_instructions import IMPLICIT_INSTRUCTIONS
from build_implicit_eval import match_template

REPO = Path(__file__).resolve().parents[3]

UPSTREAMS = {
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "qwen3": "Qwen/Qwen3-14B",
}

# author mode: the upstream writes the learning instruction itself. The system
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

USER_TMPL_AUTHOR_GENERIC = (
    "Read the following conversation transcript.\n\n<transcript>\n{conv}\n</transcript>\n\n"
    "The user may indicate, directly or indirectly, why they are sharing this "
    "information. Write the learning instruction specifying what the memory system "
    "should memorize from this conversation to serve the user. Output only the "
    "instruction sentence, nothing else."
)

SYSTEM = {"author": SYSTEM_AUTHOR, "author_generic": SYSTEM_AUTHOR_GENERIC}
MAX_NEW = {"author": 64, "author_generic": 96}


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
    ap.add_argument("--mode", choices=list(SYSTEM), default="author")
    ap.add_argument("--source", type=Path,
                    default=REPO / "data/personamem/val_ood.json")
    ap.add_argument("--outdir", type=Path, default=REPO / "data/personamem/inferred_eval")
    ap.add_argument("--limit", type=int, default=None, help="audit mode: only first N examples")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    data = json.load(open(args.source))
    if args.limit:
        data = data[:args.limit]
    print(f"[infer-li] {len(data)} examples, upstream={args.upstream}, mode={args.mode}")

    tok = AutoTokenizer.from_pretrained(UPSTREAMS[args.upstream], padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        UPSTREAMS[args.upstream], torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    # Same per-topic round-robin variant assignment as build_implicit_eval.py, so
    # each example carries the identical utterance as its implicit-eval counterpart.
    per_topic_idx = defaultdict(int)
    rows = []
    for i, ex in enumerate(data):
        t = ex["metadata"]["learn_topics"][0]
        k = per_topic_idx[t]; per_topic_idx[t] += 1
        utt = IMPLICIT_INSTRUCTIONS[t][k % len(IMPLICIT_INSTRUCTIONS[t])]
        insert_pos = None
        if args.mode == "author_generic":
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
            prompt = USER_TMPL_AUTHOR.format(conv=conv, final_turn=utt)
        rows.append({"idx": i, "topic_true": t, "variant": k % len(IMPLICIT_INSTRUCTIONS[t]),
                     "utterance": utt, "prompt": prompt,
                     "insert_after_line": insert_pos,
                     "oracle_template": match_template(ex["learning_instruction"])})

    system, max_new = SYSTEM[args.mode], MAX_NEW[args.mode]
    for s in range(0, len(rows), args.batch_size):
        chunk = rows[s:s + args.batch_size]
        outs = generate_batch(model, tok, [r["prompt"] for r in chunk],
                              args.upstream == "qwen3", system, max_new)
        for r, raw in zip(chunk, outs):
            r["raw_output"] = raw
            li = raw.strip().split("\n")[0].strip().strip('"').strip()
            if li and not li.endswith("."):
                li += "."
            r["learning_instruction"] = li
            # proxy only (true measure is the end-to-end eval): does the
            # authored scope mention the true topic term?
            r["inference_correct"] = r["topic_true"].lower().rstrip("s")[:6] in li.lower()
            r["phrase"] = li
            r["matched_topic"] = None
        done = min(s + args.batch_size, len(rows))
        if done % 80 < args.batch_size or done == len(rows):
            acc = sum(r["inference_correct"] for r in rows[:done]) / done
            print(f"[infer-li] {done}/{len(rows)} inference_acc={acc:.3f}", flush=True)

    for r in rows:
        del r["prompt"]

    n_ok = sum(r["inference_correct"] for r in rows)
    print(f"[infer-li] inference accuracy (proxy) {n_ok}/{len(rows)} = {n_ok/len(rows):.3f}")
    print("[infer-li] per-topic inference accuracy:")
    by_t = defaultdict(list)
    for r in rows:
        by_t[r["topic_true"]].append(r["inference_correct"])
    for t in sorted(by_t):
        v = by_t[t]
        print(f"    {t:<12} {sum(v)}/{len(v)}")
    print("[infer-li] instructions that miss the topic term:", Counter(
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
        out_path = args.outdir / f"val_ood_inferred_{args.upstream}_{args.mode}.json"
        json.dump(out, open(out_path, "w"))
        print(f"[infer-li] wrote {out_path}")


if __name__ == "__main__":
    main()
