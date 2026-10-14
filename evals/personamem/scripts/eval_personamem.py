"""Multi-choice preference-selectivity eval — first-token argmax + exact match.

Probes the full lower-triangular matrix [seq_idx_of_doc][seq_idx_of_io_pairs]:
after each `memorize` step `s`, probe every pair from docs `0..s`. This matches
the existing trainer/eval pattern in src/utils/metrics.py.

Per record we save the renamed/expanded letter fields:
  - preference_aligned_letter  (the answer aligned with the preference)
  - memoryllm_baseline_letter            (vanilla no-memory pick — copied from source)
  - target_letter              (preference_aligned if learn, baseline if ignore)
  - predicted_letter           (model's argmax)
  - is_target_pick             (predicted == target)
  - is_preference_pick         (predicted == preference_aligned)
  - is_baseline_pick           (predicted == baseline)

Plus full context: rollout_idx, seq_idx_of_doc (memory state),
seq_idx_of_io_pairs (probe origin), pair_idx, topic, learn_or_ignore,
pref_type, instruction_mode, source, preference, user_query, options,
letter_probs, and (if --save-verbose) full_prompt, learning_instruction, document.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO_ROOT))
from utils.checkpoints import normalize_wrapper_state_dict

from gnm import build_model  # noqa: E402
from config_memoryllm_train import (  # noqa: E402
    MemoryllmTrainConfig, ModelCfg, SamplingCfg, OptCfg, DataCfg,
    LoopCtrlCfg, DistributedCfg, LoggingCfg,
)

LETTERS = ["A", "B", "C", "D"]
SYSTEM_PROMPT = (
    "You are a helpful assistant. Below you will be given a question and four candidate "
    "responses labeled A, B, C, D. Reply with exactly one letter and nothing else."
)


def make_minimal_cfg(model_name: str, model_state_dict_path: str | None) -> MemoryllmTrainConfig:
    return MemoryllmTrainConfig(
        run_name="eval_personamem",
        seed=42,
        log=LoggingCfg(wandb_project="eval_personamem", batch_log_interval=100),
        distributed=DistributedCfg(cuda_visible_devices="0", nproc_per_node=1, master_port=29512),
        model=ModelCfg(model=model_name, model_state_dict_path=model_state_dict_path),
        sampling=SamplingCfg(max_new_tokens=1),
        loop=LoopCtrlCfg(),
        opt=None,
        data=DataCfg(),
    )


def load_model(cfg) -> torch.nn.Module:
    model = build_model(cfg)
    sd_path = cfg.model.model_state_dict_path
    if sd_path:
        print(f"[eval] loading state dict from {sd_path}")
        sd = torch.load(sd_path, map_location="cpu", weights_only=False)
        sd = normalize_wrapper_state_dict(sd)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[eval] missing {len(missing)}, unexpected {len(unexpected)}")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model


def render_prompt(tok, query: str, options: dict[str, str]) -> str:
    user = (
        f"Question: {query}\n\n"
        f"A: {options['A']}\n"
        f"B: {options['B']}\n"
        f"C: {options['C']}\n"
        f"D: {options['D']}\n\n"
        "Answer with a single letter (A, B, C, or D)."
    )
    return tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.inference_mode()
def predict_letters_batch(model, prompts: list[str], letter_ids: torch.Tensor
                            ) -> list[tuple[str, dict[str, float]]]:
    """Batched first-token argmax over A/B/C/D. Pads with left-padding so the last
    token of every sequence is the answer-letter position."""
    tok = model.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # left-pad so position [-1] is consistent across sequences
    prev_side = tok.padding_side
    tok.padding_side = "left"
    try:
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False
                   ).to(model.model.device)
    finally:
        tok.padding_side = prev_side
    mem_len = model.model.memory.shape[1]
    B, T = enc["input_ids"].shape
    mem_mask = enc["attention_mask"].new_ones((B, mem_len + 1))
    attn_full = torch.cat([mem_mask, enc["attention_mask"]], dim=1)
    out = model(input_ids=enc["input_ids"], attention_mask=attn_full)
    logits = out.logits if hasattr(out, "logits") else out[0]
    last_logits = logits[:, -1, :]                            # (B, V) — last position is answer-letter
    letter_logits = last_logits[:, letter_ids]                # (B, 4)
    probs = torch.softmax(letter_logits.float(), dim=-1)
    pred_idxs = letter_logits.argmax(dim=-1).tolist()
    out_recs = []
    for i, pi in enumerate(pred_idxs):
        out_recs.append((LETTERS[pi], {LETTERS[j]: float(probs[i, j].item()) for j in range(4)}))
    return out_recs


def _build_record(*, rollout_idx, seq_idx_of_doc, seq_idx_of_io_pairs, pair_idx, ex,
                   pair, prompt, pred_letter, letter_probs, save_verbose,
                   baseline_source: str):
    pref_letter = pair["correct_letter"]            # preference-aligned
    mem_base = pair.get("memoryllm_baseline_letter")
    lla_base = pair.get("llama3_baseline_letter")
    selected_base = pair.get(f"{baseline_source}_baseline_letter")
    is_collision = (pref_letter == selected_base) if selected_base else None
    loi = pair["learn_or_ignore"]
    target_letter = pref_letter if loi == "learn" else selected_base
    rec = {
        "rollout_idx": rollout_idx,
        "seq_idx_of_doc": seq_idx_of_doc,
        "seq_idx_of_io_pairs": seq_idx_of_io_pairs,
        "pair_idx": pair_idx,
        "topic": pair["topic"],
        "learn_or_ignore": loi,
        "pref_type": pair.get("pref_type"),
        "instruction_mode": ex["metadata"].get("instruction_mode"),
        "source": ex["metadata"].get("source"),
        "preference": pair["preference"],
        "user_query": pair["user_query"],
        "options": pair["options"],
        "preference_aligned_letter": pref_letter,
        "memoryllm_baseline_letter": mem_base,
        "llama3_baseline_letter": lla_base,
        "baseline_source": baseline_source,
        "selected_baseline_letter": selected_base,
        "is_collision": is_collision,
        "target_letter": target_letter,
        "predicted_letter": pred_letter,
        "letter_probs": letter_probs,
        "is_target_pick": pred_letter == target_letter if target_letter else None,
        "is_preference_pick": pred_letter == pref_letter,
        "is_baseline_pick": pred_letter == selected_base if selected_base else None,
        "is_memoryllm_baseline_pick": pred_letter == mem_base if mem_base else None,
        "is_llama3_baseline_pick": pred_letter == lla_base if lla_base else None,
    }
    if save_verbose:
        rec["full_prompt"] = prompt
        rec["learning_instruction"] = ex["learning_instruction"]
        rec["document"] = ex["document"]
    return rec


def run_eval_rollouts(model, eval_data, seq_len, n_rollouts, rollout_offset, output_path, save_verbose,
                       baseline_source: str = "memoryllm"):
    rollouts = [eval_data[i:i + seq_len] for i in range(0, len(eval_data), seq_len)
                if len(eval_data[i:i + seq_len]) == seq_len]
    rollouts = rollouts[rollout_offset:rollout_offset + n_rollouts]
    print(f"[eval] seq_len={seq_len}, rollouts={len(rollouts)} (offset={rollout_offset})")
    tok = model.tokenizer
    letter_ids = torch.tensor([tok(L, add_special_tokens=False)["input_ids"][0] for L in LETTERS],
                               device=model.model.device)
    results = []
    started = time.time()
    BATCH_SIZE = 12  # max probes per forward pass
    for rollout_idx, rollout in enumerate(rollouts):
        model.reset_memory()
        memorize_failed = False
        for s in range(seq_len):
            ex_s = rollout[s]
            try:
                model.memorize(learning_instruction=ex_s["learning_instruction"],
                               document=ex_s["document"])
            except Exception as e:
                print(f"[eval] rollout {rollout_idx} seq {s} memorize failed: {e}", flush=True)
                memorize_failed = True
                break

            # Collect ALL probes from docs[0..s] for this memory state, then batch.
            probe_specs = []  # (seq_idx_of_io_pairs, pair_idx, ex_p, pair, prompt)
            for p in range(s + 1):
                ex_p = rollout[p]
                for pair_idx, pair in enumerate(ex_p["input_output_pairs"]):
                    if not pair.get("correct_letter"):
                        continue
                    prompt = render_prompt(tok, pair["user_query"], pair["options"])
                    probe_specs.append((p, pair_idx, ex_p, pair, prompt))

            for i in range(0, len(probe_specs), BATCH_SIZE):
                chunk = probe_specs[i:i + BATCH_SIZE]
                prompts = [c[4] for c in chunk]
                try:
                    preds = predict_letters_batch(model, prompts, letter_ids)
                except Exception as e:
                    print(f"[eval] batch predict failed r{rollout_idx} s{s}: {e}", flush=True)
                    preds = [("?", {L: 0.25 for L in LETTERS})] * len(chunk)
                for (p, pair_idx, ex_p, pair, prompt), (pred_letter, letter_probs) in zip(chunk, preds):
                    rec = _build_record(
                        rollout_idx=rollout_idx + rollout_offset,
                        seq_idx_of_doc=s,
                        seq_idx_of_io_pairs=p,
                        pair_idx=pair_idx,
                        ex=ex_p, pair=pair, prompt=prompt,
                        pred_letter=pred_letter, letter_probs=letter_probs,
                        save_verbose=save_verbose,
                        baseline_source=baseline_source,
                    )
                    results.append(rec)
        if memorize_failed:
            continue
        elapsed = time.time() - started
        rate = (rollout_idx + 1) / elapsed if elapsed > 0 else 0
        print(f"[eval] rollout {rollout_idx + 1}/{len(rollouts)} ({rate:.2f} rollouts/s, "
              f"{len(results)} probes so far)", flush=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] DONE — {len(results)} probes -> {output_path}")
    return results


def aggregate(results: list[dict]) -> dict:
    """Headline scalars on the diagonal (seq_idx_of_doc == seq_idx_of_io_pairs).

    Four metrics are reported:
      - accuracy_overall:                rate is_target_pick on LEARN probes (any)
      - accuracy_excluding_collisions:   rate is_target_pick on LEARN probes where pref != baseline
      - selectivity_overall:             rate is_baseline_pick on IGNORE probes (any)
      - selectivity_excluding_collisions: rate is_baseline_pick on IGNORE probes where pref != baseline

    A "collision" is a probe where the chosen baseline already picks the preference-aligned letter
    (so memorize-vs-baseline behavior is indistinguishable on it). The baseline used here is whichever
    is active per the eval `baseline_source` setting (memoryllm or llama3).
    """
    buckets = {
        "accuracy_overall":               {"hits": 0, "n": 0},
        "accuracy_excluding_collisions":  {"hits": 0, "n": 0},
        "selectivity_overall":            {"hits": 0, "n": 0},
        "selectivity_excluding_collisions":{"hits": 0, "n": 0},
        "successfully_avoided_leaking_excluding_collisions": {"hits": 0, "n": 0},
    }
    for r in results:
        if r["seq_idx_of_doc"] != r["seq_idx_of_io_pairs"]:
            continue
        loi = r["learn_or_ignore"]
        is_collision = bool(r.get("is_collision"))
        if loi == "learn":
            buckets["accuracy_overall"]["n"] += 1
            buckets["accuracy_overall"]["hits"] += int(bool(r["is_target_pick"]))
            if not is_collision:
                buckets["accuracy_excluding_collisions"]["n"] += 1
                buckets["accuracy_excluding_collisions"]["hits"] += int(bool(r["is_target_pick"]))
        else:  # ignore
            buckets["selectivity_overall"]["n"] += 1
            buckets["selectivity_overall"]["hits"] += int(bool(r["is_baseline_pick"]))
            if not is_collision:
                buckets["selectivity_excluding_collisions"]["n"] += 1
                buckets["selectivity_excluding_collisions"]["hits"] += int(bool(r["is_baseline_pick"]))
                buckets["successfully_avoided_leaking_excluding_collisions"]["n"] += 1
                buckets["successfully_avoided_leaking_excluding_collisions"]["hits"] += int(not bool(r["is_preference_pick"]))

    def rate(d):
        return d["hits"] / d["n"] if d["n"] else None

    diag = {f"{k}_n": v["n"] for k, v in buckets.items()}
    diag.update({k: rate(v) for k, v in buckets.items()})

    return {
        "n_probes_total": len(results),
        "baseline_source": (results[0].get("baseline_source") if results else None),
        "diag": diag,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-path", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-state-dict-path", default=None)
    ap.add_argument("--output-path", type=Path, required=True)
    ap.add_argument("--seq-len", type=int, default=1)
    ap.add_argument("--n-rollouts", type=int, default=50)
    ap.add_argument("--rollout-offset", type=int, default=0)
    ap.add_argument("--save-verbose", action="store_true")
    ap.add_argument("--baseline-source", choices=["memoryllm", "llama3"], default="memoryllm",
                    help="Which baseline letter to use for ignore-probe target + collision detection.")
    ap.add_argument("--shuffle-seed", type=int, default=None,
                    help="Deterministically permute eval_data before chunking into rollouts "
                         "(seed-robustness runs; changes rollout composition only).")
    args = ap.parse_args()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = make_minimal_cfg(args.model, args.model_state_dict_path)
    print(f"[eval] building model={args.model}, state_dict={args.model_state_dict_path}, "
          f"baseline_source={args.baseline_source}")
    model = load_model(cfg)
    print(f"[eval] model ready")

    with open(args.eval_path) as f:
        eval_data = json.load(f)
    print(f"[eval] loaded {len(eval_data)} examples from {args.eval_path}")
    if args.shuffle_seed is not None:
        import random as _random
        _random.Random(args.shuffle_seed).shuffle(eval_data)
        print(f"[eval] shuffled eval_data with seed {args.shuffle_seed}")

    results = run_eval_rollouts(model, eval_data, args.seq_len, args.n_rollouts,
                                 args.rollout_offset, args.output_path, args.save_verbose,
                                 baseline_source=args.baseline_source)
    agg = aggregate(results)
    print(json.dumps(agg, indent=2))

    summary_path = args.output_path.with_suffix(".summary.json")
    with open(summary_path, "w") as f:
        json.dump({"aggregate": agg, "args": vars(args), "n_probes": len(results)}, f, indent=2, default=str)
    print(f"[eval] summary -> {summary_path}")


if __name__ == "__main__":
    main()
