"""ICL preference-MC eval — drives Llama-3-8B-Instruct (or any registered ICL
wrapper) through the same lower-triangular matrix evaluation as
eval_personamem.py, but uses the wrapper's own
render_mc_prompt() so the system-prompt template is selected by the
registry key (the paper uses llama3_8b_instruct_personamem).

Output is written in the results/<run>/<config>/ summary layout:
  - run_config.json
  - temp_results_val_id_epoch_0.jsonl   (copy of per-probe results)
  - summary.json                          (matrix / retention / diag / most_recent)
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "eval_personamem" / "scripts"))
from utils.checkpoints import normalize_wrapper_state_dict

from gnm import build_model  # noqa: E402
from config_memoryllm_train import (  # noqa: E402
    MemoryllmTrainConfig, ModelCfg, SamplingCfg, OptCfg, DataCfg,
    LoopCtrlCfg, DistributedCfg, LoggingCfg,
)
from build_summary import summarize_one  # noqa: E402

LETTERS = ["A", "B", "C", "D"]


def make_minimal_cfg(model_name: str) -> MemoryllmTrainConfig:
    return MemoryllmTrainConfig(
        run_name="eval_icl_pref",
        seed=42,
        log=LoggingCfg(wandb_project="eval_icl_pref", batch_log_interval=100),
        distributed=DistributedCfg(cuda_visible_devices="0", nproc_per_node=1, master_port=29512),
        model=ModelCfg(model=model_name),
        sampling=SamplingCfg(max_new_tokens=1),
        loop=LoopCtrlCfg(),
        opt=None,
        data=DataCfg(),
    )


@torch.inference_mode()
def predict_letter(model_wrapper, prompt: str, letter_ids: torch.Tensor) -> tuple[str, dict]:
    """First-token argmax over A/B/C/D using the wrapper's underlying model."""
    tok = model_wrapper.tokenizer
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False, truncation=True,
              max_length=8000).to(model_wrapper.model.device)
    out = model_wrapper.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    logits = out.logits if hasattr(out, "logits") else out[0]
    last = logits[0, -1, :]
    letter_logits = last[letter_ids]
    probs = torch.softmax(letter_logits.float(), dim=-1)
    pred_idx = int(letter_logits.argmax().item())
    return LETTERS[pred_idx], {LETTERS[i]: float(probs[i].item()) for i in range(4)}


def _build_record(*, rollout_idx, seq_idx_of_doc, seq_idx_of_io_pairs, pair_idx, ex,
                   pair, prompt, pred_letter, letter_probs, save_verbose,
                   baseline_source: str = "memoryllm"):
    pref_letter = pair["correct_letter"]
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


def run_eval(model_wrapper, eval_data, seq_len, n_rollouts, rollout_offset,
              results_path, save_verbose, baseline_source: str = "memoryllm"):
    rollouts = [eval_data[i:i + seq_len] for i in range(0, len(eval_data), seq_len)
                if len(eval_data[i:i + seq_len]) == seq_len]
    if rollout_offset:
        rollouts = rollouts[rollout_offset:]
    if n_rollouts:
        rollouts = rollouts[:n_rollouts]
    print(f"[icl-pref] seq_len={seq_len}, rollouts={len(rollouts)} (offset={rollout_offset})")
    tok = model_wrapper.tokenizer
    letter_ids = torch.tensor([tok(L, add_special_tokens=False)["input_ids"][0] for L in LETTERS],
                               device=model_wrapper.model.device)
    results = []
    started = time.time()
    for rollout_idx, rollout in enumerate(rollouts):
        model_wrapper.reset_memory()
        for s in range(seq_len):
            ex_s = rollout[s]
            try:
                model_wrapper.memorize(learning_instruction=ex_s["learning_instruction"],
                                        document=ex_s["document"])
            except Exception as e:
                print(f"[icl-pref] memorize failed at r{rollout_idx} s{s}: {e}", flush=True)
                break
            for p in range(s + 1):
                ex_p = rollout[p]
                for pair_idx, pair in enumerate(ex_p["input_output_pairs"]):
                    if not pair.get("correct_letter"):
                        continue
                    prompt = model_wrapper.render_mc_prompt(pair["user_query"], pair["options"])
                    try:
                        pred, probs = predict_letter(model_wrapper, prompt, letter_ids)
                    except Exception as e:
                        pred = "?"
                        probs = {L: 0.25 for L in LETTERS}
                        print(f"[icl-pref] predict failed at r{rollout_idx} s{s} p{p}: {e}",
                              flush=True)
                    rec = _build_record(
                        rollout_idx=rollout_idx + rollout_offset,
                        seq_idx_of_doc=s, seq_idx_of_io_pairs=p, pair_idx=pair_idx,
                        ex=ex_p, pair=pair, prompt=prompt,
                        pred_letter=pred, letter_probs=probs, save_verbose=save_verbose,
                        baseline_source=baseline_source,
                    )
                    results.append(rec)
        if (rollout_idx + 1) % 1 == 0:
            elapsed = time.time() - started
            rate = (rollout_idx + 1) / elapsed if elapsed > 0 else 0
            print(f"[icl-pref] rollout {rollout_idx + 1}/{len(rollouts)} "
                  f"({rate:.2f} rollouts/s, {len(results)} probes so far)", flush=True)
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[icl-pref] DONE — {len(results)} probes -> {results_path}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-path", type=Path, required=True,
                    help="Path to val JSON (e.g. val_seen_topics.json).")
    ap.add_argument("--model", default="llama3_8b_instruct_personamem",
                    help="Registry key (the paper uses llama3_8b_instruct_personamem).")
    ap.add_argument("--model-state-dict-path", default=None,
                    help="Optional path to a trained checkpoint to load over the registry's pretrained init.")
    ap.add_argument("--save-root", type=Path, required=True,
                    help="Output directory (summary layout).")
    ap.add_argument("--seq-len", type=int, default=10)
    ap.add_argument("--n-rollouts", type=int, default=None,
                    help="None = all rollouts in file (after offset).")
    ap.add_argument("--rollout-offset", type=int, default=0)
    ap.add_argument("--save-verbose", action="store_true")
    ap.add_argument("--baseline-source", choices=["memoryllm", "llama3"], default="memoryllm")
    ap.add_argument("--shuffle-seed", type=int, default=None,
                    help="Deterministically permute eval_data before chunking into rollouts "
                         "(seed-robustness runs; changes rollout composition only).")
    args = ap.parse_args()

    args.save_root.mkdir(parents=True, exist_ok=True)
    print(f"[icl-pref] model={args.model}, save_root={args.save_root}")

    cfg = make_minimal_cfg(args.model)
    model = build_model(cfg)
    if args.model_state_dict_path:
        print(f"[icl-pref] loading state dict from {args.model_state_dict_path}")
        sd = torch.load(args.model_state_dict_path, map_location="cpu", weights_only=False)
        sd = normalize_wrapper_state_dict(sd)
        if isinstance(sd, dict) and "model" in sd and not any(k.startswith("model.") for k in sd):
            sd = sd["model"]
        sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[icl-pref] missing {len(missing)}, unexpected {len(unexpected)}")
    model.model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.model.eval()
    print(f"[icl-pref] model ready: {type(model).__name__}")

    with open(args.eval_path) as f:
        eval_data = json.load(f)
    print(f"[icl-pref] loaded {len(eval_data)} examples from {args.eval_path}")
    if args.shuffle_seed is not None:
        import random as _random
        _random.Random(args.shuffle_seed).shuffle(eval_data)
        print(f"[icl-pref] shuffled eval_data with seed {args.shuffle_seed}")

    results_path = args.save_root / "results.json"
    results = run_eval(model, eval_data, args.seq_len, args.n_rollouts,
                        args.rollout_offset, results_path, args.save_verbose,
                        baseline_source=args.baseline_source)

    # 1. run_config.json — record what we ran
    run_cfg = {
        "model": args.model,
        "eval_path": str(args.eval_path),
        "seq_len": args.seq_len,
        "n_rollouts": args.n_rollouts,
        "rollout_offset": args.rollout_offset,
        "n_probes": len(results),
        "wrapper_class": type(model).__name__,
    }
    with open(args.save_root / "run_config.json", "w") as f:
        json.dump(run_cfg, f, indent=2, default=str)

    # 2. summary.json — matrix / retention / diag / most_recent for each metric
    summary = summarize_one(results, seq_len=args.seq_len)
    with open(args.save_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 3. temp_results_*.jsonl — per-probe full data (val_seen → val_id epoch_0)
    shutil.copy2(results_path, args.save_root / "temp_results_val_id_epoch_0.jsonl")

    # ---- Quick CLI summary (overall + excluding_collisions for both metrics) ----
    ao = summary["accuracy_overall"]
    ax = summary["accuracy_excluding_collisions"]
    so = summary["selectivity_overall"]
    sx = summary["selectivity_excluding_collisions"]
    def H(x, y):
        if x is None or y is None or (x + y) == 0:
            return None
        return 2 * x * y / (x + y)
    fmt = lambda v: f"{v:.4f}" if v is not None else "  -  "
    print()
    print(f"[icl-pref] {args.model}  baseline_source={args.baseline_source}  ({len(results)} probes)")
    print(f"  diag full        acc={fmt(ao['full'])}  acc_xcoll={fmt(ax['full'])}  "
          f"sel={fmt(so['full'])}  sel_xcoll={fmt(sx['full'])}  "
          f"H(acc_xcoll,sel_xcoll)={fmt(H(ax['full'], sx['full']))}")
    print(f"  diag most_recent acc={fmt(ao['most_recent'])}  acc_xcoll={fmt(ax['most_recent'])}  "
          f"sel={fmt(so['most_recent'])}  sel_xcoll={fmt(sx['most_recent'])}  "
          f"H(acc_xcoll,sel_xcoll)={fmt(H(ax['most_recent'], sx['most_recent']))}")
    print(f"  outputs -> {args.save_root}/{{summary.json, run_config.json, temp_results_val_id_epoch_0.jsonl}}")


if __name__ == "__main__":
    main()
