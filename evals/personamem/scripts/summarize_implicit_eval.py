"""Summarize the implicit-instruction eval runs and compare against the paper run.

Adapted from merge_personamem_eval.py (single-file runs, no shards). Enriches records
with trained-baseline letters from the original all_probes eval file (probes in
the implicit/bare variants are byte-identical, only learning_instruction
changed), then builds summaries under both collision modes and prints a
side-by-side against personamem/gnm.

Usage:
  python summarize_implicit_eval.py            # both conditions
  python summarize_implicit_eval.py implicit   # one condition
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "evals/personamem/scripts"
sys.path.insert(0, str(SCRIPTS))

from build_summary import _matrix_for, _retention_curve, build_metrics, _is_collision_any  # noqa: E402

RES = REPO / "evals/personamem/paper_results/implicit_instructions"
SOURCE_EVAL = REPO / "data/personamem/val_ood.json"
PAPER_SUMMARY = (REPO / "evals/personamem/paper_results/short_horizon/"
                 "gnm/summary.json")
SEQ_LEN = 10

METRICS = [
    "accuracy_overall",
    "accuracy_excluding_collisions",
    "selectivity_overall",
    "selectivity_excluding_collisions",
    "successfully_avoided_leaking_overall",
    "successfully_avoided_leaking_excluding_collisions",
]


def probe_hash(query: str, options: dict) -> str:
    canonical = f"Q:{query}\nA:{options['A']}\nB:{options['B']}\nC:{options['C']}\nD:{options['D']}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def is_collision_any_trained(r: dict) -> bool:
    pa = r.get("preference_aligned_letter") or r.get("correct_letter")
    return (pa == r.get("memoryllm_trained_baseline_letter")) or (pa == r.get("icl_ft_baseline_letter"))


def summarize(records, collision_mode):
    n_pre = len(records)
    if collision_mode == "any":
        records = [r for r in records if not _is_collision_any(r)]
    elif collision_mode == "any_trained":
        records = [r for r in records if not is_collision_any_trained(r)]
    elif collision_mode != "selected":
        raise ValueError(collision_mode)
    metrics = build_metrics("selected")
    summary = {
        "n_probes_total": len(records),
        "n_probes_pre_filter": n_pre,
        "seq_len": SEQ_LEN,
        "collision_mode": collision_mode,
    }
    for name, (eligible, hit) in metrics.items():
        matrix, counts = _matrix_for(records, eligible, hit, SEQ_LEN)
        diag = np.array([matrix[i, i] if counts[i, i] > 0 else 0.0 for i in range(SEQ_LEN)])
        diag_counts = np.array([counts[i, i] for i in range(SEQ_LEN)])
        diag_total_hits = sum(matrix[i, i] * counts[i, i] for i in range(SEQ_LEN) if counts[i, i] > 0)
        diag_total_n = int(diag_counts.sum())
        full = (diag_total_hits / diag_total_n) if diag_total_n else None
        i = SEQ_LEN - 1
        summary[name] = {
            "full": full,
            "full_n": diag_total_n,
            "most_recent": float(matrix[i, i]) if counts[i, i] > 0 else None,
            "most_recent_n": int(counts[i, i]),
            "over_time": diag.tolist(),
            "over_time_counts": diag_counts.tolist(),
            "retention": _retention_curve(matrix, counts, SEQ_LEN)[0],
            "retention_counts": _retention_curve(matrix, counts, SEQ_LEN)[1],
            "matrix": matrix.tolist(),
            "matrix_counts": counts.tolist(),
        }
    return summary


def main():
    conditions = sys.argv[1:] or ["implicit"]

    source = json.load(open(SOURCE_EVAL))
    by_hash = {}
    for d in source:
        for p in d.get("input_output_pairs", []):
            opts = p.get("options")
            if not opts:
                continue
            by_hash[probe_hash(p["user_query"], opts)] = (
                p.get("memoryllm_trained_baseline_letter"),
                p.get("icl_ft_baseline_letter"),
            )

    paper = json.load(open(PAPER_SUMMARY))
    summaries = {}

    for cond in conditions:
        results_path = RES / cond / "full.json"
        records = json.load(open(results_path))
        n_enriched = 0
        for r in records:
            h = probe_hash(r["user_query"], r["options"])
            if h in by_hash:
                r["memoryllm_trained_baseline_letter"], r["icl_ft_baseline_letter"] = by_hash[h]
                n_enriched += 1
        print(f"[{cond}] {len(records)} records, enriched {n_enriched} with trained-baseline letters")

        for mode in ["selected", "any_trained"]:
            s = summarize(records, mode)
            out = RES / cond / f"summary_{mode}.json"
            json.dump(s, open(out, "w"), indent=2)
            print(f"[{cond}] wrote {out.relative_to(REPO)}  (n {s['n_probes_pre_filter']} -> {s['n_probes_total']})")
        summaries[cond] = summarize(records, "any_trained")
        json.dump(records, open(RES / cond / "results.json", "w"))

    print()
    print("=" * 110)
    print("Comparison vs paper (collision_mode=any_trained), diagonal metrics")
    print("=" * 110)
    header = f"{'metric':<50} {'paper':>12}"
    for cond in conditions:
        header += f" {cond:>16}"
    print(header)
    for m in METRICS:
        row = f"{m:<50} {paper[m]['full']:>12.3f}"
        for cond in conditions:
            v = summaries[cond][m]["full"]
            row += f" {v:>16.3f}" if v is not None else f" {'-':>16}"
        print(row + "   (full)")
        row = f"{'':<50} {paper[m]['most_recent']:>12.3f}"
        for cond in conditions:
            v = summaries[cond][m]["most_recent"]
            row += f" {v:>16.3f}" if v is not None else f" {'-':>16}"
        print(row + "   (most_recent)")


if __name__ == "__main__":
    main()
