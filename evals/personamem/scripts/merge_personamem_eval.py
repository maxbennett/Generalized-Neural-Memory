"""Merge a sharded PersonaMem eval (produced by run_personamem_eval.sh) into a
summary dir: results.json + summary.json (standard collision filter)
+ summary_any_trained.json (the paper's any_trained filter: drop probes whose
preference letter equals EITHER trained model's own no-memory pick). Records
are enriched at merge time by hashing (user_query, options) against the source
dataset.

  python merge_personamem_eval.py --shards-dir results/personamem_gnm/_shards \
      --out-dir ../../evals/personamem/paper_results/short_horizon/gnm \
      --model-key memoryllm_chat --ckpt models/gnm_personamem/model.pth
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "evals/personamem/scripts"
sys.path.insert(0, str(SCRIPTS))

from build_summary import _matrix_for, _retention_curve, build_metrics, _is_collision_any  # noqa: E402

SOURCE_EVAL = REPO / "data/personamem/test.json"
SEQ_LEN = 10


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
        most_recent = float(matrix[i, i]) if counts[i, i] > 0 else None
        most_recent_n = int(counts[i, i])
        retention, retention_counts = _retention_curve(matrix, counts, SEQ_LEN)
        summary[name] = {
            "full": full,
            "full_n": diag_total_n,
            "most_recent": most_recent,
            "most_recent_n": most_recent_n,
            "over_time": diag.tolist(),
            "over_time_counts": diag_counts.tolist(),
            "retention": retention,
            "retention_counts": retention_counts,
            "matrix": matrix.tolist(),
            "matrix_counts": counts.tolist(),
        }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model-key", default="memoryllm_chat")
    ap.add_argument("--ckpt", default="models/gnm_personamem/model.pth")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    shards_dir = args.shards_dir
    out_dir = args.out_dir

    # 1. Load + concat shards.
    records = []
    for shard_path in sorted(shards_dir.glob("shard_*.json")):
        if shard_path.name.endswith(".summary.json"):
            continue
        rs = json.load(open(shard_path))
        records.extend(rs)
        print(f"[merge] {shard_path.name}: {len(rs)} records")
    print(f"[merge] total: {len(records)}")

    # 2. Enrich with trained-baseline letters from source eval.
    source = json.load(open(SOURCE_EVAL))
    by_hash = {}
    for d in source:
        for p in d.get("input_output_pairs", []):
            opts = p.get("options")
            if not opts:
                continue
            h = probe_hash(p["user_query"], opts)
            by_hash[h] = (p.get("memoryllm_trained_baseline_letter"),
                          p.get("icl_ft_baseline_letter"))
    n_enriched = 0
    for r in records:
        h = probe_hash(r["user_query"], r["options"])
        if h in by_hash:
            mtbl, icl = by_hash[h]
            r["memoryllm_trained_baseline_letter"] = mtbl
            r["icl_ft_baseline_letter"] = icl
            n_enriched += 1
    print(f"[merge] enriched {n_enriched}/{len(records)} records with trained-baseline letters")

    # 3. Write the summary dir: full records + both summaries.
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(records, f)
    sel_summary = summarize(records, "selected")
    with open(out_dir / "summary.json", "w") as f:
        json.dump(sel_summary, f, indent=2)
    filt_summary = summarize(records, "any_trained")
    with open(out_dir / "summary_any_trained.json", "w") as f:
        json.dump(filt_summary, f, indent=2)
    run_config = {
        "model": args.model_key,
        "model_state_dict_path": str(REPO / args.ckpt),
        "eval_path": str(SOURCE_EVAL),
        "seq_len": SEQ_LEN,
        "n_rollouts": 40,
        "rollout_offset": 0,
        "baseline_source": "memoryllm",
        "n_probes": len(records),
        "shards": len(list(shards_dir.glob("shard_*.json"))),
        "collision_filters": ["selected (summary.json)", "any_trained (summary_any_trained.json)"],
        "note": args.note,
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)
    print(f"[merge] wrote {out_dir}/")

    # 4. Print headline.
    print()
    print("=" * 80)
    print(f"{out_dir.name} — headline")
    print("=" * 80)
    for tag, s in [("standard (selected)", sel_summary), ("any_trained filter", filt_summary)]:
        print(f"\n  [{tag}]  n_pre={s['n_probes_pre_filter']}  n_post={s['n_probes_total']}")
        for m in ["accuracy_overall", "accuracy_excluding_collisions",
                  "selectivity_overall", "selectivity_excluding_collisions",
                  "successfully_avoided_leaking_overall", "successfully_avoided_leaking_excluding_collisions"]:
            row = s[m]
            mr = row["most_recent"]
            full = row["full"]
            mr_str = f"{mr:.3f}" if mr is not None else "  -  "
            full_str = f"{full:.3f}" if full is not None else "  -  "
            print(f"    {m:<48} most_recent={mr_str}  full={full_str}  n_diag={row['full_n']}")


if __name__ == "__main__":
    main()
