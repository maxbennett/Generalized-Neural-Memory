"""Build summary dirs from per-probe eval JSONs.

For each per-probe results JSON in `results/v12_final/`, compute three metrics:
  - accuracy           — rate `is_target_pick` on LEARN probes (target = preference_aligned)
  - selectivity        — rate `is_baseline_pick` on IGNORE probes (target = baseline)
  - ignoring_correctly — rate NOT `is_preference_pick` on IGNORE probes
                         (always >= selectivity under strict-filter)

For each metric we compute four views (matching src/utils/metrics.py conventions):
  - <metric>_full         scalar mean across all eligible diag probes
  - <metric>_most_recent  scalar mean on the just-memorized doc
                          (seq_idx_of_doc == seq_idx_of_io_pairs == seq_len-1)
  - <metric>_over_time[s] = matrix[s][s]  diagonal — accuracy at memory state s
  - <metric>_retention[t] = mean over (y,x) where y - x == t (lower-triangular by retention offset)
  - <metric>_matrix[y][x] full lower-triangular 10x10

Saves one summary.json per config plus an aggregate v12_final_summary.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "evals/personamem/results/personamem_final"
SAVED_EVALS = REPO / "evals/personamem/results/short_horizon"

SEQ_LEN = 10

# Each metric: (eligible-probe predicate, hit predicate)
# predicates are functions of a record dict.
#
# Each record carries:
#   - is_collision: (correct == selected_baseline) — set at eval time by --baseline-source.
#   - memoryllm_baseline_letter, llama3_baseline_letter: BOTH baselines stored.
#   - preference_aligned_letter: the correct (preference-aligned) letter.
#
# `collision_mode`:
#   - "selected" (default): exclude probes where correct == selected baseline (the legacy
#     definition). Asymmetric — different definitions for memoryllm vs llama3 evals.
#   - "any": exclude probes where correct == EITHER memoryllm_baseline OR llama3_baseline.
#     Filters down to probes that are non-trivial for BOTH models — fairer head-to-head
#     because both models lose the same set of "freebie" probes.


def _is_collision_any(r: dict) -> bool:
    """A probe is 'any-collision' iff the preference letter equals EITHER baseline letter."""
    pa = r.get("preference_aligned_letter") or r.get("correct_letter")
    return (pa == r.get("memoryllm_baseline_letter")) or (pa == r.get("llama3_baseline_letter"))


def build_metrics(collision_mode: str = "selected") -> dict:
    """Returns the METRICS dict gated on the chosen collision_mode."""
    if collision_mode == "selected":
        is_coll = lambda r: bool(r.get("is_collision"))
    elif collision_mode == "any":
        is_coll = _is_collision_any
    else:
        raise ValueError(f"unknown collision_mode={collision_mode!r}; expected 'selected' or 'any'")
    return {
        "accuracy_overall": (
            lambda r: r["learn_or_ignore"] == "learn",
            lambda r: bool(r.get("is_target_pick")),
        ),
        "accuracy_excluding_collisions": (
            lambda r, ic=is_coll: r["learn_or_ignore"] == "learn" and not ic(r),
            lambda r: bool(r.get("is_target_pick")),
        ),
        "selectivity_overall": (
            lambda r: r["learn_or_ignore"] == "ignore",
            lambda r: bool(r.get("is_baseline_pick")),
        ),
        "selectivity_excluding_collisions": (
            lambda r, ic=is_coll: r["learn_or_ignore"] == "ignore" and not ic(r),
            lambda r: bool(r.get("is_baseline_pick")),
        ),
        # "successfully_avoided_leaking" — looser metric: model just needs to NOT pick the
        # preference letter on ignore probes. NEW: also reported _overall (incl. collisions)
        # so it lines up with the training-time WandB-logged "selectivity" definition.
        "successfully_avoided_leaking_overall": (
            lambda r: r["learn_or_ignore"] == "ignore",
            lambda r: not bool(r.get("is_preference_pick")),
        ),
        "successfully_avoided_leaking_excluding_collisions": (
            lambda r, ic=is_coll: r["learn_or_ignore"] == "ignore" and not ic(r),
            lambda r: not bool(r.get("is_preference_pick")),
        ),
    }


# Default at module import time (collision_mode="selected") — preserves prior behavior
# of any caller doing `from build_summary import METRICS`.
METRICS = build_metrics("selected")


def _matrix_for(records, eligible, hit, seq_len=SEQ_LEN):
    """Return (matrix[y][x], counts[y][x]) matching the summary layout:
    y = seq_idx_of_io_pairs (probe origin), x = seq_idx_of_doc (memory state).
    Upper-triangular: cell filled when x >= y."""
    sums = np.zeros((seq_len, seq_len), dtype=np.float64)
    counts = np.zeros((seq_len, seq_len), dtype=np.int64)
    for r in records:
        if not eligible(r):
            continue
        y = r.get("seq_idx_of_io_pairs")
        x = r.get("seq_idx_of_doc")
        if not isinstance(y, int) or not isinstance(x, int):
            continue
        if x < y or y >= seq_len or x < 0 or x >= seq_len:
            continue
        counts[y, x] += 1
        sums[y, x] += int(hit(r))
    matrix = np.zeros_like(sums)
    mask = counts > 0
    matrix[mask] = sums[mask] / counts[mask]
    return matrix, counts


def _retention_curve(matrix, counts, seq_len=SEQ_LEN):
    """For each retention offset t in 0..seq_len-1, average across cells (y, y+t)
    weighted by per-cell counts. t = seq_idx_of_doc - seq_idx_of_io_pairs ('time_ago')."""
    out_rates = np.zeros(seq_len, dtype=np.float64)
    out_counts = np.zeros(seq_len, dtype=np.int64)
    for t in range(seq_len):
        cells = [(y, y + t) for y in range(seq_len - t)]
        total_hits = 0.0
        total_n = 0
        for y, x in cells:
            n = int(counts[y, x])
            if n == 0:
                continue
            total_hits += matrix[y, x] * n
            total_n += n
        out_rates[t] = total_hits / total_n if total_n else 0.0
        out_counts[t] = total_n
    return out_rates.tolist(), out_counts.tolist()


def summarize_one(records: list[dict], seq_len: int = SEQ_LEN,
                   collision_mode: str = "selected") -> dict:
    # collision_mode="any": GLOBAL pre-filter that drops any probe where the preference letter
    # equals EITHER baseline letter. After this filter, no remaining probe is a "collision" by
    # any definition, so `_overall` and `_excluding_collisions` will be identical (both compute
    # on the same filtered set). Use this to compare two models on apples-to-apples probes.
    n_total_pre = len(records)
    if collision_mode == "any":
        records = [r for r in records if not _is_collision_any(r)]
    elif collision_mode != "selected":
        raise ValueError(f"unknown collision_mode={collision_mode!r}; expected 'selected' or 'any'")
    # Per-metric filtering still uses `is_collision` (selected) for `_excluding_collisions`.
    # Under collision_mode='any', `is_collision` is False for every remaining record (since
    # selected collisions ⊆ any collisions), so the per-metric filter is a no-op there.
    metrics = build_metrics("selected")
    summary = {"n_probes_total": len(records), "n_probes_pre_filter": n_total_pre,
               "seq_len": seq_len, "collision_mode": collision_mode}
    for name, (eligible, hit) in metrics.items():
        matrix, counts = _matrix_for(records, eligible, hit, seq_len)
        diag = np.array([matrix[i, i] if counts[i, i] > 0 else 0.0 for i in range(seq_len)])
        diag_counts = np.array([counts[i, i] for i in range(seq_len)])

        # full = mean over all eligible diag probes (matches existing fact_accuracy convention)
        diag_total_hits = sum(matrix[i, i] * counts[i, i] for i in range(seq_len) if counts[i, i] > 0)
        diag_total_n = int(diag_counts.sum())
        full = (diag_total_hits / diag_total_n) if diag_total_n else None

        # most_recent = matrix[seq_len-1][seq_len-1]
        i = seq_len - 1
        most_recent = float(matrix[i, i]) if counts[i, i] > 0 else None
        most_recent_n = int(counts[i, i])

        retention, retention_counts = _retention_curve(matrix, counts, seq_len)

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


# (name, per-probe results file under --results-root)
CONFIGS = [
    ("gnm",    "gnm.json"),
    ("icl_ft", "icl_ft.json"),
]


def fmt(x, d=3):
    if x is None:
        return "  -  "
    return f"{x:.{d}f}"


def main():
    global RESULTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, default=RESULTS)
    ap.add_argument("--saved-evals-root", type=Path, default=SAVED_EVALS)
    args = ap.parse_args()
    RESULTS = args.results_root

    args.saved_evals_root.mkdir(parents=True, exist_ok=True)
    aggregate = {}
    for name, results_file in CONFIGS:
        rp = RESULTS / results_file
        if not rp.exists():
            print(f"[skip] {name}: {rp.name} not found")
            continue
        records = json.load(open(rp))
        if records and "seq_idx_of_io_pairs" not in records[0]:
            print(f"[warn] {name}: legacy schema (no seq_idx_of_io_pairs) — re-run eval needed")
            continue
        s = summarize_one(records)
        aggregate[name] = s

        cfg_dir = args.saved_evals_root / name
        cfg_dir.mkdir(parents=True, exist_ok=True)

        # 1) summary.json — full per-config metrics (accuracy/selectivity/ignoring × full/most_recent/over_time/retention/matrix)
        with open(cfg_dir / "summary.json", "w") as f:
            json.dump(s, f, indent=2)

        # 2) run_config.json — extract eval args from the script's auto-generated <output>.summary.json
        rc_src = rp.with_suffix(".summary.json")
        run_config = {}
        if rc_src.exists():
            try:
                run_config = json.load(open(rc_src)).get("args", {})
            except Exception as e:
                print(f"[warn] {name}: failed to read {rc_src}: {e}")
        with open(cfg_dir / "run_config.json", "w") as f:
            json.dump(run_config, f, indent=2, default=str)

        # 3) per-record dump (full data) — copy the per-probe JSON in
        import shutil
        dst = cfg_dir / "temp_results_val_ood_epoch_0.jsonl"
        shutil.copy2(rp, dst)

        print(f"[ok] {name}: n={s['n_probes_total']} -> {cfg_dir}/  "
              f"(summary.json, run_config.json, temp_results_val_ood_epoch_0.jsonl)")

    # ---- Headline table (most_recent) ----
    print("=" * 130)
    print("MOST RECENT  (probe from just-memorized doc; seq_idx_of_doc == seq_idx_of_io_pairs == seq_len-1)")
    print("=" * 130)
    print(f"{'config':<38} {'acc_all':>9} {'acc_xcoll':>10} {'sel_all':>9} {'sel_xcoll':>10}")
    print("-" * 130)
    for name, _ in CONFIGS:
        s = aggregate.get(name)
        if not s:
            continue
        ao = s["accuracy_overall"]; ax = s["accuracy_excluding_collisions"]
        so = s["selectivity_overall"]; sx = s["selectivity_excluding_collisions"]
        print(f"{name:<38} "
              f"{fmt(ao['most_recent']):>9} {fmt(ax['most_recent']):>10} "
              f"{fmt(so['most_recent']):>9} {fmt(sx['most_recent']):>10}")

    # ---- Diag-mean (full) table ----
    print()
    print("=" * 130)
    print("FULL  (mean across all diagonal probes, all memory states)")
    print("=" * 130)
    print(f"{'config':<38} {'acc_all':>9} {'acc_xcoll':>10} {'sel_all':>9} {'sel_xcoll':>10}")
    print("-" * 130)
    for name, _ in CONFIGS:
        s = aggregate.get(name)
        if not s:
            continue
        ao = s["accuracy_overall"]; ax = s["accuracy_excluding_collisions"]
        so = s["selectivity_overall"]; sx = s["selectivity_excluding_collisions"]
        print(f"{name:<38} "
              f"{fmt(ao['full']):>9} {fmt(ax['full']):>10} "
              f"{fmt(so['full']):>9} {fmt(sx['full']):>10}")

    # ---- Retention curves ----
    for metric in list(METRICS.keys()):
        print()
        print("=" * 100)
        print(f"RETENTION — {metric}  (offset 0 = just memorized, 9 = oldest)")
        print("=" * 100)
        header = f"{'config':<38}" + "".join(f"{i:>6}" for i in range(SEQ_LEN))
        print(header)
        print("-" * 100)
        for name, _ in CONFIGS:
            s = aggregate.get(name)
            if not s:
                continue
            r = s[metric]["retention"]
            row = f"{name:<38}" + "".join(f"{fmt(v, 2):>6}" for v in r)
            print(row)


if __name__ == "__main__":
    main()
