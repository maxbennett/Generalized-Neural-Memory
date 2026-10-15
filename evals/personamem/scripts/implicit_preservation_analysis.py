"""AC4-style preserved/leak/P(leak|deviation) analysis for the implicit-instruction runs.

Diag ignore probes, paper trained_any
collision filter, buckets vs GNM's OWN no-context baseline
(memoryllm_trained_baseline_letter). The reference table (rag/results.json)
provides baseline letters and the filter set, keyed by
(rollout_idx, seq_idx_of_doc, seq_idx_of_io_pairs, pair_idx) — valid because the
implicit runs use identical probes, example order, and rollout chunking.

Rows: paper GNM (explicit+descriptor), paper ICL-FT (context), then the implicit
conditions (GNM `implicit`, ICL-FT `icl_ft_implicit`; other condition dirs can be
passed on argv). Paired McNemar vs the paper rows.
"""
import json
import sys
from math import sqrt, erf
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
D = REPO / "evals/personamem/paper_results/short_horizon"
RES = REPO / "evals/personamem/paper_results/implicit_instructions"
KEY = lambda r: (r["rollout_idx"], r["seq_idx_of_doc"], r["seq_idx_of_io_pairs"], r["pair_idx"])

ref = json.load(open(D / "rag/results.json"))
tb = {KEY(r): r for r in ref}
keep = lambda t: t["preference_aligned_letter"] not in (
    t["icl_ft_baseline_letter"], t["memoryllm_trained_baseline_letter"])


def norm_p(z):
    return 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))


def score(records_by_key, ownbase_field):
    """Return (n, acc_stats, per-key ignore buckets) on paper-filtered diag probes."""
    n_ig = pres = leak = n_ln = acc = 0
    buckets = {}
    for k, t in tb.items():
        r = records_by_key.get(k)
        if r is None or r["seq_idx_of_doc"] != r["seq_idx_of_io_pairs"] or not keep(t):
            continue
        p = r["predicted_letter"]
        if r["learn_or_ignore"] == "learn":
            n_ln += 1
            acc += p == t["preference_aligned_letter"]
            continue
        n_ig += 1
        b = p == t[ownbase_field]
        l = p == t["preference_aligned_letter"]
        pres += b
        leak += l
        buckets[k] = (b, l)
    return n_ig, n_ln, acc, pres, leak, buckets


conditions = {
    "GNM paper (explicit+desc)": ({KEY(r): r for r in json.load(open(D / "gnm/results.json"))},
                                  "memoryllm_trained_baseline_letter"),
    "ICL-FT paper (explicit+desc)": ({KEY(r): r for r in json.load(open(D / "icl_ft/results.json"))},
                                     "icl_ft_baseline_letter"),
}
for cond in sys.argv[1:] or ["implicit", "icl_ft_implicit"]:
    recs = json.load(open(RES / cond / "results.json"))
    if cond.startswith("icl_ft"):
        conditions[f"ICL-FT {cond[7:]}"] = ({KEY(r): r for r in recs}, "icl_ft_baseline_letter")
    else:
        conditions[f"GNM {cond}"] = ({KEY(r): r for r in recs}, "memoryllm_trained_baseline_letter")

print("=" * 100)
print("IGNORE probes, diagonal, paper trained_any collision filter (AC4 methodology)")
print("=" * 100)
print(f"{'condition':<30} {'acc%':>6} {'n_ig':>5} {'preserved%':>11} {'leak%':>6} {'drift%':>7} {'P(leak|dev)%':>13}")
all_buckets = {}
for name, (by, field) in conditions.items():
    n_ig, n_ln, acc, pres, leak, buckets = score(by, field)
    all_buckets[name] = buckets
    dev = n_ig - pres
    print(f"{name:<30} {100*acc/n_ln:>6.1f} {n_ig:>5} {100*pres/n_ig:>11.1f} {100*leak/n_ig:>6.1f} "
          f"{100*(n_ig-pres-leak)/n_ig:>7.1f} {100*leak/dev if dev else 0:>13.1f}")

print()
for name, buckets in all_buckets.items():
    if "paper" in name:
        continue
    ref_name = "ICL-FT paper (explicit+desc)" if name.startswith("ICL-FT") else "GNM paper (explicit+desc)"
    g0 = all_buckets[ref_name]
    common = [k for k in g0 if k in buckets]
    for stat, idx in [("preserved", 0), ("leak", 1)]:
        b10 = sum(1 for k in common if g0[k][idx] and not buckets[k][idx])
        b01 = sum(1 for k in common if not g0[k][idx] and buckets[k][idx])
        z = (b10 - b01) / sqrt(b10 + b01) if b10 + b01 else 0.0
        print(f"McNemar {name} vs {ref_name.split(' (')[0]} on {stat:<9}: b={b10:<3} c={b01:<3} p={norm_p(z):.4f}")
    # leak-given-deviation two-proportion test vs paper GNM
    gd = [g0[k][1] for k in common if not g0[k][0]]
    cd = [buckets[k][1] for k in common if not buckets[k][0]]
    x1, n1, x2, n2 = sum(gd), len(gd), sum(cd), len(cd)
    if n1 and n2:
        p_ = (x1 + x2) / (n1 + n2)
        z = (x1 / n1 - x2 / n2) / sqrt(p_ * (1 - p_) * (1 / n1 + 1 / n2)) if 0 < p_ < 1 else 0.0
        print(f"P(leak|dev) {name}: {x2}/{n2}={100*x2/n2:.1f}%  vs paper {x1}/{n1}={100*x1/n1:.1f}%  p={norm_p(z):.3f}")
    print()
