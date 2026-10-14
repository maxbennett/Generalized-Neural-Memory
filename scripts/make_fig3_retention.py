"""Camera-ready Figure 3 replacement: retention vs lag (1..400) in the paper's
Fig-3 styling — Facts (Score) / Format (Accuracy) / Refusal (F1) panels with CI
bands, five systems incl. the LLM-filter write-time baselines (MemoryAgent-*).

  python scripts/make_fig3_retention.py
Outputs: plots/fig3_counterfact_retention.{png,pdf}, plots/fig3_counterfact_retention_values.md
"""

import glob
import json
import math
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import hmean

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEARCH_DIRS = [
    os.path.join(REPO, "evals/counterfact/results/long_horizon"),
    os.path.join(REPO, "evals/counterfact/paper_results/long_horizon"),
]
LAGS = [1, 3, 10, 25, 50, 100, 200, 400]
# equally-spaced categorical x-axis: each lag plotted at its ladder index
XPOS = {l: i for i, l in enumerate(LAGS)}

plt.style.use("seaborn-v0_8")
pal = sns.color_palette("deep", 10)

# label -> (acc patterns, refusal patterns, color)
# Figure labels for the write-time LLM-filter baselines.
SYSTEMS = [
    ("GNM", ["gnm_long_horizon_f?.json",
             "gnm_long_horizon_flag1000_f?.json"],
            ["gnm_long_horizon_refusal*.json"],
            pal[0]),
    ("ICL-FT", ["llama3_8b_instruct_windowed_f?.json",
                "llama3_8b_instruct_windowed_flag1000_f?.json"],
               ["llama3_8b_instruct_windowed_refusal*.json"],
               pal[8]),
    ("RAG-FT", ["llama3_8b_instruct_rag_f?.json",
                "llama3_8b_instruct_rag_flag1000_f?.json"],
               ["llama3_8b_instruct_rag_refusal*.json"],
               pal[3]),
    ("MemoryAgent-ICL", ["memoryagent_icl_f?.json"],
                     ["memoryagent_icl_refusal_f?.json",
                      "memoryagent_icl_refusal_s1001_f?.json"],
                     pal[4]),
    ("MemoryAgent-RAG", ["memoryagent_rag_f?.json"],
                     ["memoryagent_rag_refusal_f?.json",
                      "memoryagent_rag_refusal_s1001_f?.json"],
                     pal[2]),
]


def _files(patterns):
    out = []
    seen_names = set()
    for d in SEARCH_DIRS:
        for pat in patterns:
            for f in sorted(glob.glob(os.path.join(d, pat)) + glob.glob(os.path.join(d, "*", pat))):
                name = os.path.basename(f)
                if name not in seen_names:  # fresh results/ shadow the shipped paper_results copies
                    seen_names.add(name)
                    out.append(f)
    return out


def load_binary(patterns, key):
    per_lag = defaultdict(list)
    for f in _files(patterns):
        d = json.load(open(f))
        for pr in d.get(key, []):
            per_lag[pr["lag"]].append(pr["correct"])
    return per_lag



def load_refusals(patterns):
    per_lag = defaultdict(list)
    for f in _files(patterns):
        d = json.load(open(f))
        for pr in d.get("refusals", []):
            per_lag[pr["lag"]].append((pr["should_refuse"], pr["refused"]))
    return per_lag


def f1_from(tuples):
    tp = sum(1 for s, r in tuples if s and r)
    fp = sum(1 for s, r in tuples if not s and r)
    fn = sum(1 for s, r in tuples if s and not r)
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


def hm(vals):
    if any(v <= 0 for v in vals):
        return 0.0
    return float(hmean(vals))


def fact_score_curve(acc_pats, rng, n_boot=2000):
    comp = {k: load_binary(acc_pats, k) for k in ("probes", "sel", "spec")}
    xs, ys, lo, hi, ns = [], [], [], [], []
    for l in LAGS:
        sets = [comp[k].get(l, []) for k in ("probes", "sel", "spec")]
        if any(len(v) < 5 for v in sets):
            continue
        xs.append(l)
        ys.append(hm([float(np.mean(v)) for v in sets]))
        boots = []
        for _ in range(n_boot):
            ms = [float(np.mean([v[i] for i in rng.integers(0, len(v), len(v))]))
                  for v in sets]
            boots.append(hm(ms))
        lo.append(float(np.percentile(boots, 2.5)))
        hi.append(float(np.percentile(boots, 97.5)))
        ns.append(min(len(v) for v in sets))
    return xs, ys, lo, hi, ns



def binary_curve(acc_pats, key):
    """Raw per-lag mean of a binary probe key ('probes' = accuracy, 'sel', 'spec')
    with 95% binomial CI."""
    per_lag = load_binary(acc_pats, key)
    xs, ys, lo, hi, ns = [], [], [], [], []
    for l in LAGS:
        v = per_lag.get(l, [])
        if len(v) < 5:
            continue
        p = float(np.mean(v))
        h = 1.96 * math.sqrt(p * (1 - p) / len(v))
        xs.append(l); ys.append(p); lo.append(max(0.0, p - h)); hi.append(min(1.0, p + h))
        ns.append(len(v))
    return xs, ys, lo, hi, ns


def refusal_curve(ref_pats, rng, n_boot=2000):
    per_lag = load_refusals(ref_pats)
    xs, ys, lo, hi, ns = [], [], [], [], []
    for l in LAGS:
        t = per_lag.get(l, [])
        if len(t) < 5:
            continue
        xs.append(l)
        ys.append(f1_from(t))
        boots = [f1_from([t[i] for i in rng.integers(0, len(t), len(t))])
                 for _ in range(n_boot)]
        lo.append(float(np.percentile(boots, 2.5)))
        hi.append(float(np.percentile(boots, 97.5)))
        ns.append(len(t))
    return xs, ys, lo, hi, ns


PANELS_MAIN = [("Facts (Score)", "fact"), ("Facts (Accuracy)", "acc"), ("Refusal (F1)", "ref")]
PANELS_EXPANDED = [("Facts (Score)", "fact"), ("Facts (Accuracy)", "acc"),
                   ("Facts (Selectivity)", "sel"), ("Refusal (F1)", "ref")]


def render(all_curves, panels, out_stem, table):
    fig, axs = plt.subplots(1, len(panels), figsize=(4.33 * len(panels), 3.7))
    for label, color, curves in all_curves:
        for i, (title, key) in enumerate(panels):
            xs, ys, lo, hi, ns = curves[key]
            if not xs:
                continue
            px = [XPOS.get(x, x) for x in xs]
            axs[i].plot(px, np.array(ys) * 100, "-", label=label, color=color,
                        lw=4, zorder=3)
            axs[i].fill_between(px, np.array(lo) * 100, np.array(hi) * 100,
                                color=color, alpha=0.18, lw=0, zorder=2)
            for x, y, l_, h_, n in zip(xs, ys, lo, hi, ns):
                table[(title, label)][x] = f"{y*100:.1f} [{l_*100:.1f},{h_*100:.1f}] (n={n})"
    for i, (title, key) in enumerate(panels):
        axs[i].set_title(title, fontsize=18)
        axs[i].set_xticks([XPOS[l] for l in LAGS])
        axs[i].set_xticklabels([str(l) for l in LAGS])
        axs[i].tick_params(axis="x", labelsize=12)
        axs[i].tick_params(axis="y", labelsize=12)
        axs[i].set_xlabel("Documents Since Seen", fontsize=17)
        axs[i].set_ylim(0, None)
    axs[0].set_ylabel("% Performance", fontsize=17)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(labels), loc="lower center",
               bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=18,
               columnspacing=1.2, handlelength=1.6)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ("png", "pdf"):
        out = os.path.join(REPO, f"plots/{out_stem}.{ext}")
        fig.savefig(out, dpi=600, bbox_inches="tight")
        print("wrote", out)
    plt.close(fig)


def main():
    rng = np.random.default_rng(0)
    all_curves = []
    for label, acc_pats, ref_pats, color in SYSTEMS:
        all_curves.append((label, color, {
            "fact": fact_score_curve(acc_pats, rng),
            "acc": binary_curve(acc_pats, "probes"),
            "sel": binary_curve(acc_pats, "sel"),
            "ref": refusal_curve(ref_pats, rng),
        }))

    table = defaultdict(dict)  # (panel, label) -> {lag: "y [lo,hi] (n)"}
    render(all_curves, PANELS_MAIN, "fig3_counterfact_retention", table)
    render(all_curves, PANELS_EXPANDED, "fig3_counterfact_retention_expanded", table)

    # value table for the appendix / sanity reading
    lines = ["# Figure 3 (retention vs lag) — plotted values", ""]
    for title, _ in PANELS_EXPANDED:
        lines.append(f"## {title}")
        lines.append("| System | " + " | ".join(f"lag {l}" for l in LAGS) + " |")
        lines.append("|" + "---|" * (len(LAGS) + 1))
        for label, *_ in SYSTEMS:
            row = table.get((title, label), {})
            lines.append(f"| {label} | " + " | ".join(row.get(l, "–") for l in LAGS) + " |")
        lines.append("")
    out_md = os.path.join(REPO, "plots/fig3_counterfact_retention_values.md")
    open(out_md, "w").write("\n".join(lines) + "\n")
    print("wrote", out_md)


if __name__ == "__main__":
    main()
