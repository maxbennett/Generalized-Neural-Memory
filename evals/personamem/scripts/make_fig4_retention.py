"""PersonaMem retention-vs-lag chart (counterfact Fig-3 styling): Accuracy /
Preserved / P(leak|deviation) panels with 95% binomial CI bands.

  python scripts/make_fig4_retention.py
Outputs: <repo>/plots/fig4_personamem_retention.{png,pdf}
"""

import glob
import hashlib
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RES = os.path.join(HERE, "..", "results", "long_horizon")
if not os.path.isdir(RES):
    RES = os.path.join(HERE, "..", "paper_results", "long_horizon")
PAPER_RECS = os.path.join(
    REPO, "evals/personamem/paper_results/short_horizon/rag/results.json")
LAGS = [1, 3, 10, 25, 50, 100, 200, 300]   # 399 dropped: n<=3 learn probes

plt.style.use("seaborn-v0_8")
XPOS = {l: i for i, l in enumerate(LAGS)}  # equally-spaced categorical x (matches fig3)
pal = sns.color_palette("deep", 10)

SYSTEMS = [
    ("GNM",        ["gnm_long_horizon_ord.json", "gnm_long_horizon_p1??.json",
                    "gnm_long_horizon_tail_p1??.json",
                    "gnm_long_horizon_fullci_p1??.json", "gnm_long_horizon_tailci_p1??.json"],                              "trained", pal[0]),
    ("ICL-FT",     ["iclft_windowed_ord.json", "iclft_windowed_p1??.json",
                    "iclft_windowed_tail_p1??.json",
                    "iclft_windowed_fullci_p1??.json", "iclft_windowed_tailci_p1??.json"],                                    "iclft",   pal[8]),
    ("MemoryAgent-ICL", ["memoryagent_icl_ord.json", "memoryagent_icl_p1??.json",
                      "memoryagent_icl_tail_p1??.json"], "llama3", pal[4]),
    ("MemoryAgent-RAG", ["memoryagent_rag_ord.json", "memoryagent_rag_p1??.json",
                      "memoryagent_rag_tail_p1??.json"], "llama3", pal[2]),
]


def probe_hash(query, options):
    return hashlib.sha256((query + json.dumps(options, sort_keys=True)).encode()).hexdigest()


def load_baseline_map():
    recs = json.load(open(PAPER_RECS))
    return {probe_hash(r["user_query"], r["options"]):
            (r.get("memoryllm_trained_baseline_letter"), r.get("icl_ft_baseline_letter"))
            for r in recs}


def curves(patterns, own_key, base_map):
    per = defaultdict(lambda: {"acc": [0, 0], "pres": [0, 0], "leak": [0, 0], "lgd": [0, 0]})
    rng = np.random.default_rng(0)
    files = sorted(set(f for p in patterns for f in glob.glob(os.path.join(RES, p)) + glob.glob(os.path.join(RES, '*', p))))
    for f in files:
        if "smoke" in f:
            continue
        for r in json.load(open(f))["records"]:
            bases = base_map.get(r.get("probe_hash") or probe_hash(r["user_query"], r["options"]))
            if bases is None:
                continue
            trained_base, iclft_base = bases
            if own_key == "iclft":
                own = iclft_base
            elif own_key == "llama3":
                own = r.get("llama3_baseline_letter")
            else:
                own = trained_base
            pref = r["preference_aligned_letter"]
            if pref == trained_base or pref == iclft_base:   # paper any_trained filter
                continue
            L, pred = r["lag"], r["predicted_letter"]
            if r["learn_or_ignore"] == "learn":
                per[L]["acc"][0] += int(pred == pref); per[L]["acc"][1] += 1
            else:
                per[L]["pres"][0] += int(pred == own); per[L]["pres"][1] += 1
                per[L]["leak"][0] += int(pred == pref); per[L]["leak"][1] += 1
                if pred != own:
                    per[L]["lgd"][0] += int(pred == pref); per[L]["lgd"][1] += 1
    out = {}
    # Score = hmean(accuracy, 100-leak) per lag, bootstrap CI over the two binomials
    xs, ys, lo, hi = [], [], [], []
    for L in LAGS:
        ca, na = per[L]["acc"]; cl, nl = per[L]["leak"]
        if na < 5 or nl < 5:
            continue
        a, l = ca / na, cl / nl
        def hm(x, y):
            return 2 * x * y / (x + y) if x + y > 0 else 0.0
        xs.append(L); ys.append(hm(a, 1 - l))
        boots = [hm(rng.binomial(na, a) / na, 1 - rng.binomial(nl, l) / nl) for _ in range(2000)]
        lo.append(float(np.percentile(boots, 2.5))); hi.append(float(np.percentile(boots, 97.5)))
    out["score"] = (xs, ys, lo, hi)
    for key in ("acc", "pres", "leak", "lgd"):
        xs, ys, lo, hi = [], [], [], []
        for L in LAGS:
            c, n = per[L][key]
            if n < 5:
                continue
            p = c / n
            h = 1.96 * math.sqrt(p * (1 - p) / n)
            xs.append(L); ys.append(p); lo.append(max(0, p - h)); hi.append(min(1, p + h))
        out[key] = (xs, ys, lo, hi)
    return out


def main():
    base_map = load_baseline_map()
    panels = [("Score", "score"), ("Accuracy", "acc"), ("Preserved", "pres"),
              ("P(leak | deviation)", "lgd")]
    INVERTED = {"lgd"}  # lower is better; invert so up = good everywhere
    fig, axs = plt.subplots(1, 4, figsize=(17.3, 4.5))
    for label, pats, own_key, color in SYSTEMS:
        cs = curves(pats, own_key, base_map)
        for i, (title, key) in enumerate(panels):
            xs, ys, lo, hi = cs[key]
            if not xs:
                continue
            px = [XPOS[x] for x in xs]
            axs[i].plot(px, np.array(ys) * 100, "-", label=label, color=color, lw=5, zorder=3)
            axs[i].fill_between(px, np.array(lo) * 100, np.array(hi) * 100,
                                color=color, alpha=0.18, lw=0, zorder=2)
    for i, (title, key) in enumerate(panels):
        axs[i].set_title(title, fontsize=23)
        axs[i].set_xticks([XPOS[l] for l in LAGS])
        axs[i].set_xticklabels([str(l) for l in LAGS])
        axs[i].minorticks_off()
        axs[i].tick_params(axis="x", labelsize=14)
        axs[i].tick_params(axis="y", labelsize=14)
        axs[i].set_xlabel("Documents Since Seen", fontsize=21)
    # data-driven y-limits: zoom panels whose curves live far from 0
    for i in range(len(panels)):
        los, his = [], []
        for line in axs[i].get_lines():
            ys = line.get_ydata()
            if len(ys):
                los.append(min(ys)); his.append(max(ys))
        for coll in axs[i].collections:
            arr = coll.get_paths()[0].vertices[:, 1] if coll.get_paths() else []
            if len(arr):
                los.append(float(np.min(arr))); his.append(float(np.max(arr)))
        if los:
            lo_v, hi_v = min(los), max(his)
            pad = max(3.0, 0.06 * (hi_v - lo_v))
            if panels[i][1] in INVERTED:
                axs[i].set_ylim(min(102, hi_v + pad), max(0, lo_v - pad))  # inverted: up = better
            else:
                axs[i].set_ylim(max(0, lo_v - pad), min(102, hi_v + pad))
    # fixed axis bounds: Preserved floor at 60; P(l|d) inverted axis down to 80
    axs[2].set_ylim(50, axs[2].get_ylim()[1])
    axs[3].set_ylim(80, axs[3].get_ylim()[1])
    axs[3].axhline(100 / 3, color="#777777", lw=1.2, ls=":", zorder=1)
    axs[3].annotate("chance (33%)", xy=(XPOS[LAGS[-1]], 100 / 3), xytext=(0, 8),
                    textcoords="offset points", ha="right", fontsize=14, color="#555555")
    for i, (title, key) in enumerate(panels):
        if key in INVERTED:
            axs[i].annotate("axis inverted — higher is better", xy=(0.98, 0.03),
                            xycoords="axes fraction", ha="right", fontsize=12, color="#777777")
    axs[0].set_ylabel("% Performance", fontsize=21)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(labels), loc="lower center",
               bbox_to_anchor=(0.5, -0.06), frameon=False, fontsize=21,
               columnspacing=1.4, handlelength=1.6)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ("png", "pdf"):
        out = os.path.join(REPO, f"plots/fig4_personamem_retention.{ext}")
        fig.savefig(out, dpi=600, bbox_inches="tight")
        print("wrote", out)


if __name__ == "__main__":
    main()
