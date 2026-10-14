"""Ablation bar chart: GNM vs GNM (ablation) vs MemLLM, two panels.
Left: CounterFACT Fact Score (seq-10 diagonal H-mean of acc/sel/spec).
Right: PersonaMem Score = H-mean(accuracy, 100-leak), paper any_trained filter.
Bootstrap/binomial 95% CIs. Outputs plots/appendix_ablation_bars.{png,pdf}.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import hmean

plt.style.use("seaborn-v0_8")
pal = sns.color_palette("deep", 10)
rng = np.random.default_rng(0)

# CounterFACT diagonal components (evals/counterfact/paper_results/short_horizon/<run>/summary.json)
CF = {"GNM": (.958, .771, .810), "GNM (ablation)": (.753, .249, .726), "MemLLM": (.817, .190, .756)}
CF_N = (1572, 1105, 1525)  # paper probe counts (acc, sel, spec)
# PersonaMem acc/leak (evals/personamem/paper_results/short_horizon, diagonal, any_trained filter)
PM = {"GNM": (.936, .061), "GNM (ablation)": (.336, .307), "MemLLM": (.217, .209)}
PM_N = (235, 657)  # (learn n, ignore n)


def cf_score(c, B=4000):
    a, s, p = c
    na, ns, nsp = CF_N
    v = hmean([a, s, p]) * 100
    boots = [hmean([rng.binomial(na, a) / na, rng.binomial(ns, s) / ns,
                    rng.binomial(nsp, p) / nsp]) * 100 for _ in range(B)]
    return v, np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def pm_score(c, B=4000):
    acc, leak = c
    nl, ni = PM_N
    v = hmean([acc, 1 - leak]) * 100
    boots = [hmean([rng.binomial(nl, acc) / nl, 1 - rng.binomial(ni, leak) / ni]) * 100
             for _ in range(B)]
    return v, np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def main():
    cf = {k: cf_score(v) for k, v in CF.items()}
    pm = {k: pm_score(v) for k, v in PM.items()}
    # GNM blue; ablation gray; MemLLM turquoise (pal[9], matching the memory-analysis chart)
    colors = [pal[0], pal[7], pal[9]]
    fig, axs = plt.subplots(1, 2, figsize=(9, 2.7))
    for ax, (title, data) in zip(axs, [("CounterFACT — Fact Score", cf), ("PersonaMem — Score", pm)]):
        names = list(data)
        for i, nm in enumerate(names):
            v, lo, hi = data[nm]
            ax.bar(i, v, color=colors[i], width=0.62, zorder=3,
                   yerr=[[v - lo], [hi - v]], capsize=3, error_kw=dict(lw=1.1, ecolor="#444444"))
            ax.annotate(f"{v:.1f}", (i, v), xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=9.5)
        ax.set_title(title, fontsize=12)
        ax.set_ylim(0, 104)
        ax.tick_params(axis="y", labelsize=9)
    axs[0].set_ylabel("% Performance", fontsize=11)
    fig.tight_layout()
    import os
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(repo, f"plots/appendix_ablation_bars.{ext}"), dpi=600, bbox_inches="tight")
        print("wrote", f"plots/appendix_ablation_bars.{ext}")


if __name__ == "__main__":
    main()
