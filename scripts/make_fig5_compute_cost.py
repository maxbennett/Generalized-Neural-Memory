"""Final measured cost figure: "Cost of Memorization" (FLOPs per write, bars) +
"Cost of Inference" (FLOPs per generated token vs docs in memory, 1-50).
From scripts/profile_costs.py JSONs (COST_SIZES=1,5,10,25,50), counterfact and
personamem averaged. All numbers torch-profiler measured.

  python scripts/make_fig5_compute_cost.py
Outputs plots/fig5_compute_cost.{png,pdf}
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plt.style.use("seaborn-v0_8"); pal = sns.color_palette("deep", 10)
def _profile_path(e):
    fresh = os.path.join(REPO, f"evals/counterfact/results/cost_profile_{e}.json")
    shipped = {"counterfact": os.path.join(REPO, "evals/counterfact/paper_results/cost_profile_counterfact.json"),
               "personamem": os.path.join(REPO, "evals/personamem/paper_results/cost_profile_personamem.json")}[e]
    return fresh if os.path.exists(fresh) else shipped

d = {e: json.load(open(_profile_path(e)))["systems"]
     for e in ("counterfact", "personamem")}
SYS = ["ICL", "RAG", "GNM", "MemoryAgent"]  # JSON keys in cost_profile_*.json
DISP = {s: ("MemoryAgent" if s == "MemoryAgent" else s) for s in SYS}  # display labels
COL = {"ICL": pal[8], "RAG": pal[5], "GNM": pal[0], "MemoryAgent": pal[4]}
SIZES = [1, 5, 10, 25, 50]
wf = {s: (d["counterfact"][s]["write_flops"] + d["personamem"][s]["write_flops"]) / 2 for s in SYS}
inf_f = {s: [(d["counterfact"][s]["infer"][str(z)]["flops_per_token"] +
              d["personamem"][s]["infer"][str(z)]["flops_per_token"]) / 2 for z in SIZES] for s in SYS}

fig, axs = plt.subplots(1, 2, figsize=(12.8, 3.7))
ax = axs[0]
for i, s in enumerate(SYS):
    v = wf[s] / 1e13
    ax.bar(i, v, color=COL[s], width=0.62, zorder=3)
    lab = f"{wf[s]:.1e}".replace("e+12", "$\\times 10^{12}$").replace("e+13", "$\\times 10^{13}$") if wf[s] > 0 else "≈0"
    ax.annotate(lab, (i, v), xytext=(0, 5), textcoords="offset points", ha="center",
                fontsize=14, fontweight="bold")
ax.set_xticks(range(4)); ax.set_xticklabels([DISP[s] for s in SYS], fontsize=16)
ax.set_ylabel(r"FLOPs / doc write ($\times 10^{13}$)", fontsize=17)
ax.set_title("Cost of Memorization", fontsize=19)
ax.tick_params(axis="y", labelsize=15)
ax.set_ylim(0, 2.9)
ax = axs[1]
for s in SYS:
    ax.plot(SIZES, np.array(inf_f[s]) / 1e12, "-", color=COL[s], lw=4, label=DISP[s])
ax.set_xticks(SIZES); ax.set_xticklabels([str(z) for z in SIZES], fontsize=15)
ax.tick_params(axis="y", labelsize=15)
ax.set_xlabel("Documents in memory", fontsize=17)
ax.set_ylabel(r"FLOPs / token ($\times 10^{12}$)", fontsize=17)
ax.set_title("Cost of Inference", fontsize=19)
ax.legend(fontsize=16, loc="upper left")
fig.tight_layout()
fig.subplots_adjust(wspace=0.28)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(REPO, f"plots/fig5_compute_cost.{ext}"), dpi=600, bbox_inches="tight")
    print("wrote", f"plots/fig5_compute_cost.{ext}")
