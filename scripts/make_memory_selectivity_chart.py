"""Memory-selectivity (target alignment) figure for the paper — extracted from
notebooks/ablation.ipynb cell 4 (identical data/logic), camera-ready layout:
group headers (CounterFACT | PersonaMem) spanning panel pairs, a vertical
divider between the groups, and one shared legend beneath all panels.
Metric: cos_diff_last (last-token cosine-of-difference, M4).

  python scripts/make_memory_selectivity_chart.py
Outputs: plots/fig6_memory_selectivity.{png,pdf}
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

plt.style.use('seaborn-v0_8')
pal = sns.color_palette("deep", 10)
model_colors = {"Ablation": pal[7], "MemLLM": pal[9], "GNM": pal[0]}

FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND = 18, 17, 15, 15

with open(os.path.join(REPO, "evals/counterfact/paper_results/memory_analysis/memory_selectivity_counterfact.json")) as f:
    cf = json.load(f)["metrics"]["cos_diff_last"]
with open(os.path.join(REPO, "evals/personamem/paper_results/memory_analysis/memory_selectivity_personalization.json")) as f:
    pf = json.load(f)["metrics"]["cos_diff_last"]

cf_gnm_mean    = np.array(cf["gnm_mean_per_layer"]);                 cf_gnm_se    = np.array(cf["gnm_se_per_layer"])
cf_abl_mean    = np.array(cf["ablation_mean_per_layer"]);            cf_abl_se    = np.array(cf["ablation_se_per_layer"])
cf_memllm_mean = np.array(cf["memoryllm_untrained_mean_per_layer"]); cf_memllm_se = np.array(cf["memoryllm_untrained_se_per_layer"])
cf_mid = slice(15, -1)
cf_gnm_mid, cf_abl_mid, cf_memllm_mid = (float(a[cf_mid].mean()) for a in (cf_gnm_mean, cf_abl_mean, cf_memllm_mean))

pf_gnm_mean    = np.array(pf["memoryllm_trained_mean_per_layer"]);   pf_gnm_se    = np.array(pf["memoryllm_trained_se_per_layer"])
pf_abl_mean    = np.array(pf["memoryllm_ablation_mean_per_layer"]);  pf_abl_se    = np.array(pf["memoryllm_ablation_se_per_layer"])
pf_memllm_mean = np.array(pf["memoryllm_untrained_mean_per_layer"]); pf_memllm_se = np.array(pf["memoryllm_untrained_se_per_layer"])
pf_mid = slice(7, 22)
pf_gnm_mid, pf_abl_mid, pf_memllm_mid = (float(a[pf_mid].mean()) for a in (pf_gnm_mean, pf_abl_mean, pf_memllm_mean))

def ci(se):         return 1.96 * se
def mid_ci(se, sl): return 1.96 * se[sl].mean()

layers_cf = np.arange(len(cf_gnm_mean))
layers_pf = np.arange(len(pf_gnm_mean))

# 5 columns: thin empty spacer between the two experiment groups
fig, axes = plt.subplots(1, 5, figsize=(13.4, 3.55),
                         gridspec_kw={'width_ratios': [3.72, 1.25, 0.2, 3.72, 1.25]})
ax_cf_line, ax_cf_bar, ax_gap, ax_pf_line, ax_pf_bar = axes
ax_gap.axis('off')


def line_panel(ax, layers, memllm, memllm_se, abl, abl_se, gnm, gnm_se, title):
    memllm_line, = ax.plot(layers, memllm, '^--', color=model_colors["MemLLM"],
                           label='MemLLM', markersize=4, linewidth=1.3, alpha=0.9)
    ax.fill_between(layers, memllm - ci(memllm_se), memllm + ci(memllm_se),
                    color=model_colors["MemLLM"], alpha=0.15)
    abl_line, = ax.plot(layers, abl, 's-', color=model_colors["Ablation"],
                        label='GNM (Ablation)', markersize=4, linewidth=1.5)
    ax.fill_between(layers, abl - ci(abl_se), abl + ci(abl_se),
                    color=model_colors["Ablation"], alpha=0.2)
    gnm_line, = ax.plot(layers, gnm, 'o-', color=model_colors["GNM"],
                        label='GNM', markersize=4, linewidth=1.8)
    ax.fill_between(layers, gnm - ci(gnm_se), gnm + ci(gnm_se),
                    color=model_colors["GNM"], alpha=0.2)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('Layer', fontsize=FS_LABEL)
    ax.set_ylabel('Target Alignment', fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_TITLE)
    ax.set_xlim(0, layers.max())
    ax.tick_params(axis='both', labelsize=FS_TICK)
    return gnm_line, abl_line, memllm_line


def bar_panel(ax, memllm_mid, abl_mid, gnm_mid, memllm_se, abl_se, gnm_se, sl, title):
    ax.bar(['MemLLM'],   [memllm_mid], color=model_colors["MemLLM"])
    ax.bar(['Ablation'], [abl_mid],    color=model_colors["Ablation"])
    ax.bar(['GNM'],      [gnm_mid],    color=model_colors["GNM"])
    ax.errorbar(['MemLLM', 'Ablation', 'GNM'], [memllm_mid, abl_mid, gnm_mid],
                yerr=[mid_ci(memllm_se, sl), mid_ci(abl_se, sl), mid_ci(gnm_se, sl)],
                fmt='none', capsize=4, elinewidth=1.3, color='black')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_title(title, fontsize=FS_TITLE)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')


handles = line_panel(ax_cf_line, layers_cf, cf_memllm_mean, cf_memllm_se, cf_abl_mean, cf_abl_se,
                     cf_gnm_mean, cf_gnm_se, 'CounterFACT Experiment')
bar_panel(ax_cf_bar, cf_memllm_mid, cf_abl_mid, cf_gnm_mid, cf_memllm_se, cf_abl_se, cf_gnm_se, cf_mid,
          f'Mean (Layers {cf_mid.start}–{len(cf_gnm_mean)+cf_mid.stop})')
line_panel(ax_pf_line, layers_pf, pf_memllm_mean, pf_memllm_se, pf_abl_mean, pf_abl_se,
           pf_gnm_mean, pf_gnm_se, 'PersonaMem Experiment')
bar_panel(ax_pf_bar, pf_memllm_mid, pf_abl_mid, pf_gnm_mid, pf_memllm_se, pf_abl_se, pf_gnm_se, pf_mid,
          f'Mean (Layers {pf_mid.start}–{pf_mid.stop})')

ax_cf_line.legend(handles=list(handles), fontsize=FS_LEGEND, loc='upper left', frameon=False,
                  bbox_to_anchor=(-0.02, 1.02), handlelength=1.1, handletextpad=0.5)
fig.tight_layout()

for ext in ("png", "pdf"):
    out = os.path.join(REPO, f"plots/fig6_memory_selectivity.{ext}")
    fig.savefig(out, dpi=600, bbox_inches="tight")
    print("wrote", out)
