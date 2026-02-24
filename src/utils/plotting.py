import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# from torchmetrics import Metric, MetricCollection, MeanMetric
# from utils.metrics import BaseMetricsTracker
# import json
import os
from pathlib import Path
import wandb

METRIC_PRETTY_NAMES = {
    "accuracy": "Accuracy",
    "specificity": "Specificity",
    "selectivity": "Selectivity",
    "fact_accuracy": "Fact Accuracy",
    "fact_selectivity": "Fact Selectivity",
    "fact_specificity": "Fact Specificity",
    "format_accuracy": "Format Accuracy",
    "format_selectivity": "Format Selectivity"
}

METRICS_TO_PLOT_HEATMAPS = [
    "neural_memory_updates",
    "accuracy_matrix",
    "specificity_matrix",
    "selectivity_matrix",
    "fact_accuracy_matrix",
    "fact_selectivity_matrix",
    "fact_specificity_matrix",
    "format_accuracy_matrix",
    "format_selectivity_matrix"]

METRICS_TO_PLOT_BAR_CHARTS = [
    "accuracy",
    "specificity",
    "selectivity",
    "fact_accuracy",
    "fact_selectivity",
    "fact_specificity",
    "unlearning_fact_accuracy",
    "unlearning_fact_selectivity",
    "unlearning_fact_specificity",
    "format_accuracy",
    "format_selectivity",
    "refusal_accuracy",
    "refusal_precision",
    "refusal_recall",
    "refusal_selectivity",
    "refusal_specificity"]

METRICS_TO_PLOT_LINE_CHARTS = [
    "retention",
    "fact_retention",
    "fact_accuracy_over_time",
    "format_retention",
    "refusal_retention",
    "refusal_precision_over_time",
    "refusal_recall_over_time",
    "flops_per_token"
]


def _ensure_output_dir(path: str | Path | None):
    if path is None:
        return
    Path(path).mkdir(parents=True, exist_ok=True)

def _ordered_model_names_from_groups(groups: dict[str, list]) -> list[str]:
    """Stable, deduped model order across groups by first appearance."""
    seen, ordered = set(), []
    for gname in groups:
        for tr in groups[gname]:
            name = tr.model_name or "unnamed"
            if name not in seen:
                seen.add(name)
                ordered.append(name)
    return ordered


def generate_plots(
    *,
    trackers: list | None = None,                   # simple mode
    groups: dict[str, list] | None = None,          # grouped mode: {"val_id":[...], "val_ood":[...]}
    metrics_for_heatmaps: list[str] = METRICS_TO_PLOT_HEATMAPS,
    metrics_for_bars: list[str] = METRICS_TO_PLOT_BAR_CHARTS,
    metrics_for_line_charts: list[str] = METRICS_TO_PLOT_LINE_CHARTS,
    epoch: int,
    directory_to_save: str | None = None,
    wandb_run=None
):
    """
    - Creates heatmaps for any tracker’s heatmap metrics (per-tracker like before).
    - Creates bar charts:
        * If trackers provided (>=2): simple bars per metric.
        * If groups provided: grouped bars per metric, cluster by model, bars by group.

    Notes:
      - We log with wandb.Image(fig) BEFORE plt.close(fig).
      - If directory_to_save is given, PNGs are written too.
    """
    _ensure_output_dir(directory_to_save)
    assert wandb_run is not None or directory_to_save is not None, "Nowhere to save plots: no wandb run or output directory specified"
    payload = {}

    # ---------
    # HEATMAPS
    # ---------
    
    if trackers is not None:
        for tr in trackers:
            if not hasattr(tr, "epoch_results") or tr.epoch_results is None:
                continue
            for metric in tr.epoch_results.keys():
                if metric not in metrics_for_heatmaps:
                    continue
                try:
                    # data = tr.epoch_results[metric][epoch]
                    data = tr.epoch_results[metric][-1]
                    arr = np.array(data)
                    if arr.size == 0:
                        continue
                    if arr.ndim == 0:
                        arr = arr.reshape(1, 1)
                    elif arr.ndim == 1:
                        arr = arr.reshape(-1, 1)
                except Exception:
                    continue

                fig = plot_heatmap(
                    model_pretty_name=(tr.pretty_name if hasattr(tr, "pretty_name") else (tr.model_name or "model")),
                    metric_pretty_name=metric,
                    metric_key=metric,
                    values_to_graph=arr
                )

                if wandb_run is not None:
                    payload[f"epoch_{epoch}/{tr.dataset_name}/{metric}"] = wandb.Image(fig)

                if directory_to_save:
                    out = Path(directory_to_save) / f"{metric}_{tr.model_name or 'model'}_heatmap_{epoch}.png"
                    fig.savefig(out, bbox_inches="tight")

                plt.close(fig)


    # ---------
    # LINECHARTS
    # ---------
    if trackers is not None:
        for tr in trackers:
            if not hasattr(tr, "epoch_results") or tr.epoch_results is None:
                continue
            for metric in tr.epoch_results.keys():
                if metric not in metrics_for_line_charts:
                    continue
                try:
                    # series = tr.epoch_results[metric][epoch]
                    series = tr.epoch_results[metric][-1]
                    x_values = list(range(len(series)))
                    y_values = [float(np.nanmean(np.array(v))) for v in series]
                except Exception:
                    continue

                fig = plot_linechart(
                    x_values=x_values,
                    y_values=y_values,
                    title=f"{tr.pretty_name if hasattr(tr, 'pretty_name') else (tr.model_name or 'model')} - {METRIC_PRETTY_NAMES.get(metric, metric)} ",
                    xlabel="Documents seen since target fact seen",
                    ylabel=METRIC_PRETTY_NAMES.get(metric, metric)
                )

                if wandb_run is not None:
                    payload[f"epoch_{epoch}/{tr.dataset_name}/{metric}"] = wandb.Image(fig)

                if directory_to_save:
                    out = Path(directory_to_save) / f"{metric}_{tr.model_name or 'model'}_linechart.png"
                    fig.savefig(out, bbox_inches="tight")

                plt.close(fig)

    # ---------
    # BARCHARTS
    # ---------

    # SIMPLE: multiple trackers, no groups
    if groups is None and len(trackers) >= 1:
        x_axis_names = [t.dataset_name or f"dataset_{i}" for i, t in enumerate(trackers)]
        # for metric in metrics_for_bars:
        for metric in trackers[0].epoch_results.keys():
            if metric not in metrics_for_bars:
                continue
            # vals = np.array([t.epoch_results[metric][epoch] for t in trackers], dtype=float)
            vals = np.array([t.epoch_results[metric][-1] for t in trackers], dtype=float)
            pretty = METRIC_PRETTY_NAMES.get(metric, metric)
            title = f"{pretty} Comparison"
            fig = plot_barchart(
                categories=x_axis_names,
                series=vals,              
                title=title,
                xlabel="Dataset",
                ylabel=pretty
            )
            if wandb_run is not None:
                payload[f"epoch_{epoch}/{metric}_barchart"] = wandb.Image(fig)
            if directory_to_save:
                out = Path(directory_to_save) / f"{metric}_barchart.png"
                fig.savefig(out, bbox_inches="tight")
            plt.close(fig)

    # GROUPED: {"val_id":[...], "val_ood":[...]}
    elif groups is not None and len(groups) > 1:
        group_names = list(groups.keys())
        model_names = _ordered_model_names_from_groups(groups)

        # quick index for each group by tracker.name
        for metric in metrics_for_bars:
            mat = []
            for m in model_names:
                row = []
                for g in group_names:
                    lookup = { (t.model_name or "unnamed"): t for t in groups[g] }
                    tr = lookup.get(m, None)
                    v = tr.epoch_results[metric][-1] if tr is not None else np.nan
                    # v = tr.epoch_results[metric][epoch] if tr is not None else np.nan
                    row.append(v)
                mat.append(row)
            series = np.array(mat, dtype=float)  # shape (n_models, n_groups)

            pretty = METRIC_PRETTY_NAMES.get(metric, metric)
            title = f"Grouped Comparison — {pretty} (epoch {epoch})"
            fig = plot_barchart(
                categories=model_names,
                series=series,              # 2D
                title=title,
                ylabel=pretty,
                xlabel="Model",
                group_names=group_names
            )
            if wandb_run is not None:
                payload[f"epoch_{epoch}_barcharts/{metric}_grouped"] = wandb.Image(fig)
            if directory_to_save:
                out = Path(directory_to_save) / f"{metric}_grouped_bar_epoch_{epoch}.png"
                fig.savefig(out, bbox_inches="tight")
            plt.close(fig)

    # One consolidated wandb.log call
    if payload and (wandb_run is not None):
        wandb_run.log(payload)


def plot_heatmap(*, model_pretty_name:str, metric_pretty_name:str, metric_key:str, values_to_graph:np.ndarray):
    print(f"[plot_heatmap] Plotting heatmap for metric '{metric_key}'")
    arr = values_to_graph
    
    if metric_key == "neural_memory_updates":
        y_label = "Prefix token index" 
        x_label = "Sequence idx"        
    else:
        y_label = "Input-output pair index"
        x_label = "Metalearner index"
    
    # size = values_to_graph.shape[0]
    # If 1-D, promote to 2-D (N,1)
    if arr.ndim == 1:
        print(f"[plot_heatmap] Promoting 1-D array to 2-D for metric '{metric_key}'")
        arr = arr.reshape(-1, 1)

    # plt.figure(figsize=(max(4, arr.shape[1]), max(4, arr.shape[0])))
    fig, ax = plt.subplots(figsize=(max(4, arr.shape[1]), max(4, arr.shape[0])))
    sns.heatmap(arr, annot=True, fmt=".2f", cmap="viridis", cbar=True,
                xticklabels=[str(i) for i in range(arr.shape[1])],
                yticklabels=[str(i) for i in range(arr.shape[0])])
    ax.set_title(f"{model_pretty_name} - {metric_pretty_name}", fontsize=14, wrap=True, pad=12)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.tight_layout()
    return fig

    # print(f"[plot_heatmap] Saving heatmap to directory: {directory_to_save}")
    # # os.makedirs(directory_to_save, exist_ok=True)
    # if name_suffix:
    #     plt.savefig(f"{directory_to_save}/{metric_key}_{model_name}_heatmap_{name_suffix}.png")
    # else:
    #     plt.savefig(f"{directory_to_save}/{metric_key}_{model_name}_heatmap.png")
    # plt.close()

def plot_barchart(
    *,
    categories: list[str],           # x-axis labels (e.g., model names)
    series: np.ndarray,              # shape (N,) or (N, G) where G=len(group_names)
    title: str,
    ylabel: str,
    xlabel: str,
    group_names: list[str] | None = None,
    dpi: int = 140,
):
    """
    Atomic bar chart creator.
      - If series.ndim == 1: simple bars (len(series) == len(categories))
      - If series.ndim == 2: grouped bars with group_names labeling each bar in a cluster
    Returns: fig (you handle save/log/close outside)
    """
    print(f"[plot_barchart] Plotting bar chart: {title}")
    series = np.asarray(series, dtype=float)
    assert series.ndim in (1, 2), "series must be 1D or 2D"
    assert len(categories) == series.shape[0], "len(categories) must equal series.shape[0]"

    fig_h = 5.5 if series.ndim == 2 else 5.0
    fig_w = max(9.5, 1.1 * len(categories))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    x = np.arange(len(categories))
    ymax = np.nanmax(series) if np.isfinite(np.nanmax(series)) else 1.0
    pad = 0.12 * (ymax if ymax > 0 else 1.0)

    if series.ndim == 1:
        bars = ax.bar(x, series)
        for rect, v in zip(bars, series):
            if np.isfinite(v):
                ax.text(rect.get_x() + rect.get_width()/2.0, v + pad*0.1, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=9)
    else:
        assert group_names is not None and len(group_names) == series.shape[1], \
            "group_names must match series.shape[1]"
        G = series.shape[1]
        width = min(0.8 / max(G, 1), 0.28)
        for j in range(G):
            offs = x + (j - (G-1)/2.0) * width
            bars = ax.bar(offs, series[:, j], width, label=group_names[j])
            for rect, v in zip(bars, series[:, j]):
                if np.isfinite(v):
                    ax.text(rect.get_x() + rect.get_width()/2.0, v + pad*0.1, f"{v:.3f}",
                            ha="center", va="bottom", fontsize=9)
        ax.legend(frameon=False, ncols=min(len(group_names), 3), loc="upper left", bbox_to_anchor=(0, 1.02))

    ax.set_title(title, fontsize=14, pad=10)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.set_ylim(0, ymax * 1.12 if ymax > 0 else 1.0)
    fig.tight_layout()
    return fig

def plot_linechart(
    *,
    x_values: list | np.ndarray,
    y_values: list | np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    dpi: int = 140,
):
    """
    Atomic line chart creator.
    Returns: fig (you handle save/log/close outside)
    """
    fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)

    ax.plot(x_values, y_values, marker='o')

    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True)
    int_ticks = np.arange(int(min(x_values)), int(max(x_values)) + 1)
    ax.set_xticks(int_ticks)
    fig.tight_layout()
    return fig



# def _extract_scalar(tracker: MetricsTracker, metric: str, epoch: int) -> float:
#     """
#     Pull a scalar for (metric, epoch) from a tracker's epoch_results.
#     If the stored value is an array, returns np.nanmean of it.
#     """
#     try:
#         series = tracker.epoch_results.get(metric, None)
#         if series is None:
#             return np.nan
#         value = series[epoch]
#         arr = np.array(value)
#         if arr.size == 0:
#             return np.nan
#         return float(np.nanmean(arr))
#     except Exception:
#         return np.nan
# def generate_epoch_plots(*, train_tracker: MetricsTracker, 
#                         val_tracker_id: MetricsTracker, 
#                         val_tracker_ood:MetricsTracker, 
#                         training_run_dir: str, 
#                         epoch: int = None,
#                         wandb_run):

#     directory_path_for_train_plots = f"{training_run_dir}/results/plots_train"
#     directory_path_for_val_plots_id = f"{training_run_dir}/results/plots_val_id"
#     directory_path_for_val_plots_ood = f"{training_run_dir}/results/plots_val_ood"

#     os.makedirs(directory_path_for_train_plots, exist_ok=True)
#     os.makedirs(directory_path_for_val_plots_id, exist_ok=True)
#     os.makedirs(directory_path_for_val_plots_ood, exist_ok=True)

#     generate_plots(tracker=train_tracker,  directory_to_save=directory_path_for_train_plots, epoch=epoch, is_final=False)
#     generate_plots(tracker=val_tracker_id,  directory_to_save=directory_path_for_val_plots_id, epoch=epoch, is_final=False)
#     generate_plots(tracker=val_tracker_ood,  directory_to_save=directory_path_for_val_plots_ood, epoch=epoch, is_final=False)
    
#     run = wandb_run

#     plots_dir_train = Path(directory_path_for_train_plots)
#     plots_dir_ood   = Path(directory_path_for_val_plots_ood)
#     plots_dir_id    = Path(directory_path_for_val_plots_id)
    
#     acc_matrix_train_path = plots_dir_train / f'accuracy_matrix_model_heatmap_{epoch}.png'
#     specificity_matrix_train_path = plots_dir_train / f'specificity_matrix_model_heatmap_{epoch}.png'
#     selectivity_matrix_train_path = plots_dir_train / f'selectivity_matrix_model_heatmap_{epoch}.png'
#     neural_memory_updates_matrix_train_path = plots_dir_train / f'neural_memory_updates_model_heatmap_{epoch}.png'

#     acc_matrix_ood_path = plots_dir_ood / f'accuracy_matrix_model_heatmap_{epoch}.png'
#     specificity_matrix_ood_path = plots_dir_ood / f'specificity_matrix_model_heatmap_{epoch}.png'
#     selectivity_matrix_ood_path = plots_dir_ood / f'selectivity_matrix_model_heatmap_{epoch}.png'
#     neural_memory_updates_matrix_ood_path = plots_dir_ood / f'neural_memory_updates_model_heatmap_{epoch}.png'

#     acc_matrix_id_path = plots_dir_id / f'accuracy_matrix_model_heatmap_{epoch}.png'
#     specificity_matrix_id_path = plots_dir_id / f'specificity_matrix_model_heatmap_{epoch}.png'
#     selectivity_matrix_id_path = plots_dir_id / f'selectivity_matrix_model_heatmap_{epoch}.png'
#     neural_memory_updates_matrix_id_path = plots_dir_id / f'neural_memory_updates_model_heatmap_{epoch}.png'
#     try:
#         run.log({f"epoch_{epoch}_checkpoint/train/accuracy_matrix": wandb.Image(str(acc_matrix_train_path))})
#         run.log({f"epoch_{epoch}_checkpoint/train/specificity_matrix": wandb.Image(str(specificity_matrix_train_path))})
#         run.log({f"epoch_{epoch}_checkpoint/train/selectivity_matrix": wandb.Image(str(selectivity_matrix_train_path))})
#         run.log({f"epoch_{epoch}_checkpoint/train/neural_memory_updates_matrix": wandb.Image(str(neural_memory_updates_matrix_train_path))})
#         run.log({f"epoch_{epoch}_checkpoint/val_ood/accuracy_matrix": wandb.Image(str(acc_matrix_ood_path))})
#         run.log({f"epoch_{epoch}_checkpoint/val_ood/specificity_matrix": wandb.Image(str(specificity_matrix_ood_path))})
#         run.log({f"epoch_{epoch}_checkpoint/val_ood/selectivity_matrix": wandb.Image(str(selectivity_matrix_ood_path))})
#         run.log({f"epoch_{epoch}_checkpoint/val_ood/neural_memory_updates_matrix": wandb.Image(str(neural_memory_updates_matrix_ood_path))})
#         run.log({f"epoch_{epoch}_checkpoint/val_id/accuracy_matrix": wandb.Image(str(acc_matrix_id_path))})
#         run.log({f"epoch_{epoch}_checkpoint/val_id/specificity_matrix": wandb.Image(str(specificity_matrix_id_path))})
#         run.log({f"epoch_{epoch}_checkpoint/val_id/selectivity_matrix": wandb.Image(str(selectivity_matrix_id_path))})
#         run.log({f"epoch_{epoch}_checkpoint/val_id/neural_memory_updates_matrix": wandb.Image(str(neural_memory_updates_matrix_id_path))})
#     except Exception as e:
#         print(f"Error logging to wandb at epoch {epoch}: {e}")
# def generate_plots(*, trackers: list[MetricsTracker], logger, directory_to_save: str=None, epoch: int = None, compare: bool = False, is_final: bool = False):
    
#     payload = {}

#     for tracker in trackers:
#         performance_summary = tracker.epoch_results
#         if performance_summary.keys() is None:
#             print("[generate_plots] No performance summary available to plot. - probably forgetting to call .log_epoch()")
#             return
        
#         # GENERATE INDIVIDUAL GRAPHS
#         for metric in performance_summary.keys():
#             print(f"[generate_plots] Processing metric '{metric}'")
#             if metric in METRICS_TO_PLOT_HEATMAPS:
#                 try:
#                     data = performance_summary[metric][epoch]
#                     values_to_graph = np.array(data)
#                     if values_to_graph.size == 0:
#                         print(f"[generate_plots] Skip {metric}: empty")
#                         continue
#                     if values_to_graph.ndim == 0:          # scalar -> 1x1
#                         print(f"[generate_plots] Promoting 0-D array to 2-D for metric '{metric}'")
#                         values_to_graph = values_to_graph.reshape(1, 1)
#                     elif values_to_graph.ndim == 1:        # vector -> Nx1
#                         print(f"[generate_plots] Promoting 1-D array to 2-D for metric '{metric}'")
#                         values_to_graph = values_to_graph.reshape(-1, 1)
#                 except Exception as e:
#                     print(f"Skipping metric '{metric}' due to error: {e}")
#                     continue
#                 fig = plot_heatmap(metric_pretty_name=metric,
#                             model_pretty_name="model",
#                             model_name="model",
#                             metric_key=metric,
#                             values_to_graph=values_to_graph,
#                             directory_to_save=directory_to_save,
#                             name_suffix=None if is_final else f"{epoch}"
#                             )
#                 if tracker.name:
#                     payload[f"epoch_{epoch}/{tracker.name}/{metric}"] = wandb.Image(fig)
#                 else: 
#                     payload[f"epoch_{epoch}/model/{metric}"] = wandb.Image(fig)
#                 plt.close(fig)
    
#     # GENERATE COMPARE GRAPHS (if compare=True)
#     if compare:
#         if len(trackers) <2:
#             print("[generate_plots] Compare flag is set but less than 2 trackers provided. Skipping compare plots.")
#         pass


#     if payload:
#         logger.log(payload) 
# def plot_barchart(*, model_pretty_name:str, metric_pretty_name:str, metric_key:str, values_to_graph: np.ndarray):
#     """
#     Compare the performance of multiple models based on their metrics trackers.
#     Creates one bar chart per metric.
#     """
#     arr = values_to_graph

#     # set titles
#     if metric_key == "neural_memory_updates":
#         y_label = "Prefix token index" 
#         x_label = "Sequence idx"        
#     else:
#         y_label = "Input-output pair index"
#         x_label = "Metalearner index"


#     fig, ax = plt.subplots(figsize=(12, 6))
#     ax = sns.barplot(x=model_names, y=metric_values, palette="muted")
#     metric_pretty_name = METRIC_PRETTY_NAMES.get(metric_name, metric_name)
#     ax.set_title(f"Model Comparison: {metric_pretty_name}", fontsize=16)
#     ax.set_ylabel(metric_pretty_name, fontsize=14)
#     ax.set_xlabel("Model", fontsize=14)
#     ax.set_ylim(0, max(metric_values) * 1.1)
#     for i, v in enumerate(metric_values):
#         ax.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom', fontsize=12)
    
#     # Rotate and wrap model names
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', wrap=True)

#     plt.tight_layout()
#     plt.savefig(f"{directory_to_save}/{metric_name}_bar_chart.png")
#     plt.close()
