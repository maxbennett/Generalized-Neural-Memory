import re
import string

# import jieba
# from fuzzywuzzy import fuzz
import difflib
import numpy as np
import numbers
from typing import List
from collections import Counter
# from rouge import Rouge
from abc import abstractmethod


import torch
from torch import Tensor
from torchmetrics import Metric, MetricCollection, MeanMetric, SumMetric
from utils.distributed import rprint
from instructions import is_refusal

# from torch.profiler import profile, ProfilerActivity

def make_serializable(val):
    # Torch scalars/arrays
    if isinstance(val, Tensor):
        return val.detach().cpu().item() if val.numel() == 1 else val.detach().cpu().tolist()

    # NumPy scalars & arrays
    if isinstance(val, np.generic):   # np.float32, np.int64, np.bool_
        return val.item()
    if isinstance(val, np.ndarray):
        return val.tolist()

    # Python numbers
    if isinstance(val, numbers.Number):
        return val

    # Containers
    if isinstance(val, dict):
        return {k: make_serializable(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [make_serializable(v) for v in val]

    # Last resort: leave as-is (avoid forcing to str)
    return val


# -----------------------
# Trackers
# -----------------------

class BaseMetricsTracker:
    def __init__(self, *, model_name: str, dataset_name:str, device: torch.device | None = None, starting_epoch: int = 0):
        
        self.epoch_results = dict()
        self.model_name = model_name
        self.dataset_name = dataset_name
        self._pad_epochs_pending = int(max(0, starting_epoch))  
        
        coll = self.build_collection()
        if not isinstance(coll, MetricCollection):
            raise TypeError(
                f"{self.__class__.__name__}.build_collection() must return a torchmetrics.MetricCollection"
            )
        self.collection = coll
        if device is not None:
            self.collection.to(device)
    

    @abstractmethod
    def build_collection(self) -> MetricCollection:
        """Return a fully-initialized MetricCollection for this tracker."""
        ...

    def state_dict(self) -> dict:
        return {
            "epoch_results": self.epoch_results
        }

    def load_state_dict(self, state: dict):
        self.epoch_results = state.get("epoch_results", {})
    

    def compute(self):    
        return self.collection.compute()

    def reset(self):
        self.collection.reset()

    def to(self, device):
        self.collection.to(device)
        return self
    
    def log_epoch(self):
        """
        Computes metrics and stores results in a serializable format.
        """
        results = {}
        for name, metric in self.collection.items():
            result = metric.compute()
            if result is None:
                continue  # Skip if never updated
            # result might be a dict (flattened), so merge
            if isinstance(result, dict):
                results.update(result)
            else:
                results[name] = result

        serialized_results = make_serializable(results)

        # --- NEW: pre-pad stump epochs once, right before first real append ---
        if self._pad_epochs_pending > 0:
            # make sure all keys that will exist this epoch are initialized
            for k in serialized_results.keys():
                if k not in self.epoch_results:
                    self.epoch_results[k] = []
            # also include any keys that might have been created earlier (e.g., from load_state_dict)
            for k in self.epoch_results.keys():
                while len(self.epoch_results[k]) < self._pad_epochs_pending:
                    self.epoch_results[k].append(None)
            self._pad_epochs_pending = 0
        # --- END NEW ---

        for key, value in serialized_results.items():
            if key in self.epoch_results:
                self.epoch_results[key].append(value)
            else:
                self.epoch_results[key] = [value]
 

    def __getitem__(self, key: str):
        return self.collection[key]



class ExperimentTracker(BaseMetricsTracker):

    def __init__(self, *, model_name: str, dataset_name:str, sequence_length: int, device: torch.device | None = None, starting_epoch: int = 0, target_types: list[str], unlearning: bool = False):
        self.seq_len = sequence_length
        self.target_types = target_types
        self.unlearning = unlearning
        super().__init__(model_name=model_name, dataset_name=dataset_name, device=device, starting_epoch=starting_epoch)
        
        
    def build_collection(self) -> MetricCollection:
        metrics = {
            "loss": MeanMetric(),
            "compositions": SumMetric(),
            "non_compositions": SumMetric(),
            "fact_accuracy": MeanMetric(),
            "fact_selectivity": MeanMetric(),
            "fact_specificity": MeanMetric(),
            "fact_accuracy_matrix": MeanMatrix(x_axis_length=self.seq_len, y_axis_length=self.seq_len),
            "fact_selectivity_matrix": MeanMatrix(x_axis_length=self.seq_len, y_axis_length=self.seq_len),
            "fact_specificity_matrix": MeanMatrix(x_axis_length=self.seq_len, y_axis_length=self.seq_len),
            "fact_retention": MeanLine(x_axis_length=self.seq_len),
            "fact_accuracy_over_time": MeanLine(x_axis_length=self.seq_len),
            "flops_per_token": MeanLine(x_axis_length=self.seq_len)
        }
        if "formats" in self.target_types or "fact_format_compositions" in self.target_types:
            metrics.update({
                "format_accuracy": MeanMetric(),
                "format_selectivity": MeanMetric(),
                "format_retention": MeanLine(x_axis_length=self.seq_len),
                "format_accuracy_matrix": MeanMatrix(x_axis_length=self.seq_len, y_axis_length=self.seq_len),
                "format_selectivity_matrix": MeanMatrix(x_axis_length=self.seq_len, y_axis_length=self.seq_len),
                # "format_accuracy_0": MeanMetric(),
                # "format_retention_0": MeanLine(x_axis_length=self.seq_len),
                # "format_accuracy_1": MeanMetric(),
                # "format_retention_1": MeanLine(x_axis_length=self.seq_len),
                # "format_accuracy_2": MeanMetric(),
                # "format_retention_2": MeanLine(x_axis_length=self.seq_len)
            })
        if "refusal_categories" in self.target_types or "refusal_subcategories" in self.target_types or "fact_refusal_compositions" in self.target_types:
            metrics.update({
                "refusal_accuracy": MeanMetric(),
                "refusal_recall": MeanMetric(),
                "refusal_precision": MeanMetric(),
                "refusal_specificity": MeanMetric(),
                "refusal_recall_over_time": MeanLine(x_axis_length=self.seq_len),
                "refusal_precision_over_time": MeanLine(x_axis_length=self.seq_len),
                "refusal_precision_matrix": MeanMatrix(x_axis_length=self.seq_len, y_axis_length=self.seq_len),
                "refusal_recall_matrix": MeanMatrix(x_axis_length=self.seq_len, y_axis_length=self.seq_len)
            })
        if self.unlearning:
            metrics.update({
                "unlearning_fact_accuracy": MeanMetric(),
                "unlearning_fact_selectivity": MeanMetric(),
                "unlearning_fact_specificity": MeanMetric()
            })
        
        # metrics.update({
        #     "attn_mem_mass_target": MeanLine(x_axis_length=32),   # we'll reuse seq_len or override later
        #     "attn_mem_mass_ignored": MeanLine(x_axis_length=32),
        #     "attn_mem_entropy_target": MeanLine(x_axis_length=32),
        #     "attn_mem_entropy_ignored": MeanLine(x_axis_length=32),
        # })
        return MetricCollection(metrics)
    def count_compositions(self):
        self.collection["compositions"].update(1)
    def count_non_compositions(self):
        self.collection["non_compositions"].update(1)
    def update_flops_metrics(self, total_flops, *, seq_idx_of_doc: int, out_gen=None, input_ids=None):
        # total_flops = 0
        # for evt in prof.key_averages():
        #     if getattr(evt, "flops", None) is not None:
        #         total_flops += evt.flops
        if out_gen is not None:
            num_new_tokens_per_seq = out_gen.sequences.size(1) - input_ids.size(1)
            num_seqs = out_gen.sequences.size(0)
            num_new_tokens = max(1, num_new_tokens_per_seq * num_seqs)

            flops_per_token = total_flops / num_new_tokens

        else:
            flops_per_token = total_flops  # fallback if we don't have generation info
        self.collection["flops_per_token"].update(seq_idx_of_doc, flops_per_token)

    # def update_attention_probe_metrics(
    #     self,
    #     *,
    #     is_target_query: bool,
    #     layer: int,
    #     attn_mass: float,
    #     attn_entropy: float,
    # ):
    #     if is_target_query:
    #         self.collection["attn_mem_mass_target"].update(layer, attn_mass)
    #         self.collection["attn_mem_entropy_target"].update(layer, attn_entropy)
    #     else:
    #         self.collection["attn_mem_mass_ignored"].update(layer, attn_mass)
    #         self.collection["attn_mem_entropy_ignored"].update(layer, attn_entropy)
    def update_unlearning_fact_metrics(self, correct: int, *, pair: dict,
               seq_idx_of_doc: int, seq_idx_of_io_pairs: int,
               metadata_blob: dict):
        if not pair['should_refuse']:
            if pair['fact_id_targeted_for_fact_learning'] == True and pair['type'] == "paraphrase":
                if seq_idx_of_doc == seq_idx_of_io_pairs:
                    self.collection["unlearning_fact_accuracy"].update(correct)

            elif pair['fact_id_targeted_for_fact_learning'] == True and pair['type'] == "true_output_neighbor":
                if seq_idx_of_doc == seq_idx_of_io_pairs:
                    self.collection["unlearning_fact_specificity"].update(correct)

            elif pair['fact_id_targeted_for_fact_learning'] == False and pair['type'] == "paraphrase":
                if seq_idx_of_doc == seq_idx_of_io_pairs:
                    self.collection["unlearning_fact_selectivity"].update(correct)
    
    def update_fact_metrics(self, correct: int, *, pair: dict,
               seq_idx_of_doc: int, seq_idx_of_io_pairs: int,
               metadata_blob: dict):
        
        if not pair['should_refuse']:
            if pair['fact_id_targeted_for_fact_learning'] == True and pair['type'] == "paraphrase":
                if seq_idx_of_doc == seq_idx_of_io_pairs:
                    self.collection["fact_accuracy"].update(correct)
                    self.collection["fact_accuracy_over_time"].update(seq_idx_of_doc, correct)
                if seq_idx_of_doc >= seq_idx_of_io_pairs:
                    time_ago = seq_idx_of_doc - seq_idx_of_io_pairs
                    self.collection["fact_retention"].update(time_ago, correct)
                self.collection["fact_accuracy_matrix"].update(seq_idx_of_doc, seq_idx_of_io_pairs, correct)

            elif pair['fact_id_targeted_for_fact_learning'] == True and pair['type'] == "true_output_neighbor":
                if seq_idx_of_doc == seq_idx_of_io_pairs:
                    self.collection["fact_specificity"].update(correct)
                self.collection["fact_specificity_matrix"].update(seq_idx_of_doc, seq_idx_of_io_pairs, correct)

            elif pair['fact_id_targeted_for_fact_learning'] == False and pair['type'] == "paraphrase":
                if seq_idx_of_doc == seq_idx_of_io_pairs:
                    self.collection["fact_selectivity"].update(correct)
                self.collection["fact_selectivity_matrix"].update(seq_idx_of_doc, seq_idx_of_io_pairs, correct)
    
    def update_format_metrics(self, *,
                                format_correct: int, 
                                seq_idx_of_doc: int, 
                                seq_idx_of_io_pairs: int,
                                target_format: str,
                                time_since_format_instruction: int | None,
                                metadata_blob: dict
                                # data_idx: int
                                ):
        
        target_to_learn = metadata_blob['target_to_learn']
        
        
        if time_since_format_instruction is not None and seq_idx_of_doc >= seq_idx_of_io_pairs:
            self.collection["format_retention"].update(time_since_format_instruction, format_correct)
            self.collection["format_accuracy_matrix"].update(seq_idx_of_doc, seq_idx_of_io_pairs, format_correct)
            if time_since_format_instruction == 0: # this means that this is currently the target format
                self.collection["format_accuracy"].update(format_correct)
        
        

        if seq_idx_of_doc == seq_idx_of_io_pairs and target_to_learn != "format" and target_format is None:
            self.collection["format_selectivity"].update(format_correct)
        
        # if data_idx == 0:
        #     if time_since_format_instruction is not None and seq_idx_of_doc >= seq_idx_of_io_pairs:
        #         self.collection["format_retention_0"].update(time_since_format_instruction, format_correct)
        #         if time_since_format_instruction == 0: # this means that this is currently the target format
        #             self.collection["format_accuracy_0"].update(format_correct)

        # elif data_idx == 1:
        #     if time_since_format_instruction is not None and seq_idx_of_doc >= seq_idx_of_io_pairs:
        #         self.collection["format_retention_1"].update(time_since_format_instruction, format_correct)
        #         if time_since_format_instruction == 0: # this means that this is currently the target format
        #             self.collection["format_accuracy_1"].update(format_correct)
            
        # elif data_idx == 2:
        #     if time_since_format_instruction is not None and seq_idx_of_doc >= seq_idx_of_io_pairs:
        #         self.collection["format_retention_2"].update(time_since_format_instruction, format_correct)
        #         if time_since_format_instruction == 0: # this means that this is currently the target format
        #             self.collection["format_accuracy_2"].update(format_correct)
            


    def update_refusal_metrics(self, *,
                                    generated_text: str,
                                    pair: dict,
                                    seq_idx_of_doc: int,
                                    seq_idx_of_io_pairs: int,
                                    metadata_blob: dict
                                    ):
        pred_refusal  = is_refusal(generated_text)
        true_refusal  = pair['should_refuse']
        refusal_correct = (pred_refusal == true_refusal)

        if pair['type'] == "paraphrase":
            if true_refusal:
                self.collection["refusal_recall_matrix"].update(seq_idx_of_doc, seq_idx_of_io_pairs, int(refusal_correct))
            if pred_refusal:
                self.collection["refusal_precision_matrix"].update(seq_idx_of_doc, seq_idx_of_io_pairs, int(refusal_correct))
            if seq_idx_of_doc == seq_idx_of_io_pairs:
                self.collection["refusal_accuracy"].update(int(refusal_correct))
                if true_refusal:
                    self.collection["refusal_recall"].update(int(refusal_correct))
                if pred_refusal:
                    self.collection["refusal_precision"].update(int(refusal_correct))
            if seq_idx_of_doc >= seq_idx_of_io_pairs:
                time_ago = seq_idx_of_doc - seq_idx_of_io_pairs
                if true_refusal:
                    self.collection["refusal_recall_over_time"].update(time_ago, int(refusal_correct))
                if pred_refusal:
                    self.collection["refusal_precision_over_time"].update(time_ago, int(refusal_correct))
        elif pair['type'] == "true_output_neighbor":
            if seq_idx_of_doc == seq_idx_of_io_pairs:
                if pair['neighbor_of_should_refuse']:
                    self.collection["refusal_specificity"].update(int(refusal_correct))


    def update_loss(self, loss: float):
        """Update the training loss for a specific step."""
        with torch.no_grad():
            self.collection["loss"].update(loss)

# -----------------------
# Metrics
# -----------------------
class MeanMatrix(Metric):
    def __init__(self, x_axis_length: int, y_axis_length: int):
        super().__init__()
        self.x_axis_length = x_axis_length
        self.y_axis_length = y_axis_length
        self.add_state(
            "total",
            default=torch.zeros(y_axis_length, x_axis_length, dtype=torch.int32),
            dist_reduce_fx="sum"
        )
        self.add_state(
            "values",
            default=torch.zeros(y_axis_length, x_axis_length, dtype=torch.float32),
            dist_reduce_fx="sum"
        )

    def update(self, x_axis: int, y_axis: int, value: float):
        self.total[y_axis, x_axis] += 1
        self.values[y_axis, x_axis] += value

    def reset(self):
        self.total.zero_()
        self.values.zero_()

    def compute(self):
        mask = self.total > 0
        avg = torch.zeros_like(self.values).float() 
        avg[mask] = self.values[mask].float() / self.total[mask].float()
        return avg
    
class MeanLine(Metric):
    def __init__(self, x_axis_length: int):
        super().__init__()
        self.add_state(
            "total",
            default=torch.zeros(x_axis_length, dtype=torch.int32),
            dist_reduce_fx="sum"
        )
        self.add_state(
            "values",
            default=torch.zeros(x_axis_length, dtype=torch.float32),
            dist_reduce_fx="sum"
        )

    def update(self, x_axis: int, value: float):
        self.total[x_axis] += 1
        self.values[x_axis] += value

    def reset(self):
        self.total.zero_()
        self.values.zero_()

    def compute(self):
        mask = self.total > 0
        avg = torch.zeros_like(self.values).float() 
        avg[mask] = self.values[mask].float() / self.total[mask].float()
        return avg
