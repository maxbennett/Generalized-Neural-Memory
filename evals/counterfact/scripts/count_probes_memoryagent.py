"""CPU replay of the seq-10 eval dataloader to count probes per metric bucket.

Replicates the validate() probe stream (seed 42, 2 ranks, DistributedSampler,
drop_last, chunk clipping via min-across-ranks) without running any model, so we
can attach binomial standard errors to the run-level metrics. Probe counts are
model-independent (chunk clipping depends only on data), so one pass serves
every row of the comparison table.
"""
import sys, os, json
from functools import partial
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "MemoryLLM/src"))

from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed  # noqa
from utils.repro import seed_everything
from utils.data_helpers import filter_all_future_paraphrases, chunk_list
from gnm_data import DocDataset, docs_collate_fn
from config_memoryllm_train import load_cfg, ModelCfg

EVAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = load_cfg(os.path.join(EVAL_DIR, "params_memoryagent_short_horizon.yaml"),
               modelcfg=ModelCfg(model="llama3_8b_instruct", pretty_name="count", model_state_dict_path=None))
SEQ = cfg.loop.sequence_length
WORLD = 2


class _FakeSampler(DistributedSampler):
    pass


def rank_pass(rank):
    """Iterate all datasets for one rank; return per-(dataset,batch,step) chunk lists."""
    seed_everything(cfg.seed, rank=rank)
    out = {}  # (d, i, seq_idx) -> list of chunk sizes and pair metadata
    for d, path in enumerate(cfg.data.val_ood_datasets):
        ds = DocDataset(os.path.join(EVAL_DIR, path) if not os.path.isabs(path) else path,
                        shuffle=False, limit=cfg.data.val_limit)
        sampler = DistributedSampler(ds, num_replicas=WORLD, rank=rank, shuffle=False)
        dl = DataLoader(ds, batch_size=SEQ, sampler=sampler, drop_last=True, shuffle=False,
                        collate_fn=partial(docs_collate_fn, train_or_test="test",
                                           eligible_target_types=cfg.data.eligible_target_types_val_ood,
                                           holdout_targets=cfg.data.holdout_targets_val_ood,
                                           holdout_target_values=cfg.data.holdout_target_values_val_ood,
                                           sequence_length=SEQ,
                                           io_sampling_params=cfg.sampling,
                                           unlearning=False))
        sampler.set_epoch(0)
        for i, (documents, learning_instructions, input_output_pairs, metadata) in enumerate(dl):
            batch_size = len(documents[0])
            for b in range(batch_size):
                target_format = None
                fmt_seq_idx = None
                for seq_idx in range(SEQ):
                    tt = metadata[seq_idx][b]['target_type_to_learn']
                    if tt == "formats":
                        target_format = metadata[seq_idx][b]['format_in_doc']
                        if fmt_seq_idx is None:
                            fmt_seq_idx = seq_idx
                    pairs_all = []
                    for io_idx in range(seq_idx + 1):
                        p = filter_all_future_paraphrases(input_output_pairs[io_idx][b], seq_idx)
                        for pair in p:
                            pair = dict(pair)
                            pair["seq_idx_of_io_pairs"] = io_idx
                            pairs_all.append(pair)
                    chunks = chunk_list(pairs_all, cfg.sampling.num_io_pairs_to_chunk)
                    out[(d, i, b, seq_idx)] = (chunks, tt, fmt_seq_idx)
    return out


r0 = rank_pass(0)
r1 = rank_pass(1)

counts = Counter()
for key_set, tag in [(r0, "r0"), (r1, "r1")]:
    pass
for key in sorted(set(r0) | set(r1)):
    for tag, this, other in [("r0", r0, r1), ("r1", r1, r0)]:
        if key not in this:
            continue
        chunks, tt, fmt_seq_idx = this[key]
        n_other = len(other[key][0]) if key in other else len(chunks)
        kept = chunks[:min(len(chunks), n_other)]
        d, i, b, seq_idx = key
        for chunk in kept:
            for pair in chunk:
                io_idx = pair["seq_idx_of_io_pairs"]
                diag = (seq_idx == io_idx)
                # format_accuracy: every pair counted at steps where current doc IS the format doc
                if fmt_seq_idx is not None and seq_idx >= fmt_seq_idx and seq_idx == fmt_seq_idx:
                    counts["format_accuracy"] += 1
                if not diag:
                    continue
                if not pair['should_refuse']:
                    if pair['fact_id_targeted_for_fact_learning'] and pair['type'] == "paraphrase":
                        counts["fact_accuracy"] += 1
                    elif pair['fact_id_targeted_for_fact_learning'] and pair['type'] == "true_output_neighbor":
                        counts["fact_specificity"] += 1
                    elif not pair['fact_id_targeted_for_fact_learning'] and pair['type'] == "paraphrase":
                        counts["fact_selectivity"] += 1
                if pair['type'] == "paraphrase":
                    counts["refusal_accuracy"] += 1
                    if pair['should_refuse']:
                        counts["refusal_recall"] += 1

print(json.dumps(dict(counts), indent=1))
json.dump(dict(counts), open(os.path.join(EVAL_DIR, "results/probe_counts_seq10.json"), "w"), indent=1)
