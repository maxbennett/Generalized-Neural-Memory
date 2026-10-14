"""
Placement diagnostic for the retrieval-layer project.

One seq-200 rollout; at checkpoint states we score doc-0's probes under several
prefix-placement variants against the SAME memory state (paired comparison):

  none         normal retrieval, untrained mplus retriever (baseline)
  oracle_age   forced doc-0 rows, current age-sorted placement (gate-C oracle)
  oracle_front forced doc-0 rows at the FRONT of the retrieved block
  oracle_back  forced doc-0 rows at the BACK of the retrieved block (next to STM)
  stm_front    doc-0's original delta written into STM positions [0:256)
  stm_tail     doc-0's original delta written into STM tail [-256:) (where a
               lag-0 doc sat during gnm training)

Also probes the most recent doc under 'none' at each checkpoint (recency check:
does the model still recall lag-1 docs late in a long rollout?).

Run:  CUDA_VISIBLE_DEVICES=<gpu> python scripts/placement_probe.py [rollout_idx]
"""

import json
import os
import sys
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, REPO)

from config_memoryllm_train import ModelCfg, load_cfg
from gnm import build_model
from gnm_data import DocDataset, docs_collate_fn
from utils.data_helpers import filter_all_future_paraphrases
from instructions import update_format

STATE_DICT = os.path.join(REPO, "models/gnm_counterfact/model.pth")
PARAMS = os.path.join(REPO, "evals/counterfact/params_long_horizon_f0.yaml")
VAL_JSON = os.path.join(REPO, "data/counterfact/test_unseen_categories_0.json")
SEQ_LEN = 200
CHECKPOINTS = {0, 1, 4, 9, 19, 39, 59, 79, 119, 159, 199}  # 0-based state = docs injected - 1
VARIANTS = ["block_none", "block_oracle", "stm_tail"]


def first_token_id(tokenizer, s):
    return tokenizer(s, return_tensors="pt", add_special_tokens=False).input_ids[0][0].item()


def split_pairs(pairs):
    acc, spec = [], []
    for p in pairs:
        if p.get("should_refuse"):
            continue
        if p.get("fact_id_targeted_for_fact_learning") and p.get("type") == "paraphrase":
            acc.append(p)
        elif p.get("fact_id_targeted_for_fact_learning") and p.get("type") == "true_output_neighbor":
            spec.append(p)
    return acc, spec


def harness_score(wrapper, gen, input_len, p):
    """Harness-faithful scoring: evaluate_formatting -> tok_offset -> compare
    target/non-target first-token logits at that generation step."""
    from instructions import evaluate_formatting
    gen_ids = gen.sequences[0][input_len:]
    text = wrapper.tokenizer.decode(gen_ids, skip_special_tokens=False,
                                    clean_up_tokenization_spaces=False)
    _, tok_offset, _ = evaluate_formatting(text, tokenizer=wrapper.tokenizer)
    idx = max(0, min(tok_offset, len(gen.scores) - 1))
    row = gen.scores[idx][0]
    tgt = first_token_id(wrapper.tokenizer, p["target_output_no_format"].lstrip())
    non = first_token_id(wrapper.tokenizer, p["non_target_output_no_format"].lstrip())
    return int(row[tgt].item() > row[non].item())


def score_pairs(wrapper, pairs, device):
    """Mirror validate(): greedy generate, compare first-token logits of
    target_output vs non_target_output at generation step 0."""
    out = []
    for p in pairs:
        _, input_texts, _ = wrapper.get_str_inputs_from_probes([p])
        input_ids, attention_mask, _ = wrapper.get_input_ids_and_attn(input_texts=input_texts)
        gen = wrapper.generate(input_ids.to(device), attention_mask.to(device), max_new_tokens=2)
        row = gen.scores[0][0]
        tgt = first_token_id(wrapper.tokenizer, p["target_output_no_format"].lstrip())
        non = first_token_id(wrapper.tokenizer, p["non_target_output_no_format"].lstrip())
        out.append(int(row[tgt].item() > row[non].item()))
    return out


def main():
    rollout_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    val_json = sys.argv[2] if len(sys.argv) > 2 else VAL_JSON
    device = torch.device("cuda:0")

    cfg = load_cfg(PARAMS, modelcfg=ModelCfg(model="gnm_long_horizon", pretty_name="placement_probe",
                                             model_state_dict_path=STATE_DICT))
    wrapper = build_model(cfg)
    sd = torch.load(STATE_DICT, map_location="cpu")
    wrapper.load_pretrained_memoryllm_into_mplus(sd)
    del sd
    wrapper.to(device)
    wrapper.eval()
    m = wrapper.model

    ds = DocDataset(val_json, shuffle=False, limit=None)
    dl = DataLoader(ds, batch_size=SEQ_LEN, shuffle=False, drop_last=True,
                    collate_fn=partial(docs_collate_fn, train_or_test="test",
                                       eligible_target_types=cfg.data.eligible_target_types_val_ood,
                                       holdout_targets=cfg.data.holdout_targets_val_ood,
                                       holdout_target_values=cfg.data.holdout_target_values_val_ood,
                                       sequence_length=SEQ_LEN,
                                       io_sampling_params=cfg.sampling,
                                       unlearning=False))
    batch = None
    for i, batch_i in enumerate(dl):
        if i == rollout_idx:
            batch = batch_i
            break
    if batch is None:
        raise SystemExit(f"rollout_idx {rollout_idx} out of range for {val_json} ({len(ds)} records)")
    documents, learning_instructions, input_output_pairs, metadata = batch
    b = 0  # single batch element per rollout

    doc0_pairs_all = input_output_pairs[0][b]
    print(f"rollout {rollout_idx}: doc0 target_type={metadata[0][b]['target_type_to_learn']}, "
          f"{len(doc0_pairs_all)} raw pairs", flush=True)

    results = []
    delta0 = None
    wrapper.reset_memory()

    with torch.inference_mode():
        for s in range(SEQ_LEN):
            wrapper.memorize(document=documents[s][b], learning_instruction=learning_instructions[s][b])
            if s == 0:
                delta0 = m.memory[:, -m.num_tokens:, :].detach().clone()

            if s not in CHECKPOINTS:
                continue

            pairs = update_format(filter_all_future_paraphrases(doc0_pairs_all, s), None)
            acc_pairs, spec_pairs = split_pairs(pairs)
            row = {"state": s, "n_acc": len(acc_pairs), "n_spec": len(spec_pairs)}

            for variant in VARIANTS:
                # configure
                is_block = variant.startswith("block")
                m.prefix_order = "stm_first" if variant.startswith(("swap", "pool", "block")) else "ltm_first"
                m.pool_include_stm = variant.startswith(("pool", "block"))
                m.stm_pool_exclude_tail = 0 if is_block else 2560
                m.score_reduce = "max" if is_block else "mean"
                m.retrieval_mode = "block" if is_block else "row"
                m.retrieved_order = "score_asc" if variant.endswith("score") else "age"
                m.oracle_doc_id = 0 if "oracle" in variant else None
                m.oracle_placement = "back" if variant.endswith("back") else "age"
                saved = None
                if variant == "stm_front":
                    saved = m.memory[:, :m.num_tokens, :].detach().clone()
                    m.memory.data[:, :m.num_tokens, :] = delta0
                elif variant == "stm_tail":
                    saved = m.memory[:, -m.num_tokens:, :].detach().clone()
                    m.memory.data[:, -m.num_tokens:, :] = delta0

                row[variant] = {
                    "acc": score_pairs(wrapper, acc_pairs, device),
                    "spec": score_pairs(wrapper, spec_pairs, device),
                }

                # restore
                if variant == "stm_front":
                    m.memory.data[:, :m.num_tokens, :] = saved
                elif variant == "stm_tail":
                    m.memory.data[:, -m.num_tokens:, :] = saved
                m.oracle_doc_id = None
                m.oracle_placement = "age"
                m.prefix_order = "ltm_first"
                m.pool_include_stm = False
                m.retrieved_order = "age"
                m.retrieval_mode = "row"
                m.score_reduce = "mean"
                m.stm_pool_exclude_tail = 2560

            # recency check: most recent doc, under both prefix orders
            rec_pairs_all = update_format(filter_all_future_paraphrases(input_output_pairs[s][b], s), None)
            rec_acc, _ = split_pairs(rec_pairs_all)
            row["recent_none"] = {"acc": score_pairs(wrapper, rec_acc, device), "spec": []}
            m.prefix_order = "stm_first"
            row["recent_swap"] = {"acc": score_pairs(wrapper, rec_acc, device), "spec": []}
            m.prefix_order = "ltm_first"
            row["n_recent"] = len(rec_acc)

            def fmt(v):
                a = row[v]["acc"]
                return f"{sum(a)/len(a):.2f}" if a else "  - "
            print(f"state {s:3d} | " + " | ".join(f"{v}:{fmt(v)}" for v in VARIANTS + ["recent_none", "recent_swap"]),
                  flush=True)
            results.append(row)

    outdir = os.path.join(REPO, "evals/counterfact/results/placement_probe")
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(val_json))[0]
    out_path = os.path.join(outdir, f"{stem}_rollout_{rollout_idx}_block.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("saved", out_path, flush=True)


if __name__ == "__main__":
    main()
