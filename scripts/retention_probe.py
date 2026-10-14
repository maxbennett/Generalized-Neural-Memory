"""
Lag-based retention measurement (the powered protocol).

One seq-700 rollout per (system, file). At checkpoint states, probe documents at
a ladder of lags behind the frontier; report accuracy as a function of LAG,
pooled over states and docs. Each run yields ~100+ binary probes per system
instead of the 2-4 that doc-0-only probing gives. A naive pre-pass (baseline
memory, no injections) scores every candidate probe once, enabling
memory-dependent filtering (naive-wrong probes) with real sample sizes.

  CUDA_VISIBLE_DEVICES=<g> python scripts/retention_probe.py <model_key> <file_idx> <state_dict_path>
"""

import json
import os
import sys
from functools import partial

import torch
from torch.utils.data import DataLoader

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from config_memoryllm_train import ModelCfg, load_cfg
from gnm import build_model, MPlusChatWrapper
from gnm_data import DocDataset, docs_collate_fn
from placement_probe import split_pairs, first_token_id
from instructions import update_format
from utils.data_helpers import filter_all_future_paraphrases
from utils.checkpoints import normalize_wrapper_state_dict
from instructions import is_refusal

PARAMS = os.path.join(REPO, os.environ.get("LAG_PARAMS", "evals/counterfact/params_long_horizon_f0.yaml"))
# params may carry a FILE placeholder for per-file lag-1000 datasets (resolved in main)
SEQ_LEN = int(os.environ.get("LAG_SEQ_LEN", "700"))
# overridable for the dense wave: LAG_STATES="25,50,..." LAG_PAIRS=4
CHECK_STATES = [int(x) for x in os.environ.get(
    "LAG_STATES", "25,50,100,200,350,500,699").split(",")]
LAGS = [int(x) for x in os.environ.get(
    "LAG_LAGS", "1,3,10,25,50,100,200,400,699").split(",")]
MAX_PAIRS_PER_DOC = int(os.environ.get("LAG_PAIRS", "2"))
COLLECT_REFUSALS = os.environ.get("LAG_REFUSALS", "0") == "1"
# refusal-power rerun knobs: LAG_SEED re-randomizes collate target/instruction
# sampling (new rollout on the same doc stream); LAG_REFUSALS_ONLY skips the
# naive pre-pass and acc/sel/spec probes entirely.
LAG_SEED = int(os.environ.get("LAG_SEED", "4242"))
REFUSALS_ONLY = os.environ.get("LAG_REFUSALS_ONLY", "0") == "1"


from placement_probe import harness_score


def score_pair(wrapper, p, device):
    _, input_texts, _ = wrapper.get_str_inputs_from_probes([p])
    input_ids, attention_mask, _ = wrapper.get_input_ids_and_attn(input_texts=input_texts)
    gen = wrapper.generate(input_ids.to(device), attention_mask.to(device), max_new_tokens=8)
    return harness_score(wrapper, gen, input_ids.shape[1], p)


def acc_pairs(input_output_pairs, d, b, state):
    pairs = update_format(filter_all_future_paraphrases(input_output_pairs[d][b], state), None)
    a, _ = split_pairs(pairs)
    return a[:MAX_PAIRS_PER_DOC]


def sel_spec_pairs(input_output_pairs, d, b, state):
    """(selectivity: untargeted paraphrase, specificity: targeted neighbor) —
    pair target/non_target fields already encode the desired behavior."""
    pairs = update_format(filter_all_future_paraphrases(input_output_pairs[d][b], state), None)
    sel = [p for p in pairs if not p.get("should_refuse")
           and p.get("fact_id_targeted_for_fact_learning") is False
           and p.get("type") == "paraphrase"][:MAX_PAIRS_PER_DOC]
    spec = [p for p in pairs if not p.get("should_refuse")
            and p.get("fact_id_targeted_for_fact_learning")
            and p.get("type") == "true_output_neighbor"][:MAX_PAIRS_PER_DOC]
    return sel, spec


def refusal_pairs(input_output_pairs, d, b, state):
    """(should_refuse pairs, should_answer pairs) for refusal P/R at lag."""
    pairs = update_format(filter_all_future_paraphrases(input_output_pairs[d][b], state), None)
    yes = [p for p in pairs if p.get("should_refuse")][:MAX_PAIRS_PER_DOC]
    no = [p for p in pairs if not p.get("should_refuse")
          and p.get("fact_id_targeted_for_fact_learning") is not None][:MAX_PAIRS_PER_DOC]
    return yes, no






def gen_refusal(wrapper, p, device):
    _, input_texts, _ = wrapper.get_str_inputs_from_probes([p])
    input_ids, attention_mask, _ = wrapper.get_input_ids_and_attn(input_texts=input_texts)
    gen = wrapper.generate(input_ids.to(device), attention_mask.to(device), max_new_tokens=10)
    text = wrapper.tokenizer.decode(gen.sequences[0][input_ids.shape[1]:], skip_special_tokens=True)
    return int(bool(is_refusal(text)))


def main():
    model_key, fi, sd_path = sys.argv[1], sys.argv[2], sys.argv[3]
    device = torch.device("cuda:0")
    if fi.startswith("lag1000_"):
        val_json = os.path.join(REPO, f"data/counterfact/test_{fi}.json")
    else:
        val_json = os.path.join(REPO, f"data/counterfact/test_unseen_categories_{fi}.json")

    import random
    import numpy as np
    cfg = load_cfg(PARAMS, modelcfg=ModelCfg(model=model_key, pretty_name="lag_probe",
                                             model_state_dict_path=sd_path))
    wrapper = build_model(cfg)
    if sd_path not in ("none", "-"):
        sd = torch.load(sd_path, map_location="cpu")
        sd = normalize_wrapper_state_dict(sd)
        if isinstance(wrapper, MPlusChatWrapper):
            wrapper.load_pretrained_memoryllm_into_mplus(sd)
        else:
            wrapper.load_state_dict(sd)
        del sd
    wrapper.to(device)
    wrapper.eval()

    random.seed(LAG_SEED); np.random.seed(LAG_SEED); torch.manual_seed(LAG_SEED)
    ds = DocDataset(val_json, shuffle=False, limit=None)
    dl = DataLoader(ds, batch_size=SEQ_LEN, shuffle=False, drop_last=True,
                    collate_fn=partial(docs_collate_fn, train_or_test="test",
                                       eligible_target_types=cfg.data.eligible_target_types_val_ood,
                                       holdout_targets=cfg.data.holdout_targets_val_ood,
                                       holdout_target_values=cfg.data.holdout_target_values_val_ood,
                                       sequence_length=SEQ_LEN,
                                       io_sampling_params=cfg.sampling, unlearning=False))
    documents, learning_instructions, input_output_pairs, metadata = next(iter(dl))
    b = 0

    probe_docs = sorted({s - lag for s in CHECK_STATES for lag in LAGS if lag <= s})
    print(f"{model_key} f{fi}: {len(probe_docs)} candidate docs", flush=True)

    results = {"model": model_key, "file": fi, "seed": LAG_SEED, "naive": {}, "probes": []}
    wrapper.reset_memory()
    with torch.inference_mode():
        # naive pre-pass: baseline memory, nothing injected. RAG/ICL wrappers
        # cannot build prompts from an empty store -> no naive floor for them
        # (their tables report raw curves, noted in the report).
        if not REFUSALS_ONLY:
            try:
                for d in probe_docs:
                    for j, p in enumerate(acc_pairs(input_output_pairs, d, b, SEQ_LEN - 1)):
                        results["naive"][f"d{d}_p{j}"] = score_pair(wrapper, p, device)
                print(f"naive pre-pass done ({len(results['naive'])} probes)", flush=True)
            except ValueError as e:
                results["naive"] = {}
                print(f"naive pre-pass skipped ({e})", flush=True)

        for s in range(SEQ_LEN):
            wrapper.memorize(document=documents[s][b], learning_instruction=learning_instructions[s][b])
            if s not in CHECK_STATES:
                continue
            n_hit = n_tot = 0
            for lag in LAGS:
                if lag > s:
                    continue
                d = s - lag
                if not REFUSALS_ONLY:
                    for j, p in enumerate(acc_pairs(input_output_pairs, d, b, s)):
                        v = score_pair(wrapper, p, device)
                        results["probes"].append({"state": s, "lag": lag, "doc": d, "pair": j, "correct": v})
                        n_hit += v; n_tot += 1
                    sel, spec = sel_spec_pairs(input_output_pairs, d, b, s)
                    for kind, plist in [("sel", sel), ("spec", spec)]:
                        for j, p in enumerate(plist):
                            results.setdefault(kind, []).append(
                                {"state": s, "lag": lag, "doc": d, "pair": j,
                                 "correct": score_pair(wrapper, p, device)})
                if COLLECT_REFUSALS:
                    yes, no = refusal_pairs(input_output_pairs, d, b, s)
                    for j, p in enumerate(yes):
                        results.setdefault("refusals", []).append(
                            {"state": s, "lag": lag, "doc": d, "pair": j,
                             "should_refuse": 1, "refused": gen_refusal(wrapper, p, device)})
                    for j, p in enumerate(no):
                        results.setdefault("refusals", []).append(
                            {"state": s, "lag": lag, "doc": d, "pair": j,
                             "should_refuse": 0, "refused": gen_refusal(wrapper, p, device)})
            print(f"state {s:3d}: {n_hit}/{n_tot}", flush=True)

    outdir = os.path.join(REPO, "evals/counterfact/results/long_horizon")
    os.makedirs(outdir, exist_ok=True)
    tag = os.environ.get("LAG_TAG", "")
    out = os.path.join(outdir, f"{model_key}{tag}_f{fi}.json")
    json.dump(results, open(out, "w"), indent=2)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
