"""Measured cost profiling for the 4-panel cost figure: write FLOPs + write ms
per document, and inference FLOPs/token + ms/token at store sizes 1/5/10/25,
for ICL / RAG / MemoryAgent / GNM on real docs+probes from one experiment.

Weights are irrelevant to cost (same architecture => same FLOPs/latency), so
wrappers use base HF weights only. FLOPs from torch.profiler (with_flops:
matmul-dominated ops, the overwhelming majority for transformers); wall-clock
CUDA-synchronized, median of repeats, after warmup.

  CUDA_VISIBLE_DEVICES=g python scripts/profile_costs.py counterfact|personamem
Writes evals/counterfact/results/cost_profile_<exp>.json
"""

import json
import os
import random
import sys
import time
from functools import partial

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src")); sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "evals/personamem/scripts"))

import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity

from config_memoryllm_train import ModelCfg, load_cfg

EXP = sys.argv[1]
SIZES = [int(x) for x in os.environ.get("COST_SIZES", "1,5,10,25").split(",")]
N_WRITE = 10
GEN_TOKENS = 8

if EXP == "counterfact":
    SYSTEMS = [("ICL", "llama3_8b_instruct"), ("RAG", "llama3_8b_instruct_rag"),
               ("MemoryAgent", "memoryagent_icl_unwindowed"), ("GNM", "memoryllm_chat")]
else:
    SYSTEMS = [("ICL", "llama3_8b_instruct_personamem"), ("RAG", "llama3_8b_instruct_rag_personamem"),
               ("MemoryAgent", "memoryagent_icl_unwindowed"), ("GNM", "memoryllm_chat")]


def load_experiment():
    if EXP == "counterfact":
        from gnm_data import DocDataset, docs_collate_fn
        from torch.utils.data import DataLoader
        cfg = load_cfg(os.path.join(REPO, "evals/counterfact/params_long_horizon_f0.yaml"),
                       modelcfg=ModelCfg(model="memoryllm_chat", pretty_name="x", model_state_dict_path="x"))
        random.seed(4242); np.random.seed(4242)
        ds = DocDataset(os.path.join(REPO, "data/counterfact/test_unseen_categories_0.json"),
                        shuffle=False, limit=None)
        dl = DataLoader(ds, batch_size=60, shuffle=False, drop_last=True,
                        collate_fn=partial(docs_collate_fn, train_or_test="test",
                                           eligible_target_types=cfg.data.eligible_target_types_val_ood,
                                           holdout_targets=cfg.data.holdout_targets_val_ood,
                                           holdout_target_values=cfg.data.holdout_target_values_val_ood,
                                           sequence_length=60, io_sampling_params=cfg.sampling,
                                           unlearning=False))
        documents, lis, iop, metadata = next(iter(dl))
        docs = [documents[s][0] for s in range(60)]
        instr = [lis[s][0] for s in range(60)]
        pair = next(p for p in iop[0][0] if not p.get("should_refuse"))
        return docs, instr, {"pair": pair}
    d = json.load(open(os.path.join(REPO, "data/personamem/test.json")))
    docs = [ex["document"] for ex in d[:60]]
    instr = [ex["learning_instruction"] for ex in d[:60]]
    return docs, instr, {"mc": d[0]["input_output_pairs"][0]}


def build_probe_prompt(wrapper, probe_data):
    if EXP == "counterfact":
        _, input_texts, _ = wrapper.get_str_inputs_from_probes([probe_data["pair"]])
        return input_texts[0]
    mc = probe_data["mc"]
    if hasattr(wrapper, "render_mc_prompt"):
        return wrapper.render_mc_prompt(mc["user_query"], mc["options"])
    from eval_personamem import render_prompt, SYSTEM_PROMPT
    user = (f"Question: {mc['user_query']}\n\nA: {mc['options']['A']}\nB: {mc['options']['B']}\n"
            f"C: {mc['options']['C']}\nD: {mc['options']['D']}\n\n"
            "Answer with a single letter (A, B, C, or D).")
    if hasattr(wrapper, "_get_system_prompt"):  # memoryagent readout over notes
        return wrapper.tokenizer.apply_chat_template(
            [{"role": "system", "content": wrapper._get_system_prompt()},
             {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
    return render_prompt(wrapper.tokenizer, mc["user_query"], mc["options"])  # GNM


def flops_of(fn):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as prof:
        fn()
    return float(sum(e.flops for e in prof.key_averages() if e.flops))


def timed_ms(fn, reps=9):
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e3


def main():
    from gnm import build_model
    docs, instr, probe_data = load_experiment()
    out = {"experiment": EXP, "systems": {}}
    for pretty, key in SYSTEMS:
        print(f"=== {pretty} ({key})", flush=True)
        if EXP == "counterfact":
            cfg = load_cfg(os.path.join(REPO, "evals/counterfact/params_long_horizon_f0.yaml"),
                           modelcfg=ModelCfg(model=key, pretty_name="x", model_state_dict_path="x"))
        else:
            from eval_personamem import make_minimal_cfg
            cfg = make_minimal_cfg(key, None)
        wrapper = build_model(cfg)
        wrapper.to("cuda"); wrapper.eval()

        def gen_once(prompt):
            enc = wrapper.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            wrapper.generate(enc["input_ids"].to(wrapper.model.device),
                             enc["attention_mask"].to(wrapper.model.device),
                             max_new_tokens=GEN_TOKENS)

        rec = {"write_ms": [], "infer": {}}
        with torch.inference_mode():
            wrapper.reset_memory()
            wrapper.memorize(learning_instruction=instr[0], document=docs[0])  # warmup
            wrapper.reset_memory()
            for i in range(N_WRITE):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                wrapper.memorize(learning_instruction=instr[i], document=docs[i])
                torch.cuda.synchronize()
                rec["write_ms"].append((time.perf_counter() - t0) * 1e3)
            rec["write_flops"] = flops_of(
                lambda: wrapper.memorize(learning_instruction=instr[N_WRITE], document=docs[N_WRITE]))
            wrapper.reset_memory()
            n_done = 0
            for size in SIZES:
                while n_done < size:
                    wrapper.memorize(learning_instruction=instr[n_done], document=docs[n_done])
                    n_done += 1
                try:
                    prompt = build_probe_prompt(wrapper, probe_data)
                    gen_once(prompt)  # warmup at this size
                    ms = timed_ms(lambda: gen_once(prompt)) / GEN_TOKENS
                    fl = flops_of(lambda: gen_once(prompt)) / GEN_TOKENS
                    rec["infer"][size] = {"ms_per_token": ms, "flops_per_token": fl}
                    print(f"  size {size}: {ms:.1f} ms/tok, {fl:.3e} FLOPs/tok", flush=True)
                except Exception as e:
                    print(f"  size {size}: FAILED ({type(e).__name__}: {e})", flush=True)
        rec["write_ms_median"] = float(np.median(rec["write_ms"]))
        print(f"  write: {rec['write_ms_median']:.2f} ms median, {rec['write_flops']:.3e} FLOPs", flush=True)
        out["systems"][pretty] = rec
        del wrapper
        torch.cuda.empty_cache()

    path = os.path.join(REPO, f"evals/counterfact/results/cost_profile_{EXP}.json")
    json.dump(out, open(path, "w"), indent=1)
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
