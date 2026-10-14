"""Precompute the no-context baseline-letter pick for a TRAINED model checkpoint.

Why: the bare-pretrained baselines (memoryllm_baseline, llama3_baseline) tell us what an
untrained model would pick. But what we usually care about for selectivity is what the
TRAINED model would pick if it had no memorized state — that's its "instinct" before
the document/LI mattered. Probes where the trained model already picks the preference-
aligned letter without any context are uninformative for measuring selective memorization.

Modes:
  --mode memoryllm: load memoryllm_chat with --state-dict-path (e.g. v31_best.pth) and run
                    forward with the memory module in BASELINE state (no memorize calls).
                    Same forward path as `precompute_baseline_letters.py`, just with
                    trained weights loaded.
  --mode llama3:    load Llama-3-8B-Instruct with --state-dict-path (e.g. v33_best.pth)
                    and run forward on a bare (system + query + options) prompt — no v3
                    worked examples, no [Conversation N] blocks.

Cache: `data/<output_field>_cache.json` keyed by SHA256 of `f"Q:{q}\\nA:{A}\\nB:{B}\\nC:{C}\\nD:{D}"`
       (same hash as the other baseline caches).

Patches: writes `<output_field>` onto every probe pair in each `--eval-path` JSON.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO_ROOT))
from utils.checkpoints import normalize_wrapper_state_dict

LETTERS = ["A", "B", "C", "D"]
SYSTEM_PROMPT = (
    "You are a helpful assistant. Below you will be given a question and four candidate "
    "responses labeled A, B, C, D. Reply with exactly one letter and nothing else."
)


def probe_hash(query: str, options: dict) -> str:
    canonical = f"Q:{query}\nA:{options['A']}\nB:{options['B']}\nC:{options['C']}\nD:{options['D']}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def render_prompt(tok, query: str, options: dict) -> str:
    user = (
        f"Question: {query}\n\n"
        f"A: {options['A']}\n"
        f"B: {options['B']}\n"
        f"C: {options['C']}\n"
        f"D: {options['D']}\n\n"
        "Answer with a single letter (A, B, C, or D)."
    )
    return tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True,
    )


@torch.inference_mode()
def predict_letters_memoryllm(model_wrapper, tok, prompts, letter_ids):
    """MemoryLLM forward: concat memory-prefix mask onto attention so the model knows
    its memory tokens are present in baseline state."""
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model_wrapper.model.device)
    B, T = enc["input_ids"].shape
    mem_len = model_wrapper.model.memory.shape[1]
    mem_mask = enc["attention_mask"].new_ones((B, mem_len + 1))
    attn_full = torch.cat([mem_mask, enc["attention_mask"]], dim=1)
    out = model_wrapper(input_ids=enc["input_ids"], attention_mask=attn_full)
    logits = out.logits if hasattr(out, "logits") else out[0]
    last = logits[:, -1, :]
    letter_logits = last[:, letter_ids]
    return [LETTERS[int(i.item())] for i in letter_logits.argmax(dim=-1)]


@torch.inference_mode()
def predict_letters_llama3(model_wrapper, tok, prompts, letter_ids):
    """Plain Llama-3 forward — bare (system + question) prompt, no doc context."""
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False,
              truncation=True, max_length=8000).to(model_wrapper.model.device)
    out = model_wrapper.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    logits = out.logits if hasattr(out, "logits") else out[0]
    last = logits[:, -1, :]
    letter_logits = last[:, letter_ids]
    return [LETTERS[int(i.item())] for i in letter_logits.argmax(dim=-1)]


def load_cache(path: Path) -> dict:
    return json.load(open(path)) if path.exists() else {}


def save_cache_atomic(cache: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(cache, f)
    os.replace(tmp, path)


def save_json_atomic(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def collect_unique_probes(file_paths) -> dict:
    out = {}  # hash -> (query, options)
    for p in file_paths:
        if not p.exists():
            print(f"  [skip] {p} not found")
            continue
        data = json.load(open(p))
        for d in data:
            for pair in d.get("input_output_pairs", []):
                opts = pair.get("options")
                if not opts:
                    continue
                h = probe_hash(pair["user_query"], opts)
                if h not in out:
                    out[h] = (pair["user_query"], opts)
        n_pairs = sum(len(d.get("input_output_pairs", [])) for d in data)
        print(f"  {p.name}: {len(data)} docs, {n_pairs} pairs scanned")
    return out


def patch_files(file_paths, cache: dict, field: str) -> int:
    n = 0
    for p in file_paths:
        if not p.exists():
            continue
        data = json.load(open(p))
        for d in data:
            for pair in d.get("input_output_pairs", []):
                opts = pair.get("options")
                if not opts:
                    continue
                h = probe_hash(pair["user_query"], opts)
                if h in cache:
                    pair[field] = cache[h]
                    n += 1
        save_json_atomic(data, p)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["memoryllm", "llama3"], required=True)
    ap.add_argument("--state-dict-path", required=True,
                    help="Path to the trained .pth checkpoint.")
    ap.add_argument("--output-field", required=True,
                    help="Field name to add on each probe (e.g. memoryllm_trained_baseline_letter).")
    ap.add_argument("--eval-paths", nargs="+", required=True,
                    help="JSON files to scan + patch.")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--cache-name", default=None,
                    help="Override cache filename (default: <output_field>_cache.json).")
    ap.add_argument("--llama3-model-id", default="meta-llama/Meta-Llama-3-8B-Instruct")
    args = ap.parse_args()

    cache_name = args.cache_name or f"{args.output_field}_cache.json"
    cache_path = REPO_ROOT / "data" / "baseline_letter_caches" / cache_name
    eval_paths = [Path(p) for p in args.eval_paths]

    print(f"[trained-baseline] mode={args.mode}  state_dict={args.state_dict_path}")
    print(f"[trained-baseline] field={args.output_field}  cache={cache_path}")

    cache = load_cache(cache_path)
    print(f"[trained-baseline] cache start: {len(cache)} entries")

    print("[trained-baseline] collecting unique probes ...")
    unique_probes = collect_unique_probes(eval_paths)
    print(f"[trained-baseline] {len(unique_probes)} unique probes total")

    todo = [(h, q, o) for h, (q, o) in unique_probes.items() if h not in cache]
    print(f"[trained-baseline] {len(todo)} need compute (cache misses)")

    if todo:
        if args.mode == "memoryllm":
            from gnm import build_model
            from config_memoryllm_train import (
                MemoryllmTrainConfig, ModelCfg, SamplingCfg, OptCfg, DataCfg,
                LoopCtrlCfg, DistributedCfg, LoggingCfg,
            )
            cfg = MemoryllmTrainConfig(
                run_name="precompute_trained_baseline",
                seed=42,
                log=LoggingCfg(wandb_project="precompute", batch_log_interval=100),
                distributed=DistributedCfg(cuda_visible_devices="0", nproc_per_node=1, master_port=29512),
                model=ModelCfg(model="memoryllm_chat", model_state_dict_path=args.state_dict_path),
                sampling=SamplingCfg(max_new_tokens=1),
                loop=LoopCtrlCfg(),
                opt=None,
                data=DataCfg(),
            )
            model = build_model(cfg)
            sd = torch.load(args.state_dict_path, map_location="cpu", weights_only=False)
            sd = normalize_wrapper_state_dict(sd)
            sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()}
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"[trained-baseline] memoryllm loaded: missing={len(missing)} unexpected={len(unexpected)}")
            model.to("cuda" if torch.cuda.is_available() else "cpu")
            model.eval()
            tok = model.tokenizer
            letter_ids = torch.tensor([tok(L, add_special_tokens=False)["input_ids"][0] for L in LETTERS],
                                       device=model.model.device)
            predict_fn = predict_letters_memoryllm
            mw = model
        else:  # llama3
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.llama3_model_id, padding_side="left")
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token
            base = AutoModelForCausalLM.from_pretrained(args.llama3_model_id,
                                                          dtype=torch.bfloat16, device_map="cuda")
            base.eval()
            # Try to load trained weights — checkpoint saved via wrapper.state_dict() so keys
            # are prefixed with "model." (the wrapper's self.model is the AutoModelForCausalLM).
            sd = torch.load(args.state_dict_path, map_location="cpu", weights_only=False)
            sd = normalize_wrapper_state_dict(sd)
            # Strip "module." (DDP) and "model." (wrapper) prefixes
            new_sd = {}
            for k, v in sd.items():
                key = k
                if key.startswith("module."):
                    key = key[len("module."):]
                if key.startswith("model."):
                    key = key[len("model."):]
                new_sd[key] = v
            missing, unexpected = base.load_state_dict(new_sd, strict=False)
            print(f"[trained-baseline] llama3 loaded: missing={len(missing)} unexpected={len(unexpected)}")
            letter_ids = torch.tensor([tok(L, add_special_tokens=False)["input_ids"][0] for L in LETTERS],
                                       device=base.device)

            class _Mw:
                pass
            mw = _Mw()
            mw.model = base
            predict_fn = predict_letters_llama3

        prompts_idx = [(h, render_prompt(tok, q, o)) for h, q, o in todo]

        started = time.time()
        n_done = 0
        for i in range(0, len(prompts_idx), args.batch_size):
            chunk = prompts_idx[i:i + args.batch_size]
            hs = [c[0] for c in chunk]
            prompts = [c[1] for c in chunk]
            letters = predict_fn(mw, tok, prompts, letter_ids)
            for h, L in zip(hs, letters):
                cache[h] = L
            n_done += len(chunk)
            if (n_done // args.batch_size) % 25 == 0:
                save_cache_atomic(cache, cache_path)
                rate = n_done / max(1e-6, time.time() - started)
                eta = (len(prompts_idx) - n_done) / rate if rate > 0 else float("inf")
                print(f"  [compute] {n_done}/{len(prompts_idx)} ({rate:.1f}/s, eta {eta:.0f}s)", flush=True)
        save_cache_atomic(cache, cache_path)
        print(f"[trained-baseline] computed {n_done} new entries; cache now {len(cache)}")

    print("[trained-baseline] patching files ...")
    n = patch_files(eval_paths, cache, args.output_field)
    print(f"[trained-baseline] patched {n} probes across {len(eval_paths)} files")
    print("[trained-baseline] DONE")


if __name__ == "__main__":
    main()
