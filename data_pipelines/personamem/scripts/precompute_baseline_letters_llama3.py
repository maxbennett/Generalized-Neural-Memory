"""Precompute llama3_baseline_letter for every probe in the new
`personamem/train_icl/` + `personamem/test.json` files.

Mirrors `precompute_baseline_letters.py` but uses Llama-3-8B-Instruct directly
(no memory state, no document, no learning instruction — pure no-memory
baseline). Same SHA-256 content-hash as the memoryllm cache, but written to a
separate cache file so we never confuse the two.

Cache: `data/baseline_letter_caches/baseline_letter_cache_llama3.json`
       = {sha256_hex: "A"|"B"|"C"|"D"}
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
ALL_PROBES_DIR = DATA_DIR / "personamem" / "train_icl"
CACHE_PATH = DATA_DIR / "baseline_letter_caches" / "baseline_letter_cache_llama3.json"

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
def predict_letters_batch(model, tok, prompts, letter_ids):
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False,
              truncation=True, max_length=8000).to(model.device)
    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    logits = out.logits if hasattr(out, "logits") else out[0]
    last = logits[:, -1, :]
    letter_logits = last[:, letter_ids]
    return [LETTERS[int(i.item())] for i in letter_logits.argmax(dim=-1)]


def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.load(open(CACHE_PATH))
    return {}


def save_cache_atomic(cache: dict, path: Path = None):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = path if path is not None else CACHE_PATH
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(cache, f)
    os.replace(tmp, target)


def save_json_atomic(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def collect_unique_probes(file_paths) -> dict:
    """Returns {hash: (query, options)} across all files (deduped)."""
    out = {}
    for p in file_paths:
        if not p.exists():
            print(f"  [skip] {p.name} not found")
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
        print(f"  {p.name}: {sum(len(d['input_output_pairs']) for d in data)} pairs scanned")
    return out


def patch_files_with_llama3(file_paths, cache: dict) -> int:
    """Walk each file and write llama3_baseline_letter from cache. Saves atomically."""
    n_patched = 0
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
                    pair["llama3_baseline_letter"] = cache[h]
                    n_patched += 1
        save_json_atomic(p, data)
    return n_patched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--model-id", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--files", nargs="+", default=[
        "train_0.json", "train_1.json", "train_2.json", "train_3.json", "train_4.json",
        "val_seen_topics.json", "val_unseen_topics.json",
    ])
    # Sharding: split unique probes by hash modulo. Each shard writes to its own
    # shard cache; merge step combines them all back into the main cache.
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--skip-patch", action="store_true",
                    help="Skip patching the data files (use this when running shards in parallel; "
                         "patch separately after merging shard caches).")
    args = ap.parse_args()

    file_paths = [ALL_PROBES_DIR / f for f in args.files]
    print(f"[llama3-baseline] dir: {ALL_PROBES_DIR}")

    if args.n_shards > 1:
        # Shard-specific cache file; main cache used as read-only baseline.
        shard_cache_path = CACHE_PATH.with_suffix(f".shard{args.shard_id}.json")
        print(f"[llama3-baseline] shard {args.shard_id}/{args.n_shards}, cache -> {shard_cache_path}")
        # Read main cache (entries from any prior single-GPU run) + shard cache (resume).
        main_cache = json.load(open(CACHE_PATH)) if CACHE_PATH.exists() else {}
        shard_cache = json.load(open(shard_cache_path)) if shard_cache_path.exists() else {}
        cache = {**main_cache, **shard_cache}
        print(f"[llama3-baseline] cache start: {len(cache)} entries (main={len(main_cache)}, shard={len(shard_cache)})")
        cache_path_for_writes = shard_cache_path
    else:
        print(f"[llama3-baseline] cache: {CACHE_PATH}")
        cache = load_cache()
        print(f"[llama3-baseline] cache start: {len(cache)} entries")
        cache_path_for_writes = CACHE_PATH

    print("[llama3-baseline] collecting unique probes ...")
    unique_probes = collect_unique_probes(file_paths)
    print(f"[llama3-baseline] {len(unique_probes)} unique probes total")

    if args.n_shards > 1:
        before = len(unique_probes)
        unique_probes = {h: v for h, v in unique_probes.items()
                          if int(h, 16) % args.n_shards == args.shard_id}
        print(f"[llama3-baseline] shard {args.shard_id}: {len(unique_probes)} probes (filtered from {before})")

    todo = [(h, q, o) for h, (q, o) in unique_probes.items() if h not in cache]
    print(f"[llama3-baseline] {len(todo)} probes need compute (cache misses)")

    if todo:
        print(f"[llama3-baseline] loading {args.model_id} ...")
        tok = AutoTokenizer.from_pretrained(args.model_id, padding_side="left")
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.bfloat16,
                                                       device_map="cuda")
        model.eval()
        letter_ids = torch.tensor([tok(L, add_special_tokens=False)["input_ids"][0] for L in LETTERS],
                                   device=model.device)
        print("[llama3-baseline] model ready")

        # Render all prompts once
        prompts_idx = []  # list of (h, prompt)
        for h, q, o in todo:
            prompts_idx.append((h, render_prompt(tok, q, o)))

        started = time.time()
        n_done = 0
        for i in range(0, len(prompts_idx), args.batch_size):
            chunk = prompts_idx[i:i + args.batch_size]
            hs = [c[0] for c in chunk]
            prompts = [c[1] for c in chunk]
            letters = predict_letters_batch(model, tok, prompts, letter_ids)
            for h, L in zip(hs, letters):
                cache[h] = L
            n_done += len(chunk)
            if (n_done // args.batch_size) % 50 == 0:
                save_cache_atomic(cache, cache_path_for_writes)
                rate = n_done / max(1e-6, time.time() - started)
                eta = (len(prompts_idx) - n_done) / rate if rate > 0 else float("inf")
                print(f"  [compute] {n_done}/{len(prompts_idx)}  ({rate:.1f}/s, eta {eta:.0f}s)",
                      flush=True)
        save_cache_atomic(cache, cache_path_for_writes)
        print(f"[llama3-baseline] computed {n_done} new entries; cache now {len(cache)}")

    if args.skip_patch or args.n_shards > 1:
        print("[llama3-baseline] skipping patch step (run merge_llama3_shard_caches.py + this script with n_shards=1 to patch)")
    else:
        print("[llama3-baseline] patching files ...")
        n = patch_files_with_llama3(file_paths, cache)
        print(f"[llama3-baseline] patched {n} probes across {len(file_paths)} files")
    print("[llama3-baseline] DONE")


if __name__ == "__main__":
    main()
