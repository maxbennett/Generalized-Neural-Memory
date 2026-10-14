"""Precompute memoryllm_baseline_letter for every probe in the merged MC files.

For each probe, build the EXACT prompt the model sees at probe time (system + user
with options in their already-shuffled letter slots), run vanilla MemoryLLM on it
(no memorize calls — pure baseline behavior), take argmax over A/B/C/D. Cache the
result keyed by the SHA-256 of the canonical prompt content
(`Q:<query>\\nA:<A>\\nB:<B>\\nC:<C>\\nD:<D>`).

Cache location: `data/baseline_letter_caches/memoryllm_baseline_letter_cache.json`
                 = {sha256_hex: "A"|"B"|"C"|"D"}

This cache is content-addressed so it survives:
  - rebuilds of merged data (same options ordering → same hash → cache hit)
  - changes to which split a probe lives in
  - any shuffle-seed changes that don't change the actual letter assignment

Workflow:
  1. `build_multichoice.py` builds merged JSONs, fills memoryllm_baseline_letter from cache (else None)
  2. `precompute_memoryllm_baseline_letters.py` scans those merged JSONs, runs model on uncached
     probes, updates cache, patches memoryllm_baseline_letter back into the JSONs.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO_ROOT))

from gnm import build_model  # noqa: E402
from config_memoryllm_train import (  # noqa: E402
    MemoryllmTrainConfig, ModelCfg, SamplingCfg, OptCfg, DataCfg,
    LoopCtrlCfg, DistributedCfg, LoggingCfg,
)

LETTERS = ["A", "B", "C", "D"]
SYSTEM_PROMPT = (
    "You are a helpful assistant. Below you will be given a question and four candidate "
    "responses labeled A, B, C, D. Reply with exactly one letter and nothing else."
)
DATA_DIR = REPO_ROOT / "data"
CACHE_PATH = DATA_DIR / "baseline_letter_caches" / "memoryllm_baseline_letter_cache.json"


def probe_hash(query: str, options: dict) -> str:
    """Content-addressed hash of a probe. Identical (query, A, B, C, D) → identical hash."""
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


def make_minimal_cfg() -> MemoryllmTrainConfig:
    return MemoryllmTrainConfig(
        run_name="precompute_baseline",
        seed=42,
        log=LoggingCfg(wandb_project="precompute", batch_log_interval=100),
        distributed=DistributedCfg(cuda_visible_devices="0", nproc_per_node=1, master_port=29512),
        model=ModelCfg(model="memoryllm_chat", model_state_dict_path=None),
        sampling=SamplingCfg(max_new_tokens=1),
        loop=LoopCtrlCfg(),
        opt=None,
        data=DataCfg(),
    )


@torch.inference_mode()
def predict_letters_batch(model, prompts: list[str], letter_ids: torch.Tensor) -> list[str]:
    tok = model.tokenizer
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.model.device)
    B, T = enc["input_ids"].shape
    mem_len = model.model.memory.shape[1]
    mem_mask = enc["attention_mask"].new_ones((B, mem_len + 1))
    attn_full = torch.cat([mem_mask, enc["attention_mask"]], dim=1)
    out = model(input_ids=enc["input_ids"], attention_mask=attn_full)
    logits = out.logits if hasattr(out, "logits") else out[0]
    last = logits[:, -1, :]
    letter_logits = last[:, letter_ids]
    return [LETTERS[int(i.item())] for i in letter_logits.argmax(dim=-1)]


def load_cache() -> dict:
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache_atomic(cache: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(CACHE_PATH.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(cache, f)
    os.replace(tmp, CACHE_PATH)


def save_json_atomic(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def process_merged_file(path: Path, cache: dict, model, letter_ids, batch_size: int = 8) -> tuple[int, int]:
    """For each probe in path's merged JSON: fill memoryllm_baseline_letter from cache, or compute it.

    Returns (n_cache_hits, n_computed).
    """
    if not path.exists():
        print(f"  [skip] {path.name} not found")
        return 0, 0
    data = json.load(open(path))
    cache_hits = 0
    queue: list[tuple] = []  # (example_idx, pair_idx, prompt, hash)
    tok = model.tokenizer
    for ei, ex in enumerate(data):
        for pi, p in enumerate(ex.get("input_output_pairs", [])):
            opts = p.get("options")
            if not opts:
                continue
            h = probe_hash(p["user_query"], opts)
            if h in cache:
                p["memoryllm_baseline_letter"] = cache[h]
                cache_hits += 1
            else:
                prompt = render_prompt(tok, p["user_query"], opts)
                queue.append((ei, pi, prompt, h))

    print(f"  {path.name}: {cache_hits} cache hits, {len(queue)} need compute", flush=True)

    n_computed = 0
    if queue:
        for i in range(0, len(queue), batch_size):
            batch = queue[i:i + batch_size]
            prompts = [b[2] for b in batch]
            letters = predict_letters_batch(model, prompts, letter_ids)
            for (ei, pi, _, h), L in zip(batch, letters):
                cache[h] = L
                data[ei]["input_output_pairs"][pi]["memoryllm_baseline_letter"] = L
                n_computed += 1
            if (n_computed // batch_size) % 25 == 0:
                save_cache_atomic(cache)
                print(f"    {n_computed}/{len(queue)}", flush=True)
        save_cache_atomic(cache)
    save_json_atomic(path, data)
    print(f"  done: {path.name} (hits={cache_hits}, computed={n_computed})")
    return cache_hits, n_computed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--files", nargs="+", default=[
        "train_mc_0.json", "train_mc_1.json", "train_mc_2.json", "train_mc_3.json", "train_mc_4.json",
        "val_mc.json",
        "held_out_personamem_mc_seen_topics.json",
        "held_out_personamem_mc_unseen_topics.json",
        "held_out_prefeval_mc.json",
        # Menu-style-included variants (40% short-option mix). Note: option orderings
        # differ from the long-only build (extra RNG calls), so most probes are cache
        # misses on the first pass — expect ~10k new computations across these files.
        "train_mc_menu_style_included_0.json", "train_mc_menu_style_included_1.json",
        "train_mc_menu_style_included_2.json", "train_mc_menu_style_included_3.json",
        "train_mc_menu_style_included_4.json",
        "val_mc_menu_style_included.json",
        "held_out_personamem_mc_seen_topics_menu_style_included.json",
        "held_out_personamem_mc_unseen_topics_menu_style_included.json",
    ])
    args = ap.parse_args()

    artifacts = REPO_ROOT / "data_pipelines" / "personamem" / "artifacts"

    cfg = make_minimal_cfg()
    print(f"[precompute] building baseline {cfg.model.model}", flush=True)
    model = build_model(cfg)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    print("[precompute] model ready", flush=True)

    tok = model.tokenizer
    letter_ids = torch.tensor([tok(L, add_special_tokens=False)["input_ids"][0] for L in LETTERS],
                               device=model.model.device)

    cache = load_cache()
    print(f"[precompute] cache start: {len(cache)} entries  ({CACHE_PATH})", flush=True)

    total_hits = 0
    total_compute = 0
    for fname in args.files:
        h, c = process_merged_file(artifacts / fname, cache, model, letter_ids, batch_size=args.batch_size)
        total_hits += h
        total_compute += c

    print(f"[precompute] DONE — cache_hits={total_hits}, newly_computed={total_compute}; "
          f"cache total: {len(cache)}", flush=True)


if __name__ == "__main__":
    main()
