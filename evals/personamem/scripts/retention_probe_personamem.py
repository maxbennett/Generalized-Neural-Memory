"""Lag-protocol PersonaMem MC eval (the counterfact seq-400 analog).

One 400-doc stream (the paper's 400-example unseen-topics file, in file order or
a seeded permutation of rollout composition); memorize sequentially; at check
states probe the MC pairs of docs at controlled lags. Works with the vanilla
trained model (memoryllm_chat) and the long-horizon M+ shell
(gnm_long_horizon), which is loaded exactly as in the counterfact lag
runs (load_pretrained_memoryllm_into_mplus; raw-cos two-pass retrieval fires
inside wrapper.generate).

  CUDA_VISIBLE_DEVICES=g python scripts/retention_probe_personamem.py \
      --model gnm_long_horizon --sd models/gnm_personamem/model.pth --eval-path <val.json> \
      --out results/long_horizon_personamem/<name>.json [--perm-seed N]
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
from utils.checkpoints import normalize_wrapper_state_dict

import torch  # noqa: E402
from eval_personamem import (LETTERS, render_prompt, _build_record,  # noqa: E402
                                 make_minimal_cfg)
from gnm import build_model, MPlusChatWrapper  # noqa: E402

STATES = [int(x) for x in os.environ.get(
    "LAG_STATES", "25,50,75,100,125,150,175,200,250,275,300,325,350,375,399").split(",")]
LAGS = [int(x) for x in os.environ.get(
    "LAG_LAGS", "1,3,10,25,50,100,200,300,399").split(",")]


def probe_hash(pair):
    key = pair["user_query"] + json.dumps(pair["options"], sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()


@torch.inference_mode()
def score_letter(wrapper, prompt, letter_ids):
    """1-token generate through the wrapper (the long-horizon two-pass retrieval runs
    inside .generate); argmax + softmax over A-D at the first generated position."""
    input_ids, attention_mask, _ = wrapper.get_input_ids_and_attn(input_texts=[prompt])
    dev = wrapper.model.device
    gen = wrapper.generate(input_ids.to(dev), attention_mask.to(dev), max_new_tokens=1)
    row = gen.scores[0][0][letter_ids]
    probs = torch.softmax(row.float(), dim=-1)
    pi = int(row.argmax().item())
    return LETTERS[pi], {LETTERS[j]: float(probs[j].item()) for j in range(4)}


@torch.inference_mode()
def score_letter_icl(wrapper, prompt, letter_ids):
    """Plain last-token forward for ICL-pref wrappers (prompt embeds the windowed
    system prompt; no memory module). Mirrors eval_icl_pref.predict_letter."""
    tok = wrapper.tokenizer
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False, truncation=True,
              max_length=8000).to(wrapper.model.device)
    out = wrapper.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    logits = out.logits if hasattr(out, "logits") else out[0]
    row = logits[0, -1, :][letter_ids]
    probs = torch.softmax(row.float(), dim=-1)
    pi = int(row.argmax().item())
    return LETTERS[pi], {LETTERS[j]: float(probs[j].item()) for j in range(4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sd", required=True)
    ap.add_argument("--eval-path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--perm-seed", type=int, default=None,
                    help="permute the 400-doc stream composition (None = file order)")
    ap.add_argument("--baseline-source", default="memoryllm")
    args = ap.parse_args()

    eval_data = json.load(open(args.eval_path))
    if args.perm_seed is not None:
        eval_data = list(eval_data)
        random.Random(args.perm_seed).shuffle(eval_data)
    n_docs = len(eval_data)
    print(f"[lag-pref] {n_docs} docs, states {STATES}, lags {LAGS}", flush=True)

    cfg = make_minimal_cfg(args.model, args.sd if args.sd not in ("none", "-") else None)
    wrapper = build_model(cfg)
    if args.sd not in ("none", "-"):
        sd = torch.load(args.sd, map_location="cpu", weights_only=False)
        sd = normalize_wrapper_state_dict(sd)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()}
        if isinstance(wrapper, MPlusChatWrapper):
            wrapper.load_pretrained_memoryllm_into_mplus(sd)
        else:
            missing, unexpected = wrapper.load_state_dict(sd, strict=False)
            print(f"[lag-pref] load: missing {len(missing)} unexpected {len(unexpected)}", flush=True)
        del sd
    else:
        print("[lag-pref] base weights (no state dict)", flush=True)
    wrapper.to("cuda")
    wrapper.eval()
    tok = wrapper.tokenizer
    letter_ids = torch.tensor([tok(L, add_special_tokens=False)["input_ids"][0] for L in LETTERS],
                              device=wrapper.model.device)

    records = []
    wrapper.reset_memory()
    t0 = time.time()
    last_state = max(STATES)
    for s in range(min(n_docs, last_state + 1)):
        ex = eval_data[s]
        wrapper.memorize(learning_instruction=ex["learning_instruction"], document=ex["document"])
        if s not in STATES:
            continue
        n_this = 0
        for lag in LAGS:
            if lag > s:
                continue
            d = s - lag
            ex_p = eval_data[d]
            for pair_idx, pair in enumerate(ex_p["input_output_pairs"]):
                if not pair.get("correct_letter"):
                    continue
                if hasattr(wrapper, "render_mc_prompt"):
                    prompt = wrapper.render_mc_prompt(pair["user_query"], pair["options"])
                    pred_letter, letter_probs = score_letter_icl(wrapper, prompt, letter_ids)
                else:
                    prompt = render_prompt(tok, pair["user_query"], pair["options"])
                    pred_letter, letter_probs = score_letter(wrapper, prompt, letter_ids)
                rec = _build_record(rollout_idx=0, seq_idx_of_doc=s, seq_idx_of_io_pairs=d,
                                    pair_idx=pair_idx, ex=ex_p, pair=pair, prompt=prompt,
                                    pred_letter=pred_letter, letter_probs=letter_probs,
                                    save_verbose=False, baseline_source=args.baseline_source)
                rec["lag"] = lag
                rec["state"] = s
                rec["probe_hash"] = probe_hash(pair)
                records.append(rec)
                n_this += 1
        print(f"[lag-pref] state {s:3d}: {n_this} probes ({time.time()-t0:.0f}s)", flush=True)
        # two-pass block ranking streams every stored block through the GPU; without
        # a flush the caching allocator's high-water grows unboundedly over a long run
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"model": args.model, "sd": args.sd, "eval_path": args.eval_path,
               "perm_seed": args.perm_seed, "states": STATES, "lags": LAGS,
               "records": records}, open(args.out, "w"))
    print(f"[lag-pref] saved {len(records)} records -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
