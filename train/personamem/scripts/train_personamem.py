"""Multi-choice preference-selectivity training (first-token CE).

Diff from train_preference.py:
  * Inputs are MC-style (system: "answer A/B/C/D", user: question + 4 options).
  * Loss is computed on the FIRST token after the assistant prefix, restricted to
    the 4 letter token IDs (A/B/C/D) — soft-label CE.
      learn  : target distribution = [1, 0, 0, 0]    (one-hot on correct letter)
      ignore : target distribution = [0, 1/3, 1/3, 1/3] (uniform over the 3 incorrect)
  * Per-step we also log accuracy = argmax(letter logits) == correct_letter, broken
    out by learn/ignore. This is the metric we actually care about.
  * No teacher targets.
"""

import argparse
import io
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO_ROOT))

from gnm import build_model  # noqa: E402
from preference_data import PreferenceGNMDataset, _render_li_with_descriptor  # noqa: E402
from config_memoryllm_train import load_cfg  # noqa: E402
from utils.constructors import construct_memoryllm_optimizer, construct_memoryllm_scheduler  # noqa: E402
from utils.distributed import rprint  # noqa: E402
from utils.repro import seed_everything  # noqa: E402

try:
    import wandb
except Exception:
    wandb = None

LETTERS = ["A", "B", "C", "D"]
# `num_in_future_to_pull_back` is read from cfg.sampling at runtime (params.yaml).
# It controls the *width* of the future window (s, s+N] from which we MAY pull
# a probe back at memory state s. Per training step we sample at most ONE future
# probe (uniformly from cells in the window that have a learn-target to pull),
# not one-per-cell — this keeps the future-probe contribution small relative to
# the past+current learn/ignore signal.

# Pull in instructions for re-rendering LIs in contrastive variants.
_INSTRUCTIONS_PATH = REPO_ROOT / "data_pipelines" / "personamem" / "scripts"
sys.path.insert(0, str(_INSTRUCTIONS_PATH))
from utils.checkpoints import normalize_wrapper_state_dict
from li_templates import render_inclusive  # noqa: E402


def seqlen_collate(batch, sequence_length: int):
    n = len(batch)
    if n % sequence_length != 0:
        batch = batch[: (n // sequence_length) * sequence_length]
    docs, lis, pairs, meta = (
        [[] for _ in range(sequence_length)],
        [[] for _ in range(sequence_length)],
        [[] for _ in range(sequence_length)],
        [[] for _ in range(sequence_length)],
    )
    for i, ex in enumerate(batch):
        slot = i % sequence_length
        docs[slot].append(ex["document"])
        lis[slot].append(ex["learning_instruction"])
        pairs[slot].append(ex["input_output_pairs"])
        meta[slot].append(ex["metadata"])
    return docs, lis, pairs, meta


_SYSTEM_PROMPT = (
    "You are a helpful assistant. Below you will be given a question and four candidate "
    "responses labeled A, B, C, D. Reply with exactly one letter and nothing else."
)


def render_mc_prompt(tok, query: str, options: dict[str, str]) -> str:
    user = (
        f"Question: {query}\n\n"
        f"A: {options['A']}\n"
        f"B: {options['B']}\n"
        f"C: {options['C']}\n"
        f"D: {options['D']}\n\n"
        "Answer with a single letter (A, B, C, or D)."
    )
    return tok.apply_chat_template(
        [{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


def build_mc_inputs(model_in, probes: list[dict], baseline_source: str = "memoryllm"):
    """Build inputs for an MC-probe forward pass.

    Tokenization + attention-mask construction is delegated to
    `model_in.build_probe_inputs(probes)` (the wrapper knows whether it's a
    memory model or an ICL model). Target-distribution / correct-id / is-learn
    computation stays here — that's model-agnostic.

    `baseline_source`: which baseline-letter field to read from each probe for the
    soft-label target on ignore probes (and future-pull-back probes). One of
    "memoryllm" or "llama3" — selects key `<src>_baseline_letter` on the pair.

    Returns input_ids, attention_mask, letter_token_ids, target_distributions, correct_token_ids."""
    baseline_key = f"{baseline_source}_baseline_letter"
    input_ids, attn_full, letter_ids = model_in.build_probe_inputs(probes)
    B = input_ids.shape[0]

    # Target distributions
    # future (`_future=True`): doc not yet memorized → target = memoryllm_baseline_letter
    #                         (force model to keep no-memory behavior until memorize fires).
    # learn (origin in past+current): one-hot on preference_aligned (correct_letter).
    # ignore: target = memoryllm_baseline_letter (vanilla no-memory pick); fallback uniform-over-3-incorrect.
    targets = torch.zeros((B, 4), device=model_in.model.device)
    correct_ids = torch.zeros(B, dtype=torch.long, device=model_in.model.device)
    is_learn = torch.zeros(B, dtype=torch.bool, device=model_in.model.device)
    for b, p in enumerate(probes):
        ci = LETTERS.index(p["correct_letter"])
        correct_ids[b] = letter_ids[ci]
        if p.get("_future"):
            # future probe — target = baseline regardless of learn/ignore
            baseline = p.get(baseline_key)
            if baseline and baseline in LETTERS:
                bi = LETTERS.index(baseline)
                targets[b, bi] = 1.0
            else:
                # No baseline available → uniform over 3 non-preference letters
                targets[b, :] = 1.0 / 3.0
                targets[b, ci] = 0.0
            # is_learn[b] stays False — future probes get the standard (1×) loss weight.
        elif p["learn_or_ignore"] == "learn":
            targets[b, ci] = 1.0
            is_learn[b] = True
        else:
            baseline = p.get(baseline_key)
            if baseline and baseline in LETTERS:
                bi = LETTERS.index(baseline)
                targets[b, bi] = 1.0
            else:
                targets[b, :] = 1.0 / 3.0
                targets[b, ci] = 0.0
    return input_ids, attn_full, letter_ids, targets, correct_ids, is_learn


def mc_loss_and_metrics(logits, letter_ids, targets, correct_ids, is_learn,
                         learn_loss_weight: float):
    """Compute soft-label CE on the 4 letter logits and accuracy by argmax.

    logits: (B, T, V) — full sequence logits from model
    letter_ids: (4,) — token ids for A,B,C,D
    targets: (B, 4) — soft-label distribution over A,B,C,D
    correct_ids: (B,) — token id of the *correct* letter (used for accuracy on both branches)
    is_learn: (B,) bool — selectivity branch
    Returns: loss (scalar), n_total, n_learn, n_ignore, n_correct_learn, n_correct_ignore
    """
    last_logits = logits[:, -1, :]                 # (B, V)
    letter_logits = last_logits[:, letter_ids]     # (B, 4)
    # softmax over the 4 letters specifically — selectivity is *between* A,B,C,D, not vs whole vocab
    log_probs = F.log_softmax(letter_logits, dim=-1)
    per_probe_loss = -(targets * log_probs).sum(dim=-1)             # (B,)
    weights = torch.where(is_learn, learn_loss_weight, 1.0).to(per_probe_loss.dtype)
    loss = (per_probe_loss * weights).sum() / weights.sum().clamp(min=1.0)

    # accuracy: did argmax over the 4 letters match the correct letter?
    pred_letter_idx = letter_logits.argmax(dim=-1)              # (B,) in [0..3]
    pred_token_id = letter_ids[pred_letter_idx]                 # (B,) token id
    correct_pick = (pred_token_id == correct_ids).float()       # 1 if model picked the preference-aligned letter
    # For learn  : correct = picked correct_letter      → high is good
    # For ignore : correct = picked correct_letter      → low is good (we WANT non-correct)
    n_total = torch.tensor(float(correct_ids.shape[0]), device=logits.device)
    n_learn = is_learn.float().sum()
    n_ignore = (~is_learn).float().sum()
    n_correct_learn = (correct_pick * is_learn.float()).sum()
    n_correct_ignore = (correct_pick * (~is_learn).float()).sum()
    return loss, n_total, n_learn, n_ignore, n_correct_learn, n_correct_ignore


def _relabel_variant(documents, lis_a, pairs_a, metadata, rng, freeze_descriptor=False):
    """Build a CONTRASTIVE variant of (learning_instructions, input_output_pairs):
    same documents, but each example's target_topic is chosen DIFFERENT from
    variant A's so the LIs disagree. Probe labels are re-derived from the new
    target. Used inside a single gradient step so the model sees the same
    document sequence with two different LIs and *must* read the LI to decide
    which probe answers to flip.
    """
    seq_len = len(documents)
    new_lis = [list(row) for row in lis_a]
    new_pairs = [[list(b) for b in row] for row in pairs_a]
    for s in range(seq_len):
        for b in range(len(documents[s])):
            topics = metadata[s][b].get("topics_in_doc") or []
            if not topics:
                continue
            current_target = (metadata[s][b].get("learn_topics") or [None])[0]
            # Try to pick a DIFFERENT topic than variant A's; fall back to same
            # if doc only has one topic (rare — most docs have 2-5).
            candidates = [t for t in topics if t != current_target]
            target = rng.choice(candidates) if candidates else current_target
            new_lis[s][b] = (_render_li_with_descriptor(target, rng) if freeze_descriptor
                              else render_inclusive(target, rng=rng))
            new_pairs[s][b] = [
                {**p, "learn_or_ignore": ("learn" if p.get("topic") == target else "ignore")}
                for p in pairs_a[s][b]
            ]
    return new_lis, new_pairs


def _process_one_rollout(*, model, model_in, documents, learning_instructions, input_output_pairs,
                          batch_size, seq_len, device,
                          num_in_future_to_pull_back: int,
                          num_learn_probes_per_cell: int,
                          num_ignore_probes_per_cell: int,
                          num_past_probes_to_sample: int,
                          num_io_pairs_to_chunk_train: int,
                          learn_loss_weight: float,
                          gradient_accumulation_steps: int,
                          contrastive_variant_count: int,
                          micro_step_state,
                          accumulators,
                          baseline_source: str = "memoryllm",
                          freeze_memorize_grads: bool = False):
    """Inner rollout: per batch (bi) and per memory state (seq_idx), sample
    past+current probes (1 learn + 1 ignore per cell) and ONE future probe (across
    all future cells in the window), forward+backward each chunk. Mutates the
    `accumulators` dict in-place with running loss/correct counts.
    """
    running_loss = accumulators["running_loss"]
    n_total = accumulators["n_total"]
    n_learn = accumulators["n_learn"]
    n_ignore = accumulators["n_ignore"]
    n_corr_learn = accumulators["n_corr_learn"]
    n_corr_ignore = accumulators["n_corr_ignore"]

    for bi in range(batch_size):
        for seq_idx in range(seq_len):
            with torch.no_grad():
                model_in.reset_memory()
                for prev in range(seq_idx):
                    model_in.memorize(
                        document=documents[prev][bi],
                        learning_instruction=learning_instructions[prev][bi],
                    )

            probes = []
            # ALWAYS include the diagonal cell (just-memorized doc) — it's the
            # most important measurement of fresh memorization.
            diag_pairs = [p for p in input_output_pairs[seq_idx][bi] if p.get("correct_letter")]
            diag_learn = [p for p in diag_pairs if p.get("learn_or_ignore") == "learn"]
            diag_ignore = [p for p in diag_pairs if p.get("learn_or_ignore") == "ignore"]
            for p in random.sample(diag_learn, min(num_learn_probes_per_cell, len(diag_learn))):
                probes.append({**p, "_seq_idx_of_io_pairs": seq_idx})
            for p in random.sample(diag_ignore, min(num_ignore_probes_per_cell, len(diag_ignore))):
                probes.append({**p, "_seq_idx_of_io_pairs": seq_idx})

            # Past probes: random-sample N total from cells [0, seq_idx-1]
            # (matches counterfact's max_total_io_pairs_per_sample cap — keeps the
            # batch size constant across memory states instead of growing linearly).
            if num_past_probes_to_sample > 0 and seq_idx > 0:
                past_candidates = []
                for seq_idx_of_io in range(seq_idx):
                    cell_probes = [p for p in input_output_pairs[seq_idx_of_io][bi]
                                    if p.get("correct_letter")]
                    for p in cell_probes:
                        past_candidates.append((seq_idx_of_io, p))
                if past_candidates:
                    n_sample = min(num_past_probes_to_sample, len(past_candidates))
                    for seq_idx_of_io, p in random.sample(past_candidates, n_sample):
                        probes.append({**p, "_seq_idx_of_io_pairs": seq_idx_of_io})

            # Future-pull-back: collect candidate learn probes from cells (s, s+N],
            # then sample ONE total. Keeps the future-probe contribution at 1
            # probe per training step regardless of N.
            future_lo = seq_idx + 1
            future_hi = min(seq_idx + 1 + num_in_future_to_pull_back, seq_len)
            future_candidates = []
            for seq_idx_of_io in range(future_lo, future_hi):
                cell_probes = [p for p in input_output_pairs[seq_idx_of_io][bi] if p.get("correct_letter")]
                cell_probes = [p for p in cell_probes if p.get("learn_or_ignore") == "learn"]
                cell_probes = [p for p in cell_probes if p.get(f"{baseline_source}_baseline_letter")]
                for p in cell_probes:
                    future_candidates.append((seq_idx_of_io, p))
            if future_candidates:
                seq_idx_of_io, p = random.choice(future_candidates)
                probes.append({**p, "_seq_idx_of_io_pairs": seq_idx_of_io, "_future": True})

            # DDP probe-count clip
            n_local = torch.tensor(len(probes), device=device)
            n_min = n_local.clone()
            dist.all_reduce(n_min, op=dist.ReduceOp.MIN)
            probes = probes[: int(n_min.item())]
            if not probes:
                continue

            chunks = [probes[i:i + num_io_pairs_to_chunk_train]
                      for i in range(0, len(probes), num_io_pairs_to_chunk_train)]
            n_chunks_local = torch.tensor(len(chunks), device=device)
            n_chunks_min = n_chunks_local.clone()
            dist.all_reduce(n_chunks_min, op=dist.ReduceOp.MIN)
            chunks = chunks[: int(n_chunks_min.item())]

            # Loss scaling: divide by total chunks so that summed gradients form
            # a true mean over all chunks in this gradient step. Includes the
            # contrastive-variant multiplier (2 for v21 contrastive).
            num_total_chunks_factor = max(
                1,
                gradient_accumulation_steps * len(chunks) * seq_len * contrastive_variant_count,
            )

            for chunk in chunks:
                model_in.memorize(
                    document=documents[seq_idx][bi],
                    learning_instruction=learning_instructions[seq_idx][bi],
                )
                # Ablation: when freeze_memorize_grads=True, sever the gradient
                # path from the memory-write operation by detaching last_memory_update.
                # The probe-forward still uses the (now non-grad) memory delta, but
                # backward gradients stop at the detached tensor — they never flow
                # back into inject_memory's compute graph. Memorize itself ran with
                # grad tracking enabled (we just throw the graph away); model.training
                # mode + last_memory_update set means the forward path stays valid.
                if freeze_memorize_grads and getattr(model_in, "last_memory_update", None) is not None:
                    model_in.last_memory_update = model_in.last_memory_update.detach()
                input_ids, attn, letter_ids, targets, correct_ids, is_learn = build_mc_inputs(
                    model_in, chunk, baseline_source=baseline_source)
                out = model(input_ids=input_ids, attention_mask=attn)
                logits = out.logits if hasattr(out, "logits") else out[0]
                loss, nt, nl, ni, ncl, nci = mc_loss_and_metrics(
                    logits, letter_ids, targets, correct_ids, is_learn,
                    learn_loss_weight=learn_loss_weight,
                )
                (loss / num_total_chunks_factor).backward()

                micro_step_state[0] += 1
                with torch.no_grad():
                    running_loss += loss.detach().float() * nt
                    n_total += nt
                    n_learn += nl
                    n_ignore += ni
                    n_corr_learn += ncl
                    n_corr_ignore += nci
                model_in.delete_last_memory_update()
                del out, logits, loss

    accumulators["running_loss"] = running_loss
    accumulators["n_total"] = n_total
    accumulators["n_learn"] = n_learn
    accumulators["n_ignore"] = n_ignore
    accumulators["n_corr_learn"] = n_corr_learn
    accumulators["n_corr_ignore"] = n_corr_ignore


def train_one_epoch(*, model, model_in, dl, optimizer, gradient_accumulation_steps,
                     max_grad_norm, seq_len, device, epoch, global_step,
                     max_steps_per_epoch, micro_step_state,
                     num_in_future_to_pull_back: int,
                     num_learn_probes_per_cell: int,
                     num_ignore_probes_per_cell: int,
                     num_past_probes_to_sample: int,
                     num_io_pairs_to_chunk_train: int,
                     learn_loss_weight: float,
                     contrastive_pair: bool,
                     freeze_descriptor: bool = False,
                     baseline_source: str = "memoryllm",
                     freeze_memorize_grads: bool = False):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    running_loss = torch.zeros(1, device=device)
    n_total = torch.zeros(1, device=device)
    n_learn = torch.zeros(1, device=device)
    n_ignore = torch.zeros(1, device=device)
    n_corr_learn = torch.zeros(1, device=device)
    n_corr_ignore = torch.zeros(1, device=device)

    # Accumulators dict so the rollout helper can mutate them.
    accumulators = {
        "running_loss": running_loss,
        "n_total": n_total,
        "n_learn": n_learn,
        "n_ignore": n_ignore,
        "n_corr_learn": n_corr_learn,
        "n_corr_ignore": n_corr_ignore,
    }
    contrast_rng = random.Random(epoch * 0xC0FFEE + dist.get_rank() * 7)
    contrastive_variant_count = 2 if contrastive_pair else 1

    accum_count = 0
    for batch_idx, batch in enumerate(dl):
        if max_steps_per_epoch and batch_idx >= max_steps_per_epoch:
            break
        documents, learning_instructions, input_output_pairs, _meta = batch
        batch_size = len(documents[0])

        # Variant A: dataset-randomized labels (already done at __getitem__ time).
        _process_one_rollout(
            model=model, model_in=model_in,
            documents=documents,
            learning_instructions=learning_instructions,
            input_output_pairs=input_output_pairs,
            batch_size=batch_size, seq_len=seq_len, device=device,
            num_in_future_to_pull_back=num_in_future_to_pull_back,
            num_learn_probes_per_cell=num_learn_probes_per_cell,
            num_ignore_probes_per_cell=num_ignore_probes_per_cell,
            num_past_probes_to_sample=num_past_probes_to_sample,
            num_io_pairs_to_chunk_train=num_io_pairs_to_chunk_train,
            learn_loss_weight=learn_loss_weight,
            gradient_accumulation_steps=gradient_accumulation_steps,
            contrastive_variant_count=contrastive_variant_count,
            micro_step_state=micro_step_state,
            accumulators=accumulators,
            baseline_source=baseline_source,
            freeze_memorize_grads=freeze_memorize_grads,
        )

        # Variant B (contrastive): same docs, DIFFERENT target_topic per example,
        # re-rendered LIs and re-derived learn/ignore labels. Forces the model to
        # use the LI rather than memorize a (doc → answer) mapping.
        if contrastive_pair:
            lis_b, pairs_b = _relabel_variant(
                documents, learning_instructions, input_output_pairs, _meta, contrast_rng,
                freeze_descriptor=freeze_descriptor,
            )
            _process_one_rollout(
                model=model, model_in=model_in,
                documents=documents,
                learning_instructions=lis_b,
                input_output_pairs=pairs_b,
                batch_size=batch_size, seq_len=seq_len, device=device,
                num_in_future_to_pull_back=num_in_future_to_pull_back,
                num_learn_probes_per_cell=num_learn_probes_per_cell,
                num_ignore_probes_per_cell=num_ignore_probes_per_cell,
                num_past_probes_to_sample=num_past_probes_to_sample,
                num_io_pairs_to_chunk_train=num_io_pairs_to_chunk_train,
                learn_loss_weight=learn_loss_weight,
                gradient_accumulation_steps=gradient_accumulation_steps,
                contrastive_variant_count=contrastive_variant_count,
                micro_step_state=micro_step_state,
                accumulators=accumulators,
                baseline_source=baseline_source,
                freeze_memorize_grads=freeze_memorize_grads,
            )

        # Pull mutated values back out for grad-step bookkeeping below.
        running_loss = accumulators["running_loss"]
        n_total = accumulators["n_total"]
        n_learn = accumulators["n_learn"]
        n_ignore = accumulators["n_ignore"]
        n_corr_learn = accumulators["n_corr_learn"]
        n_corr_ignore = accumulators["n_corr_ignore"]

        accum_count += 1
        if accum_count % gradient_accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            for t in (running_loss, n_total, n_learn, n_ignore, n_corr_learn, n_corr_ignore):
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
            mean_loss = (running_loss / n_total.clamp(min=1)).item()
            acc_learn = (n_corr_learn / n_learn.clamp(min=1)).item()
            acc_ignore_correct = (n_corr_ignore / n_ignore.clamp(min=1)).item()
            selectivity = 1.0 - acc_ignore_correct  # % of IGNORE probes where pred ≠ correct_letter
            if dist.get_rank() == 0:
                rprint(f"[epoch {epoch} step {global_step}] loss={mean_loss:.4f} "
                       f"acc_learn={acc_learn:.3f} selectivity={selectivity:.3f} "
                       f"grad={grad_norm:.2f}")
                if wandb is not None and wandb.run is not None:
                    wandb.log({
                        "loss/train": mean_loss,
                        "acc/train/learn": acc_learn,
                        "acc/train/selectivity": selectivity,
                        "debug/grad_norm": float(grad_norm),
                        "debug/opt_step": global_step,
                        "debug/micro_step": micro_step_state[0],
                    })
            for t in (running_loss, n_total, n_learn, n_ignore, n_corr_learn, n_corr_ignore):
                t.zero_()

    return global_step


@torch.no_grad()
def evaluate_split(*, model, model_in, dl, seq_len, device, split_name: str,
                    num_io_pairs_to_chunk_eval: int,
                    learn_loss_weight: float,
                    baseline_source: str = "memoryllm"):
    """Roll through a dataloader, after each memorize step probe pairs from EVERY
    seq_idx (full seq_len × seq_len grid, including FUTURE pairs whose source doc
    has not yet been memorized — that's the upper triangle, used as a sanity check
    that memorize() actually changes the model's predictions).

    Returned matrices: shape (seq_len, seq_len), x-axis = seq_idx_of_doc
    (memorize step), y-axis = seq_idx_of_io_pairs (probe origin doc). Lower triangle
    + diagonal = the relevant doc has been memorized; upper triangle = it has not.
    """
    model.eval()

    # Per-cell counters: shape (seq_len, seq_len), [doc][pair]
    n_learn_cell = torch.zeros((seq_len, seq_len), device=device)
    n_ignore_cell = torch.zeros((seq_len, seq_len), device=device)
    n_corr_learn_cell = torch.zeros((seq_len, seq_len), device=device)
    n_corr_ignore_cell = torch.zeros((seq_len, seq_len), device=device)
    loss_sum = torch.zeros(1, device=device)
    n_total_loss = torch.zeros(1, device=device)

    for batch in dl:
        documents, learning_instructions, input_output_pairs, _meta = batch
        batch_size = len(documents[0])
        for bi in range(batch_size):
            model_in.reset_memory()
            for seq_idx_of_doc in range(seq_len):
                model_in.memorize(
                    document=documents[seq_idx_of_doc][bi],
                    learning_instruction=learning_instructions[seq_idx_of_doc][bi],
                )
                # Probe pairs from ALL seq_idx (full grid). For VALIDATION we keep
                # every probe regardless of upper/lower triangle and don't apply the
                # training-time future-pull-back filter — eval is for *measuring*
                # behavior, and filtering future cells to learn-only causes DDP
                # min-clip to drop entire cells when any rank has 0 learn probes
                # for that io_idx (creates blank heatmap cells).
                # val_loss is computed only on the lower-triangle + diagonal (cells
                # where the relevant doc has been memorized) — upper triangle is
                # tracked for diagnostics but excluded from loss.
                for seq_idx_of_io in range(seq_len):
                    cell_probes = [p for p in input_output_pairs[seq_idx_of_io][bi] if p.get("correct_letter")]
                    probes = cell_probes
                    # DDP probe-count clip
                    n_local = torch.tensor(len(probes), device=device)
                    n_min = n_local.clone()
                    dist.all_reduce(n_min, op=dist.ReduceOp.MIN)
                    probes = probes[: int(n_min.item())]
                    if not probes:
                        continue
                    for i in range(0, len(probes), num_io_pairs_to_chunk_eval):
                        chunk = probes[i:i + num_io_pairs_to_chunk_eval]
                        input_ids, attn, letter_ids, targets, correct_ids, is_learn = build_mc_inputs(
                            model_in, chunk, baseline_source=baseline_source)
                        out = model(input_ids=input_ids, attention_mask=attn)
                        logits = out.logits if hasattr(out, "logits") else out[0]
                        loss, nt, nl, ni, ncl, nci = mc_loss_and_metrics(
                            logits, letter_ids, targets, correct_ids, is_learn,
                            learn_loss_weight=learn_loss_weight,
                        )
                        # Loss only meaningful when the relevant doc has been memorized
                        # (lower-triangle + diagonal). Upper-triangle cells are blind
                        # probes — drag val_loss toward the uniform floor (~1.39).
                        if seq_idx_of_io <= seq_idx_of_doc:
                            loss_sum += loss * nt
                            n_total_loss += nt
                        n_learn_cell[seq_idx_of_doc, seq_idx_of_io] += nl
                        n_ignore_cell[seq_idx_of_doc, seq_idx_of_io] += ni
                        n_corr_learn_cell[seq_idx_of_doc, seq_idx_of_io] += ncl
                        n_corr_ignore_cell[seq_idx_of_doc, seq_idx_of_io] += nci
                        del out, logits

    for t in (loss_sum, n_total_loss, n_learn_cell, n_ignore_cell, n_corr_learn_cell, n_corr_ignore_cell):
        dist.all_reduce(t, op=dist.ReduceOp.SUM)

    nl = n_learn_cell.cpu().numpy()
    ni = n_ignore_cell.cpu().numpy()
    ncl = n_corr_learn_cell.cpu().numpy()
    nci = n_corr_ignore_cell.cpu().numpy()

    # Per-cell rates; NaN where the cell has no probes (e.g. upper triangle, or rare empties)
    with np.errstate(invalid="ignore", divide="ignore"):
        acc_matrix = np.where(nl > 0, ncl / np.maximum(nl, 1), np.nan)
        sel_matrix = np.where(ni > 0, 1.0 - nci / np.maximum(ni, 1), np.nan)

    # Aggregate scalars over the full grid
    n_learn_total = nl.sum()
    n_ignore_total = ni.sum()
    n_corr_learn_total = ncl.sum()
    n_corr_ignore_total = nci.sum()
    val_acc_learn = float(n_corr_learn_total / max(1, n_learn_total))
    val_acc_ignore_correct = float(n_corr_ignore_total / max(1, n_ignore_total))
    val_selectivity = 1.0 - val_acc_ignore_correct
    val_loss = float((loss_sum / n_total_loss.clamp(min=1)).item())

    # Diagonal-only aggregate: cells where seq_idx_of_doc == seq_idx_of_io_pairs
    # (the just-memorized doc — cleanest signal, no decay, no "future probe")
    diag_idx = np.arange(seq_len)
    nl_diag = nl[diag_idx, diag_idx]
    ni_diag = ni[diag_idx, diag_idx]
    ncl_diag = ncl[diag_idx, diag_idx]
    nci_diag = nci[diag_idx, diag_idx]
    acc_learn_diag = float(ncl_diag.sum() / max(1, nl_diag.sum()))
    acc_ignore_correct_diag = float(nci_diag.sum() / max(1, ni_diag.sum()))
    selectivity_diag = 1.0 - acc_ignore_correct_diag

    return {
        "split": split_name,
        "loss": val_loss,
        "acc_learn": val_acc_learn,
        "selectivity": val_selectivity,
        "acc_ignore_picks_correct": val_acc_ignore_correct,
        "selectivity_gap": val_acc_learn - val_acc_ignore_correct,
        "n_learn": int(n_learn_total),
        "n_ignore": int(n_ignore_total),
        "acc_matrix": acc_matrix,    # shape (seq_len, seq_len), x=seq_idx_of_doc, y=seq_idx_of_io_pairs
        "sel_matrix": sel_matrix,
        # Diagonal-only — used for bar charts (cleaner signal than full grid).
        "acc_learn_diag": acc_learn_diag,
        "selectivity_diag": selectivity_diag,
        "n_learn_diag": int(nl_diag.sum()),
        "n_ignore_diag": int(ni_diag.sum()),
    }


def plot_split_barchart(splits: list[dict], metric_key: str, title: str, ylabel: str,
                          random_baseline: float = 0.25):
    """One bar per split (x-axis = split name, one metric).
    `random_baseline` is the expected value under uniform 4-way letter picks
    (0.25 for accuracy on LEARN, 0.75 for selectivity on IGNORE)."""
    split_names = [s["split"] for s in splits]
    vals = [float(s.get(metric_key, 0.0)) for s in splits]
    fig, ax = plt.subplots(figsize=(max(4.5, 1.3 * len(split_names) + 1), 4))
    bars = ax.bar(split_names, vals, color=["#5b8def", "#ef9a3b", "#3bb273"][: len(split_names)])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(random_baseline, color="gray", linestyle=":", linewidth=1,
                label=f"random ({random_baseline:.2f})")
    ax.set_ylim(min(-0.05, min(vals) - 0.05), 1.0)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def plot_seq_heatmap(matrix: np.ndarray, title: str):
    """seq_len × seq_len heatmap. x = seq_idx_of_doc (memorize step),
    y = seq_idx_of_io_pairs (probe origin doc). NaN cells (e.g. upper triangle) are dimmed.
    """
    seq_len = matrix.shape[0]
    fig, ax = plt.subplots(figsize=(max(4, seq_len * 0.7 + 1), max(4, seq_len * 0.7 + 1)))
    im = ax.imshow(matrix.T, cmap="viridis", vmin=0, vmax=1)  # transpose so x=doc, y=pair
    ax.set_xticks(range(seq_len))
    ax.set_yticks(range(seq_len))
    ax.set_xticklabels([str(i) for i in range(seq_len)])
    ax.set_yticklabels([str(i) for i in range(seq_len)])
    ax.set_xlabel("seq_idx_of_doc (memorize step)")
    ax.set_ylabel("seq_idx_of_io_pairs (probe origin)")
    ax.set_title(title)
    for d in range(seq_len):
        for p in range(seq_len):
            v = matrix[d, p]
            if np.isnan(v):
                ax.text(d, p, "·", ha="center", va="center", color="lightgray", fontsize=10)
            else:
                ax.text(d, p, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def log_epoch_plots(epoch: int, splits: list[dict]):
    """Push per-epoch charts to wandb under the `epoch_{N}/...` namespace."""
    if wandb is None or wandb.run is None:
        return
    payload = {}
    ep_prefix = f"epoch_{epoch}"

    # --- bar charts: one for accuracy across splits, one for selectivity ---
    fig = plot_split_barchart(splits, "acc_learn_diag",
                              f"Accuracy (diagonal — just-memorized) — epoch {epoch}",
                              "accuracy on LEARN probes (higher = better)",
                              random_baseline=0.25)
    payload[f"{ep_prefix}/accuracy_barchart"] = wandb.Image(fig)
    plt.close(fig)

    fig = plot_split_barchart(splits, "selectivity_diag",
                              f"Selectivity (diagonal — just-memorized) — epoch {epoch}",
                              "selectivity (% IGNORE probes where pred ≠ correct)",
                              random_baseline=0.75)
    payload[f"{ep_prefix}/selectivity_barchart"] = wandb.Image(fig)
    plt.close(fig)

    # --- per-split heatmaps: accuracy + selectivity over (seq_idx_of_doc × seq_idx_of_io_pairs) ---
    for s in splits:
        fig = plot_seq_heatmap(s["acc_matrix"], f"Accuracy — {s['split']} (epoch {epoch})")
        payload[f"{ep_prefix}/{s['split']}/accuracy_heatmap"] = wandb.Image(fig)
        plt.close(fig)
        fig = plot_seq_heatmap(s["sel_matrix"], f"Selectivity — {s['split']} (epoch {epoch})")
        payload[f"{ep_prefix}/{s['split']}/selectivity_heatmap"] = wandb.Image(fig)
        plt.close(fig)
    wandb.log(payload)


def save_checkpoint(model_in, path: Path, epoch: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "state_dict": model_in.state_dict()}, path)
    rprint(f"saved checkpoint -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=Path, required=True)
    ap.add_argument("--save-dir", type=Path, default=Path("../../saved_models/gnm_personamem"))
    ap.add_argument("--max-steps-per-epoch", type=int, default=None)
    args = ap.parse_args()

    cfg = load_cfg(str(args.params))
    cfg.validate()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    seed_everything(cfg.seed, rank=rank)
    if rank == 0 and wandb is not None:
        try:
            wandb.init(project=cfg.log.wandb_project, name=cfg.run_name, config=cfg.to_flat_dict())
        except Exception as e:
            rprint(f"wandb init failed: {e}")

    model_wrapper = build_model(cfg).to(device)
    # NOTE: gradient checkpointing is INTENTIONALLY DISABLED for memory models.
    # MemoryLLM's cat_memory_and_hiddens uses torch.randperm to drop memory tokens at
    # forward time. With grad checkpointing on, the forward is re-run during backward
    # and randperm returns DIFFERENT indices — autograd then operates on tensors whose
    # shapes/contents don't match the original forward, which intermittently triggers
    # CUDA "illegal memory access". For MemoryLLM our MC docs are short (~150-300 tokens),
    # so we don't need it. ICL wrappers (Llama-3 fine-tune): no memory-token randperm,
    # AND probe forwards include the docs in the system prompt → ~1500-tokens per probe
    # × full 8B model + 8bit AdamW state runs OOM on H100 without checkpointing. So we
    # ENABLE gradient checkpointing only for ICL wrappers (no `.memory` attribute).
    if hasattr(model_wrapper.model, "config"):
        model_wrapper.model.config.use_cache = False
    is_icl_wrapper = not hasattr(model_wrapper.model, "memory")
    if is_icl_wrapper and hasattr(model_wrapper.model, "gradient_checkpointing_enable"):
        # use_reentrant=False is the modern path; avoids the require-grad-on-input
        # constraint of the reentrant variant and plays nicely with DDP+static_graph.
        try:
            model_wrapper.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            # older HF — fall back to plain enable
            model_wrapper.model.gradient_checkpointing_enable()
        rprint("[trainer] gradient_checkpointing ENABLED (ICL wrapper).")
    if cfg.model.model_state_dict_path:
        rprint(f"loading state dict from {cfg.model.model_state_dict_path}")
        sd = torch.load(cfg.model.model_state_dict_path, map_location="cpu", weights_only=False)
        sd = normalize_wrapper_state_dict(sd)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()}
        model_wrapper.load_state_dict(sd, strict=False)

    model = DDP(model_wrapper, find_unused_parameters=False,
                device_ids=[local_rank], output_device=local_rank)
    model._set_static_graph()
    model_in = model.module

    optimizer = construct_memoryllm_optimizer(model=model_wrapper, cfg=cfg)
    scheduler = construct_memoryllm_scheduler(optimizer=optimizer, cfg=cfg)
    if rank == 0 and scheduler is not None:
        rprint(f"[trainer] LR scheduler: {type(scheduler).__name__}, warmup_epochs={cfg.opt.lr_warmup_epochs}, peak_lr={cfg.opt.learning_rate}")

    seq_len = cfg.loop.sequence_length
    seq_len_val = cfg.loop.sequence_length_validation or seq_len
    bs_per_device = max(1, cfg.opt.batch_size_per_device)
    dl_batch = bs_per_device * seq_len
    dl_batch_val = bs_per_device * seq_len_val
    rprint(f"sequence_length={seq_len} (train) / {seq_len_val} (val); dataloader batch={dl_batch}/{dl_batch_val}")

    train_paths = cfg.data.training_datasets
    val_path = cfg.data.val_id_path
    val_dataset = PreferenceGNMDataset(val_path, shuffle=False, limit=cfg.data.val_limit)
    val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    val_collate = lambda b: seqlen_collate(b, sequence_length=seq_len_val)
    val_dl = DataLoader(val_dataset, batch_size=dl_batch_val, sampler=val_sampler,
                         collate_fn=val_collate, drop_last=True, pin_memory=True)

    # Optional OOD splits for per-epoch monitoring. Each path in val_ood_datasets gets
    # its own dataloader + split_name (derived from the file stem). Used for per-split
    # heatmaps and val curves so we can monitor multiple held-out distributions
    # (e.g. personamem unseen topics + prefeval transfer) throughout training.
    ood_dls = []  # list of (split_name, dataloader)
    OOD_LIMIT = 100
    ood_paths = cfg.data.val_ood_datasets or []
    for ood_path in ood_paths:
        stem = Path(ood_path).stem
        # Strip pipeline-export prefixes/suffixes for tidy wandb keys.
        if stem.startswith("held_out_"):
            stem = stem[len("held_out_"):]
        for sfx in ("_strict_rebuilt", "_mc_strict", "_strict", "_mc"):
            if stem.endswith(sfx):
                stem = stem[: -len(sfx)]
                break
        # Don't prefix `val_` if it already starts with one (e.g. val_unseen_topics
        # is already self-describing). PrefEval is held-out from training but isn't
        # an "unseen" topic split — it's a separate dataset entirely.
        split_name = stem if stem.startswith("val_") else f"val_{stem}"
        ood_dataset = PreferenceGNMDataset(ood_path, shuffle=False, limit=OOD_LIMIT)
        ood_sampler = DistributedSampler(ood_dataset, shuffle=False, drop_last=True)
        ood_dl = DataLoader(ood_dataset, batch_size=dl_batch_val, sampler=ood_sampler,
                             collate_fn=val_collate, drop_last=True, pin_memory=True)
        ood_dls.append((split_name, ood_dl))
        rprint(f"OOD eval split [{split_name}]: {ood_path}  limit={OOD_LIMIT}")

    # Train-sample split for per-epoch monitoring (read on training distribution).
    train_sample_dl = None
    TRAIN_SAMPLE_LIMIT = 60
    if train_paths:
        train_sample_ds = PreferenceGNMDataset(train_paths[0], shuffle=False, limit=TRAIN_SAMPLE_LIMIT)
        train_sample_sampler = DistributedSampler(train_sample_ds, shuffle=False, drop_last=True)
        train_sample_dl = DataLoader(train_sample_ds, batch_size=dl_batch_val, sampler=train_sample_sampler,
                                       collate_fn=val_collate, drop_last=True, pin_memory=True)

    best_val = float("inf")
    epochs_since_improve = 0
    global_step = 0
    micro_step_state = [0]

    for epoch in range(cfg.loop.start_epoch, cfg.loop.max_epochs):
        for ds_idx, train_path in enumerate(train_paths):
            rprint(f"[epoch {epoch}] dataset {ds_idx + 1}/{len(train_paths)}: {train_path}")
            # Explicit per-epoch seed so (a) the within-file subset chosen each epoch
            # is different and (b) DistributedSampler's per-epoch shuffle is also different.
            ds_seed = cfg.seed * 1_000_003 + epoch * len(train_paths) + ds_idx
            train_dataset = PreferenceGNMDataset(train_path, shuffle=True, limit=cfg.data.train_limit,
                                                  seed=ds_seed, randomize_li=True,
                                                  freeze_descriptor=cfg.sampling.freeze_descriptor)
            train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
            train_sampler.set_epoch(epoch * len(train_paths) + ds_idx)
            train_collate = lambda b: seqlen_collate(b, sequence_length=seq_len)
            train_dl = DataLoader(train_dataset, batch_size=dl_batch, sampler=train_sampler,
                                   collate_fn=train_collate, drop_last=True, pin_memory=True)
            global_step = train_one_epoch(
                model=model, model_in=model_in, dl=train_dl, optimizer=optimizer,
                gradient_accumulation_steps=cfg.opt.gradient_accumulation_steps,
                max_grad_norm=cfg.opt.max_grad_norm, seq_len=seq_len,
                device=device, epoch=epoch, global_step=global_step,
                max_steps_per_epoch=args.max_steps_per_epoch,
                micro_step_state=micro_step_state,
                num_in_future_to_pull_back=cfg.sampling.num_in_future_to_pull_back,
                num_learn_probes_per_cell=cfg.sampling.num_learn_probes_per_cell,
                num_ignore_probes_per_cell=cfg.sampling.num_ignore_probes_per_cell,
                num_past_probes_to_sample=cfg.sampling.num_past_probes_to_sample,
                num_io_pairs_to_chunk_train=cfg.sampling.num_io_pairs_to_chunk_train,
                learn_loss_weight=cfg.sampling.learn_loss_weight,
                contrastive_pair=cfg.sampling.contrastive_pair,
                freeze_descriptor=cfg.sampling.freeze_descriptor,
                baseline_source=cfg.data.baseline_source,
                freeze_memorize_grads=cfg.sampling.freeze_memorize_grads,
            )

        # Per-epoch evaluation across train_sample / val (seen) / val_ood (unseen)
        splits = []
        eval_kw = dict(
            num_io_pairs_to_chunk_eval=cfg.sampling.num_io_pairs_to_chunk_eval,
            learn_loss_weight=cfg.sampling.learn_loss_weight,
            baseline_source=cfg.data.baseline_source,
        )
        if train_sample_dl is not None:
            splits.append(evaluate_split(model=model, model_in=model_in, dl=train_sample_dl,
                                          seq_len=seq_len_val, device=device, split_name="train_sample",
                                          **eval_kw))
        splits.append(evaluate_split(model=model, model_in=model_in, dl=val_dl,
                                      seq_len=seq_len_val, device=device, split_name="val_seen",
                                      **eval_kw))
        for split_name, ood_dl in ood_dls:
            splits.append(evaluate_split(model=model, model_in=model_in, dl=ood_dl,
                                          seq_len=seq_len_val, device=device, split_name=split_name,
                                          **eval_kw))

        val_loss = next(s for s in splits if s["split"] == "val_seen")["loss"]
        if rank == 0:
            rprint(f"[epoch {epoch}] best={best_val:.4f}")
            for s in splits:
                rprint(f"  [{s['split']}] loss={s['loss']:.4f}  "
                       f"diag: acc_learn={s['acc_learn_diag']:.3f} selectivity={s['selectivity_diag']:.3f} "
                       f"(n_learn={s['n_learn_diag']}, n_ignore={s['n_ignore_diag']})  |  "
                       f"full-grid: acc_learn={s['acc_learn']:.3f} selectivity={s['selectivity']:.3f}")
            if wandb is not None and wandb.run is not None:
                # Scalar series — top-level keys for cross-epoch line charts.
                # Diagonal-only metrics are the primary signal (just-memorized cells).
                scalar_payload = {"debug/micro_step": micro_step_state[0], "debug/epoch": epoch}
                for s in splits:
                    scalar_payload[f"loss/{s['split']}"] = s["loss"]
                    scalar_payload[f"acc/{s['split']}/learn"] = s["acc_learn_diag"]
                    scalar_payload[f"acc/{s['split']}/selectivity"] = s["selectivity_diag"]
                    scalar_payload[f"acc_full_grid/{s['split']}/learn"] = s["acc_learn"]
                    scalar_payload[f"acc_full_grid/{s['split']}/selectivity"] = s["selectivity"]
                wandb.log(scalar_payload)
                # Per-epoch artifacts grouped under epoch_{N}/...
                log_epoch_plots(epoch, splits)
            if val_loss < best_val - cfg.loop.early_stop_min_delta:
                best_val = val_loss
                epochs_since_improve = 0
                save_checkpoint(model_in, args.save_dir / f"{cfg.run_name}_best.pth", epoch)
            else:
                epochs_since_improve += 1
            save_checkpoint(model_in, args.save_dir / f"{cfg.run_name}_recent.pth", epoch)

        stop_signal = torch.tensor([1 if epochs_since_improve > cfg.loop.early_stop_patience else 0], device=device)
        dist.broadcast(stop_signal, src=0)
        if cfg.loop.early_stop and stop_signal.item():
            rprint("early stopping triggered")
            break

        # End-of-epoch LR scheduler step (warmup → cosine, per-epoch).
        if scheduler is not None:
            scheduler.step()
            if rank == 0:
                cur_lr = optimizer.param_groups[0]['lr']
                rprint(f"[epoch {epoch}] scheduler stepped → lr now {cur_lr:.2e}")

    if rank == 0 and wandb is not None and wandb.run is not None:
        wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
