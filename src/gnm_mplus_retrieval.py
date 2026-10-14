"""
Retrieval-layer additions for GNM: run the trained gnm-all (MemoryLLM) checkpoint
inside the M+ shell with always-on LTM token retrieval, plus exact provenance
tracking of which injected document produced every memory row.

Everything for this project lives in this module so core code stays untouched:
- modeling_mplus.py: NOT modified (we subclass MPlus here).
- src/gnm.py: only a tiny import hook at the bottom of the file; the wrappers
  register themselves into gnm._REGISTRY (see register_wrappers / module bottom).

Classes
-------
MPlusWithProvenance(MPlus)
    Records the random-drop indices that `drop_memory` already computes and
    maintains per-layer provenance arrays (doc-id per memory row) for STM,
    cached-dropped, and LTM pools. Also supports an "oracle retrieval" mode
    (force rows of a chosen doc into the retrieved LTM set) used for the
    upper-bound gate before retriever training.

GNMPlusChatWrapper(MPlusChatWrapper)
    Same RoPE override + weight splice as the parent, but:
    (a) instantiates MPlusWithProvenance,
    (b) seeds the LTM with the 2560 oldest gnm STM tokens that the 12800->10240
        splice truncation throws away (replicates the successful _ltm_seeded
        variant, with gnm content instead of mplus content),
    (c) re-snapshots baselines AFTER splice+seed, so reset_memory restores
        gnm-all's STM / gnm-seeded LTM rather than mplus-8b's pretrained memory
        (the parent deliberately keeps mplus's baseline; we deliberately don't).


    Oracle upper bound: doc-0 rows are force-included in retrieval (matches
    probe_first_only evals, where doc 0 is the probed document).
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from modeling_mplus import MPlus
from transformers import AutoConfig, AutoTokenizer

from gnm import MPlusChatWrapper, ICLChatWrapper
from config_memoryllm_train import MemoryllmTrainConfig

# Provenance labels for rows that did not come from an injected document.
PROV_BACKGROUND = -1  # rows of the (frozen) gnm STM present before the rollout
PROV_SEED = -2        # rows seeded into the LTM from the truncated gnm STM


class MPlusWithProvenance(MPlus):

    def __init__(self, config):
        super().__init__(config)
        # Provenance tracking is inactive until reset_provenance() is called
        # (the wrapper calls it after splice+seed and on every reset_memory).
        self.stm_prov = None
        self.ltm_prov = None
        self.cached_dropped_prov = None
        self._drop_ptr = 0
        self._current_doc_id = PROV_BACKGROUND
        self.oracle_doc_id = None
        # pure-LTM doc store (retrieval_mode == "store"): frozen STM, every doc
        # diverted here as a complete context-independent block
        self.store_blocks = None   # [layer] -> list of [num_tokens, d] cpu tensors
        self.store_keys = None     # [layer] -> list of [num_tokens, key_dim] cpu tensors
        self.store_ids = []
        self.static_pad = None     # [L, num_ltm_tokens, d] background rows (train-dist filler)

    # ------------------------------------------------------------------
    # provenance bookkeeping
    # ------------------------------------------------------------------

    def reset_provenance(self):
        self.stm_prov = [
            np.full(self.memory.shape[1], PROV_BACKGROUND, dtype=np.int64)
            for _ in range(self.L)
        ]
        self.ltm_prov = [
            np.full(len(self.ltm[i]), PROV_SEED, dtype=np.int64)
            for i in range(self.L)
        ]
        self.cached_dropped_prov = [np.zeros(0, dtype=np.int64) for _ in range(self.L)]
        # within-doc row position (0..num_tokens-1); -1 for background/seed rows
        self.stm_pos = [np.full(self.memory.shape[1], -1, dtype=np.int64) for _ in range(self.L)]
        self.ltm_pos = [np.full(len(self.ltm[i]), -1, dtype=np.int64) for i in range(self.L)]
        self.cached_dropped_pos = [np.zeros(0, dtype=np.int64) for _ in range(self.L)]
        self._drop_ptr = 0
        self._current_doc_id = PROV_BACKGROUND

    def provenance_summary(self):
        """Per-pool counts of rows per doc id (for tests/analysis). Layer 0 view
        plus a consistency check that all layers have identical counts."""
        import collections
        out = {}
        for name, arrs in [("stm", self.stm_prov), ("cached", self.cached_dropped_prov), ("ltm", self.ltm_prov)]:
            counts0 = collections.Counter(arrs[0].tolist())
            out[name] = dict(sorted(counts0.items()))
        return out

    # `drop_memory` has exactly one call site in modeling_mplus.py
    # (base_update_memory_with_delta_memory, non-retriever path): called once per
    # layer, in layer order, with unsequeezed=False. We piggyback on it to move
    # provenance labels the same way the memory rows move.
    def drop_memory(self, current_memory, drop_length=None, unsequeezed=True,
                    return_remaining_indices=False, return_dropped_indices=False):
        tracking = self.stm_prov is not None and not unsequeezed
        current_memory, remaining_indices, dropped_indices = super().drop_memory(
            current_memory, drop_length, unsequeezed,
            return_remaining_indices=True, return_dropped_indices=True)

        if tracking:
            idx = self._drop_ptr
            self._drop_ptr += 1
            old = self.stm_prov[idx]
            rem = remaining_indices.detach().cpu().numpy()
            drp = dropped_indices.detach().cpu().numpy()
            n_new = old.shape[0] - rem.shape[0]  # == drop count == appended delta rows
            self.cached_dropped_prov[idx] = np.concatenate([self.cached_dropped_prov[idx], old[drp]])
            self.stm_prov[idx] = np.concatenate([
                old[rem],
                np.full(n_new, self._current_doc_id, dtype=np.int64),
            ])
            old_pos = self.stm_pos[idx]
            self.cached_dropped_pos[idx] = np.concatenate([self.cached_dropped_pos[idx], old_pos[drp]])
            self.stm_pos[idx] = np.concatenate([old_pos[rem], np.arange(n_new, dtype=np.int64)])

        result = [current_memory]
        if return_remaining_indices:
            result.append(remaining_indices)
        if return_dropped_indices:
            result.append(dropped_indices)
        return tuple(result) if len(result) > 1 else current_memory

    def update_memory_with_delta_memory(self, delta_memory, cached_contexts_indicators=None,
                                        retriever_weights=None, delta_memory_ages=None,
                                        dropped_delta_memory=None, dropped_delta_memory_ages=None):
        tracking = self.stm_prov is not None
        if tracking:
            if retriever_weights is not None:
                raise NotImplementedError(
                    "Provenance tracking assumes the random-drop path (use_retriever=False).")
            if dropped_delta_memory is not None:
                raise NotImplementedError(
                    "Provenance tracking does not mirror the dropped_delta_memory path.")
            self._drop_ptr = 0
        out = super().update_memory_with_delta_memory(
            delta_memory, cached_contexts_indicators=cached_contexts_indicators,
            retriever_weights=retriever_weights, delta_memory_ages=delta_memory_ages,
            dropped_delta_memory=dropped_delta_memory,
            dropped_delta_memory_ages=dropped_delta_memory_ages)
        if tracking:
            assert self._drop_ptr == self.L, (
                f"expected one drop_memory call per layer, got {self._drop_ptr}/{self.L}")
            # After a merge (triggered inside the super() call) cached provenance
            # is cleared by our merge_cached_memory override; otherwise it must
            # line up with the cached-dropped tensor.
            if self.cached_dropped_memories is not None:
                assert self.cached_dropped_prov[0].shape[0] == self.cached_dropped_memories.shape[1], (
                    f"cached prov {self.cached_dropped_prov[0].shape[0]} != "
                    f"cached rows {self.cached_dropped_memories.shape[1]}")
        return out

    def reset_store(self):
        self.store_blocks = [[] for _ in range(self.L)]
        self.store_keys = [[] for _ in range(self.L)]
        self.store_ids = []

    def store_add(self, delta, doc_id):
        """delta: [L, num_tokens, d] freshly-written block for one document."""
        with torch.no_grad():
            for l in range(self.L):
                block = delta[l].detach()
                layer = self.model.layers[l]
                k = layer.self_attn.key_proj(layer.input_layernorm(
                    block.to(layer.input_layernorm.weight.device).to(layer.input_layernorm.weight.dtype)))
                self.store_blocks[l].append(block.cpu())
                self.store_keys[l].append(k.detach().cpu())
        self.store_ids.append(int(doc_id))

    def _store_select(self, idx, hidden_states, dev, dt):
        """Rank stored doc blocks; return the top-K complete blocks, best doc
        adjacent to the input; pad the front with static background rows."""
        k_total = self.num_ltm_blocks * self.num_tokens
        n_docs_max = k_total // self.num_tokens
        layer = self.model.layers[idx]
        if getattr(self, "_capture_hiddens", None) is not None:
            self._capture_hiddens[idx] = hidden_states[0].detach()
        if not self.store_ids:
            return self.static_pad[idx].to(dev).to(dt), torch.zeros(0, dtype=torch.long)
        forced = getattr(self, "_forced_docs", None)
        if forced is not None:
            with torch.no_grad():
                chosen = [self.store_ids.index(d) for d in reversed(forced) if d in self.store_ids]
                rows = torch.cat([self.store_blocks[idx][i].to(dev).to(dt) for i in chosen], dim=0)                     if chosen else torch.zeros(0, self.d, device=dev, dtype=dt)
                if getattr(self, "_record_block_docs", None) is not None:
                    self._record_block_docs[idx] = [self.store_ids[i] for i in reversed(chosen)]
                if rows.shape[0] < k_total:
                    pad = self.static_pad[idx][: k_total - rows.shape[0]].to(dev).to(dt)
                    rows = torch.cat([pad, rows], dim=0)
            return rows.detach(), torch.zeros(0, dtype=torch.long)
        with torch.no_grad():
            K = torch.cat([k.to(dev).to(dt) for k in self.store_keys[idx]], dim=0)
            q = layer.self_attn.query_proj(layer.input_layernorm(hidden_states[0]))
            # cosine space — must match training/val ranking exactly (raw dots
            # have pathological +-300 scale from the warm start)
            qn = torch.nn.functional.normalize(q.float(), dim=-1)
            Kn = torch.nn.functional.normalize(K.float(), dim=-1)
            s = (qn @ Kn.transpose(-2, -1)).max(dim=0).values  # [n_docs*256]
            per_doc = s.view(len(self.store_ids), self.num_tokens)
            logits = per_doc.topk(min(16, self.num_tokens), dim=1).values.mean(dim=1)
            if self.oracle_doc_id is not None and self.oracle_doc_id in self.store_ids:
                logits = logits.clone()
                logits[self.store_ids.index(self.oracle_doc_id)] = float("inf")
            order = torch.argsort(logits, descending=True)[:n_docs_max]
            chosen = sorted(order.tolist(), key=lambda i: logits[i].item())  # best LAST
            rows = torch.cat([self.store_blocks[idx][i].to(dev).to(dt) for i in chosen], dim=0)
            if getattr(self, "_record_block_docs", None) is not None:
                self._record_block_docs[idx] = [self.store_ids[i] for i in reversed(chosen)]
            if rows.shape[0] < k_total:
                pad = self.static_pad[idx][: k_total - rows.shape[0]].to(dev).to(dt)
                rows = torch.cat([pad, rows], dim=0)
        return rows.detach(), torch.zeros(0, dtype=torch.long)

    RAWCOS_LAYERS = (8, 12, 16, 20, 24)

    def rank_docs_rawcos(self, cap):
        """Global doc ranking by raw-hidden-state cosine over mid layers.
        cap: {layer: [T, d] captured probe hiddens}. Zero learned parameters."""
        import torch.nn.functional as F
        with torch.no_grad():
            totals = None
            used = 0
            for l in self.RAWCOS_LAYERS:
                if l not in cap:
                    continue
                dev = cap[l].device
                h = F.normalize(cap[l].float(), dim=-1)
                per_doc = []
                for i in range(len(self.store_ids)):
                    rows = F.normalize(self.store_blocks[l][i].to(dev).float(), dim=-1)
                    per_doc.append((h @ rows.T).max())
                sc = torch.stack(per_doc)
                totals = sc if totals is None else totals + sc
                used += 1
            if totals is None or used == 0:
                return []
            order = torch.argsort(totals, descending=True)
            n = (self.num_ltm_blocks * self.num_tokens) // self.num_tokens
            return [self.store_ids[i] for i in order[:n].tolist()]

    def _ensure_ltm_keys(self, idx):
        """Materialize ltm_keys[idx] if it is None (we clear keys after seeding /
        reset so they are rebuilt with the current key_proj). update_ltm
        concatenates onto ltm_keys, so keys must exist before any merge — a
        merge can fire before the first retrieval ever built them lazily."""
        if self.ltm_keys[idx] is not None:
            return
        layer = self.model.layers[idx]
        dev = layer.input_layernorm.weight.device
        dt = layer.input_layernorm.weight.dtype
        with torch.no_grad():
            tmp = []
            for batch_ltm in torch.split(self.ltm[idx], 64):
                batch_ltm = batch_ltm.to(dev).to(dt)
                tmp.append(layer.self_attn.key_proj(layer.input_layernorm(batch_ltm)))
            self.ltm_keys[idx] = torch.cat(tmp, dim=0).detach().cpu()

    def update_ltm(self, dropped_memories=None, dropped_memory_ages=None, device=None, cached_dropped_keys=None):
        for idx in range(self.L):
            self._ensure_ltm_keys(idx)
        return super().update_ltm(dropped_memories, dropped_memory_ages,
                                  device=device, cached_dropped_keys=cached_dropped_keys)

    def merge_cached_memory(self):
        tracking = self.stm_prov is not None
        if tracking:
            # Mirror update_ltm's append + prune decision per layer:
            #   rf = concat(old_rf, initial_rf) - decay; keep = rf > 0.01;
            #   prune only if len(keep) > num_ltm_blocks*num_tokens.
            cap = self.num_ltm_blocks * self.num_tokens
            new_ltm_prov, new_ltm_pos = [], []
            for idx in range(self.L):
                prov = np.concatenate([self.ltm_prov[idx], self.cached_dropped_prov[idx]])
                pos = np.concatenate([self.ltm_pos[idx], self.cached_dropped_pos[idx]])
                rf_after = np.concatenate([
                    np.asarray(self.ltm_recall_frequencies[idx], dtype=np.float64),
                    np.full(self.cached_dropped_prov[idx].shape[0],
                            float(self.initial_rf_when_moving_stm_to_ltm)),
                ]) - self.decay_frequency
                keep = np.where(rf_after > 0.01)[0]
                if len(keep) > cap:
                    prov = prov[keep]
                    pos = pos[keep]
                new_ltm_prov.append(prov)
                new_ltm_pos.append(pos)
        super().merge_cached_memory()
        if tracking:
            for idx in range(self.L):
                assert new_ltm_prov[idx].shape[0] == self.ltm[idx].shape[0], (
                    f"layer {idx}: provenance mirror diverged from update_ltm "
                    f"({new_ltm_prov[idx].shape[0]} vs {self.ltm[idx].shape[0]})")
                self.ltm_prov[idx] = new_ltm_prov[idx]
                self.ltm_pos[idx] = new_ltm_pos[idx]
                self.cached_dropped_prov[idx] = np.zeros(0, dtype=np.int64)
                self.cached_dropped_pos[idx] = np.zeros(0, dtype=np.int64)

    # ------------------------------------------------------------------
    # oracle retrieval (upper-bound gate)
    # ------------------------------------------------------------------

    def get_ltm(self, idx, hidden_states, random_retriever_length=False):
        """Generalized retrieval. Falls through to MPlus.get_ltm unless one of the
        extensions is active:
          - oracle_doc_id: rows of that doc get their score boosted above the
            sigmoid range so topk always includes them (upper-bound diagnostics).
          - pool_include_stm: candidate pool additionally contains STM rows older
            than the readable tail window (stm_pool_exclude_tail, default 2560),
            so a complete doc block can be re-presented even while its rows are
            still (partially) resident mid-STM.
          - retrieved_order='score_asc': retrieved rows ordered most-relevant-
            LAST, i.e. adjacent to the input under prefix_order='stm_first' —
            inside the only prefix region the model reliably reads facts from.
        Selection math matches MPlus.get_ltm (sigmoid(q@K.T).mean over query
        tokens, topk, tile-pad when the pool is small)."""
        if getattr(self, "retrieval_mode", "row") == "store":
            return self._store_select(idx, hidden_states, hidden_states.device, hidden_states.dtype)
        oracle = self.oracle_doc_id is not None and self.stm_prov is not None
        include_stm = getattr(self, "pool_include_stm", False) and self.stm_prov is not None
        order = getattr(self, "retrieved_order", "age")
        if not oracle and not include_stm and order == "age":
            return super().get_ltm(idx, hidden_states, random_retriever_length)

        num_ltm_tokens = self.num_ltm_tokens if hasattr(self, "num_ltm_tokens") else self.num_ltm_blocks * self.num_tokens
        self._ensure_ltm_keys(idx)
        layer = self.model.layers[idx]
        dev, dt = hidden_states.device, hidden_states.dtype

        # training hook: harvest the exact per-layer query inputs retrieval sees
        if getattr(self, "_capture_hiddens", None) is not None:
            self._capture_hiddens[idx] = hidden_states[0].detach()

        with torch.no_grad():
            pool_x = [self.ltm[idx]]
            pool_k = [self.ltm_keys[idx]]
            pool_prov = [self.ltm_prov[idx]]
            pool_pos = [self.ltm_pos[idx]]
            pool_age = [np.asarray(self.ltm_ages[idx], dtype=np.float64)]
            if self.cached_dropped_memories is not None:
                pool_x.append(self.cached_dropped_memories[idx])
                pool_k.append(self.cached_dropped_keys[idx])
                pool_prov.append(self.cached_dropped_prov[idx])
                pool_pos.append(self.cached_dropped_pos[idx])
                pool_age.append(np.asarray(self.cached_dropped_memory_ages[idx], dtype=np.float64))
            if include_stm:
                W = getattr(self, "stm_pool_exclude_tail", num_ltm_tokens)
                stm_rows = self.memory[idx][:-W] if W > 0 else self.memory[idx]
                ks = []
                for b in torch.split(stm_rows, 1024):
                    bb = b.to(dev).to(dt)
                    ks.append(layer.self_attn.key_proj(layer.input_layernorm(bb)))
                pool_x.append(stm_rows.detach())
                pool_k.append(torch.cat(ks, dim=0))
                pool_prov.append(self.stm_prov[idx][:-W] if W > 0 else self.stm_prov[idx])
                pool_pos.append(self.stm_pos[idx][:-W] if W > 0 else self.stm_pos[idx])
                ma = np.asarray(self.memory_ages[idx], dtype=np.float64)
                pool_age.append(ma[:-W] if W > 0 else ma)

            prov = np.concatenate(pool_prov)
            ages = np.concatenate(pool_age)
            K = torch.cat([k.to(dev).to(dt) for k in pool_k], dim=0)
            n_pool = K.shape[0]

            queries = layer.self_attn.query_proj(layer.input_layernorm(hidden_states[0]))
            if getattr(self, "score_reduce", "mean") == "max":
                # max over query tokens: the most-activated token ranks the row,
                # so constant chat-template tokens cannot dilute discrimination
                predictions = (queries @ K.transpose(-2, -1)).sigmoid().max(dim=0).values
            else:
                predictions = (queries @ K.transpose(-2, -1)).sigmoid().mean(dim=0)

            if getattr(self, "retrieval_mode", "row") == "block":
                pos_all = np.concatenate(pool_pos)
                return self._block_select(pool_x, prov, pos_all, predictions,
                                          num_ltm_tokens, dev, dt, idx)

            if oracle:
                forced = np.where(prov == self.oracle_doc_id)[0]
                if forced.shape[0] > 0:
                    predictions = predictions.clone()
                    predictions[torch.from_numpy(forced).to(dev)] = 2.0

            if n_pool <= num_ltm_tokens:
                sel = torch.arange(n_pool)
                while sel.shape[0] < num_ltm_tokens:
                    sel = torch.cat([torch.arange(n_pool), sel])
                sel = sel[-num_ltm_tokens:]
            else:
                sel = torch.topk(predictions, k=num_ltm_tokens).indices.cpu()

            sel_np = sel.numpy()
            bounds = np.cumsum([0] + [x.shape[0] for x in pool_x])
            rows = torch.empty((sel_np.shape[0], self.d), device=dev, dtype=dt)
            for pi, x in enumerate(pool_x):
                m = (sel_np >= bounds[pi]) & (sel_np < bounds[pi + 1])
                if m.any():
                    local = torch.from_numpy((sel_np[m] - bounds[pi]).astype(np.int64))
                    rows[torch.from_numpy(np.where(m)[0].astype(np.int64))] = x[local].to(dev).to(dt)

            sel_prov = prov[sel_np]
            sel_age = ages[sel_np]
            sel_score = predictions[sel.to(dev)].detach().float().cpu().numpy()

            if order == "score_asc":
                o = np.argsort(sel_score, kind="stable").copy()          # most relevant last
            else:
                o = np.argsort(sel_age, kind="stable")[::-1].copy()      # oldest first (M+ default)
            rows = rows[torch.from_numpy(o.astype(np.int64)).to(dev)]
            sel_prov = sel_prov[o]

            placement = getattr(self, "oracle_placement", "age")
            if oracle and placement != "age":
                rows = self._apply_placement(rows, sel_prov, placement)

            n_ltm = len(self.ltm_prov[idx])
            ltm_indices = torch.from_numpy(sel_np[sel_np < n_ltm].astype(np.int64))
        return rows.detach(), ltm_indices

    def cat_memory_and_hiddens(self, idx, hidden_states, delta_memory=None,
                               is_injection=False, cat_to_maximum_memory=False,
                               random_retriever_length=False):
        """prefix_order='ltm_first' (default) keeps M+'s original layout
        [retrieved LTM | STM | input]. 'stm_first' moves the retrieved block to
        the most-recent prefix positions: [STM | retrieved LTM | input] — where
        learnable docs lived during gnm's seq-10 training (memory is appended at
        the tail), which the stm_tail placement diagnostic showed is the only
        region the model reliably recalls facts from."""
        if getattr(self, "prefix_order", "ltm_first") == "ltm_first":
            return super().cat_memory_and_hiddens(
                idx, hidden_states, delta_memory=delta_memory, is_injection=is_injection,
                cat_to_maximum_memory=cat_to_maximum_memory,
                random_retriever_length=random_retriever_length)

        stm = self.get_stm(idx, hidden_states, delta_memory, is_injection, cat_to_maximum_memory)
        ltm_indices = None
        if (not is_injection) and (delta_memory is None or cat_to_maximum_memory):
            ltm, ltm_indices = self.get_ltm(idx, hidden_states, random_retriever_length=random_retriever_length)
            hidden_states = torch.cat([stm, ltm.unsqueeze(0), hidden_states], dim=1)
        else:
            hidden_states = torch.cat([stm, hidden_states], dim=1)
        if self.add_bos_embedding:
            hidden_states = torch.cat([self.bos_embedding[idx].unsqueeze(0).repeat(len(hidden_states), 1, 1), hidden_states], dim=1)
        return hidden_states, ltm_indices

    def _block_select(self, pool_x, prov, pos, predictions, k_total, dev, dt, idx):
        """Document-level retrieval: rank docs by aggregated row scores, then
        retrieve each top doc's COMPLETE 256-row block (rows gathered from
        wherever they live, in original within-doc order). Best-matching doc is
        placed last (adjacent to the input under prefix_order='stm_first').
        Remaining budget is padded with seed/background rows at the front.
        With the full STM in the pool, blocks are complete by row conservation."""
        block = self.num_tokens
        n_docs = k_total // block
        pred_np = predictions.detach().float().cpu().numpy()
        doc_ids = np.unique(prov[prov >= 0])
        scores = {}
        for d in doc_ids:
            rows = pred_np[prov == d]
            scores[int(d)] = float(np.sort(rows)[-32:].mean())
        if self.oracle_doc_id is not None and int(self.oracle_doc_id) in scores:
            scores[int(self.oracle_doc_id)] = float("inf")
        ranked = sorted(scores, key=scores.get, reverse=True)[:n_docs]
        if getattr(self, "_record_block_docs", None) is not None:
            self._record_block_docs[idx] = list(ranked)

        flat = []
        for d in reversed(ranked):  # best doc last => adjacent to input
            seld = np.where(prov == d)[0]
            seld = seld[np.argsort(pos[seld], kind="stable")]
            flat.append(seld)
        used = int(sum(len(s) for s in flat))
        if used < k_total:
            pad = np.where(prov == PROV_SEED)[0][: k_total - used]
            if used + pad.shape[0] < k_total:
                bg = np.where(prov == PROV_BACKGROUND)[0][: k_total - used - pad.shape[0]]
                pad = np.concatenate([pad, bg])
            flat.insert(0, pad)
        sel_np = np.concatenate(flat).astype(np.int64)
        if sel_np.shape[0] < k_total:
            # pool degenerate (e.g., heavy pruning): pad with best unused rows,
            # tiling as a last resort — prefix length must stay invariant
            unused = np.setdiff1d(np.arange(prov.shape[0]), sel_np, assume_unique=False)
            if unused.shape[0] > 0:
                order = np.argsort(pred_np[unused])[::-1]
                extra = unused[order][: k_total - sel_np.shape[0]]
                sel_np = np.concatenate([extra, sel_np])
            while sel_np.shape[0] < k_total:
                sel_np = np.concatenate([sel_np, sel_np])[: k_total] if sel_np.shape[0] else np.zeros(k_total, dtype=np.int64)
        sel_np = sel_np[-k_total:]
        assert sel_np.shape[0] == k_total, (sel_np.shape[0], k_total)

        bounds = np.cumsum([0] + [x.shape[0] for x in pool_x])
        rows = torch.empty((k_total, self.d), device=dev, dtype=dt)
        for pi, x in enumerate(pool_x):
            m_ = (sel_np >= bounds[pi]) & (sel_np < bounds[pi + 1])
            if m_.any():
                local = torch.from_numpy((sel_np[m_] - bounds[pi]).astype(np.int64))
                rows[torch.from_numpy(np.where(m_)[0].astype(np.int64))] = x[local].to(dev).to(dt)

        n_ltm = len(self.ltm_prov[idx])
        ltm_indices = torch.from_numpy(sel_np[sel_np < n_ltm])
        return rows.detach(), ltm_indices

    def _apply_placement(self, x, prov_sel, placement):
        """Reorder retrieved rows so oracle-doc rows sit at the front or back of
        the retrieved block (diagnostic for position sensitivity). 'age' keeps
        the original oldest-first ordering."""
        mask = torch.from_numpy(prov_sel == self.oracle_doc_id).to(x.device)
        if placement == "front":
            return torch.cat([x[mask], x[~mask]], dim=0)
        if placement == "back":
            return torch.cat([x[~mask], x[mask]], dim=0)
        return x


class GNMPlusChatWrapper(MPlusChatWrapper):
    """GNM weights in the M+ shell with provenance tracking, gnm-seeded LTM, and
    post-splice reset baselines. See module docstring."""

    ORACLE_DOC_ID = None  # subclass hook

    def __init__(self, cfg: MemoryllmTrainConfig):
        nn.Module.__init__(self)  # skip parent __init__ (we load a different model class)
        self.cfg = cfg
        # Same RoPE override as MPlusChatWrapper: the gnm LoRA/QKV were trained
        # under memoryllm-8b-chat's linear-scaled RoPE, not M+'s llama3 RoPE.
        config = AutoConfig.from_pretrained(cfg.model.model_path)
        memllm_cfg = AutoConfig.from_pretrained("YuWangX/memoryllm-8b-chat")
        config.rope_scaling = memllm_cfg.rope_scaling
        config.rope_theta = memllm_cfg.rope_theta
        config.max_position_embeddings = memllm_cfg.max_position_embeddings
        self.model = MPlusWithProvenance.from_pretrained(
            cfg.model.model_path, config=config,
            attn_implementation="sdpa", torch_dtype=torch.bfloat16)
        self.model = self.model.to(torch.bfloat16)
        self.model.put_ltm_to_numpy()
        self.model.oracle_doc_id = self.ORACLE_DOC_ID
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_path)
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        self.max_new_tokens = cfg.sampling.max_new_tokens

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.model.generation_config is None:
            from transformers import GenerationConfig
            self.model.generation_config = GenerationConfig()
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.tokenizer.padding_side = "left"

        self._docs_injected = 0
        self._snapshot_baselines()

    def _snapshot_baselines(self):
        # Same as parent but tolerates ltm_keys entries being None (we clear them
        # after seeding so keys are lazily recomputed with the current key_proj —
        # important once the retriever has been retrained).
        self.baseline_memory = self.model.memory.detach().clone().cpu()
        self.baseline_ltm = [t.detach().clone() for t in self.model.ltm]
        self.baseline_ltm_keys = [None if t is None else t.detach().clone() for t in self.model.ltm_keys]
        self.baseline_ltm_ages = [np.array(a, copy=True) for a in self.model.ltm_ages]
        self.baseline_ltm_recall_frequencies = [np.array(a, copy=True) for a in self.model.ltm_recall_frequencies]
        self.baseline_memory_ages = [np.array(a, copy=True) for a in self.model.memory_ages]

    def load_pretrained_memoryllm_into_mplus(self, trained_state_dict: dict):
        super().load_pretrained_memoryllm_into_mplus(trained_state_dict)

        # Seed the LTM with the STM tokens the splice truncation threw away:
        # gnm memory is [L, 12800, d], M+ STM is [L, 10240, d] and the splice
        # keeps the LAST 10240, so the first 2560 (oldest) rows per layer are
        # exactly ltm_initial_size worth of trained-gnm memory content.
        mem = trained_state_dict.get("model.memory")
        stm_len = self.model.memory.shape[1]
        if mem is not None and mem.dim() == 3 and mem.shape[1] > stm_len:
            n_seed = mem.shape[1] - stm_len
            seed = mem[:, :n_seed, :]
            for idx in range(len(self.model.ltm)):
                self.model.ltm[idx] = seed[idx].detach().clone().to(torch.bfloat16).cpu()
                self.model.ltm_keys[idx] = None  # lazily rebuilt with current key_proj
                # Mark seeds as the oldest rows so retrieved-prefix age ordering
                # keeps them in front (they predate every injected doc).
                self.model.ltm_ages[idx] = np.arange(n_seed)[::-1].astype(np.int64) + 1_000_000
                self.model.ltm_recall_frequencies[idx] = np.full(
                    n_seed, float(self.model.initial_rf_when_moving_stm_to_ltm))
            try:
                from utils.distributed import rprint
            except Exception:
                rprint = print
            rprint(f"[GNMPlusChatWrapper] seeded LTM with {n_seed} truncated gnm STM tokens per layer")

        # Unlike the parent: snapshot the POST-splice state, so reset_memory
        # restores gnm-all's STM and the gnm-seeded LTM (not mplus-8b's memory).
        self._snapshot_baselines()
        self.model.reset_provenance()

    def memorize(self, *, learning_instruction: str, document: str):
        # Mirror the parent's update condition to keep doc-id assignment in sync
        # with whether an actual memory update happens.
        will_update = not (torch.is_grad_enabled() and self.model.training)
        if will_update:
            self.model._current_doc_id = self._docs_injected
        try:
            super().memorize(learning_instruction=learning_instruction, document=document)
        finally:
            self.model._current_doc_id = PROV_BACKGROUND
            if will_update:
                self._docs_injected += 1

    def reset_memory(self):
        self.model.memory.data.copy_(self.baseline_memory.to(self.model.memory.device))
        for idx in range(len(self.model.ltm)):
            self.model.ltm[idx] = self.baseline_ltm[idx].clone()
            self.model.ltm_keys[idx] = (
                None if self.baseline_ltm_keys[idx] is None else self.baseline_ltm_keys[idx].clone())
            self.model.ltm_ages[idx] = np.array(self.baseline_ltm_ages[idx], copy=True)
            self.model.ltm_recall_frequencies[idx] = np.array(self.baseline_ltm_recall_frequencies[idx], copy=True)
            self.model.memory_ages[idx] = np.array(self.baseline_memory_ages[idx], copy=True)
        self.model.cached_dropped_memories = None
        self.model.cached_dropped_memory_ages = None
        self.model.cached_dropped_keys = None
        self.model.update_step = 0
        self.model.past_key_values = None
        self.model.reset_provenance()
        self._docs_injected = 0
        self.last_memory_update = None


class GNMPlusPlacedRetrievalWrapper(GNMPlusChatWrapper):
    """Placement-corrected retrieval: prefix [STM | retrieved | input] with
    retrieved rows ordered most-relevant-last (adjacent to the input, inside the
    readable tail window the placement diagnostics identified), and the pool
    including older STM rows so complete doc blocks are always re-presentable."""

    PREFIX_ORDER = "stm_first"
    RETRIEVED_ORDER = "score_asc"
    POOL_INCLUDE_STM = True
    SCORE_REDUCE = "max"
    # 0: ALL STM rows are retrieval candidates. In the stm_first layout the
    # retrieved block displaces the STM from the readable tail window, so even
    # fresh docs are only readable if retrieval re-presents them (the pool
    # diagnostic's recent_swap column showed this directly).
    STM_POOL_EXCLUDE_TAIL = 0

    def __init__(self, cfg: MemoryllmTrainConfig):
        super().__init__(cfg)
        self.model.prefix_order = self.PREFIX_ORDER
        self.model.retrieved_order = self.RETRIEVED_ORDER
        self.model.pool_include_stm = self.POOL_INCLUDE_STM
        self.model.stm_pool_exclude_tail = self.STM_POOL_EXCLUDE_TAIL
        self.model.score_reduce = self.SCORE_REDUCE


class GNMPlusBlockRetrievalWrapper(GNMPlusPlacedRetrievalWrapper):
    """Document-block retrieval on top of the V2 layout: rank docs by aggregated
    row scores and retrieve complete 256-row provenance blocks of the top 10,
    best doc adjacent to the input. Complete blocks are what the placement
    diagnostics showed the model actually reads."""

    def __init__(self, cfg: MemoryllmTrainConfig):
        super().__init__(cfg)
        self.model.retrieval_mode = "block"
        # Disable M+'s recall-frequency decay: without the encoder retriever
        # there is nothing to refresh frequencies, so stock bookkeeping deletes
        # every LTM row ~10 merges (~600 injections) after it arrives. With
        # decay 0 the LTM is lossless at any sequence length.
        self.model.decay_frequency = 0


class GNMLongHorizonWrapper(GNMPlusBlockRetrievalWrapper):
    """Pure-LTM-store architecture: STM frozen as static
    background forever; every document diverted to an append-only store of M+ blocks;
    the ONLY read path is trained top-10 block retrieval into the 2560-token
    tail window. Reproduces the seq-10 training distribution at every length."""

    def __init__(self, cfg: MemoryllmTrainConfig):
        super().__init__(cfg)
        self.model.retrieval_mode = "store"

    def load_pretrained_memoryllm_into_mplus(self, trained_state_dict):
        super().load_pretrained_memoryllm_into_mplus(trained_state_dict)
        # static pad = the truncated 2560 oldest gnm rows (background content)
        mem = trained_state_dict.get("model.memory")
        stm_len = self.model.memory.shape[1]
        if mem is not None and mem.shape[1] > stm_len:
            self.model.static_pad = mem[:, : mem.shape[1] - stm_len, :].detach().clone().to(torch.bfloat16)
        else:
            self.model.static_pad = self.model.memory[:, :self.model.num_ltm_blocks * self.model.num_tokens, :].detach().clone()
        self.model.reset_store()

    def to(self, *args, **kwargs):
        out = super().to(*args, **kwargs)
        if self.model.static_pad is not None:
            self.model.static_pad = self.model.static_pad.to(self.model.memory.device)
        return out

    def memorize(self, *, learning_instruction: str, document: str):
        memory_context = f"<|learning_instruction_start|>{learning_instruction}<|learning_instruction_end|><|document_start|>{document}<|document_end|>"
        ctx_ids = self.tokenizer(memory_context, return_tensors='pt', add_special_tokens=False).input_ids.to(self.model.device)
        delta = self.model.inject_memory(ctx_ids, update_memory=False, use_retriever=False)
        d = delta[0] if delta.dim() == 4 else delta
        self.model.store_add(d, len(self.model.store_ids))

    RAWCOS_TWO_PASS = True

    def generate(self, input_ids, attention_mask=None, *, max_new_tokens=12, **gen_kwargs):
        if not (self.RAWCOS_TWO_PASS and self.model.store_ids):
            return super().generate(input_ids, attention_mask, max_new_tokens=max_new_tokens, **gen_kwargs)
        # pass 1: capture probe hiddens under whatever retrieval, ignore output
        self.model._capture_hiddens = {}
        with torch.inference_mode():
            self.model(input_ids=input_ids, attention_mask=attention_mask)
        cap = self.model._capture_hiddens
        self.model._capture_hiddens = None
        # rank in raw cosine space, pin the choice for every layer in pass 2
        self.model._forced_docs = self.model.rank_docs_rawcos(cap)
        try:
            return super().generate(input_ids, attention_mask, max_new_tokens=max_new_tokens, **gen_kwargs)
        finally:
            self.model._forced_docs = None

    def reset_memory(self):
        # STM is static; only the store resets
        self.model.reset_store()
        self.model.past_key_values = None
        self.last_memory_update = None










class WindowedICLChatWrapper(ICLChatWrapper):
    """Context-window ICL baseline for long rollouts: keeps only the most recent
    (instruction, document) pairs whose rendered system prompt fits the token
    budget — i.e., what in-context learning can actually retain when the
    document stream exceeds the context window. Without this cap the stock ICL
    wrapper builds an unbounded prompt and breaks past ~20 documents."""

    TOKEN_BUDGET = 7400  # leaves room for the probe turn within Llama-3's 8k

    def _get_system_prompt(self) -> str:
        saved_lis, saved_docs = self.remembered_lis, self.remembered_docs
        try:
            lo, hi, best = 1, len(saved_docs), 1
            while lo <= hi:
                mid = (lo + hi) // 2
                self.remembered_lis = saved_lis[-mid:]
                self.remembered_docs = saved_docs[-mid:]
                prompt = super()._get_system_prompt()
                n = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
                if n <= self.TOKEN_BUDGET:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            self.remembered_lis = saved_lis[-best:]
            self.remembered_docs = saved_docs[-best:]
            return super()._get_system_prompt()
        finally:
            self.remembered_lis, self.remembered_docs = saved_lis, saved_docs


def register_wrappers(registry: dict):
    registry["gnm_long_horizon"] = {
        "wrapper": GNMLongHorizonWrapper,
        "hf_id": "YuWangX/mplus-8b",
        "tokenizer_id": "YuWangX/memoryllm-8b-chat",
    }
    registry["llama3_8b_instruct_windowed"] = {
        "wrapper": WindowedICLChatWrapper,
        "hf_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "tokenizer_id": "meta-llama/Meta-Llama-3-8B-Instruct",
    }


# Self-register into gnm's wrapper registry. This works in both import orders:
# - `import gnm` first: gnm's bottom hook imports this module, and by the time we
#   get here gnm._REGISTRY exists (defined above the hook).
# - `import gnm_mplus_retrieval` first: the `from gnm import ...` above fully
#   executes gnm (whose hook sees this partially-initialized module and skips),
#   then we register here.
import gnm as _gnm  # noqa: E402
register_wrappers(_gnm._REGISTRY)
