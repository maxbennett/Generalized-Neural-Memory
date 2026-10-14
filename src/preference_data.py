"""PreferenceGNMDataset — thin wrapper over the merged JSON files for preference-selective training.

Returns the same dict shape as DocDataset.__getitem__ so docs_collate_fn works unchanged.
Each input_output_pair carries `topic`, `learn_or_ignore`, `pref_type`, etc. for analysis.

Optional training-only augmentation (`randomize_li=True`):
  At every __getitem__ call, randomly pick a NEW learn target from
  `metadata["topics_in_doc"]`, re-render the inclusive learning instruction
  ("only learn topic X"), and relabel each probe's `learn_or_ignore` accordingly.
  This breaks the static (doc, LI) pairing so the model can't memorize
  "doc D → preference for X" across epochs and is forced to attend to the LI
  to decide what to learn.

  All current data is INCLUSIVE only (verified across train/val/prefeval); we
  do not sample exclusive/learn-all modes. Targets are chosen uniformly from
  the topics in the doc.

  Reproducibility:
    The dataset is constructed fresh each epoch with `seed=ds_seed` (where
    ds_seed = cfg.seed * 1_000_003 + epoch * len(train_paths) + ds_idx in
    the trainer). The per-instance RNG is seeded from `(seed XOR 0xA5A5A5A5)`,
    and DataLoader uses num_workers=0 (main process), so __getitem__ calls
    consume RNG state in a deterministic order. Same (cfg.seed, epoch, ds_idx)
    → identical augmented batches. The augmentation completes BEFORE the
    item leaves the dataset, so by the time the trainer reads the batch the
    learn/ignore labels (including for future-pull-back probes) are fixed.

  This must be OFF for validation so val_loss / heatmaps are computed against
  a fixed pairing (otherwise val metrics drift epoch-to-epoch from RNG).
"""

import json
import random
import sys
from pathlib import Path
from torch.utils.data import Dataset

# Pull templates from the data-pipeline module. Lazy sys.path insert so the
# package layout doesn't need __init__.py files.
_INSTRUCTIONS_PATH = (Path(__file__).resolve().parents[1]
                       / "data_pipelines" / "personamem" / "scripts")
if str(_INSTRUCTIONS_PATH) not in sys.path:
    sys.path.insert(0, str(_INSTRUCTIONS_PATH))
from li_templates import render_inclusive  # noqa: E402
from build_multichoice import INSTRUCTION_TEMPLATES_INCLUSIVE  # noqa: E402
from topic_descriptors import topic_with_hint  # noqa: E402


def _render_li_with_descriptor(target: str, rng: random.Random) -> str:
    """LI render path: pick a random template from the 6-template set used
    at data-gen time, format with `topic_with_hint(target, rng=None)` so the
    descriptor is always idx-0 (canonical). Aligns runtime training-time LIs
    with the LI format stored in val_*.json."""
    template = rng.choice(INSTRUCTION_TEMPLATES_INCLUSIVE)
    return template.format(t=topic_with_hint(target, rng=None))


def _resample_li_and_relabel(ex: dict, rng: random.Random,
                              freeze_descriptor: bool = False) -> dict:
    """Return a NEW example dict with a freshly-sampled inclusive learn-target
    and relabeled probes. Source `ex` is not mutated.

    `freeze_descriptor`: when True, render LI via INSTRUCTION_TEMPLATES_INCLUSIVE
    + topic_with_hint(rng=None) instead of legacy bare-topic render_inclusive."""
    topics = ex.get("metadata", {}).get("topics_in_doc") or []
    pairs = ex.get("input_output_pairs", []) or []
    if not topics or not pairs:
        # Fall through to the original LI/labels if metadata is missing.
        return {
            "document": ex["document"],
            "learning_instruction": ex["learning_instruction"],
            "input_output_pairs": pairs,
            "metadata": ex.get("metadata", {}),
        }

    target = rng.choice(topics)
    new_li = (_render_li_with_descriptor(target, rng) if freeze_descriptor
              else render_inclusive(target, rng=rng))
    new_meta = dict(ex.get("metadata") or {})
    new_meta["instruction_mode"] = "inclusive"
    new_meta["learn_topics"] = [target]
    new_meta["ignore_topics"] = [t for t in topics if t != target]
    new_meta["randomized_li"] = True

    # Relabel: probes whose topic matches the new target become "learn",
    # everything else becomes "ignore". The probe's correct_letter
    # (preference-aligned) and memoryllm_baseline_letter are unchanged — those are
    # properties of the question/options, not of which topic is targeted.
    new_pairs = [
        {**p, "learn_or_ignore": ("learn" if p.get("topic") == target else "ignore")}
        for p in pairs
    ]

    return {
        "document": ex["document"],
        "learning_instruction": new_li,
        "input_output_pairs": new_pairs,
        "metadata": new_meta,
    }


class PreferenceGNMDataset(Dataset):
    def __init__(self, data_path: str, shuffle: bool = False, limit: int | None = None,
                 seed: int | None = None, randomize_li: bool = False,
                 freeze_descriptor: bool = False):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        if shuffle:
            # Explicit local RNG so different (epoch, dataset_idx) combos always get
            # different subsets of the file rather than depending on global RNG state.
            rng = random.Random(seed) if seed is not None else random
            rng.shuffle(self.data)
        if limit is not None:
            self.data = self.data[:limit]
        self.randomize_li = randomize_li
        self.freeze_descriptor = freeze_descriptor
        # Per-instance RNG for LI randomization. Seeded deterministically so same
        # (data_path, seed) always reproduces the same augmented sequence given
        # the same iteration order (which is itself deterministic via DistributedSampler.set_epoch).
        self._rng = random.Random((seed if seed is not None else 0) ^ 0xA5A5A5A5)
        print(f"[PreferenceGNMDataset] loaded {len(self.data)} examples from {data_path}"
              + (" (LI randomization ON)" if randomize_li else "")
              + (" (freeze_descriptor=True: idx-0 desc + 6-template set)" if freeze_descriptor else ""))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        ex = self.data[idx]
        if self.randomize_li:
            return _resample_li_and_relabel(ex, self._rng,
                                              freeze_descriptor=self.freeze_descriptor)
        return {
            "document": ex["document"],
            "learning_instruction": ex["learning_instruction"],
            "input_output_pairs": ex["input_output_pairs"],
            "metadata": ex["metadata"],
        }
