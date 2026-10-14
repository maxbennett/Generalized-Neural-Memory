"""Build the `_all_probes` datasets by copying the pre-rebuild artifact files
(which contain every probe — collisions and non-collisions alike) into a new
output directory.

Differences vs the existing `data/personamem/train_gnm/` files:
- No collision filter applied (every probe with a populated baseline_letter is kept).
- No re-bundling (docs preserved as build_multichoice.py emitted them).
- `baseline_letter` is renamed to `memoryllm_baseline_letter` per probe.

`llama3_baseline_letter` is left None here; populated by
`precompute_baseline_letters_llama3.py` in a separate step.
"""

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ART = REPO / "data_pipelines/personamem/artifacts"
OUT = REPO / "data/personamem/train_icl"

SOURCE_TARGET = [
    (ART / "train_mc_0.json", OUT / "train_0.json"),
    (ART / "train_mc_1.json", OUT / "train_1.json"),
    (ART / "train_mc_2.json", OUT / "train_2.json"),
    (ART / "train_mc_3.json", OUT / "train_3.json"),
    (ART / "train_mc_4.json", OUT / "train_4.json"),
    (ART / "held_out_personamem_mc_seen_topics.json",   OUT / "dev.json"),
    (ART / "held_out_personamem_mc_unseen_topics.json", REPO / "data/personamem/test.json"),
]


def transform_doc(doc: dict) -> dict:
    new_pairs = []
    for p in doc.get("input_output_pairs", []):
        np = dict(p)
        if "baseline_letter" in np:
            np["memoryllm_baseline_letter"] = np.pop("baseline_letter")
        else:
            np.setdefault("memoryllm_baseline_letter", None)
        # placeholder; populated by precompute_baseline_letters_llama3.py
        np.setdefault("llama3_baseline_letter", None)
        new_pairs.append(np)
    new_doc = dict(doc)
    new_doc["input_output_pairs"] = new_pairs
    new_meta = dict(new_doc.get("metadata") or {})
    new_meta["all_probes"] = True
    new_doc["metadata"] = new_meta
    return new_doc


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"src dir: {ART}")
    print(f"out dir: {OUT}")
    print()
    print(f"{'src':<60} {'->':<3} {'dst':<35} {'n_docs':>7} {'n_pairs':>8}  {'n_baseline_None':>16}")
    print("-" * 140)
    for src, dst in SOURCE_TARGET:
        if not src.exists():
            print(f"  [skip] missing source: {src}")
            continue
        data = json.load(open(src))
        new_data = [transform_doc(d) for d in data]
        n_pairs = sum(len(d["input_output_pairs"]) for d in new_data)
        n_none = sum(1 for d in new_data for p in d["input_output_pairs"]
                     if p["memoryllm_baseline_letter"] is None)
        with open(dst, "w") as f:
            json.dump(new_data, f)
        print(f"{str(src.name):<60} ->  {str(dst.name):<35} {len(new_data):>7} {n_pairs:>8}  {n_none:>16}")
    print()
    print(f"Done. {len(SOURCE_TARGET)} files written to {OUT}.")


if __name__ == "__main__":
    main()
