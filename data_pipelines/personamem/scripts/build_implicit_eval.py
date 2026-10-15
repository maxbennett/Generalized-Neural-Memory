"""Build the implicit-instruction variant of the PersonaMem Val-OOD eval file.

Copies data/personamem/val_ood.json and rewrites ONLY the top-level
`learning_instruction` field of each example (documents, probes, precomputed
baseline letters, metadata, and example order are untouched, so the any_trained
collision filter and probe-level pairing with the paper run stay valid). Each
example's explicit instruction is replaced by one of the hand-authored
intent-based utterances in implicit_instructions.py, assigned round-robin per
topic (the i-th example of a topic gets variant i % 3).

Output: data/personamem/implicit_eval/val_ood_implicit.json
"""

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

from implicit_instructions import IMPLICIT_INSTRUCTIONS

REPO = Path(__file__).resolve().parents[3]

# Must mirror build_multichoice.py INSTRUCTION_TEMPLATES_INCLUSIVE exactly.
TEMPLATES = [
    "Pay attention only to preferences about {t}.",
    "Only remember preferences related to {t}.",
    "Focus on the user's preferences for {t} and disregard all other preferences.",
    "Of the preferences listed, only learn the ones about {t}.",
    "Ignore everything except preferences about {t}.",
    "Track only {t} preferences from this conversation.",
]


def match_template(li: str) -> str:
    """Return the template whose prefix/suffix bracket the stored LI."""
    matches = []
    for tpl in TEMPLATES:
        prefix, suffix = tpl.split("{t}")
        if li.startswith(prefix) and li.endswith(suffix) and len(li) > len(prefix) + len(suffix):
            matches.append(tpl)
    if len(matches) != 1:
        raise ValueError(f"expected exactly 1 template match, got {matches!r} for LI: {li!r}")
    return matches[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path,
                    default=REPO / "data/personamem/val_ood.json")
    ap.add_argument("--outdir", type=Path,
                    default=REPO / "data/personamem/implicit_eval")
    args = ap.parse_args()
    bank = IMPLICIT_INSTRUCTIONS

    data = json.load(open(args.source))
    print(f"[build-implicit] {args.source.name}: {len(data)} examples")

    implicit = copy.deepcopy(data)
    per_topic_idx = defaultdict(int)
    variant_counts = Counter()

    for i, ex in enumerate(data):
        target = ex["metadata"]["learn_topics"][0]
        if target not in bank:
            raise KeyError(f"example {i}: no implicit instructions for topic {target!r}")

        k = per_topic_idx[target]
        per_topic_idx[target] += 1
        variant = k % len(bank[target])
        implicit[i]["learning_instruction"] = bank[target][variant]
        variant_counts[(target, variant)] += 1
        match_template(ex["learning_instruction"])   # every source LI must be one of the 6 templates

    # Validate: everything except learning_instruction must be byte-identical.
    for orig, new in zip(data, implicit):
        o = {k: v for k, v in orig.items() if k != "learning_instruction"}
        n = {k: v for k, v in new.items() if k != "learning_instruction"}
        if o != n:
            raise AssertionError("non-LI field diverged")
    n_changed = sum(o["learning_instruction"] != n["learning_instruction"]
                    for o, n in zip(data, implicit))
    print(f"[build-implicit] {n_changed}/{len(data)} LIs rewritten, all other fields identical")

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_implicit = args.outdir / "val_ood_implicit.json"
    json.dump(implicit, open(out_implicit, "w"))
    print(f"[build-implicit] wrote {out_implicit}")

    print("\n[build-implicit] per-topic variant usage:")
    for (topic, variant), c in sorted(variant_counts.items()):
        print(f"    {topic:<12} v{variant}: {c}")

    print("\n[build-implicit] samples (original -> implicit):")
    seen = set()
    for i, ex in enumerate(data):
        t = ex["metadata"]["learn_topics"][0]
        if t in seen:
            continue
        seen.add(t)
        print(f"  [{t}]")
        print(f"    orig:     {ex['learning_instruction']}")
        print(f"    implicit: {implicit[i]['learning_instruction']}")


if __name__ == "__main__":
    main()
