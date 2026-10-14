"""Build implicit-instruction variants of the PersonaMem MC final-eval file.

Copies data/personamem/test.json and
rewrites ONLY the top-level `learning_instruction` field of each example
(documents, probes, precomputed baseline letters, metadata, and example order
are untouched, so the any_trained collision filter and probe-level pairing
with the paper run stay valid). Two conditions:

  - explicit_bare: same template as the original example, but the rendered
    topic-with-descriptor is replaced by the bare lowercase topic name.
    Isolates the descriptor's contribution from implicitness.
  - implicit: hand-authored intent-based instructions (implicit_instructions.py),
    assigned round-robin per topic (i-th example of a topic gets variant i % 3).

Outputs to data/personamem/implicit_eval/.
"""

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

from implicit_instructions import IMPLICIT_INSTRUCTIONS, IMPLICIT_INSTRUCTIONS_V2, IMPLICIT_INSTRUCTIONS_V3

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

BARE_NAMES = {t: t.lower() for t in IMPLICIT_INSTRUCTIONS}


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
                    default=REPO / "data/personamem/test.json")
    ap.add_argument("--outdir", type=Path,
                    default=REPO / "data/personamem/implicit_eval")
    ap.add_argument("--bank", choices=["v1", "v2", "v3"], default="v1",
                    help="v2 = clarity rewrites for topics that scored <0.80 in the v1 run")
    args = ap.parse_args()
    bank = {"v1": IMPLICIT_INSTRUCTIONS, "v2": IMPLICIT_INSTRUCTIONS_V2, "v3": IMPLICIT_INSTRUCTIONS_V3}[args.bank]
    suffix = {"v1": "", "v2": "_v2", "v3": "_v3"}[args.bank]

    data = json.load(open(args.source))
    print(f"[build-implicit] {args.source.name}: {len(data)} examples")

    implicit = copy.deepcopy(data)
    bare = copy.deepcopy(data)
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

        tpl = match_template(ex["learning_instruction"])
        bare[i]["learning_instruction"] = tpl.format(t=BARE_NAMES[target])

    # Validate: everything except learning_instruction must be byte-identical.
    for name, variant_data in [("implicit", implicit), ("explicit_bare", bare)]:
        for orig, new in zip(data, variant_data):
            o = {k: v for k, v in orig.items() if k != "learning_instruction"}
            n = {k: v for k, v in new.items() if k != "learning_instruction"}
            if o != n:
                raise AssertionError(f"{name}: non-LI field diverged")
        n_changed = sum(o["learning_instruction"] != n["learning_instruction"]
                        for o, n in zip(data, variant_data))
        print(f"[build-implicit] {name}: {n_changed}/{len(data)} LIs rewritten, all other fields identical")

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_implicit = args.outdir / f"val_unseen_topics_implicit{suffix}.json"
    out_bare = args.outdir / "val_unseen_topics_explicit_bare.json"
    json.dump(implicit, open(out_implicit, "w"))
    json.dump(bare, open(out_bare, "w"))
    print(f"[build-implicit] wrote {out_implicit}")
    print(f"[build-implicit] wrote {out_bare}")

    print("\n[build-implicit] per-topic variant usage:")
    for (topic, variant), c in sorted(variant_counts.items()):
        print(f"    {topic:<12} v{variant}: {c}")

    print("\n[build-implicit] samples (original -> bare -> implicit):")
    seen = set()
    for i, ex in enumerate(data):
        t = ex["metadata"]["learn_topics"][0]
        if t in seen:
            continue
        seen.add(t)
        print(f"  [{t}]")
        print(f"    orig:     {ex['learning_instruction']}")
        print(f"    bare:     {bare[i]['learning_instruction']}")
        print(f"    implicit: {implicit[i]['learning_instruction']}")


if __name__ == "__main__":
    main()
