"""One-shot script: build the 200-example dataset for fig6_memory_selectivity_personamem.ipynb.

Pulls non-collision probes from `data/personamem/val_ood.json`,
samples 200 held-out target probes + 200 non-held-out distractor probes, pairs them, and
regenerates each merged document via the same `make_doc` pipeline used at training time.

Output: data/analysis/personamem_memory_analysis.json
"""

import json
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "data_pipelines" / "personamem" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_multichoice import (
    CONVERSATION_OPENINGS,
    CONVERSATION_FOLLOWUPS,
    CONVERSATION_CLOSINGS,
    INSTRUCTION_TEMPLATES_INCLUSIVE,
    make_doc,
)
from topic_descriptors import topic_with_hint

HELDOUT_TOPICS = {
    "Sports", "Music", "Travel", "Fashion", "Technology",
    "Photography", "Gardening", "Education", "Science", "Pets",
}

SRC = REPO / "data" / "personamem" / "val_ood.json"
OUT = REPO / "data" / "personamem_memory_analysis.json"

USER_RE = re.compile(r"^User:\s*(.*)$", re.M)


def main():
    rng = random.Random(42)

    print(f"Loading {SRC}")
    src_docs = json.load(open(SRC))
    print(f"  {len(src_docs)} source docs")

    # Step 1: flatten probes, recover pref_first_person from each doc's User: lines.
    all_probes = []
    bad_docs = 0
    for ex in src_docs:
        user_lines = USER_RE.findall(ex["document"])
        pairs = ex.get("input_output_pairs") or []
        if len(user_lines) != len(pairs):
            bad_docs += 1
            continue
        for ul, pp in zip(user_lines, pairs):
            pp = dict(pp)
            pp["pref_first_person"] = ul.strip()
            all_probes.append(pp)
    print(f"  flattened {len(all_probes)} probes (bad docs: {bad_docs})")

    # Step 2: filter to non-collision (correct_letter != memoryllm_baseline_letter).
    def is_non_collision(p):
        cl = p.get("correct_letter")
        bl = p.get("memoryllm_baseline_letter")
        return cl is not None and bl is not None and cl != bl

    non_coll = [p for p in all_probes if is_non_collision(p)]
    print(f"  non-collision probes: {len(non_coll)} / {len(all_probes)}")

    # Step 3: partition + dedupe on source_row_idx so the same underlying preference
    # row never appears twice (the all-probes pool repeats rows across docs).
    def dedupe(probes):
        seen, out = set(), []
        for p in probes:
            ridx = p.get("source_row_idx")
            if ridx in seen:
                continue
            seen.add(ridx)
            out.append(p)
        return out

    held_pool = dedupe([p for p in non_coll if p.get("topic") in HELDOUT_TOPICS])
    nonheld_pool = dedupe([p for p in non_coll
                           if p.get("topic") not in HELDOUT_TOPICS
                           and p.get("topic") != "sensitive_info"])
    print(f"  held-out target candidates (deduped): {len(held_pool)}")
    print(f"  non-held-out distractor candidates (deduped, excl sensitive_info): {len(nonheld_pool)}")
    assert len(held_pool) >= 200, f"not enough held-out probes: {len(held_pool)}"
    assert len(nonheld_pool) >= 200, f"not enough non-held-out probes: {len(nonheld_pool)}"

    # Step 4: sample without replacement, pair.
    targets = rng.sample(held_pool, 200)
    distractors = rng.sample(nonheld_pool, 200)
    assert len({t["source_row_idx"] for t in targets}) == 200
    assert len({d["source_row_idx"] for d in distractors}) == 200

    # Step 5+6+7: regenerate docs and instructions.
    examples = []
    skipped_substr = 0
    for i, (t, d) in enumerate(zip(targets, distractors)):
        # Coin-flip the order
        prefs = [(t["topic"], t["pref_first_person"]),
                 (d["topic"], d["pref_first_person"])]
        if rng.random() < 0.5:
            prefs = list(reversed(prefs))
        document = make_doc(prefs, rng)

        learning_instruction = rng.choice(INSTRUCTION_TEMPLATES_INCLUSIVE).format(
            t=topic_with_hint(t["topic"], rng)
        )

        # Strip trailing sentence punctuation: the BPE tokenizer greedily merges
        # ".\n" into a single token when in-context, so the standalone tokenization
        # (ending in ".") never matches contiguously. Counterfact's answers had no
        # trailing punctuation and didn't hit this. The alignment direction is the
        # same with or without the period token.
        target_answer = t["pref_first_person"].rstrip(".!? 。！？")
        distractor_answer = d["pref_first_person"].rstrip(".!? 。！？")

        # Sanity: both spans must be substrings of the regenerated document.
        if target_answer not in document or distractor_answer not in document:
            skipped_substr += 1
            continue

        examples.append({
            "learning_instruction": learning_instruction,
            "document": document,
            "target_answer": target_answer,
            "distractor_answer": distractor_answer,
            "target_topic": t["topic"],
            "distractor_topic": d["topic"],
            "target_source_row_idx": t["source_row_idx"],
            "distractor_source_row_idx": d["source_row_idx"],
        })

    print(f"  built {len(examples)} examples (skipped {skipped_substr} for substring mismatch)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(examples, f, indent=2)
    print(f"Wrote {OUT}")

    # Print one example as a smoke-test.
    print("\n=== SAMPLE EXAMPLE 0 ===")
    e = examples[0]
    print(f"target_topic: {e['target_topic']}  /  distractor_topic: {e['distractor_topic']}")
    print(f"learning_instruction: {e['learning_instruction']}")
    print(f"target_answer: {e['target_answer']}")
    print(f"distractor_answer: {e['distractor_answer']}")
    print(f"document:\n{e['document']}")


if __name__ == "__main__":
    main()
