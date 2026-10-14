"""Generate merged training/eval documents for preference-selective memory.

Per training example:
- Pick a persona
- Pick N in {2,3,4} of that persona's rows with distinct `topic_preference`s
- Sample F in {5,10,15} filler chunks from the persona's chat history (avoiding window overlap)
- Concatenate windows + fillers in random order with separators
- Pick instruction mode (inclusive/exclusive)
- Build per-row input_output_pairs with target_output resolved per learn/ignore label
  (using precomputed teacher targets if available; falls back to placeholder targets if not)

Outputs:
  train_0.json ... train_4.json   (5 variants from PM-train, seeds 42-46)
  val.json                         (1 variant from PM-val, seed 100)
  held_out_personamem.json         (1 variant from PM-benchmark, seed 200)
  held_out_prefeval.json           (1 variant from PrefEval, seed 300, grouped by topic)
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from li_templates import render_inclusive, render_exclusive, render_learn_all


SEPARATOR = "\n\n--- (later in conversation) ---\n\n"
FILLER_CHUNK_TURN_COUNT = 4          # ~500 tokens at ~120 tokens/turn (only used if FILLER_F_CHOICES != [0])
FILLER_F_CHOICES = [0]               # no filler — the ignore-topic windows ARE the in-doc distractors
N_CHOICES = [2, 3]                   # capped at 3 to stay under MemoryLLM-chat's 8k window
EXAMPLES_PER_PERSONA_TRAIN = 8       # tunable; with 800 personas → ~6.4k merged examples per variant
EXAMPLES_PER_PERSONA_VAL = 3
EXAMPLES_PER_PERSONA_BENCHMARK = 5
EXAMPLES_PER_TOPIC_PREFEVAL = 4

MAX_CHARS_PER_WINDOW = 6000          # ~1,500 tokens; truncates long windows to fit the 8k budget
MAX_DOC_CHARS = 22000                # ~5,500 tokens leaves room for instruction + chat template + query in 8k

# Topics held out from learning instructions during training. The model never sees these
# topic names as `target_topic_for_instruction` (in inclusive/exclusive mode) during train.
# At eval time, we generate a separate file (`held_out_personamem_unseen_topics.json`) that
# FORCES the target topic to be in this set, testing whether the model generalizes the
# learning-instruction format to novel topic names. Mirrors counterfact's holdout pattern.
HELDOUT_TOPICS = {
    "Sports", "Music", "Travel", "Fashion", "Technology",
    "Photography", "Gardening", "Education", "Science", "Pets",
}


def turns_to_text(turns: list[dict]) -> str:
    parts = []
    for t in turns:
        role = t.get("role", "user")
        content = t.get("content", "").strip()
        parts.append(f"[{role}]: {content}")
    return "\n".join(parts)


def load_chat_history(dataset_root: Path, link: str) -> list[dict]:
    """Load chat-history-32k JSON. Returns the chat list, or [] on failure."""
    if not link:
        return []
    path = dataset_root / link
    if not path.exists():
        return []
    try:
        with open(path) as f:
            obj = json.load(f)
    except Exception:
        return []
    ch = obj.get("chat_history") if isinstance(obj, dict) else obj
    return ch if isinstance(ch, list) else []


def find_window_indices(chat_history: list[dict], window_turns: list[dict]) -> tuple[int, int] | None:
    """Locate the [start, end] of `window_turns` inside `chat_history` by content match
    on the first user turn within window_turns. Returns None if not found.
    """
    if not chat_history or not window_turns:
        return None
    first = next((t for t in window_turns if t.get("role") == "user"), None)
    if not first or not first.get("content"):
        return None
    needle = first["content"].strip()[:60]
    if not needle:
        return None
    for i, t in enumerate(chat_history):
        c = t.get("content", "")
        if isinstance(c, str) and needle in c[: max(200, len(needle) + 50)]:
            start = i
            end = min(len(chat_history), start + len(window_turns))
            return (start, end)
    return None


def sample_filler_chunks(chat_history: list[dict], excluded_ranges: list[tuple[int, int]],
                         f: int, rng: random.Random) -> list[list[dict]]:
    """Sample `f` non-overlapping ~FILLER_CHUNK_TURN_COUNT-turn slices that avoid `excluded_ranges`."""
    if not chat_history or f <= 0:
        return []
    n = len(chat_history)
    excluded_ranges = sorted(excluded_ranges)
    valid_starts = []
    for s in range(0, n - FILLER_CHUNK_TURN_COUNT, max(1, FILLER_CHUNK_TURN_COUNT // 2)):
        end = s + FILLER_CHUNK_TURN_COUNT
        overlap = any(not (end <= ex_lo or s >= ex_hi) for ex_lo, ex_hi in excluded_ranges)
        if not overlap:
            valid_starts.append(s)
    rng.shuffle(valid_starts)
    chunks = []
    used_ranges = list(excluded_ranges)
    for s in valid_starts:
        if len(chunks) >= f:
            break
        end = s + FILLER_CHUNK_TURN_COUNT
        if any(not (end <= ex_lo or s >= ex_hi) for ex_lo, ex_hi in used_ranges):
            continue
        chunks.append(chat_history[s:end])
        used_ranges.append((s, end))
    return chunks


def make_merged_example(persona_records: list[dict],
                         chat_history: list[dict],
                         dataset_root: Path,
                         rng: random.Random,
                         *, allow_filler: bool, max_n: int = 3,
                         target_topic_holdout_mode: str = "no_heldout") -> dict | None:
    """target_topic_holdout_mode:
        "no_heldout"    — target topic in instruction must NOT be in HELDOUT_TOPICS (used for train/val/seen-eval)
        "force_heldout" — target topic MUST be in HELDOUT_TOPICS (used for unseen-topics eval; learn_all mode is skipped)
        "any"           — no restriction
    """
    """Build one merged training example from a list of records that all share a persona_id.

    persona_records: filtered to records eligible for this persona (e.g., source==personamem).
    chat_history: pre-loaded chat-history list for this persona (or [] if unavailable, no filler).
    """
    # Distinct topics — filter according to holdout mode so held-out topics never appear
    # in training/val/seen-eval documents at all (matches counterfact's strict holdout).
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for r in persona_records:
        by_topic[r["topic"]].append(r)
    if target_topic_holdout_mode == "no_heldout":
        topics = [t for t, recs in by_topic.items() if recs and t not in HELDOUT_TOPICS]
    else:
        # "force_heldout" or "any" — include all topics; the instruction-mode logic below
        # will enforce that target_topic IS held-out for force_heldout.
        topics = [t for t, recs in by_topic.items() if recs]
    if len(topics) < 2:
        return None
    n = min(rng.choice(N_CHOICES), len(topics), max_n)
    chosen_topics = rng.sample(topics, n)
    chosen_records = [rng.choice(by_topic[t]) for t in chosen_topics]

    # Find the window indices in the chat history (for excluded_ranges)
    excluded = []
    for r in chosen_records:
        rng_idx = find_window_indices(chat_history, r["window_turns"])
        if rng_idx:
            excluded.append(rng_idx)

    # Sample fillers (only if allow_filler and chat history available)
    if allow_filler and chat_history:
        f = rng.choice(FILLER_F_CHOICES)
        filler_chunks = sample_filler_chunks(chat_history, excluded, f, rng)
    else:
        filler_chunks = []

    # Build segments — truncate long preference windows to MAX_CHARS_PER_WINDOW.
    segments = []
    for r in chosen_records:
        text = turns_to_text(r["window_turns"])
        if len(text) > MAX_CHARS_PER_WINDOW:
            text = text[:MAX_CHARS_PER_WINDOW]
        segments.append({"text": text, "topic": r["topic"], "kind": "preference_window", "source_row_idx": r["row_idx"]})
    for fc in filler_chunks:
        text = turns_to_text(fc)
        segments.append({"text": text, "topic": "_filler", "kind": "filler", "source_row_idx": None})
    rng.shuffle(segments)

    # Drop fillers from the tail until the total fits MAX_DOC_CHARS.
    while sum(len(s["text"]) for s in segments) + (len(SEPARATOR) * (len(segments) - 1)) > MAX_DOC_CHARS:
        # Find the last filler segment and drop it. If no fillers remain, truncate the longest preference window further.
        idx = next((i for i in range(len(segments) - 1, -1, -1) if segments[i]["kind"] == "filler"), None)
        if idx is not None:
            segments.pop(idx)
            continue
        # No fillers — trim the longest preference window by 10%.
        idx = max(range(len(segments)), key=lambda i: len(segments[i]["text"]))
        new_len = int(len(segments[idx]["text"]) * 0.9)
        if new_len < 200:
            break
        segments[idx]["text"] = segments[idx]["text"][:new_len]

    document = SEPARATOR.join(s["text"] for s in segments)

    # Topic-label-per-segment for retriever supervision (token-level resolved at training time).
    segment_topic_map = [
        {"topic": s["topic"], "kind": s["kind"], "source_row_idx": s["source_row_idx"], "char_len": len(s["text"])}
        for s in segments
    ]

    # Pick instruction mode and target topic, respecting the holdout policy.
    if target_topic_holdout_mode == "force_heldout":
        # Skip learn_all (no target topic to enforce holdout on); pick a heldout target.
        candidate_targets = [t for t in chosen_topics if t in HELDOUT_TOPICS]
        if not candidate_targets:
            return None  # this example doesn't include any heldout topic — skip
        instruction_mode = rng.choice(["inclusive", "exclusive"])
        target_topic_for_instruction = rng.choice(candidate_targets)
    elif target_topic_holdout_mode == "no_heldout":
        instruction_mode = rng.choice(["inclusive", "exclusive", "learn_all"])
        if instruction_mode in ("inclusive", "exclusive"):
            candidate_targets = [t for t in chosen_topics if t not in HELDOUT_TOPICS]
            if not candidate_targets:
                # All chosen topics are heldout — fall back to learn_all (which has no target name).
                instruction_mode = "learn_all"
                target_topic_for_instruction = None
            else:
                target_topic_for_instruction = rng.choice(candidate_targets)
        else:
            target_topic_for_instruction = None
    else:  # "any"
        instruction_mode = rng.choice(["inclusive", "exclusive", "learn_all"])
        target_topic_for_instruction = rng.choice(chosen_topics) if instruction_mode != "learn_all" else None

    if instruction_mode == "inclusive":
        learn_topics = {target_topic_for_instruction}
        ignore_topics = set(chosen_topics) - learn_topics
        learning_instruction = render_inclusive(target_topic_for_instruction, rng=rng)
    elif instruction_mode == "exclusive":
        ignore_topics = {target_topic_for_instruction}
        learn_topics = set(chosen_topics) - ignore_topics
        learning_instruction = render_exclusive(target_topic_for_instruction, rng=rng)
    else:  # learn_all
        learn_topics = set(chosen_topics)
        ignore_topics = set()
        learning_instruction = render_learn_all(rng=rng)

    # Build input_output_pairs — one per chosen record's user_query.
    # Three teacher signals are exposed explicitly in each pair so the dataset is
    # self-documenting; `target_output` is the one we actually train on, resolved
    # by learn_or_ignore.
    pairs = []
    for r in chosen_records:
        is_learn = r["topic"] in learn_topics
        pref_aware = r.get("target_with_preference")
        naive = r.get("target_without_preference")
        target = pref_aware if is_learn else naive
        pairs.append({
            "type": "preference_probe",
            "topic": r["topic"],
            "learn_or_ignore": "learn" if is_learn else "ignore",
            "user_query": r["user_query"],
            "preference": r["preference"],
            "explanation": r.get("explanation"),
            "pref_type": r.get("pref_type"),
            "is_sensitive": r.get("is_sensitive", False),
            "source_row_idx": r["row_idx"],
            # Three teacher signals (kept side-by-side for inspection / future ablations):
            "aligned_response": r.get("aligned_response"),                     # GPT-5 (PersonaMem dataset)
            "llama3_preference_aware_response": pref_aware,                   # Llama-3-Instruct teacher w/ preference
            "llama3_naive_response": naive,                                   # Llama-3-Instruct teacher w/o preference
            # Resolved target_output: what the trainer uses for CE loss.
            "target_output": target,                                           # = pref_aware (learn) | naive (ignore)
        })

    return {
        "document": document,
        "learning_instruction": learning_instruction,
        "input_output_pairs": pairs,
        "metadata": {
            "source": chosen_records[0]["source"],
            "persona_id": chosen_records[0].get("persona_id"),
            "instruction_mode": instruction_mode,
            "target_topic_for_instruction": target_topic_for_instruction,
            "topics_in_doc": list(chosen_topics),
            "learn_topics": sorted(learn_topics),
            "ignore_topics": sorted(ignore_topics),
            "filler_segments_count": len(filler_chunks),
            "N": n,
            "segment_topic_map": segment_topic_map,
        },
    }


def generate_personamem_split(records_path: Path,
                               dataset_root: Path,
                               seed: int,
                               examples_per_persona: int,
                               allow_filler: bool,
                               target_topic_holdout_mode: str = "no_heldout") -> list[dict]:
    with open(records_path) as f:
        records = json.load(f)

    by_persona: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_persona[r["persona_id"]].append(r)

    rng = random.Random(seed)
    out = []
    chat_cache: dict[str, list[dict]] = {}
    skipped_too_few = 0
    for pid, recs in by_persona.items():
        topics_for_mode = (
            {r["topic"] for r in recs if r["topic"] not in HELDOUT_TOPICS}
            if target_topic_holdout_mode == "no_heldout"
            else {r["topic"] for r in recs}
        )
        if len(topics_for_mode) < 2:
            skipped_too_few += 1
            continue
        # For force_heldout mode, the persona must have at least one heldout topic.
        if target_topic_holdout_mode == "force_heldout":
            if not any(r["topic"] in HELDOUT_TOPICS for r in recs):
                continue
        link = recs[0].get("chat_history_link") or ""
        if link not in chat_cache:
            chat_cache[link] = load_chat_history(dataset_root, link)
        chat_history = chat_cache[link]
        for _ in range(examples_per_persona):
            ex = make_merged_example(recs, chat_history, dataset_root, rng,
                                      allow_filler=allow_filler,
                                      target_topic_holdout_mode=target_topic_holdout_mode)
            if ex is None:
                continue
            out.append(ex)
    return out


def generate_prefeval(records_path: Path,
                      seed: int,
                      examples_per_topic: int) -> list[dict]:
    """PrefEval has no persona axis. Group by topic, then merge across topics."""
    with open(records_path) as f:
        records = json.load(f)

    # Group by topic so we can sample N distinct-topic rows.
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_topic[r["topic"]].append(r)
    topics = list(by_topic.keys())

    rng = random.Random(seed)
    out = []
    for anchor_topic, anchor_records in by_topic.items():
        for _ in range(examples_per_topic):
            n = rng.choice(N_CHOICES)
            # Pick anchor_topic + N-1 other topics.
            other_topics = [t for t in topics if t != anchor_topic]
            n = min(n, 1 + len(other_topics))
            chosen_topics = [anchor_topic] + rng.sample(other_topics, n - 1)
            chosen_records = [rng.choice(by_topic[t]) for t in chosen_topics]

            segments = []
            for r in chosen_records:
                text = turns_to_text(r["window_turns"])
                segments.append({"text": text, "topic": r["topic"], "kind": "preference_window", "source_row_idx": r["row_idx"]})
            rng.shuffle(segments)
            document = SEPARATOR.join(s["text"] for s in segments)
            segment_topic_map = [
                {"topic": s["topic"], "kind": s["kind"], "source_row_idx": s["source_row_idx"], "char_len": len(s["text"])}
                for s in segments
            ]

            instruction_mode = rng.choice(["inclusive", "exclusive", "learn_all"])
            if instruction_mode == "inclusive":
                target_topic = rng.choice(chosen_topics)
                learn_topics = {target_topic}
                ignore_topics = set(chosen_topics) - learn_topics
                learning_instruction = render_inclusive(target_topic, rng=rng)
            elif instruction_mode == "exclusive":
                target_topic = rng.choice(chosen_topics)
                ignore_topics = {target_topic}
                learn_topics = set(chosen_topics) - ignore_topics
                learning_instruction = render_exclusive(target_topic, rng=rng)
            else:  # learn_all
                target_topic = None
                learn_topics = set(chosen_topics)
                ignore_topics = set()
                learning_instruction = render_learn_all(rng=rng)

            pairs = []
            for r in chosen_records:
                is_learn = r["topic"] in learn_topics
                pairs.append({
                    "type": "preference_probe",
                    "topic": r["topic"],
                    "learn_or_ignore": "learn" if is_learn else "ignore",
                    "user_query": r["user_query"],
                    "preference": r["preference"],
                    "explanation": r.get("explanation"),
                    "pref_type": r.get("pref_type"),
                    "is_sensitive": r.get("is_sensitive", False),
                    "source_row_idx": r["row_idx"],
                    "aligned_response": r.get("aligned_response"),  # PrefEval has aligned_op
                    "llama3_preference_aware_response": None,        # PrefEval is eval-only
                    "llama3_naive_response": None,
                    "target_output": None,                           # eval-only, no training target
                })

            out.append({
                "document": document,
                "learning_instruction": learning_instruction,
                "input_output_pairs": pairs,
                "metadata": {
                    "source": "prefeval",
                    "persona_id": None,
                    "instruction_mode": instruction_mode,
                    "target_topic_for_instruction": target_topic,
                    "topics_in_doc": list(chosen_topics),
                    "learn_topics": sorted(learn_topics),
                    "ignore_topics": sorted(ignore_topics),
                    "filler_segments_count": 0,
                    "N": n,
                    "segment_topic_map": segment_topic_map,
                },
            })
    return out


def main() -> None:
    ART = Path(__file__).resolve().parents[1] / "artifacts"
    DATASET_ROOT = Path(__file__).resolve().parents[3] / "data" / "personamem_v2"

    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts-dir", type=Path, default=ART)
    ap.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    ap.add_argument("--variants", type=int, default=5, help="Number of train variants (5 → train_0..train_4).")
    ap.add_argument("--allow-filler", type=lambda s: s.lower() == "true", default=True)
    args = ap.parse_args()

    pm_train = args.artifacts_dir / "personamem_pool_train.json"
    pm_val = args.artifacts_dir / "personamem_pool_val.json"
    pm_bench = args.artifacts_dir / "personamem_pool_benchmark.json"
    pe_pool = args.artifacts_dir / "prefeval_pool.json"

    # Train variants — target topic in instruction is NEVER in HELDOUT_TOPICS.
    for i in range(args.variants):
        out = generate_personamem_split(
            pm_train, args.dataset_root, seed=42 + i,
            examples_per_persona=EXAMPLES_PER_PERSONA_TRAIN, allow_filler=args.allow_filler,
            target_topic_holdout_mode="no_heldout",
        )
        out_path = args.artifacts_dir / f"train_{i}.json"
        with open(out_path, "w") as f:
            json.dump(out, f)
        print(f"wrote {len(out):>6} examples -> {out_path.name}")

    # Val (in-distribution validation — same holdout policy as train).
    out = generate_personamem_split(pm_val, args.dataset_root, seed=100,
                                     examples_per_persona=EXAMPLES_PER_PERSONA_VAL,
                                     allow_filler=args.allow_filler,
                                     target_topic_holdout_mode="no_heldout")
    with open(args.artifacts_dir / "val.json", "w") as f:
        json.dump(out, f)
    print(f"wrote {len(out):>6} examples -> val.json")

    # Held-out PersonaMem (benchmark personas) — split into seen-topics and unseen-topics.
    # SEEN: target topic NOT held-out (tests persona generalization only)
    out = generate_personamem_split(pm_bench, args.dataset_root, seed=200,
                                     examples_per_persona=EXAMPLES_PER_PERSONA_BENCHMARK,
                                     allow_filler=args.allow_filler,
                                     target_topic_holdout_mode="no_heldout")
    with open(args.artifacts_dir / "held_out_personamem_seen_topics.json", "w") as f:
        json.dump(out, f)
    print(f"wrote {len(out):>6} examples -> held_out_personamem_seen_topics.json")

    # UNSEEN: target topic IS held-out (tests language generalization on novel topic names)
    out = generate_personamem_split(pm_bench, args.dataset_root, seed=201,
                                     examples_per_persona=EXAMPLES_PER_PERSONA_BENCHMARK,
                                     allow_filler=args.allow_filler,
                                     target_topic_holdout_mode="force_heldout")
    with open(args.artifacts_dir / "held_out_personamem_unseen_topics.json", "w") as f:
        json.dump(out, f)
    print(f"wrote {len(out):>6} examples -> held_out_personamem_unseen_topics.json")

    # Held-out PrefEval (no filler — PrefEval has no chat history)
    out = generate_prefeval(pe_pool, seed=300, examples_per_topic=EXAMPLES_PER_TOPIC_PREFEVAL)
    with open(args.artifacts_dir / "held_out_prefeval.json", "w") as f:
        json.dump(out, f)
    print(f"wrote {len(out):>6} examples -> held_out_prefeval.json")


if __name__ == "__main__":
    main()
