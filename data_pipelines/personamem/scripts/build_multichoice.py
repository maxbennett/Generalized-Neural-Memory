"""Build multi-choice preference-selectivity data.

Each merged example:
- Document = a list of N=2-5 preference statements (one line each), much shorter
  than conversation windows (~100-300 tokens vs ~3-4k).
- Learning instruction = inclusive ("only learn topic X"), exclusive
  ("learn all except topic Y"), or learn_all.
- Probes (one per source row) are multiple-choice:
    options A-D = shuffle of {aligned_response} ∪ {3 incorrect_answers}
    correct_letter = letter of aligned_response after shuffle
- Per probe `learn_or_ignore` is set by instruction mode + probe topic.

Outputs (under <artifacts>):
  train_mc_{0..4}.json     (5 variants from PersonaMem train personas)
  val_mc.json              (PersonaMem train personas, disjoint seed)
  held_out_personamem_mc_seen_topics.json
  held_out_personamem_mc_unseen_topics.json
  held_out_prefeval_mc.json
"""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from topic_descriptors import topic_with_hint

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"

# Same heldout set as the conversation-format pipeline so eval comparisons line up.
HELDOUT_TOPICS = {
    "Sports", "Music", "Travel", "Fashion", "Technology",
    "Photography", "Gardening", "Education", "Science", "Pets",
}

LETTERS = ["A", "B", "C", "D"]
N_CHOICES = [2, 3, 4, 5]  # number of preferences per merged document

# v4: only the "inclusive" mode — single positive instruction format ("focus only on
# preferences about topic X; ignore everything else"). Dropping learn_all (no ignore
# probes, biases the model toward always picking correct) and exclusive (negative
# framing — harder to learn from).
# Topic strings include parenthetical descriptors (e.g. "Sports (athletes, teams, ...)")
# so the model has semantic anchors beyond the bare topic name.
INSTRUCTION_TEMPLATES_INCLUSIVE = [
    "Pay attention only to preferences about {t}.",
    "Only remember preferences related to {t}.",
    "Focus on the user's preferences for {t} and disregard all other preferences.",
    "Of the preferences listed, only learn the ones about {t}.",
    "Ignore everything except preferences about {t}.",
    "Track only {t} preferences from this conversation.",
]


def make_options(aligned: str, incorrect: list[str], rng: random.Random) -> tuple[dict, str]:
    """Shuffle 1 correct + 3 incorrect into A-D and report the correct letter."""
    items = [(aligned, True)] + [(x, False) for x in incorrect]
    rng.shuffle(items)
    options = {LETTERS[i]: items[i][0] for i in range(4)}
    correct_letter = next(LETTERS[i] for i in range(4) if items[i][1])
    return options, correct_letter


def _probe_hash(query: str, options: dict) -> str:
    """Same hash as precompute_memoryllm_baseline_letters.py — content-addressed."""
    canonical = f"Q:{query}\nA:{options['A']}\nB:{options['B']}\nC:{options['C']}\nD:{options['D']}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


CONVERSATION_OPENINGS = [
    "Hi! I'd love to get to know you better. Tell me about yourself.",
    "Hello! What's something you'd like me to know about you?",
    "Hey, I want to learn a bit about you. Where would you like to start?",
    "Welcome — help me understand who you are. Could you share something about yourself?",
    "Hi! To help you better, I'd love to learn more about you. What should I know first?",
    "Hey there. Could you tell me a bit about yourself so I can be more helpful?",
    "Hi — let's get to know each other. What's something important about you?",
    "Hi! What's a good thing for me to know about you?",
]

CONVERSATION_FOLLOWUPS = [
    "Got it, thanks for sharing. What else would you like me to know?",
    "Thanks. Anything else you'd like to share?",
    "Noted! What else?",
    "Interesting, thanks for telling me. Tell me something else about you.",
    "Great. What's another thing I should know?",
    "Appreciated. What else would be helpful for me to know?",
    "Got it. Anything more?",
    "Thanks for that. What else comes to mind?",
    "Helpful — what else?",
    "Okay, noted. Anything else important about you?",
    "Thanks. Is there anything else?",
    "Got it — what else should I know?",
]

CONVERSATION_CLOSINGS = [
    "Thanks for sharing all that — I'll keep it in mind.",
    "Great, I have a much better sense of you now. Thanks!",
    "Noted. I'll remember these when helping you.",
    "Got it all. Thanks for letting me know.",
    "Appreciate you telling me — I'll factor it in going forward.",
    "Thanks. I'll keep all of this in mind in our future conversations.",
]


def make_doc(preferences: list[tuple[str, str]], rng: random.Random) -> str:
    """Render preferences as a single coherent multi-turn user/assistant conversation.

    The assistant opens by inviting the user to share something about themselves.
    The user reveals one preference per turn, in first person. The assistant uses
    a different naturalistic follow-up between each, then a closing acknowledgment.

    Format: plain text dialogue with `User:` / `Assistant:` prefixes.
    """
    lines: list[str] = []
    lines.append(f"Assistant: {rng.choice(CONVERSATION_OPENINGS)}")
    n = len(preferences)
    # Sample N-1 distinct follow-ups so they don't repeat within a doc.
    follow_ups = rng.sample(CONVERSATION_FOLLOWUPS,
                              k=min(n - 1, len(CONVERSATION_FOLLOWUPS)))
    for i, (_, p) in enumerate(preferences):
        lines.append(f"User: {p}")
        if i < n - 1:
            lines.append(f"Assistant: {follow_ups[i]}")
    lines.append(f"Assistant: {rng.choice(CONVERSATION_CLOSINGS)}")
    return "\n".join(lines)


def build_one_example(
    rows: list[dict],
    instruction_mode: str,
    target_topic: str | None,
    rng: random.Random,
    baseline_cache: dict | None = None,
    menu_style_fraction: float = 0.0,
) -> dict:
    """Inclusive-only. Renders rows as a multi-turn conversation using first-person preferences.

    menu_style_fraction: per-probe probability of using short menu-style options
        (`aligned_response_short` / `incorrect_answers_short`) instead of the long
        paragraph-form options. Each probe rolls independently. Requires the row to
        have those fields populated by `rewrite_options_menu_style.py`; falls back
        to long form if missing.
    """
    assert instruction_mode == "inclusive", f"only inclusive is supported, got {instruction_mode}"
    assert target_topic is not None
    # Use the Haiku-rewritten first-person preference if available; otherwise fall back
    # to the raw preference (PrefEval rows are already first-person).
    doc = make_doc(
        [(r["topic"], r.get("pref_first_person") or r["preference"]) for r in rows],
        rng,
    )
    li = rng.choice(INSTRUCTION_TEMPLATES_INCLUSIVE).format(t=topic_with_hint(target_topic, rng))

    pairs = []
    for r in rows:
        topic = r["topic"]
        loi = "learn" if topic == target_topic else "ignore"
        # Per-probe coin flip: menu-style options if available, else long form.
        # IMPORTANT: save/restore rng state around the random() call so option-shuffle
        # order for long-style probes matches what the long-only build would have produced.
        # This means long-form probes hash to the same key as the existing memoryllm_baseline_letter
        # cache entries, so we don't need to re-run vanilla MemoryLLM on those (~60% of probes).
        if (menu_style_fraction > 0.0
                and r.get("aligned_response_short")
                and isinstance(r.get("incorrect_answers_short"), list)
                and len(r["incorrect_answers_short"]) == 3):
            _rng_state = rng.getstate()
            use_menu_style = rng.random() < menu_style_fraction
            rng.setstate(_rng_state)
        else:
            use_menu_style = False
        if use_menu_style:
            aligned_choice = r["aligned_response_short"]
            incorrect_choices = r["incorrect_answers_short"]
        else:
            aligned_choice = r["aligned_response"]
            incorrect_choices = r["incorrect_answers"]
        options, correct_letter = make_options(aligned_choice, incorrect_choices, rng)
        # Content-addressed lookup: hash matches whatever precompute_memoryllm_baseline_letters.py computed.
        memoryllm_baseline_letter = None
        if baseline_cache is not None:
            memoryllm_baseline_letter = baseline_cache.get(_probe_hash(r["user_query"], options))
        pairs.append({
            "type": "preference_probe_mc",
            "topic": topic,
            "learn_or_ignore": loi,
            "user_query": r["user_query"],
            "options": options,
            "correct_letter": correct_letter,
            "memoryllm_baseline_letter": memoryllm_baseline_letter,  # None on first build, filled in by precompute
            "preference": r["preference"],
            "pref_type": r.get("pref_type"),
            "is_sensitive": bool(r.get("is_sensitive", False)),
            "source_row_idx": int(r["row_idx"]),
            "explanation": r.get("explanation"),
            "options_style": "menu" if use_menu_style else "long",
        })

    return {
        "document": doc,
        "learning_instruction": li,
        "input_output_pairs": pairs,
        "metadata": {
            "source": rows[0].get("source"),
            "persona_id": rows[0].get("persona_id"),
            "instruction_mode": instruction_mode,
            "topics_in_doc": [r["topic"] for r in rows],
            "learn_topics": [r["topic"] for r, p in zip(rows, pairs) if p["learn_or_ignore"] == "learn"],
            "ignore_topics": [r["topic"] for r, p in zip(rows, pairs) if p["learn_or_ignore"] == "ignore"],
            "N": len(rows),
            "menu_style_fraction_requested": menu_style_fraction,
        },
    }


def select_target_topic(chosen_topics: list[str], holdout_mode: str, rng: random.Random) -> tuple[str, str | None]:
    """Returns (instruction_mode, target_topic). v4: always inclusive."""
    if holdout_mode == "force_heldout":
        candidates = [t for t in chosen_topics if t in HELDOUT_TOPICS]
        if not candidates:
            return None, None
        return "inclusive", rng.choice(candidates)
    if holdout_mode == "no_heldout":
        candidates = [t for t in chosen_topics if t not in HELDOUT_TOPICS]
        if not candidates:
            return None, None
        return "inclusive", rng.choice(candidates)
    # "any"
    return "inclusive", rng.choice(chosen_topics)


def build_examples(
    pool: list[dict],
    n_examples: int,
    seed: int,
    holdout_mode: str = "no_heldout",
    group_key: str | None = "persona_id",
    baseline_cache: dict | None = None,
    menu_style_fraction: float = 0.0,
) -> list[dict]:
    """Sample N preferences (per merged example) from pool, build merged examples.

    `group_key=None` puts everything into a single group (useful for PrefEval which has
    no persona structure). Otherwise group by the given key (e.g. persona_id).

    `menu_style_fraction`: see build_one_example.
    """
    rng = random.Random(seed)
    # Drop sensitive_info — doesn't fit a "tell me about yourself" conversation frame.
    pool = [r for r in pool
            if r.get("aligned_response")
            and len(r.get("incorrect_answers") or []) == 3
            and r.get("topic") != "sensitive_info"]
    by_group: dict = defaultdict(list)
    if group_key is None:
        by_group["__all__"] = pool
    else:
        for r in pool:
            gk = r.get(group_key)
            if gk in (None, "None"):
                gk = r["row_idx"]  # row stands alone — won't make a multi-topic doc
            by_group[gk].append(r)
    groups = [g for g, recs in by_group.items() if len(recs) >= 2]
    if not groups:
        return []
    out = []
    attempts = 0
    while len(out) < n_examples and attempts < n_examples * 10:
        attempts += 1
        g = rng.choice(groups)
        recs = by_group[g]
        # pick distinct topics
        by_topic = defaultdict(list)
        for r in recs:
            by_topic[r["topic"]].append(r)
        if holdout_mode == "no_heldout":
            topics = [t for t in by_topic if t not in HELDOUT_TOPICS]
        else:
            topics = list(by_topic.keys())
        if len(topics) < 2:
            continue
        n = min(rng.choice(N_CHOICES), len(topics))
        chosen_topics = rng.sample(topics, n)
        chosen = [rng.choice(by_topic[t]) for t in chosen_topics]
        instr_mode, target = select_target_topic(chosen_topics, holdout_mode, rng)
        if instr_mode is None:
            continue  # force_heldout but no heldout topics
        ex = build_one_example(chosen, instr_mode, target, rng, baseline_cache=baseline_cache,
                                menu_style_fraction=menu_style_fraction)
        out.append(ex)
    return out


def write(path: Path, items: list[dict]):
    with open(path, "w") as f:
        json.dump(items, f)
    print(f"  wrote {len(items)} -> {path.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    ap.add_argument("--n-train-per-variant", type=int, default=512)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--n-heldout-pm-seen", type=int, default=500)
    ap.add_argument("--n-heldout-pm-unseen", type=int, default=400)
    ap.add_argument("--n-heldout-prefeval", type=int, default=300)
    ap.add_argument("--menu-style-fraction", type=float, default=0.0,
                    help="Per-probe probability of using short menu-style options "
                         "(requires aligned_response_short / incorrect_answers_short on rows "
                         "from rewrite_options_menu_style.py). When >0, only train + val are "
                         "(re)built, with --output-suffix appended; held-out files are NOT touched.")
    ap.add_argument("--output-suffix", type=str, default="",
                    help="Suffix appended to output filenames so existing artifacts aren't overwritten. "
                         "Example: '_menu_style_included' → train_mc_menu_style_included_0.json. "
                         "Required when --menu-style-fraction > 0.")
    ap.add_argument("--prefeval-style", action="store_true",
                    help="Use prefeval-style options (requires aligned_response_prefeval_style and "
                         "incorrect_answers_prefeval_style on rows from rewrite_options_prefeval_style.py). "
                         "When set, the pool's `aligned_response` / `incorrect_answers` are REPLACED "
                         "with the prefeval-style versions before doc generation. Train + val only "
                         "(held-out files not touched). Requires --output-suffix.")
    args = ap.parse_args()

    if args.menu_style_fraction > 0.0 and not args.output_suffix:
        ap.error("--output-suffix is required when --menu-style-fraction > 0")
    if args.prefeval_style and not args.output_suffix:
        ap.error("--output-suffix is required when --prefeval-style")
    if args.prefeval_style and args.menu_style_fraction > 0.0:
        ap.error("--prefeval-style and --menu-style-fraction are mutually exclusive")

    # Pool selection:
    # - If menu_style_fraction > 0, prefer *_menu_style.json (which has both first-person prefs
    #   AND short option variants). Falls back to _rewritten.json or original.
    # - Otherwise prefer *_rewritten.json (Haiku-rewritten first-person preferences), fall
    #   back to original. Originals are kept untouched for reproducibility.
    def load_pool(stem: str):
        candidates = []
        if args.prefeval_style:
            candidates.append(args.artifacts / f"{stem}_prefeval_style.json")
        if args.menu_style_fraction > 0.0:
            candidates.append(args.artifacts / f"{stem}_menu_style.json")
        candidates.append(args.artifacts / f"{stem}_rewritten.json")
        candidates.append(args.artifacts / f"{stem}.json")
        for path in candidates:
            if path.exists():
                print(f"  loading pool: {path.name}")
                pool = json.load(open(path))
                # When --prefeval-style is set, swap the prefeval-style options into the
                # standard fields so build_one_example uses them transparently. Drop rows
                # that don't have prefeval-style fields populated (e.g., Sonnet failures).
                if args.prefeval_style:
                    swapped = []
                    n_dropped = 0
                    for r in pool:
                        psa = r.get("aligned_response_prefeval_style")
                        psi = r.get("incorrect_answers_prefeval_style")
                        if not psa or not isinstance(psi, list) or len(psi) != 3:
                            n_dropped += 1
                            continue
                        new_r = dict(r)
                        new_r["aligned_response"] = psa
                        new_r["incorrect_answers"] = psi
                        # Strip aligned_response_short/incorrect_answers_short so build_one_example
                        # never picks the menu-style branch for prefeval rows.
                        new_r.pop("aligned_response_short", None)
                        new_r.pop("incorrect_answers_short", None)
                        swapped.append(new_r)
                    print(f"    [prefeval-style] {len(swapped)}/{len(pool)} rows kept "
                           f"({n_dropped} dropped — no prefeval_style fields)")
                    pool = swapped
                return pool
        raise FileNotFoundError(f"no pool found for {stem} (looked in {[p.name for p in candidates]})")
    pm_train = load_pool("personamem_pool_train")
    pm_val = pm_train  # use same pool, different seed for the val split
    pm_bench = load_pool("personamem_pool_benchmark")
    pe = load_pool("prefeval_pool")

    # Sanity check: when mixing, warn if pool rows don't have the short variants.
    if args.menu_style_fraction > 0.0:
        n_with_short = sum(1 for r in pm_train if r.get("aligned_response_short")
                            and isinstance(r.get("incorrect_answers_short"), list)
                            and len(r["incorrect_answers_short"]) == 3)
        print(f"  pm_train rows with short-option variants: {n_with_short}/{len(pm_train)} "
              f"({100*n_with_short/max(1,len(pm_train)):.1f}%)")
        if n_with_short == 0:
            ap.error("no rows have short-option variants — run rewrite_options_menu_style.py first")

    # Stable baseline-letter cache (data/baseline_letter_caches/memoryllm_baseline_letter_cache.json), keyed by (source, row_idx).
    # Survives merged-data rebuilds. If absent, memoryllm_baseline_letter falls back to None and the
    # trainer treats ignore probes with a uniform-over-3-incorrect target.
    cache_path = Path(__file__).resolve().parents[3] / "data" / "baseline_letter_caches" / "memoryllm_baseline_letter_cache.json"
    if cache_path.exists():
        baseline_cache = json.load(open(cache_path))
        print(f"  baseline cache: {len(baseline_cache)} entries  ({cache_path})")
    else:
        baseline_cache = None
        print(f"  baseline cache: not found at {cache_path} — probes will have memoryllm_baseline_letter=None")

    sfx = args.output_suffix
    msf = args.menu_style_fraction

    print(f"Building train variants...  (menu_style_fraction={msf}, suffix={sfx!r})")
    for v in range(5):
        ex = build_examples(pm_train, args.n_train_per_variant, seed=42 + v, holdout_mode="no_heldout",
                             baseline_cache=baseline_cache, menu_style_fraction=msf)
        write(args.artifacts / f"train_mc{sfx}_{v}.json", ex)

    print("Building val...")
    ex = build_examples(pm_val, args.n_val, seed=200, holdout_mode="no_heldout",
                         baseline_cache=baseline_cache, menu_style_fraction=msf)
    write(args.artifacts / f"val_mc{sfx}.json", ex)

    # Always build held-out PersonaMem (seen + unseen). When menu-style mixing is on,
    # the held-out PM files get the same suffix and the same mix — this matches the
    # training-time option distribution so eval is apples-to-apples with training.
    if not args.prefeval_style:
        print(f"Building held-out PersonaMem (seen-topics)...  (menu_style_fraction={msf})")
        ex = build_examples(pm_bench, args.n_heldout_pm_seen, seed=300, holdout_mode="no_heldout",
                             baseline_cache=baseline_cache, menu_style_fraction=msf)
        write(args.artifacts / f"held_out_personamem_mc_seen_topics{sfx}.json", ex)

        print(f"Building held-out PersonaMem (unseen-topics, instruction targets a heldout topic)...  (menu_style_fraction={msf})")
        ex = build_examples(pm_bench, args.n_heldout_pm_unseen, seed=400, holdout_mode="force_heldout",
                             baseline_cache=baseline_cache, menu_style_fraction=msf)
        write(args.artifacts / f"held_out_personamem_mc_unseen_topics{sfx}.json", ex)

        # PrefEval is the headline eval — never regenerated with the menu-style mix.
        # Its options are already short by source-data construction, so no mixing is needed.
        if msf == 0.0:
            print("Building held-out PrefEval...")
            ex = build_examples(pe, args.n_heldout_prefeval, seed=500, holdout_mode="any", group_key=None,
                                 baseline_cache=baseline_cache)
            write(args.artifacts / "held_out_prefeval_mc.json", ex)
        else:
            print("Skipping held-out PrefEval (kept stable — its source options are already short menu-style)")
    else:
        print("--prefeval-style: skipping held-out builds (we keep using existing val/prefeval data)")


if __name__ == "__main__":
    main()
