"""Convert PersonaMem-v2 CSVs to a flat per-row record list.

For each row:
- Drop rows with pref_type == 'ask_to_forget' (we are doing static-only training).
- Parse JSON-ish snippet, user_query, incorrect_answers.
- Window = `PRIOR_TURNS_TO_PREPEND` chat-history turns immediately preceding the snippet,
  followed by the snippet itself. This gives a small conversational onramp without the
  noise of large ±N windows. ~2 prior turns is enough context without dominating.
- Output unified-schema records (one JSON file per split).
"""

import argparse
import ast
import json
from pathlib import Path

import pandas as pd

SENSITIVE_PREF_TYPES = {"health_and_medical_conditions", "therapy_background", "sensitive_info"}
PRIOR_TURNS_TO_PREPEND = 0  # snippet-only — prior turns slow training and the snippet itself carries the preference signal


def parse_json_or_pyliteral(text):
    """The CSVs serialize JSON-ish content; try strict JSON first, then ast.literal_eval."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    s = str(text).strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            return None


def parse_snippet_turns(raw) -> list[dict]:
    if isinstance(raw, list):
        return [
            {"role": t.get("role", "user"), "content": t.get("content", "")}
            for t in raw if isinstance(t, dict)
        ]
    return []


def parse_user_query(raw, fallback_str) -> str:
    if isinstance(raw, dict):
        return raw.get("content", "")
    if isinstance(raw, list) and raw:
        last = raw[-1]
        if isinstance(last, dict):
            return last.get("content", "")
        return str(last)
    return str(fallback_str).strip()


def find_snippet_start(snippet_turns: list[dict], full_chat: list[dict]) -> int | None:
    """Find the index of the snippet's first user turn inside the full chat history."""
    if not snippet_turns or not full_chat:
        return None
    first = next((t for t in snippet_turns if t.get("role") == "user"), None)
    if not first or not first.get("content"):
        return None
    needle = first["content"].strip()[:60]
    if not needle:
        return None
    for i, t in enumerate(full_chat):
        c = t.get("content", "")
        if isinstance(c, str) and needle in c[: max(200, len(needle) + 50)]:
            return i
    return None


_chat_history_cache: dict[str, list[dict] | None] = {}


def get_prior_turns(chat_history_link: str, snippet_turns: list[dict],
                    dataset_root: Path, n_prior: int = PRIOR_TURNS_TO_PREPEND) -> list[dict]:
    """Return the `n_prior` chat-history turns immediately preceding the snippet, or [] on failure."""
    if not chat_history_link or n_prior <= 0:
        return []
    if chat_history_link not in _chat_history_cache:
        path = dataset_root / chat_history_link
        if not path.exists():
            _chat_history_cache[chat_history_link] = None
        else:
            try:
                with open(path) as f:
                    obj = json.load(f)
                _chat_history_cache[chat_history_link] = obj.get("chat_history") if isinstance(obj, dict) else obj
            except Exception:
                _chat_history_cache[chat_history_link] = None
    full_chat = _chat_history_cache[chat_history_link]
    if not isinstance(full_chat, list):
        return []
    start = find_snippet_start(snippet_turns, full_chat)
    if start is None or start == 0:
        return []
    lo = max(0, start - n_prior)
    return [{"role": t.get("role", "user"), "content": t.get("content", "")} for t in full_chat[lo:start]]


def normalize_row(row: pd.Series, row_idx: int, split: str, dataset_root: Path) -> dict | None:
    pref_type = row.get("pref_type")
    pref_type_str = str(pref_type) if pref_type and not (isinstance(pref_type, float) and pd.isna(pref_type)) else None

    # Skip ask_to_forget — we are doing static-only training.
    if pref_type_str == "ask_to_forget":
        return None

    snippet_turns = parse_snippet_turns(parse_json_or_pyliteral(row.get("related_conversation_snippet")))
    user_query_text = parse_user_query(parse_json_or_pyliteral(row.get("user_query")), row.get("user_query", ""))
    correct_answer = str(row.get("correct_answer", "")).strip()
    incorrect_raw = parse_json_or_pyliteral(row.get("incorrect_answers"))
    incorrect = [str(x) for x in incorrect_raw] if isinstance(incorrect_raw, list) else []

    if not snippet_turns or not user_query_text or not correct_answer:
        return None

    chat_history_link = str(row.get("chat_history_32k_link", ""))
    prior_turns = get_prior_turns(chat_history_link, snippet_turns, dataset_root, PRIOR_TURNS_TO_PREPEND)
    window_turns = prior_turns + snippet_turns

    is_sensitive = bool(row.get("sensitive_info", False)) or (pref_type_str in SENSITIVE_PREF_TYPES)

    return {
        "source": "personamem_v2",
        "split": split,
        "persona_id": int(row["persona_id"]),
        "row_idx": int(row_idx),
        "topic": str(row.get("topic_preference", "")),
        "topic_query": str(row.get("topic_query", "")),
        "pref_type": pref_type_str,
        "preference": str(row.get("preference", "")),
        "explanation": None,  # PrefEval has this; PersonaMem does not
        "who": str(row.get("who", "self")),
        "is_sensitive": is_sensitive,
        "conversation_scenario": str(row.get("conversation_scenario", "")),
        "snippet_turns": snippet_turns,
        "window_turns": window_turns,
        "n_prior_turns": len(prior_turns),
        "user_query": user_query_text,
        "aligned_response": correct_answer,  # GPT-5 generated; diagnostic only
        "incorrect_answers": incorrect,
        "target_with_preference": None,
        "target_without_preference": None,
    }


def build_split(csv_path: Path, split_name: str, dataset_root: Path) -> list[dict]:
    df = pd.read_csv(csv_path)
    records = []
    skipped_atf = 0
    skipped_missing = 0
    no_prior_count = 0
    for i, row in df.iterrows():
        if str(row.get("pref_type", "")) == "ask_to_forget":
            skipped_atf += 1
            continue
        rec = normalize_row(row, i, split_name, dataset_root)
        if rec is None:
            skipped_missing += 1
            continue
        if rec.get("n_prior_turns", 0) == 0:
            no_prior_count += 1
        records.append(rec)
    print(f"  {split_name}: kept={len(records)}, dropped_ask_to_forget={skipped_atf}, "
          f"dropped_missing_fields={skipped_missing}, "
          f"prior_lookup_failed={no_prior_count} ({no_prior_count/max(1,len(records))*100:.1f}%)")
    return records


def summarize(records: list[dict]) -> None:
    if not records:
        return
    persona_ids = {r["persona_id"] for r in records}
    topics = {r["topic"] for r in records}
    sens = sum(1 for r in records if r["is_sensitive"])
    by_persona_topics: dict[int, set] = {}
    snippet_lens = []
    for r in records:
        by_persona_topics.setdefault(r["persona_id"], set()).add(r["topic"])
        snippet_lens.append(sum(len(t["content"]) for t in r["snippet_turns"]))
    min_topics = min(len(v) for v in by_persona_topics.values())
    import statistics
    print(f"    personas: {len(persona_ids)} | distinct topics: {len(topics)} | sensitive: {sens}")
    print(f"    min distinct topics per persona: {min_topics}")
    print(f"    snippet chars: median={int(statistics.median(snippet_lens))}, "
          f"mean={int(statistics.mean(snippet_lens))}, max={max(snippet_lens)}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]  # MemoryLLM/
    default_dataset_root = repo_root / "data" / "personamem_v2"
    default_input_dir = default_dataset_root / "benchmark" / "text"
    default_output_dir = Path(__file__).resolve().parents[1] / "artifacts"

    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=default_input_dir)
    ap.add_argument("--dataset-root", type=Path, default=default_dataset_root,
                    help="Root containing data/chat_history_32k/<file>.json")
    ap.add_argument("--output-dir", type=Path, default=default_output_dir)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits = {"train": "train.csv", "val": "val.csv", "benchmark": "benchmark.csv"}
    for split, fname in splits.items():
        csv_path = args.input_dir / fname
        if not csv_path.exists():
            print(f"[SKIP] {csv_path} not found")
            continue
        print(f"=== {split} ({csv_path}) ===")
        records = build_split(csv_path, split, args.dataset_root)
        out_path = args.output_dir / f"personamem_pool_{split}.json"
        with open(out_path, "w") as f:
            json.dump(records, f)
        print(f"  wrote {len(records)} records -> {out_path}")
        summarize(records)


if __name__ == "__main__":
    main()
