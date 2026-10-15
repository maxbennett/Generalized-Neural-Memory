"""Assemble the MemoryAgent (selective write-time preprocessing) comparison table.

Merges:
  - the memoryagent seq-10 runs (results/ or paper_results/memoryagent/)
  - paper seq-10 baselines (evals/counterfact/paper_results/short_horizon/*/summary.json)
  - write-cost stats from the memoryagent notes JSONLs + GNM microbenchmark json

Writes results/memoryagent_comparison.md
"""
import json, os, glob, statistics, argparse

EVAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(EVAL_DIR))
RUNS_DIR = os.path.join(EVAL_DIR, "results")
if os.path.isdir(os.path.join(RUNS_DIR, "memoryagent_icl")):          # fresh reruns (run_memoryagent_short_horizon.sh)
    MA_ICL, MA_RAG = os.path.join(RUNS_DIR, "memoryagent_icl"), os.path.join(RUNS_DIR, "memoryagent_rag")
else:                                                                  # shipped precomputed runs
    RUNS_DIR = os.path.join(REPO, "evals/counterfact/paper_results/memoryagent")
    MA_ICL, MA_RAG = os.path.join(RUNS_DIR, "icl"), os.path.join(RUNS_DIR, "rag")
SAVED = os.path.join(REPO, "evals/counterfact/paper_results/short_horizon")

METRICS = [
    ("fact_accuracy", "Fact acc"),
    ("fact_selectivity", "Fact sel"),
    ("format_accuracy", "Format acc"),
    ("format_selectivity", "Format sel"),
    ("refusal_recall", "Refusal recall"),
    ("refusal_accuracy", "Refusal acc"),
]

ROWS = [
    # (label, summary path, notes glob or None)
    ("GNM (ours)",              f"{SAVED}/gnm/summary.json", None),
    ("MemoryLLM (no instr.)",   f"{SAVED}/memoryllm/summary.json", None),
    ("ICL untrained",           f"{SAVED}/llama3/summary.json", None),
    ("RAG untrained",           f"{SAVED}/rag_untrained/summary.json", None),
    ("RAG fine-tuned",          f"{SAVED}/rag_trained/summary.json", None),
    ("MemoryAgent ICL", f"{MA_ICL}/summary.json", f"{MA_ICL}/notes.rank*.jsonl"),
    ("MemoryAgent RAG", f"{MA_RAG}/summary.json", f"{MA_RAG}/notes.rank*.jsonl"),
]


def note_stats(pattern):
    recs = []
    for p in sorted(glob.glob(pattern)):
        with open(p) as f:
            recs.extend(json.loads(l) for l in f if l.strip())
    if not recs:
        return None
    return {
        "n": len(recs),
        "median_write_s": statistics.median(r["write_seconds"] for r in recs),
        "median_prefill_tok": statistics.median(r["prefill_tokens"] for r in recs),
        "median_decode_tok": statistics.median(r["decode_tokens"] for r in recs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-json", default=None, help="write_cost_bench.json from microbenchmark")
    ap.add_argument("--out", default=os.path.join(EVAL_DIR, "results/memoryagent_comparison.md"))
    args = ap.parse_args()

    bench = json.load(open(args.bench_json)) if args.bench_json and os.path.exists(args.bench_json) else {}

    lines = []
    lines.append("# MemoryAgent (selective write-time preprocessing) baseline — short-horizon (seq-10) protocol\n")
    lines.append("All runs: paper protocol (seed 42, 3 test_unseen files, full probing, "
                 "sampling knobs from evals/counterfact/paper_results/short_horizon/rag_untrained/run_config.json).\n")
    header = "| Model | " + " | ".join(h for _, h in METRICS) + " | Write cost / doc |"
    sep = "|" + "---|" * (len(METRICS) + 2)
    lines += [header, sep]

    for label, spath, npattern in ROWS:
        if not os.path.exists(spath):
            lines.append(f"| {label} | " + " | ".join(["MISSING"] * len(METRICS)) + " | |")
            continue
        s = json.load(open(spath))
        cells = []
        for key, _ in METRICS:
            if key in s and s[key]:
                vals = s[key]
                cells.append(f"{sum(vals)/len(vals):.3f}")
            else:
                cells.append("—")
        cost = ""
        if npattern:
            ns = note_stats(npattern)
            if ns:
                cost = (f"{ns['median_write_s']:.2f}s; {int(ns['median_prefill_tok'])} prefill + "
                        f"{int(ns['median_decode_tok'])} decode tok (n={ns['n']})")
        elif label.startswith("GNM") and "gnm_write" in bench:
            b = bench["gnm_write"]
            cost = f"{b['median_s']:.2f}s; 1 fwd pass over ~{int(b['median_doc_li_tokens'])} tok"
        elif "ICL" in label or "RAG" in label:
            cost = "~0 (stores raw text; cost deferred to query time)"
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {cost} |")

    if bench:
        lines.append("\n## Write-cost microbenchmark (same GPU, same 30 docs)\n")
        for k, b in bench.items():
            lines.append(f"- **{k}**: median {b['median_s']*1000:.0f} ms, mean {b['mean_s']*1000:.0f} ms, "
                         f"p90 {b['p90_s']*1000:.0f} ms (n={b['n']})")

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
