# MemoryAgent (selective write-time preprocessing) baseline — seq-10 mixed documents

All runs: paper protocol (seed 42, 3 test_unseen files, full probing, sampling knobs from evals/counterfact/paper_results/short_horizon/rag_untrained/run_config.json).

| Model | Fact acc | Fact sel | Format acc | Format sel | Refusal recall | Refusal acc | Write cost / doc |
|---|---|---|---|---|---|---|---|
| GNM (ours) | 0.958 | 0.774 | 0.674 | 1.000 | 0.981 | 0.952 |  |
| MemoryLLM (no instr.) | 0.819 | 0.192 | 0.000 | 1.000 | 0.001 | 0.668 |  |
| ICL untrained | 0.764 | 0.220 | 0.000 | 1.000 | 0.223 | 0.690 | ~0 (stores raw text; cost deferred to query time) |
| RAG untrained | 0.732 | 0.290 | 0.000 | 1.000 | 0.363 | 0.720 | ~0 (stores raw text; cost deferred to query time) |
| RAG fine-tuned | 0.745 | 0.618 | 0.000 | 0.999 | 0.737 | 0.784 | ~0 (stores raw text; cost deferred to query time) |
| MemoryAgent ICL | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | |
| MemoryAgent RAG | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | |
