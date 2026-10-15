# MemoryAgent (selective write-time preprocessing) baseline — short-horizon (seq-10) protocol

All runs: paper protocol (seed 42, 3 test-ood files, full probing, sampling knobs from evals/counterfact/paper_results/short_horizon/rag_untrained/run_config.json).

| Model | Fact acc | Fact sel | Format acc | Format sel | Refusal recall | Refusal acc | Write cost / doc |
|---|---|---|---|---|---|---|---|
| GNM (ours) | 0.958 | 0.774 | 0.674 | 1.000 | 0.981 | 0.952 |  |
| MemoryLLM (no instr.) | 0.819 | 0.192 | 0.000 | 1.000 | 0.001 | 0.668 |  |
| ICL untrained | 0.764 | 0.220 | 0.000 | 1.000 | 0.223 | 0.690 | ~0 (stores raw text; cost deferred to query time) |
| RAG untrained | 0.732 | 0.290 | 0.000 | 1.000 | 0.363 | 0.720 | ~0 (stores raw text; cost deferred to query time) |
| RAG fine-tuned | 0.745 | 0.618 | 0.000 | 0.999 | 0.737 | 0.784 | ~0 (stores raw text; cost deferred to query time) |
| MemoryAgent ICL | 0.701 | 0.846 | 0.063 | 0.999 | 0.712 | 0.879 | 3.14s; 1460 prefill + 145 decode tok (n=2160) |
| MemoryAgent RAG | 0.617 | 0.849 | 0.060 | 1.000 | 0.777 | 0.899 | 3.23s; 1460 prefill + 145 decode tok (n=2160) |
