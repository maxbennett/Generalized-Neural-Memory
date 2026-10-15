# Long-horizon retention runs (Figure 3)

One subdirectory per system. Files are raw outputs of `scripts/retention_probe.py`;
`scripts/make_fig3_retention.py` pools them into the figure. Naming:
`<system>[_variant]_f<shard>.json`, where `f0/f1/f2` is the held-out test shard
(`data/counterfact/test_ood_{0,1,2}.json`) — the figure pools all three.

| File family | Rollout | Feeds (record key) |
|---|---|---|
| `<sys>_f?.json` | 700-document rollout, fact-dense probe sampling | Facts panels: Score / Accuracy / Selectivity (`probes`, `sel`, `spec`) |
| `<sys>_flag1000_f?.json` | 1,000-document streams (`data/counterfact/test_ood_1000doc_*.json`) | same Facts panels — extra samples at every lag |
| `<sys>_refusal_f?.json` | 700-doc rollout with refusal-targeted probe sampling enabled and a denser late-state ladder | Refusal (F1) panel (`refusals`) |
| `<sys>_refusal_s100?_f?.json` | variance-seed reruns of the refusal rollout (collate seeds 1001–1004; MemoryAgent: 1001 only) | pooled into the same Refusal panel — refusal events are sparse, so the reruns supply its statistical power |

Two things that are easy to misread:

- **Each file contains all record types** (a fact run also logs some refusal records
  and vice versa), but the chart routes by *file family*: Facts panels read only the
  fact-run families, the Refusal panel reads only the refusal-run families. This keeps
  each panel on a single homogeneous probe-sampling protocol.
- Fact probes are scored from target-vs-nontarget token logits; refusal probes are
  scored by generating a response and detecting a refusal — which is why refusal got
  its own (cheaper-to-rerun) campaign with `params_long_horizon_refusal.yaml`.

Reruns: `scripts/final_eval_lanes.sh` (GNM / ICL-FT / RAG-FT facts),
`scripts/fig3_queue_runner.sh` (MemoryAgent facts + refusal),
`scripts/refusal_power_lanes.sh` (GNM / ICL-FT / RAG-FT refusal seeds).
Fresh outputs land in `evals/counterfact/results/long_horizon/` and shadow these files.
