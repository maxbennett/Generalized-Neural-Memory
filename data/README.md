# data/

Benchmark files are not tracked in git. `python scripts/download_data.py` fills `counterfact/`
and `personamem/` from Hugging Face (`--counterfact` / `--personamem` for one benchmark). Split
names follow the paper.

| Path | Split | Contents |
|---|---|---|
| `counterfact/train_0-4.json` | train | five training buckets |
| `counterfact/val_id.json` | val-id | new facts, seen categories — early stopping only |
| `counterfact/test_ood_0-2.json` | test-ood | held-out categories and formats — the three Table 1 / Fig. 3 streams |
| `counterfact/test_ood_1000doc_0-2.json` | test-ood | 1,000-document streams for the long-horizon supplement |
| `counterfact/fact_id_to_subject.json` | — | fact-id → subject lookup used by the eval harness |
| `personamem/train_{gnm,icl,gnm_seed_robustness}/train_0-4.json` | train | each released model's training corpus |
| `personamem/train_*/val_id.json` | Val-ID | unseen preferences, seen target topics — early stopping only |
| `personamem/val_ood.json` | Val-OOD | held-out target topics — **the** evaluation file for every reported PersonaMem number |
| `personamem/implicit_eval/val_ood_implicit.json` | Val-OOD | same examples with the hand-authored implicit learning instructions (appendix) |
| `personamem/inferred_eval/val_ood_inferred_*.json` | Val-OOD | same examples with instructions authored by a frozen LLM (appendix); `*_audit.json` = the LLM's raw outputs |

Tracked alongside: `baseline_letter_caches/` (each model's precomputed no-memory picks, used by
the PersonaMem collision filter and the seed-robustness table) and `analysis/` (inputs to the
Fig. 6 memory-selectivity notebooks).
