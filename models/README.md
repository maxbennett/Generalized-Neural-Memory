# models/

Trained checkpoints are not tracked in git. `python scripts/download_models.py` fills each
directory below from Hugging Face and converts the download to a single `model.pth`, the path
every eval and training config expects (`--counterfact` / `--personamem` restrict to one benchmark).

| Directory | HF repo | Role |
|---|---|---|
| `gnm_counterfact/` | [`gnm-counterfact`](https://huggingface.co/maxbennett/generalized-neural-memory-gnm-counterfact) | **GNM** (CounterFACT, all instruction types) |
| `gnm_counterfact_ablation/` | [`gnm-counterfact-ablation`](https://huggingface.co/maxbennett/generalized-neural-memory-gnm-counterfact-ablation) | GNM ablation (no gradients through memory write) |
| `icl_ft_counterfact/` | [`iclft-counterfact`](https://huggingface.co/maxbennett/generalized-neural-memory-iclft-counterfact) | ICL-FT baseline (CounterFACT) |
| `rag_ft_counterfact/` | [`ragft-counterfact`](https://huggingface.co/maxbennett/generalized-neural-memory-ragft-counterfact) | RAG-FT baseline (CounterFACT) |
| `gnm_personamem/` | [`gnm-personamem`](https://huggingface.co/maxbennett/generalized-neural-memory-gnm-personamem) | **GNM** (PersonaMem) |
| `gnm_personamem_ablation/` | [`gnm-personamem-ablation`](https://huggingface.co/maxbennett/generalized-neural-memory-gnm-personamem-ablation) | GNM ablation (PersonaMem) |
| `icl_ft_personamem/` | [`iclft-personamem`](https://huggingface.co/maxbennett/generalized-neural-memory-iclft-personamem) | ICL-FT baseline (PersonaMem) |

Untrained baselines (MemoryLLM, Llama-3, Qwen3-14B, Gemma3-12B, HippoRAG, M+) are fetched from
their own public HF repos at eval time and are not stored here.
