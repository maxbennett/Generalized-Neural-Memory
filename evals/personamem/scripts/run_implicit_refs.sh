#!/bin/bash
# Reference rows for the inferred-instruction experiment: seq-len 1,
# released GNM / ICL-FT PersonaMem checkpoints, oracle-explicit and implicit_v3 conditions.
# GNM oracle (GPU 0) was already launched separately; this covers the rest.
set -u
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
OUT=$R/evals/personamem/paper_results/inferred_instructions
PY=${PY:-python}
S=$R/evals/personamem/scripts
export WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $S

CUDA_VISIBLE_DEVICES=1 $PY -u eval_personamem.py \
  --eval-path $R/data/personamem/implicit_eval/val_unseen_topics_implicit_v3.json \
  --model memoryllm_chat --model-state-dict-path $R/models/gnm_personamem/model.pth \
  --output-path $OUT/gnm_implicit_v3_seq1.json --seq-len 1 --n-rollouts 400 \
  --baseline-source memoryllm > $OUT/gnm_implicit_v3_seq1.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 $PY -u eval_icl_pref.py \
  --eval-path $R/data/personamem/test.json \
  --model llama3_8b_instruct_personamem --model-state-dict-path $R/models/icl_ft_personamem/model.pth \
  --save-root $OUT/icl_oracle_seq1 --seq-len 1 --n-rollouts 400 \
  --baseline-source llama3 > $OUT/icl_oracle_seq1.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 $PY -u eval_icl_pref.py \
  --eval-path $R/data/personamem/implicit_eval/val_unseen_topics_implicit_v3.json \
  --model llama3_8b_instruct_personamem --model-state-dict-path $R/models/icl_ft_personamem/model.pth \
  --save-root $OUT/icl_implicit_v3_seq1 --seq-len 1 --n-rollouts 400 \
  --baseline-source llama3 > $OUT/icl_implicit_v3_seq1.log 2>&1 &

wait
