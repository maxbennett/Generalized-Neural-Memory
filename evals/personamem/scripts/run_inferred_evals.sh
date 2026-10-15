#!/bin/bash
# Eval runs for the authored-instruction condition, seq-len 1.
# Usage: run_inferred_evals.sh <llama3|qwen3> <gpu_gnm> <gpu_icl> [author|author_generic]
set -u
UP=$1; G1=$2; G2=$3; MODE=${4:-author}
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
OUT=$R/evals/personamem/paper_results/inferred_instructions
EVAL=$R/data/personamem/inferred_eval/val_ood_inferred_${UP}_${MODE}.json
if [ "$MODE" != "author" ]; then UP=${UP}_${MODE}; fi
PY=${PY:-python}
export WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $R/evals/personamem/scripts

CUDA_VISIBLE_DEVICES=$G1 $PY -u eval_personamem.py \
  --eval-path $EVAL --model memoryllm_chat \
  --model-state-dict-path $R/models/gnm_personamem/model.pth \
  --output-path $OUT/gnm_inferred_${UP}_seq1.json --seq-len 1 --n-rollouts 400 \
  --baseline-source memoryllm > $OUT/gnm_inferred_${UP}_seq1.log 2>&1 &

CUDA_VISIBLE_DEVICES=$G2 $PY -u eval_icl_pref.py \
  --eval-path $EVAL --model llama3_8b_instruct_personamem \
  --model-state-dict-path $R/models/icl_ft_personamem/model.pth \
  --save-root $OUT/icl_inferred_${UP}_seq1 --seq-len 1 --n-rollouts 400 \
  --baseline-source llama3 > $OUT/icl_inferred_${UP}_seq1.log 2>&1 &

wait
