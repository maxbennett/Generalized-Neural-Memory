#!/bin/bash
# Sequence-length-1 reference rows for the inferred-instruction table (appendix):
# oracle-explicit and implicit conditions for the released GNM / ICL-FT PersonaMem
# checkpoints. The inferred conditions themselves are run by run_inferred_evals.sh.
set -u
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
OUT=$R/evals/personamem/paper_results/inferred_instructions
PY=${PY:-python}
S=$R/evals/personamem/scripts
export WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $S

CUDA_VISIBLE_DEVICES=0 $PY -u eval_personamem.py \
  --eval-path $R/data/personamem/val_ood.json \
  --model memoryllm_chat --model-state-dict-path $R/models/gnm_personamem/model.pth \
  --output-path $OUT/gnm_oracle_seq1.json --seq-len 1 --n-rollouts 400 \
  --baseline-source memoryllm > $OUT/gnm_oracle_seq1.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 $PY -u eval_personamem.py \
  --eval-path $R/data/personamem/implicit_eval/val_ood_implicit.json \
  --model memoryllm_chat --model-state-dict-path $R/models/gnm_personamem/model.pth \
  --output-path $OUT/gnm_implicit_seq1.json --seq-len 1 --n-rollouts 400 \
  --baseline-source memoryllm > $OUT/gnm_implicit_seq1.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 $PY -u eval_icl_pref.py \
  --eval-path $R/data/personamem/val_ood.json \
  --model llama3_8b_instruct_personamem --model-state-dict-path $R/models/icl_ft_personamem/model.pth \
  --save-root $OUT/icl_oracle_seq1 --seq-len 1 --n-rollouts 400 \
  --baseline-source llama3 > $OUT/icl_oracle_seq1.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 $PY -u eval_icl_pref.py \
  --eval-path $R/data/personamem/implicit_eval/val_ood_implicit.json \
  --model llama3_8b_instruct_personamem --model-state-dict-path $R/models/icl_ft_personamem/model.pth \
  --save-root $OUT/icl_implicit_seq1 --seq-len 1 --n-rollouts 400 \
  --baseline-source llama3 > $OUT/icl_implicit_seq1.log 2>&1 &

wait
