#!/bin/bash
# PersonaMem (Table 2) eval: 40 episodes of Val-OOD (data/personamem/val_ood.json) at seq_len=10,
# sharded [7,7,7,7,6,6] across 6 GPUs. This is the exact protocol used for the
# paper's PersonaMem numbers (GNM, ICL-FT, and the ablation all use it).
#
# Usage:
#   run_personamem_eval.sh <model> <ckpt-or-null> <out-dir> [baseline-source]
# Examples (from repo root):
#   evals/personamem/scripts/run_personamem_eval.sh \
#     memoryllm_chat models/gnm_personamem/model.pth results/personamem_gnm memoryllm
#   evals/personamem/scripts/run_personamem_eval.sh \
#     memoryllm_chat models/gnm_personamem_ablation/model.pth results/personamem_gnm_ablation memoryllm
# Merge the 6 shards afterwards with merge_personamem_eval.py, then aggregate
# with build_summary.py.
set -uo pipefail

MODEL=${1:?model registry key (e.g. memoryllm_chat)}
CKPT=${2:?checkpoint path relative to repo root, or 'null'}
OUT=${3:?output dir relative to repo root}
BASELINE=${4:-memoryllm}

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
PY=${PY:-python}
SCRIPT=$REPO/evals/personamem/scripts/eval_personamem.py
EVAL_PATH=$REPO/data/personamem/val_ood.json
[ "$CKPT" != "null" ] && CKPT=$REPO/$CKPT
SHARDS_DIR=$REPO/$OUT/_shards
mkdir -p "$SHARDS_DIR"

run_shard () {
  local gpu=$1; local offset=$2; local n=$3
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    $PY -u $SCRIPT \
      --eval-path "$EVAL_PATH" \
      --model "$MODEL" \
      --model-state-dict-path "$CKPT" \
      --output-path "$SHARDS_DIR/shard_${gpu}.json" \
      --seq-len 10 \
      --n-rollouts $n \
      --rollout-offset $offset \
      --baseline-source "$BASELINE" \
      > "$SHARDS_DIR/shard_${gpu}.log" 2>&1
}

run_shard 0  0 7 &
run_shard 1  7 7 &
run_shard 2 14 7 &
run_shard 3 21 7 &
run_shard 4 28 6 &
run_shard 5 34 6 &
wait
echo "[personamem_eval] all shards done -> $SHARDS_DIR"
