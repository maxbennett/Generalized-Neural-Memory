#!/bin/bash
# 6-shard eval of Llama-3 RAG (PersonaMem prompt + top-4 retrieval) on
# data/personamem/val_ood.json. Mirrors the 40-rollout split
# [7,7,7,7,6,6] used for the PersonaMem evals.
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
PY=${PY:-python}
SCRIPT=$REPO/evals/personamem/scripts/eval_icl_pref.py
EVAL_PATH=$REPO/data/personamem/val_ood.json
OUT_DIR=$REPO/evals/personamem/results/personamem_rag
SHARDS_DIR=$OUT_DIR/_shards
mkdir -p "$SHARDS_DIR"

run_shard () {
  local gpu=$1; local offset=$2; local n=$3
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    $PY -u $SCRIPT \
      --eval-path "$EVAL_PATH" \
      --model llama3_8b_instruct_rag_personamem \
      --save-root "$SHARDS_DIR/shard_${gpu}" \
      --seq-len 10 \
      --n-rollouts $n \
      --rollout-offset $offset \
      --baseline-source memoryllm \
      > "$SHARDS_DIR/shard_${gpu}.log" 2>&1
}

run_shard 0  0 7 &
run_shard 1  7 7 &
run_shard 2 14 7 &
run_shard 3 21 7 &
run_shard 4 28 6 &
run_shard 5 34 6 &
wait
echo "[llama3_rag_eval] all shards done"
