#!/bin/bash
# Seed-robustness protocol (paper appendix): for each seed, train the GNM and
# ICL-FT PersonaMem models, precompute each model's own no-memory baseline
# letters, then run the sharded seq-10 eval. Seeds reported in the paper:
# 42 (= the released main checkpoints) plus 43, 44, 0, 1, 2.
#
#   scripts/run_seed_robustness.sh 0 1        # run seeds 0 and 1
#
# 4-GPU DDP training. GPUs are picked dynamically: each phase waits until
# enough GPUs are free (<2GB used), so the chain coexists with other jobs.
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
T=$REPO/train/personamem
S=$REPO/evals/personamem/scripts
PY=${PY:-python}
TORCHRUN=${TORCHRUN:-torchrun}
EVAL=$REPO/data/personamem/test.json
OUT=$REPO/evals/personamem/paper_results/seed_robustness
mkdir -p $OUT
export WANDB_MODE=offline OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd $REPO

log() { echo "[seeds] $(date +%m-%d\ %H:%M) $1" >> $OUT/chain.log; }

# Echo a comma list of N free GPU indices (<2GB used); poll every 3 min until
# available. Logs a waiting heartbeat every 30 min.
pick_gpus() {
  local n=$1 waited=0
  while true; do
    local free
    free=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
           | awk -F', ' '$2 < 2000 {print $1}')
    local cnt
    cnt=$(echo "$free" | grep -c '[0-9]')
    if [ "$cnt" -ge "$n" ]; then
      echo "$free" | head -n "$n" | paste -sd,
      return
    fi
    if [ $((waited % 1800)) -eq 0 ]; then
      log "waiting for $n free GPUs (currently $cnt free)"
    fi
    sleep 180; waited=$((waited + 180))
  done
}

run_gnm() {
  local s=$1 port=$2
  local stem=gnm_seed${s}
  local run=gnm_personamem_seed${s}
  local ckpt=$REPO/saved_models/${run}/${run}_best.pth

  local gpus; gpus=$(pick_gpus 4)
  log "GNM seed $s training start on GPUs $gpus"
  CUDA_VISIBLE_DEVICES=$gpus PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $TORCHRUN --rdzv-backend=c10d --rdzv-endpoint=localhost:$port --nnodes=1 --nproc_per_node=4 \
    $T/scripts/train_personamem.py --params $T/params_gnm_seed${s}.yaml \
    --save-dir $REPO/saved_models/${run} \
    > $T/training_gnm_seed${s}.log 2>&1
  log "GNM seed $s training exit $? (training_gnm_seed${s}.log)"

  if [ -f "$ckpt" ]; then
    local g1; g1=$(pick_gpus 1)
    log "GNM seed $s own-baseline precompute start on GPU $g1"
    CUDA_VISIBLE_DEVICES=$g1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY -u $REPO/data_pipelines/personamem/scripts/precompute_trained_baseline_letters.py \
      --mode memoryllm --state-dict-path $ckpt \
      --output-field ${stem}_baseline_letter --cache-name ${stem}_baseline_letter_cache.json \
      --eval-paths $EVAL --batch-size 16 > $OUT/${stem}_base.log 2>&1
    log "GNM seed $s baseline exit $?"

    local g2; g2=$(pick_gpus 2)
    local ga=${g2%,*} gb=${g2#*,}
    log "GNM seed $s eval start (2 shards) on GPUs $g2"
    ( CUDA_VISIBLE_DEVICES=$ga PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      $PY -u $S/eval_personamem.py --eval-path $EVAL --model memoryllm_chat \
        --model-state-dict-path $ckpt --output-path $OUT/${stem}_shard0.json \
        --seq-len 10 --n-rollouts 20 --rollout-offset 0 --baseline-source memoryllm \
        > $OUT/${stem}_shard0.log 2>&1 ) &
    ( CUDA_VISIBLE_DEVICES=$gb PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      $PY -u $S/eval_personamem.py --eval-path $EVAL --model memoryllm_chat \
        --model-state-dict-path $ckpt --output-path $OUT/${stem}_shard1.json \
        --seq-len 10 --n-rollouts 20 --rollout-offset 20 --baseline-source memoryllm \
        > $OUT/${stem}_shard1.log 2>&1 ) &
    wait
    log "GNM seed $s eval done"
  else
    log "ERROR: GNM seed $s best ckpt missing, skipping eval"
  fi
}

run_icl() {
  local s=$1 port=$2
  local stem=icl_ft_seed${s}
  local run=icl_ft_personamem_seed${s}
  local ckpt=$REPO/saved_models/${run}/${run}_best.pth

  local gpus; gpus=$(pick_gpus 4)
  log "ICL-FT seed $s training start on GPUs $gpus"
  CUDA_VISIBLE_DEVICES=$gpus PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $TORCHRUN --rdzv-backend=c10d --rdzv-endpoint=localhost:$port --nnodes=1 --nproc_per_node=4 \
    $T/scripts/train_personamem.py --params $T/params_icl_ft_seed${s}.yaml \
    --save-dir $REPO/saved_models/${run} \
    > $T/training_icl_ft_seed${s}.log 2>&1
  log "ICL-FT seed $s training exit $? (training_icl_ft_seed${s}.log)"

  if [ -f "$ckpt" ]; then
    local g1; g1=$(pick_gpus 1)
    log "ICL-FT seed $s own-baseline precompute start on GPU $g1"
    CUDA_VISIBLE_DEVICES=$g1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY -u $REPO/data_pipelines/personamem/scripts/precompute_trained_baseline_letters.py \
      --mode llama3 --state-dict-path $ckpt \
      --output-field ${stem}_baseline_letter --cache-name ${stem}_baseline_letter_cache.json \
      --eval-paths $EVAL --batch-size 16 > $OUT/${stem}_base.log 2>&1
    log "ICL-FT seed $s baseline exit $?"

    local g1b; g1b=$(pick_gpus 1)
    log "ICL-FT seed $s eval start on GPU $g1b"
    CUDA_VISIBLE_DEVICES=$g1b PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY -u $S/eval_icl_pref.py --eval-path $EVAL --model llama3_8b_instruct_personamem \
      --model-state-dict-path $ckpt --save-root $OUT/${stem} \
      --seq-len 10 --n-rollouts 40 --baseline-source memoryllm \
      > $OUT/${stem}.log 2>&1
    log "ICL-FT seed $s eval done"
  else
    log "ERROR: ICL-FT seed $s best ckpt missing, skipping eval"
  fi
}

PORT=29811
for s in "$@"; do
  run_gnm "$s" $PORT; PORT=$((PORT + 2))
  run_icl "$s" $PORT; PORT=$((PORT + 2))
done
log "SEED CHAIN COMPLETE ($*)"
