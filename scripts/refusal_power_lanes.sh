#!/bin/bash
# Refusal P/R variance reruns: 4 systems x 3 files x 4 collate seeds
# = 48 seq-700 refusal-only rollouts. Same protocol as the original wave
# (params_long_horizon_refusal.yaml, LAG_PAIRS=4) plus 6 extra check-states
# >=450 to fatten the lag-400 cell. Old seed-4242 data remains valid and
# pool-compatible (evals/counterfact/results/long_horizon/).
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 WANDB_MODE=offline
export LAG_STATES="25,50,100,150,200,275,350,425,450,475,500,525,550,575,600,625,650,675,699"
export LAG_PAIRS=4 LAG_REFUSALS=1 LAG_REFUSALS_ONLY=1
export LAG_PARAMS=evals/counterfact/params_long_horizon_refusal.yaml
PY=${PY:-python}

EXP3=models/gnm_counterfact/model.pth
RAG=models/rag_ft_counterfact/model.pth
ICL=models/icl_ft_counterfact/model.pth
L=evals/counterfact/results/long_horizon
mkdir -p $L

# job list: model ckpt tagprefix file seed
JOBS=()
for seed in 1001 1002 1003 1004; do
  for f in 0 1 2; do
    JOBS+=("gnm_long_horizon|$EXP3|_refusal_s${seed}|$f|$seed")
    JOBS+=("memoryllm_chat|$EXP3|_dense_refusal_s${seed}|$f|$seed")
    JOBS+=("llama3_8b_instruct_rag|$RAG|_refusal_s${seed}|$f|$seed")
    JOBS+=("llama3_8b_instruct_windowed|$ICL|_refusal_s${seed}|$f|$seed")
  done
done

run_lane () {
  local lane=$1
  for (( i=lane; i<${#JOBS[@]}; i+=8 )); do
    IFS='|' read -r model ckpt tag f seed <<< "${JOBS[$i]}"
    echo "[lane $lane] $(date +%H:%M) start ${model}${tag} f${f}" >> $L/power_lanes.log
    LAG_TAG=$tag LAG_SEED=$seed CUDA_VISIBLE_DEVICES=$lane \
      $PY scripts/retention_probe.py $model $f $ckpt \
      > $L/power_${model}${tag}_f${f}.log 2>&1
    echo "[lane $lane] $(date +%H:%M) done  ${model}${tag} f${f} (exit $?)" >> $L/power_lanes.log
  done
  echo "[lane $lane] $(date +%H:%M) LANE DONE" >> $L/power_lanes.log
}

for lane in 0 1 2 3 4 5 6 7; do
  run_lane $lane &
done
wait
echo "[all] $(date +%H:%M) ALL LANES DONE" >> $L/power_lanes.log
