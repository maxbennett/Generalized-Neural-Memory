#!/bin/bash
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 WANDB_MODE=offline
export LAG_STATES="25,50,100,150,200,275,350,425,500,550,600,650,699"
export LAG_PAIRS=4 LAG_REFUSALS=1
LANE=$1
EXP3=models/gnm_counterfact/model.pth
RAG=models/rag_ft_counterfact/model.pth
ICL=models/icl_ft_counterfact/model.pth
L=evals/counterfact/results/long_horizon
run() {
  echo "[lane $LANE] $(date +%H:%M) start $1$4 f$3" >> $L/final_lanes.log
  LAG_TAG=$4 CUDA_VISIBLE_DEVICES=$LANE python scripts/retention_probe.py $1 $3 $2 > $L/final_$1$4_f$3.log 2>&1
  echo "[lane $LANE] $(date +%H:%M) done  $1$4 f$3 (exit $?)" >> $L/final_lanes.log
}
case $LANE in
  0) run gnm_long_horizon $EXP3 0 ""; run llama3_8b_instruct_rag $RAG 0 "" ;;
  1) run gnm_long_horizon $EXP3 1 ""; run llama3_8b_instruct_rag $RAG 1 "" ;;
  2) run gnm_long_horizon $EXP3 2 ""; run llama3_8b_instruct_rag $RAG 2 "" ;;
  4) run llama3_8b_instruct_windowed $ICL 0 "" ;;
  5) run llama3_8b_instruct_windowed $ICL 1 "" ;;
  6) run llama3_8b_instruct_windowed $ICL 2 "" ;;
esac
echo "[lane $LANE] $(date +%H:%M) LANE DONE" >> $L/final_lanes.log
