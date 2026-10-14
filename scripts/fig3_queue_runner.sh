#!/bin/bash
# Figure-3 MemoryAgent lanes: long-horizon fact/sel/spec runs plus refusal runs
# (2 collate seeds) for MemoryAgent-ICL and MemoryAgent-RAG. Complements
# final_eval_lanes.sh (GNM / ICL-FT / RAG-FT facts), refusal_power_lanes.sh,
# Claims only GPUs with zero compute processes
# (shared box); phase 0 smoke-tests the code path before the real queue.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 WANDB_MODE=offline HF_HUB_OFFLINE=1
PY=${PY:-python}

EXP3=models/gnm_counterfact/model.pth
RAG=models/rag_ft_counterfact/model.pth
ICL=models/icl_ft_counterfact/model.pth
L=evals/counterfact/results/long_horizon
RUN=evals/counterfact/results/fig3_queue
mkdir -p "$L" "$RUN"

DENSE_STATES="25,50,100,150,200,275,350,425,500,550,600,650,699"
REFUSAL_STATES="25,50,100,150,200,275,350,425,450,475,500,525,550,575,600,625,650,675,699"

# job spec: mode|model|ckpt|tag|file|seed   (mode: acc / refusal / smoke_acc)
JOBS=()
# -- phase 0: smokes (must pass; runner aborts if any fails)
JOBS+=("smoke_acc|memoryagent_icl|none|_smoke|0|4242")
JOBS+=("smoke_acc|memoryagent_rag|none|_smoke|0|4242")
# -- phase 1: memoryagent fact/sel/spec runs
for f in 0 1 2; do
  JOBS+=("acc|memoryagent_icl|none||$f|4242")
  JOBS+=("acc|memoryagent_rag|none||$f|4242")
done
# -- phase 2: memoryagent refusal (2 collate seeds, power-rerun state ladder)
for seed in 4242 1001; do
  for f in 0 1 2; do
    st=""; [ "$seed" != 4242 ] && st="_s${seed}"
    JOBS+=("refusal|memoryagent_icl|none|_refusal${st}|$f|$seed")
    JOBS+=("refusal|memoryagent_rag|none|_refusal${st}|$f|$seed")
  done
done

NEXT_FILE=$RUN/next_idx
LOCK=$RUN/queue.lock
[ -f "$NEXT_FILE" ] || echo 0 > "$NEXT_FILE"
SMOKE_FAILED=$RUN/SMOKE_FAILED

pop_job () {  # prints job index, -1 (queue empty), or -2 (waiting on smoke gate)
  (
    flock 9
    idx=$(cat "$NEXT_FILE")
    if [ "$idx" -ge "${#JOBS[@]}" ]; then
      echo -1
    elif [ "$idx" -ge 2 ] && ! { [ -f "$RUN/smoke_ok_0" ] && [ -f "$RUN/smoke_ok_1" ]; }; then
      echo -2
    else
      echo $((idx+1)) > "$NEXT_FILE"; echo "$idx"
    fi
  ) 9>"$LOCK"
}

run_job () {
  local gpu=$1 idx=$2
  IFS='|' read -r mode model ckpt tag f seed <<< "${JOBS[$idx]}"
  local logf="$RUN/job${idx}_${model}${tag}_f${f}.log"
  echo "[gpu $gpu] $(date +%H:%M) start #$idx ${mode} ${model}${tag} f${f}" >> "$RUN/queue.log"
  local env_common=(LAG_TAG="$tag" LAG_SEED="$seed" CUDA_VISIBLE_DEVICES="$gpu")
  case $mode in
    acc)
      env "${env_common[@]}" LAG_STATES="$DENSE_STATES" LAG_PAIRS=4 LAG_REFUSALS=0 \
        LAG_PARAMS=evals/counterfact/params_long_horizon_f0.yaml \
        $PY scripts/retention_probe.py "$model" "$f" "$ckpt" > "$logf" 2>&1 ;;
    refusal)
      env "${env_common[@]}" LAG_STATES="$REFUSAL_STATES" LAG_PAIRS=4 LAG_REFUSALS=1 LAG_REFUSALS_ONLY=1 \
        LAG_PARAMS=evals/counterfact/params_long_horizon_refusal.yaml \
        $PY scripts/retention_probe.py "$model" "$f" "$ckpt" > "$logf" 2>&1 ;;
    smoke_acc)
      env "${env_common[@]}" LAG_SEQ_LEN=30 LAG_STATES="25,29" LAG_LAGS="1,3,10,25" LAG_PAIRS=2 LAG_REFUSALS=0 \
        LAG_PARAMS=evals/counterfact/params_long_horizon_f0.yaml \
        $PY scripts/retention_probe.py "$model" "$f" "$ckpt" > "$logf" 2>&1 ;;
  esac
  local rc=$?
  echo "[gpu $gpu] $(date +%H:%M) done  #$idx ${model}${tag} f${f} (exit $rc)" >> "$RUN/queue.log"
  if [ $rc -ne 0 ] && [[ $mode == smoke_* ]]; then
    touch "$SMOKE_FAILED"
    echo "[gpu $gpu] SMOKE FAILED on #$idx — aborting queue" >> "$RUN/queue.log"
  fi
  # smoke output sanity: probes/formats key non-empty; passing smokes open the gate
  if [ $rc -eq 0 ] && [[ $mode == smoke_* ]]; then
    if $PY - "$L/${model}${tag}_f${f}.json" "$mode" <<'EOF' >> "$RUN/queue.log" 2>&1
import json, sys
d = json.load(open(sys.argv[1]))
key = "probes"
n = len(d.get(key, []))
assert n > 0, f"smoke produced 0 {key} records"
print(f"smoke OK: {sys.argv[1]} has {n} {key} records")
EOF
    then touch "$RUN/smoke_ok_${idx}"
    else touch "$SMOKE_FAILED"
    fi
  fi
}

gpu_worker () {  # runs jobs on one gpu until queue empty or smoke failure
  local gpu=$1
  while true; do
    [ -f "$SMOKE_FAILED" ] && break
    # phase gate: real jobs (idx>=2) only start after both smokes verified OK
    idx=$(pop_job)
    [ "$idx" -lt 0 ] && break
    run_job "$gpu" "$idx"
  done
  rm -f "$RUN/worker_gpu${gpu}.pid"
}

echo "[runner] $(date +%H:%M) queue started, ${#JOBS[@]} jobs" >> "$RUN/queue.log"
while true; do
  [ -f "$SMOKE_FAILED" ] && { echo "[runner] smoke failure — exiting" >> "$RUN/queue.log"; exit 1; }
  next=$( (flock 9; cat "$NEXT_FILE") 9>"$LOCK" )
  active=$(ls "$RUN"/worker_gpu*.pid 2>/dev/null | wc -l)
  if [ "$next" -ge "${#JOBS[@]}" ] && [ "$active" -eq 0 ]; then
    echo "[runner] $(date +%H:%M) ALL JOBS DONE" >> "$RUN/queue.log"; break
  fi
  if [ "$next" -lt "${#JOBS[@]}" ]; then
    for gpu in 0 1 2 3 4 5 6 7; do
      [ -f "$RUN/worker_gpu${gpu}.pid" ] && kill -0 "$(cat "$RUN/worker_gpu${gpu}.pid")" 2>/dev/null && continue
      rm -f "$RUN/worker_gpu${gpu}.pid"
      # free = zero compute processes on this GPU
      busy=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader | wc -l)
      if [ "$busy" -eq 0 ]; then
        gpu_worker "$gpu" &
        echo $! > "$RUN/worker_gpu${gpu}.pid"
        echo "[runner] $(date +%H:%M) claimed gpu $gpu (worker $!)" >> "$RUN/queue.log"
      fi
    done
  fi
  sleep 60
done
