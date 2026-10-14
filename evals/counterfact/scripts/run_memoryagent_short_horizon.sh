#!/bin/bash
# MemoryAgent (write-time LLM filter) baselines — Table-1 short-horizon runs.
# 2 lanes x 2 GPUs. Run from evals/counterfact/.
set -u

declare -A LANES=(
  ["memoryagent_icl"]="memoryagent_icl_unwindowed|0,1|MemoryAgent ICL"
  ["memoryagent_rag"]="memoryagent_rag|2,3|MemoryAgent RAG"
)

for KEY in "${!LANES[@]}"; do
  IFS='|' read -r MODEL GPUS PRETTY <<< "${LANES[$KEY]}"
  mkdir -p "results/$KEY"
  CUDA_VISIBLE_DEVICES="$GPUS" \
  MEMORYAGENT_NOTES_PATH="results/$KEY/notes" \
  WANDB_MODE=disabled \
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/eval.py \
    --model "$MODEL" \
    --model-state-dict-path null \
    --pretty-name "$PRETTY" \
    --run-key "$KEY" \
    --outdir "results/$KEY" \
    --params=params_memoryagent_short_horizon.yaml \
    > "results/$KEY/launch.log" 2>&1 &
  echo "launched $KEY (model=$MODEL, gpus=$GPUS, pid=$!)"
done
wait
echo "ALL_SELMEM_LANES_DONE"
