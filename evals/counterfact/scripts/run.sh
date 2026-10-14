#!/bin/bash
# Run every entry in a params file's `evals:` list sequentially (Table-1 protocol
# by default). Run from evals/counterfact/.
#   scripts/run.sh                      # uses params_short_horizon.yaml
#   scripts/run.sh params_comp_gen_refusals.yaml
set -u

PARAMS=${1:-params_short_horizon.yaml}

python - "$PARAMS" <<'EOF' | while IFS='|' read -r MODEL KEY PRETTY SD; do
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
for e in cfg["evals"]:
    sd = e.get("model_state_dict_path") or "null"
    print(f'{e["model"]}|{e["run_key"]}|{e["pretty_name"]}|{sd}')
EOF
  GPUS=$(python -c "import yaml,sys; print(yaml.safe_load(open('$PARAMS'))['distributed']['cuda_visible_devices'])")
  NPROC=$(python -c "import yaml,sys; print(yaml.safe_load(open('$PARAMS'))['distributed']['nproc_per_node'])")
  mkdir -p "results/$KEY"
  echo "=== eval $KEY (model=$MODEL)"
  CUDA_VISIBLE_DEVICES=$GPUS WANDB_MODE=${WANDB_MODE:-disabled} \
  torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC scripts/eval.py \
    --model "$MODEL" \
    --model-state-dict-path "$SD" \
    --pretty-name "$PRETTY" \
    --run-key "$KEY" \
    --outdir "results/$KEY" \
    --params="$PARAMS"
done
echo "ALL_EVALS_DONE"
