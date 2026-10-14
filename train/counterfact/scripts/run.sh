#!/bin/bash
# Train on the CounterFACT benchmark. Pass a params file (default params_gnm.yaml):
# Defaults to params_gnm.yaml. Run from train/counterfact/.
set -u

GPUS=$(python -c "import yaml; print(yaml.safe_load(open('${1:-params_gnm.yaml}'))['distributed']['cuda_visible_devices'])")
NPROC=$(python -c "import yaml; print(yaml.safe_load(open('${1:-params_gnm.yaml}'))['distributed']['nproc_per_node'])")
PORT=$(python -c "import yaml; print(yaml.safe_load(open('${1:-params_gnm.yaml}'))['distributed']['master_port'])")

CUDA_VISIBLE_DEVICES=$GPUS MASTER_PORT=$PORT \
torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC scripts/train.py --params=${1:-params_gnm.yaml}
