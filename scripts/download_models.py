#!/usr/bin/env python3
"""Download pre-trained models from Hugging Face and convert to .pth format.

  python download_models.py                 # everything
  python download_models.py --counterfact   # CounterFACT-benchmark models only
  python download_models.py --personamem    # PersonaMem-benchmark models only
"""

import argparse
import json
import os
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import torch
from safetensors.torch import load_file

COUNTERFACT_MODELS = {
    "maxbennett/generalized-neural-memory-gnm-counterfact": "models/gnm_counterfact",
    "maxbennett/generalized-neural-memory-iclft-counterfact": "models/icl_ft_counterfact",
    "maxbennett/generalized-neural-memory-ragft-counterfact": "models/rag_ft_counterfact",
    "maxbennett/generalized-neural-memory-gnm-counterfact-ablation": "models/gnm_counterfact_ablation",
}

PERSONAMEM_MODELS = {
    "maxbennett/generalized-neural-memory-gnm-personamem": "models/gnm_personamem",
    "maxbennett/generalized-neural-memory-iclft-personamem": "models/icl_ft_personamem",
    "maxbennett/generalized-neural-memory-gnm-personamem-ablation": "models/gnm_personamem_ablation",
}


def download_model(repo_id: str, local_dir: str):
    """Download a model from Hugging Face."""
    local_dir = os.path.join(REPO, local_dir)
    print(f"Downloading {repo_id} to {local_dir}...")
    subprocess.run(
        ["hf", "download", repo_id, "--local-dir", local_dir],
        check=True
    )


def convert_safetensors_to_pth(model_dir: str):
    """Convert sharded safetensors to a single .pth file."""
    model_dir = os.path.join(REPO, model_dir)
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    output_path = os.path.join(model_dir, "model.pth")

    if not os.path.exists(index_path):
        print(f"  Skipping conversion: {index_path} not found")
        return

    if os.path.exists(output_path):
        print(f"  {output_path} already exists, skipping conversion")
        return

    print(f"  Converting to {output_path}...")

    with open(index_path) as f:
        index = json.load(f)

    shard_files = set(index["weight_map"].values())

    state_dict = {}
    for shard in sorted(shard_files):
        shard_path = os.path.join(model_dir, shard)
        print(f"    Loading {shard}...")
        state_dict.update(load_file(shard_path))

    print(f"    Saving {output_path}...")
    torch.save(state_dict, output_path)
    print(f"  Done!")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counterfact", action="store_true",
                    help="only the CounterFACT-benchmark models")
    ap.add_argument("--personamem", action="store_true",
                    help="only the PersonaMem-benchmark models")
    args = ap.parse_args()

    models = {}
    if args.counterfact or not args.personamem:
        models.update(COUNTERFACT_MODELS)
    if args.personamem or not args.counterfact:
        models.update(PERSONAMEM_MODELS)

    print("=" * 60)
    print("Downloading and converting GNM models")
    print("=" * 60)

    for repo_id, local_dir in models.items():
        print(f"\n{'=' * 60}")
        print(f"Processing: {repo_id}")
        print("=" * 60)

        download_model(repo_id, local_dir)
        convert_safetensors_to_pth(local_dir)

    print("\n" + "=" * 60)
    print("All models downloaded and converted!")
    print("=" * 60)


if __name__ == "__main__":
    main()
