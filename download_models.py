#!/usr/bin/env python3
"""Download pre-trained models from Hugging Face and convert to .pth format."""

import os
import json
import subprocess
import torch
from safetensors.torch import load_file

MODELS = {
    "maxbennett/generalized-neural-memory-gnm-all": "models/gnm_all",
    "maxbennett/generalized-neural-memory-icl-all": "models/icl_all",
    "maxbennett/generalized-neural-memory-rag-all": "models/rag_all",
    "maxbennett/generalized-neural-memory-gnm-all-ablation": "models/gnm_all_ablation",
    "maxbennett/generalized-neural-memory-gnm-facts": "models/gnm_facts",
    "maxbennett/generalized-neural-memory-icl-facts": "models/icl_facts",
    "maxbennett/generalized-neural-memory-rag-facts": "models/rag_facts",
}


def download_model(repo_id: str, local_dir: str):
    """Download a model from Hugging Face."""
    print(f"Downloading {repo_id} to {local_dir}...")
    subprocess.run(
        ["hf", "download", repo_id, "--local-dir", local_dir],
        check=True
    )


def convert_safetensors_to_pth(model_dir: str):
    """Convert sharded safetensors to a single .pth file."""
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
    print("=" * 60)
    print("Downloading and converting GNM models")
    print("=" * 60)
    
    for repo_id, local_dir in MODELS.items():
        print(f"\n{'=' * 60}")
        print(f"Processing: {repo_id}")
        print("=" * 60)
        
        # Download
        download_model(repo_id, local_dir)
        
        # Convert
        convert_safetensors_to_pth(local_dir)
    
    print("\n" + "=" * 60)
    print("All models downloaded and converted!")
    print("=" * 60)


if __name__ == "__main__":
    main()