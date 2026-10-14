#!/usr/bin/env python3
"""Download the benchmark datasets from Hugging Face into the paths the
configs expect.

  python download_data.py                # both benchmarks
  python download_data.py --counterfact  # CounterFACT benchmark + eval streams
  python download_data.py --personamem   # PersonaMem benchmark (train + eval)
  python download_data.py --results      # also fetch precomputed paper results
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# dataset repo -> local dir (paths match what evals/ and train/ params expect)
DATASETS = {
    "counterfact": [
        ("maxbennett/generalized-neural-memory-counterfact", "data/counterfact"),
    ],
    "personamem": [
        ("maxbennett/generalized-neural-memory-personamem", "data/personamem"),
    ],
}

# The counterfact repo nests files under train/ val/ test/ retention/; the
# configs expect a flat data/counterfact/. Flatten after download.
FLATTEN = {"data/counterfact"}


def flatten(local_dir: str):
    root = REPO / local_dir
    for sub in ("train", "val", "test", "retention"):
        d = root / sub
        if not d.is_dir():
            continue
        for f in d.iterdir():
            dest = root / f.name
            if not dest.exists():
                shutil.move(str(f), str(dest))
        try:
            d.rmdir()
        except OSError:
            pass



def fetch(repo_id: str, local_dir: str):
    local_dir = str(REPO / local_dir)
    print(f"Downloading {repo_id} -> {local_dir}")
    subprocess.run(
        ["hf", "download", repo_id, "--repo-type", "dataset",
         "--local-dir", local_dir],
        check=True
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counterfact", action="store_true")
    ap.add_argument("--personamem", action="store_true")
    args = ap.parse_args()

    groups = []
    if args.counterfact or not args.personamem:
        groups.append("counterfact")
    if args.personamem or not args.counterfact:
        groups.append("personamem")

    for g in groups:
        for repo_id, local_dir in DATASETS[g]:
            fetch(repo_id, local_dir)
            if local_dir in FLATTEN:
                flatten(local_dir)

    print("Done.")


if __name__ == "__main__":
    main()
