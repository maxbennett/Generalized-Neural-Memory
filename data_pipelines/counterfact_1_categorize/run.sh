#!/bin/bash
# Stage 1 of CounterFACT benchmark regeneration: categorize the raw CounterFACT
# dump (expects HF-datasets arrow files under data/raw/counterfact/).
# Run from this directory.
set -eu

python add_category_to_raw_counterfact_from_mapping.py \
  --in ../../data/raw/counterfact/ \
  --out artifacts/counterfact_raw_categorized/ \
  --mapping mapping.json

python run_category_analytics.py \
  --in artifacts/counterfact_raw_categorized/ \
  --out artifacts/counterfact_raw_categorized/counterfact_raw_categorized_analytics.json
