#!/bin/bash
# Stage 2 of CounterFACT benchmark regeneration: format the categorized facts,
# swap in the held-out subcategories, and generate the mixed-document streams.
# Expects the stage-1 output copied to data/counterfact_with_categories/
# (see README). Run from this directory.
set -eu

python scripts/format_counterfact.py \
  --in ../../data/counterfact_with_categories \
  --out artifacts/counterfact_formatted_categorized/

python scripts/swap_hold_out_subcategories.py --params params.yaml \
  --in artifacts/counterfact_formatted_categorized/ \
  --out artifacts/counterfact_formatted_categorized_swapped/

python scripts/generate_documents.py --params params.yaml \
  --in artifacts/counterfact_formatted_categorized_swapped/ \
  --out artifacts/counterfact_doc_data_categorized/
