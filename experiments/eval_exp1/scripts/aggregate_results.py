import yaml, sys, argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--results-root", required=True)
ap.add_argument("--selector", required=True)
ap.add_argument("--this-run-id", required=True)
ap.add_argument("--out-csv", required=True)
args = ap.parse_args()

# --- Load params ---
with open("params.yaml") as f:
    params = yaml.safe_load(f)
run_cfg = params.get("run", {})

run_id = run_cfg.get("run_id")
aggregate_on = run_cfg.get("aggregate_on")

# --- Simple logic ---
if aggregate_on is None:
    if not run_id:
        sys.exit("❌ ERROR: run_id is null and aggregate_on not set — cannot proceed.")
    aggregate_on = run_id
elif aggregate_on == "null":
    # In case YAML parses it weirdly as a string
    if not run_id:
        sys.exit("❌ ERROR: run_id is null and aggregate_on explicitly 'null'.")
    aggregate_on = run_id
else:
    # If aggregate_on is manually provided, enforce that it must exist
    if not Path(f"results/{aggregate_on}").exists():
        sys.exit(f"❌ ERROR: results/{aggregate_on} does not exist. Provide a valid run_id.")

# --- Print for visibility ---
print(f"[aggregate_results] Using aggregate_on={aggregate_on}")

# --- continue with the rest of your script ---
out_csv = Path(f"results/{aggregate_on}/aggregated_results.csv")
out_csv.parent.mkdir(parents=True, exist_ok=True)