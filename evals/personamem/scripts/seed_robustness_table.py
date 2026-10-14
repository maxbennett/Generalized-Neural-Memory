"""Seed-robustness table (paper appendix): per-seed joint collision filter.

For each seed s: drop diag probes where the preference-aligned letter equals
EITHER GNM-seed-s's OR ICL-FT-seed-s's own no-context pick (same convention as
the paper's joint filter, applied per seed pair). Metrics per model: acc,
preserved (vs own no-context pick), leak, P(leak|dev), Score = HM(acc, 100-leak).
Seed 42 = the released paper checkpoints; seeds 43/44/0/1/2 = independent retrains.
"""
import hashlib
import json
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
O = REPO / "evals/personamem/paper_results/seed_robustness"
D = REPO / "evals/personamem/paper_results/short_horizon"
ph = lambda q, o: hashlib.sha256(f"Q:{q}\nA:{o['A']}\nB:{o['B']}\nC:{o['C']}\nD:{o['D']}".encode()).hexdigest()
KEY = lambda r: (r["rollout_idx"], r["seq_idx_of_doc"], r["seq_idx_of_io_pairs"], r["pair_idx"])
hm = lambda a, b: 2 * a * b / (a + b) if a + b else 0.0

ref = json.load(open(D / "rag/results.json"))
gnm42_base = {ph(r["user_query"], r["options"]): r["memoryllm_trained_baseline_letter"] for r in ref}
icl42_base = {ph(r["user_query"], r["options"]): r["icl_ft_baseline_letter"] for r in ref}

def cache_f(name):
    return json.load(open(REPO / f"data/baseline_letter_caches/{name}.json"))

def load_shards(stem):
    return (json.load(open(O / f"{stem}_shard0.json")) + json.load(open(O / f"{stem}_shard1.json")))

SEEDS = {
    42: {"gnm_recs": json.load(open(D / "gnm/results.json")), "gnm_base": gnm42_base,
         "icl_recs": json.load(open(D / "icl_ft/results.json")),            "icl_base": icl42_base},
    43: {"gnm_recs": load_shards("gnm_seed43"), "gnm_base": cache_f("gnm_seed43_baseline_letter_cache"),
         "icl_recs": json.load(open(O / "icl_ft_seed43/results.json")), "icl_base": cache_f("icl_ft_seed43_baseline_letter_cache")},
    44: {"gnm_recs": load_shards("gnm_seed44"), "gnm_base": cache_f("gnm_seed44_baseline_letter_cache"),
         "icl_recs": json.load(open(O / "icl_ft_seed44/results.json")), "icl_base": cache_f("icl_ft_seed44_baseline_letter_cache")},
}
# Seeds 0/1 (2026-07-31 extension): include each pair once BOTH sides are done.
for extra in (0, 1, 2):
    try:
        SEEDS[extra] = {
            "gnm_recs": load_shards(f"gnm_seed{extra}"),
            "gnm_base": cache_f(f"gnm_seed{extra}_baseline_letter_cache"),
            "icl_recs": json.load(open(O / f"icl_ft_seed{extra}/results.json")),
            "icl_base": cache_f(f"icl_ft_seed{extra}_baseline_letter_cache"),
        }
    except FileNotFoundError:
        print(f"(seed {extra} pair not complete yet — omitted)\n")

def metrics(recs, own_base, keep):
    nl = acc = nig = pres = leak = leak_dev = 0
    for r in recs:
        if r["seq_idx_of_doc"] != r["seq_idx_of_io_pairs"]:
            continue
        h = ph(r["user_query"], r["options"])
        if not keep(h):
            continue
        ob = own_base.get(h)
        pa, p = r["preference_aligned_letter"], r["predicted_letter"]
        if r["learn_or_ignore"] == "learn":
            nl += 1; acc += p == pa
        else:
            nig += 1; pres += p == ob; leak += p == pa; leak_dev += (p == pa) and (p != ob)
    dev = nig - pres
    return {"acc": 100*acc/nl, "preserved": 100*pres/nig, "leak": 100*leak/nig,
            "pld": 100*leak_dev/dev if dev else 0.0, "score": hm(100*acc/nl, 100-100*leak/nig),
            "n_learn": nl, "n_ignore": nig}

rows = {"GNM": {}, "ICL-FT": {}}
print("| seed | model | acc% | preserved% | leak% | P(leak\\|dev)% | Score | n (learn/ignore) |")
print("|---|---|---:|---:|---:|---:|---:|---|")
for s, d in SEEDS.items():
    # per-seed joint filter: pref differs from BOTH this seed's GNM and ICL-FT no-context picks
    gb, ib = d["gnm_base"], d["icl_base"]
    # collision letters looked up against probe hash; probes missing from either cache are dropped
    def keep(h, gb=gb, ib=ib):
        # use each seed's records to obtain the preference letter lazily — handled by caller filter below
        return True
    # build per-hash preference letter from ref (fixed probe set)
    pref = {ph(r["user_query"], r["options"]): r["preference_aligned_letter"] for r in ref}
    keepf = lambda h, gb=gb, ib=ib, pref=pref: h in pref and pref[h] != gb.get(h) and pref[h] != ib.get(h)
    for model, recs, base in [("GNM", d["gnm_recs"], gb), ("ICL-FT", d["icl_recs"], ib)]:
        m = metrics(recs, base, keepf)
        rows[model][s] = m
        print(f"| {s} | {model} | {m['acc']:.1f} | {m['preserved']:.1f} | {m['leak']:.1f} | "
              f"{m['pld']:.1f} | {m['score']:.1f} | {m['n_learn']}/{m['n_ignore']} |")
for model in ["GNM", "ICL-FT"]:
    ms = rows[model]
    if len(ms) >= 2:
        cells = []
        for k in ["acc", "preserved", "leak", "pld", "score"]:
            v = [ms[s][k] for s in ms]
            cells.append(f"{st.mean(v):.1f} ± {st.stdev(v):.1f}")
        print(f"| mean±sd | {model} | " + " | ".join(cells) + " | |")
