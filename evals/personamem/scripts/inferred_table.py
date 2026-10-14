"""Autonomous infer->execute table: seq-len 1 conditions.

All runs are seq-len 1 (each doc memorized/contexted alone, its own probes only),
released GNM / ICL-FT PersonaMem checkpoints. Probes are keyed by SHA-256 of
(query+options) — rollout indices differ from the paper's seq-10 runs, but the
probe set is identical, so baseline letters and the paper joint collision filter
(pref != GNM's AND != ICL-FT's own no-context pick, from rag/results.json)
carry over exactly. Metrics per condition: acc (learn probes), preserved (pred ==
own no-context pick, ignore probes), leak (pred == preference letter, ignore),
P(leak|dev) (leaks among deviations; chance 33.3), Score = HM(acc, 100-leak).
Exact McNemar between selected condition pairs.
"""
import hashlib
import json
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
D = REPO / "evals/personamem/paper_results/short_horizon"
RES = REPO / "evals/personamem/paper_results/inferred_instructions"

ph = lambda q, o: hashlib.sha256(f"Q:{q}\nA:{o['A']}\nB:{o['B']}\nC:{o['C']}\nD:{o['D']}".encode()).hexdigest()
hm = lambda a, b: 2 * a * b / (a + b) if a + b else 0.0

ref = json.load(open(D / "rag/results.json"))
info = {}
for r in ref:
    h = ph(r["user_query"], r["options"])
    info[h] = {"pref": r["preference_aligned_letter"],
               "gnm_base": r["memoryllm_trained_baseline_letter"],
               "icl_base": r["icl_ft_baseline_letter"]}
keep = lambda h: h in info and info[h]["pref"] != info[h]["gnm_base"] and info[h]["pref"] != info[h]["icl_base"]


def load(path):
    recs = json.load(open(path))
    out = {}
    for r in recs:
        if r["seq_idx_of_doc"] != r["seq_idx_of_io_pairs"]:
            continue
        h = ph(r["user_query"], r["options"])
        if keep(h):
            out[h] = r
    return out


def metrics(by_hash, own_key):
    nl = acc = nig = pres = leak = ldev = 0
    outcomes = {}  # hash -> dict for McNemar pairing
    for h, r in by_hash.items():
        pref, ob = info[h]["pref"], info[h][own_key]
        p = r["predicted_letter"]
        if r["learn_or_ignore"] == "learn":
            nl += 1; ok = p == pref; acc += ok
            outcomes[h] = {"kind": "learn", "acc": ok}
        else:
            nig += 1
            b, l = p == ob, p == pref
            pres += b; leak += l; ldev += l and not b
            outcomes[h] = {"kind": "ignore", "pres": b, "leak": l}
    dev = nig - pres
    return {"acc": 100*acc/nl, "preserved": 100*pres/nig, "leak": 100*leak/nig,
            "pld": 100*ldev/dev if dev else 0.0, "score": hm(100*acc/nl, 100-100*leak/nig),
            "nl": nl, "nig": nig, "outcomes": outcomes}


def mcnemar(o1, o2, kind, field):
    b = c = 0
    for h in o1:
        if h not in o2 or o1[h]["kind"] != kind:
            continue
        x, y = o1[h][field], o2[h][field]
        b += x and not y
        c += y and not x
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = min(1.0, 2 * sum(comb(n, k) for k in range(0, min(b, c) + 1)) / 2**n)
    return b, c, p


CONDS = {
    "GNM oracle-explicit":   (RES / "gnm_oracle_seq1.json",                 "gnm_base"),
    "GNM implicit (v3)":     (RES / "gnm_implicit_v3_seq1.json",            "gnm_base"),
    "GNM inferred-llama3":   (RES / "gnm_inferred_llama3_seq1.json",        "gnm_base"),
    "GNM inferred-qwen3":    (RES / "gnm_inferred_qwen3_seq1.json",         "gnm_base"),
    "GNM inferred-generic-llama3": (RES / "gnm_inferred_llama3_author_generic_seq1.json", "gnm_base"),
    "GNM inferred-generic-qwen3": (RES / "gnm_inferred_qwen3_author_generic_seq1.json", "gnm_base"),
    "GNM inferred-intent-llama3": (RES / "gnm_inferred_llama3_author_intent_seq1.json", "gnm_base"),
    "GNM inferred-intent-qwen3": (RES / "gnm_inferred_qwen3_author_intent_seq1.json", "gnm_base"),
    "GNM inferred-intent-end-llama3": (RES / "gnm_inferred_llama3_author_intent_end_seq1.json", "gnm_base"),
    "GNM inferred-purpose-llama3": (RES / "gnm_inferred_llama3_author_purpose_seq1.json", "gnm_base"),
    "GNM inferred-purpose-qwen3": (RES / "gnm_inferred_qwen3_author_purpose_seq1.json", "gnm_base"),
    "ICL-FT oracle-explicit": (RES / "icl_oracle_seq1/results.json",        "icl_base"),
    "ICL-FT implicit (v3)":  (RES / "icl_implicit_v3_seq1/results.json",    "icl_base"),
    "ICL-FT inferred-llama3": (RES / "icl_inferred_llama3_seq1/results.json", "icl_base"),
    "ICL-FT inferred-qwen3": (RES / "icl_inferred_qwen3_seq1/results.json", "icl_base"),
    "ICL-FT inferred-generic-llama3": (RES / "icl_inferred_llama3_author_generic_seq1/results.json", "icl_base"),
    "ICL-FT inferred-generic-qwen3": (RES / "icl_inferred_qwen3_author_generic_seq1/results.json", "icl_base"),
    "ICL-FT inferred-intent-llama3": (RES / "icl_inferred_llama3_author_intent_seq1/results.json", "icl_base"),
    "ICL-FT inferred-intent-qwen3": (RES / "icl_inferred_qwen3_author_intent_seq1/results.json", "icl_base"),
    "ICL-FT inferred-intent-end-llama3": (RES / "icl_inferred_llama3_author_intent_end_seq1/results.json", "icl_base"),
    "ICL-FT inferred-purpose-llama3": (RES / "icl_inferred_llama3_author_purpose_seq1/results.json", "icl_base"),
    "ICL-FT inferred-purpose-qwen3": (RES / "icl_inferred_qwen3_author_purpose_seq1/results.json", "icl_base"),
}

rows = {}
print("| condition | acc% | preserved% | leak% | P(leak|dev)% | Score | n (learn/ignore) |")
print("|---|---:|---:|---:|---:|---:|---|")
for name, (path, own) in CONDS.items():
    if not path.exists():
        print(f"| {name} | (missing: {path.name}) |")
        continue
    m = metrics(load(path), own)
    rows[name] = m
    print(f"| {name} | {m['acc']:.1f} | {m['preserved']:.1f} | {m['leak']:.1f} | "
          f"{m['pld']:.1f} | {m['score']:.1f} | {m['nl']}/{m['nig']} |")

PAIRS = [
    ("GNM inferred-intent-end-llama3", "ICL-FT inferred-intent-end-llama3"),
    ("GNM inferred-intent-end-llama3", "GNM inferred-intent-llama3"),
    ("GNM inferred-purpose-llama3", "ICL-FT inferred-purpose-llama3"),
    ("GNM inferred-purpose-qwen3", "ICL-FT inferred-purpose-qwen3"),
    ("GNM inferred-purpose-llama3", "GNM oracle-explicit"),
    ("GNM inferred-purpose-llama3", "GNM implicit (v3)"),
    ("GNM inferred-purpose-qwen3", "GNM implicit (v3)"),
    ("GNM inferred-generic-llama3", "ICL-FT inferred-generic-llama3"),
    ("GNM inferred-generic-qwen3", "ICL-FT inferred-generic-qwen3"),
    ("GNM inferred-intent-llama3", "GNM oracle-explicit"),
    ("GNM inferred-intent-llama3", "GNM implicit (v3)"),
    ("GNM inferred-intent-llama3", "ICL-FT inferred-intent-llama3"),
    ("GNM inferred-generic-llama3", "GNM oracle-explicit"),
    ("GNM inferred-generic-llama3", "GNM inferred-llama3"),
    ("GNM inferred-generic-llama3", "GNM implicit (v3)"),
    ("GNM inferred-llama3", "GNM oracle-explicit"),
    ("GNM inferred-qwen3", "GNM oracle-explicit"),
    ("GNM inferred-llama3", "GNM implicit (v3)"),
    ("GNM inferred-llama3", "ICL-FT inferred-llama3"),
    ("GNM inferred-qwen3", "ICL-FT inferred-qwen3"),
    ("ICL-FT inferred-llama3", "ICL-FT oracle-explicit"),
]
print("\nExact McNemar (paired on shared probes):")
for a, bname in PAIRS:
    if a not in rows or bname not in rows:
        continue
    for kind, field in [("learn", "acc"), ("ignore", "leak"), ("ignore", "pres")]:
        b, c, p = mcnemar(rows[a]["outcomes"], rows[bname]["outcomes"], kind, field)
        lbl = {"acc": "acc", "leak": "leak", "pres": "preserved"}[field]
        print(f"  {a} vs {bname:<26} [{lbl:<9}] b={b:<4} c={c:<4} p={p:.4g}")
