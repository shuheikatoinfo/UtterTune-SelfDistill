#!/usr/bin/env python3
"""Round-2 objective contrasts on the 319-item test half (PREREGISTRATION_v2.md item 5).

Input: the per-item output of measure_reading_all.py (--arms-json ...), i.e. a JSON with
backbones -> arm -> {n, hit, per_item:[{i, hit, hyp}]}.
Output (markdown to stdout): per backbone the arm accuracies, tag-kana and tag-none with a word-clustered
bootstrap CI (connected components of the (word, reading) graph; 10,000 resamples, seed 20260901) and an
exact two-sided McNemar p; tag-UT on CosyVoice 2; the voice-disagreement statistic over the 150 pairs of
items sharing a word and reading under different voices; and the count of inaudible outputs per arm.
"""
import argparse, collections, glob, itertools, json, re, os
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ITEMS = f"{REPO}/data/derived/objective_test/items.jsonl"
SPK = f"{REPO}/preregistration/items/speaker_assignment42_obj319.json"
BB = [("sara42g", "Sarashina"), ("cv242g", "CosyVoice 2"), ("iro42g", "Irodori"), ("t5g42g", "T5Gemma")]


def components(items):
    n = len(items); par = list(range(n))
    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    groups = collections.defaultdict(list)
    for i, it in enumerate(items):
        groups[("s", it["surface"])].append(i); groups[("r", it["reading"])].append(i)
    for g in groups.values():
        for a in g[1:]:
            par[f(a)] = f(g[0])
    comp = collections.defaultdict(list)
    for i in range(n):
        comp[f(i)].append(i)
    return list(comp.values())


def contrast(a, b, comps, n_boot=10000, seed=20260901):
    from scipy.stats import binomtest
    d = a - b; rng = np.random.default_rng(seed); vals = []
    for _ in range(n_boot):
        idx = np.concatenate([comps[k] for k in rng.integers(0, len(comps), len(comps))])
        vals.append(d[idx].mean())
    lo, hi = np.percentile(vals, [2.5, 97.5])
    b01 = int(((a == 0) & (b == 1)).sum()); b10 = int(((a == 1) & (b == 0)).sum())
    p = binomtest(b10, b10 + b01, 0.5).pvalue if b10 + b01 else 1.0
    return d.mean(), lo, hi, p, b10, b01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="measure_reading_all.py output JSON")
    ap.add_argument("--arms-json", default=None, help="arm map, for the inaudible-output count; $DATA in a path is expanded")
    a = ap.parse_args()
    items = [json.loads(l) for l in open(ITEMS)]; n = len(items)
    comps = components(items)
    R = json.load(open(a.results))["backbones"]
    def hits(bb, arm):
        h = np.zeros(n, dtype=int)
        for p in R[bb][arm]["per_item"]:
            h[p["i"]] = 1 if p["hit"] else 0
        return h
    print(f"items {n}, word-reading components {len(comps)}\n")
    print("| backbone | no edit | plain kana | tag (ours) | tag - kana [95% CI], p | tag - none [95% CI], p |")
    print("|---|---|---|---|---|---|")
    for bb, name in BB:
        if bb not in R: continue
        t, k, z = hits(bb, "ours"), hits(bb, "kana"), hits(bb, "noedit")
        ck = contrast(t, k, comps); cz = contrast(t, z, comps)
        print(f"| {name} | {z.mean():.3f} | {k.mean():.3f} | {t.mean():.3f} | "
              f"{ck[0]:+.3f} [{ck[1]:+.3f}, {ck[2]:+.3f}], {ck[3]:.2g} | {cz[0]:+.3f} [{cz[1]:+.3f}, {cz[2]:+.3f}], {cz[3]:.2g} |")
    if "cv242g" in R and "uttertune_released" in R["cv242g"]:
        t, u = hits("cv242g", "ours"), hits("cv242g", "uttertune_released"); c = contrast(t, u, comps)
        print(f"\nCosyVoice 2 released UtterTune: {u.mean():.3f}; tag - UT {c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}], p={c[3]:.2g} (discordant {c[4]} vs {c[5]})")
    # voice disagreement: pairs sharing the benchmark (kanji, reading) key under different voices
    spk = {int(i): v[0] for i, v in json.load(open(SPK))["cells"]["OBJ319"].items()}
    key = lambda it: re.sub(r"_\d+$", "", it["key"])
    grp = collections.defaultdict(list)
    for i, it in enumerate(items):
        grp[key(it)].append(i)
    pv = [(i, j) for g in grp.values() for i, j in itertools.combinations(g, 2) if spk[i] != spk[j]]
    print(f"\nVoice disagreement over {len(pv)} same-word pairs under different voices (share of pairs whose hits differ):")
    for bb, name in BB:
        if bb not in R: continue
        dis = lambda h: sum(h[i] != h[j] for i, j in pv) / len(pv)
        print(f"  {name}: tag {dis(hits(bb,'ours')):.3f}  none {dis(hits(bb,'noedit')):.3f}  kana {dis(hits(bb,'kana')):.3f}")
    if a.arms_json:
        import soundfile as sf
        print("\nInaudible outputs (peak < 0.05):")
        for bb, arms in json.load(open(a.arms_json)).items():
            for arm, root in arms.items():
                if "uttertune" in arm: continue
                root = os.path.expandvars(root)
                files = glob.glob(f"{root}/**/{arm}/*.wav", recursive=True)
                if bb == "iro42g" and arm == "kana": files = glob.glob(f"{root}/kana/*.wav")
                q = sum(np.abs(sf.read(fp)[0]).max() < 0.05 for fp in files)
                print(f"  {bb} {arm}: {q}/{len(files)}")


if __name__ == "__main__":
    main()
