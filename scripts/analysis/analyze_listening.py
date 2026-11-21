"""Confirmatory analysis of the listening study, exactly as pre-registered.

Written and committed BEFORE any rater judgment exists, which is the only thing that makes
"pre-registered analysis" mean anything. Every choice below is traceable to a section of
preregistration/PREREGISTRATION.md; none of it may be changed after data arrives without a
§7 deviation note in the paper.

Inputs
  --responses  CSV from the collector sheet: received_at,rater,kit,blind_id,verdict,client_ts
               verdict in {1,2,3,4}: 1 match / 2 accent differs / 3 different reading /
               4 cannot judge.
  --kits       directory of blind kits; each subdir has KEY_DO_NOT_OPEN.csv
               (blind_id,cell,arm,idx,speaker,target_word,...,is_attention_dup)

Outputs a JSON report and a human-readable summary. Order of the report is deliberate:
reliability BEFORE any system comparison (§5), because kappa bounds what the primary
outcome can resolve.

  python3 scripts/analysis/analyze_listening.py --responses r.csv --kits <dir> --out report.json
"""
import argparse, csv, glob, itertools, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/, for paths.py
import paths
from collections import defaultdict

# ---------------------------------------------------------------- pre-registered constants
N_BOOT = 10000            # §5 bootstrap resamples
ALPHA = 0.05              # §5 family-wise
CATCH_MIN = 3.0 / 4.0     # §4 (rev. 2026-09-06): known-answer catch trials, <3/4 excludes a session
ATTENTION_MIN = 0.80      # duplicate-consistency rate: REPORTED, never an exclusion criterion
SEED = 20260901           # bootstrap seed, fixed here so the report is reproducible

# §5 multiplicity: two families, Holm within each.
FAMILY_A = "superiority"      # H1 (Sarashina), H2 (Irodori), H5 (kana vs Ours, Sarashina)
FAMILY_B = "non-inferiority"  # H3 (2 backbones), H4 (UtterTune-released vs Ours, CosyVoice2)


# ---------------------------------------------------------------- exact tests, no SciPy
def _lchoose(n, k):
    from math import lgamma
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def mcnemar_exact_two_sided(b, c):
    """Exact binomial test on the discordant pairs (§5). b, c are the two discordant counts.

    Two-sided by the standard doubling of the smaller tail, clipped at 1.0. With n=0
    discordant pairs the test is undefined and returns p=1.0, which is the honest report:
    the arms made identical decisions on every item.
    """
    from math import exp
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(exp(_lchoose(n, i) + n * (-0.6931471805599453)) for i in range(k + 1))
    return min(1.0, 2.0 * tail)


def holm(pvals):
    """Holm step-down. Returns adjusted p-values in the input order (§5)."""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m, adj, running = len(pvals), [0.0] * len(pvals), 0.0
    for rank, i in enumerate(idx):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def boot_ci_paired(x, y, stat, clusters=None, raters=None, n_boot=N_BOOT, seed=SEED):
    """Paired bootstrap over the two CROSSED random factors: word and rater (§5).

    WORD. Cells D and E present the same 50 words in two speakers. Resampling the 100 rows
    independently treats the two renditions of one word as two independent observations and
    reports a CI that is roughly sqrt(2) too narrow. The resampling unit must be the word.

    RATER. Under the original design one fixed panel judged every stimulus, so rater was a
    constant and could be ignored. Under the block design (preregistration/items/blocks.json)
    each stimulus is judged by a different draw of ~5 crowdsourced raters out of ~17, and the
    inference generalizes to raters, not just to words. Ignoring that again understates the CI.
    Both factors are resampled together (a two-way cluster bootstrap): a replicate draws words
    with replacement AND raters with replacement, and keeps the rows in the intersection.

    `raters=None` reproduces the word-only bootstrap exactly, which is what the fixed-panel
    pilot data needs.
    """
    rng = random.Random(seed)
    n = len(x)
    if n == 0:
        return (float("nan"), float("nan"))
    if clusters is None:
        clusters = list(range(n))
    by_word = {}
    for i, c in enumerate(clusters):
        by_word.setdefault(c, []).append(i)
    wkeys = list(by_word)
    rkeys = sorted(set(raters)) if raters is not None else None
    vals = []
    for _ in range(n_boot):
        idx = []
        for _ in range(len(wkeys)):
            idx.extend(by_word[wkeys[rng.randrange(len(wkeys))]])
        if rkeys is not None:
            # resample raters with replacement; a rater drawn twice contributes its rows twice
            counts = {}
            for _ in range(len(rkeys)):
                r = rkeys[rng.randrange(len(rkeys))]
                counts[r] = counts.get(r, 0) + 1
            idx = [i for i in idx for _ in range(counts.get(raters[i], 0))]
        if not idx:
            continue
        vals.append(stat([x[i] for i in idx], [y[i] for i in idx]))
    if not vals:
        return (float("nan"), float("nan"))
    vals.sort()
    m = len(vals)
    return (vals[int(0.025 * m)], vals[min(m - 1, int(0.975 * m))])


def risk_diff(x, y):
    return sum(y) / len(y) - sum(x) / len(x)


def odds_ratio(x, y):
    """Haldane-Anscombe corrected; the correction is applied always, not only on zero cells,
    so the estimator does not change behaviour depending on the data it sees."""
    a, b = sum(y) + 0.5, len(y) - sum(y) + 0.5
    c, d = sum(x) + 0.5, len(x) - sum(x) + 0.5
    return (a / b) / (c / d)


def fleiss_kappa_pooled(tables):
    """Fleiss' kappa for an INCOMPLETE design: one kappa per block, pooled by stimulus count.

    Fleiss' kappa assumes a fixed set of raters judging every item. The block design gives each
    block its own rater set, so a single kappa over all 787 stimuli would mix agreement WITHIN a
    rater set with differences BETWEEN rater sets and is not interpretable. Each block's kappa is
    computed over its own stimuli and its own raters, then pooled weighted by the number of
    stimuli that contributed (blocks are equal-sized by construction, so this is nearly a plain
    mean; the weighting matters only if a block loses stimuli to attention-check exclusions).

    `tables` maps block id -> the per-item count table for that block.
    """
    ks, ws = [], []
    for _bid, table in sorted(tables.items()):
        rows = [r for r in table if sum(r) >= 2]
        if not rows:
            continue
        k = fleiss_kappa(table)
        if k == k:                       # not NaN
            ks.append(k); ws.append(len(rows))
    if not ks:
        return float("nan")
    return sum(k * w for k, w in zip(ks, ws)) / sum(ws)


def fleiss_kappa(table):
    """table: list of per-item [n_category0, n_category1, ...]; items may have unequal n."""
    items = [row for row in table if sum(row) >= 2]
    if not items:
        return float("nan")
    k = len(items[0])
    N = len(items)
    p_j = [0.0] * k
    tot = sum(sum(r) for r in items)
    for r in items:
        for j in range(k):
            p_j[j] += r[j]
    p_j = [v / tot for v in p_j]
    P_i = []
    for r in items:
        n = sum(r)
        P_i.append((sum(v * v for v in r) - n) / (n * (n - 1)))
    P_bar = sum(P_i) / N
    P_e = sum(v * v for v in p_j)
    if abs(1 - P_e) < 1e-12:
        return float("nan")
    return (P_bar - P_e) / (1 - P_e)


def cohen_kappa(a, b):
    """a, b: aligned label lists from two raters."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return float("nan")
    n = len(pairs)
    cats = sorted({x for p in pairs for x in p})
    po = sum(1 for x, y in pairs if x == y) / n
    pe = sum((sum(1 for x, _ in pairs if x == c) / n) * (sum(1 for _, y in pairs if y == c) / n)
             for c in cats)
    return (po - pe) / (1 - pe) if abs(1 - pe) > 1e-12 else float("nan")


# ---------------------------------------------------------------- data loading
def load_keys(kitdir):
    """Key on (kit, blind_id), NOT blind_id.

    Every kit numbers its items from q0001 independently, so blind_id is unique only within a
    kit: the 788 stimuli carry just 210 distinct blind_ids. Keying on blind_id alone silently
    merges cells and yields a wrong number rather than an error. The collector records the kit
    with every judgment, so the composite join is always available.
    """
    key = {}
    for f in glob.glob(os.path.join(kitdir, "*", "KEY_DO_NOT_OPEN.csv")):
        kit = os.path.basename(os.path.dirname(f))
        with open(f, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["cell"] != kit:
                    raise SystemExit(f"KEY cell {r['cell']!r} != kit dir {kit!r} in {f}")
                key[(kit, r["blind_id"])] = r
    # A missing or mistyped directory globs to nothing, and every cell then pairs zero items: the
    # report is written anyway, all nulls, and looks like a finished run.
    if not key:
        raise SystemExit(f"no KEY_DO_NOT_OPEN.csv under {kitdir}")
    return key


def load_responses(path):
    """Accept either the KV export (scripts/rater_ui/export_kv.py) or a collector CSV.

    The study stores each judgement server-side as it is made; the CSV form is kept because an
    earlier collector wrote one. Rows the outcome must never see are dropped here, at the door:
    `meta|consent` (the consent record) and the gate's own answers (`screening|...`). Catch trials
    live in cell "E" and are kept -- they are what the session exclusion is computed from.
    """
    rows = []
    if path.endswith(".json"):
        raw = json.load(open(path, encoding="utf-8"))
        src = raw.get("responses", raw) if isinstance(raw, dict) else raw
        for r in src:
            v = str(r.get("verdict", "")).strip()
            if v == "x":
                v = "4"
            kit, bid = str(r.get("kit", "")).strip(), str(r.get("blind_id", "")).strip()
            if kit == "meta" or kit == "screening" or v not in {"1", "2", "3", "4"}:
                continue
            rows.append({"rater": str(r.get("rater", "")).strip(), "item": (kit, bid),
                         "verdict": int(v), "ts": r.get("ts", ""),
                         "block": str(r.get("block", "")).strip()})
        return rows
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            v = str(r.get("verdict", "")).strip()
            if v == "x":                 # the rater UI sends 'x' for 「判断できない」 (prereg §4: option 4)
                v = "4"
            if v not in {"1", "2", "3", "4"}:
                continue
            if r["kit"].strip() in {"meta", "screening"}:
                continue
            rows.append({"rater": r["rater"].strip(),
                         "item": (r["kit"].strip(), r["blind_id"].strip()),
                         "verdict": int(v), "ts": r.get("client_ts", ""),
                         "block": str(r.get("block", "")).strip()})
    # last write wins: the UI allows a rater to revise before submitting
    dedup = {}
    for r in rows:
        dedup[(r["rater"], r["item"])] = r
    return list(dedup.values())


# ---------------------------------------------------------------- §4 rater-level gates
def catch_pass(resp, catch_key):
    """Session exclusion (§4, rev. 2026-09-06): the known-answer catch trials.

    Four items with a known correct verdict are mixed into every block, indistinguishable from the
    rest. A session below 3/4 is excluded and reported. The block's duplicate pairs measure whether
    a rater is SELF-consistent, which a consistent guesser also is; only a known answer separates
    the two, so the duplicates are reported (`consistency`) and never exclude anyone.
    """
    by_rater = defaultdict(dict)
    for r in resp:
        if r["item"][0] == "E":
            by_rater[r["rater"]][r["item"][1]] = r["verdict"]
    out = {}
    for rater in {r["rater"] for r in resp}:
        seen = by_rater.get(rater, {})
        # The key stores gold as 1 = the shown accent is the one spoken, 0 = it is not. The rater
        # answers on the three-choice scale, where 1 = matches and 2 = the accent differs; a gold of
        # 0 is therefore satisfied by verdict 2, not by the literal string "0".
        want = lambda g: 1 if str(g) in ("1", "True", "true") else 2
        hits = [1 if int(v) == want(catch_key[bid].get("gold")) else 0
                for bid, v in seen.items() if bid in catch_key]
        rate = (sum(hits) / len(hits)) if hits else None
        out[rater] = {"n_catch": len(hits), "n_correct": sum(hits), "rate": rate,
                      # a session with no catch trials recorded cannot be judged; it is reported
                      # as such rather than silently admitted
                      "passed": (rate is not None and rate >= CATCH_MIN)}
    return out


def consistency_rate(resp, key):
    """Within-rater agreement on the blocks' own duplicate pairs. Reported, not an exclusion."""
    by_rater = defaultdict(dict)
    for r in resp:
        by_rater[r["rater"]][r["item"]] = r["verdict"]
    dup_groups = defaultdict(list)
    for bid, k in key.items():
        g = (k["cell"], k["arm"], k["idx"], k["speaker"])
        dup_groups[g].append(bid)
    pairs = [v for v in dup_groups.values() if len(v) >= 2]
    out = {}
    for rater, ans in by_rater.items():
        seen = [[ans[b] for b in p if b in ans] for p in pairs]
        seen = [v for v in seen if len(v) >= 2]
        agree = sum(1 for v in seen if len(set(v)) == 1)
        out[rater] = {"n_pairs": len(seen), "n_agree": agree,
                      "rate": (agree / len(seen)) if seen else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True)
    ap.add_argument("--kits", default=None, help="directory of the blind kits (default: $DATA/blind_kits_42_norm)")
    ap.add_argument("--catch-key", default="preregistration/items/catch_KEY_DO_NOT_SHIP.json",
                    help="known-answer catch trials: blind_id -> gold (never shipped to raters)")
    ap.add_argument("--out", default="data/analysis/listening_report.json")
    ap.add_argument("--interim-min-raters", type=int, default=3,
                    help="INTERIM LOOK ONLY: run below the registered minimum of three raters. The report is stamped "
                         "as interim and exploratory; nothing it prints is a confirmatory result.")
    ap.add_argument("--round2", action="store_true",
                    help="PREREGISTRATION_v2.md item 7: cells A2..D2, family of five (tag > kana on each backbone, tag > no edit on Sarashina)")
    a = ap.parse_args()

    key = load_keys(a.kits or f"{paths.DATA}/blind_kits_42_norm")
    resp = load_responses(a.responses)
    catch_key = {r["blind_id"]: r for r in json.load(open(a.catch_key, encoding="utf-8"))} \
        if os.path.exists(a.catch_key) else {}
    rep = {"n_key_items": len(key), "n_raw_judgments": len(resp),
           "preregistered": "preregistration/PREREGISTRATION.md",
           "n_boot": N_BOOT, "seed": SEED}

    # ---- §4 session gate, applied and reported before anything else
    att = catch_pass(resp, catch_key)
    rep["catch_gate"] = att
    rep["consistency"] = consistency_rate(resp, key)
    rep["n_catch_key"] = len(catch_key)
    admitted = [r for r, v in att.items() if v["passed"]]
    rep["raters_admitted"] = sorted(admitted)
    rep["raters_excluded"] = sorted(r for r in att if r not in admitted)
    resp = [r for r in resp if r["rater"] in admitted]
    rep["n_judgments_after_gate"] = len(resp)
    if len(admitted) < 3:
        if len(admitted) >= a.interim_min_raters:
            rep["INTERIM"] = (f"{len(admitted)} raters: below the pre-registered minimum of 3. Everything below is an "
                              "interim, exploratory look; no test in it is confirmatory.")
        else:
            rep["ABORT"] = (f"only {len(admitted)} raters passed the catch-trial gate; the "
                            "pre-registered minimum is 3. No confirmatory test is reported.")
            _emit(rep, a.out)
            return

    # ---- per-item rater labels, attention clones dropped from the outcome (§4)
    real = {b: k for b, k in key.items()
            if str(k.get("is_attention_dup", "")).lower() != "true"}
    resp = [r for r in resp if r["item"][0] != "E"]   # catch trials gate sessions, never the outcome
    labels = defaultdict(dict)          # blind_id -> rater -> verdict
    for r in resp:
        if r["item"] in real:
            labels[r["item"]][r["rater"]] = r["verdict"]

    # ---- §5 reliability FIRST, before any system comparison
    raters = sorted(admitted)
    bin_of = lambda v: 1 if v == 1 else 0            # §4 primary outcome is binary
    tab = []
    for b in sorted(labels):
        vs = [bin_of(v) for v in labels[b].values() if v != 4]
        if len(vs) >= 2:
            tab.append([len(vs) - sum(vs), sum(vs)])
    # Registered form (prereg §5, rev. 2026-09-03): kappa WITHIN each block, pooled by stimulus
    # count. Under the block design each block has its own rater set, so the single global
    # kappa below mixes within-set agreement with between-set differences; it is kept only as
    # a reference figure. Blocks come from preregistration/items/blocks.json, keyed by
    # (cell, word-unit idx); a stimulus not found there is a design error and is reported.
    # Each rater's block is carried on the judgement itself (the site stores it), so the block
    # structure is read from the data rather than from a separate file that can drift out of date;
    # preregistration/items/blocks.json described an earlier 5x5 layout and no longer matches what
    # was fielded. A stimulus judged under more than one block would be a design error and is
    # reported as such.
    block_of, blocks_meta = {}, None
    _seen = defaultdict(set)
    for r in resp:
        if r.get("block", "").startswith("block-"):
            _seen[r["item"]].add(r["block"])
    rep["items_in_multiple_blocks"] = sorted(str(i) for i, v in _seen.items() if len(v) > 1)
    for item, v in _seen.items():
        block_of[item] = sorted(v)[0]
    tab_by_block, unblocked = defaultdict(list), 0
    for b in sorted(labels):
        vs = [bin_of(v) for v in labels[b].values() if v != 4]
        if len(vs) < 2:
            continue
        bid = block_of.get(b)
        if bid is None:
            unblocked += 1; continue
        tab_by_block[bid].append([len(vs) - sum(vs), sum(vs)])
    rep["reliability"] = {
        "fleiss_kappa_pooled_by_block": fleiss_kappa_pooled(tab_by_block) if tab_by_block else None,
        "kappa_by_block": {str(bid): {"fleiss_kappa": fleiss_kappa(t), "n_items": len(t)}
                           for bid, t in sorted(tab_by_block.items())},
        "n_blocks": len(tab_by_block), "n_items_without_block": unblocked,
        "fleiss_kappa_global_reference_only": fleiss_kappa(tab),
        "n_items_in_kappa": len(tab),
        "pairwise_cohen_kappa": {
            f"{x}|{y}": cohen_kappa(
                [bin_of(labels[b][x]) if x in labels[b] else None for b in sorted(labels)],
                [bin_of(labels[b][y]) if y in labels[b] else None for b in sorted(labels)])
            for x, y in itertools.combinations(raters, 2)},
    }

    # ---- consensus label (§4): majority over raters who judged the item
    consensus, ties = {}, 0
    for b, per in labels.items():
        vs = [v for v in per.values() if v != 4]
        if not vs:
            continue
        ones = sum(1 for v in vs if v == 1)
        if 2 * ones == len(vs):
            ties += 1
            continue
        consensus[b] = 1 if 2 * ones > len(vs) else 0
    rep["n_consensus_items"] = len(consensus)
    rep["n_tied_items_excluded"] = ties

    # §4(a) per-arm (2) vs (3) split: separates accent errors from reading errors
    split = defaultdict(lambda: {"2": 0, "3": 0})
    for b, per in labels.items():
        k = real[b]
        for v in per.values():
            if v in (2, 3):
                split[f"{k['cell']}|{k['arm']}"][str(v)] += 1
    rep["secondary_2v3_split"] = {k: dict(v) for k, v in split.items()}

    # ---- paired item vectors per (cell, arm)
    def vec(cell, arm, spk=None):
        """(idx, speaker) -> outcome. spk=None pools speakers; a per-speaker hypothesis must
        pass one; under the 42-voice design spk=None keeps every voice."""
        out = {}
        for b, k in real.items():
            if k["cell"] == cell and k["arm"] == arm and b in consensus:
                if spk is not None and k["speaker"] != spk:
                    continue      # KEY stores the speaker as "spk0"/"spk1", not an int
                out[(k["idx"], k["speaker"])] = consensus[b]
        return out

    def paired(cell, arm_a, arm_b, spk=None):
        """Returns (x, y, keys, clusters). keys are (idx, speaker); the CLUSTER is idx alone,
        because the same idx in two speakers is the same word measured twice (§5)."""
        va, vb = vec(cell, arm_a, spk), vec(cell, arm_b, spk)
        keys = sorted(set(va) & set(vb))
        return ([va[k] for k in keys], [vb[k] for k in keys], keys, [k[0] for k in keys])

    # ---- pre-registered hypotheses, declared as data so nothing is invented later
    SUP = [  # Family A (§5): superiority, McNemar exact two-sided, Holm over 5 tests
        ("Hk-sarashina", "A2_sarashina_difficult", "kana", "ours", None, "Sarashina: tag > kana"),
        ("Hk-cv2", "B2_cv2_difficult", "kana", "ours", None, "CosyVoice 2: tag > kana"),
        ("Hk-irodori", "C2_irodori_difficult", "kana", "ours", None, "Irodori: tag > kana"),
        ("Hk-t5gemma", "D2_t5gemma_difficult", "kana", "ours", None, "T5Gemma: tag > kana"),
        ("Hn-sarashina", "A2_sarashina_difficult", "noedit", "ours", None, "Sarashina: tag > no edit"),
    ] if a.round2 else [
        # 42-voice design (prereg rev. 2026-09-03): one held-out voice per word-unit, so every
        # test runs over ALL voices (spk=None). The old "spk0"/"spk1" filters would keep only the
        # ~2 words whose hash landed on voice #0 and silently gut the test. H6/H7 (replication
        # on a second voice) were withdrawn: with 27-31 voices per cell that property is inside
        # H1/H2. Holm now runs over three tests.
        ("H1", "A_sarashina_difficult", "noedit", "ours", None, "Sarashina, 27 voices: Ours > No-edit"),
        ("H2", "C_irodori_difficult", "noedit", "ours", None, "Irodori, 28 voices: Ours > No-edit"),
        ("H5", "A_sarashina_difficult", "kana", "ours", None,
         "Sarashina, 27 voices: Ours > kana input (the study's existential contrast)"),
]
    # Family B is ESTIMATION, not testing (revised 2026-09-01). Every one of these is at an
    # effective n of 50 distinct words, where the 95% CI half-width (~0.12) is the same size as
    # the pre-registered margin of 0.15: the test cannot resolve the question it is asked, and
    # reporting it as a test would present a resolution limit as a finding. Point estimate and
    # word-clustered CI are reported instead, and no non-inferiority verdict is issued.
    EST = [] if a.round2 else [
        ("E1", "D_sarashina_katakana", "noedit", "ours", None,
         "Sarashina: editing a different word leaves the judged word's accent intact"),
        ("E2", "B_cv2_datafree_vs_recorded", "uttertune_released", "ours", None,
         "CosyVoice2: data-free vs 15,097 recorded pairs"),
    ]

    fam_a, raw_p = [], []
    for hid, cell, arm0, arm1, spk, desc in SUP:
        x, y, ks, cl = paired(cell, arm0, arm1, spk)
        b = sum(1 for i, j in zip(x, y) if i == 1 and j == 0)   # arm0 correct, arm1 wrong
        c = sum(1 for i, j in zip(x, y) if i == 0 and j == 1)   # arm1 correct, arm0 wrong
        p = mcnemar_exact_two_sided(b, c)
        raw_p.append(p)
        fam_a.append({"id": hid, "desc": desc, "cell": cell, "speaker": spk,
                      "n_paired": len(ks), "n_distinct_words": len(set(cl)),
                      "p_" + arm0: (sum(x) / len(x) if x else None),
                      "p_" + arm1: (sum(y) / len(y) if y else None),
                      "discordant_b": b, "discordant_c": c, "p_raw": p,
                      "risk_diff": risk_diff(x, y) if x else None,
                      "risk_diff_ci95": boot_ci_paired(x, y, risk_diff, cl) if x else None,
                      "odds_ratio": odds_ratio(x, y) if x else None,
                      "odds_ratio_ci95": boot_ci_paired(x, y, odds_ratio, cl) if x else None})
    for h, p in zip(fam_a, holm(raw_p)):
        h["p_holm"] = p
        h["significant_at_0.05_family"] = bool(p < ALPHA)
    rep["family_A_superiority"] = fam_a

    # ---- pre-registered STRATIFIED SECONDARY on cell A (prereg rev. 2026-09-04). Descriptive:
    # no verdict, no Holm. Cell A is 34 heiban / 16 nucleus-bearing prescribed accents, as
    # constituted (generated, not sampled; no representativeness claim either way). Whether a
    # reading-only input already lands on heiban is what this block MEASURES; it was nearly
    # assumed instead, which would have moved the test set toward the hypothesis.
    def stratum_of(k):
        return "nucleus" if "\uff3c" in (k.get("prescribed_accent") or "") else "heiban"
    def paired_stratum(cell, arm_a, arm_b, stratum):
        keep = {(k["idx"], k["speaker"]) for b, k in real.items()
                if k["cell"] == cell and stratum_of(k) == stratum}
        x, y, ks, cl = paired(cell, arm_a, arm_b, None)
        sel = [i for i, k in enumerate(ks) if k in keep]
        return [x[i] for i in sel], [y[i] for i in sel], [ks[i] for i in sel], [cl[i] for i in sel]
    strat = []
    for arm0, arm1 in (("kana", "ours"), ("noedit", "ours")):
        for st in ("heiban", "nucleus"):
            x, y, ks, cl = paired_stratum("A2_sarashina_difficult" if a.round2 else "A_sarashina_difficult", arm0, arm1, st)
            if not x:
                strat.append({"contrast": f"{arm0} vs {arm1}", "stratum": st, "n_paired": 0}); continue
            b = sum(1 for i, j in zip(x, y) if i == 1 and j == 0); c = sum(1 for i, j in zip(x, y) if i == 0 and j == 1)
            strat.append({"contrast": f"{arm0} vs {arm1}", "stratum": st, "n_paired": len(ks),
                          "n_distinct_words": len(set(cl)),
                          "p_" + arm0: sum(x) / len(x), "p_" + arm1: sum(y) / len(y),
                          "discordant_b": b, "discordant_c": c, "p_mcnemar_unadjusted": mcnemar_exact_two_sided(b, c),
                          "risk_diff": risk_diff(x, y), "risk_diff_ci95": boot_ci_paired(x, y, risk_diff, cl),
                          "note": "descriptive; no confirmatory claim attached to either stratum"})
    rep["secondary_stratified_cellA_by_prescribed_type"] = strat

    est = []
    for hid, cell, arm0, arm1, spk, desc in EST:
        x, y, ks, cl = paired(cell, arm0, arm1, spk)
        lo, hi = boot_ci_paired(x, y, risk_diff, cl) if x else (None, None)
        est.append({"id": hid, "desc": desc, "cell": cell, "n_paired": len(ks),
                    "n_distinct_words": len(set(cl)),
                    "p_" + arm0: (sum(x) / len(x) if x else None),
                    "p_" + arm1: (sum(y) / len(y) if y else None),
                    "risk_diff": risk_diff(x, y) if x else None, "ci95": [lo, hi],
                    "reported_as": "estimate with 95% word-clustered CI; no test, no verdict"})
    rep["family_B_estimates"] = est

    # ---- §4(b) accent-only re-analysis: items where NEITHER arm was consensus-(3)
    def consensus3(b):
        vs = [v for v in labels.get(b, {}).values() if v != 4]
        return bool(vs) and sum(1 for v in vs if v == 3) * 2 > len(vs)
    acc_only = {}
    for hid, cell, arm0, arm1, spk, _ in SUP:
        ids0 = {(real[b]["idx"], real[b]["speaker"]): b
                for b in real if real[b]["cell"] == cell and real[b]["arm"] == arm0}
        ids1 = {(real[b]["idx"], real[b]["speaker"]): b
                for b in real if real[b]["cell"] == cell and real[b]["arm"] == arm1}
        keep = [k for k in set(ids0) & set(ids1)
                if not consensus3(ids0[k]) and not consensus3(ids1[k])
                and ids0[k] in consensus and ids1[k] in consensus]
        x = [consensus[ids0[k]] for k in keep]
        y = [consensus[ids1[k]] for k in keep]
        b_ = sum(1 for i, j in zip(x, y) if i == 1 and j == 0)
        c_ = sum(1 for i, j in zip(x, y) if i == 0 and j == 1)
        acc_only[hid] = {"n_items": len(keep), "p_" + arm0: (sum(x) / len(x) if x else None),
                         "p_" + arm1: (sum(y) / len(y) if y else None),
                         "p_raw": mcnemar_exact_two_sided(b_, c_) if x else None,
                         "note": "reading errors removed; the remaining contrast is accent only"}
    rep["secondary_accent_only"] = acc_only

    # ---- §5.4 sensitivity: drop items whose normative accent has no dictionary entry
    # The author's NHK check records one verdict per item: "ok" (found in the dictionary),
    # "fix" (found, and the preregistered target was corrected to match it) or "other_source"
    # (no NHK entry; the target rests on the author's own judgement). §5.4 repeats Family A
    # without the "other_source" items, so a reader can see whether the result depends on them.
    # item_id "difficult:001" is 1-indexed over the item list; KEY idx is 0-indexed.
    VSET = {"A_sarashina_difficult": "difficult", "B_cv2_datafree_vs_recorded": "difficult",
            "C_irodori_difficult": "difficult", "D_sarashina_katakana": "katakana",
            "E_irodori_katakana": "katakana", "A2_sarashina_difficult": "difficult",
            "B2_cv2_difficult": "difficult", "C2_irodori_difficult": "difficult", "D2_t5gemma_difficult": "difficult"}
    # Read the RELEASED record, not the author GUI's raw state: preregistration/items/
    # target_verification.json is the artifact a replicator has, and it carries both the
    # original and the corrected target so the correction itself stays auditable.
    TV = "preregistration/items/target_verification.json"
    drop, vstat = defaultdict(set), {}
    if os.path.exists(TV):
        tv = json.load(open(TV, encoding="utf-8"))
        for setname, rec in tv["sets"].items():
            bad = {int(x["item_id"].split(":")[1]) - 1 for x in rec.get("non_ok_items", [])
                   if x["verdict"] == "other_source"}
            vstat[setname] = {k: v for k, v in rec.items() if k != "non_ok_items"}
            if len(bad) != rec.get("n_other_source", len(bad)):
                raise SystemExit(f"{TV}: n_other_source disagrees with non_ok_items for {setname}")
            for cell, which in VSET.items():
                if which == setname:
                    drop[cell] = bad
    else:
        vstat = {"MISSING": TV}
    rep["author_verification"] = vstat
    rep["sensitivity_author_assigned"] = _sensitivity(SUP, real, consensus, drop)

    _emit(rep, a.out)


def _sensitivity(SUP, real, consensus, drop_by_cell):
    """§5.4: repeat Family A excluding items whose normative accent has no dictionary entry."""
    out = {}
    for hid, cell, arm0, arm1, spk, _ in SUP:
        dropped = drop_by_cell.get(cell, set())
        g = defaultdict(dict)
        for b, k in real.items():
            if k["cell"] == cell and k["arm"] in (arm0, arm1) and b in consensus:
                if int(k["idx"]) in dropped or (spk is not None and k["speaker"] != spk):
                    continue
                g[(k["idx"], k["speaker"])][k["arm"]] = consensus[b]
        pairs = [v for v in g.values() if arm0 in v and arm1 in v]
        x = [v[arm0] for v in pairs]
        y = [v[arm1] for v in pairs]
        b_ = sum(1 for i, j in zip(x, y) if i == 1 and j == 0)
        c_ = sum(1 for i, j in zip(x, y) if i == 0 and j == 1)
        out[hid] = {"n_items": len(pairs), "n_dropped_item_indices": len(dropped),
                    "p_" + arm0: (sum(x) / len(x) if x else None),
                    "p_" + arm1: (sum(y) / len(y) if y else None),
                    "p_raw": mcnemar_exact_two_sided(b_, c_) if x else None}
    return out


def _emit(rep, out):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(rep, ensure_ascii=False, indent=1, default=str))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
