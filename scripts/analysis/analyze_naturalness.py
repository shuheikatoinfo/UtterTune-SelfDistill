"""Registered analysis of the same-sentence naturalness preference (PREREGISTRATION.md, naturalness
amendments).

Input is the KV export written by scripts/rater_ui/export_kv.py and the item key that pins which
side of each pair is the edited rendition. The statistic is the share preferring the unedited
rendition minus the share preferring the edited one, ties split evenly, per backbone; the interval
is a bootstrap over pairs and raters (the same resampling as the registered interim look, so the two
are comparable); the registered non-inferiority margin is 0.15 on the upper limit.

Rules that decide what enters, all fixed before the data they act on were seen:
  * every judgement is kept, complete sitting or not (no gate exists for this task);
  * a rater's side balance is reported and excludes no one;
  * a second sitting by the same platform account is excluded whole (--exclude-rater), the account
    match being made from the platform's worker list, never from the answers;
  * --until reproduces the interim run on the judgements that existed at that time;
  * --exclude-speeders (added 2026-09-15, after the data were seen; the primary analysis by the
    author's decision, the all-kept run reported beside it): a sitting whose median interval between consecutive judgements
    is shorter than the SHORTEST pair's two clips played once (4.7 s) cannot have heard both clips
    of a typical pair, and is excluded whole. The rule reads only timestamps, never the answers or
    the arm key.

    python3 scripts/analysis/analyze_naturalness.py --responses kv.json \
        --exclude-rater <rater-id> --exclude-speeders --out data/analysis/naturalness_report.json
"""
import argparse, collections, datetime, json, os, random, statistics, sys, wave
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/, for paths.py
import paths

MARGIN = 0.15
N_BOOT = 10000
SEED = 0
BACKBONES = ["sarashina", "irodori", "cv2", "t5gemma"]


def load(path, until=None):
    raw = json.load(open(path, encoding="utf-8"))
    src = raw.get("responses", raw) if isinstance(raw, dict) else raw
    last = {}
    for r in src:
        if r.get("kit") != "nat":
            continue
        ts = r.get("ts", "")
        if until and ts >= until:
            continue
        last[(r["rater"], r["blind_id"])] = (r.get("verdict"), ts)   # last write wins
    return last


def stat(xs):
    n = len(xs)
    ours = sum(1 for x in xs if x == 1)
    ne = sum(1 for x in xs if x == -1)
    tie = n - ours - ne
    d = ((ne + tie / 2) - (ours + tie / 2)) / n if n else float("nan")
    return {"n": n, "ours": ours, "noedit": ne, "tie": tie, "D": d}


def boot(rows, n_boot=N_BOOT, seed=SEED):
    rnd = random.Random(seed)
    pairs = sorted({p for _, p, _ in rows})
    raters = sorted({r for r, _, _ in rows})
    by = collections.defaultdict(list)
    for r, p, x in rows:
        by[(r, p)].append(x)
    ds = []
    for _ in range(n_boot):
        ps = [rnd.choice(pairs) for _ in pairs]
        rs = set(rnd.choice(raters) for _ in raters)
        xs = [x for p in ps for r in rs for x in by.get((r, p), [])]
        if xs:
            ds.append(stat(xs)["D"])
    ds.sort()
    return ds[int(0.025 * len(ds))], ds[int(0.975 * len(ds))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True)
    ap.add_argument("--items", default="preregistration/items/naturalness_ab_easy50.json")
    ap.add_argument("--exclude-rater", action="append", default=[],
                    help="rater id of a sitting excluded whole (second sitting of one account)")
    ap.add_argument("--until", help="ISO timestamp; keep judgements made before it (interim run)")
    ap.add_argument("--exclude-speeders", action="store_true",
                    help="sensitivity: drop sittings whose median response interval is below the shortest pair's duration")
    ap.add_argument("--clips", default=None, help="directory of the kit's clips (default: $DATA/naturalness_kit_norm)")
    ap.add_argument("--clip-durations", default=None,
                    help="JSON {clip file name: seconds}; used instead of reading the clips (the released results ship one)")
    ap.add_argument("--exclude-items", help="JSON whose 'trained_word_items' keys are pair ids to drop whole: pairs "
                    "whose edited word was in that backbone's teacher inventory")
    ap.add_argument("--out", default="data/analysis/naturalness_report.json")
    a = ap.parse_args()

    key = json.load(open(a.items, encoding="utf-8"))
    items = {i["id"]: i for i in key["items"]}
    resp = load(a.responses, a.until)

    # response speed per sitting, from timestamps alone
    known = json.load(open(a.clip_durations)) if a.clip_durations else None
    clips = None if known is not None else (a.clips or f"{paths.DATA}/naturalness_kit_norm")
    dur = {}
    for pid in items:
        t = 0.0
        for side in "AB":
            name = f"{pid}{side}.wav"
            if known is not None:
                t += known.get(name, 0.0)
                continue
            f = os.path.join(clips, name)
            if os.path.exists(f):
                with wave.open(f) as w:
                    t += w.getnframes() / w.getframerate()
        dur[pid] = t
    shortest = min(v for v in dur.values() if v > 0) if any(dur.values()) else None
    seq = collections.defaultdict(list)
    for (r, pid), (v, ts) in resp.items():
        if pid in items and ts:
            seq[r].append(datetime.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp())
    speed = {}
    for r, ts in seq.items():
        ts.sort()
        gaps = [b - x for x, b in zip(ts, ts[1:]) if b - x < 600]      # breaks longer than 10 min are not responses
        speed[r] = statistics.median(gaps) if gaps else None
    speeders = sorted(r for r, m in speed.items() if shortest and m is not None and m < shortest and len(seq[r]) >= 10)
    if a.exclude_speeders:
        a.exclude_rater = list(a.exclude_rater) + speeders

    dropped_items = set()
    if a.exclude_items:
        dropped_items = set(json.load(open(a.exclude_items, encoding="utf-8"))["trained_word_items"])
    rows, dropped_excluded = [], collections.Counter()
    started = set()
    for (r, pid), (v, ts) in resp.items():
        it = items.get(pid)
        if not it or pid in dropped_items:
            continue
        started.add(r)
        if r in a.exclude_rater:
            dropped_excluded[r] += 1
            continue
        if v == "x":
            x = 0
        elif v in ("A", "B"):
            x = 1 if it[v] == "ours" else -1
        else:
            continue
        rows.append((r, pid, x))

    per_pair = collections.Counter(p for _, p, _ in rows)
    per_rater = collections.defaultdict(list)          # answered pairs, before any item exclusion
    for (r, pid), (v, ts) in resp.items():
        if pid in items and r not in a.exclude_rater and v in ("A", "B", "x"):
            per_rater[r].append(v)
    # a sitting is complete when it covers every pair of the group it was assigned; the groups are
    # not in the key, so completeness is approximated by the rater's count reaching the smallest
    # group size (47) -- the payment report is the authority on completion, this is for the summary
    complete = sum(1 for r, xs in per_rater.items() if len(xs) >= 47)

    rep = {"responses": a.responses, "until": a.until, "excluded_raters": a.exclude_rater,
           "excluded_items_trained_word": sorted(dropped_items),
           "speeders_flagged": speeders, "speeders_excluded": bool(a.exclude_speeders),
           "shortest_pair_s": shortest,
           "median_response_interval_s": {r: (round(m, 1) if m is not None else None) for r, m in sorted(speed.items())},
           "judgements_dropped_by_exclusion": dict(dropped_excluded),
           "sittings_started": len(started), "sittings_complete_approx": complete,
           "n_judgements": len(rows), "n_pairs": len(per_pair),
           "judgements_per_pair": {"min": min(per_pair.values()), "median": statistics.median(per_pair.values()),
                                   "max": max(per_pair.values())},
           "margin": MARGIN, "n_boot": N_BOOT, "seed": SEED, "by_backbone": {}, "same_reading_stratum": {},
           "rater_side_balance": {}}

    for bb in BACKBONES + ["ALL"]:
        sub = [t for t in rows if bb == "ALL" or items[t[1]]["backbone"] == bb]
        s = stat([x for _, _, x in sub])
        lo, hi = boot(sub)
        s.update({"ci95": [lo, hi], "non_inferior": hi < MARGIN})
        rep["by_backbone"][bb] = s
        sub2 = [t for t in sub if items[t[1]]["same_reading"]]
        if bb != "ALL":
            s2 = stat([x for _, _, x in sub2])
            lo2, hi2 = boot(sub2)
            s2.update({"ci95": [lo2, hi2], "pairs": len({p for _, p, _ in sub2})})
            rep["same_reading_stratum"][bb] = s2
    rep["same_reading_pairs_total"] = sum(1 for i in items.values() if i["same_reading"] and i["id"] not in dropped_items)

    # side balance, per rater, reported and never used to exclude (registered)
    side = collections.defaultdict(collections.Counter)
    for (r, pid), (v, _) in resp.items():
        if pid in items and r not in a.exclude_rater:
            side[r][v] += 1
    rep["rater_side_balance"] = {r: dict(c) for r, c in sorted(side.items())}

    json.dump(rep, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"speeders flagged (median interval < {shortest:.1f}s): {speeders}  excluded={a.exclude_speeders}")
    print(f"judgements {rep['n_judgements']}  sittings started {rep['sittings_started']}  "
          f"complete~ {complete}  per pair min/med/max "
          f"{rep['judgements_per_pair']['min']}/{rep['judgements_per_pair']['median']}/{rep['judgements_per_pair']['max']}")
    print(f"{'backbone':<10}{'n':>6}{'ours':>6}{'noedit':>7}{'tie':>5}{'D':>8}{'95% CI':>20}  NI")
    for bb, s in rep["by_backbone"].items():
        ci = "[%+.3f,%+.3f]" % tuple(s["ci95"])
        print(f"{bb:<10}{s['n']:>6}{s['ours']:>6}{s['noedit']:>7}{s['tie']:>5}{s['D']:>+8.3f}"
              f"{ci:>20}  {'yes' if s['non_inferior'] else 'no'}")
    print("same-reading stratum:", {bb: f"{s['D']:+.2f} ({s['pairs']} pairs)" for bb, s in rep["same_reading_stratum"].items()})
    print("->", a.out)


if __name__ == "__main__":
    main()
