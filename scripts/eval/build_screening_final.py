#!/usr/bin/env python3
"""Freeze the screening set from the items the author's dry runs validated.

An item is only usable if the audio actually carries the accent the item claims. That cannot be
assumed from the synthesis request -- two dry runs showed it is often false -- and it cannot be
checked automatically, so it is checked by ear, once per item, before recruitment opens.

Kept: every item whose gold matched the author's verdict in every dry run it appeared in.
Dropped, with reasons recorded in the output:
  scr02 三時 / scr04 一人 / scr08 汁物 / scr11 二回  the base did not render the dictionary type
  scr19 茄子                                        the reading itself was wrong (ナスビ -> ナス)
  scrc03 手紙 / scrc05 魚                            rendered with a fall, not heiban
This is validity filtering of an instrument, not selection on an outcome: no listening data exists
yet, and the author's own answers can never be evidence about this set's difficulty afterwards.

Selection is then 5 items per accent type (頭高/中高/平板/尾高) with gold balanced 10/10, taken in
the file order of each type so the choice is not tuned to anything.
"""
import argparse, hashlib, json, os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/, for paths.py
import paths

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = [("preregistration/items/screening_20.json", f"{paths.DATA}/screening_v2"),
       ("preregistration/items/screening_v4_heiban_odaka.json", f"{paths.DATA}/screening_v4")]
# A mismatch between the author's dry-run verdict and an item's gold means one of two things: the
# audio does not carry the accent the item claims, or the author slipped. They are not the same
# defect and cannot be told apart from the answer file, so every mismatch is re-listened to before
# anything is discarded. That re-listen reinstated 手紙 and 魚, whose audio is heiban after all.
DROP_REASON = {
    "scr02": "re-listened: base did not render the dictionary accent",
    "scr04": "re-listened: base did not render the dictionary accent",
    "scr08": "re-listened: base did not render the dictionary accent",
    "scr11": "re-listened: base did not render the dictionary accent",
    "scr19": "reading itself wrong: ナスビ rendered as ナス (also caught by ASR)"}
REINSTATED = {
    "scrc03": "re-listened: audio is heiban as the dictionary says; the dry-run answer was a slip",
    "scrc05": "re-listened: audio is heiban as the dictionary says; the dry-run answer was a slip"}
# The gate is 20 items, 5 per accent type, gold 10/10. Everything else that survived verification
# becomes the catch-trial pool, mixed into the blocks themselves: the gate tests ability at
# admission, the catch trials test whether the rater kept listening.
GATE = {"頭高": 5, "中高": 5, "平板": 5, "尾高": 5}

# The gate is 20 items, 5 per accent type, gold 10/10. Everything else that survived verification
# becomes the catch-trial pool, mixed into the blocks themselves: the gate tests ability at
# admission, the catch trials test whether the rater kept listening.
GATE = {"頭高": 5, "中高": 5, "平板": 5, "尾高": 5}

ap = argparse.ArgumentParser()
ap.add_argument("--dry", default="preregistration/items/screening_dryrun_author.json")
ap.add_argument("--out", default="preregistration/items/screening_gate10.json")
ap.add_argument("--catch-out", default="preregistration/items/screening_catch.json")
ap.add_argument("--audio-out", default=f"{paths.DATA}/screening_final")
ap.add_argument("--catch-audio", default=f"{paths.DATA}/screening_catch")
ap.add_argument("--write", action="store_true")
a = ap.parse_args()

dry = json.load(open(os.path.join(REPO, a.dry), encoding="utf-8"))
items, audio = [], {}
for rel, d in SRC:
    for r in json.load(open(os.path.join(REPO, rel), encoding="utf-8")):
        items.append(r); audio[r["id"]] = os.path.join(d, r["id"] + ".wav")


def acc_type(r):
    if "dict_type" in r:
        return "平板" if r["dict_type"] == "heiban" else "尾高"
    tr = r["true_reading"].replace("’", "'")
    if "'" not in tr:
        return "平板"
    mora = lambda s: len([c for c in s if c not in "ャュョ"])
    n, tot = mora(tr[:tr.index("'")]), mora(tr.replace("'", ""))
    return "頭高" if n == 1 else ("尾高" if n == tot else "中高")


kept, dropped = [], []
for r in items:
    g = "1" if r["gold"] in (1, True, "1") else "2"
    seen = [d[r["id"]] for d in dry.values() if r["id"] in d]
    ok = (seen and all(v == g for v in seen)) or r["id"] in REINSTATED
    (kept if ok else dropped).append(r)
assert {r["id"] for r in dropped} == set(DROP_REASON), \
    (f"drop set moved: {sorted({r['id'] for r in dropped})}; every mismatch must be re-listened to"
     " and recorded in DROP_REASON or REINSTATED before the set can be frozen")

sel, odd = [], 0
for ti, t in enumerate(("頭高", "中高", "平板", "尾高")):
    pool = [r for r in kept if acc_type(r) == t]
    want1 = [r for r in pool if r["gold"] in (1, True, "1")]
    want0 = [r for r in pool if r not in want1]
    # An even-sized type takes one of each gold class; the extra item of an ODD-sized type has to
    # alternate between the classes, or the gate leans 6/4 instead of 5/5.
    order = (want1, want0) if odd % 2 == 0 else (want0, want1)
    odd += GATE[t] % 2
    take, i = [], 0
    while len(take) < GATE[t] and (i < len(want1) or i < len(want0)):
        for src in order:
            if i < len(src) and len(take) < GATE[t]:
                take.append(src[i])
        i += 1
    sel += take

# fixed, content-derived order: types must not appear in blocks, and the order must not depend on
# how this script happened to iterate.
sel.sort(key=lambda r: hashlib.sha256(("ORD:" + r["id"]).encode()).hexdigest())
out = []
for r in sel:
    out.append(dict(id=r["id"], word=r["word"], carrier=r["carrier"],
                    reading="".join(c for c in (r.get("reading") or r.get("true_reading", "")) if c not in "'’"),
                    accent_type=acc_type(r), shown_accent=r["shown_accent"],
                    gold="1" if r["gold"] in (1, True, "1") else "0",
                    source="screening_20" if r["id"].startswith("scr0") or r["id"].startswith("scr1") else "screening_v4"))

catch = [r for r in kept if r["id"] not in {x["id"] for x in sel}]
from collections import Counter
print(f"  kept {len(kept)} of {len(items)}  ->  gate {len(out)}, catch pool {len(catch)}")
print("  型:", dict(Counter(r["accent_type"] for r in out)),
      " gold1:", sum(r["gold"] == "1" for r in out), "/ gold0:", sum(r["gold"] == "0" for r in out))
for r in out:
    print(f"  {r['id']:<8}{r['word']:<6}{r['accent_type']:<5}提示={r['shown_accent']:<10}gold={r['gold']}")
if a.write:
    json.dump(out, open(os.path.join(REPO, a.out), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.makedirs(a.audio_out, exist_ok=True)
    for r in out:
        shutil.copy2(audio[r["id"]], os.path.join(a.audio_out, r["id"] + ".wav"))
    json.dump({"dropped": {k: DROP_REASON[k] for k in sorted(DROP_REASON)},
               "reinstated_after_relisten": {k: REINSTATED[k] for k in sorted(REINSTATED)}},
              open(os.path.join(REPO, "preregistration/items/screening_dropped.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)
    cout = [dict(id=r["id"], word=r["word"], carrier=r["carrier"],
                 reading="".join(c for c in (r.get("reading") or r.get("true_reading", "")) if c not in "'’"),
                 accent_type=acc_type(r), shown_accent=r["shown_accent"],
                 gold="1" if r["gold"] in (1, True, "1") else "0") for r in catch]
    cout.sort(key=lambda r: hashlib.sha256(("ORD:" + r["id"]).encode()).hexdigest())
    json.dump(cout, open(os.path.join(REPO, a.catch_out), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.makedirs(a.catch_audio, exist_ok=True)
    for r in cout:
        shutil.copy2(audio[r["id"]], os.path.join(a.catch_audio, r["id"] + ".wav"))
    print(f"  -> {a.out} ({len(out)}) + {a.catch_out} ({len(cout)}), wavs in {a.audio_out} / {a.catch_audio}")
    print("     catch pool:", dict(Counter(r["accent_type"] for r in cout)),
          "gold1", sum(r["gold"] == "1" for r in cout))
