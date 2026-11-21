"""Write the ONE text set every backbone's v2 teacher generator reads (2026-09-16).

Rows: {word, reading (accented, apostrophe notation), plain (kanji carrier), phon (carrier with the
target replaced by <PHON_START>reading<PHON_END>)}. The eval-collision filter is the union of the
filters used in round 1: any inventory word matching a target of difficult-50, katakana-50, EASY-50
(the naturalness set; only the CosyVoice 2 script excluded it in round 1) or blockC-1000, by
surface or canonical reading, is dropped. Ten carriers per word, no sampling.

    python3 scripts/teachers/build_text_v3.py --out $DATA/v2/teachers_text_v3.jsonl
"""
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/, for paths.py
import paths
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts", "teachers"))
from reading_check import canon
TD = paths.ITEM_SOURCE
BLOCKC = f"{paths.DATA}/blockC_wordlist/blockc_1000.jsonl"

ap = argparse.ArgumentParser()
ap.add_argument("--inventory", default=os.path.join(REPO, "data/cleanroom/inventory_500.jsonl"))
ap.add_argument("--out", required=True)
a = ap.parse_args()

surf, read = set(), set()
def add_span(path, pat):
    for ln in open(path, encoding="utf-8"):
        for m in re.findall(pat, ln):
            s = re.sub(r"[＼￣]", "", m); surf.add(s); surf.add(canon(s)); read.add(canon(s))
add_span(f"{TD}/test_difficult_50_with_accent.txt", r"<b>(.+?)</b>")
add_span(f"{TD}/test_katakana_50_with_accent.txt", r"<b>(.+?)</b>")
add_span(f"{TD}/test_difficult_50_lora.txt", r"<PHON_START>(.*?)<PHON_END>")
add_span(f"{TD}/test_katakana_50_phon.txt", r"<PHON_START>(.*?)<PHON_END>")
add_span(f"{TD}/test_easy_50_phon.txt", r"<PHON_START>(.*?)<PHON_END>")
for ln in open(BLOCKC, encoding="utf-8"):
    r = json.loads(ln); surf.add(r["surface"]); read.add(canon(r["reading"]))

rows, dropped, bad = [], set(), 0
for ln in open(a.inventory, encoding="utf-8"):
    r = json.loads(ln)
    native = "<PHON_START>" + r["accented_reading"] + "<PHON_END>"
    if native not in r["carrier_student"] or r["word"] not in r["carrier_plain"]:
        bad += 1; continue
    if r["word"] in surf or canon(r["word"]) in surf or canon(r["accented_reading"]) in read:
        dropped.add(r["word"]); continue
    rows.append({"word": r["word"], "reading": r["accented_reading"], "plain": r["carrier_plain"],
                 "phon": r["carrier_student"], "accent_type": r["accent_type"], "category": r["category"]})
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
words = {r["word"] for r in rows}
print(f"inventory rows kept {len(rows)} over {len(words)} words; dropped {len(dropped)} eval-colliding words "
      f"({'/'.join(sorted(dropped))}); malformed {bad} -> {a.out}")
