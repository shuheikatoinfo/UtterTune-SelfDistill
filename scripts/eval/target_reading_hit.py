#!/usr/bin/env python3
"""Target-reading HIT rate — a carrier-robust reading diagnostic (CPU, uses STORED hyps).

Motivation: full-sentence kana-CER is dominated by carrier-word ASR noise on
lower-intelligibility backbones (Irodori-500M-v3 full-sentence CER ~0.6 even on readable words).
The EDIT only touches the TARGET word, so the honest reading question is: "does the synthesized
audio actually say the target word's intended reading?" This checks whether the target katakana
appears (contiguous, katakana-normalized) in the ALREADY-STORED kana-ASR hypothesis — no GPU, no
re-synthesis. It is an auxiliary DIAGNOSTIC (not a replacement for the reported metric); it is
uniform across arms (every arm scored identically), so it is integrity-clean.

Reads:
  * target readings: the <PHON_START>KANA<PHON_END> span of test_difficult_50_lora.txt (apostrophe/
    slash stripped) = the intended target reading per difficult sentence.
  * stored hyps: $DATA/.../blockA_metrics.json (items[].hyp_kana, whisper-large-v3) for
    sarashina_difficult / cv2_difficult, and $DATA/.../blockB1_irodori/blockB1_irodori.json
    (eval.<arm>.per_line[].hyp, kana-whisper) for Irodori B1.
Prints per-(set,arm) HIT rate = fraction of items whose target reading is found in the hyp.
"""
from __future__ import annotations
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/, for paths.py
import paths

TESTS = paths.ITEM_SOURCE
ROOT = paths.DATA
CAN = str.maketrans({"ヂ": "ジ", "ヅ": "ズ", "ヴ": "ブ", "ヲ": "オ"})


def kata(s: str) -> str:
    """Normalize to katakana-only comparison key (hiragana->katakana, drop non-kana, canon)."""
    if not s:
        return ""
    out = []
    for ch in s:
        o = ord(ch)
        if 0x3041 <= o <= 0x3096:  # hiragana -> katakana
            out.append(chr(o + 0x60))
        else:
            out.append(ch)
    return re.sub(r"[^ァ-ヶー]", "", "".join(out)).translate(CAN)


def target_readings() -> list[str]:
    """Target reading per difficult line = content between <PHON_START>..<PHON_END>, marks stripped."""
    out = []
    with open(f"{TESTS}/test_difficult_50_lora.txt", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            m = re.search(r"<PHON_START>(.*?)<PHON_END>", ln)
            span = m.group(1) if m else ""
            out.append(kata(span.replace("'", "").replace("/", "")))
    return out


def hit(target: str, hyp: str) -> bool:
    """Target reading found in hyp. Contiguous substring; fall back to 'all morae present in order'
    for long readings the ASR may split (subsequence) — reported separately."""
    return bool(target) and target in kata(hyp)


def subseq(target: str, hyp: str) -> bool:
    h = kata(hyp)
    it = iter(h)
    return bool(target) and all(c in it for c in target)


def score(name: str, targets: list[str], hyps: list[str]):
    n = min(len(targets), len(hyps))
    sub = sum(subseq(targets[i], hyps[i]) for i in range(n))
    exact = sum(hit(targets[i], hyps[i]) for i in range(n))
    print(f"  {name:16s} n={n:3d}  substring-hit={exact/n:5.1%} ({exact:2d})  subsequence-hit={sub/n:5.1%} ({sub:2d})")
    return exact, sub, n


def main():
    tgt = target_readings()
    print(f"target readings: {len(tgt)} (e.g. {tgt[:3]})\n")

    # --- Block A difficult: Sarashina + CV2 from blockA_metrics.json (stored whisper-large-v3 hyps)
    bam = os.environ.get("OUT_JSON", os.path.join(ROOT, "blockA_metrics.json"))
    if os.path.exists(bam):
        d = json.load(open(bam))
        for setk in ("sarashina_difficult", "cv2_difficult"):
            s = d["sets"].get(setk)
            if not s:
                continue
            print(f"[{setk}]  (hyps: whisper-large-v3)")
            for arm, a in s["arms"].items():
                hyps = [it.get("hyp_kana", "") for it in a.get("items", [])]
                score(arm, tgt, hyps)
            print()

    # --- Block B1 Irodori from blockB1_irodori.json (stored kana-whisper hyps)
    b1 = os.path.join(ROOT, "blockB1_irodori", "blockB1_irodori.json")
    if os.path.exists(b1):
        d = json.load(open(b1))
        ev = d.get("eval", {})
        print(f"[blockB1_irodori]  (hyps: kana-whisper; n so far may be < 50 if still running)")
        for arm, a in ev.items():
            hyps = [pl.get("hyp", "") for pl in a.get("per_line", [])]
            score(arm, tgt, hyps)
        print()
    else:
        print("[blockB1_irodori] not present yet (B1 still running) — re-run this after it finishes.\n")

    print("HIT = target word's intended reading found in the stored ASR hyp (carrier-robust; auxiliary).")


if __name__ == "__main__":
    main()
