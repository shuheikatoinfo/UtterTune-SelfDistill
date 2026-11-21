"""One reading measurement for every backbone, in one pass, with one ASR configuration.

Why: the numbers in the paper previously came from two different scripts with two different
kana-constraint mechanisms (logit mask vs. suppress-kanji) and, for one backbone, from a
earlier, unreleased adapter. Two rows measured differently are not comparable, and a contaminated
adapter contradicts the disjointness the method claims. This script removes both problems: every
arm of every backbone is transcribed by the same model with the same decoding constraint, from
audio resampled explicitly to 16 kHz (see scripts/eval/asr_audio.py for why that matters), and
scored with the same canonicalisation.

Metric of record: canonical target-reading hit — does the intended reading of the target word
appear in the kana transcript, after long-vowel canonicalisation.

  scripts/eval/.venv_metrics/bin/python scripts/eval/measure_reading_all.py --out <json>
"""
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/, for paths.py
import paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, soundfile as sf, torch
from asr_audio import to_asr_input

ROOT = paths.DATA
D = f"{ROOT}/cleanroom_rebuild"

# (backbone, arm) -> directory of NN.wav. Every entry is a CLEAN-ROOM editor.
ARMS = {
    "sarashina": {
        "noedit": f"{ROOT}/blockA_sarashina_difficult_4spk_cleanroom_v2/spk0/noedit",
        "kana":   f"{ROOT}/blockA_sarashina_difficult_4spk_cleanroom_v2/spk0/kana",
        "ours":   f"{ROOT}/blockA_sarashina_difficult_4spk_cleanroom_v2/spk0/ours",
    },
    "cosyvoice2": {
        "noedit": f"{D}/cv2_compare/ours_cleanroom/noedit",
        "kana":   f"{D}/cv2_compare/ours_cleanroom/kana",
        "ours":   f"{D}/cv2_compare/ours_cleanroom/ours",
        "uttertune_released": f"{D}/cv2_compare/uttertune_released/ours",
    },
    "irodori": {
        "noedit": f"{D}/irodori_eval_v2/difficult_spk0/noedit",
        "kana":   f"{D}/irodori_eval_v2/difficult_spk0/kana",
        "ours":   f"{D}/irodori_eval_v2/difficult_spk0/ours",
    },
    "t5gemma": {
        "noedit": f"{D}/t5g_cleanroom_out/blockA_t5gemma_difficult/noedit",
        "kana":   f"{D}/t5g_cleanroom_out/blockA_t5gemma_difficult/kana",
        "ours":   f"{D}/t5g_cleanroom_out/blockA_t5gemma_difficult/ours",
    },
}

CAN = str.maketrans({"ヂ": "ジ", "ヅ": "ズ", "ヴ": "ブ", "ヲ": "オ"})
U_DAN = set("ウクグスズツヅヌフブプムユュルヴ"); O_DAN = set("オコゴソゾトドノホボポモヨョロヲ")
E_DAN = set("エケゲセゼテデネヘベペメレェ")

def norm(s):
    s = "".join(chr(ord(c) + 0x60) if 0x3041 <= ord(c) <= 0x3096 else c for c in (s or ""))
    return re.sub(r"[^ァ-ヶー]", "", s).translate(CAN)

def canon(s):
    out = []
    for c in norm(s):
        p = out[-1] if out else ""
        out.append("ー" if (c == "ー" or (c == "ウ" and (p in U_DAN or p in O_DAN))
                            or (c == "オ" and p in O_DAN) or (c == "イ" and p in E_DAN)) else c)
    return "".join(out)

# TEST_DIR / TEST_SET pick which item set's target readings the metric scores against; the
# defaults reproduce the difficult-50 measurements exactly. The synthesis scripts take the same
# two variables, so a run and its measurement cannot silently disagree about the item set.
TEST_DIR = os.environ.get("TEST_DIR") or paths.ITEM_SOURCE
ITEM_SPK = {}
if os.environ.get("ITEM_SPK_JSON"):
    import json as _json
    _a = _json.load(open(os.environ["ITEM_SPK_JSON"]))["cells"][os.environ["ITEM_SPK_CELL"]]
    ITEM_SPK = {int(k): v[0] for k, v in _a.items()}
TEST_SET = os.environ.get("TEST_SET", "difficult_50")


# TARGETS_FILE decouples the ground truth from the input. The target reading is a property of the
# BENCHMARK ITEM; the tag is an input we choose, and its span and accent have both been changed
# deliberately (a phrase-scoped tag reads ソノホカ where the item's target is ホカ). Reading the
# truth off the tag made the unedited and kana arms move when only the edited arm's input changed,
# which is not a measurement, it is the ruler moving. One line per item, in the set's order.
TARGETS_FILE = os.environ.get("TARGETS_FILE")


def targets(normalize=canon):
    out = []
    if TARGETS_FILE:
        for ln in open(TARGETS_FILE, encoding="utf-8"):
            if ln.strip():
                out.append(normalize(ln.strip().replace("'", "").replace("/", "")))
        return out
    for ln in open(f"{TEST_DIR}/test_{TEST_SET}_lora.txt", encoding="utf-8"):
        if not ln.strip():
            continue
        m = re.search(r"<PHON_START>(.*?)<PHON_END>", ln)
        out.append(normalize((m.group(1) if m else "").replace("'", "").replace("/", "")))
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{D}/remeasure/reading_all.json")
    # kana-whisper outputs katakana directly. Plain whisper-large-v3 transcribes in kanji, and the
    # katakana-only normalisation then deletes most of the sentence (七味猛霊が静かな森に… ->
    # ガカナニクヒッソリムダ), which is what produced absurd hit rates on the first attempt. A
    # kana-constrained decode is required; this model provides it natively for every backbone,
    # and 24/44.1/48 kHz are all handled because the audio is resampled explicitly to 16 kHz.
    ap.add_argument("--asr", default="sbintuitions/kana-whisper")
    # Override the (backbone -> arm -> dir) map without duplicating the metric. Any per-speaker or
    # per-set measurement MUST come through this script, so every number in the paper is produced
    # by one implementation of canon()/targets().
    ap.add_argument("--arms-json", default=None,
                    help="JSON file replacing ARMS: {backbone: {arm: wav_dir}}")
    args = ap.parse_args()
    if args.arms_json:
        ARMS = json.load(open(args.arms_json, encoding="utf-8"))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    torch.backends.cudnn.enabled = False          # this box's cuDNN fails on some conv shapes
    from transformers import pipeline
    TGT = targets()
    TGT_N = targets(normalize=norm)   # exact-mora form, see the hit test
    asr = pipeline("automatic-speech-recognition", model=args.asr, device=0,
                   dtype=torch.float16, generate_kwargs={"language": "ja", "task": "transcribe"})
    KANA_OK = None  # kana constraint applied by post-hoc canonicalisation, identical for every arm

    res = {"asr": args.asr, "n_items": len(TGT), "backbones": {}}
    for bk, arms in ARMS.items():
        res["backbones"][bk] = {}
        for arm, d in arms.items():
            if not os.path.isdir(d):
                print(f"  SKIP {bk}/{arm}: no dir {d}", flush=True)
                continue
            hits, n, per = 0, 0, []
            # 42-voice layout: one held-out voice per ITEM, files under <d>/spk<s(i)>/<arm>/NN.wav.
            # Uses the SAME env as the synthesis scripts (ITEM_SPK_JSON / ITEM_SPK_CELL), so the
            # scorer and the synthesis cannot disagree about which voice an item was given.
            for i, tgt in enumerate(TGT):
                if ITEM_SPK:
                    # arm label -> on-disk subdir (the released-adapter arm was synthesized with ARMS=ours)
                    sub = "ours" if arm == "uttertune_released" else arm
                    p = os.path.join(d, f"spk{ITEM_SPK[i]}", sub, f"{i:02d}.wav")
                    if not os.path.exists(p):
                        # Irodori switches the reference per item inside one process, so its output
                        # is flat (<arm>/NN.wav) with no speaker directory. Same assignment, same
                        # voices -- only the layout differs.
                        p = os.path.join(d, arm, f"{i:02d}.wav")
                else:
                    p = os.path.join(d, f"{i:02d}.wav")
                if not os.path.exists(p) or not tgt:
                    continue
                a, sr = sf.read(p, dtype="float32")
                raw = str(asr(to_asr_input(a, sr))["text"])
                # HIT = canon match OR exact-mora match. canon() alone collapses a word-INITIAL
                # オ/ウ/イ after an o/u/e-dan mora across the word boundary ("コノネコオス" ->
                # "コノネコース"), scoring an exact rendition of "オス" as a miss (found 2026-09-03,
                # audit #6). The exact-mora alternative cannot create a false positive: it is the
                # stricter of the two tests. The raw hypothesis is stored so any future rescoring
                # is a file operation, not another ASR pass.
                ok = (tgt in canon(raw)) or (TGT_N[i] in norm(raw))
                hits += ok; n += 1; per.append({"i": i, "hit": bool(ok), "hyp": raw})
            res["backbones"][bk][arm] = {"n": n, "hit": hits / n if n else None, "per_item": per}
            print(f"  {bk:12s} {arm:20s} n={n:3d} hit={hits/n if n else float('nan'):.3f}", flush=True)
    json.dump(res, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"-> {args.out}")
