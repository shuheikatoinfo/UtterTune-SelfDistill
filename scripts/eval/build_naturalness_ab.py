#!/usr/bin/env python3
"""Freeze the naturalness A/B stimuli from the easy set.

Why the easy set. The paper's instrument is a same-sentence A/B preference between the unedited and
the edited rendition, which only measures naturalness if both renditions SAY THE SAME THING. On the
difficult set they do not: the edit exists precisely to change the reading, and measured over the
319-item objective set the two arms produce an identical transcript for 1-16% of items depending on
backbone. A listener asked which sounds better would be answering which word they recognised. The
UtterTune preprint has the same split -- its subjective MOS is reported on Test Set 1 (easy) only,
with the automatic UTMOSv2 proxy on both -- and this follows that.

Even on the easy set the equality is not free: measured here, ASR hears the target reading in both
arms for only 56-84% of pairs depending on backbone. An earlier version of this script ADMITTED
ONLY those pairs. That was wrong, and in our own favour: the pairs it dropped are overwhelmingly the
ones where the unedited arm got the reading wrong -- exactly the cases where the edit did something
-- so the surviving set is the mildest possible intervention, where the tag asks for what the model
would have said anyway. Naturalness measured there cannot show a cost that the mechanism only incurs
when it acts. It is also selection on a post-treatment outcome, which tilts the set toward items the
unedited arm happened to handle well.

Every complete pair therefore ships. The rater is told that pronunciation is not what is being
judged. Whether the two arms realise the same reading is recorded per pair (`same_reading`) so the
matched subset can be reported as a secondary analysis, with the disagreement between the two
readings of the result visible rather than resolved by us in advance.
"""
import argparse, collections, glob, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/, for paths.py
import paths
import numpy as np

CAN = str.maketrans({"ヂ": "ジ", "ヅ": "ズ", "ヴ": "ブ"})
norm = lambda s: re.sub(r"[^ァ-ヶー]", "", s or "").translate(CAN)
ROOT = f"{paths.DATA}/easy50_nat"
TD = paths.ITEM_SOURCE

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="preregistration/items/naturalness_ab_easy50.json")
ap.add_argument("--audio-out", default=f"{paths.DATA}/naturalness_kit")
ap.add_argument("--write", action="store_true")
ap.add_argument("--root", default=ROOT, help="tree holding <backbone>/**/{noedit,ours}/NN.wav")
ap.add_argument("--per-backbone", type=int, default=0,
                help="round 2 (PREREGISTRATION_v2.md item 8): keep the N items per backbone with the lowest "
                     "sha256('NAT2:<backbone>:<idx>'), chosen before any audio is inspected; 0 = all")
ap.add_argument("--id-prefix", default="n", help="pair id prefix (round 1: n, round 2: m)")
a = ap.parse_args()
ROOT = a.root

lora = [l.rstrip("\n") for l in open(f"{TD}/test_easy_50_lora.txt", encoding="utf-8") if l.strip()]
van = [l.rstrip("\n") for l in open(f"{TD}/test_easy_50_vanilla.txt", encoding="utf-8") if l.strip()]
yom = [l.rstrip("\n") for l in open(f"{TD}/test_easy_50_yomi.txt", encoding="utf-8") if l.strip()]

# Five sentences have a target word that is already katakana (チューリップ, ノート, ケーキ, ハート,
# クロワッサン). The three text-in backbones render an edited version of them like any other: the
# tagged sentence is simply a different input string. Irodori's edited arm is not text-in -- it
# locates the target inside the sentence and wraps only that span, and it derived the span by
# diffing plain against reading text, which is empty here. That is a difference in how the edit is
# ADDRESSED, not in what the model can do, so the span is named directly through the router's
# registry instead (in round 1; from round 2 the Irodori synthesis falls back to the tagged word) and
# all four backbones carry all 50.
tgt = [norm(re.sub(r"[’'/]", "", re.search(r"<PHON_START>(.*?)<PHON_END>", l).group(1))) for l in lora]


def audible(path, peak=0.05):
    """A pair only exists if both clips can be heard. After the guard's retry cap some T5Gemma items
    still come back at peak 0.006-0.04 -- inaudible at a normal setting, and loudness normalisation
    would only raise the noise floor. Such a pair cannot be judged, so it is excluded and the count
    is reported per backbone. This removes a backbone's WORST output, so the count matters as much
    as the preference it leaves behind."""
    import soundfile as sf
    a, _ = sf.read(path, dtype="float32")
    return float(np.abs(a).max()) >= peak


def clips(bb):
    """Locate {arm: {item_index: path}}; the four backbones write slightly different trees."""
    out = collections.defaultdict(dict)
    for arm in ("noedit", "ours"):
        for p in glob.glob(f"{ROOT}/{bb}/**/{arm}/*.wav", recursive=True) + \
                 glob.glob(f"{ROOT}/{bb}/{arm}/*.wav"):
            m = re.search(r"(\d+)\.wav$", p)
            if m:
                out[arm][int(m.group(1))] = p
    return out


def hyps(bb, av):
    """Prefer the transcript the degeneracy guard already made; ASR anything it did not cover."""
    got = {}
    for f in glob.glob(f"{ROOT}/{bb}/**/guard_takes.json", recursive=True):
        for r in json.load(open(f, encoding="utf-8")):
            if r.get("hyp") is not None:
                got[(r["arm"], r["idx"])] = norm(r["hyp"])
    todo = [(arm, i, p) for arm, d in av.items() for i, p in d.items() if (arm, i) not in got]
    if todo:
        import torch, soundfile as sf
        from transformers import pipeline
        torch.backends.cudnn.enabled = False
        kw = pipeline("automatic-speech-recognition", model="sbintuitions/kana-whisper", device=0,
                      torch_dtype=torch.float16, generate_kwargs={"language": "ja", "task": "transcribe"})
        for arm, i, p in todo:
            w, sr = sf.read(p, dtype="float32")
            if w.ndim > 1: w = w.mean(1)
            if sr != 16000:   # resample here: the pipeline's raw-input path distorts 44.1/48 kHz audio
                import librosa; w = librosa.resample(w, orig_sr=sr, target_sr=16000); sr = 16000
            got[(arm, i)] = norm(kw({"raw": w, "sampling_rate": sr})["text"])
    return got


rows, report = [], {}
for bb in ("sarashina", "cv2", "irodori", "t5gemma"):
    av = clips(bb)
    if not av.get("noedit") or not av.get("ours"):
        report[bb] = {"pairs": 0, "admitted": 0, "note": "no audio"}
        continue
    h = hyps(bb, av)
    both_all = sorted(set(av["noedit"]) & set(av["ours"]))
    if a.per_backbone:
        import hashlib as _h
        sel = set(sorted(range(len(van)), key=lambda i: _h.sha256(f"NAT2:{bb}:{i}".encode()).hexdigest())[:a.per_backbone])
        both_all = [i for i in both_all if i in sel]
    both = [i for i in both_all if audible(av["noedit"][i]) and audible(av["ours"][i])]
    inaudible = len(both_all) - len(both)
    same = [i for i in both
            if tgt[i] and tgt[i] in h.get(("noedit", i), "") and tgt[i] in h.get(("ours", i), "")]
    report[bb] = {"pairs": len(both), "excluded_inaudible": inaudible,
                  "same_reading": len(same),
                  "same_reading_rate": round(len(same) / max(len(both), 1), 3)}
    for i in both:
        rows.append(dict(backbone=bb, idx=i, sentence=van[i], target_reading=tgt[i],
                         same_reading=i in same,
                         hyp_noedit=h.get(("noedit", i), ""), hyp_ours=h.get(("ours", i), ""),
                         noedit=av["noedit"][i], ours=av["ours"][i]))

print(f"  {'backbone':<12}{'対':>5}{'無音で除外':>11}{'読み一致':>9}{'割合':>7}")
for bb, r in report.items():
    print(f"  {bb:<12}{r['pairs']:>5}{r.get('excluded_inaudible', 0):>11}"
          f"{r.get('same_reading', 0):>9}{r.get('same_reading_rate', 0):>7.0%}")
print(f"  合計 {len(rows)} 対（全対を出題。読み一致は副次分析の層）")

if a.write:
    import hashlib, shutil
    os.makedirs(a.audio_out, exist_ok=True)
    out = []
    for k, r in enumerate(sorted(rows, key=lambda r: hashlib.sha256(
            f"NAT:{r['backbone']}:{r['idx']}".encode()).hexdigest())):
        pid = f"{a.id_prefix}{k:03d}"
        # A/B side is fixed by hash, not by arm, so the edited rendition is not always on one side
        flip = int(hashlib.sha256(f"NATSIDE:{pid}".encode()).hexdigest(), 16) % 2 == 1
        for side, arm in (("A", "ours" if flip else "noedit"), ("B", "noedit" if flip else "ours")):
            shutil.copy2(r[arm], os.path.join(a.audio_out, f"{pid}{side}.wav"))
        out.append(dict(id=pid, backbone=r["backbone"], idx=r["idx"], sentence=r["sentence"],
                        target_reading=r["target_reading"], same_reading=r["same_reading"],
                        hyp_noedit=r["hyp_noedit"], hyp_ours=r["hyp_ours"],
                        A="ours" if flip else "noedit", B="noedit" if flip else "ours"))
    json.dump({"n": len(out), "report": report,
               "note": "A/B side fixed by hash; the mapping lives here and is never shipped to raters.",
               "items": out}, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  -> {a.out} + {2*len(out)} wav in {a.audio_out}")
