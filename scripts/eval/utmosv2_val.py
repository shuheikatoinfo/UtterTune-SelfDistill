#!/usr/bin/env python3
"""UTMOSv2 of every tag-arm condition on the validation sets (mean over items), plus the unedited and kana arms.
    VALROOT=... python3 scripts/eval/utmosv2_val.py  -> <VALROOT>/utmosv2.json"""
import glob, json, os, sys, statistics as st
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/, for paths.py
import paths
import torch; torch.backends.cudnn.enabled = False
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_metrics import utmos
VR = os.environ.get("VALROOT") or f"{paths.DATA}/v2/sara_val"; out = {}
for s in ("sel105", "deveasy"):
    base = f"{VR}/{s}/registered"
    conds = {"unedited": sorted(glob.glob(f"{base}/spk*/noedit/*.wav")), "kana": sorted(glob.glob(f"{base}/spk*/kana/*.wav"))}
    for d in sorted(glob.glob(f"{VR}/{s}/*/")):
        conds[os.path.basename(d.rstrip("/"))] = sorted(glob.glob(f"{d}/spk*/ours/*.wav"))
    for c, files in conds.items():
        if not files: continue
        sc = utmos(files); out[f"{s} {c}"] = {"n": len(files), "utmosv2_mean": round(float(st.mean(sc)), 3)}; print(s, c, out[f"{s} {c}"], flush=True)
json.dump(out, open(f"{VR}/utmosv2.json", "w"), indent=1)
