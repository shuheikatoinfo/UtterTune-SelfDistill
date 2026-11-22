#!/usr/bin/env python3
"""Equalise loudness across a set of listening clips.

An A/B preference asks which rendition sounds more natural. Loudness is not naturalness, but it is
the single strongest confound in a paired listening test -- the louder clip is preferred. Measured
on this set the systematic bias is small (ours minus no-edit averages +0.09 dB) but the per-pair
spread is not: the median absolute difference is 0.75 dB, 42% of pairs exceed 1 dB and 13% exceed
3 dB, with T5Gemma worst at a 1.90 dB median. A note asking raters to ignore it would not work; the
effect is not under their control.

Every clip is normalized to the same integrated loudness and peak-limited. The transform is applied
identically to both arms, so it cannot favor either.
"""
import argparse, json, os, shutil
import numpy as np, soundfile as sf
import pyloudnorm as pl

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--recursive", action="store_true", help="walk sub-directories, mirroring the tree")
ap.add_argument("--lufs", type=float, default=-23.0)
ap.add_argument("--peak", type=float, default=-1.0, help="dBFS ceiling after normalization")
ap.add_argument("--floor", type=float, default=-45.0,
                help="leave a clip alone if its own loudness is below this: it is degenerate "
                     "(near-silent, usually a runaway take), and normalizing it amplifies noise")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

before, after, clipped, skipped = [], [], 0, []
def walk(root):
    if not a.recursive:
        for f in sorted(os.listdir(root)):
            if f.endswith(".wav"):
                yield f
        return
    for dp, _, fs in os.walk(root):
        for f in sorted(fs):
            if f.endswith(".wav"):
                yield os.path.relpath(os.path.join(dp, f), root)


for f in walk(a.src):
    os.makedirs(os.path.dirname(os.path.join(a.out, f)) or a.out, exist_ok=True)
    x, sr = sf.read(os.path.join(a.src, f), dtype="float64")
    lo = pl.Meter(sr).integrated_loudness(x)
    if not np.isfinite(lo) or lo < a.floor:
        sf.write(os.path.join(a.out, f), x.astype("float32"), sr)
        skipped.append(f)
        continue
    y = pl.normalize.loudness(x, lo, a.lufs)
    ceiling = 10 ** (a.peak / 20)
    pk = np.abs(y).max()
    if pk > ceiling:                      # scale the whole clip down rather than clip samples
        y = y * (ceiling / pk); clipped += 1
    sf.write(os.path.join(a.out, f), y.astype("float32"), sr)
    before.append(lo); after.append(pl.Meter(sr).integrated_loudness(y))

b, af = np.array(before), np.array(after)
print(f"  {len(b)} clips  正規化前 {b.mean():.1f} ± {b.std():.1f} LUFS  →  後 {af.mean():.1f} ± {af.std():.1f}")
print(f"  ピーク制限で追加減衰したクリップ: {clipped}")
print(f"  劣化クリップとして正規化を見送り: {len(skipped)} {skipped[:5]}")
print(f"  -> {a.out}")
