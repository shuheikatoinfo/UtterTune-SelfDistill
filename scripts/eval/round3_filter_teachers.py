#!/usr/bin/env python3
"""Round 3 (registration section 16, item 2): the slow-teacher filter, one number on every backbone: a teacher is dropped
when its speech runs longer than 0.22 s per character of the plain carrier. Sarashina: 5.5 semantic tokens/char (25 Hz);
CosyVoice 2: waveform duration; Irodori: 5.5 latent frames/char (25 Hz); T5Gemma: 11 codec frames/char (50 Hz).
    python3 scripts/eval/round3_filter_teachers.py <cv2|irodori|t5gemma> <in> <out>
Writes <out> and <out>.filter.json (counts). Never modifies the input."""
import json, re, sys, os, soundfile as sf
bb, src, dst = sys.argv[1:4]; LIMIT = 0.22; kept, dropped, rates = [], 0, []
strip = lambda t: re.sub(r"<[^<>]*>", "", t)
if bb == "cv2":
    rows = [l.rstrip("\n") for l in open(src, encoding="utf-8") if l.strip()]
    for l in rows:
        spk, txt, tp, wp = l.split("\t"); r = sf.info(wp).duration / max(1, len(strip(txt))); rates.append(r)
        if r <= LIMIT: kept.append(l)
        else: dropped += 1
    open(dst, "w", encoding="utf-8").write("".join(l + "\n" for l in kept))
else:
    rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    for j in rows:
        r = (j["num_frames"] / 25.0 / max(1, len(j["original_text"]))) if bb == "irodori" else (j["n_frames"] / 50.0 / max(1, len(j["teacher_input"])))
        rates.append(r)
        if r <= LIMIT: kept.append(j)
        else: dropped += 1
    with open(dst, "w", encoding="utf-8") as f:
        for j in kept: f.write(json.dumps(j, ensure_ascii=False) + "\n")
rs = sorted(rates); q = {p: round(rs[int(p * (len(rs) - 1))], 3) for p in (0.1, 0.5, 0.9)}
info = {"backbone": bb, "rule": "drop if seconds per plain character > 0.22 (section 16 item 2)", "in": len(rows), "kept": len(kept), "dropped": dropped, "s_per_char_q10_50_90": q, "src": src}
json.dump(info, open(dst + ".filter.json", "w"), indent=1); print(info)
