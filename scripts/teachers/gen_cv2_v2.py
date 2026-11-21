"""Teacher generation for CosyVoice 2 (the registered rule, as on the other backbones).

Same rule as the other backbones: the frozen base reads the PLAIN kanji carrier; a take is kept
if it passes the degeneration guard (audio >= MIN_SEC) AND the ASR of record hears the target
reading; else the seed is bumped, up to MAX_TAKES; a carrier with no passing take is dropped and
logged. CosyVoice 2 cannot synthesize without a prompt clip, so the SHARED bank of 2,000 self-sampled
voices (Irodori's anchor bank; --pool-json) is cycled over carriers, each clip with its own
transcript as prompt text; Irodori cycles the same 2,000 as reference latents. No best-of.

Output: manifest.tsv (spk, phon_text, tokens.npy, wav) as train_lora_cv2.py expects, plus provenance.

    $COSYVOICE_DIR/.venv/bin/python scripts/teachers/gen_cv2_v2.py \
        --text .../teachers_text_v3.jsonl --out $DATA/v2/cv2/teachers
"""
import argparse, collections, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/, for paths.py
import paths
import numpy as np, soundfile as sf
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO + "/scripts/cv"); sys.path.insert(0, REPO + "/scripts/teachers")
from cv_synth import make_measurement_synth, _vad_prompt, FICTIONAL_TEXT
from reading_check import Checker
BOOT = f"{paths.DATA}/bootstrap_speakers"

ap = argparse.ArgumentParser()
ap.add_argument("--text", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--speakers", default=",".join(str(i) for i in range(20)))
ap.add_argument("--pool-json", help="shared voice pool: [{spk, src_wav, text}]; overrides --speakers")
ap.add_argument("--max-takes", type=int, default=5)
ap.add_argument("--min-sec", type=float, default=0.5)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--shard", default="0/1", help="i/n: rows i::n; voice index and seed keyed on the global row index")
a = ap.parse_args()
WAV = os.path.join(a.out, "wav"); TOK = os.path.join(a.out, "tok")
os.makedirs(WAV, exist_ok=True); os.makedirs(TOK, exist_ok=True)
if a.pool_json:
    import soundfile as sf, librosa, shutil
    _pool = json.load(open(a.pool_json, encoding="utf-8"))["voices"]
    _pd = os.path.join(os.path.dirname(a.pool_json), "wav24k"); os.makedirs(_pd, exist_ok=True)
    POOL, PTEXT, spk_idx = [], [], []
    for k, v in enumerate(_pool):
        dst = os.path.join(_pd, v["spk"] + ".wav")
        _ok = False
        if os.path.exists(dst):
            try: _ok = sf.info(dst).frames > 0
            except Exception: _ok = False   # a copy truncated by an interrupted run is rewritten
        if not _ok:
            y, sr = sf.read(v["src_wav"], dtype="float32")
            if y.ndim > 1: y = y.mean(1)
            if sr != 24000: y = librosa.resample(y, orig_sr=sr, target_sr=24000)
            sf.write(dst, y, 24000)
        POOL.append(_vad_prompt(dst)); PTEXT.append(v["text"]); spk_idx.append(v["spk"])
    assert all(PTEXT), "every shared voice needs its transcript"
else:
    spk_idx = [int(s) for s in a.speakers.split(",")]
    POOL = [_vad_prompt(os.path.join(BOOT, f"fictional{k}.wav")) for k in spk_idx]
    PTEXT = [FICTIONAL_TEXT] * len(POOL)
rows = [json.loads(l) for l in open(a.text, encoding="utf-8")]
for _i, _r in enumerate(rows):
    _r["_gi"] = _i
_si, _sn = (int(x) for x in a.shard.split("/"))
rows = rows[_si::_sn]
if a.limit:
    rows = rows[:a.limit]
checker = Checker(device=0)
g = make_measurement_synth("cv2", prompt_wav=POOL[0], prompt_text=FICTIONAL_TEXT)
tokenizer = g.model.frontend._extract_speech_token
print(f"loaded (sr={g.sr}); {len(rows)} carriers; pool {len(POOL)}", flush=True)

manifest, prov, drops, takes_hist = [], [], [], collections.Counter()
per_spk = collections.Counter(); t0 = time.time()
for li, r in enumerate(rows):
    i = r["_gi"]; si = i % len(POOL); g.prompt_wav = POOL[si]; g.prompt_text = PTEXT[si]
    key = f"{r['word']}_{i}"; wp = os.path.join(WAV, f"{key}.wav"); tp = os.path.join(TOK, f"{key}.npy")
    wav, used, hyp, fails = None, None, "", []
    for t in range(a.max_takes):
        w = g.synth(r["plain"], text_frontend=True, seed=8000 + 131 * t + i)
        guard = len(w) >= int(a.min_sec * g.sr)
        heard, hyp = (False, "")
        if guard:
            heard, hyp = checker.heard(np.asarray(w, dtype="float32"), g.sr, r["reading"])
        if guard and heard:
            wav, used = w, t; break
        fails.append({"take": t, "guard": bool(guard), "hyp": hyp})
    if wav is None:
        drops.append({"word": r["word"], "plain": r["plain"], "fails": fails}); continue
    sf.write(wp, wav, g.sr)
    np.save(tp, tokenizer(wp)[0].squeeze(0).cpu().numpy().astype(np.int32))
    manifest.append((spk_idx[si], r["phon"], tp, wp)); per_spk[spk_idx[si]] += 1; takes_hist[used] += 1
    prov.append({"key": key, "word": r["word"], "speaker": str(spk_idx[si]), "take": used, "hyp": hyp})
    if li % 50 == 0:
        print(f"  {li}/{len(rows)} kept={len(manifest)} dropped={len(drops)}  {time.time()-t0:.0f}s", flush=True)
with open(os.path.join(a.out, "manifest.tsv"), "w", encoding="utf-8") as f:
    for spk, txt, tp, wp in manifest:
        f.write(f"{spk}\t{txt}\t{tp}\t{wp}\n")
json.dump({"rule": "voice pool cycled (--pool-json or --speakers); first take passing guard (>=min_sec) and reading check; max_takes",
           "max_takes": a.max_takes, "kept": len(manifest), "dropped": len(drops),
           "words_in": len({r['word'] for r in rows}), "words_kept": len({p['word'] for p in prov}),
           "per_speaker_kept": dict(per_spk), "takes_hist": {str(k): v for k, v in sorted(takes_hist.items())},
           "items": prov, "dropped_rows": drops}, open(os.path.join(a.out, "provenance.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(f"DONE kept={len(manifest)} dropped={len(drops)} in {time.time()-t0:.0f}s -> {a.out}/manifest.tsv", flush=True)
