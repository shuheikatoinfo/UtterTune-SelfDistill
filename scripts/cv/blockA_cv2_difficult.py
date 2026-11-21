"""Block A — CosyVoice2 (data-free self-distilled adapter), difficult set, 3 arms.
CV2 is the demoted-but-retained backbone (a Block A data point + weight release). Its test files
natively use the <PHON_START>/<PHON_END> tokens (no Sarashina-token swap needed). Arms:
  noedit : base CV2 (measurement synth), plain kanji.
  kana   : base CV2, target->katakana (the UtterTune kana-input baseline).
  ours   : CV2 + cv2_lora_pilot (our data-free adapter), <PHON>+' notation, text_frontend=False.
Reading Kana-CER vs pyopenjtalk full-kana of the yomi reference; accent = ear/rater on the wavs.
"""
import sys, os, re, json, difflib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/, for paths.py
import paths
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, soundfile as sf, pyopenjtalk
from cv_synth import make_measurement_synth
from cv_quality import KanaScorer
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'eval'))
from degenerate import synth_valid, safe_asr, save_guard_record

# TEST_DIR / TEST_SET make the item set swappable without touching the synthesis path:
# the same code, the same recipe, only a different input file. Needed for the 105-item
# SELECTION split and the 319-item TEST split (preregistration/items/objective_*.jsonl).
# Default: difficult-50.
TD = (os.environ.get("TEST_DIR") or paths.ITEM_SOURCE)
TEST_SET = os.environ.get("TEST_SET", "difficult_50")
OUT = (os.environ.get("BLOCKA_CV2_OUT") or f"{paths.DATA}/blockA_cv2_difficult")
LORA = (os.environ.get("CV2_LORA") or f"{paths.DATA}/cv2_lora_pilot")
_ARMS_ENV = [a.strip() for a in os.environ.get("ARMS", "noedit,kana,ours").split(",") if a.strip()]
for d in _ARMS_ENV:
    os.makedirs(os.path.join(OUT, d), exist_ok=True)

vanilla = [l.rstrip("\n") for l in open(f"{TD}/test_{TEST_SET}_vanilla.txt", encoding="utf-8") if l.strip()]
# difficult:011 (斟酌) carried the pre-correction heiban reading; the author's NHK verification
# made it atamadaka, so the synthesis input is the corrected copy -- the SAME file Sarashina and
# Irodori use. Readings are identical between the two files (reading metric unaffected); the
# accent arms are not.
DIFF_LORA = os.environ.get("DIFF_LORA",
    f"{paths.REPO}/data/derived/test_difficult_50_lora_fixed.txt"
    if TEST_SET == "difficult_50" else f"{TD}/test_{TEST_SET}_lora.txt")
lora = [l.rstrip("\n") for l in open(DIFF_LORA, encoding="utf-8") if l.strip()]  # CV2 <PHON_*> native
yomi = [l.rstrip("\n") for l in open(f"{TD}/test_{TEST_SET}_yomi.txt", encoding="utf-8") if l.strip()]
n = min(len(vanilla), len(lora), len(yomi))
print(f"difficult sentences: {n}", flush=True)

CAN = str.maketrans({"ヂ": "ジ", "ヅ": "ズ", "ヴ": "ブ"})
def norm(s): return re.sub(r"[^ァ-ヶー]", "", s).translate(CAN)
scorer = KanaScorer(device=0)
TAKES = []          # per-item guard record (reported)


# ITEM_SPK_JSON + ITEM_SPK_CELL: per-WORD reference prompt for the 42-voice listening kit. The
# prompt is an instance attribute of CVSynth, so it is switched per item without reloading; the
# same _vad_prompt() prep as every other CV2 path. Output layout spk{s}/{arm}/{i:02d}.wav = sweep42/.
_ASSIGN = None
if os.environ.get("ITEM_SPK_JSON"):
    from cv_synth import _vad_prompt
    _sel = json.load(open(f"{paths.DATA}/eval_speakers/selected_42.json"))["selected"]
    _PROMPT = {i: (_vad_prompt(s["wav"]), s["text"]) for i, s in enumerate(_sel)}
    _ASSIGN = json.load(open(os.environ["ITEM_SPK_JSON"]))["cells"][os.environ["ITEM_SPK_CELL"]]
    _ASSIGN = {int(k): list(v) for k, v in _ASSIGN.items()}
    print(f"[assign] per-word prompts, cell={os.environ['ITEM_SPK_CELL']}: "
          f"{sum(len(v) for v in _ASSIGN.values())} (word,speaker) pairs over {len(_PROMPT)} refs42", flush=True)


def run(g, arm, texts, text_frontend):
    cers = []
    for i in range(n):
        spks = _ASSIGN.get(i, [None]) if _ASSIGN is not None else [None]
        wav = None
        for sidx in spks:
            if sidx is not None:
                g.prompt_wav, g.prompt_text = _PROMPT[sidx]
                od = os.path.join(OUT, f"spk{sidx}", arm)
            else:
                od = os.path.join(OUT, arm)
            os.makedirs(od, exist_ok=True)
            wp = os.path.join(od, f"{i:02d}.wav")
            if os.path.exists(wp):
                # Already synthesised from an identical input: keep it. A partial re-run must not
                # regenerate audio it is not meant to touch.
                continue
            # uniform degeneracy guard, identical for every arm (scripts/eval/degenerate.py)
            wav, _, ginfo = synth_valid(
                lambda sd: (g.synth(texts[i], text_frontend=text_frontend, seed=sd), g.sr),
                5000 + i, asr=lambda w, r: safe_asr(lambda x, y: scorer.transcribe(x, y), w, r))
            sf.write(wp, wav, g.sr)
            TAKES.append(dict(arm=arm, spk=sidx, idx=i, **ginfo))
            _guard_hyp = ginfo.get("hyp")
            if ginfo["takes"] > 1:
                print(f"    guard {arm}/{i:02d}: {ginfo['takes']} takes {ginfo['rejected']}", flush=True)
        if wav is None:
            # every take for this item was already on disk: nothing was synthesised, so there is no
            # waveform to score here. The auxiliary CER simply has no entry for it.
            continue
        # the guard already transcribed this waveform with this model; a second identical ASR pass
        # per item was pure waste. (This CER is an auxiliary; the reported metric is computed by
        # scripts/eval/measure_reading_all.py in a separate uniform pass over the written wavs.)
        hyp = norm(_guard_hyp if _guard_hyp is not None
                   else safe_asr(lambda w, r: scorer.transcribe(w, r), wav, g.sr))
        ref = norm(pyopenjtalk.g2p(yomi[i], kana=True))
        cers.append(1 - difflib.SequenceMatcher(None, ref, hyp).ratio())
    c = np.array(cers) if cers else np.array([float("nan")])
    print(f"  {arm:8s}: Kana-CER {c.mean():.3f}  CER>0.5 {int((c>0.5).sum())}", flush=True)
    return c


# ARMS lets a second adapter (e.g. the released UtterTune weights) be run for its "ours" arm only:
# noedit and kana never touch the adapter, so synthesizing them once per SPEAKER rather than once
# per (speaker, adapter) removes pure duplication. Default: all three.
ARMS = [a.strip() for a in os.environ.get("ARMS", "noedit,kana,ours").split(",") if a.strip()]
cer = {}
if {"noedit", "kana"} & set(ARMS):
    gb = make_measurement_synth("cv2")           # base (noedit + kana)
    if "noedit" in ARMS:
        cer["noedit"] = run(gb, "noedit", vanilla, True)
    if "kana" in ARMS:
        cer["kana"] = run(gb, "kana", yomi, True)
    del gb
if "ours" in ARMS:
    go = make_measurement_synth("cv2", lora_dir=LORA)   # + data-free adapter; tags preserved
    cer["ours"] = run(go, "ours", lora, False)

res = {a: {"kana_cer_mean": float(c.mean()), "cer>0.5": int((c > 0.5).sum())}
       for a, c in cer.items()}
res["arms"] = ARMS
res["prompt_wav"] = os.environ.get("CV_PROMPT_WAV", "(default fictional)")
res["n"] = n
json.dump(res, open(os.path.join(OUT, "blockA_cv2_difficult.json"), "w"), ensure_ascii=False, indent=1)
save_guard_record(os.path.join(OUT, "guard_takes.json"), TAKES)
print(f"DONE -> {OUT}  (CV2 Block A difficult; accent = ear/rater)", flush=True)
