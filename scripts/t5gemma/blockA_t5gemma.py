"""Block A (backbone-agnostic) — T5Gemma-TTS, the ENCODER-DECODER AR backbone (design §2.2, axis-3 driver).

Mirror of scripts/blockA_sarashina_difficult.py, three arms over the same test set:

  noedit : plain kanji on the frozen base (No-edit).
  kana   : the target word rewritten as whole-word katakana (UtterTune kana-input reading baseline;
           fixes READING, not accent). No tags.
  ours   : the UtterTune-tagged line, i.e. <PHON_START>kana'/<PHON_END> with ' = accent nucleus,
           / = phrase boundary.

GATE QUESTION (③ ear gate — this is the whole reason the T5Gemma 'ours' arm exists right now):
  T5Gemma-TTS has NO trained pronunciation-control tokens. Its text side is the *stock* Gemma
  SentencePiece tokenizer (google/t5gemma-2b-2b-ul2, vocab 256000). Verified on CPU:
     '<PHON_START>' -> ['<','PH','ON','_','START','>']   (6 junk subwords, NOT one control token)
     '<PHON_END>'   -> ['<','PH','ON','_','END','>']
     "'"  -> id 235303   (ordinary apostrophe token)
     '/'  -> id 235283   (ordinary slash token)
  So the 'ours' arm here is the T5Gemma analogue of the "PronSteering-empty" pass-through check we
  ran on Sarashina: it measures whether feeding the RAW tags either
     (a) DEGRADES the utterance (the encoder tries to voice "< PH ON _ START >" / the ' and / as
         audible glitches, tanking CER), or
     (b) is ROBUSTLY IGNORED / passed through (the model just reads the katakana inside, ' and /
         do nothing — reading may improve like the kana arm, but accent is uncontrolled).
  Either way, this is a data-free probe of a frozen backbone: if (b), T5Gemma is a candidate for the
  same input-layer tag-substitution + self-distill recipe (accent would then require training PHON
  tokens, exactly as on Sarashina); if the reading also breaks, the backbone is a poor axis-3 point.
  There is NO reliable automatic accent-nucleus metric — accent is decided by the ear / raters on the
  wavs this script emits. Kana-whisper CER (vs the katakana-reading reference) is the READING signal only.

CPU-forbidden note: this script loads the 2b-2b model + XCodec2 + kana-whisper and must run on GPU.

Usage (venv = $T5GEMMA_DIR/.venv_t5gemma):
  cd $T5GEMMA_DIR
  .venv_t5gemma/bin/python <repo>/scripts/t5gemma/blockA_t5gemma.py --set difficult
  # smoke test first: add --limit 3
  # other sets: --set easy | --set katakana   (provisional file maps; 'difficult' is the validated headline)
"""
import sys, os, re, json, difflib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/, for paths.py
import paths

T5G = paths.T5GEMMA_DIR
sys.path.insert(0, T5G)  # so `data.tokenizer`, `duration_estimator`, `inference_tts_utils` resolve

import fire
import numpy as np
import torch
import soundfile as sf
import pyopenjtalk
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'eval'))
from degenerate import synth_valid, safe_asr
# The guard ASR must see 16 kHz audio: transformers 4.57 does not resample a raw-array payload, so
# 44.1 kHz T5Gemma audio would reach kana-whisper unchanged (PREREGISTRATION_v2 section 8).
from asr_audio import to_asr_input
TAKES = []          # per-item guard record (reported)
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from data.tokenizer import AudioTokenizer
from duration_estimator import estimate_duration
from inference_tts_utils import inference_one_sample

# TEST_DIR / TEST_SET make the item set swappable without touching the synthesis path:
# the same code, the same recipe, only a different input file. Needed for the 105-item
# SELECTION split and the 319-item TEST split (preregistration/items/objective_*.jsonl).
# Default: difficult-50.
TD = (os.environ.get("TEST_DIR") or paths.ITEM_SOURCE)
TEST_SET = os.environ.get("TEST_SET", "difficult_50")

# Cross-backbone reference voice: reuse the exact narration prompt + transcript used by
# blockA_sarashina_difficult.py so the T5Gemma arm is voice-comparable to the Sarashina arm.
SP = f"{paths.DATA}/_handoff"
DEFAULT_REF = SP + "/sara-ref/samples/styles/ナレーション_prompt.wav"
DEFAULT_REF_MIRROR = f"{paths.DATA}/_handoff/sara-ref/samples/styles/ナレーション_prompt.wav"
DEFAULT_REFTEXT = "彼らは王権の正当性を重視し革命の影響を抑え込むことを目指しました"

# per-set arm -> (filename, strip_phon_tags?). strip=True removes the <PHON_*> wrappers but keeps the
# inner (katakana) text -> that is the tag-free "kana" reading baseline. strip=False = raw tags (gate).
SETS = {
    "difficult": {
        "noedit": (f"test_{TEST_SET}_vanilla.txt", False),
        "kana":   (f"test_{TEST_SET}_yomi.txt", False),   # whole target word already in katakana, no tags
    # difficult:011 (斟酌) carried the pre-correction heiban reading; the author's NHK
    # verification made it atamadaka, so the synthesis input is taken from the corrected
    # copy -- the SAME file Sarashina and Irodori use. Readings are identical between the
    # two files, so the reading metric is unaffected; the accent arms are not.
        "ours":   (f"{paths.REPO}/data/derived/test_difficult_50_lora_fixed.txt" if TEST_SET == "difficult_50"
                   else f"test_{TEST_SET}_lora.txt", False),
        "ref":    (f"test_{TEST_SET}_yomi.txt", False),   # reading reference for kana-CER
    },
    # easy / katakana: provisional maps (headline+GPU-gate is 'difficult'). phon = word-level tag,
    # phonfull = whole-sentence UtterTune notation (carries the ' accent marks).
    "easy": {
        "noedit": ("test_easy_50.txt", False),
        "kana":   ("test_easy_50_phon.txt", True),
        "ours":   ("test_easy_50_phonfull.txt", False),
        "ref":    ("test_easy_50_phon.txt", True),
    },
    "katakana": {
        "noedit": ("test_katakana_50.txt", False),
        "kana":   ("test_katakana_50_phon.txt", True),
        "ours":   ("test_katakana_50_phonfull.txt", False),
        "ref":    ("test_katakana_50_phon.txt", True),
    },
}

CAN = str.maketrans({"ヂ": "ジ", "ヅ": "ズ", "ヴ": "ブ"})
_PHON_RE = re.compile(r"<PHON_START>|<PHON_END>|</?b>")

# T5Gemma control tokens. The literal "<PHON_START>" is NOT a token here: Gemma splits it into
# ['<','PH','ON','_','START','>'] and the decoder speaks those subwords aloud (measured: reading hit
# 0.74 -> 0.46, target word replaced by "fuon-endoo"). Gemma's reserved <unusedN> rows ARE single
# tokens (ids 7/8) that never occur in text, so the training data and this eval both use them.
# Must match TAG_S/TAG_E in gen_teachers_t5gemma.py.
TAG_S, TAG_E = "<unused0>", "<unused1>"


def retag(s):
    """Test files carry the portable <PHON_*> spelling; map it to this backbone's real tokens."""
    return s.replace("<PHON_START>", TAG_S).replace("<PHON_END>", TAG_E)


def kana_only(s):
    """Keep only katakana/long-vowel, canonicalize — matches blockA_sarashina norm() for kana-CER."""
    return re.sub(r"[^ァ-ヶー]", "", s).translate(CAN)


def strip_tags(s):
    """Remove the <PHON_*>/<b> wrappers but keep the inner text (used for the kana baseline arm)."""
    return _PHON_RE.sub("", s)


def clean_for_g2p(s):
    """Strip tags AND UtterTune/PronSteering notation so pyopenjtalk sees clean kana/kanji text."""
    s = strip_tags(s)
    return re.sub(r"[＼'/\[\]｜|]", "", s)


def _read(fname):
    path = fname if os.path.isabs(fname) else os.path.join(TD, fname)
    with open(path, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f if l.strip()]


def _device_str(model, fallback="cuda"):
    dm = getattr(model, "hf_device_map", None)
    if not dm:
        return fallback
    for dev in dm.values():
        if dev in ("cpu", "disk"):
            continue
        if isinstance(dev, int):
            return f"cuda:{dev}"
        return str(dev)
    return fallback


def main(
    set="difficult",
    model_dir=T5G + "/pretrained/T5Gemma-TTS-2b-2b",
    reference_speech=DEFAULT_REF,
    reference_text=DEFAULT_REFTEXT,
    out_root=paths.DATA,
    ours_mode="tokens",       # "tokens" = map <PHON_*> to the real <unusedN> control tokens (the
                              #   method; matches training). "raw" = feed the literal <PHON_*>
                              #   string, which this tokenizer shreds -- kept only as the negative
                              #   probe that established why "tokens" is required. "strip" = tags
                              #   removed, ' / kept.
    top_k=30,
    top_p=0.9,
    temperature=0.8,
    shard="0/1",              # i/n: synthesize only items with index % n == i (seeds and voices stay keyed on
                              #   the global index); a final run with shard=0/1 over the merged wavs scores all.
    seed0=5000,
    limit=None,               # cap #sentences for a smoke test (e.g. 3)
    arms="noedit,kana,ours",
    ours_surface=0,           # reading-only self-distill: feed the ours arm the SURFACE (vanilla) text
                              #   (not the PHON-tag file) so a merged-LoRA model_dir is tested on plain kanji.
    ours_file=None,           # reading+accent: feed the ours arm this file verbatim (abs path ok); ' kept
                              #   (accent nucleus is a stock Gemma token). Overrides ours_surface.
):
    set_name = set
    assert set_name in SETS, f"unknown set {set_name}; choose {list(SETS)}"
    cfg_files = SETS[set_name]
    # Fire turns a bare "a,b,c" into a tuple, and str(tuple) then yields "('a', 'b', 'c')",
    # which split(",") shreds into "('a'" etc. Accept both a real sequence and a string.
    if isinstance(arms, (list, tuple)):
        arms = [str(a).strip() for a in arms if str(a).strip()]
    else:
        arms = [a.strip() for a in str(arms).split(",") if a.strip()]

    no_ref = reference_speech is None or str(reference_speech).lower() in {"none", "", "null"}
    if not no_ref and not os.path.exists(reference_speech) and os.path.exists(DEFAULT_REF_MIRROR):
        reference_speech = DEFAULT_REF_MIRROR  # fall back to the $DATA mirror of the narration prompt
    prefix_transcript = "" if no_ref else str(reference_text or "")

    out = os.path.join(out_root, f"blockA_t5gemma_{set_name}")
    for a in arms:
        os.makedirs(os.path.join(out, a), exist_ok=True)

    # --- load text per arm + the reading reference ---
    texts = {}
    for a in arms:
        fname, strip = cfg_files[a]
        if a == "ours" and ours_file:
            fname, strip = ours_file, False                # accented-kana input verbatim (' kept)
        elif a == "ours" and ours_surface:
            fname, strip = cfg_files["noedit"][0], False   # plain kanji -> merged-LoRA model
        lines = _read(fname)
        if a == "ours" and ours_mode == "strip":
            strip = True
        texts[a] = [strip_tags(x) if strip else x for x in lines]
        if a == "ours" and ours_mode == "tokens":
            texts[a] = [retag(x) for x in texts[a]]
            n_tag = sum(1 for x in texts[a] if TAG_S in x)
            assert n_tag == len(texts[a]), (
                f"ours_mode=tokens but {len(texts[a]) - n_tag} lines carry no control token; "
                f"the adapter was trained on tagged input and untagged input is out of distribution")
    ref_fname, ref_strip = cfg_files["ref"]
    ref_lines = _read(ref_fname)
    ref_texts = [clean_for_g2p(x) for x in ref_lines]
    # plain (noedit) lines drive the shared duration budget so every arm of sentence i gets the SAME
    # target length — removes a confound and keeps the arms honestly comparable.
    plain_lines = [clean_for_g2p(x) for x in _read(cfg_files["noedit"][0])]

    n = min([len(texts[a]) for a in arms] + [len(ref_texts), len(plain_lines)])
    if limit:
        n = min(n, int(limit))
    print(f"[T5Gemma Block A] set={set_name} n={n} arms={arms} ours_mode={ours_mode} "
          f"ref={'NONE' if no_ref else os.path.basename(reference_speech)}", flush=True)

    # --- load model (bf16, device_map=auto) ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto")
    if not getattr(model, "hf_device_map", None) and device != "cpu":
        model = model.to(device)
    device = _device_str(model, device)
    model.eval()
    cfg = model.config

    tk_name = getattr(cfg, "text_tokenizer_name", None) or getattr(cfg, "t5gemma_model_name", None)
    text_tokenizer = AutoTokenizer.from_pretrained(tk_name)
    audio_tokenizer = AudioTokenizer(backend="xcodec2", model_name=getattr(cfg, "xcodec2_model_name", None))
    SR = int(audio_tokenizer.sample_rate)
    codec_sr = int(getattr(cfg, "encodec_sr", 50))
    prompt_end_frame = 0
    if not no_ref:
        info = sf.info(reference_speech)
        prompt_end_frame = int(100 * info.samplerate)  # cut_off_sec=100; ref is short so loads whole file

    kw = pipeline("automatic-speech-recognition", model="sbintuitions/kana-whisper", device=0,
                  torch_dtype=torch.float16, generate_kwargs={"language": "ja", "task": "transcribe"})

    # ITEM_SPK_JSON: one reference voice per ITEM (42-voice objective table). reference_speech is
    # consumed per call below, so the voice changes without reloading the model.
    ITEM_REF = {}
    if os.environ.get("ITEM_SPK_JSON"):
        _a = json.load(open(os.environ["ITEM_SPK_JSON"]))["cells"][os.environ["ITEM_SPK_CELL"]]
        _refs = json.load(open(f"{paths.DATA}/eval_speakers/selected_42.json"))["selected"]
        ITEM_REF = {int(k): (_refs[v[0]]["wav"], _refs[v[0]]["text"]) for k, v in _a.items()}
        print(f"[assign] per-item references: {len(ITEM_REF)} items over "
              f"{len({w for w, _ in ITEM_REF.values()})} voices", flush=True)

    def _ref_of(i):
        return ITEM_REF.get(i, (reference_speech, prefix_transcript))

    # shared per-sentence duration budget (from the plain text)
    durs = [float(estimate_duration(target_text=plain_lines[i],
                                    reference_speech=None if no_ref else _ref_of(i)[0],
                                    reference_transcript=None if no_ref else _ref_of(i)[1],
                                    target_lang="ja", reference_lang="ja")) for i in range(n)]

    def synth(text, dur, seed, ref=None, ptext=None, pend=None):
        torch.manual_seed(seed); np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        decode_config = {"top_k": top_k, "top_p": top_p, "min_p": 0, "temperature": temperature,
                         "stop_repetition": 3, "codec_audio_sr": SR, "codec_sr": codec_sr,
                         "silence_tokens": [], "sample_batch_size": 1}
        concat, gen = inference_one_sample(
            model=model, model_args=cfg, text_tokenizer=text_tokenizer, audio_tokenizer=audio_tokenizer,
            audio_fn=None if no_ref else (ref or reference_speech), target_text=text, lang="ja", device=device,
            decode_config=decode_config, prompt_end_frame=(pend if pend is not None else prompt_end_frame),
            target_generation_length=dur, prefix_transcript=(ptext if ptext is not None else prefix_transcript),
            multi_trial=[], repeat_prompt=0, return_frames=False)
        return gen[0].cpu().squeeze().numpy().astype("float32")

    _si, _sn = (int(x) for x in str(shard).split("/"))
    def run(arm):
        recs, cers = [], []
        for i in range(n):
            if i % _sn != _si:
                continue
            wp = os.path.join(out, arm, f"{i:02d}.wav")
            _guard_hyp = None
            if os.path.exists(wp):
                a = sf.read(wp, dtype="float32")[0]
            else:
                _rw, _rt = _ref_of(i)
                _pe = prompt_end_frame if not ITEM_REF else int(100 * sf.info(_rw).samplerate)
                # uniform degeneracy guard, identical for every arm (scripts/eval/degenerate.py)
                a, _, ginfo = synth_valid(
                    lambda sd: (synth(texts[arm][i], durs[i], seed=sd, ref=_rw, ptext=_rt, pend=_pe), SR),
                    seed0 + i, asr=lambda w, r: str(safe_asr(lambda x, y: kw(to_asr_input(x, y))["text"], w, r)))
                sf.write(wp, a, SR)
                TAKES.append(dict(arm=arm, idx=i, **ginfo))
                if ginfo["takes"] > 1:
                    print(f"    guard {arm}/{i:02d}: {ginfo['takes']} takes {ginfo['rejected']}", flush=True)
                _guard_hyp = ginfo.get("hyp")
            # the guard already transcribed this exact waveform with this exact model; reuse it
            hyp = kana_only(str(_guard_hyp if _guard_hyp is not None
                                 else safe_asr(lambda w, r: kw(to_asr_input(w, r))["text"], a, SR)))
            ref = kana_only(pyopenjtalk.g2p(ref_texts[i], kana=True))
            cer = 1.0 - difflib.SequenceMatcher(None, ref, hyp).ratio()
            cers.append(cer)
            recs.append({"i": i, "text": texts[arm][i], "ref_kana": ref, "hyp_kana": hyp,
                         "kana_cer": round(cer, 4), "dur_s": round(durs[i], 2), "wav_len_s": round(len(a) / SR, 2)})
            if i % 10 == 0:
                print(f"  {arm} {i}/{n} cer={cer:.2f} dur={durs[i]:.1f}s", flush=True)
        json.dump({"arm": arm, "records": recs}, open(os.path.join(out, arm, "meta.json"), "w"),
                  ensure_ascii=False, indent=1)
        return np.array(cers)

    results, summary = {}, {}
    for a in arms:
        c = run(a)
        results[a] = c
        summary[a] = {"kana_cer_mean": float(c.mean()), "kana_cer_median": float(np.median(c)),
                      "cer>0.5": int((c > 0.5).sum())}
    summary["n"] = n
    summary["set"] = set_name
    summary["ours_mode"] = ours_mode
    summary["reference"] = "none" if no_ref else os.path.basename(reference_speech)
    summary["gate"] = ("T5Gemma has NO trained PHON tokens: <PHON_START> tokenizes to 6 junk subwords, "
                       "' and / are ordinary punctuation. 'ours' vs 'kana'/'noedit' = raw-tag "
                       "degrade-vs-passthrough probe. Accent is judged by ear/raters on the wavs, "
                       "NOT by this CER (reading only).")
    json.dump(summary, open(os.path.join(out, f"blockA_{set_name}.json"), "w"), ensure_ascii=False, indent=1)

    print(f"\n== T5Gemma Block A {set_name} (n={n}) — reading Kana-CER (accent = ear/rater on wavs) ==", flush=True)
    for a in arms:
        c = results[a]
        print(f"  {a:16s}: Kana-CER {c.mean():.3f} (med {np.median(c):.3f})  CER>0.5 {int((c>0.5).sum())}", flush=True)
    print(f"DONE -> {out}", flush=True)
    print("GATE READ: if 'ours' Kana-CER ~ 'kana' (both << 'noedit') the raw tags pass through and "
          "T5Gemma reads the katakana inside (accent still uncontrolled -> needs PHON-token self-distill). "
          "If 'ours' Kana-CER blows up (>> 'noedit'), the raw < PH ON _ START > junk is voiced/degrades.", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
