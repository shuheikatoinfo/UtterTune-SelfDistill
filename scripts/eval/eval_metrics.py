#!/usr/bin/env python
"""Reusable OBJECTIVE-METRICS package for the paper's evaluation (experiment-design §4.4).

Three auxiliary objective metrics, all driven by lazily-loaded model singletons so importing
this module is cheap and you only pay for the models you actually call:

  1. UTMOSv2  -- naturalness MOS prediction (screening).      utmos(wav_paths) -> [float]
     model: sarulab-speech/UTMOSv2, config 'fusion_stage3' fold0 (SSL backbone facebook/wav2vec2-base)
  2. ERes2Net -- speaker-similarity cosine.                   eres2net_cosine(a, b) -> float
     model id: iic/speech_eres2net_sv_zh-cn_16k-common (modelscope / 3D-Speaker), 192-d embeddings
  3. whisper-large-v3 kana-constrained CER -- intelligibility. whisper_kana_cer(wav, ref) -> float
     model: openai/whisper-large-v3, decoding logit-masked to U+3000..U+30FF (kana range)

Runs on the RTX 5090 (sm_120) inside scripts/eval/.venv_metrics (torch 2.8.0+cu128); kept fully
separate from the CosyVoice .venv so it cannot disturb that environment.

CLI:  python eval_metrics.py --wav-dir DIR --ref REF.jsonl --out OUT.json [--ref-field F] [--ref-g2p]
      -> per-file UTMOSv2 + kana-CER as JSON (and prints means).
"""
from __future__ import annotations
import os
import re
import json
import argparse
from functools import lru_cache

import numpy as np

# ---------------------------------------------------------------------------
# Katakana normalization (shared with the existing blockA eval pipeline).
# Keep only katakana + prolonged mark; canonicalize voiced variants so that
# ヂ/ヅ/ヴ do not spuriously inflate CER against dictionary readings.
# ---------------------------------------------------------------------------
_CAN = str.maketrans({"ヂ": "ジ", "ヅ": "ズ", "ヴ": "ブ"})


def kata_norm(s: str) -> str:
    """Normalize an ASR/reference string to a bare katakana string for CER.

    Hiragana is folded to katakana first (whisper under the kana constraint emits
    hiragana for native readings), then everything except katakana + 'ー' is dropped.
    """
    import jaconv
    return re.sub(r"[^ァ-ヶー]", "", jaconv.hira2kata(str(s))).translate(_CAN)


def pron_from_text(text: str) -> str:
    """Full katakana reading of a (kanji/mixed) sentence via pyopenjtalk, normalized.

    Used to build kana references from a 'yomi' line whose target word is already in
    katakana and the rest in kanji (see data/test_sentences/*_yomi.txt).
    """
    import pyopenjtalk
    return kata_norm(pyopenjtalk.g2p(text, kana=True))


def strip_tags(s: str) -> str:
    """Drop UtterTune-style <...> pron tags before normalizing an existing kana reference."""
    return re.sub(r"<[^>]*>", "", str(s))


# ---------------------------------------------------------------------------
# 1. UTMOSv2 -- naturalness MOS
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _utmos_model():
    import utmosv2
    return utmosv2.create_model(pretrained=True, config="fusion_stage3", fold=0, device="auto")


def utmos(wav_paths, verbose: bool = False):
    """Predict UTMOSv2 naturalness MOS for one path or a list of paths.

    Returns a float for a single path, else a list[float] aligned with wav_paths.
    """
    single = isinstance(wav_paths, (str, os.PathLike))
    paths = [wav_paths] if single else list(wav_paths)
    m = _utmos_model()
    scores = [float(m.predict(input_path=str(p), verbose=verbose)) for p in paths]
    return scores[0] if single else scores


# ---------------------------------------------------------------------------
# 2. ERes2Net -- speaker embedding + cosine similarity
# ---------------------------------------------------------------------------
ERES2NET_MODEL_ID = "iic/speech_eres2net_sv_zh-cn_16k-common"


@lru_cache(maxsize=1)
def _eres2net_pipeline():
    from modelscope.pipelines import pipeline
    return pipeline(task="speaker-verification", model=ERES2NET_MODEL_ID)


def eres2net_embed(wav) -> np.ndarray:
    """L2-normalized 192-d ERes2Net speaker embedding for a single wav path."""
    p = _eres2net_pipeline()
    emb = np.asarray(p([str(wav)], output_emb=True)["embs"][0], dtype=np.float64)
    n = np.linalg.norm(emb)
    return emb / n if n > 0 else emb


def eres2net_cosine(wav_a, wav_b) -> float:
    """Cosine similarity between the ERes2Net embeddings of two wavs."""
    p = _eres2net_pipeline()
    embs = np.asarray(p([str(wav_a), str(wav_b)], output_emb=True)["embs"], dtype=np.float64)
    a, b = embs[0], embs[1]
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 0 else 0.0


def cosine(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    a = np.asarray(emb_a, dtype=np.float64)
    b = np.asarray(emb_b, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# 3. whisper-large-v3 kana-constrained CER -- intelligibility
# ---------------------------------------------------------------------------
WHISPER_MODEL_ID = "openai/whisper-large-v3"
# Allowed output code-point range: U+3000..U+30FF (CJK symbols/punct + hiragana + katakana).
_KANA_LO, _KANA_HI = 0x3000, 0x30FF


@lru_cache(maxsize=1)
def _whisper():
    import torch
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperProcessor,
        LogitsProcessor,
        LogitsProcessorList,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    proc = WhisperProcessor.from_pretrained(WHISPER_MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(
        WHISPER_MODEL_ID, torch_dtype=dtype
    ).to(device).eval()
    tok = proc.tokenizer

    def _allowed(tid: int) -> bool:
        s = tok.decode([tid])
        return len(s) > 0 and all(_KANA_LO <= ord(c) <= _KANA_HI for c in s)

    allowed = [i for _, i in tok.get_vocab().items() if _allowed(i)]
    eos = model.generation_config.eos_token_id
    if isinstance(eos, (list, tuple)):
        allowed.extend(int(e) for e in eos)
    else:
        allowed.append(int(eos))

    mask = torch.full((model.config.vocab_size,), float("-inf"), dtype=torch.float32)
    mask[torch.tensor(sorted(set(allowed)))] = 0.0

    class _KanaConstraint(LogitsProcessor):
        def __call__(self, input_ids, scores):
            return scores + mask.to(scores.device, scores.dtype)

    lp = LogitsProcessorList([_KanaConstraint()])
    return proc, model, lp, device, dtype


def _load_audio_16k(wav) -> np.ndarray:
    import soundfile as sf
    import librosa
    a, sr = sf.read(str(wav), dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != 16000:
        a = librosa.resample(a, orig_sr=sr, target_sr=16000)
    return a


def whisper_kana_transcribe(wav, max_new_tokens: int = 200) -> str:
    """Kana-constrained whisper-large-v3 transcription, normalized to katakana."""
    proc, model, lp, device, dtype = _whisper()
    audio = _load_audio_16k(wav)
    feat = proc(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype)
    import torch
    with torch.no_grad():
        ids = model.generate(
            feat, language="ja", task="transcribe",
            max_new_tokens=max_new_tokens, logits_processor=lp,
        )
    return kata_norm(proc.batch_decode(ids, skip_special_tokens=True)[0])


def whisper_kana_cer(wav, ref_katakana: str) -> float:
    """Character error rate of the kana-constrained transcription vs a katakana reference.

    ref_katakana is normalized with kata_norm (tags stripped), so you may pass a raw
    dictionary reading such as 'ショー<ガツ>ニワ...' directly.
    """
    import jiwer
    ref = kata_norm(strip_tags(ref_katakana))
    hyp = whisper_kana_transcribe(wav)
    if not ref:
        return 0.0 if not hyp else 1.0
    return float(jiwer.cer(ref, hyp))


# ---------------------------------------------------------------------------
# CLI: per-file UTMOSv2 + kana-CER over a directory of wavs against a ref jsonl
# ---------------------------------------------------------------------------
_REF_FIELDS = ["reading", "ref_katakana", "normalized_pron", "pron", "yomi", "ref", "text"]


def _match_wavs(wav_dir, records):
    """Pair jsonl records to wav paths.

    Priority per record: explicit 'wav' field -> '{key}.wav' if it exists -> positional
    (sorted wav order). Returns list of (wav_path, ref_string) in record order.
    """
    wavs_sorted = sorted(
        [os.path.join(wav_dir, f) for f in os.listdir(wav_dir) if f.lower().endswith(".wav")]
    )
    pairs = []
    for i, rec in enumerate(records):
        wp = None
        if rec.get("wav"):
            wp = rec["wav"] if os.path.isabs(rec["wav"]) else os.path.join(wav_dir, rec["wav"])
        elif rec.get("key") and os.path.exists(os.path.join(wav_dir, f"{rec['key']}.wav")):
            wp = os.path.join(wav_dir, f"{rec['key']}.wav")
        elif i < len(wavs_sorted):
            wp = wavs_sorted[i]
        pairs.append((wp, rec))
    return pairs


def _pick_ref(rec, field):
    if field:
        return rec.get(field, "")
    for f in _REF_FIELDS:
        if rec.get(f):
            return rec[f]
    return ""


def main():
    ap = argparse.ArgumentParser(description="Per-file UTMOSv2 + kana-CER over a wav dir.")
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--ref", required=True, help="jsonl with per-item reference readings")
    ap.add_argument("--out", required=True, help="output json path")
    ap.add_argument("--ref-field", default=None, help="jsonl field holding the reference reading")
    ap.add_argument("--ref-g2p", action="store_true",
                    help="run pyopenjtalk on the ref field (yomi with kanji) to get kana")
    ap.add_argument("--limit", type=int, default=None, help="only process first N records")
    ap.add_argument("--no-utmos", action="store_true")
    ap.add_argument("--no-cer", action="store_true")
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.ref, encoding="utf-8") if l.strip()]
    if args.limit:
        records = records[: args.limit]
    pairs = _match_wavs(args.wav_dir, records)

    items = []
    for wp, rec in pairs:
        if wp is None or not os.path.exists(wp):
            print(f"  [skip] no wav for record key={rec.get('key')}", flush=True)
            continue
        ref_raw = _pick_ref(rec, args.ref_field)
        ref = pron_from_text(strip_tags(ref_raw)) if args.ref_g2p else kata_norm(strip_tags(ref_raw))
        row = {"wav": wp, "key": rec.get("key"), "ref_kana": ref}
        if not args.no_utmos:
            row["utmos"] = utmos(wp)
        if not args.no_cer:
            hyp = whisper_kana_transcribe(wp)
            row["hyp_kana"] = hyp
            import jiwer
            row["kana_cer"] = 0.0 if not ref else float(jiwer.cer(ref, hyp)) if hyp or ref else 0.0
        items.append(row)
        msg = f"  {row.get('key') or os.path.basename(wp)}"
        if "utmos" in row:
            msg += f"  UTMOS={row['utmos']:.3f}"
        if "kana_cer" in row:
            msg += f"  kana_CER={row['kana_cer']:.3f}"
        print(msg, flush=True)

    summary = {"n": len(items)}
    if items and "utmos" in items[0]:
        summary["utmos_mean"] = float(np.mean([r["utmos"] for r in items]))
    if items and "kana_cer" in items[0]:
        summary["kana_cer_mean"] = float(np.mean([r["kana_cer"] for r in items]))
    out = {"summary": summary, "items": items,
           "models": {"utmos": "sarulab-speech/UTMOSv2 fusion_stage3/fold0",
                      "whisper": WHISPER_MODEL_ID, "eres2net": ERES2NET_MODEL_ID}}
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"\nsummary: {summary}\n-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
