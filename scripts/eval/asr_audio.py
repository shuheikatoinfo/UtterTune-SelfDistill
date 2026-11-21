"""Single entry point for feeding audio to a HuggingFace ASR pipeline.

Why this exists: passing raw audio with its NATIVE sampling_rate lets the pipeline resample
internally, and that path BREAKS on 48 kHz input — the model receives aliased audio and returns a
fluent-looking transcript of something else. Same wav, same model, same code, in two independent
environments (torch 2.8.0+cu128 and 2.11.0+cu130):

    {"raw": a, "sampling_rate": 48000}                  -> ツマエーノーヤ、スーセキノオーブンニ…
    librosa.resample(a, 48000, 16000), sampling_rate=16000 -> チミモーリョーガ、シズカナモリニ… (correct)

24 kHz input is unaffected, so the failure looks backbone-specific (44.1/48 kHz backbones) when it is
not.

Always resample explicitly before the ASR call. Whisper-family models operate at 16 kHz.
"""
from __future__ import annotations
import numpy as np

TARGET_SR = 16000


def to_asr_input(wav, sr: int) -> dict:
    """Return the pipeline payload with audio resampled to 16 kHz (mono, float32)."""
    a = np.asarray(wav, dtype=np.float32)
    if a.ndim > 1:
        a = a.mean(axis=1)
    sr = int(sr)
    if sr != TARGET_SR:
        try:
            import librosa
            a = librosa.resample(a, orig_sr=sr, target_sr=TARGET_SR)
        except Exception:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(sr, TARGET_SR)
            a = resample_poly(a, TARGET_SR // g, sr // g)
        a = np.asarray(a, dtype=np.float32)
    return {"raw": a, "sampling_rate": TARGET_SR}
