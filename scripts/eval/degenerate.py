#!/usr/bin/env python3
"""One definition of "degenerate output", and one uniform retry policy, shared by every backbone.

Why this exists. A length check alone (under 0.5 s) catches only one way an autoregressive TTS
fails. The others are a single mora repeated (ンンンン…, エーエーエー…) and clips run to the generation
cap while silent; unguarded, they are scored as ordinary reading misses, at different rates per arm,
so the metric would partly measure sampler stability, not pronunciation.

The policy here is the same class as the guard it completes, and is applied IDENTICALLY to every arm:
  * reject a take if it is degenerate, redraw with a bumped seed, take the FIRST valid one;
  * never compare takes and keep the best -- that would be take selection, which this project forbids;
  * cap the retries, and RECORD the number of takes so the degeneration rate is a reported number
    rather than something inferred afterwards from a detector.
"""
import json
import re
import os
import numpy as np

# GUARD_MAX_TAKES=1 turns the guard OFF for a measurement run: the first take is kept whatever it
# is, while its rejection reason is still recorded. That is how the guard's own effect is measured
# -- the same items, the same seeds, with and without the redraw.
MAX_TAKES = int(os.environ.get("GUARD_MAX_TAKES", "6"))
MIN_S = 0.5
SILENT_RMS = 1e-3
QUIET_PEAK = 0.05      # linear; a usable take peaks far above this
QUIET_RMS = 10 ** (-45 / 20)   # -45 dBFS
SILENT_FRAC = 0.80
REP_RUN = 6
_MORA = re.compile(r"[ァ-ヶ][ャュョァィゥェォ]?ー?|ー")


# Deliberately NOT tested here: the longest run of silence inside a clip. 42 of the 501 listening
# clips hold an internal pause of 2 s or more, one of them 9.88 s, and every one passed. Adding the
# test would not be backbone-neutral -- 41 of those 42 are Sarashina (26 of 158 difficult, 15 of 105
# katakana) and CosyVoice2 has none, being the cleanest backbone on this measure despite degenerating
# readily in other ways. A filter that removes a quarter of one cell and nothing from two others is
# not the uniform, disclosed filter this study commits to. Measured, approved, and recorded in the
# preregistration instead.
def audio_reason(a, sr, cap_s=None):
    """Degeneracy visible in the waveform alone (no ASR needed). None = looks fine."""
    a = np.asarray(a, dtype="float32").reshape(-1)
    if a.size == 0:
        return "empty"
    d = a.size / float(sr)
    if d < MIN_S:
        return "too_short"
    if cap_s and d >= cap_s:
        return "runaway"                      # ran to the generation limit
    if d > 30.0:
        return "over_asr_limit"               # longer than Whisper reads in one pass; not speech we can score
    if float(np.sqrt((a ** 2).mean())) < SILENT_RMS:
        return "silent"
    # Audible-but-inaudible: a take can pass the silence test and still be far too quiet to judge.
    # Seven naturalness clips and one listening clip sat at peak 0.008-0.04 and -47 to -59 LUFS,
    # which is nothing a rater can hear at a normal setting; normalizing them only raises the noise.
    # Thresholds are well below normal speech here (a clip at -23 LUFS has RMS near -25 dBFS).
    if float(np.abs(a).max()) < QUIET_PEAK or float(np.sqrt((a ** 2).mean())) < QUIET_RMS:
        return "too_quiet"
    n = max(1, int(0.025 * sr))
    fr = a[: a.size // n * n].reshape(-1, n)
    if fr.size and float((( fr ** 2).mean(1) < 1e-7).mean()) > SILENT_FRAC:
        return "mostly_silent"
    return None


def text_reason(hyp):
    """Degeneracy visible in the ASR hypothesis. None = looks fine."""
    u = _MORA.findall(hyp or "")
    if len(u) < 2:
        return "no_speech"
    run = best = 1
    for x, y in zip(u, u[1:]):
        run = run + 1 if x == y else 1
        best = max(best, run)
    if best >= REP_RUN:
        return "repetition"
    if len(u) >= 12 and len(set(u)) / len(u) < 0.25:
        return "low_variety"
    return None


def synth_valid(gen, seed, asr=None, cap_s=None, max_takes=MAX_TAKES):
    """gen(seed) -> (audio, sr). Redraw on a degenerate take; return the FIRST valid one.

    Returns (audio, sr, info) with info = {"takes": n, "rejected": [reasons], "final_reason": r|None}.
    `final_reason` is set when every take was degenerate: the last take is returned anyway so no item
    is ever dropped (the project's rule), and the item is flagged for reporting.
    """
    rejected = []
    a = sr = hyp = None
    for t in range(max_takes):
        a, sr = gen(seed + 1000 * t)
        r = audio_reason(a, sr, cap_s)
        if r is None and asr is not None:
            try:
                hyp = asr(a, sr)             # handed back so callers need not transcribe twice
                r = text_reason(hyp)
            except Exception as e:
                # The ASR is the guard's instrument, not its master: if it cannot read the take
                # (e.g. Whisper refuses inputs over 30 s and demands long-form decoding), that is
                # evidence the take is degenerate, not a reason to abort the run (a runaway take can
                # exceed 30 s).
                hyp, r = None, f"asr_failed:{type(e).__name__}"
        if r is None:
            return a, sr, {"takes": t + 1, "rejected": rejected, "final_reason": None, "hyp": hyp}
        rejected.append(r)
    return a, sr, {"takes": max_takes, "rejected": rejected, "final_reason": rejected[-1], "hyp": hyp}


def safe_asr(asr, a, sr, limit_s=30.0):
    """Transcribe for SCORING without letting a broken take abort the run.

    Whisper refuses inputs over 30 s (it wants long-form decoding with timestamps), and a degenerate
    take can be longer than that. A clip we cannot read is scored as unreadable -- returning "" makes
    it a miss, which is the honest outcome -- but it must never raise, or one such take stops the
    whole run. Every ASR call in the synthesis scripts goes through here.
    """
    a = np.asarray(a, dtype="float32").reshape(-1)
    if a.size == 0 or a.size / float(sr) > limit_s:
        return ""
    try:
        return asr(a, sr)
    except Exception:
        return ""


def save_guard_record(path, rows):
    """Write the per-item guard record (one row per synthesized wav: arm, spk, idx, takes, rejected, ...).

    A run synthesizes only the wavs not yet on disk, so `rows` holds only this run's items. The file is
    therefore merged, keyed by (arm, spk, idx): rows from earlier runs are kept, and a row of this run
    replaces an earlier row for the same wav (that wav was deleted and synthesized again). Writing
    only this run's rows erased the record of every take kept from earlier runs.
    """
    merged = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for r in json.load(f):
                merged[(r["arm"], r.get("spk"), r["idx"])] = r
    for r in rows:
        merged[(r["arm"], r.get("spk"), r["idx"])] = r
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(merged.values()), f, ensure_ascii=False, indent=1)
