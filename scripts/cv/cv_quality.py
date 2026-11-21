"""Kana-whisper scoring of CosyVoice output.

KanaScorer.transcribe is the ASR half of the degeneration guard in blockA_cv2_difficult.py.
KanaScorer.score rates a take by its kana CER against the expected text's reading plus a penalty for
a duration far from the expected one (runaway or truncated take); lower is better.
"""
import os, re, difflib, tempfile
import numpy as np, soundfile as sf

_CAN = str.maketrans({"ヂ": "ジ", "ヅ": "ズ", "ヴ": "ブ"})


def _ref_kana(s):
    import pyopenjtalk
    return pyopenjtalk.g2p(s.replace("。", "").replace("、", ""), kana=True).replace(" ", "").translate(_CAN)


def _mora_count(kana):
    # rough mora estimate: kana chars minus small combiners (ゃゅょ etc. don't add a mora)
    return len(re.sub(r"[ァィゥェォャュョヮ]", "", kana))


class KanaScorer:
    def __init__(self, device=0):
        import torch
        from transformers import pipeline
        self.kw = pipeline("automatic-speech-recognition", model="sbintuitions/kana-whisper",
                           device=device, torch_dtype=torch.float16,
                           generate_kwargs={"language": "ja", "task": "transcribe"})

    def transcribe(self, wav, sr):
        # pipeline wants 16k mono; pass raw array + sr so it resamples
        return str(self.kw({"raw": np.asarray(wav, dtype=np.float32), "sampling_rate": sr})["text"])

    def score(self, wav, sr, expected_text, dur_penalty=0.4, dev_free=0.5):
        """Return (score, cer, hyp, dur). Lower score = better take."""
        ref = _ref_kana(expected_text)
        hyp_raw = self.transcribe(wav, sr)
        hyp = re.sub(r"[^ァ-ヶー]", "", hyp_raw).translate(_CAN)
        cer = 1 - difflib.SequenceMatcher(None, ref, hyp).ratio()
        dur = len(wav) / sr
        exp_dur = max(0.5, _mora_count(ref) * 0.17)  # ~0.17 s/mora
        dev = abs(dur - exp_dur) / exp_dur
        score = cer + dur_penalty * max(0.0, dev - dev_free)  # only penalize dev beyond dev_free
        return round(score, 4), round(cer, 3), hyp, round(dur, 2)

