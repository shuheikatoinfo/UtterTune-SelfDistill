"""Best-of-N + automatic rejection for CV synthesis — the robust lever against the *stochastic*
artifacts (music-like textures / runaway tails that appear on some seeds even with a ZH prompt).
Generate N takes, score each, keep the best. Score = kana-whisper CER (hallucination/music -> ASR
garbage -> high CER) + a duration-deviation penalty (runaway/truncation). Lower is better. Same
rejection-sampling philosophy as the teacher QC in the Sarashina recipe.
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


def best_of(synth, text, n, scorer, expected_text=None, base_seed=7000, text_frontend=True):
    """Synthesize `text` n times (varied seed), score each, return the best.
    Returns dict{wav, sr, best, takes:[{seed,score,cer,hyp,dur}]}."""
    expected_text = expected_text or text
    takes, best = [], None
    for k in range(n):
        seed = base_seed + 101 * k
        wav = synth.synth(text, text_frontend=text_frontend, seed=seed)
        sc, cer, hyp, dur = scorer.score(wav, synth.sr, expected_text)
        rec = {"seed": seed, "score": sc, "cer": cer, "hyp": hyp, "dur": dur}
        takes.append(rec)
        if best is None or sc < best[0]:
            best = (sc, wav, rec)
    return {"wav": best[1], "sr": synth.sr, "best": best[2], "takes": takes}
