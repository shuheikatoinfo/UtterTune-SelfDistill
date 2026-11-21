"""Teacher reading check of the registered rule, shared by all four backbones.

A candidate teacher take is kept only if the ASR of record (kana-whisper) hears the target word's
reading in it, judged with the same canonicalisation as the paper's reading metric
(scripts/eval/measure_reading_all.py: canon() / norm(); copied here so that every backbone's
teacher generator, each in its own venv, applies one implementation). The check reads the target
reading only -- never the accent, never the rest of the sentence.
"""
import re

CAN = str.maketrans({"ヂ": "ジ", "ヅ": "ズ", "ヴ": "ブ", "ヲ": "オ"})
U_DAN = set("ウクグスズツヅヌフブプムユュルヴ"); O_DAN = set("オコゴソゾトドノホボポモヨョロヲ")
E_DAN = set("エケゲセゼテデネヘベペメレェ")


def norm(s):
    s = "".join(chr(ord(c) + 0x60) if 0x3041 <= ord(c) <= 0x3096 else c for c in (s or ""))
    return re.sub(r"[^ァ-ヶー]", "", s).translate(CAN)


def canon(s):
    out = []
    for c in norm(s):
        p = out[-1] if out else ""
        out.append("ー" if (c == "ー" or (c == "ウ" and (p in U_DAN or p in O_DAN))
                            or (c == "オ" and p in O_DAN) or (c == "イ" and p in E_DAN)) else c)
    return "".join(out)


def hit(reading, hyp):
    """True if the target reading (accent marks ignored) is heard in the transcript."""
    return (canon(reading) in canon(hyp)) or (norm(reading) in norm(hyp))


class Checker:
    """kana-whisper wrapper; `heard(audio_float32, sr, reading) -> (bool, hyp)`."""
    def __init__(self, device=0, model="sbintuitions/kana-whisper"):
        import torch
        from transformers import pipeline
        self.asr = pipeline("automatic-speech-recognition", model=model, device=device,
                            torch_dtype=torch.float16,
                            generate_kwargs={"language": "ja", "task": "transcribe"})

    def transcribe(self, audio, sr):
        # Resample here, not in the pipeline: transformers' raw-input path (torchaudio) returns distorted
        # transcripts at 48 kHz (garbage) and 44.1 kHz (misheard), while 24 and 16 kHz are fine. librosa
        # to 16 kHz gives the file-path result at every rate.
        import numpy as np, librosa
        x = np.asarray(audio, dtype="float32")
        if int(sr) != 16000:
            x = librosa.resample(x, orig_sr=int(sr), target_sr=16000)
        return str(self.asr({"raw": x, "sampling_rate": 16000})["text"])

    def heard(self, audio, sr, reading):
        hyp = self.transcribe(audio, sr)
        return hit(reading, hyp), hyp
