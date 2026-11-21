#!/usr/bin/env python3
"""Synthesize with Irodori-TTS-500M-v3 or a tag-token fine-tune of it, from a reference voice.

Library: `load_runtime(checkpoint)` then `synthesize(runtime, text, ref, seed)`.
CLI, one utterance:
    python synthesis.py --text "...<PHON_START>カラ'イ<PHON_END>..." --ref-latent REF.pt \
        [--checkpoint CKPT] [--seed 0] [--guard] --out OUT.wav

Checkpoints: none (the pinned base release), a .pt written by the upstream trainer from the
tag-token init, or the round-3 checkpoint the paper reports. That last one was written by an
earlier trainer; `_make_compatible` explains the two things it carries that the upstream model
does not.

--guard runs the project's degeneracy guard (`scripts/eval/degenerate.py`, on the repository's
code branch): a degenerate take is redrawn with a bumped seed and the first valid take is kept.
"""
from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent
# The pinned upstream checkout (setup.sh) and the project's evaluation helpers on the code branch.
sys.path[:0] = [str(HERE / "upstream"), str(HERE.parent / "scripts" / "eval")]

from irodori_tts import inference_runtime
from irodori_tts.config import ModelConfig
from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest

from prepare_tag_model import (
    BASE_REPO,
    BASE_REVISION,
    BASE_TEXT_VOCAB_SIZE,
    BASE_TOKENIZER_REVISION,
    TAG_TOKENS,
    build_tag_tokenizer,
)

KANA_WHISPER = "sbintuitions/kana-whisper"


def tag_tokenizer_dir() -> Path:
    """The tag tokenizer, built once into the user cache from the pinned base tokenizer."""
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    path = cache / "irodori-pron-tags" / f"tokenizer-{BASE_TOKENIZER_REVISION[:12]}"
    if not (path / "tokenizer.json").exists():
        build_tag_tokenizer(path)
    return path


def _make_compatible(state: dict, model_cfg: dict) -> tuple[dict, dict]:
    """Return (state, model_cfg) the upstream model loads, without changing what it computes.

    * A tag-token checkpoint names its tokenizer by a path relative to where it was trained; the
      tokenizer is rebuilt from the pinned base instead (same ids, see prepare_tag_model.py).
    * The round-3 checkpoint's config holds `pronunciation_*` keys of a conditioning branch that was
      switched off (`pronunciation_conditioning: false`, no weights), and its state holds
      `meanflow_cond` (two linear layers), which no gradient reached in RF training: its tensors
      are bit-identical at steps 500 and 2,000 and its output layer is all zeros. Both are
      dropped; anything else the upstream model does not know is an error.
    """
    known = {f.name for f in fields(ModelConfig)}
    extra = {k: v for k, v in model_cfg.items() if k not in known}
    if extra:
        if any(not k.startswith("pronunciation_") for k in extra) or extra.get(
            "pronunciation_conditioning"
        ):
            raise ValueError(f"checkpoint config has keys this model does not support: {sorted(extra)}")
        model_cfg = {k: v for k, v in model_cfg.items() if k in known}
    if any(k.startswith("meanflow_cond.") for k in state):
        if state["meanflow_cond.2.weight"].abs().max().item() != 0.0:
            raise ValueError("meanflow_cond has a non-zero output layer; dropping it would change outputs")
        state = {k: v for k, v in state.items() if not k.startswith("meanflow_cond.")}
    if model_cfg["text_vocab_size"] == BASE_TEXT_VOCAB_SIZE + len(TAG_TOKENS):
        model_cfg = {**model_cfg, "text_tokenizer_repo": str(tag_tokenizer_dir())}
    return state, model_cfg


@contextmanager
def _compatible_checkpoint_loading():
    # InferenceRuntime.from_key reads the checkpoint through this module-level function and offers
    # no hook of its own, so it is wrapped for the duration of one load.
    upstream_load = inference_runtime._load_checkpoint_for_inference

    def load(path):
        state, model_cfg, train_cfg = upstream_load(path)
        return (*_make_compatible(state, model_cfg), train_cfg)

    inference_runtime._load_checkpoint_for_inference = load
    try:
        yield
    finally:
        inference_runtime._load_checkpoint_for_inference = upstream_load


def load_runtime(checkpoint: str | Path | None, *, watermark: bool = True, device: str = "cuda") -> InferenceRuntime:
    """Model and codec in fp32 on `device` (the upstream infer.py defaults).

    watermark=False skips upstream's SilentCipher watermark; the paper's evaluation clips were made
    without it, so reproducing them needs it off.
    """
    if checkpoint is None:
        checkpoint = hf_hub_download(BASE_REPO, "model.safetensors", revision=BASE_REVISION)
    with _compatible_checkpoint_loading():
        runtime = InferenceRuntime.from_key(
            RuntimeKey(checkpoint=str(checkpoint), model_device=device, codec_device=device)
        )
    if not watermark:
        runtime.watermarker.model = None
    return runtime


def synthesize(
    runtime: InferenceRuntime,
    text: str,
    *,
    seed: int,
    ref_wav: str | Path | None = None,
    ref_latent: str | Path | None = None,
) -> tuple[np.ndarray, int]:
    """One take at the upstream sampling defaults (40 steps, CFG text 3 / speaker 5), length from
    the duration predictor. The reference is a waveform or a saved DACVAE latent (T x 32)."""
    if (ref_wav is None) == (ref_latent is None):
        raise ValueError("give exactly one of ref_wav and ref_latent")
    result = runtime.synthesize(
        SamplingRequest(
            text=text,
            ref_wav=None if ref_wav is None else str(ref_wav),
            ref_latent=None if ref_latent is None else str(ref_latent),
            seed=int(seed),
        )
    )
    return result.audio.float().numpy().reshape(-1), result.sample_rate


def kana_whisper_transcriber():
    """kana-whisper as (audio, sr) -> katakana text; audio is resampled by `to_asr_input` and a
    take whisper cannot read comes back as "" instead of raising (`safe_asr`)."""
    from asr_audio import to_asr_input
    from degenerate import safe_asr
    from transformers import pipeline

    asr = pipeline(
        "automatic-speech-recognition",
        model=KANA_WHISPER,
        device=0,
        torch_dtype=torch.float16,
        generate_kwargs={"language": "ja", "task": "transcribe"},
    )
    return lambda audio, sr: str(safe_asr(lambda a, r: asr(to_asr_input(a, r))["text"], audio, sr))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--text", required=True)
    ref = parser.add_mutually_exclusive_group(required=True)
    ref.add_argument("--ref-latent", help="saved DACVAE latent (.pt, T x 32) of the reference voice")
    ref.add_argument("--ref-wav", help="reference voice as audio")
    parser.add_argument("--checkpoint", help="fine-tuned checkpoint; omit for the base release")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--guard", action="store_true", help="redraw degenerate takes (kana-whisper)")
    parser.add_argument("--no-watermark", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    runtime = load_runtime(args.checkpoint, watermark=not args.no_watermark)

    def take(seed: int) -> tuple[np.ndarray, int]:
        return synthesize(runtime, args.text, seed=seed, ref_wav=args.ref_wav, ref_latent=args.ref_latent)

    if args.guard:
        from degenerate import synth_valid

        audio, sr, info = synth_valid(take, args.seed, asr=kana_whisper_transcriber())
        print(f"guard: {info}")
    else:
        audio, sr = take(args.seed)
    sf.write(args.out, audio, sr)
    print(f"wrote {args.out} ({len(audio) / sr:.2f} s)")


if __name__ == "__main__":
    main()
