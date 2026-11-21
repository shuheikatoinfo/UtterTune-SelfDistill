#!/usr/bin/env python3
"""Build the two inputs a pronunciation-tag run of Irodori-TTS starts from, in one work directory.

    <work>/tokenizer_pron_tags/   the base text tokenizer with <PHON_START> and <PHON_END> added
    <work>/init_pron_tags.pt      the base weights with the text embedding grown by those two rows

With these two in place the upstream trainer fine-tunes unchanged: it initialises from a
checkpoint only when checkpoint and config agree on the vocabulary, so the vocabulary is grown
here, once, instead of inside the trainer.

    python prepare_tag_model.py --work-dir WORK [--seed 0]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer

BASE_REPO = "Aratako/Irodori-TTS-500M-v3"
BASE_REVISION = "236c1e56591279fc24e3c1bf6609fc06e48dde28"
BASE_SHA256 = "c4b8e7e982697664f829b7fb6bea307a25bd7ee013ad0d6114efc3e326acbd54"
# The base names this tokenizer in its own config; pinned so a later upload cannot shift the ids.
BASE_TOKENIZER_REPO = "llm-jp/llm-jp-3-150m"
BASE_TOKENIZER_REVISION = "b112feef602fff752e4dac4c30af6a2c2fa41c7a"
BASE_TEXT_VOCAB_SIZE = 99574
TAG_TOKENS = ("<PHON_START>", "<PHON_END>")
TOKENIZER_DIRNAME = "tokenizer_pron_tags"
INIT_CHECKPOINT_NAME = "init_pron_tags.pt"
# Keys the base stores beside the model config; the trainer reads them from its own config.
_INFERENCE_ONLY_KEYS = {"max_text_len", "max_caption_len", "fixed_target_latent_steps"}


def download_base_weights() -> Path:
    """Fetch the pinned base checkpoint and refuse it unless its SHA-256 is the published one."""
    path = Path(hf_hub_download(BASE_REPO, "model.safetensors", revision=BASE_REVISION))
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    if digest.hexdigest() != BASE_SHA256:
        raise RuntimeError(f"{path}: sha256 {digest.hexdigest()} != {BASE_SHA256}")
    return path


def build_tag_tokenizer(out_dir: Path) -> None:
    """Save the base tokenizer plus the two tag tokens (ids 99,574 and 99,575) to out_dir.

    The tags are added as SPECIAL tokens. As ordinary added tokens they go through this
    tokenizer's normaliser, and `<PHON_END>` then fails to match inside a sentence and is spelt
    out as `<`, `PH`, `ON`, `_`, `END`, `>`. Special tokens match as one id each; the segment that
    follows a tag gets a leading `▁`, as it does at the start of a text.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_TOKENIZER_REPO, revision=BASE_TOKENIZER_REVISION, use_fast=True
    )
    tokenizer.add_special_tokens({"additional_special_tokens": list(TAG_TOKENS)})
    ids = tokenizer.convert_tokens_to_ids(list(TAG_TOKENS))
    if ids != [BASE_TEXT_VOCAB_SIZE + i for i in range(len(TAG_TOKENS))]:
        raise RuntimeError(f"tag tokens got ids {ids}, expected the rows after {BASE_TEXT_VOCAB_SIZE}")
    tokenizer.save_pretrained(out_dir)
    # The trainer checks the text encoder width against the tokenizer repo's model config.
    AutoConfig.from_pretrained(
        BASE_TOKENIZER_REPO, revision=BASE_TOKENIZER_REVISION
    ).save_pretrained(out_dir)


def build_init_checkpoint(base_path: Path, out_path: Path, tokenizer_ref: str, seed: int) -> None:
    """Write the base weights with two new text-embedding rows, in the trainer's .pt layout.

    The new rows are drawn from N(0, 1), the initialisation `torch.nn.Embedding` gives every row of
    a freshly built text encoder; all other weights are the base's, unchanged.
    """
    with safe_open(str(base_path), framework="pt", device="cpu") as handle:
        flat_config = json.loads(handle.metadata()["config_json"])
    model_config = {k: v for k, v in flat_config.items() if k not in _INFERENCE_ONLY_KEYS}
    state = load_file(str(base_path), device="cpu")
    key = "text_encoder.text_embedding.weight"
    embedding = state[key]
    if embedding.shape[0] != BASE_TEXT_VOCAB_SIZE:
        raise RuntimeError(f"{key} has {embedding.shape[0]} rows, expected {BASE_TEXT_VOCAB_SIZE}")
    generator = torch.Generator().manual_seed(seed)
    new_rows = torch.randn((len(TAG_TOKENS), embedding.shape[1]), generator=generator)
    state[key] = torch.cat([embedding, new_rows.to(embedding.dtype)], dim=0)
    model_config["text_vocab_size"] = int(state[key].shape[0])
    model_config["text_tokenizer_repo"] = tokenizer_ref
    torch.save({"model": state, "model_config": model_config}, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0, help="seed for the two new embedding rows")
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    build_tag_tokenizer(args.work_dir / TOKENIZER_DIRNAME)
    # Relative on purpose: the trainer runs inside the work directory, and inference passes the
    # tokenizer directory explicitly, so no machine path is written into a checkpoint.
    build_init_checkpoint(
        download_base_weights(), args.work_dir / INIT_CHECKPOINT_NAME, TOKENIZER_DIRNAME, args.seed
    )
    print(f"wrote {args.work_dir / TOKENIZER_DIRNAME} and {args.work_dir / INIT_CHECKPOINT_NAME}")


if __name__ == "__main__":
    main()
