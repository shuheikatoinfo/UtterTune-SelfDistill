# Irodori-TTS: pronunciation-tag fine-tune (round 3)

Training and inference code for the paper's round-3 Irodori model, built on the public
[Irodori-TTS](https://github.com/Aratako/Irodori-TTS) code (MIT) at a pinned commit. The model is
Irodori-TTS-500M-v3 fully fine-tuned so that `<PHON_START>reading'<PHON_END>` in the input text sets
the reading and accent of the wrapped span.

## Dependency

| What | Pinned to |
|---|---|
| Irodori-TTS code | `eaf74d6a19138f743acb5b71a445fd25a57db987` (2026-06-04), cloned by `setup.sh` into `irodori/upstream/` |
| Base weights | `Aratako/Irodori-TTS-500M-v3` revision `236c1e56591279fc24e3c1bf6609fc06e48dde28`, `model.safetensors` SHA-256 `c4b8e7e9…acbd54` (checked by `prepare_tag_model.py`) |
| Text tokenizer | `llm-jp/llm-jp-3-150m` revision `b112feef602fff752e4dac4c30af6a2c2fa41c7a`, plus the two tag tokens |
| Codec | `Aratako/Semantic-DACVAE-Japanese-32dim` (fetched by the upstream code) |

```bash
bash irodori/setup.sh     # clone + checkout + apply upstream.patch + `uv sync --frozen --extra cu128`
PY=irodori/upstream/.venv/bin/python
```

## What this directory adds to the pinned code

| File | Role |
|---|---|
| `upstream.patch` | dataset: a manifest row may name its reference latent (`ref_latent_path`); see below |
| `prepare_tag_model.py` | base tokenizer + `<PHON_START>` / `<PHON_END>` (special tokens, ids 99,574 and 99,575); base weights with the text embedding grown by two N(0, 1) rows, written as the trainer's init checkpoint |
| `train_round3.yaml` | the round-3 settings for the upstream `train.py` |
| `train.sh` | runs the upstream trainer with that config from a work directory |
| `synthesis.py` | loads the base, a fine-tuned checkpoint or the paper's round-3 checkpoint; synthesizes one utterance from a reference latent or waveform; optional degeneracy guard |
| `run_blockB1.py` | the item-set CLI the evaluation scripts call (`scripts/eval/round3_*.sh`) |

Everything else (model, RF loss, Muon, WSD schedule, fixed-length targets, duration predictor,
sampler, codec) is the pinned upstream code, unchanged.

## Train

The manifest is upstream's JSONL (`text`, `latent_path`, `num_frames`, `speaker_id`), one row per
teacher, with `text` carrying the tags, plus `ref_latent_path`: the reference latent the teacher
was synthesized from. Upstream conditions on another utterance of the same `speaker_id`; with the
patch a row's own `ref_latent_path` is used when present.

```bash
$PY irodori/prepare_tag_model.py --work-dir WORK      # WORK/tokenizer_pron_tags, WORK/init_pron_tags.pt
bash irodori/train.sh WORK MANIFEST                    # WORK/checkpoints/checkpoint_0002000.pt, ...
bash irodori/train.sh WORK ROOT/teachers/irodori/train_filtered.jsonl --data-root ROOT   # released dataset
```

Round-3 settings (`train_round3.yaml`): full fine-tune, no LoRA; Muon (AdamW for embeddings and
1-D parameters), learning rate 1e-5, weight decay 0.01, WSD (100 warm-up steps, then constant),
batch 32 × 2 accumulation, bf16, gradient clip 1.0, RF loss `echo`, fixed 224-frame targets with
full mask, duration loss weight 0.1, no text or speaker condition dropout, seed 0, 2,000 steps.
The upstream trainer writes a manifest index cache next to the manifest.

## Synthesize

```bash
$PY irodori/synthesis.py --checkpoint CKPT --ref-latent REF.pt --seed 0 --guard \
    --text "このカレーはとても<PHON_START>カラ'イ<PHON_END>です。" --out out.wav
```

Without `--checkpoint` the base release is used. The tag tokenizer is rebuilt from the pinned base
tokenizer into `~/.cache/irodori-pron-tags/` on first use. `--guard` and `run_blockB1.py` need
`scripts/eval/degenerate.py` and `scripts/eval/asr_audio.py` of this repository.

## Evaluation CLI

```bash
TEST_DIR=D TEST_SET=S ITEM_SPK_JSON=A ITEM_SPK_CELL=C EVAL_SPEAKERS_JSON=V BLOCKB1_OUT=O \
  $PY irodori/run_blockB1.py --arms ours --phon-checkpoint CKPT --skip-eval
```

Arms `noedit` (base, `test_S_vanilla.txt`), `kana` (base, `test_S_yomi.txt`), `ours` (`CKPT`,
`test_S_lora.txt`, `/` phrase marks removed). Item `i` uses its assigned voice and seed `5000 + i`;
a degenerate take is redrawn (seed + 1000 per redraw, up to 6 takes, first valid take kept).
Writes `O/<arm>/NN.wav`, `O/guard_takes.json`, `O/blockB1_irodori.json`. The line files are used
as given: an item set whose tags still need routing must be routed first.

## Relation to the paper's model

This code is written on the public Irodori-TTS. Retrained with it in the round-3 setting, the model reaches the
paper's checkpoint on the 144 validation items (kana CER 0.061 against 0.057, UTMOSv2 2.77 against 2.80, both read
with this code).

