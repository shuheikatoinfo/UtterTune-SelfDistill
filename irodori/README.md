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
`scripts/eval/degenerate.py` and `scripts/eval/asr_audio.py` from the repository's code branch.

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

## Reproduction of the round-3 model

Retrained with this code on the round-3 teacher manifest (13,901 rows), one run, and read on the
round-3 validation items (tag arm, 105 selection + 39 side-effect items, the same voices, seeds and
guard) with `run_blockB1.py`. CER: mean sentence kana CER from the guard's kana-whisper transcripts,
as the checkpoint rule computes it. UTMOSv2: mean over the 144 clips; UTMOSv2 picks random audio
segments, so it is given as mean ± s.d. over three fixed seeds (the recorded value is one unseeded
draw).

| Step 2,000 | CER (144) | CER sel. 105 | CER side 39 | redraws | UTMOSv2 |
|---|---|---|---|---|---|
| Paper's checkpoint, as recorded | 0.058 | 0.063 | 0.044 | 1 | 2.806 |
| Paper's checkpoint, its recorded clips re-scored | 0.058 | 0.063 | 0.044 | 1 | 2.791 ± 0.011 |
| Paper's checkpoint, this inference code | 0.057 | 0.063 | 0.043 | 1 | 2.800 ± 0.005 |
| Retrained with this code | 0.061 | 0.068 | 0.043 | 1 | 2.769 ± 0.016 |
| Retrained, unpatched upstream reference sampling | 0.051 | 0.057 | 0.035 | 0 | 2.889 ± 0.029 |

The retrained model is within the checkpoint rule's tie margin (0.005) of the paper's checkpoint
read by the same code. One training run per row: run-to-run spread is not measured.

Why the patch: with the row's own reference the training loss follows the paper's run log
(smoothed loss at steps 100 / 400 / 800: 1.160 / 0.963 / 0.899 against 1.159 / 0.968 / 0.902);
with upstream's same-speaker sampling it stays about 0.05 higher (1.197 / 1.014 / 0.957). That run
scores better on validation, but it is not the recipe the paper's model was trained with.

What is not reproduced:

* **Target length of the tag arm.** The paper's runs did not use the length this code takes from
  the duration predictor: on the tag arm, 10 of 144 clips have the same length, and the paper's
  clips are 2.5% shorter on average. Forcing the paper's length on three items reproduces their
  waveforms (correlation 0.97 to 0.99), and on the no-edit arm the predicted lengths equal the
  paper's on all eight items checked; so the text and reference reach the model as they did.
  Where the paper's tag-arm length came from could not be determined from the configuration,
  the logs and the public code.
* **Exact weights.** The two new embedding rows are drawn from N(0, 1) with their own seed (the
  original took them from the model's initialisation), so data order and initial values differ.
* **Routing.** The earlier loader routed some line files at load time (reviewed accents on the
  difficult and easy sets). This CLI reads line files as given, except for removing `/`.

The paper's checkpoint also holds `meanflow_cond`, a two-layer module that public Irodori-TTS
does not define (its later MeanFlow support is a different module, `delta_cond_module`) and that
no gradient reached in RF training (bit-identical at steps 500 and 2,000, all-zero output layer).
A model trained with this code has the same parameters without it (637 tensors, 512,050,465
parameters); `synthesis.py` drops it when loading the paper's checkpoint.
