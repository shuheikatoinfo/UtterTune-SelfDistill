# Self-distilled pronunciation and accent control

Code, pre-registration and results for

> Shuhei Kato. *Self-Distilled Pronunciation and Accent Control for Neural Text-to-Speech.* arXiv:2609.17234.

A frozen TTS backbone reads a sentence containing a common word it already says correctly; its own output becomes
the teacher for the same sentence with that word written as a tagged, accented reading
(`<PHON_START>カラ'イ<PHON_END>`). Training on these pairs adds reading and pitch-accent control to the backbone
without any recorded speech. The paper applies one recipe to four backbones: Sarashina2.2-TTS, CosyVoice 2,
Irodori-TTS-500M-v3 and T5Gemma-TTS.

## Released artifacts

| What | Where |
|---|---|
| Sarashina2.2-TTS adapter | [shuheikatoinfo/Sarashina2.2-TTS-self-distilled-pron-accent](https://huggingface.co/shuheikatoinfo/Sarashina2.2-TTS-self-distilled-pron-accent) |
| CosyVoice 2 adapter | [shuheikatoinfo/CosyVoice2-0.5B-self-distilled-pron-accent](https://huggingface.co/shuheikatoinfo/CosyVoice2-0.5B-self-distilled-pron-accent) |
| Irodori-TTS-500M-v3 model | [shuheikatoinfo/Irodori-TTS-500M-v3-self-distilled-pron-accent](https://huggingface.co/shuheikatoinfo/Irodori-TTS-500M-v3-self-distilled-pron-accent) |
| T5Gemma-TTS adapter | [shuheikatoinfo/T5Gemma-TTS-2b-2b-self-distilled-pron-accent](https://huggingface.co/shuheikatoinfo/T5Gemma-TTS-2b-2b-self-distilled-pron-accent) |
| Text set, teacher sets, speaker prompts | [shuheikatoinfo/self-distilled-pron-accent-data](https://huggingface.co/datasets/shuheikatoinfo/self-distilled-pron-accent-data) |
| Pre-registration and its dated record | [`preregistration/`](preregistration/) (this repository's history) |
| Evaluation inputs, raters' responses, results | [`results/round3/`](results/round3/), [`data/`](data/), [`preregistration/items/`](preregistration/items/) |

The difficult-50, katakana-50 and easy-50 sentences come from [UtterTune](https://github.com/shuheikatoinfo/UtterTune)
(`data/eval/`). The objective set is drawn from the
[Joyo Kanji Yomi Benchmark](https://huggingface.co/datasets/sbintuitions/joyo-kanji-yomi-benchmark) (MIT).

## Layout

```
preregistration/   the registration (PREREGISTRATION_v2.md opens with a summary of the round the paper reports),
                   item sets, screening and catch trials, voice assignments
scripts/           teacher generation, training, synthesis with the degeneration guard, evaluation, analysis,
                   and the rater web site; paths as the registration cites them
irodori/           Irodori-TTS training and inference on top of the public Irodori-TTS code
data/              text-set inventory and the evaluation inputs
results/round3/    reports of record, raters' responses (pseudonymous IDs), kit keys, per-item retry record
```

## Reproduce the reported numbers

No GPU is needed; run from the repository root.

```bash
# listening test (Fig. 1, Table 2 left)
python scripts/analysis/analyze_listening.py --round2 \
  --responses results/round3/responses/responses_panel_r3_final_20260921_1009.json \
  --kits results/round3/kits/listening --out listening.json
# naturalness (Table 2 right)
python scripts/analysis/analyze_naturalness.py --exclude-speeders \
  --responses results/round3/responses/responses_nat_round3_final_20260921_1014.json \
  --items preregistration/items/naturalness_ab_easy50_round3.json \
  --clip-durations results/round3/kits/naturalness_clip_durations.json --out naturalness.json
# reading accuracy contrasts (Table 1)
python scripts/eval/obj319_contrasts_v2.py --results results/round3/reading_obj319_round3_ut_corrected_inputs.json
```

The first two reproduce `results/round3/listening_report_round3_4raters_20260921_1009.json` and
`naturalness_report_round3_final_10k.json` exactly.

## Train

Set the environment variables that `scripts/paths.py` names for the parts you use (`COSYVOICE_DIR`, `T5GEMMA_DIR`,
`SARASHINA_DIR`, `DATASET_DIR` for the dataset, and `DATA` for the evaluation scripts). `D` below is a local copy of
the dataset.

```bash
# Sarashina2.2-TTS (round 3: 20,000 steps)
python scripts/sarashina/train_lora_sarashina.py --teachers $D/teachers/sarashina/teachers_P4_not_slow.jsonl \
  --model-dir <sbintuitions/sarashina2.2-tts> --out out/sarashina --steps 20000 --alpha 64 --save-every 4000
# CosyVoice 2 (round 3: 1,000 steps)
python scripts/cv/train_lora_cv2.py --manifest $D/teachers/cosyvoice2/manifest_filtered.tsv --data-root $D \
  --out out/cv2 --steps 1000
# T5Gemma-TTS (round 3: 600 steps)
python scripts/t5gemma/materialize_dataset.py --teachers $D/teachers/t5gemma/teachers_filtered.jsonl --out out/t5g_data
DATASET_DIR=out/t5g_data EXP_ROOT=out/t5g NUM_STEPS=600 bash scripts/t5gemma/blockA_readlora_train.sh
# Irodori-TTS-500M-v3: see irodori/README.md
```

The routed evaluation inputs are shipped as files under `data/derived/`. The drivers under `scripts/eval/`
(`round3_*`, `sarahyp_P4_*`, `obj319_inputfix_resynth.sh`) are the ones that produced the paper's numbers, including the
cloud-GPU runs.

## Licence

Code: MIT ([`LICENSE`](LICENSE)). Weights and data carry the licences of the backbones they derive from; see each
Hugging Face card. Sarashina2.2-TTS parts are Built with Sarashina and are under the Sarashina Model NonCommercial
License Agreement; T5Gemma-TTS parts are under the Gemma Terms of Use and CC BY-NC 4.0.

## Citation

```bibtex
@misc{kato2026selfdistilled,
  title         = {Self-Distilled Pronunciation and Accent Control for Neural Text-to-Speech},
  author        = {Kato, Shuhei},
  year          = {2026},
  eprint        = {2609.17234},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SD}
}
```
