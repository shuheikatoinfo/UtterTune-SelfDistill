#!/usr/bin/env bash
# Reading and accent self-distillation LoRA on T5Gemma-TTS-2b-2b.
# Derived from examples/training/t5gemma_2b-2b-ft-lora.sh of T5Gemma-TTS (https://github.com/Aratako/T5Gemma-TTS),
# used under its licence:
#
# MIT License
#
# Copyright (c) 2025 Aratako
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Changes from the example: the released pretrained checkpoint is loaded; the data is the materialized teacher set
# (tagged carrier -> the base's acoustic tokens for the plain carrier; scripts/t5gemma/materialize_dataset.py); no prompt
# (neighbor_prompt_prob 0) and loss over the whole acoustic sequence (no_loss_on_prefix 0); LoRA on the whole backbone
# while the audio heads stay frozen; validation off.
# Needs T5GEMMA_DIR (with steps/trainer.py patched by scripts/t5gemma/trainer.patch), T5G_DATA_DIR (the materialized
# set) and EXP_ROOT. Optional: NUM_STEPS, SAVE_EVERY, BATCH_SIZE, LR, WARMUP_FRAC, MAX_NUM_TOKENS, GRAD_ACCUM.
set -euo pipefail

# T5GEMMA_DIR is read directly (no scripts/paths.sh): this script also runs on its own on training pods.
REPO="${T5GEMMA_DIR:?set T5GEMMA_DIR to the T5Gemma-TTS checkout}"
T5GEMMA_MODEL_NAME=google/t5gemma-2b-2b-ul2
XCODEC2_MODEL_NAME=NandemoGHS/Anime-XCodec2-44.1kHz-v2
PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-${REPO}/pretrained/T5Gemma-TTS-2b-2b/ckpt/pretrained.pth}"

T5G_DATA_DIR="${T5G_DATA_DIR:?set T5G_DATA_DIR to the materialized teacher set}"
EXP_ROOT="${EXP_ROOT:?set EXP_ROOT}"
NUM_GPUS="${NUM_GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_STEPS="${NUM_STEPS:-400}"
LR="${LR:-0.035}"
WARMUP_FRAC="${WARMUP_FRAC:-0.05}"
# Validation is off: no checkpoint is selected on the validation loss, and the small validation split can yield no
# batch, which the unpatched trainer does not survive. VAL_EVERY must exceed NUM_STEPS (so validation never runs) and
# be a proper divisor of --inference_every_n_steps (trainer.py asserts both); 100000 satisfies both.
VAL_EVERY="${VAL_EVERY:-100000}"
SAVE_EVERY="${SAVE_EVERY:-200}"
PRINT_EVERY="${PRINT_EVERY:-10}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"

X_SEP_TOKEN=255999
N_SPECIAL=5
AUDIO_VOCAB_SIZE=65536
MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-20000}"
VAL_MAX_NUM_TOKENS="${VAL_MAX_NUM_TOKENS:-5000}"

mkdir -p "${EXP_ROOT}"
DATASET_DIRS="['${T5G_DATA_DIR}']"
MANIFEST_NAMES="['manifest_final']"

echo "[Info] LoRA reading-only train: data=${T5G_DATA_DIR} exp=${EXP_ROOT} steps=${NUM_STEPS}"
cd "${REPO}"
TORCHRUN="${REPO}/.venv_t5gemma/bin/torchrun"   # the checkout's environment, not a torchrun on PATH
# --local_wandb 1: W&B offline; without it main.py stops when no W&B API key is configured.
"${TORCHRUN}" --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" "${REPO}/main.py" \
  --local_wandb 1 \
  --model_arch t5gemma \
  --t5gemma_model_name "${T5GEMMA_MODEL_NAME}" \
  --text_input_type text \
  --text_tokenizer_name "${T5GEMMA_MODEL_NAME}" \
  --audio_tokenizer xcodec2 \
  --xcodec2_model_name "${XCODEC2_MODEL_NAME}" \
  --audio_vocab_size "${AUDIO_VOCAB_SIZE}" \
  --progress_scale 2000 \
  --neighbor_prompt_prob 0 \
  --n_special "${N_SPECIAL}" \
  --x_sep_token "${X_SEP_TOKEN}" \
  --no_loss_on_prefix 0 \
  --min_prompt_len 0.5 \
  --audio_max_length 40 \
  --audio_min_length 0.2 \
  --text_max_length 500 \
  --encodec_sr 50 \
  --dataset_dir "${DATASET_DIRS}" \
  --manifest_name "${MANIFEST_NAMES}" \
  --encodec_folder_name xcodec2_1cb \
  --audio_folder_name audio \
  --target_time_stretch_prob 0 \
  --time_stretch_prob 0 \
  --batch_size "${BATCH_SIZE}" \
  --num_workers 2 \
  --max_num_tokens "${MAX_NUM_TOKENS}" \
  --val_max_num_tokens "${VAL_MAX_NUM_TOKENS}" \
  --num_steps "${NUM_STEPS}" \
  --lr "${LR}" \
  --warmup_fraction "${WARMUP_FRAC}" \
  --precision bfloat16 \
  --print_every_n_steps "${PRINT_EVERY}" \
  --val_every_n_steps "${VAL_EVERY}" \
  --inference_every_n_steps 100000000 \
  --save_every_n_steps "${SAVE_EVERY}" \
  --tb_write_every_n_steps 1 \
  --seed 1 \
  --exp_dir "${EXP_ROOT}" \
  --drop_long 1 \
  --pad_x 0 \
  --text_pad_token 0 \
  --num_buckets 5 \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --optimizer_name "ScaledAdam" \
  --pseudo_epoch_size 1000 \
  --reduce_lr_start_step 100000 \
  --reduce_lr_start_epoch 100000 \
  --clipping_update_period 200 \
  --validation_sample_cap 30000 \
  --t5_gradient_checkpointing 1 \
  --prune_text_modules 2 \
  --compile 0 \
  --attn_implementation sdpa \
  --ddp_find_unused_parameters 0 \
  --load_model_from "${PRETRAINED_MODEL_PATH}" \
  --use_lora 1 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
