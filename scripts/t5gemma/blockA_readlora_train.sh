#!/usr/bin/env bash
# STEP C - reading-only self-distill LoRA on the T5Gemma-TTS backbone.
# Copy of examples/training/t5gemma_2b-2b-ft-lora.sh with the reading-only recipe overrides:
#   - load the real pretrained ckpt (pretrained/.../ckpt/pretrained.pth)
#   - our teacher dataset (surface kanji text -> base's kana-reading acoustic tokens)
#   - neighbor_prompt_prob 0, no_loss_on_prefix 0  (prompt-free; loss over the whole acoustic seq)
#   - LoRA on the whole backbone; custom audio heads stay frozen.
# Env overrides: DATASET_DIR, EXP_ROOT, NUM_STEPS, BATCH_SIZE, LR, SAVE_EVERY, VAL_EVERY.
set -euo pipefail

# T5GEMMA_DIR is read directly (no scripts/paths.sh): this script is also shipped on its own to training pods.
REPO="${T5GEMMA_DIR:?set T5GEMMA_DIR to the T5Gemma-TTS checkout}"
T5GEMMA_MODEL_NAME=google/t5gemma-2b-2b-ul2
XCODEC2_MODEL_NAME=NandemoGHS/Anime-XCodec2-44.1kHz-v2
PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-${REPO}/pretrained/T5Gemma-TTS-2b-2b/ckpt/pretrained.pth}"

DATASET_DIR="${DATASET_DIR:?set DATASET_DIR}"
EXP_ROOT="${EXP_ROOT:?set EXP_ROOT}"
NUM_GPUS="${NUM_GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_STEPS="${NUM_STEPS:-400}"
LR="${LR:-0.035}"
WARMUP_FRAC="${WARMUP_FRAC:-0.05}"
# Validation is DISABLED by default. trainer.py:898 all_reduces `sum_losses`, which stays an
# int when the valid loader yields no batches -> TypeError and the run dies mid-training.
# The v2 clean-room run hit this too (7 tracebacks, last step 200/400), so its adapter came
# from a partial run. Nothing here consumes the validation score: the checkpoint is never
# selected on it (selecting would be measurement-integrity-relevant), so skipping it costs
# nothing and removes the crash. [2026-09-01]
# Must be LARGER than NUM_STEPS (so validation never fires) yet a strict DIVISOR of
# --inference_every_n_steps=100000000 and strictly smaller than it -- trainer.py:129 asserts
# inference_every > val_every AND inference_every %% val_every == 0. 100000 satisfies both.
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
DATASET_DIRS="['${DATASET_DIR}']"
MANIFEST_NAMES="['manifest_final']"

echo "[Info] LoRA reading-only train: data=${DATASET_DIR} exp=${EXP_ROOT} steps=${NUM_STEPS}"
cd "${REPO}"
TORCHRUN="${REPO}/.venv_t5gemma/bin/torchrun"   # use the venv interpreter, NOT pyenv's torchrun
# --local_wandb: run W&B in offline mode. Without it main.py aborts with
# "No API key configured" before the first step (seen 2026-08-29 on the clean-room run).
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
