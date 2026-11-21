#!/usr/bin/env bash
# Fine-tune Irodori-TTS-500M-v3 with the two pronunciation tags, round-3 settings (train_round3.yaml).
#   bash irodori/train.sh WORK_DIR MANIFEST [more train.py options, e.g. --output-dir DIR]
# WORK_DIR holds what prepare_tag_model.py wrote; checkpoints go to WORK_DIR/checkpoints unless
# --output-dir is given. The upstream trainer caches a manifest index next to MANIFEST.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
UPSTREAM=$HERE/upstream
WORK=$(cd "$1" && pwd)
MANIFEST=$(realpath "$2")
shift 2
cd "$WORK"  # train_round3.yaml names the tokenizer relative to the work directory
exec "$UPSTREAM/.venv/bin/python" "$UPSTREAM/train.py" \
  --config "$HERE/train_round3.yaml" \
  --manifest "$MANIFEST" \
  --init-checkpoint "$WORK/init_pron_tags.pt" \
  --output-dir "$WORK/checkpoints" \
  "$@"
