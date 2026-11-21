#!/usr/bin/env bash
# Fine-tune Irodori-TTS-500M-v3 with the two pronunciation tags, round-3 settings (train_round3.yaml).
#   bash irodori/train.sh WORK_DIR MANIFEST [--data-root ROOT] [more train.py options, e.g. --output-dir DIR]
# WORK_DIR holds what prepare_tag_model.py wrote; checkpoints go to WORK_DIR/checkpoints unless
# --output-dir is given. --data-root: the manifest's relative latent paths are relative to ROOT (the
# released dataset's layout); a copy with those paths resolved is written to WORK_DIR and trained on.
# The upstream trainer caches a manifest index next to the manifest it reads.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
UPSTREAM=$HERE/upstream
PY=$UPSTREAM/.venv/bin/python
WORK=$(cd "$1" && pwd)
MANIFEST=$(realpath "$2")
shift 2
if [ "${1:-}" = --data-root ]; then
  RESOLVED=$WORK/$(basename "$MANIFEST")
  "$PY" - "$MANIFEST" "$(cd "$2" && pwd)" "$RESOLVED" <<'EOF'
import json, os, sys
manifest, root, out = sys.argv[1:]
with open(manifest, encoding="utf-8") as src, open(out, "w", encoding="utf-8") as dst:
    for line in src:
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("latent_path", "ref_latent_path"):
            if row.get(key) and not os.path.isabs(row[key]):
                row[key] = os.path.join(root, row[key])
        dst.write(json.dumps(row, ensure_ascii=False) + "\n")
EOF
  MANIFEST=$RESOLVED
  shift 2
fi
cd "$WORK"  # train_round3.yaml names the tokenizer relative to the work directory
exec "$PY" "$UPSTREAM/train.py" \
  --config "$HERE/train_round3.yaml" \
  --manifest "$MANIFEST" \
  --init-checkpoint "$WORK/init_pron_tags.pt" \
  --output-dir "$WORK/checkpoints" \
  "$@"
