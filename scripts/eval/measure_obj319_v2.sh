#!/usr/bin/env bash
# Round-2 objective measurement (PREREGISTRATION_v2.md item 5): no-edit and kana arms from round 1 (obj42g),
# the "ours" arm from the v2 adapters, same metric script, same item set, same per-item voices.
set -uo pipefail
. "$(dirname "$0")/../paths.sh"; require_env DATA
REPO=$REPO_ROOT; V=$DATA/v2
export TEST_DIR=$REPO/data/derived/obj319_routed TEST_SET=obj319
export ITEM_SPK_JSON=$REPO/preregistration/items/speaker_assignment42_obj319.json ITEM_SPK_CELL=OBJ319
# the truth is the ITEM's reading, not the tag (found 2026-09-09): fixed here
export TARGETS_FILE=${TARGETS_FILE:-$REPO/data/derived/obj319_routed/targets.txt}
PY=$REPO/scripts/eval/.venv_metrics/bin/python; export DISABLE_CUDNN=1
cd $REPO && $PY scripts/eval/measure_reading_all.py --arms-json ${ARMS_JSON:-data/analysis/v2/arms_obj319_round2.json} --out ${OUT_JSON:-data/analysis/v2/reading_obj319_round2.json} "$@"
