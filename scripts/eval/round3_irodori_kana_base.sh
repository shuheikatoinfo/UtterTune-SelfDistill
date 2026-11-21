#!/usr/bin/env bash
# Irodori's kana baseline on the validation sets = the base arm ("noedit") over the kana sentences (as in round 2:
# difficult_kana / obj319_kana). Runs after the main Irodori validation chain (round3_irodori_local.sh) has finished.
set -uo pipefail
. "$(dirname "$0")/../paths.sh"; require_env DATA
R=$REPO_ROOT; T=$DATA; V=$T/v2; ITEMS=$R/preregistration/items; PY=$IRODORI_PY; cd $R
export EVAL_SPEAKERS_JSON=$T/eval_speakers/selected_42.json HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
until [ -f $V/r3/irodori/VAL_SWEEP_DONE ]; do sleep 30; done
for s in sel105 deveasy; do cell=SEL105; [ $s = deveasy ] && cell=DEVEASY
  ( SET=difficult TEST_DIR=$R/data/derived/${s}_kana TEST_SET=$s ITEM_SPK_JSON=$ITEMS/speaker_assignment42_$s.json ITEM_SPK_CELL=$cell BLOCKB1_OUT=$V/r3val/irodori/$s/base_kana $PY $IRODORI/run_blockB1.py --arms noedit --skip-eval ) > $V/logs/r3val_irodori_${s}_kana.log 2>&1; echo "$s kana-base rc=$? $(date +%T)"
done
echo IRODORI_KANA_BASE_DONE
