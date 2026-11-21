#!/usr/bin/env bash
# Section 14: the variant that met the 100-item criterion (P4, slow teachers removed) read on all 319 objective items
# against the section-12 criterion: tag arm with the adapter given explicitly, registered voices, seeds, guard; unedited
# and kana arms are the registered round-2 clips (linked, not re-synthesized).
set -uo pipefail
. "$(dirname "$0")/../paths.sh"; require_env DATA SARASHINA_DIR
R=$REPO_ROOT; T=$DATA; V=$T/v2; PY=$T/_handoff/venv_tf457/bin/python
A=$V/sara_hyp/pods/shy-3r/hyp/P4_not_slow/lora/ckpt_20000; O=$V/synth/sarashina_P4/obj319; mkdir -p $O; cd $R
for s in $V/synth/sarashina/obj319/spk*; do b=$(basename $s); mkdir -p $O/$b; for arm in noedit kana; do [ -e $O/$b/$arm ] || ln -s $s/$arm $O/$b/$arm; done; done
export EVAL_SPEAKERS_JSON=$T/eval_speakers/selected_42.json
SARA_ADAPTER_OURS=$A TEST_DIR=$R/data/derived/obj319_routed TEST_SET=obj319 ITEM_SPK_JSON=$R/preregistration/items/speaker_assignment42_obj319.json ITEM_SPK_CELL=OBJ319 BLOCK_OUT=$O \
  $PY scripts/blockA_sarashina_difficult_4spk.py --nspk 42 > $V/logs/sarashina_P4_synth_obj319.log 2>&1
echo "synth rc=$? ours=$(ls $O/spk*/ours/*.wav | wc -l) $(date +%T)"
mkdir -p $R/data/analysis/v2
python3 -c "import json; json.dump({'sara42g': {'noedit': '$O', 'kana': '$O', 'ours': '$O'}}, open('$R/data/analysis/v2/arms_obj319_sarashina_P4.json', 'w'), indent=1)"
ARMS_JSON=data/analysis/v2/arms_obj319_sarashina_P4.json OUT_JSON=data/analysis/v2/reading_obj319_sarashina_P4.json bash scripts/eval/measure_obj319_v2.sh > $V/logs/sarashina_P4_measure.log 2>&1
python3 scripts/eval/obj319_contrasts_v2.py --results data/analysis/v2/reading_obj319_sarashina_P4.json --arms-json data/analysis/v2/arms_obj319_sarashina_P4.json
echo "P4_FULL_DONE $(date +%T)"
