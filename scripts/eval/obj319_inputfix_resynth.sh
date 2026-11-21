#!/usr/bin/env bash
# Amendment 13: 74 of the 319 objective tag inputs were malformed by the routing step (a reading widened to a phrase by
# the accent review was placed on the original one-kanji span, or a suffix's reading appended twice). The corrected
# inputs replace the canonical file; the tag arm of those 74 items is re-synthesized on all four backbones with the
# registered adapters, same voices, seeds and guard; earlier clips are kept under *_dupbug.
set -uo pipefail
. "$(dirname "$0")/../paths.sh"; require_env DATA SARASHINA_DIR COSYVOICE_DIR T5GEMMA_DIR
R=$REPO_ROOT; T=$DATA; V=$T/v2; T5G=$T5GEMMA_DIR
OBJ=$R/data/derived/obj319_routed; ITEMS=$R/preregistration/items; L=$V/logs/obj319_inputfix.log
export EVAL_SPEAKERS_JSON=$T/eval_speakers/selected_42.json
say(){ echo "[fix] $1 $(date +%F_%T)" | tee -a $L; }
cd $R
cp $OBJ/test_obj319_lora_fixed20260917.txt $OBJ/test_obj319_lora.txt; say "canonical tag inputs replaced by the corrected file"
IDX=$(python3 -c "import json;print(' '.join('%02d'%i for i in json.load(open('$OBJ/changed_by_fix20260917.json'))))")
stash(){ for f in $IDX; do w=$1/$f.wav; [ -f "$w" ] && { d=$(dirname "$w" | sed "s#/obj319#/obj319_dupbug#"); mkdir -p "$d"; mv "$w" "$d/"; }; done; }
for d in $V/synth/sarashina/obj319/spk*/ours $V/synth/cv2/obj319/spk*/ours $V/synth/irodori/obj319/ours $V/synth/t5gemma/obj319/blockA_t5gemma_difficult/ours; do stash $d; done
for g in $V/synth/sarashina/obj319 $V/synth/cv2/obj319 $V/synth/irodori/obj319; do [ -f $g/guard_takes.json ] && cp $g/guard_takes.json $g/guard_takes.before_inputfix.json; done
say "stashed: sarashina $(ls $V/synth/sarashina/obj319_dupbug/spk*/ours/*.wav 2>/dev/null | wc -l) cv2 $(ls $V/synth/cv2/obj319_dupbug/spk*/ours/*.wav 2>/dev/null | wc -l) irodori $(ls $V/synth/irodori/obj319_dupbug/ours/*.wav 2>/dev/null | wc -l) t5gemma $(ls $V/synth/t5gemma/obj319_dupbug/blockA_t5gemma_difficult/ours/*.wav 2>/dev/null | wc -l)"
say "sarashina start"
SARA_ADAPTER_OURS=$V/sarashina/lora/ckpt_20000 TEST_DIR=$OBJ TEST_SET=obj319 ITEM_SPK_JSON=$ITEMS/speaker_assignment42_obj319.json ITEM_SPK_CELL=OBJ319 BLOCK_OUT=$V/synth/sarashina/obj319 \
  $T/_handoff/venv_tf457/bin/python scripts/blockA_sarashina_difficult_4spk.py --nspk 42 > $V/logs/sarashina_synth_obj319_inputfix.log 2>&1; say "sarashina rc=$? ours=$(ls $V/synth/sarashina/obj319/spk*/ours/*.wav | wc -l)"
cp $V/synth/sarashina/obj319/guard_takes.json $V/synth/sarashina/obj319/guard_takes.inputfix74.json
say "cv2 start"
ARMS=ours CV2_LORA=$V/cv2/lora TEST_DIR=$OBJ TEST_SET=obj319 ITEM_SPK_JSON=$ITEMS/speaker_assignment42_obj319.json ITEM_SPK_CELL=OBJ319 BLOCKA_CV2_OUT=$V/synth/cv2/obj319 \
  $COSYVOICE_DIR/.venv/bin/python scripts/cv/blockA_cv2_difficult.py > $V/logs/cv2_synth_obj319_inputfix.log 2>&1; say "cv2 rc=$? ours=$(ls $V/synth/cv2/obj319/spk*/ours/*.wav | wc -l)"
say "irodori start"
( export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1; SET=difficult TEST_DIR=$OBJ TEST_SET=obj319 ITEM_SPK_JSON=$ITEMS/speaker_assignment42_obj319.json ITEM_SPK_CELL=OBJ319 \
    BLOCKB1_OUT=$V/synth/irodori/obj319 $IRODORI_PY $IRODORI/run_blockB1.py --arms ours --phon-checkpoint $V/irodori/phon/checkpoint_final.pt --skip-eval ) > $V/logs/irodori_synth_obj319_inputfix.log 2>&1; say "irodori rc=$? ours=$(ls $V/synth/irodori/obj319/ours/*.wav | wc -l)"
say "t5gemma start"
( cd $T5G && TEST_DIR=$OBJ TEST_SET=obj319 ITEM_SPK_JSON=$ITEMS/speaker_assignment42_obj319.json ITEM_SPK_CELL=OBJ319 \
    $T5G/.venv_t5gemma/bin/python $R/scripts/t5gemma/blockA_t5gemma.py --model_dir $V/t5gemma/merged --out_root $V/synth/t5gemma/obj319 --arms ours ) > $V/logs/t5gemma_synth_obj319_inputfix.log 2>&1; say "t5gemma rc=$? ours=$(ls $V/synth/t5gemma/obj319/blockA_t5gemma_difficult/ours/*.wav | wc -l)"
say "measure start"
OUT_JSON=data/analysis/v2/reading_obj319_round2_inputfix.json bash scripts/eval/measure_obj319_v2.sh > $V/logs/obj319_inputfix_measure.log 2>&1
python3 scripts/eval/obj319_contrasts_v2.py --results data/analysis/v2/reading_obj319_round2_inputfix.json --arms-json data/analysis/v2/arms_obj319_round2.json 2>&1 | tee -a $L
say "INPUTFIX DONE"
