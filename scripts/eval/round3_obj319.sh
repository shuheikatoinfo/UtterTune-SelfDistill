#!/usr/bin/env bash
# Round 3 (section 16): tag arm of the 319-item objective test half with the round-3 adapters (Irodori step 2,000,
# CosyVoice 2 step 1,000, T5Gemma step 600; Sarashina = the slow-teacher-filter adapter already read in section 14).
# Same items, voices, seeds, guard and scripts as round 2; unedited, kana and UtterTune arms are the round-2 clips.
# Every adapter path is given explicitly and logged with its hash. One job at a time on the shared GPU.
set -uo pipefail
. "$(dirname "$0")/../paths.sh"; require_env DATA COSYVOICE_DIR T5GEMMA_DIR
R=$REPO_ROOT; T=$DATA; V=$T/v2
OBJ=$R/data/derived/obj319_routed; ITEMS=$R/preregistration/items; L=$V/logs/round3_obj319.log
say(){ echo "[r3obj] $1 $(date +%F_%T)" | tee -a $L; }; cd $R
export EVAL_SPEAKERS_JSON=$T/eval_speakers/selected_42.json
CK_I=$(cat $V/r3/irodori/CHOSEN_CKPT); CK_C=$(cat $V/r3/cv2/CHOSEN_CKPT); B_T=$V/r3/t5gemma/readlora_pth/bundle_step600.pth
say "adapters: irodori $CK_I $(md5sum $CK_I | cut -c1-12) | cv2 $CK_C $(md5sum $CK_C/adapter_model.safetensors | cut -c1-12) | t5gemma $B_T $(md5sum $B_T | cut -c1-12)"
# run_blockB1.py is in irodori/ (environment from irodori/setup.sh).
( HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 SET=difficult TEST_DIR=$OBJ TEST_SET=obj319 ITEM_SPK_JSON=$ITEMS/speaker_assignment42_obj319.json ITEM_SPK_CELL=OBJ319 \
    BLOCKB1_OUT=$V/synth_r3/irodori/obj319 $IRODORI_PY $IRODORI/run_blockB1.py --arms ours --phon-checkpoint $CK_I --skip-eval ) > $V/logs/r3_irodori_obj319.log 2>&1
say "irodori rc=$? ours=$(ls $V/synth_r3/irodori/obj319/ours/*.wav 2>/dev/null | wc -l)"
ARMS=ours CV2_LORA=$CK_C TEST_DIR=$OBJ TEST_SET=obj319 ITEM_SPK_JSON=$ITEMS/speaker_assignment42_obj319.json ITEM_SPK_CELL=OBJ319 BLOCKA_CV2_OUT=$V/synth_r3/cv2/obj319 \
  $COSYVOICE_DIR/.venv/bin/python scripts/cv/blockA_cv2_difficult.py > $V/logs/r3_cv2_obj319.log 2>&1
say "cv2 rc=$? ours=$(ls $V/synth_r3/cv2/obj319/spk*/ours/*.wav 2>/dev/null | wc -l)"
T5G=$T5GEMMA_DIR; M=$V/r3/t5gemma/merged_s600
[ -f $M/config.json ] || ( cd $T5G && $T5G/.venv_t5gemma/bin/python $R/scripts/t5gemma/export_readlora_hf.py --ckpt $B_T --out $M ) > $V/logs/r3_t5gemma_export.log 2>&1
say "t5gemma export rc=$? $(ls $M 2>/dev/null | wc -l) files"
( cd $T5G && TEST_DIR=$OBJ TEST_SET=obj319 ITEM_SPK_JSON=$ITEMS/speaker_assignment42_obj319.json ITEM_SPK_CELL=OBJ319 \
    $T5G/.venv_t5gemma/bin/python $R/scripts/t5gemma/blockA_t5gemma.py --model_dir $M --out_root $V/synth_r3/t5gemma/obj319 --arms ours ) > $V/logs/r3_t5gemma_obj319.log 2>&1
say "t5gemma rc=$? ours=$(ls $V/synth_r3/t5gemma/obj319/blockA_t5gemma_difficult/ours/*.wav 2>/dev/null | wc -l)"
python3 - <<PY
import json
V="$V"; d=json.load(open("$R/data/analysis/v2/arms_obj319_round2.json"))
d["sara42g"]["ours"]=f"{V}/synth/sarashina_P4/obj319"; d["cv242g"]["ours"]=f"{V}/synth_r3/cv2/obj319"
d["iro42g"]["ours"]=f"{V}/synth_r3/irodori/obj319"; d["t5g42g"]["ours"]=f"{V}/synth_r3/t5gemma/obj319/blockA_t5gemma_difficult"
json.dump(d,open("$R/data/analysis/v2/arms_obj319_round3.json","w"),indent=1)
PY
ARMS_JSON=data/analysis/v2/arms_obj319_round3.json OUT_JSON=data/analysis/v2/reading_obj319_round3.json bash scripts/eval/measure_obj319_v2.sh > $V/logs/r3_obj319_measure.log 2>&1
say "measure rc=$?"
python3 scripts/eval/obj319_contrasts_v2.py --results data/analysis/v2/reading_obj319_round3.json --arms-json data/analysis/v2/arms_obj319_round3.json > data/analysis/v2/obj319_round3_contrasts.md 2>&1
say "R3_OBJ319_DONE"
