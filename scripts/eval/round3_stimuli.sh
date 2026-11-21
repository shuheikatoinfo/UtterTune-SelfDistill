#!/usr/bin/env bash
# Round 3 stimuli for the three non-Sarashina backbones, synthesized only once the chosen checkpoint is written to
# $V/r3/<bb>/CHOSEN_CKPT (a path). difficult-50 first (listening), then easy-50 (naturalness). Local GPU, one at a time.
# Sarashina's stimuli were synthesized with P4 at 07:08-07:18 (synth_r3/sarashina).
set -uo pipefail
. "$(dirname "$0")/../paths.sh"; require_env DATA ITEM_SOURCE COSYVOICE_DIR IRODORI_DIR T5GEMMA_DIR
R=$REPO_ROOT; T=$DATA; V=$T/v2; ITEMS=$R/preregistration/items
L=$V/logs/round3_stimuli.log; say(){ echo "[r3-stim] $1 $(date +%F_%T)" | tee -a $L; }; cd $R; export EVAL_SPEAKERS_JSON=$T/eval_speakers/selected_42.json
TD=$ITEM_SOURCE
done_bb=""
while [ $(echo $done_bb | wc -w) -lt 3 ]; do
  for bb in cv2 irodori t5gemma; do
    echo "$done_bb" | grep -qw $bb && continue; [ -f $V/r3/$bb/CHOSEN_CKPT ] || continue
    CK=$(cat $V/r3/$bb/CHOSEN_CKPT); say "$bb: chosen $CK"
    case $bb in
    cv2)
      PY=$COSYVOICE_DIR/.venv/bin/python; export ARMS=ours CV2_LORA=$CK
      ITEM_SPK_JSON=$ITEMS/speaker_assignment42.json ITEM_SPK_CELL=B2_cv2_difficult BLOCKA_CV2_OUT=$V/synth_r3/cv2/difficult $PY scripts/cv/blockA_cv2_difficult.py > $V/logs/r3_cv2_synth_difficult.log 2>&1; say "cv2 difficult rc=$? ($(ls $V/synth_r3/cv2/difficult/spk*/ours/*.wav 2>/dev/null | wc -l) clips)"
      TEST_SET=easy_50 ITEM_SPK_JSON=$ITEMS/speaker_assignment42_easy50.json ITEM_SPK_CELL=EASY50 BLOCKA_CV2_OUT=$V/synth_r3/cv2/easy $PY scripts/cv/blockA_cv2_difficult.py > $V/logs/r3_cv2_synth_easy.log 2>&1; say "cv2 easy rc=$?"
      unset ARMS CV2_LORA ;;
    irodori)
      # run_blockB1.py lives in the Irodori code of this repo ($IRODORI_DIR).
      ID=$IRODORI_DIR; PY=$ID/.venv/bin/python
      export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
      ( cd $ID && SET=difficult TEST_DIR=$TD TEST_SET=difficult_50 ITEM_SPK_JSON=$ITEMS/speaker_assignment42.json ITEM_SPK_CELL=C2_irodori_difficult BLOCKB1_OUT=$V/synth_r3/irodori/difficult $PY $IRODORI_DIR/run_blockB1.py --arms ours --phon-checkpoint $CK --skip-eval ) > $V/logs/r3_irodori_synth_difficult.log 2>&1; say "irodori difficult rc=$? ($(ls $V/synth_r3/irodori/difficult/ours/*.wav 2>/dev/null | wc -l) clips)"
      ( cd $ID && SET=difficult TEST_DIR=$TD TEST_SET=easy_50 ITEM_SPK_JSON=$ITEMS/speaker_assignment42_easy50.json ITEM_SPK_CELL=EASY50 BLOCKB1_OUT=$V/synth_r3/irodori/easy $PY $IRODORI_DIR/run_blockB1.py --arms ours --phon-checkpoint $CK --skip-eval ) > $V/logs/r3_irodori_synth_easy.log 2>&1; say "irodori easy rc=$?"
      unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE ;;
    t5gemma)
      # CK is a merged model directory (exported from the chosen bundle; the pipeline pulls it as merged_s<steps>)
      T5G=$T5GEMMA_DIR; PY=$T5G/.venv_t5gemma/bin/python
      ( cd $T5G && ITEM_SPK_JSON=$ITEMS/speaker_assignment42.json ITEM_SPK_CELL=D2_t5gemma_difficult $PY $R/scripts/t5gemma/blockA_t5gemma.py --model_dir $CK --out_root $V/synth_r3/t5gemma/difficult --arms ours ) > $V/logs/r3_t5gemma_synth_difficult.log 2>&1; say "t5gemma difficult rc=$? ($(ls $V/synth_r3/t5gemma/difficult/blockA_t5gemma_difficult/ours/*.wav 2>/dev/null | wc -l) clips)"
      ( cd $T5G && TEST_SET=easy_50 ITEM_SPK_JSON=$ITEMS/speaker_assignment42_easy50.json ITEM_SPK_CELL=EASY50 $PY $R/scripts/t5gemma/blockA_t5gemma.py --model_dir $CK --out_root $V/synth_r3/t5gemma/easy --arms ours ) > $V/logs/r3_t5gemma_synth_easy.log 2>&1; say "t5gemma easy rc=$?" ;;
    esac
    done_bb="$done_bb $bb"
  done
  sleep 20
done
say "ROUND3_STIMULI_DONE"
