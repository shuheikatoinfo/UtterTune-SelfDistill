#!/usr/bin/env bash
# Round 3 (section 16): train the three non-Sarashina adapters on the filtered 50-carrier teacher sets and synthesize
# the difficult-50 (listening) and easy-50 (naturalness) tag arms. Local GPU, strictly one backbone at a time, in the
# order the teacher sets land (sentinel $V/r3/<bb>/TEACHERS_READY written by the collection step). Commands are the
# round-2 commands (v2_chain_locked4.sh) with round-3 paths; base arms are reused from round 2 where they are base-model.
set -uo pipefail
. "$(dirname "$0")/../paths.sh"; require_env DATA ITEM_SOURCE COSYVOICE_DIR T5GEMMA_DIR
R=$REPO_ROOT; T=$DATA; V=$T/v2; ITEMS=$R/preregistration/items
TD=$ITEM_SOURCE; L=$V/logs/round3_train_local.log
say(){ echo "[r3] $1 $(date +%F_%T)" | tee -a $L; }; cd $R; export EVAL_SPEAKERS_JSON=$T/eval_speakers/selected_42.json
done_bb=""
while [ $(echo $done_bb | wc -w) -lt 3 ]; do
  for bb in irodori cv2 t5gemma; do
    echo "$done_bb" | grep -qw $bb && continue; [ -f $V/r3/$bb/TEACHERS_READY ] || continue
    say "$bb: teachers ready, training"
    case $bb in
    cv2)
      PY=$COSYVOICE_DIR/.venv/bin/python
      $PY scripts/cv/train_lora_cv2.py --manifest $V/r3/cv2/manifest_filtered.tsv --out $V/r3/cv2/lora --steps 1000 > $V/logs/r3_cv2_train.log 2>&1
      grep -q "SAVED adapter" $V/logs/r3_cv2_train.log || { say "cv2 TRAIN FAILED"; done_bb="$done_bb $bb"; continue; }
      say "cv2 train done"; export ARMS=ours CV2_LORA=$V/r3/cv2/lora
      ITEM_SPK_JSON=$ITEMS/speaker_assignment42.json ITEM_SPK_CELL=B2_cv2_difficult BLOCKA_CV2_OUT=$V/synth_r3/cv2/difficult $PY scripts/cv/blockA_cv2_difficult.py > $V/logs/r3_cv2_synth_difficult.log 2>&1; say "cv2 difficult rc=$?"
      TEST_SET=easy_50 ITEM_SPK_JSON=$ITEMS/speaker_assignment42_easy50.json ITEM_SPK_CELL=EASY50 BLOCKA_CV2_OUT=$V/synth_r3/cv2/easy $PY scripts/cv/blockA_cv2_difficult.py > $V/logs/r3_cv2_synth_easy.log 2>&1; say "cv2 easy rc=$?"
      unset ARMS CV2_LORA ;;
    irodori)
      # Irodori code: irodori/ (setup.sh, prepare_tag_model.py, train.sh + train_round3.yaml, run_blockB1.py).
      PY=$IRODORI_PY; WORK=$V/r3/irodori/work
      export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
      ( $PY $IRODORI/prepare_tag_model.py --work-dir $WORK && bash $IRODORI/train.sh $WORK $V/r3/irodori/train_filtered.jsonl --output-dir $V/r3/irodori/phon --max-steps 2000 ) > $V/logs/r3_irodori_train.log 2>&1
      CK=$V/r3/irodori/phon/checkpoint_final.pt; [ -f "$CK" ] || { say "irodori TRAIN FAILED"; done_bb="$done_bb $bb"; continue; }
      say "irodori train done"
      for job in "difficult_50 $ITEMS/speaker_assignment42.json C2_irodori_difficult difficult" "easy_50 $ITEMS/speaker_assignment42_easy50.json EASY50 easy"; do
        set -- $job; ( SET=difficult TEST_DIR=$TD TEST_SET=$1 ITEM_SPK_JSON=$2 ITEM_SPK_CELL=$3 BLOCKB1_OUT=$V/synth_r3/irodori/$4 $PY $IRODORI/run_blockB1.py --arms ours --phon-checkpoint $CK --skip-eval ) > $V/logs/r3_irodori_synth_$4.log 2>&1; say "irodori $4 rc=$?"
      done; unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE ;;
    t5gemma)
      T5G=$T5GEMMA_DIR; PY=$T5G/.venv_t5gemma/bin/python
      ( cd $T5G && $PY $R/scripts/t5gemma/materialize_dataset.py --teachers $V/r3/t5gemma/teachers_filtered.jsonl --out $V/r3/t5gemma/dataset ) > $V/logs/r3_t5gemma_mat.log 2>&1
      ( cd $T5G && DATASET_DIR=$V/r3/t5gemma/dataset EXP_ROOT=$V/r3/t5gemma/readlora NUM_STEPS=400 SAVE_EVERY=100 bash $R/scripts/t5gemma/blockA_readlora_train.sh ) > $V/logs/r3_t5gemma_train.log 2>&1
      BUNDLE=$(ls -1 $V/r3/t5gemma/readlora/bundle_step*.pth 2>/dev/null | sort -V | tail -1); [ -n "$BUNDLE" ] || { say "t5gemma TRAIN FAILED"; done_bb="$done_bb $bb"; continue; }
      ( cd $T5G && $PY $R/scripts/t5gemma/export_readlora_hf.py --ckpt $BUNDLE --out $V/r3/t5gemma/merged ) > $V/logs/r3_t5gemma_export.log 2>&1
      [ -f $V/r3/t5gemma/merged/config.json ] || { say "t5gemma EXPORT FAILED"; done_bb="$done_bb $bb"; continue; }
      say "t5gemma train+export done"
      ( cd $T5G && ITEM_SPK_JSON=$ITEMS/speaker_assignment42.json ITEM_SPK_CELL=D2_t5gemma_difficult $PY $R/scripts/t5gemma/blockA_t5gemma.py --model_dir $V/r3/t5gemma/merged --out_root $V/synth_r3/t5gemma/difficult --arms ours ) > $V/logs/r3_t5gemma_synth_difficult.log 2>&1; say "t5gemma difficult rc=$?"
      ( cd $T5G && TEST_SET=easy_50 ITEM_SPK_JSON=$ITEMS/speaker_assignment42_easy50.json ITEM_SPK_CELL=EASY50 $PY $R/scripts/t5gemma/blockA_t5gemma.py --model_dir $V/r3/t5gemma/merged --out_root $V/synth_r3/t5gemma/easy --arms ours ) > $V/logs/r3_t5gemma_synth_easy.log 2>&1; say "t5gemma easy rc=$?" ;;
    esac
    done_bb="$done_bb $bb"; say "$bb stage done"
  done
  sleep 30
done
say "ROUND3_TRAIN_LOCAL_DONE"
