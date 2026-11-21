#!/usr/bin/env bash
# Round 3 Irodori (section 16 as restored 10:50: section-12 carriers, slow-teacher filter): train locally with
# checkpoints every 500 steps to 4000, then read every checkpoint on the two validation sets (base arms once).
set -uo pipefail
. "$(dirname "$0")/../paths.sh"; require_env DATA
R=$REPO_ROOT; T=$DATA; V=$T/v2; ITEMS=$R/preregistration/items
L=$V/logs/round3_irodori_local.log; say(){ echo "[r3-iro] $1 $(date +%F_%T)" | tee -a $L; }; cd $R; export EVAL_SPEAKERS_JSON=$T/eval_speakers/selected_42.json
# Irodori code: irodori/ (setup.sh, prepare_tag_model.py, train.sh + train_round3.yaml, run_blockB1.py).
PY=$IRODORI_PY; WORK=$V/r3/irodori/work
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
say "train start ($(wc -l < $V/r3/irodori/train_filtered.jsonl) rows)"
( $PY $IRODORI/prepare_tag_model.py --work-dir $WORK && bash $IRODORI/train.sh $WORK $V/r3/irodori/train_filtered.jsonl --output-dir $V/r3/irodori/phon --max-steps 4000 --save-every 500 ) > $V/logs/r3_irodori_train.log 2>&1
say "train done: $(ls $V/r3/irodori/phon | tr '\n' ' ')"
SEL=$R/data/derived/sel105_routed; DEV=$R/data/derived/deveasy
for s in sel105 deveasy; do d=$SEL; cell=SEL105; [ $s = deveasy ] && { d=$DEV; cell=DEVEASY; }; aj=$ITEMS/speaker_assignment42_$s.json
  ( SET=difficult TEST_DIR=$d TEST_SET=$s ITEM_SPK_JSON=$aj ITEM_SPK_CELL=$cell BLOCKB1_OUT=$V/r3val/irodori/$s/base $PY $IRODORI/run_blockB1.py --arms noedit --skip-eval ) > $V/logs/r3val_irodori_${s}_noedit.log 2>&1; say "$s noedit rc=$?"
  ( SET=difficult TEST_DIR=$d TEST_SET=$s ITEM_SPK_JSON=$aj ITEM_SPK_CELL=$cell BLOCKB1_OUT=$V/r3val/irodori/$s/base_kana $PY $IRODORI/run_blockB1.py --arms kana --skip-eval ) > $V/logs/r3val_irodori_${s}_kana.log 2>&1; say "$s kana rc=$?"
done
for st in 500 1000 2000 4000; do
  CK=$(ls $V/r3/irodori/phon/checkpoint_$(printf %07d $st)* 2>/dev/null | head -1); [ $st = 4000 ] && [ -z "$CK" ] && CK=$V/r3/irodori/phon/checkpoint_final.pt
  [ -n "$CK" ] || { say "no checkpoint at $st"; continue; }
  for s in sel105 deveasy; do d=$SEL; cell=SEL105; [ $s = deveasy ] && { d=$DEV; cell=DEVEASY; }; aj=$ITEMS/speaker_assignment42_$s.json
    ( SET=difficult TEST_DIR=$d TEST_SET=$s ITEM_SPK_JSON=$aj ITEM_SPK_CELL=$cell BLOCKB1_OUT=$V/r3val/irodori/$s/s$st $PY $IRODORI/run_blockB1.py --arms ours --phon-checkpoint $CK --skip-eval ) > $V/logs/r3val_irodori_${s}_s$st.log 2>&1; say "$s s$st rc=$?"
  done
done
say "IRODORI_VAL_SWEEP_DONE"; touch $V/r3/irodori/VAL_SWEEP_DONE
