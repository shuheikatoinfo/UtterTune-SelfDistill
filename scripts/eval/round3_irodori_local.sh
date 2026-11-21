#!/usr/bin/env bash
# Round 3 Irodori (section 16 as restored 10:50: section-12 carriers, slow-teacher filter): train locally with
# checkpoints every 500 steps to 4000, then read every checkpoint on the two validation sets (base arms once).
set -uo pipefail
. "$(dirname "$0")/../paths.sh"; require_env DATA IRODORI_DIR
R=$REPO_ROOT; T=$DATA; V=$T/v2; ITEMS=$R/preregistration/items
L=$V/logs/round3_irodori_local.log; say(){ echo "[r3-iro] $1 $(date +%F_%T)" | tee -a $L; }; cd $R; export EVAL_SPEAKERS_JSON=$T/eval_speakers/selected_42.json
# train.py, its configs and run_blockB1.py live in the Irodori code of this repo ($IRODORI_DIR).
ID=$IRODORI_DIR; PY=$ID/.venv/bin/python
BASE=$IRODORI_DIR/downloads/Irodori-TTS-500M-v3/model.safetensors
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
say "train start ($(wc -l < $V/r3/irodori/train_filtered.jsonl) rows)"
( cd $ID && $PY $ID/train.py --config $ID/configs/train_500m_v3_phon_anchorbank_10k_better3k.yaml --manifest $V/r3/irodori/train_filtered.jsonl --init-checkpoint $BASE --output-dir $V/r3/irodori/phon --max-steps 4000 --save-every 500 --checkpoint-best-n 0 ) > $V/logs/r3_irodori_train.log 2>&1
say "train done: $(ls $V/r3/irodori/phon | tr '\n' ' ')"
SEL=$R/data/derived/sel105_routed; DEV=$R/data/derived/deveasy
for s in sel105 deveasy; do d=$SEL; cell=SEL105; [ $s = deveasy ] && { d=$DEV; cell=DEVEASY; }; aj=$ITEMS/speaker_assignment42_$s.json
  ( cd $ID && SET=difficult TEST_DIR=$d TEST_SET=$s ITEM_SPK_JSON=$aj ITEM_SPK_CELL=$cell BLOCKB1_OUT=$V/r3val/irodori/$s/base $PY $IRODORI_DIR/run_blockB1.py --arms noedit --skip-eval ) > $V/logs/r3val_irodori_${s}_noedit.log 2>&1; say "$s noedit rc=$?"
  ( cd $ID && SET=difficult TEST_DIR=$d TEST_SET=$s ITEM_SPK_JSON=$aj ITEM_SPK_CELL=$cell BLOCKB1_OUT=$V/r3val/irodori/$s/base_kana $PY $IRODORI_DIR/run_blockB1.py --arms kana --skip-eval ) > $V/logs/r3val_irodori_${s}_kana.log 2>&1; say "$s kana rc=$?"
done
for st in 500 1000 2000 4000; do
  CK=$(ls $V/r3/irodori/phon/checkpoint_$(printf %07d $st)* 2>/dev/null | head -1); [ $st = 4000 ] && [ -z "$CK" ] && CK=$V/r3/irodori/phon/checkpoint_final.pt
  [ -n "$CK" ] || { say "no checkpoint at $st"; continue; }
  for s in sel105 deveasy; do d=$SEL; cell=SEL105; [ $s = deveasy ] && { d=$DEV; cell=DEVEASY; }; aj=$ITEMS/speaker_assignment42_$s.json
    ( cd $ID && SET=difficult TEST_DIR=$d TEST_SET=$s ITEM_SPK_JSON=$aj ITEM_SPK_CELL=$cell BLOCKB1_OUT=$V/r3val/irodori/$s/s$st $PY $IRODORI_DIR/run_blockB1.py --arms ours --phon-checkpoint $CK --skip-eval ) > $V/logs/r3val_irodori_${s}_s$st.log 2>&1; say "$s s$st rc=$?"
  done
done
say "IRODORI_VAL_SWEEP_DONE"; touch $V/r3/irodori/VAL_SWEEP_DONE
