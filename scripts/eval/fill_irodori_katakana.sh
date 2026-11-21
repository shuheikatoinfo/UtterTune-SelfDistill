#!/usr/bin/env bash
# The five easy-set sentences whose target word is already katakana. run_blockB1.py locates the edit
# span by diffing plain vs reading text, which is empty here, so it produced no edited rendition.
# The edit is still well defined: surface == reading, and the tag itself names the span. Synthesised
# through the same router the ours arm uses, with a one-entry registry naming that word.
set -uo pipefail
. "$(dirname "$0")/../paths.sh"; require_env DATA IRODORI_DIR
# synthesize_routed.py lives in the Irodori code of this repo ($IRODORI_DIR).
ID=$IRODORI_DIR
PY=$ID/.venv/bin/python
ROOT=$DATA
CK=$ROOT/cleanroom_rebuild/irodori_v2/phon_cleanroom/checkpoint_final.pt
OUT=$ROOT/easy50_nat/irodori/ours
LOG=$1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$ID" || exit 1
while IFS=$'\t' read -r i spk word reading text; do
  w=$(printf "%s/%02d.wav" "$OUT" "$i")
  [ -f "$w" ] && { echo "[skip] $i" >> "$LOG"; continue; }
  reg=/tmp/reg_$i.json
  python3 -c "import json,sys;json.dump({sys.argv[1]:{'reading':sys.argv[2]}},open(sys.argv[3],'w'),ensure_ascii=False)" "$word" "$reading" "$reg"
  timeout 600 "$PY" synthesize_routed.py --text "$text" --registry "$reg" \
    --uttertune-checkpoint "$CK" --ref-wav "$ROOT/refvoice/refs42/spk$spk.wav" \
    --seed 1234 --out "$w" >>"$LOG" 2>&1
  echo "[fill] i=$i spk=$spk word=$word -> $([ -f "$w" ] && echo OK || echo FAIL)" >> "$LOG"
done < /tmp/iro_missing.tsv
echo "FILL_DONE ours=$(ls $OUT | wc -l) $(date +%H:%M:%S)" >> "$LOG"
