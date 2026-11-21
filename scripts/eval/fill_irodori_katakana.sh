#!/usr/bin/env bash
# The five easy-set sentences whose target word is already katakana. The earlier loader located the
# edit span by diffing plain vs reading text, which is empty here, so it produced no edited rendition.
# The edit is still well defined: surface == reading, and the tag itself names the span: the word is
# wrapped as <PHON_START>reading<PHON_END> (a line already tagged is used as it is) and synthesized
# with irodori/synthesis.py.
set -uo pipefail
. "$(dirname "$0")/../paths.sh"; require_env DATA
PY=$IRODORI_PY
ROOT=$DATA
CK=$ROOT/cleanroom_rebuild/irodori_v2/phon_cleanroom/checkpoint_final.pt
OUT=$ROOT/easy50_nat/irodori/ours
LOG=$1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
while IFS=$'\t' read -r i spk word reading text; do
  w=$(printf "%s/%02d.wav" "$OUT" "$i")
  [ -f "$w" ] && { echo "[skip] $i" >> "$LOG"; continue; }
  tagged=$(python3 -c "import sys; t, w, r = sys.argv[1:]; print(t if '<PHON_START>' in t else t.replace(w, f'<PHON_START>{r}<PHON_END>', 1))" "$text" "$word" "$reading")
  timeout 600 "$PY" "$IRODORI/synthesis.py" --text "$tagged" --checkpoint "$CK" \
    --ref-wav "$ROOT/refvoice/refs42/spk$spk.wav" --seed 1234 --no-watermark --out "$w" >>"$LOG" 2>&1
  echo "[fill] i=$i spk=$spk word=$word -> $([ -f "$w" ] && echo OK || echo FAIL)" >> "$LOG"
done < /tmp/iro_missing.tsv
echo "FILL_DONE ours=$(ls $OUT | wc -l) $(date +%H:%M:%S)" >> "$LOG"
