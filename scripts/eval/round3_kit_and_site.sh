#!/usr/bin/env bash
# Round 3 listening kit and panel site (section 16). Builds only after every backbone's difficult-50 tag arm exists
# under synth_r3 (CHOSEN adapters). Steps: kit (round3 profile) -> loudness normalization (as round 2) -> silence
# trimming (lead > 0.3 s, tail > 0.5 s, every clip of every arm; new in round 3, recorded in section 16) -> one panel
# block -> static site for project jp-accent-listening-r3. Deployment is a separate, checked step (round3_panel_deploy.sh).
set -euo pipefail
. "$(dirname "$0")/../paths.sh"; require_env DATA
REPO=$REPO_ROOT; V=$DATA/v2
KIT=${KIT_ROOT:-$V/blind_kits_r3}; NORM=${KIT_ROOT:-$V/blind_kits_r3}_norm; SITE=${SITE_ROOT:-$V/cf_panel_r3}; cd $REPO
SKIP=${KIT_SKIP:-}; T5=""; echo "$SKIP" | grep -q t5gemma || T5="$V/synth_r3/t5gemma/difficult/blockA_t5gemma_difficult/ours $V/synth/t5gemma/difficult_base/blockA_t5gemma_difficult/kana"
CV=""; echo "$SKIP" | grep -q cv2 || CV="$V/synth_r3/cv2/difficult $V/synth/cv2/difficult"
for d in $V/synth_r3/sarashina/difficult $V/synth_r3/irodori/difficult/ours $V/synth/sarashina/difficult $V/synth/irodori/difficult_kana/noedit $CV $T5; do
  [ -d "$d" ] || { echo "MISSING synthesis: $d"; exit 1; }; done
[ "$(ls $V/synth_r3/sarashina/difficult/spk*/ours/*.wav | wc -l)" = 50 ] || { echo "A ours incomplete"; exit 1; }
echo "$SKIP" | grep -q cv2 || [ "$(ls $V/synth_r3/cv2/difficult/spk*/ours/*.wav | wc -l)" = 50 ] || { echo "B ours incomplete"; exit 1; }
[ "$(ls $V/synth_r3/irodori/difficult/ours/*.wav | wc -l)" = 50 ] || { echo "C ours incomplete"; exit 1; }
echo "$SKIP" | grep -q t5gemma || [ "$(ls $V/synth_r3/t5gemma/difficult/blockA_t5gemma_difficult/ours/*.wav | wc -l)" = 50 ] || { echo "D ours incomplete"; exit 1; }
rm -rf "$KIT" "$NORM" "$SITE"
CELL_SUFFIX=_r3 KIT_OUT=$KIT KIT_PROFILE=round3 KIT_SKIP="$SKIP" FULL_SENTENCE=1 python3 scripts/eval/build_blind_kits.py --build
python3 scripts/eval/normalize_loudness.py --src "$KIT" --out "$NORM" --recursive
for c in "$KIT"/*/; do c=${c%/}; n="$NORM/$(basename $c)"; for f in MANIFEST.csv KEY_DO_NOT_OPEN.csv README_rater.md; do [ -f "$c/$f" ] && cp "$c/$f" "$n/$f"; done; done
echo "no trimming (author, 15:22)"
for c in $NORM/*/; do
  c=${c%/}; [ -f "$c/KEY_DO_NOT_OPEN.csv" ] || continue
  python3 - "$c" <<'PY'
import csv, sys
c = sys.argv[1]
rows = list(csv.DictReader(open(f"{c}/KEY_DO_NOT_OPEN.csv", encoding="utf-8")))
with open(f"{c}/BLOCKS.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["blind_id", "block"]); w.writeheader()
    w.writerows({"blind_id": r["blind_id"], "block": 0} for r in rows)
print(f"  {c.split('/')[-1]}: {len(rows)} rows -> one panel block")
PY
done
python3 scripts/rater_ui/build_static.py --panel --kits "$NORM" --out "$SITE" --project jp-accent-listening-r3
python3 -c "import json;d=json.load(open('$SITE/public/index.json'));print([k['kit'] for k in d['kits']], 'block0 items', len(d['blocks']['0']))"
echo "KIT_R3_BUILT"
