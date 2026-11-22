#!/usr/bin/env bash
# Round 3 naturalness kit and site (section 16): easy-50 pairs on all four backbones. Unedited arm = round-2 base-model
# clips; edited arm = round-3 adapters (synth_r3, easy-50 tags as reviewed on 09-18). Same builder, rules and
# exclusions as round 2; new pair-id prefix q; loudness normalization only (no trimming); fee 700 yen.
set -euo pipefail
. "$(dirname "$0")/../paths.sh"; require_env DATA
R=$REPO_ROOT; V=$DATA/v2; S=$V/nat_src_r3; cd $R
SKIP=${NAT_SKIP:-}
for bb in sarashina cv2; do for arm in noedit ours; do rm -rf $S/$bb/$arm; mkdir -p $S/$bb/$arm
  src=$V/synth/$bb/easy; [ $arm = ours ] && src=$V/synth_r3/$bb/easy
  for f in $src/spk*/$arm/*.wav; do n=$(basename $f); [ -e $S/$bb/$arm/$n ] && { echo "DUP $bb $arm $n"; exit 1; }; cp $f $S/$bb/$arm/$n; done; done; done
rm -rf $S/irodori; mkdir -p $S/irodori/noedit $S/irodori/ours; cp $V/synth/irodori/easy/noedit/*.wav $S/irodori/noedit/; cp $V/synth_r3/irodori/easy/ours/*.wav $S/irodori/ours/
if ! echo "$SKIP" | grep -q t5gemma; then rm -rf $S/t5gemma; mkdir -p $S/t5gemma/noedit $S/t5gemma/ours
  cp $V/synth/t5gemma/easy_base/blockA_t5gemma_difficult/noedit/*.wav $S/t5gemma/noedit/; cp $V/synth_r3/t5gemma/easy/blockA_t5gemma_difficult/ours/*.wav $S/t5gemma/ours/; else rm -rf $S/t5gemma; fi
for bb in $(ls $S); do n=$(ls $S/$bb/ours | wc -l); m=$(ls $S/$bb/noedit | wc -l); echo "$bb: ours=$n noedit=$m"; [ "$n" = 50 ] && [ "$m" = 50 ] || { echo "INCOMPLETE $bb"; exit 1; }; done
OUT=${NAT_OUT:-$V/naturalness_kit_round3}; SITE=${NAT_SITE:-$V/cf_nat_round3}; ITEMS=${NAT_ITEMS:-preregistration/items/naturalness_ab_easy50_round3.json}; PROJECT=${NAT_PROJECT:-jp-accent-naturalness-round3}
python3 scripts/eval/build_naturalness_ab.py --root $S --out $ITEMS --audio-out $OUT --id-prefix q --write
rm -rf ${OUT}_norm; python3 scripts/eval/normalize_loudness.py --src $OUT --out ${OUT}_norm --recursive
rm -rf $SITE; python3 scripts/rater_ui/build_naturalness_site.py --items $ITEMS --clips ${OUT}_norm --out $SITE --project $PROJECT --groups 4 --fee "700円"
echo "NAT_R3_BUILT -> $SITE"
