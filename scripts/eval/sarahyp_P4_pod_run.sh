#!/usr/bin/env bash
# Section 14, on one pod per variant (SHARD k/5 picks it): the registered trainer and settings (20,000 steps, alpha 64)
# on a filtered teacher set, then the 100-item tag-arm diagnostic (same inputs, voices, seeds, guard). Variant P3 =
# registered teachers, checkpoints at 1k/2k/3k. The shipped Sarashina adapter is P4's step 20,000.
# Pod layout: /workspace/venv (the Sarashina environment), /workspace/sara-tts-weights (sbintuitions/sarashina2.2-tts),
# /workspace/bundle/{scripts (this repository's scripts/), sarashina2.2-tts, teachers/teachers_<variant>.jsonl,
# eval (obj319_routed, speaker_assignment42_obj319.json, the 42 voices and selected_42_pod.json naming them)}.
set -uo pipefail
export HF_TOKEN=$(tr "\0" "\n" < /proc/1/environ | grep ^HF_TOKEN= | cut -d= -f2-); export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
K=${SHARD%%/*}; NAMES=(P1_tag_not_initial P2_no_long_repeats P3_registered_early P4_not_slow P5_all_three); N=${NAMES[$K]}
until [ -f /workspace/sara-tts-weights/UPLOAD_DONE ]; do sleep 15; done
cd /workspace && . venv/bin/activate; E=/workspace/bundle/eval; O=/workspace/hyp/$N; mkdir -p $O
W=/workspace/sara-tts-weights
# scripts/paths.py: the synthesis script reads the weights from $DATA/_handoff/sara-tts-weights.
export DATA=/workspace/data SARASHINA_DIR=/workspace/bundle/sarashina2.2-tts
mkdir -p $DATA/_handoff && ln -sfn $W $DATA/_handoff/sara-tts-weights
diag(){ # diag <ckpt dir> <label>
  ONLY_OURS=1 SARA_ADAPTER_OURS=$1 TEST_DIR=$E/obj319_routed TEST_SET=obj319 ITEM_SPK_JSON=$E/speaker_assignment42_obj319.json ITEM_SPK_CELL=OBJ319 \
    EVAL_SPEAKERS_JSON=$E/selected_42_pod.json BLOCK_OUT=$O/diag_$2 python /workspace/bundle/scripts/blockA_sarashina_difficult_4spk.py --nspk 42 --limit 100 > $O/diag_$2.log 2>&1
  python - "$O/diag_$2" "$N $2" <<'PY' | tee -a /workspace/RESULTS.txt
import json, sys, collections
d = json.load(open(sys.argv[1] + "/guard_takes.json")); o = [x for x in d if x.get("arm") == "ours"]
print(f"[hyp] {sys.argv[2]}: tag items {len(o)} | needed a redraw {sum(x['takes'] > 1 for x in o)} | exhausted the six takes {sum(x.get('final_reason') == 'runaway' for x in o)} | reasons {dict(collections.Counter(r for x in o for r in x['rejected']))}")
PY
}
TRAIN="python /workspace/bundle/scripts/sarashina/train_lora_sarashina.py --model-dir $W --out $O/lora --steps 20000 --alpha 64"
if [ "$N" = P3_registered_early ]; then
  $TRAIN --teachers /workspace/bundle/teachers/teachers_registered.jsonl --save-every 1000 > $O/train.log 2>&1 &
  TP=$!; until [ -f $O/lora/ckpt_3000/pron_emb.pt ]; do sleep 10; done; sleep 5; kill $TP; sleep 10
  for s in 1000 2000 3000; do diag $O/lora/ckpt_$s step$s; done
elif [ -n "${WAIT_PID:-}" ]; then   # a trainer already runs on this pod (its chain was lost): wait for it, then read its adapter
  while kill -0 $WAIT_PID 2>/dev/null; do sleep 20; done
  diag $O/lora/ckpt_20000 step20000; diag $O/lora/ckpt_4000 step4000
else
  $TRAIN --teachers /workspace/bundle/teachers/teachers_$N.jsonl --save-every 4000 > $O/train.log 2>&1
  diag $O/lora/ckpt_20000 step20000; diag $O/lora/ckpt_4000 step4000
fi
echo "RUN_DONE"
