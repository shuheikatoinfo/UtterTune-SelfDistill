"""What the round-3 pods run, staged locally in the pod's own layout.

A kit is a local directory laid out exactly as /workspace on the pod; the orchestration scripts
(round3_cv2_pods.py, round3_t5gemma_pods.py) ship it with one tar and start the chain command
written for it. Everything of this repository that the pod runs is in the kit, so a fresh pod needs
only what its image and setup provide:
  CosyVoice 2: /workspace/CosyVoice (code, on PYTHONPATH), /workspace/cv2model (CosyVoice2-0.5B),
               a python with CosyVoice's requirements on PATH.
  T5Gemma:     /workspace/T5Gemma-TTS with .venv_t5gemma and pretrained/T5Gemma-TTS-2b-2b.

CosyVoice 2 kit:
  repo/       the scripts the chain runs, at their repository paths, so imports resolve as here
  data/       DATA on the pod: eval_speakers/selected_42.json (wav paths point into eval/spk/) and
              bootstrap_speakers/fictional0.wav (cv_synth's default prompt, prepared at start-up)
  eval/       validation item sets, speaker assignments, the 42 voice clips (spk/)
  r3/         chain.sh, manifest_pod.tsv and the teacher token files it names (tok/<pull dir>/<file>)
T5Gemma kit:
  r3/         materialize_dataset.py, teachers_filtered.jsonl, blockA_readlora_train.sh, chain.sh
"""
import os, shutil

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WS = "/workspace"

CV2_REPO_FILES = (
    "scripts/paths.py",
    "scripts/cv/train_lora_cv2.py",
    "scripts/cv/blockA_cv2_difficult.py",
    "scripts/cv/cv_synth.py",
    "scripts/cv/cv_quality.py",
    "scripts/eval/degenerate.py",
)
# validation sets read on the pod: (name, directory under data/derived, speaker-assignment cell)
CV2_VAL_SETS = (("sel105", "sel105_routed", "SEL105"), ("deveasy", "deveasy", "DEVEASY"))


def _place(src, dst):
    """Hard link when possible (token files are many and large), else copy."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _place_tree(src, dst):
    for d, _, fs in os.walk(src):
        for f in fs:
            _place(os.path.join(d, f), os.path.join(dst, os.path.relpath(os.path.join(d, f), src)))


def stage_cv2(kit, manifest_rows, voice_kit, fictional_wav, budgets):
    """Build the CosyVoice 2 kit. Its r3/chain.sh trains one adapter per step budget, then reads every
    adapter on both validation sets (the base arms with the first budget only), and finally appends
    CV2_R3_DONE to /workspace/r3/done.txt.

    manifest_rows: (speaker, phon_text, local token path, wav) rows of the filtered teacher manifest.
    voice_kit: directory holding selected_42.json (wav paths under /workspace/eval/spk/) and spk/.
    """
    shutil.rmtree(kit, ignore_errors=True)
    for f in CV2_REPO_FILES:
        _place(os.path.join(REPO, f), os.path.join(kit, "repo", f))
    _place(os.path.join(voice_kit, "selected_42.json"), os.path.join(kit, "data/eval_speakers/selected_42.json"))
    _place(fictional_wav, os.path.join(kit, "data/bootstrap_speakers/fictional0.wav"))
    _place_tree(os.path.join(voice_kit, "spk"), os.path.join(kit, "eval/spk"))
    for name, d, _ in CV2_VAL_SETS:
        _place_tree(os.path.join(REPO, "data/derived", d), os.path.join(kit, "eval", d))
        f = f"speaker_assignment42_{name}.json"
        _place(os.path.join(REPO, "preregistration/items", f), os.path.join(kit, "eval", f))
    pod_rows = []
    for spk, txt, tp, wp in manifest_rows:
        # the token file's pull directory: a pulled shard dir, or the round-2 pod dir (cv2_runpod/<pod>/tok)
        sub = tp.split("/runpod/")[1].split("/")[0] if "/runpod/" in tp else os.path.basename(os.path.dirname(os.path.dirname(tp)))
        rel = f"r3/tok/{sub}/{os.path.basename(tp)}"
        if not os.path.exists(os.path.join(kit, rel)):
            _place(tp, os.path.join(kit, rel))
        pod_rows.append("\t".join([spk, txt, f"{WS}/{rel}", wp]))
    with open(os.path.join(kit, "r3/manifest_pod.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(pod_rows) + "\n")
    env = (f"export DATA={WS}/data COSYVOICE_DIR={WS}/CosyVoice CV2_DIR={WS}/cv2model "
           f"PYTHONPATH={WS}/CosyVoice:{WS}/CosyVoice/third_party/Matcha-TTS "
           "HUGGING_FACE_HUB_TOKEN=$(tr '\\0' '\\n' < /proc/1/environ | grep ^HF_TOKEN= | cut -d= -f2-) "
           f"&& cd {WS}/repo")
    cmd = env
    for s_ in budgets:
        cmd += (f" && python scripts/cv/train_lora_cv2.py --manifest {WS}/r3/manifest_pod.tsv --base {WS}/cv2model"
                f" --out {WS}/r3/lora_s{s_} --steps {s_} > {WS}/r3/train_s{s_}.log 2>&1")
    first = True
    for s_ in budgets:
        for name, d, cell in CV2_VAL_SETS:
            arms = "noedit,kana,ours" if first else "ours"
            cmd += (f" && ARMS={arms} CV2_LORA={WS}/r3/lora_s{s_} TEST_DIR={WS}/eval/{d} TEST_SET={name}"
                    f" ITEM_SPK_JSON={WS}/eval/speaker_assignment42_{name}.json ITEM_SPK_CELL={cell}"
                    f" BLOCKA_CV2_OUT={WS}/r3val/{name}/s{s_} python scripts/cv/blockA_cv2_difficult.py"
                    f" > {WS}/r3/val_{name}_s{s_}.log 2>&1")
        first = False
    with open(os.path.join(kit, "r3/chain.sh"), "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n" + cmd + f" && echo CV2_R3_DONE >> {WS}/r3/done.txt\n")


def stage_t5gemma(kit, teachers, grad_ckpt, rundir, num_steps, save_every, batch):
    """Build the T5Gemma kit: materialize the teacher set, then LoRA-train with checkpoints to
    /workspace/r3/<rundir>/bundle_step<N>.pth. grad_ckpt "0" turns activation recomputation off."""
    shutil.rmtree(kit, ignore_errors=True)
    _place(os.path.join(REPO, "scripts/t5gemma/materialize_dataset.py"), os.path.join(kit, "r3/materialize_dataset.py"))
    _place(teachers, os.path.join(kit, "r3/teachers_filtered.jsonl"))
    train = open(os.path.join(REPO, "scripts/t5gemma/blockA_readlora_train.sh"), encoding="utf-8").read()
    assert train.count("--t5_gradient_checkpointing 1") == 1
    with open(os.path.join(kit, "r3/blockA_readlora_train.sh"), "w", encoding="utf-8") as f:
        f.write(train.replace("--t5_gradient_checkpointing 1", f"--t5_gradient_checkpointing {grad_ckpt}"))
    rd = f"{WS}/r3/{rundir}"
    chain = (f"cd {WS}/T5Gemma-TTS && . .venv_t5gemma/bin/activate && export T5GEMMA_DIR={WS}/T5Gemma-TTS"
             " && export HF_TOKEN=$(tr '\\0' '\\n' < /proc/1/environ | grep ^HF_TOKEN= | cut -d= -f2-)"
             " && export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN HF_HUB_ENABLE_HF_TRANSFER=0"
             f" && python {WS}/r3/materialize_dataset.py --teachers {WS}/r3/teachers_filtered.jsonl --out {WS}/r3/dataset > {WS}/r3/mat.log 2>&1"
             f" && DATASET_DIR={WS}/r3/dataset EXP_ROOT={rd} NUM_STEPS={num_steps} SAVE_EVERY={save_every} BATCH_SIZE={batch}"
             f" bash {WS}/r3/blockA_readlora_train.sh > {WS}/r3/train.log 2>&1; echo T5G_TRAIN_DONE >> {WS}/r3/train.log")
    with open(os.path.join(kit, "r3/chain.sh"), "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n" + chain + "\n")
    return rd
