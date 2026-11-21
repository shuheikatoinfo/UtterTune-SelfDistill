#!/usr/bin/env python3
"""Round 3 T5Gemma (section 16 as restored 10:50: section-12 carriers only, slow-teacher filter), everything on pods:
1. wait until every t5gv4 pod's FIRST run (section-12 shard) printed RUN_DONE, collect those 32 shards only
   (collect_v4.py with a virtual state), apply the filter -> $V/r3/t5gemma/teachers_filtered.jsonl;
2. train on one free T5Gemma pod (repo, venv and base model already there): materialize -> LoRA train, NUM_STEPS=800,
   SAVE_EVERY=200; the pod is marked BUSY_TRAIN so the validation watcher leaves it alone;
3. pull every bundle_stepN.pth to $V/r3/t5gemma/readlora/ as soon as it is written (the watcher t5g_pod_val.py
   dispatches each to a free pod for the validation reads)."""
import json, os, re, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/, for paths.py
import paths
V = f"{paths.DATA}/v2"; R = paths.REPO
SSH = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=15"]
LOGS = "".join(open(f"{V}/logs/{f}").read() for f in os.listdir(f"{V}/logs") if (f.startswith("t5gv") or f.startswith("t5gh")) and f.endswith(".log"))
def sh(ip, port, cmd, timeout=180):
    return subprocess.run(["ssh"] + SSH + ["-p", str(port), f"root@{ip}", cmd], capture_output=True, text=True, timeout=timeout).stdout.strip()
def addr(p):
    if p.get("ip"): return p["ip"], int(p["port"])
    m = re.search(re.escape(p["name"]) + r" (\S+):(\d+)", LOGS); return (m.group(1), int(m.group(2))) if m else (None, None)
say = lambda *a: print(time.strftime("%T"), *a, flush=True)
NUM_STEPS = os.environ.get("NUM_STEPS", "1600"); BATCH = os.environ.get("BATCH_SIZE", "4"); SAVE_EVERY = os.environ.get("SAVE_EVERY", "200"); RUNDIR = os.environ.get("RUNDIR", "readlora"); N_EXP = int(NUM_STEPS) // int(SAVE_EVERY)
TEACH = f"{V}/r3/t5gemma/teachers_filtered.jsonl"
# 1. section-12 shards (a second instance waits for the first one's teacher file instead of collecting again)
SKIP = (os.environ.get("RUNDIR") and os.environ.get("RUNDIR") != "readlora") or os.path.exists(f"{V}/r3/t5gemma/TEACHERS_READY")
if SKIP:
    while not os.path.exists(f"{V}/r3/t5gemma/TEACHERS_READY"): time.sleep(60)
    say("teachers ready (already collected)")
virt = None if SKIP else []
for p in (json.load(open(f"{V}/t5gv4/runpod_pods.json"))["pods"] if virt is not None else []):
    if p.get("stage") == "terminated": continue
    ip, port = addr(p)
    if ip: virt.append({"name": p["name"], "ip": ip, "port": port, "shard": p.get("shard"), "log": "/workspace/pod.log", "out": "/workspace/out/t5gemma"})
# a shard re-run elsewhere (pod never ran, or too slow) is listed in shard_overrides.json: {"<pod name>": [entries]}
OV = f"{V}/t5gv4/shard_overrides.json"
if virt is not None and os.path.exists(OV):
    ov = json.load(open(OV)); virt = [q for q in virt if q["name"] not in ov]
    for name, entries in ov.items(): virt += entries; say("override:", name, "->", [e["name"] for e in entries])
say("t5gemma section-12 shards on", len(virt) if virt is not None else "(skipped)", "pods")
while virt is not None:
    nd = []
    for q in virt:
        try: d = sh(q["ip"], q["port"], f"grep -c RUN_DONE {q['log']}")
        except Exception: d = "?"
        if d != "1": nd.append(q["name"])
    say("first runs done", len(virt) - len(nd), "of", len(virt), "| waiting on", nd[:10])
    if not nd: break
    time.sleep(90)
if virt is not None:
  os.makedirs(f"{V}/r3col_t5gemma_v4", exist_ok=True); json.dump({"pods": virt}, open(f"{V}/r3col_t5gemma_v4/runpod_pods.json", "w"), indent=1)
  env = dict(os.environ, JOB="t5gv4", STATE_OVERRIDE=f"{V}/r3col_t5gemma_v4/runpod_pods.json", OUT_OVERRIDE=f"{V}/r3col_t5gemma_v4")
  r = subprocess.run(["python3", "scripts/teachers/collect_v4.py"], cwd=R, env=env, capture_output=True, text=True); say(r.stdout[-600:], r.stderr[-400:])
  os.makedirs(f"{V}/r3/t5gemma", exist_ok=True)
  r = subprocess.run(["python3", "scripts/eval/round3_filter_teachers.py", "t5gemma", f"{V}/r3col_t5gemma_v4/teachers_all_v4.jsonl", f"{V}/r3/t5gemma/teachers_filtered.jsonl"], cwd=R, capture_output=True, text=True); say(r.stdout, r.stderr[-300:])
  open(f"{V}/r3/t5gemma/TEACHERS_READY", "w").write(time.strftime("%F %T"))
# 2. training pod: a finished t5gv5 pod (no second run), not busy
pod = None
while not pod:
    for p in [q for j in (("t5gh",) if os.environ.get("FAST_ONLY") else ("t5gh", "t5gv5")) if os.path.exists(f"{V}/{j}/runpod_pods.json") for q in json.load(open(f"{V}/{j}/runpod_pods.json"))["pods"]]:   # fast GPUs first
        ip, port = addr(p)
        if not ip: continue
        try:
            if sh(ip, port, "grep -c RUN_DONE /workspace/pod.log") == "1" and sh(ip, port, "ps -eo args | grep -c '[b]lockA_t5gemma\\|[e]xport_readlora\\|[t]rain_readlora'") == "0" \
               and sh(ip, port, "mkdir /workspace/BUSY_TRAIN.d 2>/dev/null && touch /workspace/BUSY_TRAIN && echo claimed") == "claimed":   # atomic claim: two instances never share a pod
                pod = (p["name"], ip, port); break
        except Exception: continue
    if not pod: time.sleep(60)
name, ip, port = pod; say("training on", name)
sh(ip, port, "touch /workspace/BUSY_TRAIN; mkdir -p /workspace/r3 /workspace/scripts"); RD = f"/workspace/r3/{RUNDIR}"
os.makedirs(f"{V}/t5g_trainkit", exist_ok=True)
GC = os.environ.get("T5_GRADCKPT", "1")   # 0 on an 80 GB GPU: no activation recomputation, about 2x faster per step
subprocess.run(f"sed -e 's#^REPO=.*#REPO=/workspace/T5Gemma-TTS#' -e 's#--t5_gradient_checkpointing 1#--t5_gradient_checkpointing {GC}#' {R}/scripts/t5gemma/blockA_readlora_train.sh > {V}/t5g_trainkit/blockA_readlora_train_{RUNDIR}.sh && cp {R}/scripts/t5gemma/materialize_dataset.py {V}/t5g_trainkit/", shell=True, check=True)
subprocess.run(["scp", "-q"] + SSH + ["-P", str(port), f"{V}/t5g_trainkit/materialize_dataset.py", f"{V}/r3/t5gemma/teachers_filtered.jsonl", f"root@{ip}:/workspace/r3/"], capture_output=True, timeout=1800)
subprocess.run(["scp", "-q"] + SSH + ["-P", str(port), f"{V}/t5g_trainkit/blockA_readlora_train_{RUNDIR}.sh", f"root@{ip}:/workspace/r3/blockA_readlora_train.sh"], capture_output=True, timeout=600)
cmd = ("cd /workspace/T5Gemma-TTS && . .venv_t5gemma/bin/activate && export HF_TOKEN=$(tr '\\0' '\\n' < /proc/1/environ | grep ^HF_TOKEN= | cut -d= -f2-) && export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN HF_HUB_ENABLE_HF_TRANSFER=0 && python /workspace/r3/materialize_dataset.py --teachers /workspace/r3/teachers_filtered.jsonl --out /workspace/r3/dataset > /workspace/r3/mat.log 2>&1 && "
       f"DATASET_DIR=/workspace/r3/dataset EXP_ROOT={RD} NUM_STEPS={NUM_STEPS} SAVE_EVERY={SAVE_EVERY} BATCH_SIZE={BATCH} bash /workspace/r3/blockA_readlora_train.sh > /workspace/r3/train.log 2>&1; echo T5G_TRAIN_DONE >> /workspace/r3/train.log")
open(f"{V}/t5g_trainkit/chain_{RUNDIR}.sh", "w").write("#!/usr/bin/env bash\n" + cmd + "\n")
subprocess.run(["scp", "-q"] + SSH + ["-P", str(port), f"{V}/t5g_trainkit/chain_{RUNDIR}.sh", f"root@{ip}:/workspace/r3/chain.sh"], capture_output=True, timeout=120)
sh(ip, port, "cd /workspace && (setsid nohup bash /workspace/r3/chain.sh < /dev/null > /workspace/r3/chain.log 2>&1 &)"); say("materialize+train started on", name)
# 3. pull bundles as they appear
os.makedirs(f"{V}/r3/t5gemma/{RUNDIR}", exist_ok=True); got = set(); t0 = time.time()
while len(got) < N_EXP and time.time() - t0 < 4 * 3600:
    try: ls = sh(ip, port, f"ls {RD}/bundle_step*.pth 2>/dev/null; grep -c T5G_TRAIN_DONE /workspace/r3/train.log 2>/dev/null; tail -c 200 /workspace/r3/train.log | tr '\\r' '\\n' | tail -1")
    except Exception: ls = ""
    for f in [l for l in ls.split("\n") if l.endswith(".pth")]:
        st = int(re.search(r"step(\d+)", f).group(1))
        if st in got: continue
        sz1 = sh(ip, port, f"stat -c %s {f}"); time.sleep(30); sz2 = sh(ip, port, f"stat -c %s {f}")
        if sz1 != sz2: continue
        subprocess.run(["scp", "-q"] + SSH + ["-P", str(port), f"root@{ip}:{f}", f"{V}/r3/t5gemma/{RUNDIR}/"], capture_output=True, timeout=3600); got.add(st); say("pulled bundle step", st)
    if len(got) < N_EXP: time.sleep(60)
say("T5G_POD_TRAIN_DONE", sorted(got))
