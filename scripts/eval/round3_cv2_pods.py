#!/usr/bin/env python3
"""Round 3 CosyVoice 2 (section 16 as restored 10:50: section-12 carriers only, slow-teacher filter), on pods:
1. wait for the section-12 shards: cv4-0/1/2/3/5/6/7 first runs (shards 0-3, 5-7 of 8) and the four quarter runs of
   shard 4 (cv4-0/1/5/7, SHARD 4/12/20/28 of 32, out_s4); collect (collect_v4.py, virtual state), keep rows whose
   text is a section-12 carrier, apply the filter -> $V/r3/cv2/manifest_filtered.tsv (local paths);
2. on one CosyVoice 2 pod: upload the token files and a pod-pathed manifest, train four budgets (250/500/1000/2000
   steps, registered 1000), then read every adapter on both validation sets (base arms once) with the eval kit;
3. pull adapters to $V/r3/cv2/lora_s<steps>/ and validation outputs to $V/r3val/cv2/<set>/s<steps>/ (base: base/)."""
import json, os, re, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/, for paths.py
import paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import round3_podkit
V = f"{paths.DATA}/v2"; R = paths.REPO
SSH = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=15"]
LOGS = open(f"{V}/logs/cv2v4_par.log").read()
def sh(ip, port, cmd, timeout=300):
    return subprocess.run(["ssh"] + SSH + ["-p", str(port), f"root@{ip}", cmd], capture_output=True, text=True, timeout=timeout).stdout.strip()
def addr(p):
    if p.get("ip"): return p["ip"], int(p["port"])
    m = re.search(re.escape(p["name"]) + r" (\S+):(\d+)", LOGS); return (m.group(1), int(m.group(2))) if m else (None, None)
say = lambda *a: print(time.strftime("%T"), *a, flush=True)
st = json.load(open(f"{V}/cv2v4/runpod_pods.json")); virt = []
SKIP = os.path.exists(f"{V}/r3/cv2/TEACHERS_READY")   # teachers already collected and filtered: go straight to training
for p in st["pods"]:
    if p.get("stage") == "terminated": continue
    ip, port = addr(p)
    if not ip: continue
    virt.append({"name": p["name"], "ip": ip, "port": port, "shard": p.get("shard"), "log": "/workspace/pod.log", "out": "/workspace/out/cv2/teachers"})
    if p.get("s4_shard"): virt.append({"name": p["name"] + "_s4", "ip": ip, "port": port, "shard": p["s4_shard"], "log": "/workspace/pod_s4.log", "out": "/workspace/out_s4/cv2/teachers"})
OV = f"{V}/cv2v4/shard_overrides.json"   # a shard re-run elsewhere (pod too slow): {"<pod name>": [entries]}
if os.path.exists(OV):
    ov = json.load(open(OV)); virt = [q for q in virt if q["name"] not in ov]
    for name, entries in ov.items(): virt += entries; say("override:", name, "->", [e["name"] for e in entries])
say("cv2 section-12 pieces:", [q["name"] for q in virt])
while not SKIP:
    nd = []
    for q in virt:
        try: d = sh(q["ip"], q["port"], f"grep -c RUN_DONE {q['log']}")
        except Exception: d = "?"
        if d != "1": nd.append(q["name"])
    say("done", len(virt) - len(nd), "of", len(virt), "| waiting on", nd)
    if not nd: break
    time.sleep(90)
if not SKIP:
  os.makedirs(f"{V}/r3col_cv2_v4", exist_ok=True); json.dump({"pods": virt}, open(f"{V}/r3col_cv2_v4/runpod_pods.json", "w"), indent=1)
  env = dict(os.environ, JOB="cv2v4", STATE_OVERRIDE=f"{V}/r3col_cv2_v4/runpod_pods.json", OUT_OVERRIDE=f"{V}/r3col_cv2_v4")
  r = subprocess.run(["python3", "scripts/teachers/collect_v4.py"], cwd=R, env=env, capture_output=True, text=True); say(r.stdout[-600:], r.stderr[-400:])
  v4 = {json.loads(l)["phon"] for l in open(f"{V}/teachers_text_v4_new.jsonl", encoding="utf-8") if l.strip()} | {json.loads(l)["phon"] for l in open(f"{V}/teachers_text_v3.jsonl", encoding="utf-8") if l.strip()}
  rows = [l for l in open(f"{V}/r3col_cv2_v4/teachers/manifest_all_v4.tsv", encoding="utf-8") if l.strip()]
  keep = [l for l in rows if l.split("\t")[1] in v4]; say("manifest rows", len(rows), "section-12 rows", len(keep))
  os.makedirs(f"{V}/r3/cv2", exist_ok=True); open(f"{V}/r3/cv2/manifest_v4only.tsv", "w", encoding="utf-8").writelines(keep)
  r = subprocess.run(["python3", "scripts/eval/round3_filter_teachers.py", "cv2", f"{V}/r3/cv2/manifest_v4only.tsv", f"{V}/r3/cv2/manifest_filtered.tsv"], cwd=R, capture_output=True, text=True); say(r.stdout, r.stderr[-300:])
  open(f"{V}/r3/cv2/TEACHERS_READY", "w").write(time.strftime("%F %T"))
# 2. training + validation pod: cv4-0 once its quarter run is done (it is, by now)
p = [q for q in st["pods"] if q["name"] == "cv4-0"][0]; ip, port = addr(p); say("cv2 training on cv4-0")
# everything the pod runs, staged in the pod's layout (scripts/eval/round3_podkit.py) and shipped with one tar
man = [l.rstrip("\n").split("\t") for l in open(f"{V}/r3/cv2/manifest_filtered.tsv", encoding="utf-8") if l.strip()]
KIT = f"{V}/r3/cv2/podkit"
round3_podkit.stage_cv2(KIT, man, voice_kit=f"{V}/runpod_bundle_t5gsynth/eval", fictional_wav=f"{paths.DATA}/bootstrap_speakers/fictional0.wav",
                        budgets=(250, 500, 1000, 2000, 4000, 8000, 16000))
subprocess.run(f"tar cz -C {KIT} . | ssh {' '.join(SSH)} -p {port} root@{ip} 'tar xz -C /workspace --no-same-owner'", shell=True, check=True)
sh(ip, port, "(setsid nohup bash /workspace/r3/chain.sh < /dev/null > /workspace/r3/chain.log 2>&1 &)"); say("cv2 train+validate chain started on cv4-0")
for _ in range(240):
    try:
        if sh(ip, port, "grep -c CV2_R3_DONE /workspace/r3/done.txt 2>/dev/null") == "1": break
    except Exception: pass
    time.sleep(60)
else: say("cv2 chain did not finish"); raise SystemExit(1)
os.makedirs(f"{V}/r3val/cv2", exist_ok=True)
subprocess.run(f"ssh {' '.join(SSH)} -p {port} root@{ip} 'cd /workspace/r3 && tar cz lora_s* train_s*.log val_*.log' | tar xz -C {V}/r3/cv2 --no-same-owner", shell=True)
subprocess.run(f"ssh {' '.join(SSH)} -p {port} root@{ip} 'cd /workspace/r3val && tar cz .' | tar xz -C {V}/r3val/cv2 --no-same-owner", shell=True)
open(f"{V}/r3/cv2/VAL_SWEEP_DONE", "w").write(time.strftime("%F %T")); say("CV2_VAL_SWEEP_DONE")
