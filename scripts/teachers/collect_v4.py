#!/usr/bin/env python3
"""Collect the extended-text (v4) teacher shards from RunPod pods and merge them with the registered round-2 sets.

    JOB=t5gv4|irov4|cv2v4 python3 scripts/teachers/collect_v4.py [--partial]

Never writes into the round-2 directories. Outputs:
  t5gv4 -> V/t5gemma_v4/teachers_new.jsonl, teachers_all_v4.jsonl (= round-2 teachers_generic + new)
  irov4 -> V/irodori_v4/train_new.jsonl,   train_all_v4.jsonl    (= round-2 train.jsonl + new), latent paths local
  cv2v4 -> V/cv2_v4/teachers/manifest_new.tsv, manifest_all_v4.tsv (= round-2 manifest + new), tok/wav paths local
each with a provenance JSON (pods, shards, kept, dropped).
"""
import argparse, json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/, for paths.py
import paths
V = f"{paths.DATA}/v2"; JOB = os.environ["JOB"]
SSH = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=20"]
ap = argparse.ArgumentParser(); ap.add_argument("--partial", action="store_true"); a = ap.parse_args()
pods = [p for p in json.load(open(os.environ.get("STATE_OVERRIDE", f"{V}/{JOB}/runpod_pods.json")))["pods"] if p.get("ip")]
REMOTE = {"t5gv4": "/workspace/out/t5gemma", "irov4": "/workspace/out/irodori", "cv2v4": "/workspace/out/cv2/teachers"}[JOB]
OUT = os.environ.get("OUT_OVERRIDE") or {"t5gv4": f"{V}/t5gemma_v4", "irov4": f"{V}/irodori_v4", "cv2v4": f"{V}/cv2_v4"}[JOB]; os.makedirs(f"{OUT}/runpod", exist_ok=True)
got, prov = [], []
for p in pods:
    log = p.get("log", "/workspace/pod.log")
    r = subprocess.run(SSH + ["-p", str(p["port"]), f"root@{p['ip']}", f"grep -c RUN_DONE {log}"], capture_output=True, text=True)
    if r.stdout.strip() != "1":
        print(f"  {p['name']} (shard {p.get('shard')}): not done");
        if not a.partial: sys.exit(1)
        continue
    d = f"{OUT}/runpod/{p['name']}"; os.makedirs(d, exist_ok=True)
    src = p.get("out", REMOTE)
    subprocess.run(" ".join(SSH + ["-p", str(p["port"]), f"root@{p['ip']}", f"'tar cz -C {src} .'"]) + f" | tar xz -C {d} --no-same-owner", shell=True, check=True)
    got.append((p, d)); print(f"  pulled {p['name']} shard {p.get('shard')}")
text = [json.loads(l) for l in open(f"{V}/teachers_text_v4_new.jsonl", encoding="utf-8")]
if JOB == "t5gv4":
    rows = []
    for p, d in got:
        for k in range(1, 6):
            f = f"{d}/pass{k}/teachers.jsonl"
            if os.path.exists(f): rows += [l for l in open(f, encoding="utf-8") if l.strip()]
            pf = f + ".provenance.json"
            if os.path.exists(pf): prov.append({"pod": p["name"], "shard": p["shard"], "pass": k, **{kk: vv for kk, vv in json.load(open(pf)).items() if kk != "items"}})
    open(f"{OUT}/teachers_new.jsonl", "w", encoding="utf-8").writelines(rows)
    old = [l for l in open(f"{V}/t5gemma/teachers_generic/teachers.jsonl", encoding="utf-8") if l.strip()]
    open(f"{OUT}/teachers_all_v4.jsonl", "w", encoding="utf-8").writelines(old + rows); n_new, n_old = len(rows), len(old)
elif JOB == "irov4":
    REF = f"{paths.IRODORI_DIR}/data/uttertune_anchor_bank/bootstrap_refs/"   # in the Irodori code of this repo
    rows = []
    for p, d in got:
        for l in open(f"{d}/train.jsonl", encoding="utf-8"):
            if not l.strip(): continue
            j = json.loads(l); j["latent_path"] = f"{d}/latents/" + os.path.basename(j["latent_path"]); j["ref_latent_path"] = REF + os.path.basename(j["ref_latent_path"])
            assert os.path.exists(j["latent_path"]), j["latent_path"]; rows.append(j)
        pf = f"{d}/train.jsonl.provenance.json"
        if os.path.exists(pf): prov.append({"pod": p["name"], "shard": p["shard"], **{k: v for k, v in json.load(open(pf)).items() if k != "items"}})
    order = {(t["word"], t["plain"]): i for i, t in enumerate(text)}; rows.sort(key=lambda j: order.get((j["target_surface"], j["original_text"]), 10**9))
    with open(f"{OUT}/train_new.jsonl", "w", encoding="utf-8") as f:
        for j in rows: f.write(json.dumps(j, ensure_ascii=False) + "\n")
    old = [l for l in open(f"{V}/irodori/train.jsonl", encoding="utf-8") if l.strip()]
    with open(f"{OUT}/train_all_v4.jsonl", "w", encoding="utf-8") as f:
        f.writelines(old); [f.write(json.dumps(j, ensure_ascii=False) + "\n") for j in rows]
    n_new, n_old = len(rows), len(old)
else:
    rows = []; os.makedirs(f"{OUT}/teachers", exist_ok=True)
    for p, d in got:
        for l in open(f"{d}/manifest.tsv", encoding="utf-8"):
            if not l.strip(): continue
            spk, txt, tp, wp = l.rstrip("\n").split("\t"); tp = f"{d}/tok/" + os.path.basename(tp); wp = f"{d}/wav/" + os.path.basename(wp)
            assert os.path.exists(tp), tp; rows.append((spk, txt, tp, wp))
        pf = f"{d}/provenance.json"
        if os.path.exists(pf): prov.append({"pod": p["name"], "shard": p["shard"], **{k: v for k, v in json.load(open(pf, encoding="utf-8")).items() if k != "items"}})
    with open(f"{OUT}/teachers/manifest_new.tsv", "w", encoding="utf-8") as f:
        for r in rows: f.write("\t".join(r) + "\n")
    old = [l for l in open(f"{V}/cv2/teachers/manifest.tsv", encoding="utf-8") if l.strip()]
    with open(f"{OUT}/teachers/manifest_all_v4.tsv", "w", encoding="utf-8") as f:
        f.writelines(old); [f.write("\t".join(r) + "\n") for r in rows]
    n_new, n_old = len(rows), len(old)
json.dump({"source": f"RunPod pods, JOB={JOB}, text teachers_text_v4_new.jsonl ({len(text)} carriers)", "pods": [p for p, _ in got], "shards": prov,
           "new_kept": n_new, "registered_round2": n_old, "total": n_new + n_old}, open(f"{OUT}/provenance_v4.json", "w"), indent=1, ensure_ascii=False)
print(f"{JOB}: new {n_new} + registered {n_old} = {n_new + n_old} -> {OUT}")
