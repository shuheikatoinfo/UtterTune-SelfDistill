#!/usr/bin/env python3
"""Pull every stored judgment out of Cloudflare KV into one JSON file.

The site writes each judgment as it is made (functions/api/save.js) and each gate attempt
(functions/api/screen.js), so this is the authoritative record; a rater's browser copy is only
their own safety net.

Read this before deleting anything. A bulk delete run during a live study destroyed about thirty
workers' records, and there was no export to fall back on -- so `kv_purge.py` now refuses to run
without a backup newer than the keys it is about to remove.
"""
import argparse, datetime, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/, for paths.py
import paths

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cf_kv

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=f"{paths.DATA}/listening_responses.json")
ap.add_argument("--ns", help="KV namespace id (default: $CF_KV_NAMESPACE)")
ap.add_argument("--quiet", action="store_true")
a = ap.parse_args()

kv = cf_kv.KV(a.ns)
names = kv.keys()
tick = None if a.quiet else (lambda i, n: print(f"\r  {i}/{n}", end="", flush=True))
vals = kv.get_many(names, on_chunk=tick)
if tick:
    print()

# `responses` stays the top-level list analyze_listening.py reads. The rest is split out because a
# gate verdict is not a judgment and must never be counted as one.
rows = {"responses": [], "screening": [], "gate": [], "assign": [], "other": [],
        "exported_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "namespace": kv.ns, "key_count": len(names)}
for k in names:
    v = vals.get(k)
    if v is None:                       # bulk/get omits keys that vanished mid-read
        rows["other"].append({"key": k, "missing": True})
        continue
    bucket = ("screening" if k.startswith("screen:") else
              "gate" if k.startswith(("gate:", "retake:")) else
              "assign" if k.startswith("assign:") else
              "responses" if k.startswith("resp:") else "other")
    rows[bucket].append(v if isinstance(v, dict) else {"key": k, "value": v})

os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
json.dump(rows, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"  {len(rows['responses'])} judgments, {len(rows['screening'])} gate attempts, "
      f"{len(rows['gate'])} verdicts -> {a.out}")
