#!/usr/bin/env python3
"""Build the naturalness A/B site: clips, a group split, and the page.

Ships nothing that identifies which side is the edited rendition -- the A/B mapping lives only in
preregistration/items/naturalness_ab_easy50.json, which is not copied here. The 143 pairs are split
into thirds so one sitting is about twenty minutes; a rater judges one third.
"""
import argparse, hashlib, json, os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/, for paths.py
import paths

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--items", default="preregistration/items/naturalness_ab_easy50.json")
ap.add_argument("--clips", default=f"{paths.DATA}/naturalness_kit_norm",
                help="loudness-equalised copies; a paired preference is otherwise decided by volume")
ap.add_argument("--out", default=f"{paths.DATA}/cf_nat")
ap.add_argument("--kv-id", default=os.environ.get("CF_KV_NAMESPACE"), help="default: $CF_KV_NAMESPACE")
ap.add_argument("--project", default="jp-accent-naturalness")
ap.add_argument("--groups", type=int, default=3)
ap.add_argument("--fee", default="1,000円")
ap.add_argument("--closed", action="store_true",
                help="recruitment is over: turn away anyone with no answers stored, so nobody "
                     "starts 30 minutes of work that no slot can pay for. Raters who already "
                     "have answers still get in, because they still need to re-sync them.")
a = ap.parse_args()
if not a.kv_id:
    sys.exit("set CF_KV_NAMESPACE or pass --kv-id")

d = json.load(open(os.path.join(REPO, a.items), encoding="utf-8"))
items = d["items"]
PUB = os.path.join(a.out, "public")
os.makedirs(os.path.join(PUB, "clips"), exist_ok=True)

# Groups are content-derived, and each group keeps every backbone in proportion, so a rater who
# only judges one third still sees all four systems.
by_bb = {}
for it in items:
    by_bb.setdefault(it["backbone"], []).append(it["id"])
groups = {str(g): [] for g in range(a.groups)}
for bb, ids in by_bb.items():
    for i, pid in enumerate(sorted(ids, key=lambda p: hashlib.sha256(("G:" + p).encode()).hexdigest())):
        groups[str(i % a.groups)].append(pid)
for g in groups:
    groups[g].sort(key=lambda p: hashlib.sha256(("ORD:" + g + ":" + p).encode()).hexdigest())

n = 0
for it in items:
    for side in ("A", "B"):
        src = os.path.join(a.clips, f"{it['id']}{side}.wav")
        shutil.copy2(src, os.path.join(PUB, "clips", f"{it['id']}{side}.wav")); n += 1
json.dump({"n_pairs": len(items), "groups": groups, "closed": a.closed,
           "per_group": min(len(v) for v in groups.values()), "fee": a.fee},
          open(os.path.join(PUB, "index.json"), "w", encoding="utf-8"), ensure_ascii=False)
page = open(os.path.join(REPO, "scripts/rater_ui/nat/index.html"), encoding="utf-8").read()
# An unclosed <script> leaves Chrome with a page that renders and does nothing: no error, no
# content, nothing in the console to photograph. It cost a dry run to find, so it is checked here.
assert page.count("<script>") == page.count("</script>"), \
    f"REFUSING TO SHIP: {page.count('<script>')} <script> vs {page.count('</script>')} </script>"
open(os.path.join(PUB, "index.html"), "w", encoding="utf-8").write(page)
with open(os.path.join(PUB, "_headers"), "w") as f:
    f.write("/\n  Cache-Control: no-store\n/index.html\n  Cache-Control: no-store\n"
            "/index.json\n  Cache-Control: no-store\n/clips/*\n  Cache-Control: public, max-age=604800\n")
shutil.copytree(os.path.join(REPO, "scripts/rater_ui/functions"),
                os.path.join(a.out, "functions"), dirs_exist_ok=True)
with open(os.path.join(a.out, "wrangler.toml"), "w") as f:
    f.write(f'name = "{a.project}"\ncompatibility_date = "2026-01-01"\npages_build_output_dir = "public"\n'
            f'\n[[kv_namespaces]]\nbinding = "RATER"\nid = "{a.kv_id}"\n')

# The side mapping must never reach the browser.
shipped = json.dumps(json.load(open(os.path.join(PUB, "index.json"), encoding="utf-8")))
assert '"ours"' not in shipped and '"noedit"' not in shipped, "REFUSING TO SHIP: arm labels in index.json"
print(f"  pairs {len(items)} / clips {n} / groups " +
      ", ".join(f"{g}={len(v)}" for g, v in sorted(groups.items())))
print(f"OK -> {a.out}")
