"""Build the static, deployable rater site from the blind kits.

Ships ONLY what a rater may see: clips, and a manifest carrying blind_id / target word /
target reading / prescribed accent. KEY_DO_NOT_OPEN.csv and build_note.json are never copied,
and the build fails loudly if anything KEY-like would end up in the output.

Judgments are POSTed to COLLECT_URL as they are made and mirrored in localStorage, so a lost
connection or a closed tab cannot cost the rater their session.

  python3 scripts/rater_ui/build_static.py --out /tmp/rater_site --collect-url https://...
"""
import argparse
import random, csv, json, os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/, for paths.py
import paths

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--kits", default=f"{paths.DATA}/blind_kits_v2")
ap.add_argument("--panel", action="store_true",
                help="one unit holding every stimulus: trained raters judge all of them, so there is "
                     "no block to assign and kappa is computable over the whole set")
ap.add_argument("--kv-id", default=os.environ.get("CF_KV_NAMESPACE"),
                help="Cloudflare KV namespace id for responses (default: $CF_KV_NAMESPACE); empty ships no functions")
ap.add_argument("--project", default="jp-accent-listening")
ap.add_argument("--screening", default="final",
                help="comma-separated screening sets to ship (v2,v4); empty ships none")
ap.add_argument("--out", required=True)
ap.add_argument("--collect-url", default="", help="endpoint that accepts one JSON judgment per POST")
args = ap.parse_args()
if args.kv_id is None:
    sys.exit("set CF_KV_NAMESPACE or pass --kv-id (an empty id ships no functions)")

FORBIDDEN = ("KEY", "build_note", "_align")
kits = sorted(d for d in os.listdir(args.kits)
              if os.path.isdir(os.path.join(args.kits, d)) and not d.startswith("_"))
if not kits:
    sys.exit(f"no kits under {args.kits}")

# Cloudflare Pages project layout. Assets go under public/; functions/ and wrangler.toml sit
# BESIDE it, at the project root. Keeping the config inside the asset directory published it --
# /wrangler.toml was fetchable, exposing the KV namespace id.
PUB = os.path.join(args.out, "public")
os.makedirs(PUB, exist_ok=True)
index = []
for k in kits:
    src, dst = os.path.join(args.kits, k), os.path.join(PUB, k)
    os.makedirs(os.path.join(dst, "clips"), exist_ok=True)
    with open(os.path.join(src, "MANIFEST.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    allowed = ("blind_id", "target_word", "target_reading", "prescribed_accent")
    items = [{a: r[a] for a in allowed} for r in rows]
    json.dump(items, open(os.path.join(dst, "manifest.json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    for r in rows:
        shutil.copy2(os.path.join(src, "clips", r["blind_id"] + ".wav"),
                     os.path.join(dst, "clips", r["blind_id"] + ".wav"))
    shutil.copy2(os.path.join(src, "README_rater.md"), os.path.join(dst, "README_rater.md"))
    index.append({"kit": k, "n": len(items)})
    print(f"  {k}: {len(items)} items")

# SCREENING (prereg S:4): 20 known-answer items, pass threshold >=90% (18/20). Delivered as part
# of the SAME job (design A): everyone who opens the URL takes it first; those who pass go straight
# on to their block and finish with a main completion code, those who fail stop there and get a
# screening completion code, so the two groups can be paid different rates without any manual
# tracking or scouting. The gold answers are NOT shipped -- only the audio and the shown accent --
# so the page cannot reveal them and a worker cannot read them out of the payload. Grading happens
# after the fact, from the recorded judgments, against preregistration/items/screening_20.json.
# Two screening sources: v2 tests the position of a word-internal downstep to the mora, v3 tests
# heiban vs odaka, which is only audible on the following particle and which v2 deliberately
# excluded for that reason. Both are needed because the main study contains both (238 heiban,
# 148 atamadaka, 57 nakadaka, 7 odaka).
def disp(a):
    """Accent in the one notation the rater is taught: ＼ = fall after that mora, ￣ = no fall."""
    if "＼" in a or "￣" in a:          # v4 items are authored in display notation already
        return a
    d = a.replace("'", "＼").replace("’", "＼")
    return d if "＼" in d else d + "￣"


ALL_SCR = {
    "v2": ("preregistration/items/screening_20.json", f"{paths.DATA}/screening_v2"),
    "v4": ("preregistration/items/screening_v4_heiban_odaka.json", f"{paths.DATA}/screening_v4"),
    # The frozen screening set: 20 items, 5 per accent type, gold 10/10, every one verified by ear
    # in the author's dry runs (scripts/eval/build_screening_final.py).
    "final": ("preregistration/items/screening_gate20.json", f"{paths.DATA}/screening_final_norm"),
}
# --screening picks which sets ship. v3 (the tagged heiban/odaka attempt) is deliberately absent:
# the contrast is realized on the following particle, outside the tagged span, so it could not be
# requested, and the author heard both members of all 8 pairs as the same type. v4 replaces it by
# never asking for an accent at all.
SCR_SOURCES = [(os.path.join(REPO, ALL_SCR[k][0]), ALL_SCR[k][1])
               for k in (args.screening.split(",") if args.screening else [])]
scr, scr_audio_of = [], {}
for src, aud in SCR_SOURCES:
    if not os.path.exists(src):
        continue
    with open(src, encoding="utf-8") as f:
        for r in json.load(f):
            scr.append(r); scr_audio_of[r["id"]] = aud
scr_src = SCR_SOURCES[0][0]
if scr:
    missing = [r["id"] for r in scr if not os.path.exists(os.path.join(scr_audio_of[r["id"]], r["id"] + ".wav"))]
    if missing:
        sys.exit(f"REFUSING TO SHIP: screening audio missing for {missing[:5]}")
    os.makedirs(os.path.join(PUB, "screening", "clips"), exist_ok=True)
    for r in scr:
        shutil.copy2(os.path.join(scr_audio_of[r["id"]], r["id"] + ".wav"),
                     os.path.join(PUB, "screening", "clips", r["id"] + ".wav"))
    # The item files keep accents in the apostrophe notation the synthesizer consumes
    # ("カ'シオ"), but the rater is taught exactly one notation -- ＼ = fall after that mora,
    # ￣ = no fall -- and the main blocks are already written that way. Shipping the raw field
    # would show the screening in a notation the instructions never define, and would render a
    # heiban/odaka pair identically ("ハシ" vs "ハシ"), making those items unanswerable.
    for r in scr:
        r["shown_accent"] = disp(r["shown_accent"])
        # The main task shows 語（読み）; the screening had no reading field and rendered an empty
        # "（）". Ship the reading with every accent mark stripped: the reading is not the answer
        # (the accent is), and it is shown in the main task too.
        r["reading"] = "".join(c for c in (r.get("reading") or r.get("true_reading") or "")
                               if c not in "'’＼￣")
    shown = ("id", "word", "shown_accent", "reading")   # gold and spoken_accent never reach the browser
    json.dump([{k: r[k] for k in shown} for r in scr],
              open(os.path.join(PUB, "screening", "manifest.json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    leaked_gold = [r for r in scr if "gold" in {k: r[k] for k in shown}]
    if leaked_gold:
        sys.exit("REFUSING TO SHIP: gold answers would reach the browser")
    print(f"  screening: {len(scr)} items (gold withheld)")

# BLOCK assignment (prereg S:4): a session is one block of ~95 judgments spanning all four cells,
# not one cell. <kit>/BLOCKS.csv maps blind_id -> block and carries no arm, speaker or source path,
# so shipping it does not weaken the blinding. Written into index.json as
# blocks[block] = [[kit, blind_id], ...] in a per-block fixed order.
blocks = {}
for k in kits:
    bp = os.path.join(args.kits, k, "BLOCKS.csv")
    if not os.path.exists(bp):
        sys.exit(f"REFUSING TO SHIP: {k}/BLOCKS.csv missing; run scripts/eval/assign_blocks_to_kit.py")
    with open(bp, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            blocks.setdefault(int(r["block"]), []).append([k, r["blind_id"]])
# A block design existed to keep a crowdworker's sitting near twenty minutes. Trained raters judge
# the whole set in under an hour, which removes the assignment problem entirely and restores the
# kappa over all stimuli that the block design gave up. --panel collapses the five blocks into one.
if args.panel:
    blocks = {0: [pair for b in sorted(blocks) for pair in blocks[b]]}
    CATCH_KNOWN_OVERRIDE = 8
else:
    CATCH_KNOWN_OVERRIDE = None
for b in blocks:
    random.Random(f"block:{b}").shuffle(blocks[b])

# CATCH TRIALS. A short gate admits a rater; it cannot tell whether they kept listening. Each block
# therefore carries a few checks of its own, mixed in and indistinguishable from real items:
#   - KNOWN: items from the verified screening pool, whose correct verdict is known. The kits have
#     nothing of this kind: their own checks are duplicate-consistency items (23 of them, already
#     mixed into the 473), which measure whether a rater is SELF-consistent but cannot tell a
#     careful rater from a consistent guesser. These can.
# Within-rater consistency is therefore left to the kits' 23 duplicate pairs; adding repeats here
# would have measured the same thing twice.
# They ship as an ordinary cell with ordinary blind ids, so nothing in the page or the network
# traffic marks a judgment as a check. The id -> source map is written to the local key file only,
# which this builder already refuses to ship.
CATCH_KNOWN, CATCH_REPEAT = (CATCH_KNOWN_OVERRIDE or 4), 0
catch_pool = json.load(open(os.path.join(REPO, "preregistration/items/screening_catch.json"),
                            encoding="utf-8")) if blocks else []
CATCH_AUDIO = f"{paths.DATA}/screening_catch_norm"
catch_key, cat_items = [], []
if catch_pool:
    os.makedirs(os.path.join(PUB, "E", "clips"), exist_ok=True)
    for b in sorted(blocks):
        rnd = random.Random(f"catch:{b}")
        known = rnd.sample(catch_pool, CATCH_KNOWN)
        reps = rnd.sample(blocks[b], min(CATCH_REPEAT, len(blocks[b])))
        picks = [("known", x) for x in known] + [("repeat", x) for x in reps]
        rnd.shuffle(picks)          # otherwise every block puts its repeats last, in the same slots
        for j, (kind, x) in enumerate(picks):
            bid = f"c{b}{j:02d}"
            if kind == "known":
                shutil.copy2(os.path.join(CATCH_AUDIO, x["id"] + ".wav"),
                             os.path.join(PUB, "E", "clips", bid + ".wav"))
                cat_items.append(dict(blind_id=bid, target_word=x["word"], target_reading=x["reading"],
                                      prescribed_accent=disp(x["shown_accent"])))
                catch_key.append(dict(block=b, blind_id=bid, kind=kind, source=x["id"], gold=x["gold"]))
            else:
                src_kit, src_bid = x
                shutil.copy2(os.path.join(PUB, src_kit, "clips", src_bid + ".wav"),
                             os.path.join(PUB, "E", "clips", bid + ".wav"))
                src_row = next(r for r in json.load(open(os.path.join(PUB, src_kit, "manifest.json"),
                                                        encoding="utf-8")) if r["blind_id"] == src_bid)
                cat_items.append(dict(blind_id=bid, target_word=src_row["target_word"],
                                      target_reading=src_row["target_reading"],
                                      prescribed_accent=src_row["prescribed_accent"]))
                catch_key.append(dict(block=b, blind_id=bid, kind=kind, source=f"{src_kit}/{src_bid}"))
        # Spread them out, jittered per block: an even step would put every rater's checks at the
        # same ordinal positions. A repeat is also kept well away from the item it repeats, so the
        # rater is not comparing two clips they can still hear in their head.
        step = max(1, (len(blocks[b]) + len(picks)) // (len(picks) + 1))
        for j, (kind, x) in enumerate(picks):
            at = min(len(blocks[b]), step * (j + 1) + rnd.randint(-step // 3, step // 3))
            if kind == "repeat":
                twin = blocks[b].index(list(x)) if list(x) in blocks[b] else None
                while twin is not None and abs(at - twin) < step:
                    at = min(len(blocks[b]), at + step)
            blocks[b].insert(at, ["E", f"c{b}{j:02d}"])
    json.dump(cat_items, open(os.path.join(PUB, "E", "manifest.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(catch_key, open(os.path.join(REPO, "preregistration/items/catch_KEY_DO_NOT_SHIP.json"),
                              "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  catch: {CATCH_KNOWN} known + {CATCH_REPEAT} repeat per block ({len(cat_items)} clips)")

print("  blocks: " + ", ".join(f"{b}={len(v)}" for b, v in sorted(blocks.items())))
json.dump({"kits": index, "collect_url": args.collect_url,
           "screening_n": len(scr),
           "blocks": {str(b): v for b, v in sorted(blocks.items())}},
          open(os.path.join(PUB, "index.json"), "w", encoding="utf-8"), ensure_ascii=False)
_page = open(os.path.join(REPO, "scripts/rater_ui/static_index.html"), encoding="utf-8").read()
assert _page.count("<script>") == _page.count("</script>"), "REFUSING TO SHIP: unbalanced <script>"
open(os.path.join(PUB, "index.html"), "w", encoding="utf-8").write(_page)

leaked = [os.path.join(dp, f) for dp, _, fs in os.walk(args.out) for f in fs
          if any(x in f for x in FORBIDDEN)]
if leaked:
    sys.exit(f"REFUSING TO SHIP: key-like files in output: {leaked[:5]}")
n = sum(len(fs) for _, _, fs in os.walk(args.out))
# Pages Functions: /api/screen grades the gate, /api/save stores each judgment. Serving them from
# the same origin as the site keeps the answer key off the client, makes the response readable (so a
# failed save is visible rather than silent), and removes the dependency on a second service.
fn_src = os.path.join(REPO, "scripts/rater_ui/functions")
if os.path.exists(fn_src) and args.kv_id:
    shutil.copytree(fn_src, os.path.join(args.out, "functions"), dirs_exist_ok=True)
    with open(os.path.join(args.out, "wrangler.toml"), "w") as f:
        f.write(f'name = "{args.project}"\ncompatibility_date = "2026-01-01"\n'
                f'pages_build_output_dir = "public"\n\n[[kv_namespaces]]\nbinding = "RATER"\n'
                f'id = "{args.kv_id}"\n')
    print("  functions: /api/screen (graded server-side), /api/save -> KV")

# Cloudflare Pages cache rules. The page and the manifests change between deploys and a rater
# holding a stale index.html would answer a block that no longer matches the shipped clips, so they
# are never cached; the clips themselves are immutable per blind_id and are cached hard.
# Written by the builder rather than kept as a loose file: the output directory is rebuilt from
# scratch each time, and a hand-placed _headers silently disappeared once.
with open(os.path.join(PUB, "_headers"), "w") as f:
    f.write("/\n  Cache-Control: no-store\n"
            "/index.html\n  Cache-Control: no-store\n"
            "/index.json\n  Cache-Control: no-store\n"
            "/*/manifest.json\n  Cache-Control: no-store\n"
            "/*/README_rater.md\n  Cache-Control: no-store\n"
            "/*/clips/*\n  Cache-Control: public, max-age=604800\n")

print(f"OK -> {args.out}  ({n} files, blinding verified: no KEY/build_note shipped)")
