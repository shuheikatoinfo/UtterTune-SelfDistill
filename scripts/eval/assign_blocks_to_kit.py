#!/usr/bin/env python3
"""Write <kit>/BLOCKS.csv (blind_id, block) for every row of every cell, so the rater server can
serve a BLOCK -- 95 judgements spanning all four cells -- without ever reading KEY_DO_NOT_OPEN.
Blocks are defined on word-units (preregistration/items/blocks.json: cell -> idx list); an
attention clone goes wherever its original goes (same cell + idx), so duplicate-consistency is
scored within one session. BLOCKS.csv carries no arm, speaker, idx or source path.
"""
import csv, json, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/, for paths.py
import paths
kit_root = sys.argv[1] if len(sys.argv) > 1 else f"{paths.DATA}/blind_kits_42"
blocks = json.load(open("preregistration/items/blocks.json", encoding="utf-8"))
block_of = {(c, int(u)): b["block"] for b in blocks["blocks"] for c, us in b["units"].items() for u in us}
per_block = collections.Counter(); missing = 0
for cell in sorted(d for d in os.listdir(kit_root) if os.path.isfile(f"{kit_root}/{d}/KEY_DO_NOT_OPEN.csv")):
    rows = list(csv.DictReader(open(f"{kit_root}/{cell}/KEY_DO_NOT_OPEN.csv", encoding="utf-8")))
    out = []
    for r in rows:
        b = block_of.get((cell, int(r["idx"])))
        if b is None: missing += 1; continue
        out.append({"blind_id": r["blind_id"], "block": b}); per_block[b] += 1
    with open(f"{kit_root}/{cell}/BLOCKS.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["blind_id", "block"]); w.writeheader(); w.writerows(out)
    print(f"  {cell}: {len(out)} rows -> BLOCKS.csv")
print("  judgements per block (incl. attention clones):", dict(sorted(per_block.items())), "| unmapped:", missing)
