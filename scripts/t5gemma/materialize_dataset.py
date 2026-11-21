"""STEP B - materialize teacher jsonl into the layout data/combined_dataset.py expects.

  DATASET_DIR/manifest_final/{train.txt,valid.txt}   # lines: <stem>\t<n_frames>
  DATASET_DIR/text/<stem>.txt                          # encoder input = SURFACE kanji carrier
  DATASET_DIR/xcodec2_1cb/<stem>.txt                   # one line: space-separated acoustic ints (1 codebook)

n_frames >= encodec_sr*audio_min_length (>=10 at 50Hz*0.2s) passes the filter. Holds out `valid_n` rows.
"""
import os, json, argparse, random

ap = argparse.ArgumentParser()
ap.add_argument("--teachers", required=True, help="teachers.jsonl from gen_teachers_t5gemma.py")
ap.add_argument("--out", required=True, help="DATASET_DIR")
ap.add_argument("--valid_n", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

rows = [json.loads(l) for l in open(args.teachers, encoding="utf-8") if l.strip()]
rows = [r for r in rows if r.get("n_frames", 0) >= 10 and r.get("codes")]
print(f"loaded {len(rows)} valid teacher rows")

man = os.path.join(args.out, "manifest_final")
txt = os.path.join(args.out, "text")
enc = os.path.join(args.out, "xcodec2_1cb")
for d in (man, txt, enc):
    os.makedirs(d, exist_ok=True)

for r in rows:
    stem = r["stem"]
    with open(os.path.join(txt, stem + ".txt"), "w", encoding="utf-8") as f:
        f.write(r["surface_text"].strip())
    with open(os.path.join(enc, stem + ".txt"), "w") as f:
        f.write(" ".join(str(int(c)) for c in r["codes"]))  # single codebook -> one line

random.Random(args.seed).shuffle(rows)
valid = rows[: args.valid_n]
train = rows[args.valid_n:]
if not train:  # tiny smoke set: don't starve train
    train, valid = rows, rows[:1]
with open(os.path.join(man, "train.txt"), "w") as f:
    for r in train:
        f.write(f"{r['stem']}\t{r['n_frames']}\n")
with open(os.path.join(man, "valid.txt"), "w") as f:
    for r in valid:
        f.write(f"{r['stem']}\t{r['n_frames']}\n")
print(f"train={len(train)} valid={len(valid)} -> {args.out}")
print(f"  manifest_final/{{train,valid}}.txt  text/<stem>.txt  xcodec2_1cb/<stem>.txt")
