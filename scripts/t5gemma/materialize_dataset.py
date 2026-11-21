"""Write a T5Gemma teacher set (jsonl) in the layout T5Gemma-TTS's data/combined_dataset.py reads.

  OUT/manifest_final/{train.txt,valid.txt}   # lines: <stem>\t<n_frames>
  OUT/text/<stem>.txt                          # encoder input: the tagged carrier (`surface_text`)
  OUT/xcodec2_1cb/<stem>.txt                   # one line: the base's acoustic codes for the plain carrier (1 codebook)

n_frames >= encodec_sr*audio_min_length (>=10 at 50Hz*0.2s) passes the filter. Holds out `valid_n` rows.
OUT is T5G_DATA_DIR of blockA_readlora_train.sh.
"""
import os, json, argparse, random

ap = argparse.ArgumentParser()
ap.add_argument("--teachers", required=True, help="teacher jsonl (teachers/t5gemma/ of the released dataset)")
ap.add_argument("--out", required=True, help="output directory (T5G_DATA_DIR of blockA_readlora_train.sh)")
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
