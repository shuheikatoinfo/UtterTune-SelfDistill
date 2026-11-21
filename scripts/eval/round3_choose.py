#!/usr/bin/env python3
"""Round 3: apply the registered checkpoint rule (section 16, 08:55) to a backbone's validation reads and write
$V/r3/<bb>/CHOSEN_CKPT. Rule: over the 144 validation items, lowest mean sentence kana CER of the tag arm; a
checkpoint within 0.005 of the best is a tie, broken by the higher mean UTMOSv2 of the tag arm; excluded: a checkpoint
that needed a redraw on more than 2% of the 105 selection items or exhausted the takes on any (where the guard record
exists). Also prints the table and writes <bb>_choice.json. The author's blind listening is a check on top of this.
    python3 scripts/eval/round3_choose.py <cv2|irodori|t5gemma> [--dry]
Layouts: cv2  r3val/cv2/<set>/s<N>/guard_takes.json (arm field; s250 also holds noedit and kana)
         irodori r3val/irodori/<set>/{base,base_kana,s<N>}/guard_takes.json (one arm per dir, no arm field)
         t5gemma r3val/t5gemma/<set>/{base,s<N>}/**/<arm>/meta.json (records with hyp_kana)"""
import glob, json, os, re, sys, statistics as st
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/, for paths.py
import paths
import pyopenjtalk
V = f"{paths.DATA}/v2"; R = paths.REPO; bb = sys.argv[1]; dry = "--dry" in sys.argv
CAN = str.maketrans({"ヂ": "ジ", "ヅ": "ズ", "ヴ": "ブ"}); norm = lambda s: re.sub(r"[^ァ-ヶー]", "", s or "").translate(CAN)
def lev(a, b):
    p = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        q = [i]
        for j, cb in enumerate(b, 1): q.append(min(p[j] + 1, q[j - 1] + 1, p[j - 1] + (ca != cb)))
        p = q
    return p[-1]
SETS = {"sel105": f"{R}/data/derived/sel105_routed/test_sel105", "deveasy": f"{R}/data/derived/deveasy/test_deveasy"}
REF = {s: [norm(pyopenjtalk.g2p(l.rstrip("\n"), kana=True)) for l in open(p + "_yomi.txt", encoding="utf-8") if l.strip()] for s, p in SETS.items()}
def load(bb, s, label, arm):
    """-> {idx: (cer, takes, exhausted)} for one arm of one run"""
    out = {}
    if bb == "cv2":
        d = f"{V}/r3val/cv2/{s}/{label}"; f = f"{d}/guard_takes.json"
        if not os.path.exists(f): return None
        for x in json.load(open(f)):
            if x.get("arm") == arm: out[int(x["idx"])] = (min(1.0, lev(norm(x.get("hyp")), REF[s][int(x["idx"])]) / max(1, len(REF[s][int(x["idx"])]))), x.get("takes", 1), bool(x.get("final_reason")))
    elif bb == "irodori":
        d = {"noedit": f"{V}/r3val/irodori/{s}/base", "kana": f"{V}/r3val/irodori/{s}/base_kana"}.get(arm, f"{V}/r3val/irodori/{s}/{label}"); f = f"{d}/guard_takes.json"
        if not os.path.exists(f): return None
        for x in json.load(open(f)): out[int(x["idx"])] = (min(1.0, lev(norm(x.get("hyp")), REF[s][int(x["idx"])]) / max(1, len(REF[s][int(x["idx"])]))), x.get("takes", 1), bool(x.get("final_reason")))
    else:
        d = f"{V}/r3val/t5gemma/{s}/{'base' if arm != 'ours' else label}"; fs = glob.glob(f"{d}/**/{arm}/meta.json", recursive=True)
        if not fs: return None
        for x in json.load(open(fs[0]))["records"]: out[int(x["i"])] = (min(1.0, lev(norm(x.get("hyp_kana")), REF[s][int(x["i"])]) / max(1, len(REF[s][int(x["i"])]))), x.get("takes", 1), bool(x.get("final_reason")))
    return out
labels = sorted({os.path.basename(p.rstrip("/")) for s in SETS for p in glob.glob(f"{V}/r3val/{bb}/{s}/*/") if re.search(r"s\d+$", os.path.basename(p.rstrip("/")))}, key=lambda l: (l.rstrip("0123456789"), int(re.search(r"(\d+)$", l).group(1))))
if not labels: print(bb, "no validation reads yet"); raise SystemExit(1)
base_label = labels[0]
rows = []; print(f"{bb}: checkpoints {labels}")
for s in SETS:
    ne = load(bb, s, base_label, "noedit"); ka = load(bb, s, base_label, "kana")
    if ne: print(f"  {s}: unedited CER {st.mean(v[0] for v in ne.values()):.3f} | kana CER {st.mean(v[0] for v in ka.values()):.3f}" if ka else f"  {s}: unedited CER {st.mean(v[0] for v in ne.values()):.3f}")
table = {}
for lab in labels:
    per = {}; ok = True
    for s in SETS:
        o = load(bb, s, lab, "ours")
        if not o or len(o) < len(REF[s]): ok = False; break
        per[s] = o
    if not ok: print(f"  {lab}: incomplete"); continue
    allc = [v[0] for s in SETS for v in per[s].values()]; red = sum(v[1] > 1 for v in per["sel105"].values()); exh = sum(v[2] for v in per["sel105"].values())
    table[lab] = {"cer144": round(st.mean(allc), 4), "cer_sel105": round(st.mean(v[0] for v in per["sel105"].values()), 4), "cer_deveasy": round(st.mean(v[0] for v in per["deveasy"].values()), 4), "redraw_sel105": red, "exhausted_sel105": exh, "eligible": red <= 2 and exh == 0}
    print(f"  {lab}: CER144 {table[lab]['cer144']:.4f} (sel105 {table[lab]['cer_sel105']:.3f}, deveasy {table[lab]['cer_deveasy']:.3f}) redraw {red} exhausted {exh} {'' if table[lab]['eligible'] else 'EXCLUDED'}")
elig = [l for l in table if table[l]["eligible"]]
if not elig: print("no eligible checkpoint"); raise SystemExit(2)
best = min(table[l]["cer144"] for l in elig); ties = [l for l in elig if table[l]["cer144"] - best <= 0.005]
choice = ties[0]; note = "lowest CER"
if len(ties) > 1:
    # UTMOSv2 lives in the metrics venv (scripts/eval/.venv_metrics), not in this interpreter: score via a subprocess
    import subprocess
    um = {}
    for l in ties:
        files = sorted(glob.glob(f"{V}/r3val/{bb}/*/{l}/**/ours/*.wav", recursive=True))
        if not files: um[l] = -1; continue
        code = ("import sys, json, statistics as st, torch; torch.backends.cudnn.enabled = False; sys.path.insert(0, '%s/scripts/eval'); from eval_metrics import utmos; "
                "print(json.dumps(st.mean(utmos(json.load(open(sys.argv[1]))))))" % R)
        lst = f"{V}/r3/{bb}/_utmos_files_{l}.json"; json.dump(files, open(lst, "w"))
        r = subprocess.run([f"{R}/scripts/eval/.venv_metrics/bin/python", "-c", code, lst], capture_output=True, text=True, env=dict(os.environ, DISABLE_CUDNN="1"))
        um[l] = float(r.stdout.strip().split("\n")[-1]) if r.returncode == 0 and r.stdout.strip() else -1
        if um[l] == -1: print("UTMOSv2 failed for", l, r.stderr[-300:])
    choice = max(ties, key=lambda l: um[l]); note = f"tie within 0.005 among {ties}, UTMOSv2 {dict((k, round(v, 3)) for k, v in um.items())}"
steps = int(re.search(r"(\d+)$", choice).group(1))
path = {"cv2": f"{V}/r3/cv2/lora_s{steps}", "irodori": (glob.glob(f"{V}/r3/irodori/phon/checkpoint_{steps:07d}*") or [f"{V}/r3/irodori/phon/checkpoint_final.pt"])[0], "t5gemma": f"pod:merged_{choice}"}[bb]
json.dump({"table": table, "choice": choice, "steps": steps, "path": path, "note": note}, open(f"{V}/r3/{bb}/{bb}_choice.json", "w"), indent=1)
print(f"CHOICE {bb}: {choice} ({note}) -> {path}")
if not dry: open(f"{V}/r3/{bb}/CHOSEN_CKPT" if bb != "t5gemma" else f"{V}/r3/{bb}/CHOSEN_POD", "w").write(path)
