#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clean-room FULL builder (500 words / ~5000 rows).
Uses only the hand-authored word
pool (pool_500.json) + carrier frame banks (frames_500.json) + pyopenjtalk
(public OpenJTalk dict). Only public resources are read.

Pipeline: classify accent (isolation-bug fix) -> quota+dedup selection
(odaka first, heiban heaviest) -> 10 register-varied carriers per word.
Deterministic (no randomness). Writes inventory_500.jsonl and the accent
review sheet.
"""
import json, re, sys
from collections import Counter, OrderedDict
import pyopenjtalk

PS, PE = "<PHON_START>", "<PHON_END>"
SMALL = "ァィゥェォャュョ"
KANA = set("ァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾタダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポマミムメモャヤュユョヨラリルレロワヲンヴー")
PARTS = set("がをはにへとでもの")
BLACK = set("急な 得意 苦手 元気 現在 偶然 満員 半分 生米".split())
QUOTA = OrderedDict([("household",60),("school_work",54),("food_shopping",60),
 ("transport",54),("digital_hobby_nature",66),("katakana_loanword",52),
 ("voiced_consonant",48),("katakana_proper_noun",46),("numeric_counter",60),("body_people",33)])
FILL = ["heiban","atamadaka","nakadaka"]; WEIGHT = {"heiban":3,"atamadaka":2,"nakadaka":2}

def morae(k):
    o=[]
    for ch in k:
        if ch in SMALL and o: o[-1]+=ch
        else: o.append(ch)
    return o

def phrases_and_F(text):
    labs=pyopenjtalk.extract_fullcontext(text); aps=set(); F=None
    for l in labs:
        ph=l.split('-')[1].split('+')[0]
        if ph in('sil','pau'): continue
        mI=re.search(r"/I:[^@]*@(\d+)\+",l)
        if mI: aps.add(int(mI.group(1)))
        if F is None:
            mF=re.search(r"/F:(\d+)_(\d+)#",l)
            if mF: F=(int(mF.group(1)),int(mF.group(2)))
    return len(aps),F

def analyze(word):
    toks=pyopenjtalk.run_frontend(word)
    if not toks: return None
    pron="".join(t['pron'].replace("’","") for t in toks)
    if any(c not in KANA for c in pron): return None
    m=morae(pron); ml=len(m)
    nap,F=phrases_and_F(word)
    if nap!=1 or F is None or F[0]!=ml: return None  # multi-phrase / misaligned -> drop
    if len(toks)==1:
        acc=toks[0]['acc']
    else:
        acc=F[1]
        if acc>=ml:
            _,F2=phrases_and_F(word+"がある")  # + particle frame
            acc=F2[1] if F2 and F2[1]<ml else 0
    if acc==0: t='heiban'
    elif acc==1: t='atamadaka'
    elif acc>=ml: t='odaka'
    else: t='nakadaka'
    reading="".join(m) if acc==0 else "".join(m[:acc])+"'"+"".join(m[acc:])
    return {"word":word,"accent_type":t,"accented_reading":reading}

def cnt_sub(w):
    if w.endswith("円"): return "cnt_money"
    if w.endswith("時間") or w.endswith("週間") or w.endswith("分") or w.endswith("年"): return "cnt_period"
    if w.endswith("時"): return "cnt_clock"
    if w.endswith("階"): return "cnt_floor"
    if w.endswith("月") or w.endswith("日"): return "cnt_month"
    if w.endswith("人"): return "cnt_person"
    if w.endswith("匹") or w.endswith("頭") or w.endswith("羽"): return "cnt_animal"
    if w.endswith("歳"): return "cnt_age"
    if w.endswith("パーセント"): return "cnt_ratio"
    if w.endswith("キロ") or w.endswith("センチ") or w.endswith("メートル"): return "cnt_distance"
    return "cnt_count"

def part_after(fr): return fr[fr.find("{W}")+3:fr.find("{W}")+4] in PARTS

def main(pool_path, frames_path, out_path, review_path):
    pool=json.load(open(pool_path,encoding="utf-8"))
    F=json.load(open(frames_path,encoding="utf-8"))
    # classify
    cls={}; dropped=[]
    for cat,words in pool.items():
        cls[cat]=[]
        for w in words:
            a=analyze(w)
            if a is None: dropped.append((cat,w)); continue
            a["category"]=cat; cls[cat].append(a)
    # select
    used=set(); selected=[]
    for cat,quota in QUOTA.items():
        items=[x for x in cls[cat] if x["word"] not in used and x["word"] not in BLACK]
        by={t:[x for x in items if x["accent_type"]==t] for t in ["odaka"]+FILL}
        pick=[]
        for x in by["odaka"][:quota]: pick.append(x); used.add(x["word"])
        rem=quota-len(pick); pools={t:list(by[t]) for t in FILL}
        while rem>0 and any(pools.values()):
            prog=False
            for t in FILL:
                for _ in range(WEIGHT[t]):
                    if rem>0 and pools[t]:
                        x=pools[t].pop(0)
                        if x["word"] in used: continue
                        pick.append(x); used.add(x["word"]); rem-=1; prog=True
            if not prog: break
        selected+=pick
    # generate
    rows=[]
    for gi,it in enumerate(selected):
        w,cat,typ,rd=it["word"],it["category"],it["accent_type"],it["accented_reading"]
        if cat=="numeric_counter":
            fp=F[cnt_sub(w)]
        else:
            fp=F["_neutral"]+F.get(cat,[])
            if typ=="odaka": fp=[f for f in fp if part_after(f)]
        n=len(fp); start=(gi*7)%n; chosen=[]; k=0
        while len(chosen)<10 and k<n*2:
            fr=fp[(start+k)%n]
            if fr not in chosen: chosen.append(fr)
            k+=1
        for fr in chosen:
            rows.append(OrderedDict([("word",w),("category",cat),("accent_type",typ),
                ("accented_reading",rd),("carrier_plain",fr.replace("{W}",w)),
                ("carrier_student",fr.replace("{W}",PS+rd+PE))]))
    with open(out_path,"w",encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    # review sheet
    order={"heiban":0,"atamadaka":1,"nakadaka":2,"odaka":3}
    first=OrderedDict()
    for r in rows: first.setdefault(r["word"],r)
    items=sorted(first.values(), key=lambda r:(r["category"],order[r["accent_type"]],r["word"]))
    with open(review_path,"w",encoding="utf-8") as f:
        f.write(f"# Clean-room Accent Review Sheet ({len(first)} words)\n\n")
        f.write("Manual eyeball check of pyopenjtalk-derived accent readings against NHK / OJAD.\n")
        f.write("Tick the box if the reading is correct; if wrong, append the correction in the\n")
        f.write("Check cell using the form `-> 正: カ'ナ` (accented katakana, `'` = downstep\n")
        f.write("nucleus, no apostrophe = heiban). Example: `[ ] -> 正: メガ'ネ`. A later script\n")
        f.write("parses `-> 正: <kana>` lines and writes the correction back into\n")
        f.write("`inventory_500.jsonl` (updating accented_reading + carrier_student + accent_type).\n\n")
        cur=None
        for r in items:
            if r["category"]!=cur:
                cur=r["category"]; f.write(f"\n## {cur}\n\n")
                f.write("| Check | Word | Reading | Accent | Sample carrier |\n")
                f.write("|---|---|---|---|---|\n")
            f.write(f"| [ ] | {r['word']} | {r['accented_reading']} | {r['accent_type']} | {r['carrier_plain']} |\n")
    print("dropped:",dropped)
    print("rows",len(rows),"words",len(first))
    print("per-category",dict(Counter(r["category"] for r in first.values())))
    print("per-accent",dict(Counter(r["accent_type"] for r in first.values())))

if __name__=="__main__":
    main(*sys.argv[1:5])
