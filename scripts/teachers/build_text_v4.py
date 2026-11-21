#!/usr/bin/env python3
"""EXPLORATORY follow-up (PREREGISTRATION_v2.md section 12): the registered text set (471 words x 10 carriers)
extended to up to 50 carriers per word, same words, same readings, same frame-times-word construction as
scripts/cleanroom/build_500.py. Frames: the hand-authored banks of frames_500.json (30 neutral + category banks, of
which the registered set uses 10 per word) plus the new hand-authored banks below. A word's registered 10 carriers are
kept; new carriers are added in a fixed order (no sampling). Odaka words only take frames with a particle directly
after the target, as in the registered builder. Only public resources are read.
"""
import json, collections, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/, for paths.py
import paths
sys.path.insert(0, "scripts/cleanroom")
V = f"{paths.DATA}/v2"
PS, PE = "<PHON_START>", "<PHON_END>"; PARTS = set("がをはにへとでもの")
part_after = lambda fr: fr[fr.find("{W}") + 3:fr.find("{W}") + 4] in PARTS
NEW_NEUTRAL = [
 "ねえ、{W}のこと、ちょっとだけ一緒に見てみない？", "昨日、{W}の話を聞いて驚きました。", "正直に言うと、{W}にはあまり詳しくないんです。",
 "先週の番組で、{W}が大きく取り上げられていた。", "もしよければ、{W}について教えてもらえますか？", "そういえば、{W}のことをすっかり忘れていた。",
 "子どものころから、{W}には特別な思い出がある。", "最近になって、{W}を見直す人が増えているそうだ。", "あの人は、{W}のことになると急に話が長くなる。",
 "今日の授業では、{W}が話題の中心になった。", "祖母はいつも、{W}の大切さを語っていた。", "雑誌の特集で、{W}が詳しく紹介されていました。",
 "彼女は{W}に関する本を何冊も持っている。", "みんなで{W}の写真を見ながら盛り上がった。", "旅行のあいだ、{W}のことばかり考えていた。",
 "店員さんに{W}のことを尋ねてみた。", "父は若いころ、{W}に夢中だったらしい。", "その記事は、{W}をめぐる問題を分かりやすく説明している。",
 "ところで、{W}はもう確認しましたか？", "会議の最後に、{W}について短い報告があった。", "友人の影響で、{W}に興味を持つようになった。",
 "説明を聞いて、{W}の仕組みがやっと分かった。", "どうしても{W}が気になって、夜遅くまで調べてしまった。", "先生は黒板に{W}と大きく書いた。",
 "明日までに、{W}に関する資料をまとめておきます。", "意外なことに、{W}を知らない人も多いようだ。", "引っ越してから、{W}との付き合い方が変わった。",
 "この地域では、{W}が昔から親しまれている。", "ニュースで{W}の名前を聞くたびに、あの日を思い出す。", "いつか{W}についてゆっくり話しましょう。",
 "兄に頼まれて、{W}のことを調べている。", "本当のところ、{W}はどれくらい重要なんだろう？", "その作家は、{W}を題材にした小説を書いている。",
 "私たちの世代にとって、{W}は身近な存在だ。", "アンケートでは、{W}を挙げた人がいちばん多かった。", "忙しくて、{W}のことまで手が回らない。",
 "来月の特集は{W}に決まったそうです。", "彼の説明のおかげで、{W}への見方が変わった。", "休みの日には、{W}のことをのんびり考えるのが好きだ。",
 "誰に聞いても、{W}の評判はとても良かった。", "最初は{W}に戸惑ったが、すぐに慣れた。", "母からの手紙には、{W}のことが細かく書かれていた。",
]
NEW_COUNTER = [
 "たしか{W}だったと思います。", "{W}で足りるかどうか心配だ。", "{W}と聞いて、少し驚いた。", "予定では{W}になるはずです。", "メモには{W}と書いてあった。",
 "{W}に変更してもらえますか？", "結局、{W}ということで話がまとまった。", "正確には{W}ではなかったかもしれない。", "{W}も必要だとは思わなかった。",
 "彼は小さな声で{W}と答えた。", "最初は{W}のつもりだった。", "{W}が正しい答えです。", "もう一度確認したら、{W}で合っていました。", "{W}を超えることはないでしょう。",
 "去年は{W}でしたが、今年は違います。", "{W}に決まったら連絡します。", "表示は{W}のままになっている。", "係の人は{W}と言っていました。", "{W}では少し足りない気がする。",
 "念のため、{W}と書き直しておいた。", "みんなの予想は{W}で一致していた。", "{W}の部分だけ、赤い字で書かれていた。", "書類には{W}と記入してください。", "{W}は多すぎると言われてしまった。",
 "たった{W}でも、ずいぶん印象が変わる。", "思っていたより{W}に近い結果になった。", "{W}を目安に準備を進めています。", "父は{W}と聞いて、すぐに納得した。", "{W}が限度だと説明された。",
 "友達は{W}と勘違いしていたらしい。", "{W}へ変わったのは、つい最近のことだ。", "放送では{W}と案内していた。", "{W}のところに印を付けておいた。", "正しくは{W}です、と訂正が入った。",
 "{W}もあれば十分だと思います。", "{W}とだけ書かれた紙が残っていた。", "その数字は{W}を指している。", "計算してみると、ちょうど{W}になった。", "{W}に直して、もう一度提出した。", "目標は{W}に設定されている。",
]
assert len(set(NEW_NEUTRAL)) == len(NEW_NEUTRAL) and all(part_after(f) for f in NEW_NEUTRAL)
def cnt_sub(w):   # verbatim from scripts/cleanroom/build_500.py (importing it pulls in pyopenjtalk)
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
F = json.load(open("scripts/cleanroom/frames_500.json", encoding="utf-8"))
rows = [json.loads(l) for l in open(f"{V}/teachers_text_v3.jsonl", encoding="utf-8")]
byw = collections.OrderedDict()
for r in rows: byw.setdefault(r["word"], []).append(r)
TARGET = 50; new_rows, per = [], []
for w, rs in byw.items():
    r0 = rs[0]; cat, typ, rd = r0["category"], r0["accent_type"], r0["reading"]
    pool = (F[cnt_sub(w)] + NEW_COUNTER) if cat == "numeric_counter" else (F["_neutral"] + F.get(cat, []) + NEW_NEUTRAL)
    if typ == "odaka": pool = [f for f in pool if part_after(f)]
    have = {r["plain"] for r in rs}; add = []
    for fr in pool:
        p = fr.replace("{W}", w)
        if p in have or len(rs) + len(add) >= TARGET: continue
        have.add(p); add.append({"word": w, "reading": rd, "plain": p, "phon": fr.replace("{W}", PS + rd + PE), "accent_type": typ, "category": cat, "set": "v4_new"})
    new_rows += add; per.append(len(rs) + len(add))
with open(f"{V}/teachers_text_v4_new.jsonl", "w", encoding="utf-8") as f:
    for r in new_rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"words {len(byw)} | registered carriers {len(rows)} | new carriers {len(new_rows)} | per word min {min(per)} median {sorted(per)[len(per)//2]} max {max(per)} | words under 50: {sum(p < 50 for p in per)}")
