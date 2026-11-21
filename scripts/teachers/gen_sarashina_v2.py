"""v2 teacher generation for Sarashina2.2-TTS (uniform recipe, 2026-09-16).

Rule shared by all four backbones: for each carrier, the frozen base reads the PLAIN kanji carrier;
a take is kept if it passes the degeneration guard (here: 20 <= n_sem <= 600 semantic tokens) AND
the ASR of record hears the target reading in the vocoded take (scripts/teachers/reading_check.py);
otherwise the seed is bumped, up to MAX_TAKES takes; a carrier with no passing take is dropped and
logged. No speaker enters: the teacher is the base's own text-only continuation; the reference used
to vocode a take for the check affects timbre only and is never stored.

Output rows: {word, reading, plain, phon (Sarashina tag tokens), sem, n_sem, take, hyp}.

    $DATA/_handoff/venv_tf457/bin/python scripts/teachers/gen_sarashina_v2.py \
        --text $DATA/v2/teachers_text_v3.jsonl \
        --out  $DATA/v2/sarashina/teachers.jsonl
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/, for paths.py
import paths
HANDOFF = f"{paths.DATA}/_handoff"
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, paths.SARASHINA_DIR); sys.path.insert(0, REPO + "/scripts/teachers")
import torch
from sarashina_tts.additional_tokens import SPEECH_START_TOKEN, SPEECH_END_TOKEN, PRON_START_TOKEN, PRON_END_TOKEN, PAD_TOKEN
from sarashina_tts.generate.generate import SarashinaTTSGenerator, _semantic_ids_to_str
from reading_check import Checker

ap = argparse.ArgumentParser()
ap.add_argument("--text", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--ref", default=f"{paths.DATA}/refvoice/pilot_ref_24k.wav")
ap.add_argument("--ref-json", default=f"{paths.DATA}/refvoice/pilot_ref.json")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--batch", type=int, default=24)
ap.add_argument("--max-new", type=int, default=400)
ap.add_argument("--max-takes", type=int, default=3)
ap.add_argument("--nsem-min", type=int, default=20)
ap.add_argument("--nsem-max", type=int, default=600)
ap.add_argument("--only-dropped", help="provenance.json of a previous run: process only its dropped carriers")
ap.add_argument("--take-offset", type=int, default=0, help="first take index for this run")
a = ap.parse_args()
os.makedirs(os.path.dirname(a.out), exist_ok=True)

rows = [json.loads(l) for l in open(a.text, encoding="utf-8")]
if a.limit:
    rows = rows[:a.limit]
if a.only_dropped:
    _d = {(d["word"], d["plain"]) for d in json.load(open(a.only_dropped, encoding="utf-8"))["dropped_rows"]}
    rows = [r for r in rows if (r["word"], r["plain"]) in _d]
    print(f"[extra pass] {len(rows)} carriers dropped by the previous run", flush=True)
for r in rows:
    r["phon"] = r["phon"].replace("<PHON_START>", PRON_START_TOKEN).replace("<PHON_END>", PRON_END_TOKEN)

torch.backends.cudnn.enabled = False
g = SarashinaTTSGenerator(model_dir=HANDOFF + "/sara-tts-weights", watermark=False, use_vllm=False)
tok = g.tokenizer; tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = PAD_TOKEN
model = g.text_generator.model
model.config.use_cache = True
SEM_BASE = tok.convert_tokens_to_ids("<|semantic_0|>"); SEM_MAX = tok.convert_tokens_to_ids("<|semantic_6560|>")
SE_ID = tok.convert_tokens_to_ids(SPEECH_END_TOKEN)
REFTEXT = json.load(open(a.ref_json, encoding="utf-8"))["text"]
emb = g._extract_zero_shot_embedding(a.ref); ptok = g._extract_audio_prompt_tokens(a.ref); pfeat = g._extract_audio_prompt_feat(a.ref)
real_generate = g.text_generator.generate
checker = Checker(device=0)

def extract(gen):
    out = []
    for t in gen:
        t = int(t)
        if t == SE_ID or t == tok.eos_token_id or t == tok.pad_token_id:
            break
        if SEM_BASE <= t <= SEM_MAX:
            out.append(t - SEM_BASE)
        elif out:
            break
    return out

def generate_batch(chunk, take):
    """Base reads the plain carriers; per-batch seed keyed on (first row index, take)."""
    enc = tok([r["plain"] + SPEECH_START_TOKEN for r in chunk], return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        torch.manual_seed(1234 + chunk[0]["_i"] + 100000 * take)
        o = model.generate(**enc, max_new_tokens=a.max_new, do_sample=True, temperature=0.9, top_p=0.95,
                           repetition_penalty=1.0, pad_token_id=tok.pad_token_id, eos_token_id=(tok.eos_token_id or 2))
    return [extract(gg.tolist()) for gg in o[:, enc["input_ids"].shape[1]:]]

def vocode(sem):
    g.text_generator.generate = lambda prompts, gen_kwargs: [_semantic_ids_to_str(sem)]
    try:
        with torch.no_grad():
            w = g.generate(["x"], flow_embedding=emb, audio_prompt_text=REFTEXT, audio_prompt_tokens=ptok,
                           audio_prompt_feat=pfeat, audio_prompt_path=a.ref, gen_kwargs={"max_tokens": 8})[0]
    finally:
        g.text_generator.generate = real_generate
    return w.squeeze().detach().cpu().numpy().astype("float32")

for i, r in enumerate(rows):
    r["_i"] = i
pending = list(rows); kept = []; dropped = []; t0 = time.time()
for take in range(a.take_offset, a.take_offset + a.max_takes):
    nxt = []
    for s in range(0, len(pending), a.batch):
        chunk = pending[s:s + a.batch]
        sems = generate_batch(chunk, take)
        for r, sem in zip(chunk, sems):
            ok_guard = a.nsem_min <= len(sem) <= a.nsem_max
            heard, hyp = (False, "")
            if ok_guard:
                heard, hyp = checker.heard(vocode(sem), 24000, r["reading"])
            if ok_guard and heard:
                kept.append({"word": r["word"], "reading": r["reading"], "plain": r["plain"], "phon": r["phon"],
                             "sem": sem, "n_sem": len(sem), "take": take, "hyp": hyp})
            else:
                r.setdefault("_fail", []).append({"take": take, "n_sem": len(sem), "guard": ok_guard, "hyp": hyp})
                nxt.append(r)
        print(f"  take {take}: {min(s + a.batch, len(pending))}/{len(pending)} pending -> kept {len(kept)}  {time.time()-t0:.0f}s", flush=True)
    pending = nxt
    if not pending:
        break
dropped = [{"word": r["word"], "plain": r["plain"], "fails": r["_fail"]} for r in pending]
kept.sort(key=lambda r: rows.index(next(x for x in rows if x["plain"] == r["plain"] and x["word"] == r["word"])))
with open(a.out, "w", encoding="utf-8") as f:
    for r in kept:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
json.dump({"rule": "first take passing guard (20<=n_sem<=600) and reading check; max_takes", "max_takes": a.max_takes,
           "kept": len(kept), "dropped": len(dropped), "words_in": len({r['word'] for r in rows}),
           "words_kept": len({r['word'] for r in kept}),
           "takes_hist": {str(k): sum(1 for r in kept if r["take"] == k) for k in range(a.take_offset, a.take_offset + a.max_takes)},
           "dropped_rows": dropped}, open(a.out + ".provenance.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"DONE kept={len(kept)} dropped={len(dropped)} words {len({r['word'] for r in kept})}/{len({r['word'] for r in rows})} "
      f"in {time.time()-t0:.0f}s -> {a.out}", flush=True)
