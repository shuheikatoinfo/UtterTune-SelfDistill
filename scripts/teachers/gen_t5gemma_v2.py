"""Teacher generation for T5Gemma-TTS (the registered rule, as on the other backbones).

Same rule as the other backbones: the frozen base reads the PLAIN kanji carrier with no reference
(no_ref); a take is kept if it passes the degeneration guard (>= 10 codec frames, valid codes) AND
the ASR of record hears the target reading; else the seed is bumped, up to MAX_TAKES; a carrier
with no passing take is dropped and logged. No best-of, no CER threshold. Student text = the
carrier with the target replaced by <unused0>accented reading<unused1> (Gemma's reserved rows),
exactly as in round 1 (scripts/t5gemma/gen_teachers_t5gemma.py, student="tagged").

Output: teachers.jsonl rows {stem, word, reading, surface_text, teacher_input, n_frames, take, hyp, codes}
compatible with scripts/t5gemma/materialize_dataset.py.

    cd $T5GEMMA_DIR && .venv_t5gemma/bin/python <repo>/scripts/teachers/gen_t5gemma_v2.py \
        --text .../teachers_text_v3.jsonl --out $DATA/v2/t5gemma/teachers_generic
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/, for paths.py
import paths
T5G = paths.T5GEMMA_DIR
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, T5G); sys.path.insert(0, REPO + "/scripts/t5gemma"); sys.path.insert(0, REPO + "/scripts/teachers")
import numpy as np, torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from data.tokenizer import AudioTokenizer
from duration_estimator import estimate_duration
from inference_tts_utils import inference_one_sample
from gen_teachers_t5gemma import TAG_S, TAG_E, accented, BOUNDARY_KEEP, kana_only, MARKS
from reading_check import Checker

ap = argparse.ArgumentParser()
ap.add_argument("--text", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model_dir", default=T5G + "/pretrained/T5Gemma-TTS-2b-2b")
ap.add_argument("--temperature", type=float, default=0.6)
ap.add_argument("--top_k", type=int, default=30)
ap.add_argument("--top_p", type=float, default=0.9)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--max-takes", type=int, default=5)
ap.add_argument("--min-frames", type=int, default=10)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--shard", default="0/1", help="i/n: process rows i::n (merge shards afterwards)")
ap.add_argument("--only-dropped", help="provenance.json of a previous pass: process only its dropped carriers")
ap.add_argument("--take-offset", type=int, default=0, help="first take index for this pass (second pass: 1)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

rows = []
for l in open(a.text, encoding="utf-8"):
    r = json.loads(l)
    acc = accented(r["reading"].translate(BOUNDARY_KEEP))
    if not acc or r["word"] not in r["plain"]:
        continue
    rows.append({"word": r["word"], "reading": r["reading"], "reading_kana": kana_only(r["reading"].translate(MARKS)),
                 "teacher_input": r["plain"], "surface_text": r["plain"].replace(r["word"], TAG_S + acc + TAG_E, 1)})
for i, r in enumerate(rows):
    r["stem"] = f"gen_{i}"
si, sn = (int(x) for x in a.shard.split("/"))
rows = rows[si::sn]
if a.only_dropped:
    _d = {(d["word"], d["plain"]) for d in json.load(open(a.only_dropped, encoding="utf-8"))["dropped_rows"]}
    rows = [r for r in rows if (r["word"], r["teacher_input"]) in _d]
    print(f"[second pass] {len(rows)} carriers dropped by the first pass", flush=True)
if a.limit:
    rows = rows[:a.limit]
assert all(TAG_S in r["surface_text"] and r["surface_text"] != r["teacher_input"] for r in rows)
print(f"[teacher-gen v2] rows={len(rows)} words={len({r['word'] for r in rows})}", flush=True)

model = AutoModelForSeq2SeqLM.from_pretrained(a.model_dir, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto").eval()
cfg = model.config
tk = getattr(cfg, "text_tokenizer_name", None) or getattr(cfg, "t5gemma_model_name", None)
text_tokenizer = AutoTokenizer.from_pretrained(tk)
audio_tokenizer = AudioTokenizer(backend="xcodec2", model_name=getattr(cfg, "xcodec2_model_name", None))
SR = int(audio_tokenizer.sample_rate); codec_sr = int(getattr(cfg, "encodec_sr", 50))
checker = Checker(device=0)

def synth_once(text, dur, s):
    torch.manual_seed(s); np.random.seed(s); torch.cuda.manual_seed(s)
    dc = {"top_k": a.top_k, "top_p": a.top_p, "min_p": 0, "temperature": a.temperature, "stop_repetition": 3,
          "codec_audio_sr": SR, "codec_sr": codec_sr, "silence_tokens": [], "sample_batch_size": 1}
    _, gen, _, gen_frames = inference_one_sample(
        model=model, model_args=cfg, text_tokenizer=text_tokenizer, audio_tokenizer=audio_tokenizer,
        audio_fn=None, target_text=text, lang="ja", device="cuda", decode_config=dc, prompt_end_frame=0,
        target_generation_length=dur, prefix_transcript="", multi_trial=[], repeat_prompt=0, return_frames=True)
    codes = gen_frames.squeeze().to(torch.int64).tolist()
    if isinstance(codes, int):
        codes = [codes]
    return codes, gen.detach().cpu().squeeze().numpy().astype("float32")

outp = os.path.join(a.out, "teachers.jsonl")
kept, drops, takes_hist, t0 = 0, [], {}, time.time()
with open(outp, "w", encoding="utf-8") as wf:
    for i, r in enumerate(rows):
        dur = float(estimate_duration(target_text=r["teacher_input"], reference_speech=None, reference_transcript=None,
                                      target_lang="ja", reference_lang="ja"))
        got, fails = None, []
        for t in range(a.take_offset, a.take_offset + a.max_takes):
            codes, audio = synth_once(r["teacher_input"], dur, a.seed + 131 * t + i)
            guard = (len(codes) >= a.min_frames) and (min(codes) >= 0) and (max(codes) < 65536)
            heard, hyp = (False, "")
            if guard:
                heard, hyp = checker.heard(audio, SR, r["reading"])
            if guard and heard:
                got = (codes, t, hyp); break
            fails.append({"take": t, "guard": bool(guard), "n_frames": len(codes), "hyp": hyp})
        if got is None:
            drops.append({"word": r["word"], "plain": r["teacher_input"], "fails": fails})
        else:
            codes, t, hyp = got
            wf.write(json.dumps({"stem": r["stem"], "word": r["word"], "reading": r["reading_kana"],
                                 "surface_text": r["surface_text"], "teacher_input": r["teacher_input"],
                                 "n_frames": len(codes), "take": t, "hyp": hyp, "codes": codes}, ensure_ascii=False) + "\n")
            kept += 1; takes_hist[t] = takes_hist.get(t, 0) + 1
        if i % 25 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(rows)} kept={kept} dropped={len(drops)} [{el:.0f}s, {el/(i+1):.1f}s/row]", flush=True)
json.dump({"rule": "no_ref; first take passing guard (>=min_frames, valid codes) and reading check; max_takes",
           "max_takes": a.max_takes, "kept": kept, "dropped": len(drops), "words_in": len({r['word'] for r in rows}),
           "takes_hist": {str(k): v for k, v in sorted(takes_hist.items())}, "dropped_rows": drops},
          open(outp + ".provenance.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"[DONE] kept={kept}/{len(rows)} dropped={len(drops)} wall={time.time()-t0:.0f}s -> {outp}", flush=True)
