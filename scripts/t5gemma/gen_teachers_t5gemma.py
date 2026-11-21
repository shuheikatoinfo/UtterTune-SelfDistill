"""STEP A - self-distill TEACHER generation for the T5Gemma-TTS (encoder-decoder) backbone.

READING-ONLY. Teacher input = the KANA reading substituted into a carrier (unambiguous -> the frozen
base reads it correctly). Teacher label = the acoustic XCodec2 tokens the base emits (gen_frames from
inference_one_sample(return_frames=True)). Student (trained later) input = the SURFACE kanji carrier.
So the LoRA learns: surface kanji -> acoustic tokens that SOUND the correct reading. No accent/PHON
control (out of scope: that needs encoder-vocab surgery).

Two word sources:
  --mode covered  --pairs_tsv <idx\tsurface\treading>   (e.g. difficult-50 target words, FRESH carriers)
  --mode generic  --teachers_jsonl <text-set jsonl>  (word/reading/plain; carrier = `plain`)

Output jsonl rows: {stem, word, reading, surface_text, teacher_input, n_frames, cer, codes:[int,...]}.
QC = decode the base output + kana-whisper CER vs pyopenjtalk.g2p(teacher_input, kana=True); reject
teachers whose reading the base itself garbled (disclosed uniform filter, like gen_teachers_cv2.py).

Run:  cd $T5GEMMA_DIR
  .venv_t5gemma/bin/python <this> --mode covered --pairs_tsv .../difficult50_pairs.tsv --limit_words 3
"""
import sys, os, re, json, difflib, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/, for paths.py
import paths

T5G = paths.T5GEMMA_DIR
sys.path.insert(0, T5G)

import fire
import numpy as np
import torch
import soundfile as sf
import pyopenjtalk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from data.tokenizer import AudioTokenizer
from duration_estimator import estimate_duration
from inference_tts_utils import inference_one_sample

DEFAULT_REF = f"{paths.DATA}/_handoff/sara-ref/samples/styles/ナレーション_prompt.wav"
DEFAULT_REFTEXT = "彼らは王権の正当性を重視し革命の影響を抑え込むことを目指しました"

CAN = str.maketrans({"ヂ": "ジ", "ヅ": "ズ", "ヴ": "ブ"})
MARKS = str.maketrans("", "", "'’/｜|┐［］[]・")        # strip accent/boundary marks -> pure kana
# student side KEEPS ' and / (both stock Gemma tokens); only the exotic bracket marks go.
BOUNDARY_KEEP = str.maketrans("", "", "’｜|┐［］[]・")


def kana_only(s):
    return re.sub(r"[^ァ-ヶー]", "", s).translate(CAN)


def accented(s):
    """Kana + the UtterTune notation marks, i.e. what the STUDENT is shown.
    ' (accent nucleus) and / (phrase boundary) are ordinary single tokens in the Gemma
    vocabulary, so the accent axis needs no vocabulary change -- only the two span
    delimiters do, and those are taken from Gemma's reserved <unusedN> rows."""
    return re.sub(r"[^ァ-ヶー'/]", "", s).translate(CAN)


# Span delimiters. Gemma reserves <unused0..N> as SINGLE tokens that never occur in text
# (verified: ids 7 and 8, atomic, no collision with pad/bos/eos/unk). Using them keeps the
# interface identical to Sarashina's <PRON_START>/<PRON_END> without resizing the embedding.
# The literal string "<PHON_START>" must NEVER be fed to this backbone: its tokenizer splits it
# into ['<','PH','ON','_','START','>'] and the decoder speaks those subwords aloud.
TAG_S, TAG_E = "<unused0>", "<unused1>"


# neutral carriers that isolate the target word W (NOT any difficult-50 eval sentence -> no leakage).
COVERED_CARRIERS = [
    "{W}という言葉について考えた。",
    "彼女は{W}という表現を使った。",
    "今日の授業で{W}を学んだ。",
    "その本には{W}の話が書かれていた。",
    "私は{W}についてよく知らない。",
    "先生が{W}の意味を丁寧に説明した。",
    "この文章に{W}が登場する。",
    "彼は{W}という語を口にした。",
    "レポートに{W}について書いた。",
    "会議で{W}が話題になった。",
]


def _build_rows(mode, pairs_tsv, teachers_jsonl, carriers_per_word, limit_words, teacher_plain=0,
                student="plain"):
    """Return list of dicts: {stem, word, reading_raw, reading_kana, surface_text, teacher_input}."""
    rows = []
    if mode == "covered":
        pairs = []
        with open(pairs_tsv, encoding="utf-8") as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 3:
                    pairs.append((p[0], p[1], p[2]))  # idx, surface, reading
        if limit_words:
            pairs = pairs[:limit_words]
        ncar = min(carriers_per_word, len(COVERED_CARRIERS))
        for idx, surf, read in pairs:
            read_kana = kana_only(read.translate(MARKS))
            for c, tmpl in enumerate(COVERED_CARRIERS[:ncar]):
                rows.append({
                    "stem": f"cov_{idx}_{c}",
                    "word": surf, "reading_raw": read, "reading_kana": read_kana,
                    "surface_text": tmpl.replace("{W}", surf),
                    "teacher_input": tmpl.replace("{W}", read_kana),
                })
    elif mode == "generic":
        from collections import defaultdict
        per = defaultdict(int)
        with open(teachers_jsonl, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                w = r["word"]
                if per[w] >= carriers_per_word:
                    continue
                if w not in r["plain"]:
                    continue
                read_kana = kana_only(r["reading"].translate(MARKS))
                if not read_kana:
                    continue
                c = per[w]
                per[w] += 1
                # ON-METHOD (as on Sarashina): teacher input = PLAIN KANJI carrier,
                # let the base's natural reading+accent BE the teacher. teacher_plain=1 selects this.
                # teacher_plain=0 = the old kana-forced input (breaks the accent axis; kept for A/B).
                tin = r["plain"] if teacher_plain else r["plain"].replace(w, read_kana, 1)
                # STUDENT input. The method's mechanism is the ASYMMETRY between what the student
                # is shown and what the teacher was shown: the student sees the tagged reading, the
                # label is the frozen base reading the PLAIN carrier. student="plain" together with
                # teacher_plain=1 makes the two identical -> identity distillation, no control
                # channel, adapter learns nothing. That combination is rejected in main().
                if student == "tagged":
                    acc = accented(r["reading"].translate(BOUNDARY_KEEP))
                    if not acc:
                        continue
                    stext = r["plain"].replace(w, TAG_S + acc + TAG_E, 1)
                else:
                    stext = r["plain"]
                rows.append({
                    "stem": f"gen_{len(rows)}",
                    "word": w, "reading_raw": r["reading"], "reading_kana": read_kana,
                    "surface_text": stext,
                    "teacher_input": tin,
                })
        if limit_words:
            # limit_words here caps distinct words for smoke
            keep_words = list(dict.fromkeys(r["word"] for r in rows))[:limit_words]
            rows = [r for r in rows if r["word"] in keep_words]
    else:
        raise ValueError(mode)
    return rows


def main(
    mode="covered",
    pairs_tsv=f"{paths.DATA}/blockA_t5gemma_readlora/difficult50_pairs.tsv",
    teachers_jsonl=None,  # required with mode="generic"; round 3 imports only the helpers of this module
    out=f"{paths.DATA}/blockA_t5gemma_readlora/teachers_covered",
    model_dir=T5G + "/pretrained/T5Gemma-TTS-2b-2b",
    reference_speech=DEFAULT_REF,
    reference_text=DEFAULT_REFTEXT,
    no_ref=1,            # NO-REF is self-consistent with prompt-free (neighbor_prompt_prob 0) training,
                         #   and the narration ref was shown to DEGRADE reading on hard sentences.
    carriers_per_word=8,
    temperature=0.6,
    top_k=30,
    top_p=0.9,
    seed=1234,
    bestof=2,            # best-of-K per (word,carrier): keep the lowest-CER take (teacher construction only)
    qc=0.35,             # reject teacher if base's own reading kana-CER > qc (disclosed uniform filter)
    save_wav=0,          # save a few decoded teacher wavs for eyeballing
    limit_words=None,    # smoke: cap #distinct words
    teacher_plain=0,     # ON-METHOD: 1 = feed PLAIN KANJI carrier to the base (Sarashina method);
                         #   the base's natural reading+accent is the teacher. 0 = old kana-forced.
    student="plain",     # "tagged" = student sees <unused0>accented-reading<unused1> in place of the
                         #   target word (the method). "plain" = student sees the bare carrier.
):
    if student == "plain" and int(teacher_plain) == 1:
        raise SystemExit(
            "REFUSED: student='plain' with teacher_plain=1 makes the student input IDENTICAL to the "
            "teacher input. That is identity distillation -- the adapter is trained to reproduce what "
            "the frozen base already does, no control channel is installed, and the resulting numbers "
            "describe nothing. This combination produced the invalid T5Gemma row (2026-09-01). "
            "Use student='tagged'.")
    os.makedirs(out, exist_ok=True)
    wavdir = os.path.join(out, "wav"); os.makedirs(wavdir, exist_ok=True)
    rows = _build_rows(mode, pairs_tsv, teachers_jsonl, carriers_per_word, limit_words,
                       int(teacher_plain), student)
    print(f"[teacher-gen] mode={mode} student={student} teacher_plain={teacher_plain} "
          f"rows={len(rows)} carriers/word={carriers_per_word} qc={qc} temp={temperature}", flush=True)
    if student == "tagged":
        n_tag = sum(1 for r in rows if TAG_S in r["surface_text"])
        assert n_tag == len(rows), f"tag missing from {len(rows)-n_tag} student inputs"
        n_id = sum(1 for r in rows if r["surface_text"] == r["teacher_input"])
        assert n_id == 0, f"{n_id} rows have student input == teacher input (identity)"
        print(f"[teacher-gen] asymmetry check OK: {len(rows)} rows tagged, 0 identity pairs\n"
              f"  student : {rows[0]['surface_text']}\n  teacher : {rows[0]['teacher_input']}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "GPU required"
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    cfg = model.config
    dev = "cuda"
    tk = getattr(cfg, "text_tokenizer_name", None) or getattr(cfg, "t5gemma_model_name", None)
    text_tokenizer = AutoTokenizer.from_pretrained(tk)
    audio_tokenizer = AudioTokenizer(backend="xcodec2", model_name=getattr(cfg, "xcodec2_model_name", None))
    SR = int(audio_tokenizer.sample_rate)
    codec_sr = int(getattr(cfg, "encodec_sr", 50))
    no_ref = bool(int(no_ref))
    if no_ref:
        afn, pfx, prompt_end_frame = None, "", 0
    else:
        afn, pfx = reference_speech, reference_text
        info = sf.info(reference_speech); prompt_end_frame = int(100 * info.samplerate)
    kw = pipeline("automatic-speech-recognition", model="sbintuitions/kana-whisper", device=0,
                  torch_dtype=torch.float16, generate_kwargs={"language": "ja", "task": "transcribe"})
    print(f"[loaded] SR={SR} codec_sr={codec_sr} no_ref={no_ref} bestof={bestof}", flush=True)

    def synth_once(text, dur, s):
        torch.manual_seed(s); np.random.seed(s); torch.cuda.manual_seed(s)
        dc = {"top_k": top_k, "top_p": top_p, "min_p": 0, "temperature": temperature,
              "stop_repetition": 3, "codec_audio_sr": SR, "codec_sr": codec_sr,
              "silence_tokens": [], "sample_batch_size": 1}
        concat, gen, concat_frames, gen_frames = inference_one_sample(
            model=model, model_args=cfg, text_tokenizer=text_tokenizer, audio_tokenizer=audio_tokenizer,
            audio_fn=afn, target_text=text, lang="ja", device=dev, decode_config=dc,
            prompt_end_frame=prompt_end_frame, target_generation_length=dur,
            prefix_transcript=pfx, multi_trial=[], repeat_prompt=0, return_frames=True)
        codes = gen_frames.squeeze().to(torch.int64).tolist()
        if isinstance(codes, int):
            codes = [codes]
        audio = gen.detach().cpu().squeeze().numpy().astype("float32")
        return codes, audio

    outjsonl = os.path.join(out, "teachers.jsonl")
    kept, rej, cers = 0, 0, []
    words_with_teacher = set()
    t0 = time.time()
    with open(outjsonl, "w", encoding="utf-8") as wf:
        for i, r in enumerate(rows):
            ref = kana_only(pyopenjtalk.g2p(r["teacher_input"], kana=True))
            dur = float(estimate_duration(target_text=r["teacher_input"],
                                          reference_speech=None if no_ref else reference_speech,
                                          reference_transcript=None if no_ref else pfx,
                                          target_lang="ja", reference_lang="ja"))
            best = None  # (cer, codes, audio)
            for k in range(int(bestof)):
                codes, audio = synth_once(r["teacher_input"], dur, seed + 131 * k + i)
                hyp = kana_only(str(kw({"raw": audio, "sampling_rate": SR})["text"]))
                cer = 1.0 - difflib.SequenceMatcher(None, ref, hyp).ratio()
                if best is None or cer < best[0]:
                    best = (cer, codes, audio, hyp)
                if cer <= 0.1:
                    break
            cer, codes, audio, hyp = best
            n_frames = len(codes)
            ok = (cer <= qc) and (n_frames >= 10) and (min(codes) >= 0) and (max(codes) < 65536)
            if save_wav and i < int(save_wav):
                sf.write(os.path.join(wavdir, f"{r['stem']}.wav"), audio, SR)
            if not ok:
                rej += 1
                if i < 6 or rej <= 5:
                    print(f"  REJECT {r['stem']} cer={cer:.3f} nfr={n_frames} word={r['word']} "
                          f"ref={ref[:24]} hyp={hyp[:24]}", flush=True)
                continue
            cers.append(cer); kept += 1; words_with_teacher.add(r["word"])
            wf.write(json.dumps({"stem": r["stem"], "word": r["word"], "reading": r["reading_kana"],
                                 "surface_text": r["surface_text"], "teacher_input": r["teacher_input"],
                                 "n_frames": n_frames, "cer": round(cer, 4), "codes": codes},
                                ensure_ascii=False) + "\n")
            if i < 6 or i % 25 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{len(rows)} kept={kept} rej={rej} cer={cer:.3f} nfr={n_frames} "
                      f"word={r['word']} [{el:.0f}s, {el/(i+1):.1f}s/utt]", flush=True)
    n_words = len(set(r["word"] for r in rows))
    dur_s = time.time() - t0
    print(f"\n[DONE] kept={kept}/{len(rows)} rejected={rej} (qc>{qc}) "
          f"words_covered={len(words_with_teacher)}/{n_words} "
          f"mean_cer_kept={np.mean(cers) if cers else float('nan'):.3f} "
          f"wall={dur_s:.0f}s ({dur_s/max(1,len(rows)):.1f}s/utt) -> {outjsonl}", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
