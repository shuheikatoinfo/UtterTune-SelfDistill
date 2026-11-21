"""Block A — Sarashina2.2-TTS, difficult set, 3 arms, 4 UMAP-representative speakers (HEADLINE).

Scales the validated 1-speaker pilot (blockA_sarashina_difficult.py) to the 4 eval speakers
(eval_speakers/selected_4.json: seed13/seed15/fictional3/seed06 fictional bootstrap wavs). These
fictional speakers have NO stored transcript, and Sarashina zero-shot REQUIRES audio_prompt_text,
so each ref wav is self-transcribed once with kana-whisper (bootstrap wavs are known to transcribe
cleanly) and that reading is used as the prompt text (CAM++-style speaker prompt, per
selected_4.json backbone_transfer note).

Arms per speaker (same as the pilot): noedit (base, plain kanji) / kana (base, target->katakana) /
ours (faithful2 adapter + UtterTune ' notation). Reading Kana-CER vs pyopenjtalk full-kana of the
yomi ref; accent = ear/rater on the wavs. No best-of-N (single seed 5000+i, uniform >=0.5s retry).
Output: blockA_sarashina_difficult_4spk/spk{0..3}/{arm}/NN.wav + blockA_difficult_4spk.json.

Speaker embeddings/tokens/feat are codec/flow-side (LLM-adapter-independent) so they are extracted
ONCE up front; then base arms (noedit,kana) run for all speakers, faithful2 is loaded once, and the
ours arm runs for all speakers — minimizing model reloads.

Usage: python blockA_sarashina_difficult_4spk.py [--limit N] [--nspk M]   (smoke: --limit 3 --nspk 1)
"""
import sys, os, re, json, difflib, argparse
import paths
SP = f"{paths.DATA}/_handoff"
sys.path.insert(0, paths.SARASHINA_DIR)
import torch, numpy as np, soundfile as sf
from peft import PeftModel
from sarashina_tts.additional_tokens import PRON_START_TOKEN, PRON_END_TOKEN
from sarashina_tts.generate.generate import SarashinaTTSGenerator
from transformers import pipeline
import pyopenjtalk
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval'))
from degenerate import synth_valid, safe_asr
# Calibrated from 957 generated clips: durations pile up at exactly 20.00 s (52 clips) -- that is
# the hard generation cap and a clip reaching it was truncated, not finished. An earlier value of
# 10.0 s (from max_tokens/50) was WRONG: 77 of the 112 clips over 9.9 s are perfectly fine by ASR,
# so that threshold would have rejected ~8% of legitimate audio and biased the set toward short
# utterances. [2026-09-04]
CAP_S = 19.9

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0, help="0 = all 50 sentences; N = smoke first N")
ap.add_argument("--nspk", type=int, default=4, help="number of speakers to run (smoke: 1)")
args = ap.parse_args()

# TEST_DIR / TEST_SET make the item set swappable without touching the synthesis path:
# the same code, the same recipe, only a different input file. Needed for the 105-item
# SELECTION split and the 319-item TEST split (preregistration/items/objective_*.jsonl).
# Defaults reproduce the original difficult-50 run exactly. [2026-09-03]
TD = (os.environ.get("TEST_DIR") or paths.ITEM_SOURCE)
TEST_SET = os.environ.get("TEST_SET", "difficult_50")
OUT = (os.environ.get("BLOCK_OUT") or f"{paths.DATA}/blockA_sarashina_difficult_4spk")
FAITHFUL2 = os.environ.get("SARA_ADAPTER_OURS", SP + "/adapters/lora_faithful2_tf457/ckpt_20000")
# Reference alignment: EVAL_SPEAKERS_JSON lets every backbone be driven from ONE speaker list
# (e.g. the two ITA references) instead of the bootstrap set. Entries may carry a "text"
# field; when present it is used verbatim as the prompt transcript instead of the ASR
# self-transcription below -- for a corpus reference the true transcript is known, and an
# ASR error in the prompt text degrades the clone for no reason. [2026-09-01]
SPK_JSON = os.environ.get("EVAL_SPEAKERS_JSON",
                          f"{paths.DATA}/eval_speakers/selected_4.json")
_sel = json.load(open(SPK_JSON))
SEL = _sel["selected_4"] if "selected_4" in _sel else _sel["selected"]
SEL = SEL[:args.nspk]
SPK_WAVS = [s["wav"] for s in SEL]

vanilla = [l.rstrip("\n") for l in open(f"{TD}/test_{TEST_SET}_vanilla.txt", encoding="utf-8") if l.strip()]
# difficult:011 (斟酌) carried the pre-correction heiban reading; the author's NHK verification
# made it atamadaka, so the synthesis input is taken from the corrected copy. [2026-08-29]
DIFF_LORA = os.environ.get("DIFF_LORA",
    f"{paths.REPO}/data/derived/test_difficult_50_lora_fixed.txt"
    if TEST_SET == "difficult_50" else f"{TD}/test_{TEST_SET}_lora.txt")
lora = [l.rstrip("\n").replace("<PHON_START>", PRON_START_TOKEN).replace("<PHON_END>", PRON_END_TOKEN)
        for l in open(DIFF_LORA, encoding="utf-8") if l.strip()]
yomi = [l.rstrip("\n") for l in open(f"{TD}/test_{TEST_SET}_yomi.txt", encoding="utf-8") if l.strip()]
n = min(len(vanilla), len(lora), len(yomi))
if args.limit:
    n = min(n, args.limit)
print(f"difficult sentences: {n}  speakers: {len(SPK_WAVS)}", flush=True)

CAN = str.maketrans({"ヂ": "ジ", "ヅ": "ズ", "ヴ": "ブ"})
def norm(s): return re.sub(r"[^ァ-ヶー]", "", s).translate(CAN)

g = SarashinaTTSGenerator(model_dir=SP + "/sara-tts-weights", watermark=False, use_vllm=False)
base = g.text_generator.model
base_state = {k: v.detach().clone() for k, v in base.state_dict().items()}
PS = g.tokenizer.convert_tokens_to_ids(PRON_START_TOKEN); PE = g.tokenizer.convert_tokens_to_ids(PRON_END_TOKEN)
kw = pipeline("automatic-speech-recognition", model="sbintuitions/kana-whisper", device=0,
              torch_dtype=torch.float16, generate_kwargs={"language": "ja", "task": "transcribe"})

# --- per-speaker prompt: self-transcribe (kana) + extract flow/codec-side prompt tensors ONCE ---
TAKES = []          # per-item guard record: takes used and reasons rejected (reported)
SPK = []
for _si, wav in enumerate(SPK_WAVS):
    known = (SEL[_si].get("text") or "").strip()
    if known:
        ptext = known
    else:
        a, srr = sf.read(wav, dtype="float32")
        ptext = str(kw({"raw": a, "sampling_rate": int(srr)})["text"]).strip()
    SPK.append({
        "wav": wav, "name": os.path.basename(wav), "ptext": ptext,
        "emb": g._extract_zero_shot_embedding(wav),
        "tok": g._extract_audio_prompt_tokens(wav),
        "feat": g._extract_audio_prompt_feat(wav),
    })
    print(f"  [spk] {os.path.basename(wav)} ptext='{ptext[:40]}...'", flush=True)
for i in range(len(SPK)):
    os.makedirs(os.path.join(OUT, f"spk{i}"), exist_ok=True)


def load_faithful2():
    base.load_state_dict(base_state)
    m = PeftModel.from_pretrained(base, FAITHFUL2).merge_and_unload()
    pe = torch.load(os.path.join(FAITHFUL2, "pron_emb.pt"), map_location="cpu")
    with torch.no_grad():
        ie = m.get_input_embeddings().weight
        ie[PS] = pe["emb"][0].to(ie.device, ie.dtype); ie[PE] = pe["emb"][1].to(ie.device, ie.dtype)
    m.config.use_cache = True; g.text_generator.model = m


# Uniform degeneracy guard (scripts/eval/degenerate.py). The previous check caught only
# too-short output; runaway/silent/repetition takes were scored as reading misses, at 2.4%
# of the edited arm vs 1.0% of kana. Same policy for every arm: redraw on a degenerate take,
# take the FIRST valid one (never the best), cap the retries, record the take count.
def synth(text, sp, seed, asr=None):
    def gen(sd):
        torch.manual_seed(sd)
        w = g.generate([text], flow_embedding=sp["emb"], audio_prompt_text=sp["ptext"],
                       audio_prompt_tokens=sp["tok"], audio_prompt_feat=sp["feat"], audio_prompt_path=sp["wav"],
                       gen_kwargs={"max_tokens": 500, "temperature": 0.9, "top_p": 0.95})[0]
        return w.squeeze().detach().cpu().numpy().astype("float32"), 24000
    a, _, info = synth_valid(gen, seed, asr=asr, cap_s=CAP_S)
    return a, info



# ITEM_SPK_JSON + ITEM_SPK_CELL: one reference voice per ITEM, assigned by hash before any
# audio was synthesized. Used for the 42-voice objective table, so no result depends on which
# one or two references were picked. Speaker si synthesizes only its assigned items; output
# layout stays spk{si}/{arm}/{i:02d}.wav. Unset -> unchanged behaviour. [2026-09-04]
_ASSIGN = None
if os.environ.get("ITEM_SPK_JSON"):
    _ASSIGN = json.load(open(os.environ["ITEM_SPK_JSON"]))["cells"][os.environ["ITEM_SPK_CELL"]]
    _ASSIGN = {int(k): set(v) for k, v in _ASSIGN.items()}
    print(f"[assign] per-item speakers: {sum(len(v) for v in _ASSIGN.values())} (item,speaker) pairs", flush=True)


def run(arm, texts, si):
    sp = SPK[si]
    d = os.path.join(OUT, f"spk{si}", arm); os.makedirs(d, exist_ok=True)
    cers = []
    for i in range(n):
        if _ASSIGN is not None and si not in _ASSIGN.get(i, ()):
            continue
        wp = os.path.join(d, f"{i:02d}.wav")
        _guard_hyp = None
        if os.path.exists(wp):
            a = sf.read(wp, dtype="float32")[0]
        else:
            # the ASR the guard uses for the repetition check is the SAME pass the metric uses
            a, ginfo = synth(texts[i], sp, seed=5000 + i,
                             asr=lambda w, r: str(safe_asr(lambda x, y: kw({"raw": x, "sampling_rate": y})["text"], w, r)))
            sf.write(wp, a, 24000)
            TAKES.append(dict(arm=arm, spk=si, idx=i, **ginfo)); _guard_hyp = ginfo.get("hyp")
            if ginfo["takes"] > 1:
                print(f"    guard spk{si}/{arm}/{i:02d}: {ginfo['takes']} takes {ginfo['rejected']}"
                      + (f" STILL {ginfo['final_reason']}" if ginfo["final_reason"] else ""), flush=True)
        # the guard already transcribed this waveform with this model -- do not pay for it twice
        hyp = norm(str(_guard_hyp if _guard_hyp is not None
                       else safe_asr(lambda w, r: kw({"raw": w, "sampling_rate": r})["text"], a, 24000)))
        ref = norm(pyopenjtalk.g2p(yomi[i], kana=True))
        cers.append(1 - difflib.SequenceMatcher(None, ref, hyp).ratio())
    c = np.array(cers)
    print(f"  spk{si}/{arm:7s}: Kana-CER {c.mean():.3f}  CER>0.5 {int((c>0.5).sum())}", flush=True)
    return c


res = {"n": n, "speakers": [s["name"] for s in SPK], "arms": {}}
# base arms across all speakers, then faithful2 once, then ours across all speakers
# ONLY_OURS=1 (diagnostics only): skip the base arms and synthesize the tag arm alone. Default unchanged.
_ONLY_OURS = bool(os.environ.get("ONLY_OURS"))
for arm, texts in ([] if _ONLY_OURS else [("noedit", vanilla), ("kana", yomi)]):
    res["arms"].setdefault(arm, {})
    for si in range(len(SPK)):
        c = run(arm, texts, si)
        res["arms"][arm][f"spk{si}"] = {"kana_cer_mean": float(c.mean()), "cer>0.5": int((c > 0.5).sum())}
load_faithful2()
res["arms"].setdefault("ours", {})
for si in range(len(SPK)):
    c = run("ours", lora, si)
    res["arms"]["ours"][f"spk{si}"] = {"kana_cer_mean": float(c.mean()), "cer>0.5": int((c > 0.5).sum())}

json.dump(res, open(os.path.join(OUT, "blockA_difficult_4spk.json"), "w"), ensure_ascii=False, indent=1)
json.dump(TAKES, open(os.path.join(OUT, "guard_takes.json"), "w"), ensure_ascii=False, indent=1)
print(f"\n== Sarashina Block A difficult, {len(SPK)} speakers (n={n}) ==", flush=True)
for arm in (("ours",) if _ONLY_OURS else ("noedit", "kana", "ours")):
    ms = [res["arms"][arm][f"spk{si}"]["kana_cer_mean"] for si in range(len(SPK))]
    print(f"  {arm:7s}: per-spk CER {[round(m,3) for m in ms]}  mean {np.mean(ms):.3f}", flush=True)
print(f"DONE -> {OUT}  (headline: accent by ear/rater across 4 speakers)", flush=True)
