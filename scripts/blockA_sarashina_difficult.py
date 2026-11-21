"""Block A (backbone-agnostic) — Sarashina2.2-TTS, difficult set, No-edit vs Ours.
Headline artifact: the same 49 difficult-word sentences synthesized (a) plain kanji on the base
(No-edit) and (b) with the UtterTune ' notation edit + the self-distilled faithful2 adapter (Ours).
Wavs feed the external raters' accent judgment (§4.2) and the ear; kana-whisper CER (vs the
katakana-reading reference) is the auxiliary reading signal. One speaker (pilot; the 4 UMAP
representative speakers come later).
"""
import sys, os, re, json, difflib
import paths
SP = f"{paths.DATA}/_handoff"
sys.path.insert(0, paths.SARASHINA_DIR)
import torch, numpy as np, soundfile as sf
from peft import PeftModel
from sarashina_tts.additional_tokens import PRON_START_TOKEN, PRON_END_TOKEN
from sarashina_tts.generate.generate import SarashinaTTSGenerator
from transformers import pipeline
import pyopenjtalk

TD = paths.ITEM_SOURCE
OUT = f"{paths.DATA}/blockA_sarashina_difficult"
for d in ("noedit", "kana", "ours"):
    os.makedirs(os.path.join(OUT, d), exist_ok=True)
FAITHFUL2 = SP + "/lora_faithful2_tf457/ckpt_20000"

vanilla = [l.rstrip("\n") for l in open(f"{TD}/test_difficult_50_vanilla.txt", encoding="utf-8") if l.strip()]
# the _lora file uses CV2/UtterTune token names <PHON_START>/<PHON_END>; Sarashina's own tokens are
# <|pron_start|>/<|pron_end|> (else the tokenizer reads the tag literally, "ピーエイチオンスタート").
lora = [l.rstrip("\n").replace("<PHON_START>", PRON_START_TOKEN).replace("<PHON_END>", PRON_END_TOKEN)
        for l in open(f"{TD}/test_difficult_50_lora.txt", encoding="utf-8") if l.strip()]
yomi = [l.rstrip("\n") for l in open(f"{TD}/test_difficult_50_yomi.txt", encoding="utf-8") if l.strip()]
n = min(len(vanilla), len(lora), len(yomi))
print(f"difficult sentences: {n}", flush=True)

REF = SP + "/sara-ref/samples/styles/ナレーション_prompt.wav"
REFTEXT = "彼らは王権の正当性を重視し革命の影響を抑え込むことを目指しました"
CAN = str.maketrans({"ヂ": "ジ", "ヅ": "ズ", "ヴ": "ブ"})
def norm(s): return re.sub(r"[^ァ-ヶー]", "", s).translate(CAN)

g = SarashinaTTSGenerator(model_dir=SP + "/sara-tts-weights", watermark=False, use_vllm=False)
base = g.text_generator.model
base_state = {k: v.detach().clone() for k, v in base.state_dict().items()}
PS = g.tokenizer.convert_tokens_to_ids(PRON_START_TOKEN); PE = g.tokenizer.convert_tokens_to_ids(PRON_END_TOKEN)
emb = g._extract_zero_shot_embedding(REF); tok = g._extract_audio_prompt_tokens(REF); feat = g._extract_audio_prompt_feat(REF)
kw = pipeline("automatic-speech-recognition", model="sbintuitions/kana-whisper", device=0,
              torch_dtype=torch.float16, generate_kwargs={"language": "ja", "task": "transcribe"})


def load_faithful2():
    base.load_state_dict(base_state)
    m = PeftModel.from_pretrained(base, FAITHFUL2).merge_and_unload()
    pe = torch.load(os.path.join(FAITHFUL2, "pron_emb.pt"), map_location="cpu")
    with torch.no_grad():
        ie = m.get_input_embeddings().weight
        ie[PS] = pe["emb"][0].to(ie.device, ie.dtype); ie[PE] = pe["emb"][1].to(ie.device, ie.dtype)
    m.config.use_cache = True; g.text_generator.model = m


def synth(text, seed):
    torch.manual_seed(seed)
    for _ in range(6):
        w = g.generate([text], flow_embedding=emb, audio_prompt_text=REFTEXT, audio_prompt_tokens=tok,
                       audio_prompt_feat=feat, audio_prompt_path=REF,
                       gen_kwargs={"max_tokens": 500, "temperature": 0.9, "top_p": 0.95})[0]
        a = w.squeeze().detach().cpu().numpy().astype("float32")
        if len(a) >= int(0.5 * 24000):
            return a
    return a


def run(arm, texts):
    if arm == "ours":
        load_faithful2()
    cers = []
    for i in range(n):
        wp = os.path.join(OUT, arm, f"{i:02d}.wav")
        if os.path.exists(wp):
            a = sf.read(wp, dtype="float32")[0]  # reuse cached wav (No-edit/Ours already synthesized)
        else:
            a = synth(texts[i], seed=5000 + i); sf.write(wp, a, 24000)
        hyp = norm(str(kw({"raw": a, "sampling_rate": 24000})["text"]))
        # yomi line = target word in kana + rest in kanji; pyopenjtalk -> full kana reference
        ref = norm(pyopenjtalk.g2p(yomi[i], kana=True))
        c = 1 - difflib.SequenceMatcher(None, ref, hyp).ratio()
        cers.append(c)
        if i % 10 == 0:
            print(f"  {arm} {i}/{n} cer={c:.2f}", flush=True)
    return np.array(cers)


# 3 arms: No-edit (plain kanji) / kana-baseline (target->katakana, no tags) / Ours (edit + faithful2 ')
ce_noedit = run("noedit", vanilla)
ce_kana = run("kana", yomi)   # base, no adapter — the UtterTune kana-input baseline (fixes reading, not accent)
ce_ours = run("ours", lora)
res = {a: {"kana_cer_mean": float(c.mean()), "cer>0.5": int((c > 0.5).sum())}
       for a, c in [("noedit", ce_noedit), ("kana", ce_kana), ("ours", ce_ours)]}
res["n"] = n
json.dump(res, open(os.path.join(OUT, "blockA_difficult.json"), "w"), ensure_ascii=False, indent=1)
print(f"\n== Sarashina Block A difficult (n={n}) — reading Kana-CER (accent = ear/rater) ==", flush=True)
for a, c in [("No-edit (kanji)", ce_noedit), ("kana-baseline", ce_kana), ("Ours (edit+') ", ce_ours)]:
    print(f"  {a:16s}: Kana-CER {c.mean():.3f}  CER>0.5 {int((c>0.5).sum())}", flush=True)
print(f"DONE -> {OUT}  (kana-baseline fixes reading but NOT accent; Ours ' is the accent claim)", flush=True)
