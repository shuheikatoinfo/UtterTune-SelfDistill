"""CosyVoice2 / CosyVoice3 synthesis backend for the base-measurement benches.

One code path drives both bases via the *main* CosyVoice repo's `AutoModel` (which selects
CosyVoice2 vs CosyVoice3 by which cosyvoice{2,3}.yaml sits in model_dir). This is deliberately
the plain-text path: base READING/ACCENT measurement needs NO PHON tokens and NO LoRA — those
belong to the later self-distill notation arm (docs/cv-extension.md §5).

Env overrides:
  CV_ROOT   main CosyVoice checkout (default $COSYVOICE_DIR)
  CV2_DIR   CosyVoice2-0.5B model dir (default $COSYVOICE_DIR/pretrained_models/CosyVoice2-0.5B)
  CV3_DIR   Fun-CosyVoice3-0.5B-2512 model dir
  CV_PROMPT_WAV  fixed reference voice (default {CV_ROOT}/asset/zero_shot_prompt.wav)

Voice note: reading/accent are judged on OUTPUT CONTENT (kana-CER via whisper, F0 peak), so the
voice only needs to be fixed and intelligible. Default = CosyVoice's shipped zero-shot prompt.
Pass a Japanese prompt wav (e.g. the Sarashina narration prompt) via CV_PROMPT_WAV to share one
voice across bases (CAM++ speaker space is identical across Sarashina/CV2/CV3 — see §1).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/, for paths.py
import paths

CV_ROOT = (os.environ.get("CV_ROOT") or paths.COSYVOICE_DIR)
sys.path.insert(0, CV_ROOT)
sys.path.insert(0, os.path.join(CV_ROOT, "third_party/Matcha-TTS"))

MODEL_DIRS = {
    "cv2": (os.environ.get("CV2_DIR") or f"{paths.COSYVOICE_DIR}/pretrained_models/CosyVoice2-0.5B"),
    "cv3": (os.environ.get("CV3_DIR") or f"{paths.COSYVOICE_DIR}/pretrained_models/Fun-CosyVoice3-0.5B-2512"),
}

# CV3's Japanese examples prepend an instruction prefix (test_ja_inference.py); CV2 does not.
# Only used in cross_lingual mode; zero_shot supplies a real JP prompt+transcript instead.
PREFIX = {"cv2": "", "cv3": "You are a helpful assistant.<|endofprompt|>"}

# Voice-matched to the Sarashina base bench (same wav + transcript) so CV↔Sarashina is
# apples-to-apples and the Chinese-accent / edge-artifact confounds of the shipped Chinese
# cross_lingual prompt are removed. CAM++ space is shared across bases, so this is a valid swap.
JP_PROMPT_WAV = f"{paths.DATA}/_handoff/sara-ref/samples/styles/ナレーション_prompt.wav"
JP_PROMPT_TEXT = "彼らは王権の正当性を重視し革命の影響を抑え込むことを目指しました。"

# Fictional speaker (zero reference audio): a voice bootstrap-sampled from the CAM++ speaker space
# on Sarashina (random z -> flow_embedding_only synth -> CAM++ re-embed -> iterate -> Gaussian fit
# -> sample). CAM++ encoder is shared across bases, so its embedding plugs straight into CV2/CV3.
# Used via the embedding-only path (inference_sft: llm_embedding=flow_embedding, NO prompt speech
# tokens) -> removes BOTH the Chinese timbre AND the Chinese in-context prompt-token anchor.
FICTIONAL_WAV = f"{paths.DATA}/bootstrap_speakers/fictional0.wav"
FICTIONAL_TEXT = "こんにちは。今日はとてもいい天気ですね。"  # the bootstrap SENT spoken in fictional*.wav


def _vad_prompt(src=FICTIONAL_WAV):
    """UtterTune-paper prompt prep: VAD the prompt clip itself (the paper VAD'd its 48 prompt
    clips and kept those where speech was detected; ≤4 s recommended). Cached next to the source."""
    dst = os.path.splitext(src)[0] + "_vad.wav"
    if not os.path.exists(dst):
        import soundfile as sf
        wav, sr = sf.read(src, dtype="float32")
        sf.write(dst, _trim_edges(wav, sr), sr)
    return dst


def make_measurement_synth(backend, **overrides):
    """The settled measurement configuration (docs/cv-extension.md §6) — aligned with the
    UtterTune paper's protocol (user's call): SHIPPED default inference config (no custom levers;
    they were verified CER-neutral anyway, so nothing is lost), fictional speaker as a VAD'd
    zero_shot prompt anchor (reference-audio-free), bidirectional VAD on outputs. Runaway tails are
    handled post-hoc by the VAD (8.53s->3.03s verified); mid-utterance garbage is handled at
    aggregation (CER>0.5 / UTMOSv2<2.0 drops, rates disclosed) exactly as in the paper. This makes
    our CV2 numbers directly comparable to the published UtterTune results. No take selection —
    best-of-N lives in cv_quality.py, teacher generation only.

    Reference alignment: CV_PROMPT_WAV (+ CV_PROMPT_TEXT) overrides the fictional prompt so one
    reference voice can be shared across every backbone. The prompt is VAD'd with the SAME
    _vad_prompt() prep either way, so only the clip changes. Unset -> unchanged default."""
    env_wav = os.environ.get("CV_PROMPT_WAV")
    if env_wav:
        prompt_wav, prompt_text = _vad_prompt(env_wav), os.environ.get("CV_PROMPT_TEXT")
        if not prompt_text:
            raise SystemExit("CV_PROMPT_WAV set without CV_PROMPT_TEXT: zero_shot needs the "
                             "prompt transcript, and a wrong one silently degrades the clone.")
    else:
        prompt_wav, prompt_text = _vad_prompt(), FICTIONAL_TEXT
    # CV2_TRIM=0 disables the output VAD. It exists because the trim is the ONE piece of
    # post-processing that is not uniform across backbones -- Sarashina, Irodori and T5Gemma
    # get none -- so measuring with it off makes all four identical and removes a step that
    # would otherwise have to be disclosed and defended. Default is unchanged (on).
    trim = os.environ.get("CV2_TRIM", "1") != "0"
    cfg = dict(mode="zero_shot", prompt_wav=prompt_wav, prompt_text=prompt_text, trim=trim)
    cfg.update(overrides)
    return CVSynth(backend, **cfg)


def _trim_edges(wav, sr, trigger_level=7.0):
    """Bidirectional VAD trim of the synthesized output — the exact post-processing the UtterTune
    paper applied to synthesized speech ('we observed that CosyVoice 2 occasionally generates
    non-speech'): torchaudio.functional.vad forward, then on the time-reversed signal. Disclosed,
    fixed, applied uniformly to every arm — legitimate for reported measurements (unlike take
    selection). Falls back to the input if VAD eats everything."""
    import torch, torchaudio
    if wav.size == 0:
        return wav
    x = torch.from_numpy(wav).unsqueeze(0)
    t = torchaudio.functional.vad(x, sr, trigger_level=trigger_level)
    if t.shape[-1] > 0:
        t = torchaudio.functional.vad(t.flip(-1), sr, trigger_level=trigger_level).flip(-1)
    if t.shape[-1] == 0:
        return wav
    return t.squeeze(0).numpy()


class CVSynth:
    def _apply_quality_levers(self, max_token_text_ratio, sampling):
        """Inject the DETERMINISTIC artifact-suppression levers the top-level API does not thread
        through (docs/cv-extension.md §6). Both optional; None = shipped default. These are a fixed
        inference config applied identically to every arm (base & adapter), so — unlike best-of-N
        take rejection — they do NOT cherry-pick and are legitimate for reported measurements.
          max_token_text_ratio (default 20): caps LLM generation length = (text-prompt)*ratio.
            Lowering to ~8-12 kills runaway tails / music-at-the-end. [llm.inference]
          sampling (default 25): top_k for the random-sampling fallback in RAS. Lower = more
            conservative token choice = fewer hallucinations. [llm.inference]
        (n_timesteps is NOT exposed by CV2's CausalMaskedDiffWithXvec.inference — hardcoded=10.)"""
        import functools
        llm = self.model.model.llm
        if (max_token_text_ratio is not None or sampling is not None) and not getattr(llm, "_qpatched", False):
            kw = {}
            if max_token_text_ratio is not None:
                kw["max_token_text_ratio"] = max_token_text_ratio
            if sampling is not None:
                kw["sampling"] = sampling
            llm.inference = functools.partial(llm.inference, **kw)
            llm._qpatched = True

    def _attach_lora(self, lora_dir):
        """Load a self-distilled UtterTune LoRA + embed_patch onto the CV2 llm (mirrors UtterTune
        infer.py): register <PHON_*> tokens on the frontend tokenizer, resize embeddings, attach the
        PeftModel to cv2.model.llm, then overwrite the 2 new input-embedding rows from embed_patch."""
        import torch
        from peft import PeftModel
        from safetensors.torch import load_file
        from cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer
        base_dir = MODEL_DIRS[self.backend]
        tok = get_qwen_tokenizer(token_path=f"{base_dir}/CosyVoice-BlankEN", skip_special_tokens=True)
        NEW = ["<PHON_START>", "<PHON_END>"]
        tok.tokenizer.add_special_tokens({"additional_special_tokens": NEW})
        tok.special_tokens["additional_special_tokens"].extend(
            [t for t in NEW if t not in tok.special_tokens["additional_special_tokens"]])
        base_llm = self.model.model.llm
        base_llm.llm.model.resize_token_embeddings(len(tok.tokenizer))
        new_ids = tok.tokenizer.convert_tokens_to_ids(NEW)
        hf = PeftModel.from_pretrained(base_llm, lora_dir, is_trainable=False, torch_dtype=torch.float32)
        rows = load_file(os.path.join(lora_dir, "embed_patch.safetensors"))["embed_rows"].to(
            self.model.frontend.device)
        with torch.no_grad():
            hf.base_model.llm.model.get_input_embeddings().weight[new_ids] = rows
        self.model.model.llm = hf
        self.model.frontend.tokenizer = tok

    def _register_phon_untrained(self):
        """Register <PHON_*> tokens + resize embeddings but attach NO LoRA/embed_patch — the
        'mechanism present, untrained' control (Sarashina's shipped state: pron tokens exist but
        their embeddings are ~vocab-mean init and do nothing). For the self-distill-on/off ablation."""
        from cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer
        tok = get_qwen_tokenizer(token_path=f"{MODEL_DIRS[self.backend]}/CosyVoice-BlankEN",
                                 skip_special_tokens=True)
        NEW = ["<PHON_START>", "<PHON_END>"]
        tok.tokenizer.add_special_tokens({"additional_special_tokens": NEW})
        tok.special_tokens["additional_special_tokens"].extend(
            [t for t in NEW if t not in tok.special_tokens["additional_special_tokens"]])
        self.model.model.llm.llm.model.resize_token_embeddings(len(tok.tokenizer))
        self.model.frontend.tokenizer = tok

    def __init__(self, backend, prompt_wav=None, prompt_text=None, mode="fictional",
                 spk_wav=None, spk_emb=None, trim=True, fp16=False,
                 max_token_text_ratio=None, sampling=None, lora_dir=None):
        """mode:
          'fictional'     — embedding-only (inference_sft), speaker sampled from CAM++ space, NO
                            prompt speech tokens, zero reference audio. Removes the Chinese timbre
                            AND the Chinese in-context anchor. Default. (spk_wav to re-embed, or a
                            precomputed spk_emb [1,192].)
          'zero_shot'     — JP prompt wav + transcript (in-language anchor; uses prompt tokens).
          'cross_lingual' — shipped Chinese prompt (the original honest-default arm)."""
        assert backend in MODEL_DIRS, f"backend must be one of {list(MODEL_DIRS)}"
        assert mode in ("fictional", "zero_shot", "cross_lingual")
        import torch  # noqa
        from cosyvoice.cli.cosyvoice import AutoModel
        self.backend = backend
        self.mode = mode
        self.trim = trim
        self.torch = torch
        self.model = AutoModel(model_dir=MODEL_DIRS[backend], fp16=fp16)
        self.sr = self.model.sample_rate  # 24000 for CV2/CV3
        env_wav = os.environ.get("CV_PROMPT_WAV")
        self.prefix = PREFIX[backend]
        if mode == "fictional":
            if spk_emb is not None:
                emb = spk_emb if torch.is_tensor(spk_emb) else torch.tensor(spk_emb)
                emb = emb.to(self.model.frontend.device)
            else:
                emb = self.model.frontend._extract_spk_embedding(spk_wav or env_wav or FICTIONAL_WAV)
            self.spk_emb = emb
            self._sft_id = "__fic__"
            self.model.frontend.spk2info[self._sft_id] = {"embedding": emb}
        elif mode == "zero_shot":
            self.prompt_wav = prompt_wav or env_wav or JP_PROMPT_WAV
            self.prompt_text = prompt_text or os.environ.get("CV_PROMPT_TEXT", JP_PROMPT_TEXT)
        else:  # cross_lingual (default Chinese prompt)
            self.prompt_wav = prompt_wav or env_wav or os.path.join(CV_ROOT, "asset/zero_shot_prompt.wav")
            self.prompt_text = None
        self._apply_quality_levers(max_token_text_ratio, sampling)
        if lora_dir:
            self._attach_lora(lora_dir)

    def synth(self, text, text_frontend=True, seed=None):
        """Synthesize one sentence -> (float32 np.ndarray @ self.sr).

        text_frontend=True  : honest base path (WeText normalizer runs; note it treats kanji as
                              Chinese -> this is exactly what base READING measurement should expose).
        text_frontend=False : raw string passed through unchanged (for later <PHON_*> notation arm).
        """
        if seed is not None:
            self.torch.manual_seed(seed)
        chunks = []
        if self.mode == "fictional":
            gen = self.model.inference_sft(
                text, self._sft_id, stream=False, text_frontend=text_frontend)
        elif self.mode == "zero_shot":
            gen = self.model.inference_zero_shot(
                text, self.prompt_text, self.prompt_wav, stream=False, text_frontend=text_frontend)
        else:
            gen = self.model.inference_cross_lingual(
                self.prefix + text, self.prompt_wav, stream=False, text_frontend=text_frontend)
        for out in gen:
            chunks.append(out["tts_speech"])
        wav = self.torch.cat(chunks, dim=-1).squeeze().detach().cpu().numpy().astype("float32")
        if self.trim:
            wav = _trim_edges(wav, self.sr)
        return wav
