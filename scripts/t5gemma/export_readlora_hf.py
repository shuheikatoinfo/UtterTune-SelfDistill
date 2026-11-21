"""STEP D - export the reading-only LoRA bundle -> standalone merged HF dir.

The repo's export_t5gemma_voice_hf_lora.py calls model.carefully_load_state_dict(ckpt["model"]).
That remap is written for loading a VANILLA ckpt into a LoRA model (backbone.* -> backbone.base_model.
model.*). But a LoRA TRAINING bundle already has PEFT-format keys (backbone.base_model.model....,
lora_A/lora_B, .base_layer.weight); feeding those through the same remap DOUBLE-prefixes them and the
adapter silently fails to load (merge becomes a no-op). We instead build the same LoRA-wrapped model
and load DIRECTLY (keys already match), verifying LoRA-key overlap before trusting the merge.
"""
import argparse, os, pathlib, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/, for paths.py
import paths
import torch
from transformers.models.t5gemma import T5GemmaConfig

REPO = paths.T5GEMMA_DIR
sys.path.insert(0, REPO)
from config import apply_repo_defaults
from models.t5gemma import T5GemmaVoiceModel
from hf_export.configuration_t5gemma_voice import T5GemmaVoiceConfig
from hf_export.modeling_t5gemma_voice import T5GemmaVoiceForConditionalGeneration


def _dtype(p):
    p = str(p).lower()
    return {"float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}.get(p, torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base_repo", default="google/t5gemma-2b-2b-ul2")
    ap.add_argument("--save_adapter_dir", default=None)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ns = ckpt["args"]
    setattr(ns, "use_lora", 1)
    ns = apply_repo_defaults(ns)
    target_dtype = _dtype(getattr(ns, "precision", "float32"))

    model = T5GemmaVoiceModel(ns)  # LoRA-wrapped -> keys already in PEFT format

    sd = ckpt["model"]
    own = set(model.state_dict().keys())
    ckd = set(sd.keys())
    inter = own & ckd
    lora_ck = {k for k in ckd if "lora_" in k}
    lora_hit = {k for k in lora_ck if k in own}
    print(f"[keys] ckpt={len(ckd)} model={len(own)} intersect={len(inter)} "
          f"lora_in_ckpt={len(lora_ck)} lora_matched={len(lora_hit)}")
    direct_ok = len(inter) > 0.9 * len(own) and len(lora_hit) == len(lora_ck) and len(lora_ck) > 0
    if direct_ok:
        res = model.load_state_dict({k: v.to(torch.float32) if v.is_floating_point() else v
                                     for k, v in sd.items()}, strict=False)
        print(f"[direct load] missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")
        try:
            tdt = next(model.parameters()).dtype
            model.to(dtype=tdt)
        except StopIteration:
            pass
    else:
        print("[fallback] direct-load overlap poor -> carefully_load_state_dict remap")
        model.carefully_load_state_dict(sd)

    # sanity: LoRA B weights should be non-zero after a real training run
    nz = 0
    for n, p in model.named_parameters():
        if "lora_B" in n and p.abs().sum().item() > 0:
            nz += 1
    print(f"[sanity] nonzero lora_B tensors = {nz}")

    if args.save_adapter_dir:
        os.makedirs(args.save_adapter_dir, exist_ok=True)
        model.backbone.save_pretrained(args.save_adapter_dir, safe_serialization=True)
        print(f"[adapter] saved -> {args.save_adapter_dir}")

    model.backbone = model.backbone.merge_and_unload()

    base_cfg = T5GemmaConfig.from_pretrained(args.base_repo)
    for attr in ("tie_word_embeddings", "tie_input_output_embeddings", "tie_encoder_decoder"):
        setattr(base_cfg, attr, False)
        for side in ("encoder", "decoder"):
            if hasattr(base_cfg, side):
                setattr(getattr(base_cfg, side), attr, False)
    prune = getattr(ns, "prune_text_modules", 0)
    drop_lm_head = prune >= 1 or getattr(ns, "drop_lm_head", 0)
    drop_dec_embed = prune >= 2

    pv = int(getattr(ns, "audio_vocab_size", 65536))
    voice_cfg = T5GemmaVoiceConfig(
        t5gemma_model_name=args.base_repo, t5_config_dict=base_cfg.to_dict(),
        attn_implementation=getattr(ns, "attn_implementation", "eager"),
        precision=getattr(ns, "precision", "float32"), prune_text_modules=prune,
        use_pm_rope=getattr(ns, "use_pm_rope", 1), tie_word_embeddings=False,
        tie_input_output_embeddings=False, n_codebooks=getattr(ns, "n_codebooks", 1),
        audio_vocab_size=pv, n_special=getattr(ns, "n_special", 5),
        empty_token=getattr(ns, "empty_token", pv), eog=getattr(ns, "eog", pv + 1),
        eos=getattr(ns, "eos", pv + 3), audio_pad_token=getattr(ns, "audio_pad_token", pv + 2),
        audio_mask_token=getattr(ns, "audio_mask_token", 1024),
        y_sep_token=getattr(ns, "y_sep_token", pv + 4), x_sep_token=getattr(ns, "x_sep_token", 255999),
        special_first=getattr(ns, "special_first", 0), encodec_sr=getattr(ns, "encodec_sr", 50.0),
        progress_scale=getattr(ns, "progress_scale", 2000.0),
        progress_lookahead_secs=getattr(ns, "progress_lookahead_secs", 2.0),
        extra_cutoff=getattr(ns, "extra_cutoff", 5.0),
        text_guard_frames_per_token=getattr(ns, "text_guard_frames_per_token", 0),
        add_eos_to_text=getattr(ns, "add_eos_to_text", 0),
        add_bos_to_text=getattr(ns, "add_bos_to_text", 0),
        parallel_pattern=getattr(ns, "parallel_pattern", 0),
        audio_max_length=getattr(ns, "audio_max_length", 40.0),
        audio_tokenizer=getattr(ns, "audio_tokenizer", "xcodec2"),
        xcodec2_model_name=getattr(ns, "xcodec2_model_name", None),
        codec_audio_sr=getattr(ns, "codec_audio_sr", None),
        text_tokenizer_name=getattr(ns, "text_tokenizer_name", None),
    )
    hf_model = T5GemmaVoiceForConditionalGeneration(voice_cfg)
    hf_model.to(dtype=target_dtype)

    state = model.state_dict()  # merged
    if drop_lm_head:
        for k in [k for k in state if k.startswith("backbone.lm_head.")]:
            state.pop(k)
    if drop_dec_embed:
        for k in [k for k in state if k.startswith("backbone.model.decoder.embed_tokens.")]:
            state.pop(k)
    r = hf_model.load_state_dict(state, strict=False)
    print(f"[hf load] missing={len(r.missing_keys)} unexpected={len(r.unexpected_keys)}")

    os.makedirs(args.out, exist_ok=True)
    hf_model.save_pretrained(args.out, safe_serialization=True)
    voice_cfg.save_pretrained(args.out)
    shutil.copy(os.path.join(REPO, "hf_export/modeling_t5gemma_voice.py"),
                os.path.join(args.out, "modeling_t5gemma_voice.py"))
    shutil.copy(os.path.join(REPO, "hf_export/configuration_t5gemma_voice.py"),
                os.path.join(args.out, "configuration_t5gemma_voice.py"))
    print(f"[Done] merged LoRA HF model -> {args.out}")


if __name__ == "__main__":
    main()
