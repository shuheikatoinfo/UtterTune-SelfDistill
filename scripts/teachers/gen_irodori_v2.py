"""v2 teacher generation for Irodori-TTS-500M-v3 (uniform recipe, 2026-09-16).

Same rule as the other backbones: the frozen base reads the PLAIN kanji carrier; a take is kept if it passes the degeneration guard (audio >= MIN_SEC) AND the
ASR of record hears the target reading; otherwise the seed is bumped, up to MAX_TAKES; a carrier
with no passing take is dropped and logged. The kept take's own codec latent is the teacher.

Speaker: Irodori's unconditional (no_ref) output is unintelligible (probed 2026-09-16: guidance 0/1/5,
20/32 steps, all noise), so like CosyVoice 2 it takes a reference; the rule shared by the two
speaker-conditioned backbones is the backbone's whole self-sampled voice bank cycled over carriers, as in
round 1 (CosyVoice 2: 20 fictional voices; Irodori: the first 2,000 anchor-bank references in name
order), disclosed as the one backbone-specific element. The voices are the released dataset's
shared voices ($DATASET_DIR/shared_voices/latents, unless --ref-dir names another directory).

Manifest schema matches the upstream train.py (text/latent_path/num_frames/speaker_id/original_text/...),
plus ref_latent_path, which irodori/upstream.patch conditions on.

    irodori/upstream/.venv/bin/python scripts/teachers/gen_irodori_v2.py \
        --base-checkpoint <model.safetensors> --text .../teachers_text_v3.jsonl \
        --out-latent-dir .../v2/irodori/latents --out-manifest .../v2/irodori/train.jsonl
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO + "/scripts/teachers")
sys.path.insert(0, os.path.join(REPO, "scripts"))  # for paths.py
import paths
sys.path.insert(0, os.path.join(REPO, "irodori", "upstream"))  # irodori_tts, pinned by irodori/setup.sh
import torch
from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest
from reading_check import Checker


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--text", type=Path, required=True)
    p.add_argument("--out-latent-dir", type=Path, required=True)
    p.add_argument("--out-manifest", type=Path, required=True)
    p.add_argument("--num-steps", type=int, default=20)
    p.add_argument("--cfg-scale-text", type=float, default=3.0)
    p.add_argument("--cfg-scale-speaker", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--min-sec", type=float, default=0.5)
    p.add_argument("--max-takes", type=int, default=5)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--model-device", default="cuda")
    p.add_argument("--model-precision", default="bf16")
    p.add_argument("--codec-device", default="cuda")
    p.add_argument("--codec-precision", default="fp32")
    p.add_argument("--codec-repo", default="Aratako/Semantic-DACVAE-Japanese-32dim")
    p.add_argument("--ref-dir", type=Path, help="voice reference latents (default: $DATASET_DIR/shared_voices/latents)")
    p.add_argument("--num-voices", type=int, default=2000)
    p.add_argument("--shard", default="0/1", help="i/n: process rows i::n (merge shards afterwards); voice index stays the global row index")
    p.add_argument("--name-prefix", default="", help="latent file name prefix (per shard)")
    a = p.parse_args()
    if a.ref_dir is None:
        a.ref_dir = Path(paths.DATASET_DIR) / "shared_voices" / "latents"
    refs = sorted(a.ref_dir.glob("bank_spk_*.pt"))[: a.num_voices]
    assert len(refs) == a.num_voices, f"need {a.num_voices} refs under {a.ref_dir}, found {len(refs)}"
    a.out_latent_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in a.text.read_text(encoding="utf-8").splitlines() if l.strip()]
    for gi, row in enumerate(rows):
        row["_gi"] = gi
    si, sn = (int(x) for x in a.shard.split("/"))
    rows = rows[si::sn]
    if a.limit:
        rows = rows[:a.limit]
    runtime = InferenceRuntime.from_key(RuntimeKey(
        checkpoint=str(a.base_checkpoint), model_device=a.model_device, model_precision=a.model_precision,
        codec_device=a.codec_device, codec_precision=a.codec_precision, codec_repo=a.codec_repo))
    # The teachers were made in an environment without SilentCipher; the pinned environment has it, and
    # a watermark in the teacher audio would end up in the teacher latents.
    runtime.watermarker.model = None
    codec = runtime.codec
    checker = Checker(device=0)
    written, dropped, takes_hist, drops = 0, 0, {}, []
    t0 = time.time()
    with a.out_manifest.open("w", encoding="utf-8") as mf:
        for li, row in enumerate(rows):
            gi = row["_gi"]
            audio, sr, hyp, used = None, None, "", None
            fails = []
            ref = refs[gi % len(refs)]
            for t in range(a.max_takes):
                res = runtime.synthesize(SamplingRequest(
                    text=str(row["plain"]), ref_latent=str(ref), no_ref=False, num_steps=a.num_steps,
                    cfg_scale_text=a.cfg_scale_text, cfg_scale_speaker=a.cfg_scale_speaker,
                    seed=a.seed + gi + 100000 * t))
                x = res.audio
                if x.dim() == 1:
                    x = x.view(1, 1, -1)
                elif x.dim() == 2:
                    x = x.unsqueeze(0)
                sr = int(res.sample_rate)
                guard = x.shape[-1] >= int(a.min_sec * sr)
                heard, hyp = (False, "")
                if guard:
                    heard, hyp = checker.heard(x[0, 0].float().cpu().numpy(), sr, row["reading"])
                if guard and heard:
                    audio, used = x, t
                    break
                fails.append({"take": t, "guard": bool(guard), "hyp": hyp})
            if audio is None:
                dropped += 1; drops.append({"word": row["word"], "plain": row["plain"], "fails": fails}); continue
            with torch.inference_mode():
                lat = codec.encode_waveform(audio.to(codec.device), sr)[0].detach().cpu().contiguous()
            name = f"{a.name_prefix}{written:07d}.pt"; torch.save(lat, a.out_latent_dir / name)
            mf.write(json.dumps({"text": str(row["phon"]), "latent_path": str((a.out_latent_dir / name).resolve()),
                                 "num_frames": int(lat.shape[0]), "speaker_id": f"anchor:{ref.stem}",
                                 "original_text": str(row["plain"]), "target_surface": str(row["word"]),
                                 "reading_accented": str(row["reading"]), "ref_latent_path": str(ref.resolve()),
                                 "take": used, "hyp": hyp}, ensure_ascii=False) + "\n")
            written += 1; takes_hist[used] = takes_hist.get(used, 0) + 1
            if li % 50 == 0:
                print(f"[gen] {li}/{len(rows)} written {written} dropped {dropped}  {time.time()-t0:.0f}s", flush=True)
    json.dump({"rule": "anchor-bank voices cycled (num_voices); first take passing guard (>=min_sec) and reading check; max_takes", "max_takes": a.max_takes, "voices": [r.stem for r in refs],
               "kept": written, "dropped": dropped, "words_in": len({r['word'] for r in rows}),
               "takes_hist": {str(k): v for k, v in sorted(takes_hist.items())}, "dropped_rows": drops},
              open(str(a.out_manifest) + ".provenance.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[gen] DONE wrote {written} rows ({dropped} dropped) in {time.time()-t0:.0f}s -> {a.out_manifest}", flush=True)


if __name__ == "__main__":
    main()
