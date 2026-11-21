#!/usr/bin/env python3
"""Synthesize one item set with Irodori-TTS for the evaluation scripts, under the degeneracy guard.

    TEST_DIR=D TEST_SET=S ITEM_SPK_JSON=A ITEM_SPK_CELL=C EVAL_SPEAKERS_JSON=V BLOCKB1_OUT=O \
        python run_blockB1.py --arms ours --phon-checkpoint CKPT [--skip-eval]

Arms, each reading its own line file D/test_<S>_<kind>.txt, one line per item:
    noedit  base model, plain text   (kind "vanilla")
    kana    base model, kana reading (kind "yomi")
    ours    --phon-checkpoint, tagged text (kind "lora"), "/" phrase marks removed
Item i is read by voice A["cells"][C][str(i)][0] of V["selected"], with seed 5000 + i; a
degenerate take is redrawn with the seed bumped by 1000 (scripts/eval/degenerate.py). SET is
accepted and ignored: the item set is named by TEST_DIR and TEST_SET.

Writes O/<arm>/<i:02d>.wav (48 kHz, 16-bit), O/guard_takes.json (one record per item and arm:
idx, arm, takes, rejected, final_reason, hyp = the guard's kana-whisper transcript of the kept
take) and O/blockB1_irodori.json (inputs and settings of the run). Scoring is left to
scripts/eval (round3_choose.py reads guard_takes.json), so --skip-eval changes nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import soundfile as sf

from synthesis import kana_whisper_transcriber, load_runtime, synthesize

SEED_BASE = 5000
LINE_FILE_KIND = {"noedit": "vanilla", "kana": "yomi", "ours": "lora"}


def read_lines(path: Path) -> list[str]:
    return [line.rstrip("\n") for line in path.open(encoding="utf-8") if line.strip()]


def item_reference_wavs(n_items: int) -> list[str]:
    """Reference wav per item from the speaker assignment; every item must have one."""
    cell = json.load(open(os.environ["ITEM_SPK_JSON"]))["cells"][os.environ["ITEM_SPK_CELL"]]
    voices = json.load(open(os.environ["EVAL_SPEAKERS_JSON"]))["selected"]
    missing = [i for i in range(n_items) if str(i) not in cell]
    if missing:
        raise ValueError(f"no voice assigned to items {missing[:10]}")
    return [voices[cell[str(i)][0]]["wav"] for i in range(n_items)]


def sha256_prefix(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def main() -> None:
    from degenerate import MAX_TAKES, synth_valid  # importable once synthesis set the path

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--arms", nargs="+", required=True, help="noedit, kana, ours (space or comma separated)")
    parser.add_argument("--phon-checkpoint", type=Path, help="fine-tuned checkpoint for the ours arm")
    parser.add_argument("--skip-eval", action="store_true", help="accepted; this script never scores")
    args = parser.parse_args()
    arms = [a for part in args.arms for a in part.split(",") if a]
    unknown = sorted(set(arms) - set(LINE_FILE_KIND))
    if unknown:
        parser.error(f"unknown arms {unknown}; choose from {sorted(LINE_FILE_KIND)}")
    if "ours" in arms and args.phon_checkpoint is None:
        parser.error("--arms ours needs --phon-checkpoint")

    test_dir, test_set = Path(os.environ["TEST_DIR"]), os.environ["TEST_SET"]
    out_dir = Path(os.environ["BLOCKB1_OUT"])
    texts = {a: read_lines(test_dir / f"test_{test_set}_{LINE_FILE_KIND[a]}.txt") for a in arms}
    if "ours" in texts:
        # Reviewed readings may mark a phrase boundary with "/" (ガ'ク/カンセツショー). The tag
        # notation the model was trained on has no such mark, and the paper's runs removed it.
        texts["ours"] = [line.replace("/", "") for line in texts["ours"]]
    n_items = len(next(iter(texts.values())))
    if any(len(t) != n_items for t in texts.values()):
        raise ValueError(f"line files differ in length: { {a: len(t) for a, t in texts.items()} }")
    ref_wavs = item_reference_wavs(n_items)
    print(f"[assign] per-item references: {n_items} items over {len(set(ref_wavs))} voices", flush=True)

    asr = kana_whisper_transcriber()
    records, arm_meta = [], {}
    for arm in arms:
        checkpoint = args.phon_checkpoint if arm == "ours" else None
        print(f"=== ARM: {arm} ({checkpoint or 'base release'}) ===", flush=True)
        runtime = load_runtime(checkpoint, watermark=False)
        (out_dir / arm).mkdir(parents=True, exist_ok=True)
        for i, text in enumerate(texts[arm]):
            audio, sr, info = synth_valid(
                lambda seed: synthesize(runtime, text, seed=seed, ref_wav=ref_wavs[i]),
                SEED_BASE + i,
                asr=asr,
            )
            sf.write(out_dir / arm / f"{i:02d}.wav", audio, sr)
            records.append({"idx": i, "arm": arm, **info})
            note = f" guard: {info['takes']} takes {info['rejected']}" if info["takes"] > 1 else ""
            print(f"  [{arm}] {i:02d}/{n_items} {len(audio) / sr:.2f}s{note}", flush=True)
        arm_meta[arm] = {
            "checkpoint": str(checkpoint) if checkpoint else "base release",
            "checkpoint_sha256_16": sha256_prefix(checkpoint) if checkpoint else None,
            "texts": texts[arm],
        }
        del runtime

    json.dump(records, open(out_dir / "guard_takes.json", "w"), ensure_ascii=False, indent=0)
    run_record = {
        "test_set": test_set,
        "n": n_items,
        "arms": arms,
        "seed_base": SEED_BASE,
        "max_takes": MAX_TAKES,
        "sampling": "upstream defaults: 40 steps, cfg text 3.0, speaker 5.0; fp32; no watermark",
        "reference_wavs": ref_wavs,
        "arm_meta": arm_meta,
    }
    json.dump(run_record, open(out_dir / "blockB1_irodori.json", "w"), ensure_ascii=False, indent=1)
    print(f"DONE -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
