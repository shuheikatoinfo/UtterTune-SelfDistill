#!/usr/bin/env python3
"""Emit the evaluation item lists and record their provenance.

There is no sampling and no randomness: **every item in each source set is used.**
This script exists so that the exact item text and the SHA256 of each source file are
published alongside the pre-registration and can be re-derived by anyone.

    python3 build_item_lists.py --write

Outputs (under ./items/):
    difficult_all.tsv   all difficult items
    katakana_all.tsv    all katakana items
    provenance.json     source paths, SHA256, counts

Item IDs are stable: "<set>:<1-based line number in the source file>".
"""

import argparse
import hashlib
import json
import pathlib

DEFAULT_SRC = pathlib.Path(
    "$ITEM_SOURCE"
)
SOURCES = {
    "difficult": ("test_difficult_50_vanilla.txt", "test_difficult_50_with_accent.txt"),
    "katakana": ("test_katakana_50.txt", "test_katakana_50_with_accent.txt"),
}

HERE = pathlib.Path(__file__).resolve().parent
OUT_DIR = HERE / "items"


def read_items(path):
    """Return [(line_no, text)] for non-blank lines, 1-based line numbers."""
    lines = path.read_text(encoding="utf-8").split("\n")
    return [(i + 1, s.strip()) for i, s in enumerate(lines) if s.strip()]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=pathlib.Path, default=DEFAULT_SRC)
    ap.add_argument("--write", action="store_true", help="write files (otherwise dry-run)")
    args = ap.parse_args()

    prov = {"sampling": "none — all items in each source set are used", "sets": {}}

    for name, (plain_fn, accent_fn) in SOURCES.items():
        plain_p, accent_p = args.src / plain_fn, args.src / accent_fn
        plain, accent = read_items(plain_p), read_items(accent_p)
        if len(plain) != len(accent):
            raise SystemExit(f"{name}: line count mismatch {len(plain)} vs {len(accent)}")
        acc_by_line = dict(accent)

        rows = ["item_id\tsource_line\tsentence\tsentence_with_accent"]
        for ln, txt in plain:
            rows.append(f"{name}:{ln:03d}\t{ln}\t{txt}\t{acc_by_line[ln]}")
        body = "\n".join(rows) + "\n"

        prov["sets"][name] = {
            "source_plain": str(plain_p),
            "source_accent": str(accent_p),
            "sha256_plain": sha256(plain_p),
            "sha256_accent": sha256(accent_p),
            "n_items": len(plain),
        }

        print(f"{name}_all: n={len(plain)}")
        if args.write:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            (OUT_DIR / f"{name}_all.tsv").write_text(body, encoding="utf-8")

    if args.write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "provenance.json").write_text(
            json.dumps(prov, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote -> {OUT_DIR}")
    else:
        print("(dry run; pass --write to emit files)")


if __name__ == "__main__":
    main()
