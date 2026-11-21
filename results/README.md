# Results

`round3/` holds the reports of record of the round the paper reports, the raters' responses (pseudonymous IDs), the
listening-kit keys, and the per-item retry record. The registration's summary (`preregistration/PREREGISTRATION_v2.md`)
names which file backs which number.

## `round3/retry_record_obj319.jsonl`

One row per item and arm of the 319-item table (13 arms × 319 items). The degeneration guard
(`scripts/eval/degenerate.py`) redraws a degenerate take with a new seed, up to six takes, and keeps the first take
that passes; if none passes, it keeps the last.

| Field | Meaning |
|---|---|
| `backbone`, `arm`, `idx`, `spk` | backbone; `noedit`, `kana`, `tag` or `uttertune_released`; item index; voice index |
| `takes` | takes synthesized for the item, 1 to 6 |
| `rejected` | the reason for each rejected take, in order: `too_short` (under 0.5 s), `runaway` (ran to the length cap), `silent`, `too_quiet`, `mostly_silent` (audio tests); `no_speech`, `repetition` (tests on the kana-whisper transcript) |
| `source` | how the row was obtained: written by the synthesis run, reconstructed from its log, or re-synthesized with the same inputs, seeds and guard (the registration's entry on this record, 2026-11-21) |
