# Pre-registration v2 — one recipe, one evaluation, four backbones (round 2)

**Status:** fixed 2026-09-16 (JST), before any round-2 teacher audio was inspected, before any
round-2 adapter was trained, and before any round-2 judgement was collected. Round 1 (the design
in `PREREGISTRATION.md`, with all its amendments) was analysed in full before this document was
written; nothing here is chosen on round-1 outcomes except the decision to run round 2 at all.
**Author:** Shuhei Kato. **Purpose:** replace the round-1 patchwork (four teacher-take rules, three
different accent comparators, one backbone unjudged, an untuned transfer whose teacher sets were
not reading-checked) with one design applied identically to all four backbones.

## 0. What is fixed here

1. **One text set.** `data/cleanroom/inventory_500.jsonl` (533 words x 10 carriers) minus every word
   colliding, by surface or canonical reading, with a target of difficult-50, katakana-50, easy-50 or
   blockC-1000 (the union of the round-1 filters; round 1 applied easy-50 on CosyVoice 2 only).
   Result: **471 words, 4,710 carriers** (`scripts/teachers/build_text_v3.py`; the 62 dropped words
   are listed in its log). The Joyo objective set is disjoint by its own construction (round 1 §2).
2. **One teacher rule** (`scripts/teachers/gen_*_v2.py`, `reading_check.py`): the frozen base reads
   the plain kanji carrier; the take is kept iff it passes the degeneration guard (Sarashina: 20 to
   600 semantic tokens; others: audio >= 0.5 s and valid codes) **and** the ASR of record
   (kana-whisper) hears the target word's reading in it, judged with the paper's canonicalisation;
   otherwise the seed is bumped and the carrier gets another take, **up to five takes in all**, each
   later take only for carriers whose earlier takes failed; a carrier with no passing take is dropped and
   logged with all its transcripts. (History, all before any teacher set was complete: two takes at
   12:20 JST; one at 13:40 because of T5Gemma's speed; two at 14:35 as passes over earlier drops;
   three at 14:55; five at 15:05. Cutoff: T5Gemma's later takes stop at 2026-09-17 20:00 JST, whatever
   was finished by then is kept and the cut is recorded, so that the panel can be built on 09-18.) No best-of, no CER threshold, no per-word cap. The number kept per
   backbone is reported, however small.
3. **One speaker rule.** No speaker enters where the backbone can synthesize from text alone
   (Sarashina, T5Gemma). Where it cannot, **2,000 self-sampled synthetic voices are cycled over
   carriers** on both speaker-conditioned backbones: Irodori the first 2,000 anchor-bank references
   in name order (as in round 1); CosyVoice 2 its fictional bootstrap bank extended from 20 to 2,000
   by the same nested draws from the round-1 Gaussian (`scripts/bootstrap_speaker_more.py --total
   2000`; the existing 20 are untouched). (History, before any Irodori or CosyVoice 2 teacher
   existed: five voices at 12:20 JST; each backbone's whole bank at 13:20; 2,000 on both at 13:45 so
   that the two match.) Irodori's unconditional path was probed on 2026-09-16 and produces noise
   under every guidance scale, step count and unconditional mode, so it is speaker-conditioned.
4. **Training** exactly as in round 1 per backbone (same scripts, ranks, alphas, step counts:
   Sarashina 20k, CosyVoice 2 1k, Irodori 2k, T5Gemma 400), with the v2 teacher sets. No tuning on
   any evaluation number. If a backbone's teacher set is small (T5Gemma is expected to lose most
   carriers to the reading check), it is trained anyway and its size is reported.
5. **Objective evaluation** as in round 1: the 319-item test half under one of 42 held-out
   ReazonSpeech voices per item (`items/speaker_assignment42_obj319.json`), canonical target-reading
   hit under kana-whisper, guard identical in every arm, bootstrap clustered by word, exact McNemar.
   The **no-edit and kana arms do not depend on the adapter and are reused from round 1**; only the
   "ours" arm is re-synthesized with each v2 adapter, under the same seeds and guard. Reported
   contrasts: tag - none and tag - kana per backbone. The CosyVoice 2 released-UtterTune arm is
   reused. Table 2 of the round-1 paper (teacher-voice dose) is not repeated.
6. **Listening evaluation, round 2** (`items/speaker_assignment42.json`, cells A2..D2): the
   difficult-50 set, **every backbone x {kana, tag}, plus {no edit} on Sarashina only**, one held-out
   voice per word fixed by hash (A2/B2/C2 reuse the round-1 assignment of the same backbone, D2 is
   hashed the same way), 450 stimuli + 8 known-answer catch trials + 5% duplicate items. Same
   raters as round 1 (four admitted at 18/19 on the screening gate) plus any new candidate who
   passes the same gate before the kit closes; minimum three raters, all judging everything.
   Judged blind, randomized order, whole sentences, prescribed accent shown; the rater UI, catch
   trials, screening definition and payment are those of round 1.
7. **Confirmatory family (Holm, family-wise alpha 0.05, McNemar exact two-sided, paired by word):**
   - H-k(b), b in {Sarashina, CosyVoice 2, Irodori, T5Gemma}: the tag realizes the prescribed accent
     more often than plain kana input.
   - H-n: on Sarashina the tag realizes the prescribed accent more often than no edit.
   Five tests, one family. Secondary (no test): the accent-only version of each contrast with
   reading failures removed, as in round 1; Fleiss' kappa over all stimuli before any contrast;
   per-rater catch and duplicate consistency. Round-1 figures are not pooled with round 2.
8. **Naturalness, round 2.** Same-sentence A/B (unedited vs edited, "no difference" allowed) on the
   easy set with the v2 adapters, **25 pairs per backbone = 100 pairs**, the 25 chosen per backbone
   as the lowest sha256 of the pair id (fixed rule, no inspection), two groups of 50, at least 15
   judgements per pair, CrowdWorks task format, 1,000 yen, one sitting per account. Statistic and
   margin as round 1: P(unedited) - P(edited) with ties split, non-inferiority margin 0.15 on the
   upper 95% limit, pair- and rater-clustered bootstrap. Exclusions **fixed now**: a second sitting by
   the same account (by worker list); a sitting whose median inter-response interval is below the
   shortest pair's two clips played once (speeder, timestamps only); no exclusion on answer
   balance (reported). Trained-word pairs cannot occur (item 1). Payment as round 1.
9. **Exclusion rules for the accent panel, fixed now:** a rater below 6/8 on the catch trials is
   excluded and reported; a rater whose median inter-response interval is below the shortest
   stimulus's duration is excluded and reported; tied consensus items (2-2) are excluded and counted.
10. **Reporting.** Every number above is reported whatever it shows. The paper's Table 1 is the
    round-2 table; round 1 is described in one paragraph as the preliminary round whose defects
    this design removes, with its full record in `PREREGISTRATION.md`. If round 2 cannot be
    completed by the submission deadline, the round-1 paper is submitted unchanged and round 2
    appears in the arXiv version.

## 1. Decision points already taken (so they are not taken later)

* Candidates reached through the platform are paid whatever they score, as in round 1. Volunteers
  approached directly may take part without payment; whether each did is recorded and stated.
* A backbone whose v2 teacher set has fewer than 300 carriers is still trained and evaluated; the
  size is stated in the paper.
* If the round-2 accent panel gets only three raters, the design stands (registered minimum).
* No stimulus, item, arm, rater or pair is dropped for any reason not listed in items 2, 8 and 9.

## 2. Analysis scripts

`scripts/analysis/analyze_listening.py` (cells A2..D2, family of five), `scripts/analysis/
analyze_naturalness.py` (100-pair key), `scripts/eval/measure_reading_all.py` (objective). The
teacher provenance files (`*.provenance.json`) carry every dropped carrier with its transcripts.

## 3. Timestamp

Fixed 2026-09-16 13:00 JST; git history of this file is the record. Teacher generation for
Sarashina and T5Gemma started at 12:20 JST under this rule (before this text was finished, after
the rule was); no output had been inspected beyond the four-carrier smoke tests recorded above.
