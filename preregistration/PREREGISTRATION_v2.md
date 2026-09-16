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
   backbone is reported, however small. **T5Gemma's teacher generation runs on rented RunPod GPUs**
   (decided 14:00 JST on 09-16 with 151 of 4,710 carriers done locally at 30 s per row under GPU
   contention): the same script, rule and seeds, rows sharded over pods (`scripts/teachers/runpod_t5gemma.py`);
   the local partial pass is discarded whole and no row of it enters; with the pods the five takes are
   expected to finish before the cutoff, which stays in force.
   **Checker implementation defect, found 15:20 JST on 09-16 and fixed before any teacher set was
   inspected beyond its counts:** `reading_check.Checker` handed raw audio with its native rate to the
   transformers pipeline, whose resampling is faithful at 24 and 16 kHz but distorts 44.1 kHz
   (target word misheard) and 48 kHz (unintelligible) audio; verified on one Irodori utterance whose
   file-path transcript is correct at every rate once librosa resamples to 16 kHz first. The fix resamples
   in the checker. Consequences: Sarashina and CosyVoice 2 (24 kHz) are unaffected and their sets stand;
   Irodori (48 kHz; 200 carriers done, 10 kept) and T5Gemma (44.1 kHz; pods about 35 min in) are
   regenerated from scratch with the fixed checker and their partial outputs are discarded whole.
   The same raw-rate path was used by the round-1 T5Gemma teacher QC (CER at 44.1 kHz), which is
   noted in `PREREGISTRATION.md`; the objective measurement of record (`measure_reading_all.py`)
   already resampled via `asr_audio.to_asr_input` and is unaffected.
   **Irodori's teacher generation also moved to RunPod** (15:30 JST on 09-16, at the user's direction, for
   time): same script, rule, seeds and voice assignment (both keyed on the global row index), rows sharded
   over four pods; the local run (restarted 15:17 with the fixed checker) is stopped once the pods deliver
   and its partial output is discarded whole. CosyVoice 2's teacher generation moved to RunPod as well
   (15:50 JST; same script, rule, seeds and prompt-voice assignment keyed on the global row index; four
   shards). Execution note: the Sarashina training started 15:05 stalled at step 4,880 from 15:58 (GPU
   memory oversubscribed while three jobs shared the local GPU) and was killed at 16:15 and restarted from
   scratch with the identical command (`v2_chain.sh sarashina_train`); the stalled run's log is kept as
   `logs/sarashina_train_stalled_1558.log` and nothing of it is used.
   Irodori's adapter training (2,000 steps, identical config, base checkpoint and command) runs on the
   RunPod pod that generated its teachers (16:39 JST) and the checkpoint is copied back for the local
   synthesis; no local Irodori training run is used.
   T5Gemma's materialize/train/export (400 steps, identical scripts including the local trainer patch)
   likewise run on the pod that generated its teachers (17:25 JST); the merged model is copied back for
   the local syntheses. CosyVoice 2's LoRA training (1,000 steps, identical command) runs on one of its
   teacher pods once the four shards are collected (from 17:30 JST); the adapter is copied back for the
   local syntheses. Idle pods are terminated once their outputs are on local storage.
   Irodori's easy-50 synthesis (18:05 JST) first produced 45 of 50 items: its case loader derives the
   target word from the vanilla-vs-yomi diff, which is empty for the five katakana targets (items 6, 8,
   17, 20, 47), while the other three backbones read the tagged word from the shared _lora file. The
   loader now falls back to that word when the diff is empty (a one-line change in run_blockB1.py, kept
   with a dated backup), and the easy set was re-synthesized in full with the same seeds; the 45-item
   run is kept beside it as `easy_run1_45of50` and is not used.
3. **One speaker rule.** No speaker enters where the backbone can synthesize from text alone
   (Sarashina, T5Gemma: text-only teachers, as in round 1). Where it cannot, **the same 2,000
   self-sampled synthetic voices are cycled over carriers on both speaker-conditioned backbones**:
   Irodori's anchor bank `bank_spk_00001..02000` (voices sampled by Irodori-TTS-500M-v2 with no reference,
   one fixed seed and one anchor sentence per voice, then stored as reference latents; the profile captions
   in the bank's log are labels, not conditions, as the 500M base has no caption input; no recording.
   Correction 2026-09-16 15:00 JST of the earlier wording "synthesized from written captions"), used by Irodori as reference latents and by CosyVoice 2 as prompt clips
   with each clip's own transcript (`v2/shared_voices/pool.json`). (History, before any Irodori or
   CosyVoice 2 teacher existed: five voices at 12:20 JST; each backbone's own bank at 13:20; 2,000 on
   both at 13:45; one shared set at 14:10 so that the two backbones hear identical voices.)
   Irodori's unconditional path was probed on 2026-09-16 and produces noise under every guidance
   scale, step count and unconditional mode, so it is speaker-conditioned.
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
