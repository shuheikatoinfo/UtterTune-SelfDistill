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
   T5Gemma's four evaluation syntheses (18:31 JST) also run on its pod with the identical commands, item
   files, speaker assignments and the 42 held-out evaluation voices copied to the pod; outputs are copied
   back into the same directory layout. The in-synthesis guard is unchanged from round 1.
   **Audit 18:45 JST:** the base weights the pods downloaded from Hugging Face were compared tensor by
   tensor with the local copies used for synthesis (Irodori-TTS-500M-v3 model.safetensors, CosyVoice2-0.5B
   llm/flow/hift + both ONNX files byte-wise, T5Gemma-TTS-2b-2b ckpt/pretrained.pth): all identical.
   Two small execution facts: (i) T5Gemma teacher seeds are `1234 + 131*take + i` with i the row's index
   within its shard (rows i::8), not the global index; every shard used the same layout, so the seeds are
   fixed and reproducible from the shard assignment, which is recorded in `t5gemma/runpod_pods.json`;
   Irodori and CosyVoice 2 seeds and voices are keyed on the global row index. (ii) The CosyVoice 2 prompt
   clips reached the pods as 16-bit FLAC copies of the 24 kHz pool wavs (PCM_16 quantisation of float
   audio); the VAD-trimmed prompt copies were rebuilt on the pods from those.
   Two further checks at 19:00 JST, both clean: every round-2 listening stimulus exists for all four cells
   and all arms (A2 150, B2/C2/D2 100 each) under the item's assigned voice; and all 200 naturalness
   no-edit clips are byte-identical to the round-1 easy-50 rendition under that item's assigned voice,
   so each A/B pair holds one voice fixed. The round-2 panel site was deployed at 19:10 JST
   (481 items per rater: 450 stimuli, 23 duplicates, 8 catch trials). Its two endpoints were probed once
   with a dummy account `probe_selfcheck`, which failed screening and never entered the panel; its two
   records were exported and deleted from the store the same minute.
   **Naturalness pair set, built 19:28 JST: 95 of the 100 registered pairs.** Five pairs hold one clip
   that cannot be judged (peak below 0.05 at full scale) and are excluded by the round-1 audibility rule,
   which item 8 did not list: Sarashina items 23 and 29, where the EDITED clip is silent (the adapter ran
   to the token cap), and T5Gemma items 20, 25 and 31, where the UNEDITED clip is silent (a base failure).
   The exclusion therefore helps the method on Sarashina and hurts it on T5Gemma. Both counts are reported
   per backbone and per arm in the paper, so the filter is a result rather than a hidden step; no pair is
   replaced and the hash selection is untouched. Groups: 49 and 46.
   **Superseded 09-17 06:40 JST by the rebuild under amendment 5:** with both sides of every pair from
   round 2, the T5Gemma no-edit clips are audible and only the two Sarashina pairs whose EDITED clip is
   silent (items 23 and 29) fall out: **98 of 100 pairs**, groups 51 and 47, deployed at
   jp-accent-naturalness-r2.pages.dev. Every shipped clip was verified against its round-2 source.

## 4. Amendment, 2026-09-16 20:00 JST — every listening arm is synthesized in round 2

**Why.** Items 5 and 6 reused the round-1 no-edit and kana arms on the ground that they do not depend
on the adapter. Auditing the tree before the panel opened showed that is true of the *weights* but not
of the *procedure*: the round-1 listening wavs (`sweep42/`) were produced on 09-02, and the shared
degeneracy guard (`scripts/eval/degenerate.py`, retry on no-speech / repetition / low-variety, cap 6)
entered the synthesis scripts on 09-05. The round-2 tag arm therefore had up to six attempts to avoid a
degenerate take while the reused comparators had one. The guard fired on 6 of 50 items in the round-2
Sarashina tag arm and on 1 of 50 in CosyVoice 2, so the asymmetry was live, and it runs in the method's
favour. Irodori's kana arm was worse placed still: round 1 built it with a different script
(`synthesize_routed.py`), one fixed seed for every item, and no guard at all.

**What changes.** Every arm of every listening cell is now synthesized in round 2, with the same script,
the same seed rule and the same guard as that cell's tag arm, in the same session:
Sarashina A2 (no-edit, kana, tag) and CosyVoice 2 B2 (tag) already existed from the round-2 runs;
CosyVoice 2's kana arm and Irodori's kana arm were synthesized at 19:45 JST (Irodori's as the base arm
over the kana sentences, `run_blockB1.py --arms noedit`, seed base 5000, cap 6 — identical to its tag
arm); T5Gemma D2 was round-2 on both arms already. No round-1 listening audio enters round 2. The
round-1 wavs stay on disk untouched.

**What does not change.** Items, voices, assignment, arms, tests, rater rules and exclusions are as
registered.

## 5. Amendment, 2026-09-16 20:10 JST — one guard implementation on all four backbones

The audit above left three asymmetries standing, all of which the user asked to remove while there was
still time. All three are now closed, and every affected arm is re-synthesized on the local GPU.

1. **Irodori ran a weaker guard.** Sarashina, CosyVoice 2 and T5Gemma redraw a take when the waveform
   is degenerate **or** when kana-whisper's transcript is (`text_reason`: no speech, a repetition run,
   or a mora inventory under 25%). Irodori redrew on length alone. `run_blockB1.py` now calls the same
   `synth_valid()` from `scripts/eval/degenerate.py`, with the same seed bump (+1000 per take), the same
   cap (6), the same waveform thresholds and the same ASR, and writes the same `guard_takes.json`
   record; the patched file is kept in the repository as
   `scripts/irodori/run_blockB1_guarded_20260916.py` and the original beside it as a dated backup.
   Every Irodori arm is therefore re-synthesized: listening C2 (tag, kana), the naturalness easy set
   (tag, no edit) and the objective 319 set (tag, kana, no edit).
2. **The naturalness no-edit clips came from 09-08.** They were guarded, but from a different session
   than the edited clips. The no-edit arm is now synthesized in round 2 for every backbone
   (Sarashina and Irodori already had it; CosyVoice 2 and T5Gemma are queued), so both sides of every
   A/B pair come from the same session, the same script and the same guard.
3. **The objective comparators came from 09-04**, before the shared guard reached the scripts. The
   no-edit and kana arms of the 319-item table are re-synthesized in round 2 for CosyVoice 2, Irodori
   and T5Gemma (Sarashina already had all three), on the local GPU, with the item voices and seeds
   unchanged. The table is reported on round-2 arms throughout; the round-1-arm version is kept as a
   sensitivity check.

No item, voice, assignment, arm, test or exclusion rule changes. Nothing is selected on an outcome:
the change is which implementation of one disclosed filter runs, and it now runs identically everywhere.

**Execution record, 09-16 20:47 to 09-17 06:35 JST.** The queue ran on the local GPU in this order, each
with the same guard: CosyVoice 2 easy no-edit (50), T5Gemma easy no-edit (50), CosyVoice 2 obj319 no-edit and
kana (319 each), Irodori obj319 tag, no-edit and kana (319 each; the tag arm redone under the shared guard),
T5Gemma obj319 no-edit and kana. The last step failed at once because the queue had exported
`HF_HUB_OFFLINE=1` for Irodori and T5Gemma's tokenizer load needs the hub; it was rerun at 06:35 on 09-17
without the flag, nothing else changed. The naturalness pair set was rebuilt on 09-17 from round-2 clips on
both sides for all four backbones (every staged clip verified as the item's assigned-voice round-2
rendition), and the objective table is measured with `data/analysis/v2/arms_obj319_round2.json`;
`arms_obj319_v2.json` (round-1 comparators) is the sensitivity check.

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
   raters as round 1 (the gate as amended in §7 admits six of 34 candidates: 20/20, two at 19/20,
   three at 18/20; four of the six judged in round 1) plus any new candidate who
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

## 6. Amendment, 2026-09-17 08:05 JST — the naturalness study uses all 50 easy pairs per backbone

Item 8 registered 25 pairs per backbone, chosen by hash, to halve the crowd. Before any naturalness judgement
was collected, a variance decomposition of the round-1 judgements (per-pair true-preference SD 0.37 to 0.65
against a within-pair judgement SD of 0.70 to 0.88) showed that the 95% interval of the preference is set
almost entirely by the number of pairs, not by the number of judgements per pair: at 25 pairs the half-width
is 0.16 to 0.26 whatever the crowd size, against a non-inferiority margin of 0.15; at 50 pairs it is 0.11 to
0.19. The user therefore restored the round-1 size: **all 50 easy pairs per backbone, no hash selection, four
groups of about 50 pairs, one group per sitting, at least 15 judgements per pair**, everything else in item 8
unchanged. 198 pairs ship (the two Sarashina pairs with a silent edited clip are excluded as before); groups
51/51/48/48. Both sides of every pair are round-2 audio (amendment 5). Built and deployed 07:38 JST (file
times 07:37:50 to 07:38:02); the CrowdWorks task opened at 07:46 JST and every sitting since answers ids of
the 198-pair build (each rater's answered set lies inside exactly one of its four groups; the 98-pair build
had ids m000 to m097 only). No judgement was recorded on the earlier build.

**Objective table measured 2026-09-17 09:44 JST** with the metric of record (truth = the item's registered
reading via `TARGETS_FILE`, as fixed on 09-09). A first run at 09:00 had omitted that variable and judged against the
tag-derived reading; it was discarded once the same files re-measured differently from the 09-09 record. With the
item truth, the round-1 comparator arms reproduce the round-1 numbers exactly. Results (all-round-2 arms): tag − none
+0.25 / +0.40 / +0.29 / +0.38 (Sarashina / CosyVoice 2 / Irodori / T5Gemma, all p < 1e-10); tag − kana −0.09 / −0.16 /
−0.05 / −0.09, all below zero at the 95% level. `data/analysis/v2/obj319_round2_contrasts.md`.

## 7. Amendment, 2026-09-16 20:30 JST — screening key restored to 20 items, pass line 18 of 20

- **What changed.** On 2026-09-09 item `scr07` (委員, shown as 頭高) had been withdrawn from the
  screening key and credited to every candidate, because the author, re-listening, could not decide
  whether the rendition matched the display (the note said 中高 was admissible). That call was wrong:
  a 中高 word starts low on its first mora, and this rendition does not. Measured on 2026-09-16 with
  YIN over the carrier's first word, F0 holds about 220 Hz through the first mora and falls to about
  174 Hz across the second and third: the rendition is 頭高 against the display, "differs" is the
  correct answer, and the original key stands. The set is scored as a clean 20.
- **Pass line.** The slack widens from one item to two at the same time (18 of 20, written as a
  count, not a fraction), on the author's decision, so that restoring the item cannot demote anyone
  already admitted. Under 18/20 all five previously admitted candidates remain admitted, and one
  candidate rejected at 17/19, who had answered `scr07` correctly, is admitted: six of 34 in total.
  Nobody is dropped on a rule change made after the fact. Every stored gate record was regraded from
  its own stored answers against the 20-item key; the rule and the key are in
  `scripts/rater_ui/gen_collector.py` and the generated `functions/api/screen.js`, deployed to the
  round-2 panel site at the same time.
- **Incident, 2026-09-17 morning: admitted raters could not reach the judging block.** The regrade
  wrote each gate record through a helper that already JSON-serialises its argument, and passed it a
  pre-serialised string, so all 33 records were stored as JSON string literals. The site's gate
  endpoint then returned `found` without `pass`, and the page treated every rater, admitted or not, as
  failed. Reported by an invited rater; diagnosed and repaired 2026-09-17 morning by rewriting all 33
  records as objects from the same stored answers (no answer or verdict changed; six pass, 27 fail,
  every record re-checked against the 18/20 rule). No judgements were lost: the block had not been
  reachable, so none had been made. The window during which invited raters were turned away is the
  invitation time to the repair; they are being told the link now works.

## 8. Amendment, 2026-09-17 11:40 JST — the guard's ASR half saw native-rate audio on Irodori and T5Gemma

- **Defect.** The shared guard (`scripts/eval/degenerate.py`, amendment 5) rejects a take on the waveform
  (silent, too quiet, mostly silent, under 0.5 s, runaway to the cap) **or** on kana-whisper's transcript
  (no speech, a repetition run, a mora inventory under 25%). The transcript half is fed by a per-backbone
  wrapper. On Irodori (`run_blockB1.py`, 48 kHz) and T5Gemma (`blockA_t5gemma.py`, 44.1 kHz) that wrapper
  passed the raw array with its native rate to the transformers pipeline, which in the transformers 4.57
  used by both environments does not resample: kana-whisper heard the audio at three times (Irodori) or
  2.76 times (T5Gemma) its speed. Verified 11:20 JST inside both environments on round-2 files: the same
  waveform transcribes as garbage at its native rate and correctly once resampled to 16 kHz. Sarashina and
  CosyVoice 2 (24 kHz) were checked the same way inside their environments and are faithful. This is the
  raw-rate defect of item 2 (the teacher checker), present in a second place. **The metric of record was
  never on this path**: `measure_reading_all.py` resamples through `asr_audio.to_asr_input`, and the round-2
  objective transcripts are correct.
- **Effect.** On Irodori and T5Gemma the transcript half of the guard fired on garbage: `repetition` and
  `low_variety` rejections there are noise with respect to the take's content, and a genuinely looping take
  was not caught by it. The waveform half worked. The guard was applied identically to every arm of a
  backbone, so no arm was favoured within a backbone, and nothing was selected on the measured outcome; but
  the guard on these two backbones was in effect weaker than on the other two, which is the asymmetry
  amendment 5 was meant to remove. Retry records: Irodori objective 135 of 638 (tag + no edit) and 65 of 319
  (kana) items retried, 27 and 9 to the cap; T5Gemma objective 51 of 319 base items and 7 + 46 tag items retried.
- **Fix and re-synthesis.** Both wrappers now resample through `to_asr_input` (the Irodori synthesis wrapper backed up
  as `run_blockB1.py.bak_guardasr_20260917`; repo copy and `blockA_t5gemma.py` patched the same minute). The
  following arms are re-synthesized on the local GPU with the same items, voices, seeds, cap and scripts,
  the earlier outputs kept beside them under `*_guardraw` for a sensitivity check: Irodori objective 319
  (tag, no edit, kana) and listening C2 (tag, kana); T5Gemma objective 319 (tag, no edit, kana) and listening
  D2 (tag, kana). The objective table of record is the re-synthesized one. The naturalness easy sets cannot be
  re-synthesized (judgements were collected on them from 07:46 JST); their guard state is disclosed. Whether
  the re-synthesized C2 and D2 stimuli replace the deployed ones is decided by the author before any C2/D2
  judgement exists (none exists at 11:40 JST); the panel is not redeployed on the author's initiative.

## 9. Amendment, 2026-09-17 12:05 JST — the author's decisions on amendment 8

1. **Listening C2 and D2 are replaced.** No C2/D2 judgement exists (verified again before deployment); the
   re-synthesized Irodori and T5Gemma difficult arms (tag, kana; same items, voices, seeds) replace the deployed
   stimuli on the same site. A2 and B2 are unchanged. Raters already admitted keep their admission; nobody has
   started the block.
2. **Naturalness is re-collected in full, all four backbones** (the author's decision, 12:20 JST, for one
   clean collection under one design). The Irodori and T5Gemma easy arms (tag and no edit on both) are
   re-synthesized under the corrected guard (`scripts/eval/round2_guardfix_easy.sh`, same items, voices and
   seeds); the Sarashina and CosyVoice 2 easy clips are unchanged (24 kHz; guard verified faithful in their
   environments). The 200 pairs are rebuilt with the same rules (audibility exclusion, A/B by hash), deployed as
   a new site, and judged in a new CrowdWorks task under item 8 and amendment 6 (four groups of about 50, at
   least 15 judgements per pair, same exclusions, same payment, new raters not required). The complete 09-17
   collection on the 198-pair build (104 sittings) is superseded as the record and is reported only as a
   sensitivity check, per backbone. Nothing else in item 8 changes.

