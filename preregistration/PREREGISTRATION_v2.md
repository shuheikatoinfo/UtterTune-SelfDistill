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

1. **Listening C2 and D2 are replaced.** The re-synthesized Irodori and T5Gemma difficult arms (tag, kana; same
   items, voices, seeds) replace the deployed stimuli on the same site, as cells **C2b** and **D2b**; A2 and B2
   are unchanged (the rebuilt clips are checked byte-identical to the deployed ones before deployment).
   **Correction at 12:00 JST:** one rater had already completed the whole block (473 judgements, 10:32 to
   11:18 JST) on the earlier C2/D2 stimuli before the store was checked again. The author's decision (12:00 JST,
   option A): the replacement goes ahead; that rater's A2/B2 judgements stand; their 210 C2/D2 judgements on the
   superseded stimuli are exported to the backup and removed from the working store, reported as a sensitivity
   check, and the rater is asked to judge C2b/D2b (210 items) for a proportional additional payment. The other
   admitted raters had not started. Under the corrected guard none of the Irodori difficult or objective items
   needed a redraw: every redraw the defective guard triggered on Irodori was spurious, so 17 of 50 tag and 19 of
   50 kana C2 stimuli change take (seed) with this replacement. The registered speed exclusion (median
   inter-response interval below the shortest stimulus) is the only speed rule applied; a rater may answer as
   soon as the accent is heard, so short intervals as such are not a ground for exclusion.
2. **Naturalness is re-collected in full, all four backbones** (the author's decision, 12:20 JST, for one
   clean collection under one design). The Irodori and T5Gemma easy arms (tag and no edit on both) are
   re-synthesized under the corrected guard (`scripts/eval/round2_guardfix_easy.sh`, same items, voices and
   seeds); the Sarashina and CosyVoice 2 easy clips are unchanged (24 kHz; guard verified faithful in their
   environments). The 200 pairs are rebuilt with the same rules (audibility exclusion, A/B by hash), deployed as
   a new site, and judged in a new CrowdWorks task under item 8 and amendment 6 (four groups of about 50, at
   least 15 judgements per pair, same exclusions, same payment, new raters not required). The complete 09-17
   collection on the 198-pair build (104 sittings) is superseded as the record and is reported only as a
   sensitivity check, per backbone. Nothing else in item 8 changes.

**Execution record, amendment 9 item 1 (panel), 2026-09-17 12:20 JST.** Kit rebuilt from the corrected-guard
Irodori and T5Gemma difficult arms (`scripts/teachers/v2_kit_and_site_b.sh`); A2/B2 clips verified byte-identical
to the deployed kit; cells renamed C2b/D2b; the earlier C2/D2 clip directories are kept in the deployment so a
session opened on the earlier build keeps playing. Stimuli whose audio changed with the corrected guard
(byte-different and not the same take under float noise): C2b 38 of 105 (17 tag, 19 kana, 2 duplicates), D2b 32 of
105; these 70 form block 1, which only the rater who had completed the block judges (invitation URL with `&b=1`).
Every other rater judges block 0 (481 items, A2, B2, C2b, D2b); the assignment endpoint is called with one unit so
that no new rater is ever handed block 1. Store: the completed rater's 140 judgements on unchanged C2/D2 stimuli were
re-keyed to C2b/D2b (same clip, same answer; `rekeyed_from` kept in each record) and their 70 judgements on changed
stimuli removed (exported first to `superseded_c2d2_R1_*.json`). A second rater had started at 12:04 JST
(32 judgements); at the author's decision they restart the block: their 10 C2/D2 judgements were exported and
removed, their 22 A2/B2 judgements stand. Deployed 12:20 JST and probed end to end with a dummy passing account
for both blocks (block 0: 481 items; block 1: 70 items), whose records were deleted afterwards.

**Objective, Irodori, under the corrected guard (12:22 JST, `reading_obj319_round2_irodori_gf.json`):** no edit
0.586, kana 0.909, tag 0.868; tag − kana −0.041 [−0.079, −0.003], p = 0.029; tag − none +0.282 [+0.212, +0.357].
No item of any arm needed a redraw; the no-edit and kana values coincide with the round-1 comparators (first take
everywhere). The earlier round-2 Irodori row (0.580 / 0.912 / 0.865) is the sensitivity check.

**Execution record, amendment 9 item 2 (naturalness, third build), 2026-09-17 12:50 JST.** Irodori easy (tag,
no edit; 11:50 to 11:53 JST, local) and T5Gemma easy (tag 12:16 to 12:32, no edit to 12:44 JST, local) re-synthesized
under the corrected guard with the same items, voices and seeds; the earlier arms kept as `*_guardraw`. Pair set
`preregistration/items/naturalness_ab_easy50_r3.json` (ids p000 to p197; 198 pairs: Sarashina 48 with the same two
silent-edit exclusions, others 50; A/B by hash as before). Against the 198-pair build of 07:38: every Sarashina and
CosyVoice 2 clip byte-identical; Irodori 17 no-edit and 24 tag clips changed, T5Gemma 17 and 21 (the T5Gemma
counts include same-take float-noise differences, the earlier T5Gemma easy arms having been pod-generated).
Loudness-normalised as before (396 clips, -23.0 LUFS), four content-derived groups 51/51/48/48. Deployed as a new
project, `https://jp-accent-naturalness-r3.pages.dev`, 12:47 JST; served clips verified byte-identical to the build;
the consent-to-judging path exercised with a dummy id in a headless browser (48-item group rendered with both
players), whose records were deleted. The group-assignment endpoint shares the store with the 07:46 task, so its
"emptiest group" count starts from that task's 55/55/54/56; new arrivals are still balanced. Pods for the T5Gemma
easy arms (t5gs-5) and the slow objective host (t5gs-2, t5gs-3) were terminated unused at 12:52 JST.

**Objective, T5Gemma, under the corrected guard (13:10 JST, `reading_obj319_round2_t5gemma_gf.json`):** synthesized
on RunPod 5090 pods (`runpod_bundle_t5gsynth`, four index shards; the slow host's two shards were taken over by the
two fast pods with the same script and seeds, and the pod logs are kept under `t5gsynth/logs`), measured on the local
GPU. No edit 0.436, kana 0.906, tag 0.815; tag − kana −0.091 [−0.133, −0.050], p = 1.5e-5; tag − none +0.379
[+0.298, +0.463]; inaudible outputs: tag 4 of 319, comparators 0. The earlier round-2 T5Gemma row
(0.442 / 0.909 / 0.824) is the sensitivity check. All six pods terminated by 13:05 JST.

**Naturalness re-collection opened.** The CrowdWorks task for `jp-accent-naturalness-r3` was posted by the author at
about 13:40 JST on 2026-09-17 (90 slots, 1,000 yen, posting text as in the recruitment artifact: only the URL and the
stated duration changed from the 07:46 task). The 07:46 site (`-r2`) was rebuilt with `--closed` and redeployed at
13:42 JST (groups and clips unchanged, verified), so nobody new starts a sitting that the superseded collection would
neither use nor pay for; raters with stored answers can still finish and re-sync.

**Panel page defect found and fixed, 2026-09-17 15:20 JST.** The page counted progress as the number of stored
answers in the browser, not the number of answered items of the current block. For the rater who restarted on the
rebuilt block (amendment 9), the browser still held ten answers to the superseded C2/D2 cells, so the page showed
the completion screen at 471 of 481 real answers (the ten unanswered items: A2 2, B2 2, C2b 5, D2b 1). Found from
the store (463 of 473 stimuli plus catch trials, then no movement). The counter now counts only answers to items of
the block shown (`nDone` in the page and both templates); redeployed 15:20 JST; the rater is asked to reload and
finish the ten items. No answer was lost or altered. The rater still on the earlier build, and block 1, are not
affected (their stored answers all belong to the block they are shown).

**Interim look, 2026-09-17 15:30 JST (at the author's request, to draft the paper).** Store exported to
`responses_interim_20260917_1530.json`. Panel: two raters complete (R1, R2); the registered
analysis refuses confirmatory tests below three raters, so it was run with the explicit `--interim-min-raters 2`
flag added for this purpose, which stamps the report as interim and exploratory
(`data/analysis/v2/listening_report_interim_2raters.json`; cells C2b/D2b presented to the script under their
registered names by `scripts/analysis/prep_round2_panel.py`, judgements on the superseded C2/D2 stimuli dropped,
the incomplete third rater left out). Naturalness: the registered script on the re-collection as it stood, task
not yet closed, duplicate-account check not yet possible (`naturalness_report_r3_interim.json`). No design, item,
rater or stopping decision depends on this look: every admitted rater still judges everything, the naturalness
task closes when its posted slots are filled, and the final numbers replace the interim ones whatever they show.
What the interim look shows is unfavourable to the method on Sarashina (tag 0.62 vs kana 0.55, tag vs no edit
0.61 vs 0.64) and on naturalness (non-inferiority met on no backbone) and is reported as such.

## 10. Amendment, 2026-09-17 16:00 JST — exploratory bridge block: the preliminary round's Sarashina tag stimuli

**Why.** The interim look (above) shows the round-2 Sarashina adapter not separating from kana, where the
preliminary round's adapter had (0.89 vs 0.57, a different panel of four raters). Construction was audited and is
the same in both rounds (trainer, LoRA configuration, steps, teacher sampling and format, tag-token embeddings); the
teacher set's composition and the training run's randomness differ. To tell an adapter difference from a panel
difference, the author adds one exploratory measurement; it is not part of the confirmatory family, changes no
registered cell, and is reported whatever it shows.

**What.** Block 2 of the panel site: the 50 Sarashina tag stimuli that the preliminary round's raters heard, byte for
byte (round-1 kit, loudness-normalised clips), shown with the round-2 display of the same item (no prescribed accent
differs between the two kits). The round-1 kana and no-edit Sarashina stimuli are the same renditions as round 2's A2
(per arm 33 byte-identical and 17 same take under float noise), so the rater's own A2 judgements are the
comparators. Judged only by the rater the author recruits additionally (after that rater's block 0, invitation URL
with `&b=2`), blind, same UI and instructions, for an additional payment stated to the rater. Analysis: that rater's
share of words with the prescribed accent realized, round-1 tag stimuli vs round-2 tag stimuli vs kana, descriptive
only (one rater). `scripts/rater_ui/panel_r2_block2.py`; key kept out of the deployment.

## 11. Finding, 2026-09-17 16:35 JST — the preliminary round's Sarashina tag arm was not the clean-room adapter

Auditing why the round-2 Sarashina adapter runs away where the preliminary round's did not
(`scripts/eval/diag_which_adapter.py`; base loaded once, each candidate adapter merged onto a restored copy, same
reference voice, inputs, seeds and sampling as the round-1 synthesis script):

- The preliminary round's Sarashina **listening** tag stimuli (`sweep42/sarashina/*/ours`, 09-02) are reproduced
  **bit for bit** by `_handoff/adapters/lora_faithful2_tf457/ckpt_20000` and by no clean-room checkpoint
  (`lora_cleanroom_apostrophe_v2/ckpt_4000..20000`, `lora_cleanroom_apostrophe/ckpt_20000`). The no-edit arm is
  reproduced bit for bit by the base, so the comparison is sound.
- The preliminary round's Sarashina **objective** tag arm (`obj42g/sarashina/*/ours`) is a mixture: of four items
  checked, three are bit-identical to `lora_faithful2_tf457` and one to the clean-room v2 adapter (the partial
  re-synthesis of 09-09).
- `lora_faithful2_tf457` is the earlier, unreleased adapter, trained on an earlier, unreleased teacher set. It is
  the **default** of `SARA_ADAPTER_OURS` in `blockA_sarashina_difficult_4spk.py`, and was the adapter in effect when
  the round-1 stimuli were synthesized; the clean-room adapter the round-1 paper describes was not the one heard.
- Under today's conditions the clean-room v2 adapter of the preliminary round runs to the token cap on the same
  items as the round-2 adapter (first two listening items at every checkpoint 4k to 20k; 15 of 88 objective items
  needing a redraw, against 61 of 319 for the round-2 adapter). The round-2 Sarashina result is therefore what the
  clean-room recipe gives on this backbone, in both rounds; it is not a bad training draw.

**Consequences.** (i) Round 2 set `SARA_ADAPTER_OURS` explicitly for every Sarashina synthesis (`v2_chain.sh`), so
its Sarashina arms are the registered adapter; nothing in round 2 changes. (ii) The preliminary round's Sarashina tag
results (accent 0.89 vs 0.57, naturalness non-inferior, reading 0.821) are **withdrawn as evidence for the method**:
they describe a different adapter. They are not reported in the paper as the method's result. (iii) Amendment 10's
bridge block would bridge to that other adapter and is withdrawn before anyone judged it; block 2 is removed from the
site. (iv) The within-rater comparison computed at the interim look (the same raters scoring the round-1 tag stimuli
0.84 to 0.86 and the round-2 ones 0.60 to 0.64) stands as a fact about the two adapters, not about two draws of one
recipe. The other three backbones' round-1 arms were synthesized with explicit clean-room model paths
(`scripts/sweep42_one.sh`) and are not affected.

## 12. Exploratory follow-up, 2026-09-17 16:45 JST — Sarashina with five times the carriers

**Why.** Section 11 leaves one construction difference between the earlier, unreleased adapter, which does not run away,
and both clean-room adapters, which do: its teacher set held 25,000 carriers over 500 words (50 per word, longer and
more varied sentences) against 4,830 / 4,508 over 483 / 461 words (10 per word, frame-built). The author decided, after
seeing the round-2 Sarashina result, to test whether carrier count explains the failure. This is a results-driven
change and is reported only as exploratory; **Table 1 keeps the registered round-2 Sarashina adapter**, and no
listening or naturalness judgement is collected on the follow-up adapter.

**What.** Same 471 words, readings and frame-times-word construction; up to 50 carriers per word
(`scripts/teachers/build_text_v4.py`: the registered 10 kept, 18,839 added from the existing hand-authored frame banks
plus 42 new neutral and 40 new counter frames written before any output; odaka words take only frames with a
particle after the target). Same teacher rule (guard + reading check, five takes; `gen_sarashina_v2.py --max-takes 5`,
shards 0 to 3 on the local GPU, 4 to 11 on RunPod 5090 pods with the local environment's package versions and the local
weights copied to the pods), same trainer, LoRA configuration and 20,000 steps, adapter path given explicitly. Measured:
the tag arm's redraw and cap-exhaustion rates and reading accuracy on the 319 objective items, same voices, seeds and
guard. Stopping rule fixed now: if no result exists by 2026-09-19 the follow-up is dropped and said so. Whatever it
shows is reported in one to three sentences.

**Success criterion for section 12, fixed 2026-09-17 18:32 JST, before any output of the 50-carrier adapter existed**
(training at step ~3,300 of 20,000; the first checkpoint not yet written). Reference values under the same 319 items,
voices, seeds and guard: Sarashina no edit 3/319 redraws and kana 3/319, the earlier, unreleased adapter's tag arm 2/233,
CosyVoice 2's tag arm 3/319, Irodori's 0/319, none of them with an item exhausting the six takes; the registered
Sarashina adapter's tag arm 61/319 with 24 exhausted. **Success:** the 20,000-step adapter's tag arm needs a redraw on
at most 6 of 319 items (2%), exhausts the takes on none, and reads at least 0.80 (registered adapter 0.755, kana 0.843).
**Partial:** redraws above 2% but at most about 8%. **No effect:** 10% or more. The 4,000-step early read (100 items;
the registered adapter's 4,000-step checkpoint needed 11 redraws) only decides whether the other three backbones'
extended teacher sets are generated tonight (yes unless it is about 10 again); it is not reported as a result.

## 13. Amendment, 2026-09-17 19:00 JST — 74 of the 319 objective tag inputs were malformed; corrected and re-synthesized

- **Defect.** The author noticed `ご令<PHON_START>ゴレーエン<PHON_END>` in a synthesis log. The accent review of 09-09 had
  widened 96 objective targets to a phrase (足痕 → アシア'ト, 弓状 → キュージョー) and written correct tagged lines, but the
  routing step run afterwards (`scripts/eval/route_ours_inputs.py`) took the span to wrap from the plain-versus-kana
  diff, which still names only the original kanji. On the 319-item test half, 61 inputs therefore carried the phrase's
  reading on the one-kanji span (`足<アシア'ト>`, the neighbour said twice) and 13 had an absorbed suffix's reading
  appended to a reading that already contained it (`キュージョージョー`): **74 malformed tag inputs**, on all four
  backbones' tag arms, in both rounds since 09-09. Irodori's own case loader made the same choice at run time. The
  inputs before routing contain no such case; the listening (difficult-50) and naturalness (easy-50) inputs and the
  training carriers carry no widened reading and are unaffected (checked: 0 of 50 and 0 of 50 spans change under the
  corrected rule; the two apparent repeats in difficult-50 are real reduplications, 粛々 and 延々).
- **Fix.** Both the routing script and Irodori's loader now take the span from the tagged line itself. The corrected
  file replaces the tag exactly by the reviewed span with the reviewed reading on 319 of 319 items
  (`test_obj319_lora_fixed20260917.txt`, now the canonical `test_obj319_lora.txt`; the malformed file is kept as
  `test_obj319_lora.dupbug_20260909.txt`; the 74 indices in `changed_by_fix20260917.json`). The kana and no-edit
  inputs, the truth targets, voices, seeds, guard and adapters are unchanged.
- **Re-synthesis and re-measurement.** The tag arm of the 74 items is re-synthesized on all four backbones with the
  registered adapters (`scripts/eval/obj319_inputfix_resynth.sh`; earlier clips kept under `obj319_dupbug`), and the
  objective table is re-measured (`reading_obj319_round2_inputfix.json`). That table is the record; the table on
  the malformed inputs is kept as a sensitivity check. On the malformed inputs the tag-minus-kana gap was already
  present on the 245 well-formed items (Sarashina -0.06, CosyVoice 2 -0.18, Irodori -0.03, T5Gemma -0.10), and
  Sarashina's redraw rate on them was 18% against 23% and 31% on the two malformed kinds, so the defect is not what
  produces either finding; the corrected numbers are reported whatever they show.
- **Section 12, early read (not a result).** The 50-carrier adapter's 4,000-step checkpoint needed a redraw on about
  11 of 100 items, as the registered adapter's did; by the rule fixed at 18:32 the other three backbones' extended
  teacher sets are not generated tonight. Its training was restarted from scratch after the first run lost its GPU
  memory to a diagnostic run beside it (1 step/s instead of 5); the restart follows the re-synthesis above, alone on
  the GPU, and its evaluation uses the corrected inputs.

**Block 2 restored, 2026-09-17 19:45 JST.** The author had already contracted one rater for the bridge block of
amendment 10 before section 11 withdrew it. The block (the 50 round-1 Sarashina tag stimuli, byte for byte) is put
back on the site so that the contract can be honoured and paid; its judgements are archived and **not analysed or
reported**, since section 11 shows those stimuli came from a different adapter. No other block or rater is affected.

**Objective table of record after amendment 13 (measured 2026-09-17 19:40 to 19:51 JST,
`reading_obj319_round2_inputfix.json`).** No edit / kana / tag: Sarashina 0.511 / 0.843 / 0.771, CosyVoice 2
0.282 / 0.850 / 0.690 (released UtterTune 0.856), Irodori 0.586 / 0.909 / 0.871, T5Gemma 0.436 / 0.906 / 0.809.
Tag - kana: -0.072 [-0.120, -0.019], -0.160 [-0.212, -0.107], -0.038 [-0.075, +0.000], -0.097 [-0.137, -0.058];
tag - none: +0.260, +0.408, +0.285, +0.373 (all p < 1e-12); tag - UtterTune on CosyVoice 2: -0.166 [-0.226, -0.106].
The correction moved the tag arm by +0.016, +0.003, +0.003 and -0.006; no conclusion changes. Sarashina's tag arm
still needs a redraw on 60 of 319 items and exhausts the six takes on 20 (61 and 24 on the malformed inputs), so
the malformed inputs were not the cause of the runaway.

**Naturalness re-collection closed, 2026-09-17 20:05 JST.** The author closed the CrowdWorks task; the `-r3` site was
rebuilt with `--closed` and redeployed (groups unchanged, verified live). Store exported to
`responses_final_nat_r3_20260917.json`; registered analysis (`analyze_naturalness.py --exclude-speeders`,
`data/analysis/v2/naturalness_report_r3.json`): 114 sittings started, 105 complete, two excluded by the speed rule,
5,194 judgements on 198 pairs (24 to 29 per pair). Unedited minus edited preference: Sarashina +0.34 [+0.19, +0.49],
CosyVoice 2 +0.29 [+0.13, +0.44], Irodori +0.15 [+0.04, +0.27], T5Gemma +0.28 [+0.10, +0.44]; the 0.15 margin is met on
none. Same-reading stratum (151 pairs): +0.26, +0.33, +0.15, +0.26. The numbers equal the interim look's, no judgement
having arrived after it. The duplicate-account exclusion (second sitting by one platform account) is applied when the
author supplies the platform's worker list; until then the figures are reported as before that check.
Exploratory, not registered: the pair-level breakdown (`naturalness_r3_per_pair.json`), the relation of the preference
to the sentence CER change (`naturalness_r3_cer.json`), sentence damage on the listening stimuli
(`difficult50_cer_tag_vs_kana.json`) and UTMOSv2 on the pairs (margin 0.20 from the preliminary registration;
`utmosv2_naturalness_pairs_r3.json`).


## 14. Exploratory cause hunt, 2026-09-17 20:38 JST — five Sarashina teacher-set variants trained in parallel

**Why.** Sections 11 to 13 leave the Sarashina runaway unexplained: code and configuration are identical to the
earlier, unreleased adapter's, carrier count does not move it at 4,000 steps, and the malformed inputs were not the cause.
Three measurable properties of the clean-room teacher sets differ from the earlier, unreleased set: the tagged word opens
the sentence in most carriers, some teachers contain long runs of one repeated speech token, and some teachers are
slow (many speech tokens per character). The author asked for these to be tested tonight, in parallel on rented GPUs,
because 2026-09-18 is the last day on which listeners for a further registered round could still be recruited.

**What (written before any of these adapters produced output; all five were at most a few hundred steps into
training).** Registered trainer and settings (`train_lora_sarashina.py`, 20,000 steps, alpha 64), one RTX 5090 pod each,
public backbone weights pinned to the commit of the local copy (sha256 of the large file checked). Variants are filters
of the section-12 teacher set (22,729 rows; clean-room only, no data of the earlier, unreleased set leaves the local machine):
P1 tag never sentence-initial (8,755 rows, 465 words); P2 no run of 25 or more identical speech tokens (20,750);
P4 at most 5.5 speech tokens per character (12,870); P5 all three filters (6,378); P3 the registered teacher set
unchanged, stopped after 3,000 steps, to read checkpoints at 1,000 / 2,000 / 3,000 steps. Each adapter is read on the
first 100 objective items' tag arm (corrected inputs, registered voices, seeds and guard;
`runpod_bundle_sarahyp/RUNPOD_RUN.sh`), at 20,000 and 4,000 steps. References on the same 100 items: registered adapter
at 4,000 steps 11 redraws, at 12,000 steps 21; the 50-carrier adapter at 4,000 steps 10; on all 319 items the registered
adapter 60 redraws (19%), Sarashina's unedited and kana arms 3 each (1%).

**Criterion, fixed now.** A variant counts as **removing the runaway** only if its 20,000-step adapter needs a redraw
on at most 2 of the 100 items and exhausts the six takes on none. 3 to 6 is **partial**; 7 or more is **no effect**.
A variant that meets the criterion is then measured on all 319 items against the section-12 criterion before anything
is said about it. These runs are exploratory, select among variants after seeing results, and **do not replace any
registered number**; if one succeeds, the author decides whether a further registered round (recipe change applied to
all four backbones, new stimuli, new judgements) is feasible, and the present round's results stay on record either way.
If none succeeds, the paper reports the Sarashina failure as it stands and says which causes were ruled out.

**Section 14, sixth variant added 21:18 JST (before any output of the first five).** Asked by the author whether
training and evaluation voices were separated for the earlier, unreleased adapter, the generators were compared: neither
teacher set contains a speaker (both are the base's text-only continuations), so no voice can leak; but the
earlier, unreleased set was taken **without selection** (first take of every carrier), while the clean-room recipe keeps
only takes that pass the guard and the reading check (registered Sarashina set: 3,846 of 4,508 rows are first takes,
662 are redraws, 202 carriers dropped). P6 tests that difference: the registered 4,710 carriers, first take of each
with the registered sampler and seeds, no guard, no reading check (`runpod_bundle_sarahyp6/scripts/gen_unfiltered.py`),
then the registered trainer and the same 100-item reading. Same criterion as the other five. Note for any later use:
an unselected teacher set contains carriers the base misreads, so P6 is a diagnostic of the runaway, not a candidate
recipe, unless its reading accuracy also holds.

**Section 12, result (measured 2026-09-17 20:56 to 21:36 JST; `data/analysis/v2/reading_obj319_sarashina_v4.json`,
adapter `v2/sarashina_v4/lora/ckpt_20000` given explicitly, corrected inputs, registered voices, seeds and guard).**
The 50-carrier adapter's tag arm needed a redraw on **41 of 319** items (12.9%) and exhausted the six takes on **12**
(registered adapter 60 and 20); 12 outputs were inaudible. By the criterion fixed at 18:32 this is **no effect on the
runaway** (10% or more). Reading accuracy of the tag arm was 0.850 against the registered adapter's 0.771 and plain
kana's 0.843 (tag - kana +0.006 [-0.044, +0.059]); voice disagreement 0.200. So five times the carriers raises the
objective reading accuracy to the kana baseline's level but does not remove the failure that matters to a listener.
Exploratory, results-driven, not in Table 1. Redraws again concentrate on a few reference voices (spk3 7 of 8 items,
spk20 7 of 10, spk14 5 of 12), the same voices on which the registered adapter runs away (spk14 11 of 12, spk20 8 of
10, spk16 8 of 11, spk3 6 of 8; 21 of 42 voices never needed a redraw) and the only voices on which the unedited and
kana arms needed redraws at all.

**Section 14, first reads and two further diagnostics (2026-09-17 21:47 JST).** P3 (registered teachers, early
checkpoints), 100 items each: 1,000 steps 14 redraws / 3 exhausted, 2,000 steps 15 / 2, 3,000 steps 12 / 3. The
runaway is present from the first thousand steps at the level it keeps later, so it is not a late over-training
effect and no stopping point removes it. Mid-training reads of P1 and P2 at 8,000 steps (progress checks, not
results): 15 / 6 and 10 / 0. Two diagnostics started at 21:46, before any output: **L1**, today's trainer and
environment on the earlier, unreleased teacher set (local machine only, that set never leaves it; internal control, never
reported as a result: if L1 runs away the cause is in today's code, environment or inputs, if it does not the cause
is in the clean-room teacher data); **no-tag inputs**, the registered 20,000-step adapter loaded but fed the untagged
plain and kana sentences of the same 100 items (if it runs away there too, the adapter has damaged the model's
stopping behaviour in general for some voices; if not, the tagged span triggers it).

**Section 14, results (read 2026-09-18 02:31 JST; pods finished between about 22:10 and 23:50 on 09-17; all files under
`v2/sara_hyp/pods/`).** Tag arm, first 100 objective items, registered voices, seeds and guard; redraws / items
exhausting the six takes:

| variant | teachers | 20,000 steps | 4,000 steps |
|---|---|---|---|
| P1 tag never sentence-initial | 8,755 | 11 / 2 | 13 / 3 |
| P2 no long runs of one speech token | 20,750 | 14 / 3 | 9 / 0 |
| P3 registered set, early checkpoints | 4,508 | (1,000: 14 / 3; 2,000: 15 / 2; 3,000: 12 / 3) | |
| **P4 at most 5.5 speech tokens per character** | 12,870 | **1 / 0** | 2 / 0 |
| P5 all three filters | 6,378 | crashed at item 73 (vocoder on a two-token output), 2 redraws until then | 4 / 0 |
| P6 teachers without selection | 4,710 | 18 / 3 | 16 / 5 |
| L1 today's code on the earlier, unreleased self-distilled set (local) | 25,000 | 9 / 0 | 14 / 1 |
| no-tag inputs, registered adapter | | plain 16 / 2, kana 16 / 3 | |

By the criterion fixed at 20:38, **P4 removes the runaway on the 100 items; P1, P2 and P6 have no effect**; P5 is
unread at 20,000 steps. Readings: the registered adapter runs away on untagged input as often as on tagged input, so
the adapter has shifted the model's stopping behaviour in general and the tagged span is not the trigger; the
runaway is present from the first thousand steps; selection of teachers by the reading check is not the cause (P6);
the one property that matters is **slow teachers**: the base's text-only continuations are often slow (registered
set: 56% of teachers above 5.5 speech tokens per character, 27% above 8, 7% above 12), and an adapter trained on
them lengthens every output until, on a few reference voices, generation no longer stops.

**Correction to section 11.** Section 11 and later notes called `lora_faithful2_tf457` an adapter trained on the
earlier, unreleased self-distilled set. `docs/internal/HANDOFF.md` and the file itself show otherwise: its teacher file
`faithful_teachers.jsonl` holds 14,970 rows keyed by JSUT utterance ids, that is, **speech tokens of real
recordings**. The one Sarashina adapter that never ran away was therefore trained on recorded speech, and L1 shows
that the earlier, unreleased *self-distilled* set runs away under today's code as the clean-room sets do (9 of 100). This
removes the puzzle of section 11: nothing was lost between the rounds; self-distilled text-only teachers had this
failure on Sarashina all along, and the preliminary round never exercised them. It also makes the withdrawal of the
preliminary Sarashina results doubly necessary (their adapter used recorded speech, which the method excludes).

**Follow-ups started 02:30 to 02:33 JST on 09-18, before any output:** P4's 20,000-step adapter on all 319 items against
the section-12 criterion (local, `scripts/eval/sarahyp_P4_full.sh`); P7, the *registered* 4,508-row set with the
same filter (1,965 rows, 453 words), to see whether the filter alone suffices without the extra carriers; P9, the
section-12 set filtered at 8 instead of 5.5 tokens per character (18,372 rows), to see how sharp the threshold is.
Four pods terminated at 02:33; two remain for P7 and P9. All exploratory; Table 1 is unchanged.

**Panel, registered minimum reached (2026-09-18 02:38 JST).** A third valid rater (R3) completed all 481 items at
22:33 on 09-17 (catch 6/8, consistency 20/23). The registered analysis was run without the interim flag
(`data/analysis/v2/listening_report_3raters_20260918_0234.json`; KV export `v2/responses_panel_r2_20260918_0234.json`):
Fleiss' kappa 0.89 over 437 stimuli, no ties. Tag vs kana: Sarashina 0.65 vs 0.56 (Holm p 0.85), CosyVoice 2 0.83 vs
0.38 (Holm p 1e-4), Irodori 0.86 vs 0.48 (Holm p 1e-4), T5Gemma 0.78 vs 0.57 (raw p 0.041, Holm p 0.12); Sarashina tag
vs no edit 0.65 vs 0.67. A fourth rater (R4) is at 139 of 481; if that rater completes before the panel is
closed, the analysis is rerun on all complete valid raters and that run replaces this one, whatever it shows. The
panel closes when the author closes recruitment; no result decides that.

## 15. Exploratory, 2026-09-18 02:46 JST — validation on the selection split; carriers matched to running text

**Author's decision (2026-09-18 02:46).** Beyond the runaway, look on Sarashina for a teacher-set condition that lowers the
damage the edit does to the rest of the sentence (sentence kana CER), which sections 9 to 13 found associated with the
naturalness preference. Naturalness cannot be searched objectively; CER can. Conditions are compared **only on the
selection split** (`preregistration/items/objective_selection.jsonl`, 105 items, component-disjoint from the 319 test
items, frozen before any adapter was measured, registered purpose: model choice), tag inputs routed by the corrected
router (`data/derived/sel105_routed/`), one voice per item by `sha256('SEL105:'+key) mod 42`
(`speaker_assignment42_sel105.json`, fixed before any audio). Measures: redraws and exhausted items under the
registered guard; sentence kana CER of the tag arm against the kana and unedited arms of the same items and voices;
whether the target reading is heard. A condition that does not run away and lowers the validation CER is then checked
by the author by ear, and only then applied to the other three backbones. The 319 test items and the listening sets are
not used to choose anything from here on. (Section 14's variants were read on the first 100 *test* items; that was a
diagnosis of a failure, and it is disclosed as selection on test data for the runaway criterion only.)

**Hypothesis of the author.** Teachers should follow the distribution the edit meets in use: in the registered
teacher set the tagged word opens the sentence in 93% of carriers and carriers are 14 to 19 characters long, while in
running text an edited word sits anywhere in a sentence of ordinary length. Matching these is a property of the
deployment condition, not of test items. **v5 carriers** (`scripts/teachers/build_text_v5.py`): the same 471 words
and readings, 24 new neutral and 16 new counter frames written before any output and without consulting evaluation
sentences, word at relative position 0.43 / 0.56 / 0.69 (10th / 50th / 90th percentile), length 24 / 27 / 31
characters; 10,868 carriers; teachers by the registered rule (guard, reading check, five takes) on six pods.
Planned adapters, all with the slow-teacher filter of section 14 (at most 5.5 speech tokens per character), registered
trainer and settings: **M2** section-12 pool plus v5; **M3** the same, resampled so that relative tag position is
flat over thirds of the sentence. Baselines on the same split: registered adapter, section-12 adapter, P4.

**Section 14, P4 on all 319 items (2026-09-18 02:50 JST; `data/analysis/v2/reading_obj319_sarashina_P4.json`, adapter
`v2/sara_hyp/pods/shy-3r/hyp/P4_not_slow/lora/ckpt_20000` given explicitly).** Redraws **4 of 319** (1.3%), none
exhausting the six takes, no inaudible output; tag reading accuracy **0.875** (kana 0.843, registered adapter 0.771;
tag - kana +0.031 [-0.016, +0.080]); voice disagreement 0.147; sentence kana CER from the guard's stored transcripts
0.077 against kana's 0.074 and the registered adapter's 0.173. All three parts of the section-12 criterion are met
(at most 6 redraws, none exhausted, reading at least 0.80). Caveat, stated once: P4 was chosen among six variants on
the first 100 of these 319 items, so this table is not an unbiased test result; it is exploratory and replaces
nothing. From here on choices are made on the selection split only (section 15).

**Section 15, second validation set (02:53 JST, before any audio of it).** The naturalness deficit was measured on
words the base already reads right, so side effects are also validated on such words: `data/derived/deveasy/`, 39
common words in ordinary sentences (length 18 / 21 / 24 characters, relative tag position 0.14 / 0.40 / 0.65), tags
from the front-end's own reading and accent, words disjoint from the 471 teacher words and from the targets of the
easy-50 and difficult-50 sets (`scripts/eval/build_deveasy.py`; 61 candidates written, 22 dropped by the disjointness
rule). Measure: sentence kana CER of the tagged rendition minus the unedited rendition of the same sentence and voice.

**Section 15, carriers v5b (02:58 JST, before any v5 or v5b output).** Asked by the author whether v5 matches the
evaluation sentences: alone it does not. Summary statistics (length in characters and relative tag position, 10th /
50th / 90th percentile): objective 319 items 13 / 20 / 26 and 0.08 / 0.43 / 0.73; easy-50 22 / 29 / 36 and 0.00 / 0.35 /
0.74; difficult-50 21 / 24 / 32 and 0.06 / 0.18 / 0.47; registered teachers 14 / 16 / 19 with the tag first in 99%; v5
24 / 27 / 31 and 0.43 / 0.56 / 0.69. v5 leaves uncovered short sentences with a late word, words early but not first,
and long sentences. **v5b** (`scripts/teachers/build_text_v5b.py`, 10,655 carriers, 24 neutral and 12 counter frames)
fills these: length 10 / 26 / 39, position 0.07 / 0.25 / 0.63. Only these summary statistics of the evaluation sets
were used; no evaluation sentence was read for the purpose. Six further pods generate v5b with the registered rule.
The planned adapters M2 and M3 now draw on the section-12 pool plus v5 plus v5b.

**Panel hold, 2026-09-18 03:00 JST.** At the author's request the one invited rater who has not started (R5) sees the hold screen again (`index.json` hold list), because a further round may replace the stimuli; the rater in progress (R4, 139 of 481, last active 22:48) is not affected. No rater was active at deployment; stimuli and order unchanged.

**Section 15, two further conditions (03:07 JST, decided by the author before any validation number existed).**
Independent of the carrier-distribution hypothesis, two conditions that need no new teachers, each on the P4 pool
(12,870 rows), registered trainer and settings, read on the two validation sets only:
**C1 clean sentences**: the registered reading check looks at the target word only, so a teacher whose *rest* is
garbled passes; C1 keeps teachers whose whole-sentence kana CER (stored generation transcript against the front-end
reading of the carrier with the target reading substituted) is at most 0.05, a threshold fixed now: 8,125 rows (in
the pool 36% exceed 0.05 and 10% exceed 0.10). **C2 untagged copies**: all 12,870 tagged rows plus an untagged copy
(plain text as the prompt, same speech tokens) of a seeded random 30% (3,862 rows), so that the adapter is also
trained to leave untagged text alone. **C3** both (10,559 rows). Files `v2/sara_hyp/teachers_C*.jsonl`.

**Section 15, validation results (read 2026-09-18 06:50 JST; the session that watched the runs died at about 03:30 and
the pods sat idle until 06:45; all outputs are dated on the pods). `scripts/eval/sarahyp_val_summary.py`, sentence
kana CER from the guard's stored transcripts. Selection split, 105 items (unedited CER 0.109, kana 0.098, target
heard 0.59 / 0.84) and side-effect set, 39 items (unedited 0.058, kana 0.044, target heard 1.00 / 1.00):

| tag arm | sel105 redraw / exhausted | heard | CER | tag - unedited | deveasy redraw | heard | CER | tag - unedited |
|---|---|---|---|---|---|---|---|---|
| registered adapter | 26 / 11 | 0.73 | 0.196 | +0.087 | 5 (2 exhausted) | 0.85 | 0.127 | +0.069 |
| section-12 (50 carriers) | 21 / 6 | 0.79 | 0.137 | +0.027 | 7 | 1.00 | 0.037 | -0.021 |
| P4 slow teachers removed | 2 / 0 | 0.80 | 0.089 | -0.020 | 0 | 0.92 | 0.043 | -0.015 |
| C1 clean sentences | 1 / 0 | 0.84 | 0.092 | -0.017 | 0 | 0.95 | 0.045 | -0.013 |
| C2 untagged copies | 1 / 0 | 0.87 | 0.082 | -0.027 | 0 | 0.97 | 0.039 | -0.019 |
| C3 both | 0 / 0 | 0.83 | 0.081 | -0.028 | 0 | 0.95 | 0.050 | -0.009 |

Reading: on both validation sets the registered adapter is the outlier (CER twice the kana arm's, target heard less
often than kana, runaway on a fifth of the items); every condition with the slow-teacher filter sits at the kana
arm's CER on both sets. Among P4, C1, C2 and C3 the differences (0.081 to 0.092; 0.039 to 0.050) are within what 105
and 39 items can resolve and are not read as differences. C2 is nominally best on both sets and hears the target
most often (0.87; 0.97). On the side-effect set the edit still loses the target word itself on 1 to 3 of 39 items
under P4, C1 and C3 (heard 0.92 to 0.95 against 1.00 unedited), on 1 under C2. Section-14 follow-ups on the first
100 test items: P7 (registered set, filter only, 1,965 rows) 8 redraws / 1 exhausted, so the filter alone on the
small registered set is not enough; P9 (threshold 8 instead of 5.5, 18,372 rows) 1 / 0, like P4.
Teacher shards v5 and v5b: 11 of 12 pods finished (kept 1,717 to 1,762 of about 1,776 each); shard 00's pod failed
with a CUDA error and the shard is being regenerated; M2 (pool + v5 + v5b, slow removed) and M3 (M2 plus 30%
untagged copies) train as soon as it lands, on the two pods kept for that.

## 16. Round 3, registered 2026-09-18 06:58 JST — one recipe change on all four backbones, before any round-3 output

**Decision (author, 06:55).** Sections 14 and 15 found the Sarashina failure's cause (slow self-distilled teachers) and
a condition that removes it and brings the edit's sentence damage to the kana arm's level on both validation sets.
The author decided to apply the change as **one recipe change to all four backbones** and to re-collect the listening
and naturalness judgements on new stimuli. The change is fixed now, before any round-3 teacher, adapter or stimulus
exists, and is applied identically:
1. **Carriers:** the section-12 text (the registered 471 words, up to 50 carriers per word, `teachers_text_v3.jsonl`
   plus `teachers_text_v4_new.jsonl`), on every backbone. Teachers by each backbone's registered generator and rule
   (guard, reading check, five takes), unchanged.
2. **Slow-teacher filter:** a teacher is dropped when its speech representation runs longer than **0.22 seconds per
   character of the plain carrier** (Sarashina 5.5 semantic tokens at 25 Hz, the P4 rule; CosyVoice 2 by the teacher
   waveform's duration; Irodori 5.5 latent frames at 25 Hz; T5Gemma 11 codec frames at 50 Hz). The rule is the same
   number on every backbone whatever it removes (expected: about half on Sarashina, a smaller share on CosyVoice 2,
   almost nothing on Irodori and T5Gemma, whose teachers are generated with a voice prompt and are not slow).
3. **Training, guard, synthesis, voices, seeds, inputs, evaluation and analysis:** exactly as registered for round 2
   (sections 1 to 13), adapter paths given explicitly and recorded with a hash.
Not part of the change (validated as within noise, section 15): untagged copies (C2), whole-sentence teacher
selection (C1), mid/late-position carriers (v5/v5b). If M2/M3 (section 15) finish before the round-3 adapters are
trained and show a clear gain on both validation sets, the author may add v5/v5b carriers to item 1 for all four
backbones and will say so here before use; otherwise they are not used.
**Stimuli and judgements:** new difficult-50 listening stimuli and easy-50 naturalness pairs on all four backbones,
same items, voices, seeds, kits, screening, catch trials, exclusions and analyses as round 2; naturalness pay 700
yen per sitting (round 2: 1,000), a change of payment only. The round-2 panel (three raters, section "registered
minimum reached") and naturalness results stay on record as round 2; round 3 replaces them in the paper only if
its data are complete; otherwise the paper reports round 2 and says round 3 was started.
**Objective table:** the 319 test items are measured once per backbone with the round-3 adapters, after the
adapters are frozen, and reported as round 3's table of record; the selection split remains the only place where
anything was chosen.

**Section 15, M2/M3 started 07:10 JST; author's blind check.** At the author's request M2 and M3 were built without
waiting for the twelfth v5 shard (its pod failed twice; 19,136 of about 20,900 v5/v5b rows are in): pool 41,865 rows,
15,246 removed as slow, **M2 26,619 rows** (relative tag position 0.00 / 0.28 / 0.63, length 16 / 24 / 32), **M3
34,531** (M2 plus 30% untagged copies). Training started 07:09 on two pods, validation on both sets follows in the
same chain. **Blind check by the author** (`scripts/eval/build_val_blind_page.py`, local page): for every
validation item, the edited rendition of one condition against the unedited rendition of the same sentence and
voice, sides and order shuffled with a fixed seed, condition hidden until the tally is requested; choices are
"A / B more natural", "no difference", "A / B has an error". The author's judgements choose among conditions on the
validation split only and **never enter a reported number**; the round-3 recipe (section 16) is not changed by them
unless a change is written here before use.

**Section 15, early read of M2/M3 at 4,000 steps (2026-09-18 07:31 JST; not a result, the 20,000-step adapters follow).**
Selection split: M2 1 redraw / 0 exhausted, heard 0.85, CER 0.077; M3 3 / 0, 0.87, 0.070 (P4 2 / 0, 0.80, 0.089; C2
1 / 0, 0.87, 0.082). Side-effect set: M2 0, 0.97, 0.041; M3 0, 0.97, 0.044 (P4 0, 0.92, 0.043). A re-synthesis of the
C2 adapter on the local GPU gave 0.077 / heard 0.81 against the pod's 0.082 / 0.87 on the same items and seeds, which
calibrates the noise: differences of this size between conditions are not differences. So far the carrier
distribution does not move the validation CER; the blind check and the 20,000-step read decide whether v5/v5b are
added to section 16 (they would be an addition to the section-12 carriers, applied to all four backbones).

**Section 16, note (07:43 JST, before any round-3 teacher of the three backbones was collected).** The expectation
written in item 2 ("almost nothing on Irodori and T5Gemma") was wrong: applied to the round-2 teacher sets the rule
would drop 43% of Irodori's (median 0.211 s/char, 90th percentile 0.284), 36% of CosyVoice 2's (0.181 / 0.319), 8% of
T5Gemma's (0.194 / 0.220), and 56% of Sarashina's (0.234 / 0.450). **The rule is kept as registered**: one number,
0.22 s per character, on every backbone; with 50 carriers per word every backbone keeps more teachers than it had in
round 2. Filter: `scripts/eval/round3_filter_teachers.py`; training and stimulus synthesis:
`scripts/eval/round3_train_local.sh` (round-2 commands, round-3 paths; Sarashina's adapter is P4, hash in
`v2/sara_hyp/P4_adapter_sha256.txt`, its stimuli synthesized 07:08 to 07:18).

**Section 15, validation inputs checked (2026-09-18 07:52 JST).** Asked by the author whether the validation sets'
readings and accents were verified: the selection split's tags were written on 09-03 and the author's accent review of
09-09 (`objective_accent_review.tsv`, all 424 items) had been applied to the test half only; on the selection half 89 of
105 tags equal the reviewed accent and 16 differ only in where the accent mark sits relative to a long-vowel mark
(ショー'ジョー against reviewed ショ'ージョー), readings identical on all 105. The reviewed marks are now applied to
`test_sel105_lora.txt` (previous file kept as `*.prereview_20260918.txt`); every validation number reported so far
used the pre-review marks, identically for all conditions, so the comparisons stand; later runs use the reviewed file.
The side-effect set's 39 tags come from the front-end (pyopenjtalk) and were read through by the author's assistant
item by item; they are standard Tokyo accents; the author can inspect them on the listening page, where each item's tag
is shown in brackets.

**Section 15, author's review of the validation inputs applied (08:13 JST).** The author reviewed all 144 validation
targets in the click-through review UI (`preregistration/items/val_accent_reviewed.json`): 61 of 105 selection tags and
2 of 39 side-effect tags changed (22 by span, the rest by nucleus position; readings of the spans as the author wrote
them; list in `data/derived/sel105_routed/val_review_changes_20260918.json`). The inputs are rebuilt from the review
(`test_sel105_lora.txt`, `test_deveasy_lora.txt` and its kana line; previous files kept as `*.before_review_20260918.txt`).
**Router change (2026-09-18, `phon_router.py`, backup `.bak_20260918`):** when the router absorbs a following raw-kanji
suffix into the tag, a registry entry written for the merged span now wins outright; otherwise the suffix reading is
appended as before, the head's nucleus is kept, and the router reports the merged span as accent-unverified so that a
reviewer decides (the 09-18 review answered the five such spans in the selection split). On the reviewed selection
inputs the router changes no line. Every validation number so far was measured on the pre-review inputs, identically
for all conditions; all conditions are re-synthesized on the reviewed inputs and the earlier numbers are superseded.
For round 3 (section 16) the listening and naturalness inputs are the round-2 files (author-verified against the NHK
dictionary in round 1, `data/author_verification/nhk_difficult.json`), unchanged; Irodori's loader routes them at run
time and the router change can only affect an item whose tag abuts a raw-kanji suffix.

**Section 16, amendment (08:35 JST, author's decision on design grounds, before any round-3 teacher of the three
backbones was collected).** Item 1 (carriers) now reads: the section-12 text **plus the v5 and v5b carriers of
section 15** (21,523 more carriers: word in the middle or late in the sentence, sentences of running-text length), on
all four backbones. The validation reads of section 15 showed no measurable gain from these carriers (M2 within the
noise of P4 and C2 on 144 items); the author adds them because the teacher distribution should follow the
distribution the edit meets in use, not because of an effect, and the paper will describe the carrier set as it is
without claiming an effect for this part. Sarashina's round-3 adapter is therefore retrained on the full section-12 +
v5 + v5b pool with the slow-teacher filter (M2 was trained on 11 of 12 v5 shards; the twelfth is regenerated first);
P4 is superseded for round 3. Untagged copies (M3) are not part of the recipe. The v5/v5b teachers of the three
backbones are generated on the same pods right after their section-12 shards, same generator and rule.

**Section 16, Sarashina step count chosen on the selection split (08:44 JST, rule fixed before the round-3 adapter
exists).** The registered step counts were fixed a priori and never tuned. For Sarashina, whose round-3 adapter is
being retrained anyway, the checkpoint (4,000 / 8,000 / 12,000 / 16,000 / 20,000 steps) is chosen on the two
validation sets by **mean sentence kana CER of the tag arm over the 144 items (lower is better)**; a checkpoint within
0.005 of the best is a tie, and ties go to the higher mean UTMOSv2 of the tag arm; a checkpoint that needs a redraw on
more than 2% of the 105 selection items or exhausts the takes on any is excluded. The choice is written here before
any round-3 Sarashina stimulus is synthesized. The other three backbones keep their registered (untuned) step counts;
the paper says so.

**Section 15, author's blind check on the side-effect set (2026-09-18, judged 10:00 to 10:40 JST, file
`data/analysis/v2/author_blind_deveasy_20260918.json`).** 39 items x 6 conditions, edited against unedited, sides and
order shuffled, condition hidden. Prefer edited / prefer unedited / no difference (edited flagged as having an error):
registered adapter 3 / 13 / 23 (5 flagged); section-12 6 / 8 / 25 (1); P4 5 / 5 / 29 (1); C2 9 / 9 / 21 (0); M2 7 / 7 /
25 (0); M3 5 / 7 / 27 (0). The author's ear agrees with the CER: only the registered adapter loses to its own unedited
rendition; every condition with the slow-teacher filter is a tie, and P4, C2, M2 and M3 are not distinguishable. These
judgements choose nothing beyond confirming section 16's recipe and never enter a reported number.

**Section 16, amendment withdrawn (10:48 JST, author's decision after the blind check; before any round-3 stimulus
of the three backbones).** The v5/v5b carriers (amendment of 08:40) are **not** added: on the validation sets and in the
author's blind check they were indistinguishable from the slow-teacher filter alone, and their teacher generation on
CosyVoice 2 and T5Gemma is the slowest step. Item 1 stands as first registered: the section-12 carriers on all four
backbones. Sarashina's round-3 adapter is **P4** (section 14; hash in `v2/sara_hyp/P4_adapter_sha256.txt`), whose
stimuli were synthesized 07:08 to 07:18; the retrained adapter on the wider pool (R3_sarashina, five checkpoints
validated) is kept as exploratory. Irodori's training is restarted on the section-12 rows only (13,901 of 22,548 after
the filter); CosyVoice 2 and T5Gemma use their section-12 shards only. The v5/v5b teachers already generated are kept
for later exploratory use and enter nothing here.

**Section 16, stimulus preparation (10:56 JST, before any round-3 kit exists).** The author noticed clips that open
with long silence. Measured on the round-2 kits: leading silence over 1 s occurs in 16 to 19 of 50 Sarashina clips in
every arm (no edit, kana, tag; up to 9.9 s; a base-model behaviour with some reference voices), in 0 CosyVoice 2 and
Irodori clips, and in 0 to 2 T5Gemma clips; the round-3 Sarashina tag arm (P4) shows the same share as its base arms.
Since such silence is unrelated to the edit and could sway a naturalness or accent judgement, **every round-3 clip of
every arm and backbone is trimmed identically after loudness normalisation: leading silence over 0.3 s and trailing
silence over 0.5 s are cut (silence = below 1% of the clip's peak; `scripts/eval/trim_silence.py`, report written
into the kit)**. Round-2 clips were not trimmed; the difference is stated with the round-3 results. Kit and site:
`scripts/eval/round3_kit_and_site.sh` (kit profile round3: tag arms from synth_r3, base arms the round-2 base-model
clips, cell names unchanged so the registered analysis applies). Training of CosyVoice 2 and T5Gemma runs on pods
(`round3_cv2_pods.py`, `round3_t5gemma_pods.py`) with the checkpoint grids and validation reads of 08:55.

**Section 16, execution (11:10 JST).** To save time, the checkpoint rule of 08:55 is applied automatically the moment a
backbone's validation reads are complete (`scripts/eval/round3_choose.py`, choice written to `r3/<bb>/*_choice.json`),
and the stimuli are synthesized at once; the author's blind listening runs in parallel and can overrule a choice, in
which case the stimuli are re-synthesized with the overruled checkpoint and this note says so.

**Section 16, checkpoint grids widened (11:20 JST, before any of these adapters existed).** Asked by the author
whether the registered step counts might all be too small, the grids are doubled at the top: T5Gemma 200 to 1,600
steps (every 200; registered 400), CosyVoice 2 250 / 500 / 1,000 / 2,000 / 4,000 (registered 1,000); Irodori keeps 500
to 4,000 (registered 2,000; run already in progress). Same rule of 08:55 chooses among them.

**Section 16, training budgets compared in epochs (11:23 JST, before any of these adapters existed).** The registered
step counts are not comparable across backbones: in samples seen (steps x batch) over each backbone's round-3 teacher
set, Sarashina's 20,000 steps are about 12 epochs, Irodori's 4,000 about 18, CosyVoice 2's 4,000 about 2.5 (1,000
registered: 0.6) and T5Gemma's 1,600 at batch 4 about 0.4 (400 registered: 0.1). The author asked that the smaller
budgets be extended. **CosyVoice 2**: the grid gains 8,000 and 16,000 steps (about 5 and 10 epochs). **T5Gemma**: a
second training run with batch 16 and 3,200 steps (about 3 epochs, checkpoints every 400, labels b16s400 ...
b16s3200) runs in parallel with the batch-4 run; both enter the same rule. If the longer run's best checkpoint arrives
after the listening kit is built, the kit is rebuilt with it before the task is posted, and this note says so.

**Section 16, Irodori checkpoint chosen (2026-09-18 12:02 JST, `r3/irodori/irodori_choice.json`).** Teacher set
13,901 rows (section-12 carriers, 8,647 dropped as slow). Validation CER over 144 items: 500 steps 0.077, 1,000
0.064, 2,000 0.058, 4,000 0.059; redraws on the selection split 2 / 0 / 1 / 1, none exhausted. 2,000 and 4,000 tie
within 0.005; UTMOSv2 of the tag arm 2.806 against 2.730 → **2,000 steps** (the registered value). Stimuli synthesis
started 12:02. Irodori's kana baseline on the validation sets is the base arm over the kana sentences
(`round3_irodori_kana_base.sh`), as in round 2.

**Section 16, T5Gemma execution note (13:42 JST).** The batch-4 training on a 5090 pod runs at about 0.19 steps per
second (gradient checkpointing), so 1,600 steps would take 2.4 hours; the batch-16 run of 11:22 (3,200 steps) is
abandoned as infeasible today and its pod is used for validation reads. Two earlier training attempts failed before
any step (the gated base model's token was not passed, then a quoting error in the chain); about 80 minutes were lost.
The T5Gemma checkpoint for round 3 is chosen by the rule of 08:55 among the checkpoints whose validation reads are
complete when the listening kit has to be built (200-step checkpoints arrive every 18 minutes from about 13:58);
later checkpoints are read afterwards and reported, and the paper says which checkpoint was used and why.

**Section 16, correction of the epoch comparison (13:45 JST).** T5Gemma's trainer batches by a token budget (20,000
tokens per step, about 40 carriers), not by the nominal batch of 4; the round-2 log shows 400 steps = 9 epochs of the
4,529-row set, and 400 steps of the round-3 set (18,781 rows) are about 0.85 epoch, 1,600 about 3.4 epochs. The
"batch 16" run of 11:22 would not have changed the step size and is withdrawn. Today's rate (0.19 steps/s on a 5090
with activation checkpointing) equals round 2's. To finish in time, a second batch-4 run without activation
checkpointing runs on an H100 (labels hs200 ... hs1600); the 5090 run continues; both enter the same rule.

**Finding, 2026-09-18 15:02 JST — three easy-50 tags carry a wrong reading.** During the round-3 dry run the author
noticed the easy-50 tag `コモリ/カ` (item 22, 子守歌); a listing of all 50 tags shows two more: item 43 (心 in 心が温かく)
tagged シン, item 49 (夕暮れ時) tagged ユーグレ'ジ. The easy-50 targets were never dictionary-verified (round 1 verified
the difficult and katakana targets only); these tags came from the front-end and were used unchanged in rounds 1 and
2, so in the round-2 naturalness study the edited rendition of these three items asked for a wrong reading, which
counts against the edit on those items. The author reviews all 50 easy tags now (click-through UI,
`preregistration/items/easy50_accent_reviewed.json`); corrected items are re-synthesized on every backbone before the
round-3 naturalness kit is built, and the round-2 naturalness numbers are reported with this defect stated (a
sensitivity analysis excluding the three items will be added).

**Easy-50 tags reviewed (15:07 JST).** The author reviewed all 50 easy-50 tags; 10 changed
(`data/derived/easy50_r3/`, list in `changed_items.json`): three readings (子守歌 コモリ'ウタ, 心 ココ'ロ, 夕暮れ時
ユーグレドキ), one word (間 マ), six nucleus positions (一日, 音, 上, 駅前, 疲れ, 貝殻). Round 3 uses the reviewed file for
the naturalness pairs (the tag arm of the 10 items is re-synthesized on every backbone, same voices, seeds and guard);
rounds 1 and 2 used the unreviewed tags, which is stated with their results.

**Section 16, CosyVoice 2 checkpoint chosen (2026-09-18 15:10 JST, `r3/cv2/cv2_choice.json`).** Teacher set 13,751 rows
(section-12 carriers, 7,644 dropped as slow). Validation CER over 144 items: 500 steps 0.179, 1,000 0.166, 2,000 0.185,
4,000 0.186, 8,000 0.205, 16,000 0.234; no redraws anywhere. Unedited / kana: selection split 0.204 / 0.113, side-effect
set 0.078 / 0.100. **1,000 steps** (the registered value) is chosen; the provisional choice of 14:49 (same checkpoint)
already synthesized the stimuli. Note: on the side-effect set every budget's tag arm (0.139 to 0.182) is above the
unedited arm (0.078), i.e. on CosyVoice 2 the edit still damages sentences the base reads correctly, unlike Sarashina.

**Findings, 2026-09-18 15:23 JST.** (1) **CosyVoice 2 drops the final mora.** On the round-2 clips the kana-whisper
transcript lacks the sentence's last mora in 29 of 50 kana-arm clips and 24 of 50 unedited easy clips; on the round-3
tag arm in 45 of 50 difficult and 8 of 10 sampled easy clips (Sarashina 3 to 4 of 50 in every arm, Irodori 3 of 50).
The synthesis path passes the whole sentence with stream=False and nothing in the repository crops the waveform, so
this is the backbone's own early stop in this setup; it is present in both arms and stronger in the tag arm, it is
audible as a clipped ending, and it is stated with the CosyVoice 2 naturalness results. (2) **No trimming in the
round-3 kits** (author's decision 15:22): the silence-trimming rule of 10:56 is withdrawn before any kit was built
with it; clips ship as synthesized after loudness normalisation, as in round 2. (3) The T5Gemma 5090 training stopped
at step 1,000 while writing that checkpoint (no error in the log; the file is truncated); checkpoints 200, 400, 600
and 800 exist and the rule chooses among those read in time.

**Author's blind check on the round-3 stimuli sample (2026-09-18 15:35 JST, `data/analysis/v2/author_blind_r3sample_20260918.json`).**
Ten difficult-50 and ten easy-50 items per backbone (fixed seed), backbone hidden. Difficult, tag vs kana, "right =
reading and accent": tag right only / kana right only / both right / both wrong: CosyVoice 2 5 / 0 / 2 / 3, Irodori
6 / 1 / 3 / 0, Sarashina 6 / 0 / 2 / 2. Easy, tag vs no edit, more natural: edited / unedited / no difference:
CosyVoice 2 2 / 6 / 2, Irodori 1 / 0 / 9, Sarashina 0 / 5 / 5. No catastrophic failure; the accent channel works on all
three; the naturalness cost remains audible on CosyVoice 2 and, in this sample, on Sarashina's P4 stimuli, while
Irodori is a tie. These judgements choose nothing and enter no reported number.

**Section 16, T5Gemma checkpoint chosen (2026-09-18 16:07 JST, `round3_choose.py t5gemma`, `$V/r3/t5gemma/t5gemma_choice.json`).**
Validation reads complete at kit time: steps 400 and 600 (step 200 base arms and step 800 still being read; reported
when they land). Tag-arm kana CER on the 144 validation items: step 400 0.1352 (selection 0.1365, easy 0.1317), step
600 0.1353 (selection 0.1336, easy 0.1397); no redraw and no exhausted item on either. The gap is within the 0.005 tie
margin, so the UTMOSv2 tie-break applies: step 400 2.794, step 600 2.866. **Step 600 is the round-3 T5Gemma
checkpoint** (about 1.3 epochs of the filtered round-3 set at the 20,000-token batch). The stimuli for both candidates
were synthesized in parallel on the pods holding the merged models (started 15:46); the step-600 difficult-50 and
easy-50 tag arms (easy-50 with the reviewed tags of `data/derived/easy50_r3/`) go into the kits; the step-400 stimuli
are kept but not used. Steps 800 and 1,000 and later training were not available in time; this is stated with the
results. Training bundles for steps 200 to 1,000 are copied to `$V/r3/t5gemma/readlora_pth/` before the pod is cut.

**Section 16, round-3 sites released (2026-09-18, listening 16:13 JST, naturalness 16:28 JST; `round3_deploy.sh`).**
Listening: `jp-accent-listening-r3.pages.dev`, KV `worker-RATER_R3PROD`; four cells A2/B2/C2/D2 (158/105/105/105 items,
same arms, attention duplicates and catch items E as the round-2 site, byte-identical E clips and screening set), block
0 = 481 items, no hold. Verified before release: for all 473 cell items the key's source path (cell, arm, item
index), the kit copy (byte-identical to the source), the normalised clip and the served clip (byte-identical) form one
chain; rater-facing target word, reading and prescribed accent match the key and the difficult-50 text; within each
item the arms share the voice; all 501 served files fetched from the live host with matching sizes; a test
save written to and removed from the live store. Naturalness: `jp-accent-naturalness-round3.pages.dev`, KV
`worker-NAT_R3PROD`, 200 pairs (50 per backbone, all easy-50 items), 400 clips, 4 groups, fee 700 yen; the same
chain check on all 400 clips (registered items -> synth_r3 tag arm / round-2 unedited arm -> kit -> normalised ->
served), sentences equal to the reviewed easy-50 text, test save written and removed. Side assignment is the
registered deterministic hash of the pair id (as round 2); it gives A = unedited on 25/21/24/27 of 50 pairs
(Sarashina / CosyVoice 2 / Irodori / T5Gemma; binomial two-sided p >= 0.32), which is kept as the rule gives it.
T5Gemma stimuli in both sites are the step-600 checkpoint (choice of 16:07). No trimming; loudness normalisation only.

**Panel decision (16:35 JST).** Rater R4 (round-2 gate 20/20) is at 379 of 481 round-2 items (last
answer 14:47 JST today). The author asks this rater to move to round 3 and pays the round-2 work as done; the
round-2 judgements stay incomplete and, under the registered rule (complete valid raters only), do not enter the
round-2 analysis, which stands at three raters unless that rater finishes on their own. Round-2 raters who passed
the gate enter round 3 at the main task (invitation URL with `s=2`); new raters take the screening first.

**Section 16, T5Gemma step 800 read afterwards (pulled 16:15 JST, after the kit build of 16:12).** Tag-arm kana CER on
the 144 validation items: 0.1387 (selection 0.1221, easy 0.1835), no redraw. This lies within the 0.005 tie band of
steps 400 (0.1352) and 600 (0.1353): had it been read in time, the rule would have put it into the UTMOSv2 tie-break
(its UTMOSv2 is computed and recorded below). It was not available when the kit was built, so step 600 stands as the
round-3 checkpoint under the rule of 13:42; the step-800 numbers are reported with the results. The pattern is
noted, not interpreted: step 800 reads the selection split better and the easy split worse than 400 and 600.

**Section 16, step-800 UTMOSv2 (recorded 17:10 JST).** Mean UTMOSv2 over the 144 validation clips: step 800 2.832 (step 400 2.794, step 600 2.866). Under the rule this would have left step 600 chosen; the round-3 checkpoint remains step 600 as chosen at 16:07.

**Section 16, gate records carried over (17:15 JST).** The round-3 site keeps its own KV, so a round-2 rater opening a
`s=2` invitation without a gate record there is shown the 20-item screening again before the block. Rater
R2 met this at 17:00 and passed it again (19/20; the screening items are the same 20 as round 2) before
entering block 0. To spare the other three the repeat and the risk of a second, binding attempt, their round-2 gate
records (R1 19/20, R3 18/20, R4 20/20) were copied verbatim into the round-3
KV and verified through the live gate endpoint; they enter block 0 directly. The gate outcome of every round-3 rater
is the round-2 one; R2's repeat is recorded and changes nothing.

**Section 16, pods closed (17:34 JST).** All RunPod pods are terminated. On local storage: CosyVoice 2 adapters for every grid
step (250 to 16,000; chosen 1,000); T5Gemma training bundles for steps 200, 400 and 600 (model, optimizer, scheduler;
chosen 600, which loads); the step-800 bundle was not copied in full and was discarded (its validation read and
UTMOSv2 are recorded above), the step-1000 file was truncated by the training crash. The step-200 base-arm read on
the validation set was stopped unfinished (no registered number depends on it). Irodori checkpoints are local.

**Section 16, naturalness recruitment target (17:50 JST, before any result is looked at).** The registered minimum of
15 judgements per pair stands; the author sets the recruitment target at 20 valid complete sittings per group (80 in
all), so that every pair has at least 20 judgements after the registered exclusions. The task stays open until every
group reaches it; additional postings, if any, target the groups that are short. The analysis uses every valid
complete sitting, as registered; the target changes the number of raters, not the rule.

**Section 16, naturalness interim look (2026-09-18 18:23 JST, `analyze_naturalness.py --exclude-speeders`,
`data/analysis/v2/naturalness_report_round3_interim_20260918_1823.json`, export `v2/responses_nat_round3_interim_20260918_1823.json`).**
Taken at the author's request to see how many sittings the registered exclusions remove; the task stays open until
the 107 assigned sittings finish, as decided at 17:50, and no design, item, rater or stopping decision depends on
this look. 105 sittings started, about 99 complete, 5,031 judgements; two sittings excluded by the speed rule (median
interval below the shortest pair, 6.1 s), 4,964 judgements kept, 23 to 26 per pair; the duplicate-account check
waits for the platform's worker list. Unedited minus edited: Sarashina -0.008 [-0.146, +0.133] (margin met),
Irodori +0.193 [+0.074, +0.305], CosyVoice 2 +0.200 [+0.042, +0.352], T5Gemma +0.259 [+0.095, +0.417]; same-reading
stratum -0.03 / +0.21 / +0.22 / +0.19. Interim; the final run on the closed task replaces it whatever it shows.

**Section 16, listening interim look (2026-09-18 18:28 JST, two complete raters, `analyze_listening.py --round2
--interim-min-raters 2`, `data/analysis/v2/listening_report_round3_interim_20260918_1828.json`).** Taken at the
author's request; below the registered minimum of three raters, so exploratory, and no rater, item or stopping
decision depends on it. Raters R2 (catch 8/8, consistency 23/23) and R1 (7/8, 23/23); kappa
0.93 over 441 stimuli; with two raters a disagreement is a tie, 14 tied stimuli excluded as registered. Tag vs kana:
Sarashina 0.80 vs 0.57 (Holm p 0.026), CosyVoice 2 0.88 vs 0.40 (Holm p < 1e-4), Irodori 0.85 vs 0.45 (Holm p < 1e-4),
T5Gemma 0.93 vs 0.60 (Holm p 0.0016); Sarashina tag vs no edit 0.78 vs 0.63 (p 0.17). The final run on all complete
valid raters replaces it whatever it shows.

**Section 16, objective table, three backbones (2026-09-18 19:42 JST, `data/analysis/v2/reading_obj319_round3_partial3.json`,
`obj319_round3_partial3_contrasts.md`; T5Gemma's tag arm still synthesizing, to be read in the same way).** Tag arm
with the round-3 adapters (paths and hashes in `v2/logs/round3_obj319.log`), unedited, kana and UtterTune arms the
round-2 clips, one ASR pass: Sarashina 0.875 (tag - kana +0.031 [-0.016, +0.080], p 0.20), CosyVoice 2 0.784 (-0.066
[-0.114, -0.019], p 0.009; tag - UtterTune -0.072 [-0.122, -0.021]), Irodori 0.868 (-0.041 [-0.076, -0.007], p 0.019);
redraws 4 / 0 / 0 of 319, none exhausted, no inaudible output. **Check on the Sarashina caveat of section 14:** the
rate rule was chosen on the first 100 of the 319 items; on the 219 items never used for that choice the tag reads
0.904 against kana's 0.868 (first 100: 0.810 against 0.790), so the row shows no sign of an optimistic selection
effect; the caveat stays in the paper with this figure beside it.
Intervals for that split (same clustered bootstrap and exact McNemar as the table, `reading_obj319_sarashina_r3_split100_219.json`):
219 unused items, 150 components: tag 0.904 [0.858, 0.944], tag - kana +0.037 [-0.026, +0.101] (p 0.26, discordant 23
vs 15), tag - none +0.402 [+0.308, +0.491]; first 100 items: tag 0.810 [0.708, 0.900], tag - kana +0.020 [-0.048, +0.088].

**Section 16, objective table, T5Gemma (2026-09-18 20:26 JST, `reading_obj319_round3_t5g.json`, `obj319_round3_t5g_contrasts.md`).**
Step-600 model exported locally from the training bundle (hash in `v2/logs/round3_obj319.log`; missing keys 0), tag
arm synthesized 19:08 to 20:21, unedited and kana arms the round-2 clips, one ASR pass: tag 0.793 (round 2: 0.809),
tag - kana -0.113 [-0.154, -0.073] (p 1e-7), tag - none +0.357 [+0.275, +0.437]; redraws 32 of 319 (round 2: 30), 7
outputs inaudible after six takes and scored as misses (round 2: 4); voice disagreement tag 0.327. Unfavourable and
reported as it is: on T5Gemma the round-3 recipe does not improve the objective reading score. The four-backbone
single-pass read (`reading_obj319_round3.json`) follows as the table of record and is checked against these values.

**Exploratory, 2026-09-18 20:50 JST — why the objective reading score and the listening result differ
(`data/analysis/v2/asr_vs_raters_reading_r3_interim.json`).** On the 400 round-3 listening stimuli both measures exist:
the guard's kana-whisper transcript (target reading found or not) and the raters' verdict (interim, two raters; a
clip counts as "reading wrong" if either rater answered 3). Tag arms: ASR misses the target on 42 of 200 clips,
raters call the reading wrong on 3; kana arms: 35 and 1. The ASR score is therefore a conservative measure of
reading, about equally so for both arms (21% against 17.5% of clips), and the objective tag - kana gap is not read
as a gap in what a listener hears. T5Gemma's 66 objective misses concentrate on few reference voices (32 on 8 of
the 42 voices, where kana reads 0.60 to 1.00), and 7 are inaudible outputs. Threshold sensitivity quoted in the
paper: section 14's P9 (8 speech tokens per character, about 0.32 s) removed the runaway as P4 did (1 redraw in 100).
Exploratory; the counts are recomputed on the final panel.

**Note on authorship of this document (2026-09-18 21:40 JST).** This registration, including every amendment and
execution note, was drafted with a generative AI assistant (Claude, Anthropic) working under the author's direction;
the same assistant wrote and ran the experiment, evaluation and analysis code. Every design and analysis decision
recorded here was made or approved by the author (entries say "the author decided" where the decision was the
author's own initiative), and the author is responsible for the content. What fixes a decision in time is the
repository's commit history, not the prose. The paper discloses the same in its acknowledgments, as the IEEE policy
on AI-generated content requires.
