# Pre-registration — Listening evaluation of post-hoc pronunciation/accent editing in LLM-based TTS

**Status:** DRAFT — not yet timestamped.
**Author:** Shuhei Kato (KOWRO Inc.)
**Registered artifact:** this file, `build_item_lists.py`, and `items/` in the same directory.

This document fixes the design, outcome measures, and analysis of the listening evaluation
**before any listening data is collected**. There is no item selection: every item in both
source sets is evaluated, so there is no sampling decision to question. What is fixed here is
the hypothesis set, the non-inferiority margin, the analysis, the exclusion rules, and the
bounded role of the automatic accent measure.

---

## 0. Resolved design decisions (fixed before any listening data)

1. **Speakers.** Two anchor-bank references per confirmatory cell, fixed here; one of them for
   the locality cells. Speaker identity is recorded in the released materials.
2. **Backbones under human judgment.** Sarashina2.2-TTS (autoregressive) and Irodori-500M-v3
   (diffusion). CosyVoice2 and T5Gemma-TTS appear only in the objective cross-backbone reading
   table and receive no rater time — a 4-page limit cannot carry four listening tables, and the
   cross-architecture claim is answered by two paradigms.
3. **Timestamp.** This directory is committed and git-tagged before rater onboarding; an OSF
   registration may be filed in addition. The paper cites the commit hash.
4. **Editor provenance.** Every adapter evaluated here is retrained clean-room: all four
   backbones are trained on one shared, publishable text set (533 words × 10 carriers) with no
   recorded speech and no unreleased assets. Words colliding with any evaluation target by
   surface or canonical reading are dropped (50 of 533).

---

## 1. Hypotheses

Primary (confirmatory) — **Family A, superiority**. McNemar exact, paired, two-sided; Holm over
the three tests at family-wise α = 0.05 (strictest threshold 0.05/3 = 0.0167).

Every test below is measured across the 42 held-out voices, one per word-unit (§3), so each cell
spans 27–31 distinct voices. Voice generalization is therefore a property of H1/H2/H5 themselves,
not a separate replication test; the earlier H6/H7 ("replicates on a second voice") were withdrawn
for that reason (§7.1), which also removes two tests from the Holm family.

| ID | Hypothesis |
|----|------------|
| **H1** | On difficult (rare-reading) words, the edited system produces the intended accent type more often than the unedited baseline. Sarashina, 50 word-units over 27 voices. |
| **H2** | **Cross-architecture.** The same holds on a diffusion backbone (Irodori-500M-v3), i.e. H1 is not a property of one generation paradigm. 50 word-units over 28 voices. |
| **H5** | **A reading-only input cannot control accent.** Feeding the katakana reading without accent marks does not produce the specified accent type at the rate the tagged edit does. Sarashina, cell A, 50 word-units over 27 voices; stratified by prescribed type as a secondary (§7.1 rev. 2026-09-04). *This is the study's existential contrast: on the 319-item objective set kana reads at least as accurately as the tag on every backbone, so if kana also places the accent, the method has no reason to exist.* |

Reported as **estimates, not tests** — see §5 for why no non-inferiority test is pre-registered:

| ID | Statement | Reporting |
|----|-----------|-----------|
| **E1** | **Locality (control leak).** In every katakana sentence the edit lands on a **different word** than the one judged; editing elsewhere should not degrade the untouched word's accent. Sarashina, all 50 items over 30 voices. | Paired difference with 95 % bootstrap CI clustered by word. No verdict. |
| **E2** | **What zero recorded data costs.** On CosyVoice2, the data-free editor vs the released UtterTune adapter, trained on 15,097 hand-corrected recorded pairs. | Paired difference with 95 % bootstrap CI clustered by word. No verdict. |

Secondary (descriptive; no confirmatory claim):

| ID | Statement | Reporting |
|----|-----------|-----------|
| S3 | Inter-rater reliability | Fleiss' κ with 95% bootstrap CI, over **all** stimuli |
| S4 | Convergent validity between the human consensus label and the automatic accent measure (§5.3), on the human-judged stimuli. | AUROC of PASQA scores separating human-correct from human-incorrect, with 95% bootstrap CI. **No threshold is fitted to the human labels**, so no accuracy figure is reported. |

A comparison against PronSteering (S1 in an earlier draft) is not pre-registered and no stimuli
for it exist in the kit; it is not reported.

Directional expectations are stated so that a null or reversed result is reportable as such.

## 2. Materials

Item sets (fixed, external to this study, unchanged since creation):

| Set | Source file | Items | Used for |
|-----|-------------|-------|----------|
| difficult | `test_difficult_50_vanilla.txt` (+ `_with_accent`) | **50 (all)** | H1, H2, H5, H6, H7, E2 |
| katakana | `test_katakana_50.txt` (+ `_with_accent`, edit input `_phon`) | **50 (all)** | E1 |

**Edit site in the katakana set — and a repaired materials defect.**
`test_katakana_50_phon.txt` places the `<PHON_*>` edit on a **non-target** word (e.g. in the
カメラ sentence the edit lands on 撮影 → サツエー); the katakana loanword named in `_with_accent`
is left untouched and is what raters judge. That is what makes the cell a control-leak test
rather than a second editing-accuracy test.

In **8 of the 50** sentences the released file tagged the judged word itself (パイプオルガン,
ブランコ, シンフォニー, テレフォン, ホテル, ランプ, キャビン, ドミノ) — a defect of the
test-set generator, which failed to exclude the target when choosing an edit site. One of them
(ブランコ) additionally specified ブ'ランコ while the judged target is ブラ＼ンコ.

**These items are repaired, not dropped.** For each, the edit site was regenerated onto a
non-target kanji noun of the same sentence, with its accented reading taken from `pyopenjtalk`
exactly as the other 42 were built (教会, 公園, 劇場, 大事, 穏やか, 静か, 山小屋, 静か), and the
affected arm was re-synthesized. The cell therefore keeps **all 50 items** and every item now
tests the same thing. The original and repaired lines are both released
(`preregistration/items/katakana_regenerated_8.json`,
`data/derived/test_katakana_50_phon_fixed.txt`); the repair was made on the materials alone,
before any audio was judged, and no item was removed on any ground.

**No sampling is performed.** Every item in both sets is evaluated, under every condition, on
every backbone. There is therefore no seed, no subset, and no selection rule that could have
been chosen after seeing results. `build_item_lists.py` emits the lists verbatim and records
the SHA256 of each source file in `items/provenance.json`.

**Target accent types.** Each item carries its intended accent type in the `_with_accent`
source (target word wrapped in `<b>…</b>`, `＼` = downstep, `￣` = flat). These targets are
verified by the author **before any audio is presented to raters**, against the following
**fixed source hierarchy**, which is applied per item and recorded per item:

1. *NHK Japanese Pronunciation and Accent Dictionary* (new edition) — primary authority;
2. if the word has **no headword** in (1): OJAD, then 大辞林, then the NHK dictionary's own
   appendix rules for compounds — whichever first yields a single normative type;
3. if the word has no headword in (1)–(2) at all, the target is the **author's
   native-speaker default reading**, fixed **before any audio is heard** and recorded as such
   per item;
4. if a source lists the word but yields **no single** normative type (multiple accepted
   variants), the item is **excluded**.

**Why a target may be author-assigned without compromising the claim.** The hypothesis tested
is conformance to the *intended* accent (H1, §4: raters are shown the intended type and never
asked to recall the norm), not conformance to the dictionary. The target is applied
**identically to both arms**, so an author-assigned target cannot favour the edited system: it
is the No-edit baseline that benefits, because a natural default reading is also what the
unedited model's lexical prior tends to produce. Dictionary grounding is required not for the
edited arm but to keep the **No-edit baseline a meaningful control** — an arbitrary target
would be unguessable without instruction and would make the comparison vacuous. A
native-speaker default reading satisfies that requirement. Tier (3) is used for 4 of the 100
items (all in the difficult set); each is listed in
`preregistration/items/target_verification.json`, and §5.4 pre-registers a sensitivity
analysis excluding them.

The difficult set is by design a rare-vocabulary set, so absence from (1) is expected for
part of it; falling back to (2) rather than excluding avoids an exclusion rule that would
correlate with word rarity and would preferentially remove exactly the items the study is
about. **The source actually used is recorded for every item**, and the paper reports the
per-source counts and the number excluded. Every verification decision is made on the
dictionaries alone, **never on the audio**. Verification is performed in a purpose-built
review UI (`scripts/author_review/`), which writes one record per item — verdict, corrected
accent string where applicable, and source — to `data/author_verification/`; these records
are released with the code so the annotation is auditable.

## 3. Stimuli

Backbones: **Sarashina2.2-TTS**, **Irodori-TTS-500M-v3**, **CosyVoice2**. Reference voices are
the **42 held-out refs42 speakers** (ReazonSpeech; ERes2Net→UMAP→k-means(42) nearest centroid;
filtered on SNR, pause spectral flatness, transcript CER and zero fraction; trimmed;
`eval_speakers/selected_42.json`). Each **word-unit** — every arm of one word in one cell — is
synthesized under speaker(s) fixed by a hash **before any audio was selected**:
`spk(cell, word, k) = sha256("SPK:"+cell+":"+word+":"+k) mod 42`, k=1 re-hashed until it differs
from k=0 (`items/speaker_assignment42.json`, `scripts/eval/assign_speakers42.py`). Cells A, C and D
use two speakers per word (k = 0, 1); cell B uses one. Speaker is therefore nested in word-unit,
the paired comparisons stay within a speaker, and the word-clustered bootstrap (§5) already
accounts for it. The 42 voices replace the earlier two-voice design (§7.1); no voice is chosen by
ear and none appears in any training or teacher-generation stage of any backbone. The released
UtterTune adapter was trained on JSUT+JVS, so JVS is excluded as a reference everywhere.

| # | Cell | Composition | Stimuli |
|---|------|-------------|---------|
| 1 | A — control + reading-only baseline (AR) | Sarashina difficult × {No-edit, **kana input**, Ours} × 50 word-units | 150 |
| 2 | B — data-free vs recorded-data (AR) | CosyVoice2 difficult × {**UtterTune released**, Ours} × 50 | 100 |
| 3 | C — cross-architecture (diffusion) | Irodori difficult × {No-edit, Ours} × 50 | 100 |
| 4 | D — locality (AR) | Sarashina katakana × {No-edit, Ours} × 50 | 100 |
| 5 | Attention checks | duplicate-consistency items, ≈5% (§4) | 23 |
| | **Total** | | **473** |

Each word-unit is synthesized under the single voice its hash assigns (§3), so cells A–D span 27,
31, 28 and 30 of the 42 held-out references respectively. The 473 judgements are split into **5
blocks of 95–98** (§4) — 99–102 with the four known-answer catch trials each block carries — about
25 min each; with the 20-item gate a first session fits in half an hour. Each stimulus is judged by
five raters: 25 block-assignments, about 10.5 h of judging in total (JPY ~52,000 at JPY 5,000/h)
plus the gate for every candidate screened, admitted or not.

An Irodori katakana locality cell (200 stimuli) was specified in an earlier draft and removed
before any listening data was collected: it repeated cell D's question on a second backbone and
detected nothing D does not. The per-cell second speaker was removed for the reason given in
§3.1 — with one voice per word-unit drawn from 42, a duplicated second rendition adds no
independent cluster to a word-clustered analysis.

### 3.1 Why 42 voices and one per word

An earlier draft used two hand-checked voices and spent three cells replicating the result on the
second one. That is n=1 for a generalization claim, and it made two of five confirmatory tests
answer a question a larger voice sample answers for free. Drawing one voice per word-unit from the
42 filtered held-out references (§3) spreads each cell over 27–31 voices at the same rater cost per
stimulus, and the paired comparison still lives inside one voice because a word-unit keeps all its
arms on its assigned voice. Two voices per word was considered and rejected: the analysis clusters
by word, so a second voice adds no independent cluster while doubling rater cost
(450 stimuli vs 800).

This also removes a selection worry the two-voice design carried: which two voices were chosen
mattered, and objectively the two ITA references disagree (on the 319-item set the tag is 0.11
below plain kana under one and 0.03 below under the other). With 42 voices assigned by hash before
any audio was selected, no result rests on that choice.

## 4. Listening evaluation procedure

- **Raters:** five external experts in standard-Japanese accent norms (former announcers,
  narrators, or teachers with accent-instruction experience), with no interest in the
  outcome. Target 5, minimum 3.
- **Qualification screening:** every candidate rater — regardless of how they were recruited
  or whether they are known to the author — takes the same screening test before admission:
  known-answer accent judgments, in the same three-choice format and the same ＼/￣ notation as
  the study itself. Screening items are disjoint from the evaluation stimuli. Candidates who do
  not pass are compensated for the screening and not admitted; the number screened and the number
  admitted are reported. Self-declared credentials are not a substitute for the test.

  *Amendment, 2026-09-05 (before any rater was admitted and before any listening data existed).*
  The gate is **20 items** at **19/20**, and each block additionally carries **4 catch trials**.

  *Why split it.* A gate can only test ability at the moment of admission; it cannot tell whether a
  rater kept listening for the next hundred judgements. A short gate plus ongoing checks tests both,
  and the checks cost four judgements per block rather than a second sitting.

  *Gate (20 items).* 5 items in each of 頭高 / 中高 / 平板 / 尾高, gold balanced 10/10, in the same
  three-choice format and ＼/￣ notation as the study. 19/20 leaves one item of slack for a
  mishandled keypress — the author slipped on 2 of 36 in the dry runs — without admitting a
  candidate who cannot hear a contrast.

  *Catch trials (4 per block).* Known-answer items from the same verified pool as the gate. The
  blocks already carry 23 duplicate-consistency items of their own, so within-rater consistency is
  measured there and is not duplicated here; what those duplicates cannot do is separate a careful
  rater from a consistent guesser, which is what a known answer tests. They ship as an ordinary cell with ordinary blind ids and are placed at per-block
  jittered positions, never first and never adjacent, with each repeat kept well away from the item
  it repeats; nothing in the page or its network traffic marks a judgement as a check. A rater scoring
  **below 3/4** on these is reported and excluded from the primary analysis; the block's own
  duplicate pairs are reported as a within-rater agreement rate and are not an exclusion criterion.

  *Coverage.* The original 20 items test the position of a word-internal downstep only; their own
  note records that heiban/odaka contrasts were excluded because the contrast is audible only on
  the following particle. The evaluation set is predominantly heiban, so that pool did not test the
  distinction the study most depends on; heiban/odaka items were added, and gate and catch pool
  both span all four accent types.

  *How the heiban/odaka items are built.* Not by requesting an accent. A first attempt asked the
  adapter to render the same word once as heiban and once as odaka; that cannot work, and an author
  dry run confirmed it did not — both members of all eight pairs were heard as the same type. The
  contrast is realised on the following particle, outside the tagged span, so the control signal
  cannot express it, and every word chosen happened to be odaka in the dictionary anyway. The items
  that shipped instead have the base model read plain text, so each word comes out in its dictionary
  type (pyopenjtalk: `acc==0` heiban, `acc==mora_size` odaka); the difficulty lies entirely in what
  the rater is **shown** — that type, or the opposite member of the contrast.

  *Item validity.* An item is usable only if the audio carries the accent the item claims, which
  does not follow from the synthesis request. Every clip was ASR-verified for its reading and then
  judged by ear by the author across two dry runs. Where a verdict disagreed with an item's gold,
  the clip was **re-listened to before anything was discarded**: a disagreement can mean bad audio
  or a slipped answer, and the answer file cannot tell them apart. Of 36 candidates, 5 were dropped
  — four where the base did not render the dictionary accent (三時, 一人, 汁物, 二回) and one whose
  reading itself was wrong (茄子, rendered ナス) — and 2 (手紙, 魚) were reinstated when the
  re-listen found the audio correct after all. The surviving 31 items form the gate (20) and the
  catch pool (11). The author's answers, the drop list and the reinstatements are released with the
  item files (`screening_dryrun_author.json`, `screening_dropped.json`).

  Grading is done **on the server** — a Cloudflare Pages Function on the same origin as the study
  site — and not in the browser: the answer key is never sent to the client, so a candidate cannot
  read it out of the page. Judgements are stored server-side as they are made, with the browser's
  local copy kept only as the rater's own safety net and unsent judgements retried. If the grading
  request fails the session is allowed to continue and is recorded as `unverified`; every response
  is re-graded offline before payment and before entering the analysis, and `unverified` sessions
  that fail that grading are excluded there. This fail-open behaviour is a payment-fairness
  measure, not a loosening of the criterion.
- **All raters judge all 787 stimuli.** Judgments are not divided among raters. This is what
  makes κ computable over the entire stimulus set rather than over an overlap subset.
- **Task: three-way, collapsed to binary for the primary outcome.** The rater is shown the
  **intended accent type** and answers: **(1)** the target word is heard and its accent matches;
  **(2)** the target word is heard but the accent differs; **(3)** the target word is not heard at
  all (a different reading). A fourth option, "cannot judge", is available and expected to be rare.
  The **primary outcome is binary: (1) versus (2 or 3)** — the tag specifies a reading *and* an
  accent, so an utterance that says a different word has not realized the specification. This
  keeps the preregistered McNemar analysis unchanged.

  Response (3) exists because the difficult set forces it: on rare kanji the unedited base often
  produces a different word entirely (蜿蜒 → "ホモット"), and asking whether *that* realizes
  エンエン￣ is ill-posed. Without the category, raters split on interpretation and κ falls.

  **Pre-specified secondary analyses.** (a) The (2) vs (3) split per arm, which separates
  accent errors from reading errors and shows how much of the No-edit deficit is reading. (b) An
  **accent-only re-analysis** restricted to items where *neither* arm was scored (3): on that
  subset the comparison is purely about accent, which answers directly whether the accent gain is
  a reading gain in disguise. Both are fixed here, before any data. The rater is *not* asked to recall or look up the normative accent;
  normativity is established beforehand (§2). This keeps the task perceptual and keeps the
  per-judgment time under ~10 s.
- **Blinding:** condition labels are never shown; presentation order is randomized
  independently per rater. **The stimulus is the whole sentence** (§3): no cutting and no
  forced alignment, so no alignment error can be mistaken for an accent error, and the rater
  hears the target word in the prosodic context that determines its realization. The rater is
  told which word to judge.
- **Consensus label:** majority of the raters who judged the item. With 5 raters, ties are
  impossible. If fewer than 5 raters complete and an even number remains, tied items are
  excluded from the primary analysis and their count reported.
- **Attention checks: duplicate consistency, ≈5%.** A stimulus is re-presented under a second
  blind ID; a session is scored on how often it answers the pair the same way. Threshold 80 %;
  a session below it is re-run, and if not re-run its judgments are excluded and the exclusion
  reported. This is what the kit implements (37 pairs). An earlier draft said "known-correct
  items"; that would require the author's own ear to define the correct answer for a synthesized
  clip, and although a gate is not a reported number, keeping the author out of the loop entirely
  is cleaner. Consistency is checkable without anyone knowing the right answer.
- **No author judgments enter any reported number.** The author's role is limited to
  verifying target accent types against the dictionary (§2), which is annotation against an
  external norm, not evaluation of the systems.

## 5. Statistical analysis

- **Primary outcome:** binary accent correctness of the consensus label.
- **Superiority tests (Family A: H1, H2, H5, H6, H7):** McNemar exact, paired, two-sided.
  **Multiplicity:** Holm within the family at family-wise α = 0.05; strictest threshold
  0.05/5 = 0.0100.
- **No non-inferiority test is pre-registered.** An earlier draft tested locality and the
  data-free comparison for non-inferiority at Δ = 0.15. That is withdrawn, before any data, for a
  reason that is a property of the design and not of the result: every one of those comparisons
  rests on **50 distinct words**, and at 50 words with the discordance an accent judgement
  realistically produces (20–25 %), the 95 % CI half-width on the risk difference is ≈ 0.12 — the
  same size as the margin. Simulation puts the probability of concluding non-inferiority at
  0.58–0.68 *when the two arms are genuinely identical* (`power_analysis.py`). A test that fails a
  third to a half of the time on a true null does not answer the question; reporting it as a test
  would present a resolution limit as a finding. **E1 and E2 are therefore reported as a point
  estimate with a 95 % CI and no verdict**, and a reader who wants to apply a margin can apply
  their own to the interval.
- **Bootstrap unit is the WORD, not the row.** Cells with two speakers present the same 50 words
  twice; resampling rows independently treats two renditions of one word as two independent
  observations and yields a CI roughly √2 too narrow. All intervals are cluster-bootstrapped by
  word, 10,000 resamples.
- **Effect sizes:** risk difference and odds ratio, both with 95% word-clustered bootstrap CI.
- **Secondary automatic measure:** PASQA, in the bounded role fixed in §5.3. It is never a
  confirmatory outcome.
- **Reliability:** Fleiss' κ over all stimuli, with 95% bootstrap CI. Pairwise Cohen's κ
  reported as a matrix. Human–human agreement is reported **before** any system comparison,
  because it bounds what the primary outcome can resolve.
- **Crowd naturalness MOS:** analyzed with a cumulative-link mixed model (logit) with random
  intercepts for listener and sentence, respecting the ordinal scale and repeated measures.
  Fixed effect of system reported as log-odds with 95% interval. Raw mean MOS is reported
  descriptively only.
- **No exclusions on automatic metrics.** Every synthesized sample enters every automatic
  metric; nothing is filtered. Instead of excluding unintelligible outliers, the number of
  samples with CER > 0.5 is reported **as a metric** (catastrophic-failure count, per arm and
  per test set), alongside means computed over all samples. Human judgments are never
  excluded except by the attention-check rule (§4).
- **The analysis is code, fixed in advance.** `scripts/analysis/analyze_listening.py` implements
  everything above and was committed before any rater judgement existed; it is dry-run on
  synthetic responses generated from the real stimulus keys so that every branch is exercised.

### 5.1 Power and its limits (stated in advance)

**This section was rewritten on 2026-09-01, before any listening data, because the original
derivation was wrong in an optimistic direction.** It coupled the two arms from a single uniform
draw (`a = u < p0`, `b = u < p1`), which forces every discordant pair to fall the same way and
makes power look like 1.00 at n = 50. Real paired data has discordance in both directions: a word
the base happens to get right and the edited system gets wrong. McNemar depends only on the two
discordant counts, so the honest parameter is the **total discordance q**, of which q = |effect|
is the perfectly-coupled floor and anything above it is symmetric noise.

**Family A superiority, n = 50 words**, P(reject) at the strictest Holm threshold
(0.0167 for 3 tests / **0.0100 for the 5 tests actually registered**):

| accent effect | q = d | q = d+10% | q = d+20% | q = d+30% |
|---|---|---|---|---|
| +0.18 | .82 / .70 | .44 / .34 | .29 / .23 | .22 / .17 |
| +0.25 | .98 / .96 | .71 / .63 | .55 / .46 | .45 / .36 |
| **+0.35** | 1.0 / 1.0 | .93 / .90 | **.83 / .77** | .75 / .67 |
| **+0.45** | 1.0 / 1.0 | .99 / .99 | **.96 / .94** | .92 / .88 |

Read in advance, so the reading cannot be chosen after the fact:

* **The design is adequate iff the accent effect is ≳ 0.35.** That is what the method's own
  premise predicts: an unedited base has no channel through which to place an accent nucleus, so
  it matches the prescribed type only by chance. A large effect is the *expected* case, and the
  reading-accuracy effects (+0.14 to +0.38) are a floor, not a forecast — they measure whether the
  right morae appear, which the base often manages without any accent control.
* **If the observed effect is nearer 0.18–0.25, no n available to this study resolves it.** The
  result is then reported as unresolved, with the estimate and CI, and explicitly not as evidence
  of absence. No items will be added to convert a non-significant result into a significant one.
* **Growing Family A from 3 tests to 5** (H6, H7 — the speaker replications) costs ≈ 0.06 of power
  at the expected effect size. The replication is not traded against the main result.

**Why there is no non-inferiority test.** See §5: at 50 words the CI half-width (≈ 0.12) is the
same size as the Δ = 0.15 an earlier draft registered, and the test concludes non-inferiority only
0.58–0.68 of the time when the arms are genuinely identical. E1 and E2 are reported as estimates.

### 5.4 Sensitivity analysis for author-assigned targets

`preregistration/items/target_verification.json` records the author's pre-audio check of every
target against the §2 source hierarchy. **Seven difficult-set items** (`difficult:024, 026, 028,
031, 041, 042, 049`) and **one katakana item** (`katakana:006`) have no dictionary headword and
rest on the author's own judgement or on analogy from a listed variant spelling. One further item
in each set was *corrected* against the dictionary (`difficult:011`, `katakana:031`); corrections
are applied downstream of that file, so the original and the corrected target are both auditable
and the corrected target is the one synthesized and judged.

**Every Family A test is therefore reported twice**: on all 50 items (primary) and on the items
whose target comes from a dictionary headword (sensitivity, 43 for the difficult set). Both are
reported regardless of whether they agree; if they disagree, the dictionary-only result is the one
interpreted. This is fixed here, before any listening data exists, so neither version can be
selected after seeing the outcome. The count is read from the released JSON at analysis time
rather than hard-coded, and the analysis aborts if the file's own totals disagree with its item
list.

### 5.2 Pre-set non-inferiority margins (naturalness and similarity)

Non-degradation on non-target material is tested on objective measures over the full item
sets, including the "easy" set which receives no listening judgment, with margins fixed here:

| Measure | Margin Δ | Decision |
|---------|----------|----------|
| UTMOSv2 | 0.20 | non-inferior if the lower bound of the 95% bootstrap CI of the difference exceeds −Δ |
| ERes2Net | 0.01 | same |
| CER | 0.01 | same |

**MOS predictors are not used as evidence about accent.** UTMOSv2 here covers overall
naturalness only. General-purpose MOS predictors are empirically insensitive to localized
pitch-accent errors: against human accent ratings, Kawamura et al. (INTERSPEECH 2026,
arXiv:2606.20137) report SRCC −0.012 (UTMOS), −0.028 (NISQA) and −0.074 (DNSMOS P.835),
and Takagi et al. (arXiv:2606.19951) report human MOS falling 4.00 → 2.16 under
accent-phrase substitution while no predictor moves by more than 0.1 point. Accordingly no
MOS predictor is reported as an accent measure anywhere in this study.

### 5.5 Naturalness evaluation (crowdsourced, CrowdWorks) — fixed here in full

Evaluated **separately from the expert accent task** and by different people: the expert budget
buys accent judgements, which no automatic measure supplies; naturalness is what naive listeners
can judge.

**The design follows the methodological critique of MOS, not the habit.** Kirkland et al.
(SSW 2023, DOI 10.21437/SSW.2023-7) show that MOS values depend on the question wording, on
whether "naturalness" or "quality" is asked, and on the scale and instructions — none of which are
usually reported — so absolute MOS is not comparable across studies. The SSW 2025 community paper
(Bailly et al., DOI 10.21437/SSW.2025-1) argues that evaluation should be **task-oriented**: the
measure should follow the claim being made. Our claim is narrow and comparative — *applying the
edit does not make the audio worse* — so the primary instrument is a **same-sentence paired
comparison**, not an absolute rating.

**Primary: same-sentence A/B preference (non-degradation).**
The listener hears the **same sentence** synthesized twice — No-edit and Ours, order randomised
per trial — and answers: **A is more natural / B is more natural / no difference**. This holds
text, speaker and content constant, so the comparison is within-item and immune to the scale-usage
and question-framing effects that make absolute MOS unstable.
Outcome: the proportion preferring No-edit. Non-inferiority is concluded if the upper bound of the
95 % bootstrap CI of (P(prefer No-edit) − P(prefer Ours)) is **below 0.15**, ties split evenly.
Analysed with a mixed-effects model (random intercepts for listener and sentence).

**Secondary: 5-point naturalness rating, reported the way the critique asks for.**
Retained so the work is legible to readers who expect it, and reported with everything Kirkland
et al. found missing: the **exact Japanese question text**, the **scale labels and increments**,
the **instructions**, the **recruitment platform and screening**, and **CIs on every mean**.
Question, fixed here verbatim:

> この音声の**自然さ**を5段階で評価してください。（1 = 非常に不自然、2 = やや不自然、3 = どちらともいえない、4 = やや自然、5 = 非常に自然）
> 内容や声の好みではなく、**話し方の自然さ**についてお答えください。

All conditions of a given sentence are rated **within the same session by the same listener**, so
the ratings are comparable within this study. **We do not compare these values to MOS reported in
other papers, and say so in the paper.**

**Stimuli (300, one speaker each, full sentences).**

| set | arms | items | pairs / stimuli |
|---|---|---|---|
| Sarashina difficult | No-edit, Ours | 50 | 50 pairs |
| Sarashina **easy** | No-edit, Ours | 50 | 50 pairs |
| Irodori difficult | No-edit, Ours | 50 | 50 pairs |

The **easy** set carries the most weight: difficult sentences are lexically odd whatever the system
does, so an ordinary sentence is where a naturalness cost would actually show. It receives no
accent judgement (§6) and exists for this.

**Procedure follows ITU-T P.808**, the recommendation for crowdsourced speech-quality testing,
whose session structure is qualification → setup/environment check → training → rating:
- **Headphone check — recorded, not gating.** A *dichotic* count test: beep groups alternate
  strictly between the left and right channel, so the sequence is recoverable only over stereo
  headphones. (A single-channel pitch comparison is not used: laptop speakers pass it, so it does
  not test for headphones at all.) The result is stored as a **covariate**; it does not end the
  task and does not affect payment. The rationale is design-specific: in a *same-sentence* A/B
  comparison the playback device is constant within a trial, so a poor device adds noise but
  cannot bias the comparison — screening would mainly cost us listeners who are in fact wearing
  headphones. The pass rate is reported, and the primary analysis is repeated on the passing
  subset as a sensitivity check.
- **Training.** Before rating, the listener hears anchors spanning the quality range (clean
  stimuli and grossly degraded counterparts) so the scale is used consistently. No ratings are
  collected during training.
- **Rating.** The A/B trials, then the 5-point subset.

**Collection.** Each worker completes **50 A/B trials** (≈12 min) drawn across the three sets, then
rates a **25-stimulus subset** on the 5-point scale. Each pair receives **≥ 15 independent
judgements**; ≈ 90 workers, ≈ 4,500 preference judgements. Served from the same static site as the
expert kits, with a per-worker completion code pasted back into CrowdWorks for approval.
Compensation above the platform's standard rate for the time involved.

**Two separate gates, fixed in advance: payment and analysis.** Conflating them is what makes
crowdsourced studies unfair, so they are stated apart. Payment is for effort actually expended;
exclusion is about whether the data can be used. **A worker excluded from the analysis is still
paid**, because the work was done.

*Payment gate (approve / do not approve on the platform).* Every criterion below is stated in the
task description **before** the worker starts, so non-approval can never be a surprise.
- **Approve** when the submitted completion code matches a recorded session in which at least
  **95 %** of the trials were answered.
- **Approve** a partial session (50–95 % answered): partial completion is usually a technical
  problem, not bad faith. The data are excluded from the analysis; the payment is not.
- **Do not approve** only when the work was not performed: the completion code matches no
  session; fewer than **50 %** of trials answered; or the median time per trial is **below the
  combined duration of the two clips**, i.e. both could not have been heard.
- **The headphone check never affects payment.** It is a covariate, and the task description
  says so explicitly.

*Analysis-exclusion gate (paid, but data not used).*
- **Attention pairs**: 3 trials pit a deliberately degraded stimulus (band-limited and noised,
  grossly audible) against its clean counterpart, at fixed positions so every worker meets all
  three. Preferring the degraded one in ≥ 2 of 3 → excluded.
- **Straight-lining**: the identical response on ≥ 90 % of trials → excluded.
- **Speed**: median time per trial below the combined clip duration → excluded (and, per above,
  also not approved: this is the one criterion that appears in both gates, because it indicates
  the task was not performed at all).

**Re-collection after exclusion.** Following P.808, excluding a rating does not simply reduce the
count: the affected items are **re-posted until every pair again carries ≥ 15 valid judgements**,
so exclusions cannot leave some stimuli measured more thinly than others. Re-collection is
mechanical (it targets whichever pairs fall below the threshold) and never depends on which way
the existing judgements went; the number of re-collection rounds is reported.

Numbers recruited, approved, not approved, and excluded from analysis --- with the reason for each
--- are reported.

**Pre-commitment.** No naturalness analysis absent from this list will be added, and a favourable
accent result will not license dropping an unfavourable naturalness result.

### 5.3 Automatic accent measure (PASQA) — secondary, full coverage

PASQA (Kawamura, Shirahata, Mitsui, Shimizu; INTERSPEECH 2026; arXiv:2606.20137; code CC0,
weights Apache-2.0) predicts a pitch-accent-focused quality score that its authors validate
against human accent ratings (SRCC 0.828, LCC 0.814; 15 native raters, 120 samples).

**Role, and the limit that defines it.** PASQA scores accent naturalness **from audio alone,
with no access to the specified target accent type.** It therefore cannot establish
conformance to a specification, which is this study's primary claim. It is used as a
secondary, full-coverage measure and is never a substitute for the human primary outcome.

One consequence is stated in advance so that neither direction of result is over-read: where
a **non-standard** accent type is specified, an output that correctly follows the
specification is expected to receive a *lower* PASQA score, because PASQA treats departure
from the dictionary-normal accent as degradation. On the item sets used here the specified
type is the normative one, so this inversion should not bite; it does bound any extension of
this measure to two-valued control or to minimal pairs.

**Coverage.** PASQA is run over every synthesized utterance — all 50 items in each set, every
arm, every backbone, all four speakers — not only the 683 stimuli presented to raters.

**Pre-specified analyses.**

1. Mean PASQA per arm with 95% bootstrap CI, reported beside the human results.
2. Convergent validity (S4): AUROC of PASQA against the human consensus label, bootstrap CI.
3. Sequential-editing block: mean PASQA per edit count per method, with bootstrap CI. The one
   pre-specified comparison is Ours vs each baseline at the largest edit count, Holm-corrected
   within that family. This adds an accent axis to a block that otherwise reports only CER.

**Declared limits.**

- **Domain.** Neither backbone is in PASQA's training or validation distribution. Its authors
  report out-of-domain pairwise accuracy 0.780 against a human ceiling of 0.984, and name OOD
  robustness as future work.
- **Independence.** PASQA was trained on accent perturbations generated by its authors' own
  Japanese TTS system, from the same commercial Japanese LLM-TTS lineage as one of the
  backbones evaluated here. It is not a fully independent instrument and is not treated as one.
- **Version.** The exact commit and weights revision used are recorded in the released code.
  The upstream repository self-describes as temporarily public, so the artifact is pinned.

**Pre-commitment.** If the human primary result is unfavourable, PASQA will not be promoted to
the primary outcome, and no PASQA-based analysis absent from this list will be added.

## 6. What is deliberately not measured

Stated so that absence is not mistaken for omission:

- **Sequential-editing block (1→10→100→1000):** no listening judgment is collected there.
  It reports CER and the automatic accent measure (§5.3) only.
- **"Easy" (common-word) set:** objective non-inferiority only (§5.2); no listening judgment.
- **Homophone minimal pairs:** removed from the paper entirely.
- **Naturalness:** crowdsourced MOS (naive listeners) plus UTMOSv2; not part of the expert
  binary task above.
- **No automatic measure is a primary outcome anywhere.** Every confirmatory hypothesis
  (H1–H7, E1–E2) rests on human judgment. PASQA appears solely in the secondary role fixed in §5.3.

## 7. Deviations

Any deviation from this document — in materials, procedure, or analysis — will be reported
explicitly in the paper, with the reason and with both the pre-registered and the actual
analysis where they differ.

### 7.1 Revisions made before registration (2026-09-01)

This document was revised before it was tagged and before any rater judgement existed, so these
are not deviations. They are recorded because each one made the design weaker on paper, and a
reader should be able to see that they were made in that direction rather than the other.

1. **The power derivation was wrong in an optimistic direction** and is replaced (§5.1). The
   original coupled both arms from one uniform draw, which cannot produce discordance in both
   directions and reported power 1.00 at n = 50.
2. **Both non-inferiority tests are withdrawn** and become estimates with CIs (§5). At 50 words
   the CI half-width equals the registered margin, so the test concludes non-inferiority only
   0.58–0.68 of the time when the arms are truly identical.
3. **Bootstrap intervals now cluster by word**, not by row. Two speakers of the same 50 words are
   not 100 independent observations.
4. **Speaker allocation moved** from the locality cells to the headline cells, and two speaker
   replications (H6, H7) were added as confirmatory tests. Justified by measurement, not taste:
   cross-voice discordance under the edit is 0.267 vs 0.137 unedited (§3.1).
5. **The Irodori locality cell was removed** — it repeated the Sarashina locality question on a
   second backbone. Total judgements per rater: 788 → 787.
6. **Attention checks are described as what the kit implements**, duplicate-consistency pairs
   rather than known-answer items (§4).
7. **The PronSteering comparison (S1) is dropped**; no stimuli for it exist in the kit.

**Revision 2026-09-03 (voices).** The two hand-checked reference voices were replaced by the 42
filtered held-out references, one assigned per word-unit by hash before any audio was selected
(§3, §3.1). Consequences, all recorded before any listening data exists:
* **H6 and H7 withdrawn.** They tested replication on a second voice; with 27–31 voices per cell
  that property is inside H1/H2. Family A is now three tests, so Holm's strictest threshold moves
  from 0.0100 to 0.0167 — a change that FAVOURS the remaining tests, disclosed for that reason.
* Stimuli fall 787 → 472, blocks 10 → 6, rater cost by roughly 40%. No item and no arm was
  dropped; only the duplicated second-voice renditions are gone.
* The design no longer rests on which voices were picked. That mattered: on the 319-item objective
  set the tag reads 0.11 below plain kana under one ITA reference and 0.03 below under the other.

**Revision 2026-09-04 (H5 stratification).** Recorded before any listening data exists. Inspecting
cell A showed that 34 of its 50 prescribed accents are heiban (flat). Whether the backbone realizes
the prescribed type on such words *without* any control — i.e. whether a plain-kana input already
lands on heiban — is an empirical question this study is the instrument for, so it is **not**
assumed either way. Changes:
* **H5 stays on cell A** (difficult-50 as constituted: 34 heiban, 16 nucleus-bearing). This set was
  generated, not sampled, and **no claim is made that its accent composition represents rare words
  in general** — in either direction. A nucleus-balanced replacement cell was drafted
  (`items/aprime_candidates.tsv`) and **rejected**: restricting the population to the stratum where
  the tag is expected to win would tune the test set toward the hypothesis. The candidate list is
  kept only as a record of what was considered.
* **Stratified secondary on cell A, fixed now:** the kana-vs-Ours contrast and each arm's
  prescribed-type hit rate are reported separately for heiban (34) and nucleus-bearing (16) words.
  This is what answers whether a reading-only input already places heiban, and it is reported as a
  finding, not designed around. No confirmatory claim is attached to either stratum.
* **Power:** if a reading-only input already lands on heiban, up to 34 items are ties and the primary
  H5 has less power than n=50 suggests; the paired CI is reported and interpreted as such. Whether
  that happens is measured (the stratified secondary), not assumed. If the CI is wide, the remedy is
  more items chosen by a criterion that does not look at accent, not a favourable subset.
* **Cell C voices that collapse the edit** (3 of the 42 references drove Irodori's objective gain to
  or below zero on difficult-50): no exclusion. Every voice the hash assigned is kept and results are
  reported per voice; excluding them would select the reference on the outcome.

**Dry run 2026-09-04 (procedure only; no reported number).** The author completed all 473
judgements of the 42-voice kit under a separate rater id and output directory
(`rater_responses_DRYRUN/`, never merged with rater data). Outcomes used, and the only ones used:
* **Pace:** median 4.0 s and p90 9.0 s per judgement (expert, task known). The 15 s/judgement
  assumption behind 5 blocks × 95 is therefore conservative by a wide margin; the block design is
  kept unchanged, with the headroom noted.
* **Attention gate calibration:** 23/23 duplicate pairs answered identically (100%, both exact and
  on the binary outcome). The registered 80% threshold does not exclude an attentive listener.
* **Fourth option:** 9 「判断できない」 answers were recorded; the UI sends them as `x`, which the
  analysis now maps to verdict 4 (it previously dropped them).
No arm-level contrast was computed from these judgements, before or after recruitment.

## 8. Timestamp

This directory is committed and tagged in a public repository before rater onboarding
begins. The paper cites the commit hash. **【OPEN — see §0.3】** Whether an additional
OSF registration is filed is to be decided.

Recorded at timestamp: `<commit hash>` — to be filled at tagging.
