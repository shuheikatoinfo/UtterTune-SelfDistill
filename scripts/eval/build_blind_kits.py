"""CP5 — Build BLIND RATER KITS for the completed Sarashina 4-speaker headline cells.

Follows the blind_kit / blind_kit_tf457 precedent (hidden labels, fixed-seed shuffle,
KEY_DO_NOT_OPEN, catch trials = duplicated stimuli) but upgraded to the paper's §4.2 load
design: raters judge SHORT CLIPS of the target word cut with 0.5s margin on each side,
one clip per judgment, blind (no condition labels), with a rater-facing MANIFEST (word +
prescribed reading/accent, NO arm labels) and a separate KEY.csv mapping blind IDs back to
(cell, arm, sentence idx, speaker).

Two phases (so GPU use is isolated):
  --align  : whisper-large-v3 word-level timestamps for every wav in the packaged cells
             (GPU; run as: <venv_python> build_blind_kits.py --align)
  --build  : cut clips + write kits (CPU only; requires --align output)

Integrity: NO take selection — every wav of every packaged (cell, arm, idx, spk) is included.
Target-word localization is whisper word-timestamp based with a disclosed proportional
fallback (flagged per-item in KEY.csv as align_method). Randomization seed = 0 throughout.

Cells packaged (all Sarashina, 4 speakers, per STATUS §1 / design §4.2):
  blockA_difficult  : blockA_sarashina_difficult_4spk  {noedit, ours}            50x4x2 = 400
  blockA_easy       : blockA_sarashina_easy_4spk       {noedit, ours}  15 sampled idx = 120
  blockB2_difficult : blockB2_sarashina_4spk           {noedit, ours, pronsteering}   = 600
  blockB2_katakana  : blockB2_sarashina_katakana_4spk  {noedit, ours, pronsteering}   = 600
                      (Block A katakana headline cell = the {noedit, ours} subset; the KEY
                       lets you slice it out, no separate kit needed)
Plus 5% duplicate consistency/attention items per kit (precedent: catch = same stimulus
re-presented under a new blind ID; gold-anchored attention items are a TODO, see MANIFEST).
"""
import argparse, csv, json, os, random, re, shutil, sys, unicodedata
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/, for paths.py
import paths

DATA = paths.DATA
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Which synthesis generation to package. There is NO safe default: the unsuffixed directories
# still hold the audio of the adapter trained before the clean-room rebuild, and defaulting to "" silently
# builds a rater kit from superseded wavs that no reported number was measured on. Must be set.
SUF = os.environ.get("CELL_SUFFIX")
if SUF is None:
    raise SystemExit("set CELL_SUFFIX (e.g. CELL_SUFFIX=_cleanroom_v2); there is no safe default")
OUTROOT = os.environ.get("KIT_OUT", f"{DATA}/blind_kits{SUF}")
ALIGNDIR = f"{OUTROOT}/_align"
TD = paths.ITEM_SOURCE
SEED = 0
MARGIN = 0.5          # (unused in rev.4: stimuli are full sentences, not clips)
FULL_SENTENCE = os.environ.get("FULL_SENTENCE", "1") == "1"
ATTN_FRAC = 0.05      # duplicate consistency items

# ---------------------------------------------------------------- test-set parsing
def _lines(fn):
    return [l.rstrip("\n") for l in open(f"{TD}/{fn}", encoding="utf-8") if l.strip()]

TAGRE = re.compile(r"<PHON_START>(.*?)<PHON_END>")
BOLD = re.compile(r"<b>(.*?)</b>")
ACCENT_MARKS = "＼￣・'/[]"

def _strip_accent(s):
    return "".join(c for c in s if c not in ACCENT_MARKS)

def _surface_from_diff(vanilla, tagged):
    """tagged = vanilla with the target word replaced by <PHON_START>kana<PHON_END>.
    Return (surface_word, reading_kana) via common prefix/suffix against vanilla."""
    m = TAGRE.search(tagged)
    reading = _strip_accent(m.group(1))
    pre, post = tagged[:m.start()], tagged[m.end():]
    assert vanilla.startswith(pre) and vanilla.endswith(post), (vanilla, tagged)
    surface = vanilla[len(pre):len(vanilla) - len(post)] if post else vanilla[len(pre):]
    return surface, reading, pre  # pre = char offset context for proportional fallback

def _author_corrections():
    """Author target-accent corrections (prereg §2 verification, done before any audio was
    judged). The kit reads the released item files directly, so corrections must be applied
    here or raters would be shown a target the author already rejected."""
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                     "preregistration", "items", "target_verification.json")
    fixes = {}
    if os.path.exists(p):
        tv = json.load(open(p, encoding="utf-8"))
        for st, d in tv.get("sets", {}).items():
            for r in d.get("non_ok_items", []):
                if r.get("verdict") == "fix" and r.get("corrected_target"):
                    fixes[(st, int(r["item_id"].split(":")[1]) - 1)] = r["corrected_target"]
    return fixes


def load_sets():
    sets = {}
    # difficult: target word tagged in _lora; accent notation in _with_accent
    van = _lines("test_difficult_50_vanilla.txt")
    lora = _lines("test_difficult_50_lora.txt")
    acc = _lines("test_difficult_50_with_accent.txt")
    items = []
    for v, l, a in zip(van, lora, acc):
        s, r, pre = _surface_from_diff(v, l)
        ab = BOLD.search(a)
        items.append(dict(sentence=v, surface=s, reading=r, prefix=pre,
                          accent=(ab.group(1) if ab else "")))
    sets["difficult"] = items
    # easy: target word tagged in _phon; no accent file (non-degradation cell)
    van = _lines("test_easy_50.txt")
    phon = _lines("test_easy_50_phon.txt")
    items = []
    for v, p in zip(van, phon):
        s, r, pre = _surface_from_diff(v, p)
        items.append(dict(sentence=v, surface=s, reading=r, prefix=pre, accent=""))
    sets["easy"] = items
    # katakana: RATED target = <b>...</b> katakana word in _with_accent (locality target,
    # appears literally in the sentence); the edited word (phon tag) is a different word.
    van = _lines("test_katakana_50.txt")
    acc = _lines("test_katakana_50_with_accent.txt")
    phon = _lines("test_katakana_50_phon.txt")
    items = []
    for v, a, p in zip(van, acc, phon):
        ab = BOLD.search(a)
        word = _strip_accent(ab.group(1))
        assert word in v, (word, v)
        pm = TAGRE.search(p)
        items.append(dict(sentence=v, surface=word, reading=word,
                          prefix=v[:v.index(word)], accent=ab.group(1),
                          edited_word_reading=_strip_accent(pm.group(1)) if pm else ""))
    sets["katakana"] = items
    # apply the author's dictionary-verified corrections to the PRESCRIBED accent shown to raters
    for (st, i), corrected in _author_corrections().items():
        if st in sets and 0 <= i < len(sets[st]):
            old = sets[st][i]["accent"]
            sets[st][i]["accent"] = corrected
            print(f"author correction applied: {st}[{i}] {old} -> {corrected}")
    return sets

# ---------------------------------------------------------------- cells
def easy_sample_idx():
    return sorted(random.Random(SEED).sample(range(50), 15))

# katakana locality cells: 8 of the 50 sentences originally tagged the JUDGED word itself instead
# of another word (test-set generator defect). They were NOT dropped — the edit site was
# regenerated onto a non-target kanji noun so all 50 sentences test control leak the same way.
# See preregistration/items/katakana_regenerated_8.json and data/derived/.
KATAKANA_IDX = list(range(50))


def cells():
    """The five preregistered listening cells (PREREGISTRATION.md §3, 788 judgments).

    Stimuli are FULL SENTENCES: the ±0.5 s clip design was dropped because the aligner could not
    locate the target word in 11 % of difficult items, concentrated in the No-edit arm (a word the
    model mis-reads is a word the aligner cannot find), and because fragmenting adds noise that is
    anti-conservative for the non-inferiority test.

    Layouts: "spk_subdir" = <dir>/spk<N>/<arm>/NN.wav ; "dir_per_spk" = <dir_fmt % spk>/<arm>/NN.wav
    ; "flat" = <dir>/<arm>/NN.wav (single speaker, arm dirs at the top).
    """
    IRO = f"{DATA}/cleanroom_rebuild/irodori_eval_v2"
    CMP = f"{DATA}/cleanroom_rebuild/cv2_compare"

    # KIT_PROFILE=refalign builds the kit from the REFERENCE-ALIGNED synthesis: every backbone
    # driven by the same two held-out ITA references, the corrected difficult-50 input, the fixed
    # T5Gemma tagging and whichever CosyVoice2 adapter won on the SELECTION split. The default
    # profile still points at the pre-alignment directories so the earlier kit stays reproducible.
    # Speaker directories are named (amitaro/kamijou) rather than spk0/spk1 under this profile.
    if os.environ.get("KIT_PROFILE") == "round2":
        # Round 2 (PREREGISTRATION_v2.md item 6): every backbone x {kana, tag}, plus {no edit} on
        # Sarashina; one voice per word from speaker_assignment42.json cells A2..D2. EVERY arm is
        # synthesized in round 2 under the same script, seeds and degeneracy guard as its own tag arm
        # (the round-1 sweep42 wavs predate the shared guard, so reusing them would give the tag arm
        # retries the comparators never had).
        SW = f"{DATA}/sweep42"; V2 = f"{DATA}/v2/synth"
        A = json.load(open(os.environ.get(
            "KIT_ASSIGN", f"{REPO}/preregistration/items/speaker_assignment42.json")))["cells"]
        isp = lambda cell: {int(k): v for k, v in A[cell].items()}
        return [
            dict(name="A2_sarashina_difficult", layout="per_item_spk_arms",
                 arm_dirs={"noedit": f"{V2}/sarashina/difficult", "kana": f"{V2}/sarashina/difficult", "ours": f"{V2}/sarashina/difficult"},
                 arms=["noedit", "kana", "ours"], set="difficult", idx=list(range(50)), item_spk=isp("A2_sarashina_difficult")),
            dict(name="B2_cv2_difficult", layout="per_item_spk_arms",
                 arm_dirs={"kana": f"{V2}/cv2/difficult", "ours": f"{V2}/cv2/difficult"},
                 arms=["kana", "ours"], set="difficult", idx=list(range(50)), item_spk=isp("B2_cv2_difficult")),
            dict(name="C2_irodori_difficult", layout="per_item_flat",
                 arm_dirs={"kana": f"{V2}/irodori/difficult_kana", "ours": f"{V2}/irodori/difficult"},
                 arm_subdir={"kana": "noedit"},   # the kana rendition is the base arm over the kana sentences
                 arms=["kana", "ours"], set="difficult", idx=list(range(50)), item_spk=isp("C2_irodori_difficult")),
            dict(name="D2_t5gemma_difficult", layout="per_item_flat",
                 arm_dirs={"kana": f"{V2}/t5gemma/difficult_base/blockA_t5gemma_difficult",
                           "ours": f"{V2}/t5gemma/difficult/blockA_t5gemma_difficult"},
                 arms=["kana", "ours"], set="difficult", idx=list(range(50)), item_spk=isp("D2_t5gemma_difficult")),
        ]

    if os.environ.get("KIT_PROFILE") == "round3":
        # Round 3 (PREREGISTRATION_v2.md section 16): the tag arm of every backbone comes from the round-3 adapter
        # (synth_r3); the base arms (kana; no edit on Sarashina) are base-model renditions and are the round-2 clips,
        # same voices, seeds and guard. Cells keep the round-2 names so the registered analysis applies unchanged.
        V2 = f"{DATA}/v2/synth"; V3 = f"{DATA}/v2/synth_r3"
        A = json.load(open(os.environ.get(
            "KIT_ASSIGN", f"{REPO}/preregistration/items/speaker_assignment42.json")))["cells"]
        isp = lambda cell: {int(k): v for k, v in A[cell].items()}
        cells_r3 = [
            dict(name="A2_sarashina_difficult", layout="per_item_spk_arms",
                 arm_dirs={"noedit": f"{V2}/sarashina/difficult", "kana": f"{V2}/sarashina/difficult", "ours": f"{V3}/sarashina/difficult"},
                 arms=["noedit", "kana", "ours"], set="difficult", idx=list(range(50)), item_spk=isp("A2_sarashina_difficult")),
            dict(name="B2_cv2_difficult", layout="per_item_spk_arms",
                 arm_dirs={"kana": f"{V2}/cv2/difficult", "ours": f"{V3}/cv2/difficult"},
                 arms=["kana", "ours"], set="difficult", idx=list(range(50)), item_spk=isp("B2_cv2_difficult")),
            dict(name="C2_irodori_difficult", layout="per_item_flat",
                 arm_dirs={"kana": f"{V2}/irodori/difficult_kana", "ours": f"{V3}/irodori/difficult"},
                 arm_subdir={"kana": "noedit"},
                 arms=["kana", "ours"], set="difficult", idx=list(range(50)), item_spk=isp("C2_irodori_difficult")),
            dict(name="D2_t5gemma_difficult", layout="per_item_flat",
                 arm_dirs={"kana": f"{V2}/t5gemma/difficult_base/blockA_t5gemma_difficult",
                           "ours": f"{V3}/t5gemma/difficult/blockA_t5gemma_difficult"},
                 arms=["kana", "ours"], set="difficult", idx=list(range(50)), item_spk=isp("D2_t5gemma_difficult")),
        ]
        skip = [x for x in os.environ.get("KIT_SKIP", "").split(",") if x]   # e.g. KIT_SKIP=cv2,t5gemma: partial build for the author's dry run
        return [c for c in cells_r3 if not any(k in c["name"] for k in skip)]

    if os.environ.get("KIT_PROFILE") == "sweep42":
        # 42-voice design: one held-out voice per WORD-UNIT, assigned by hash before any audio was
        # selected (preregistration/items/speaker_assignment42.json). The speaker therefore varies
        # per item, not per cell, so these cells carry `item_spk` (idx -> [speaker]) and layout
        # "per_item_spk": <dir>/spk<N>/<arm>/NN.wav. Cells A/C and B-ours come from sweep42/, which
        # was synthesized over all 42 refs; only the assigned (item, speaker) pairs are packaged.
        SW = f"{DATA}/sweep42"; K42 = f"{DATA}/kit42"
        A = json.load(open(os.environ.get(
            "KIT_ASSIGN", f"{REPO}/preregistration/items/speaker_assignment42.json")))["cells"]
        isp = lambda cell: {int(k): v for k, v in A[cell].items()}
        return [
            dict(name="A_sarashina_difficult", layout="per_item_spk", dir=f"{SW}/sarashina",
                 arms=["noedit", "kana", "ours"], set="difficult", idx=list(range(50)),
                 item_spk=isp("A_sarashina_difficult")),
            # The released-adapter arm was synthesized with ARMS=ours, so on disk its wavs sit in
            # .../spk<N>/ours/ ; arm_subdir maps the KIT's arm label to the directory name.
            dict(name="B_cv2_datafree_vs_recorded", layout="per_item_spk_arms",
                 arm_dirs={"uttertune_released": f"{K42}/cv2_uttertune_released", "ours": f"{SW}/cv2"},
                 arm_subdir={"uttertune_released": "ours", "ours": "ours"},
                 arms=["uttertune_released", "ours"], set="difficult", idx=list(range(50)),
                 item_spk=isp("B_cv2_datafree_vs_recorded")),
            dict(name="C_irodori_difficult", layout="per_item_spk", dir=f"{SW}/irodori",
                 arms=["noedit", "ours"], set="difficult", idx=list(range(50)),
                 item_spk=isp("C_irodori_difficult")),
            dict(name="D_sarashina_katakana", layout="per_item_spk", dir=f"{K42}/sarashina_katakana",
                 arms=["noedit", "ours"], set="katakana", idx=KATAKANA_IDX,
                 item_spk=isp("D_sarashina_katakana")),
        ]

    if os.environ.get("KIT_PROFILE") == "refalign":
        RA = f"{DATA}/refalign"
        SPKN = ["amitaro", "kamijou"]
        CV2_OURS = os.environ.get("KIT_CV2_OURS")
        if not CV2_OURS:
            raise SystemExit("KIT_PROFILE=refalign needs KIT_CV2_OURS (the adapter chosen on the "
                             "SELECTION split, e.g. .../refalign/cv2/%s/ours_cleanroom/ours)")
        KATA = os.environ.get("KIT_KATAKANA_DIR")
        if not KATA:
            raise SystemExit("KIT_PROFILE=refalign needs KIT_KATAKANA_DIR (cell D katakana "
                             "synthesis under the aligned references; not yet produced)")
        return [
            dict(name="A_sarashina_difficult", layout="name_subdir",
                 dir=f"{RA}/sarashina_difficult", spk_names=["spk0", "spk1"],
                 arms=["noedit", "kana", "ours"],
                 arms_by_spk={0: ["noedit", "kana", "ours"], 1: ["noedit", "ours"]},
                 set="difficult", idx=list(range(50)), spks=[0, 1]),
            dict(name="B_cv2_datafree_vs_recorded", layout="explicit",
                 arm_dirs={"uttertune_released": f"{RA}/cv2/{SPKN[0]}/uttertune_released/ours",
                           "ours": CV2_OURS % SPKN[0] if "%s" in CV2_OURS else CV2_OURS},
                 set="difficult", idx=list(range(50)), spks=[0]),
            dict(name="C_irodori_difficult", layout="name_subdir",
                 dir=f"{RA}/irodori", spk_names=SPKN,
                 arms=["noedit", "ours"], set="difficult", idx=list(range(50)), spks=[0, 1]),
            # katakana is produced by the Sarashina 4spk script, whose output layout is spk0/spk1
            # (cell A likewise); only the Irodori and CV2 runners write amitaro/kamijou dirs.
            dict(name="D_sarashina_katakana", layout="name_subdir",
                 dir=KATA, spk_names=["spk0", "spk1"],
                 arms=["noedit", "ours"], set="katakana", idx=list(range(50)), spks=[0, 1]),
        ]
    return [
        # A carries the study's existential contrast (kana vs ours: can a reading-only input
        # place an accent?). spk1 adds the speaker replication; it needs only the noedit/ours
        # pair, so the kana arm stays on spk0 and no rater time is spent duplicating it.
        dict(name="A_sarashina_difficult", layout="spk_subdir",
             dir=f"{DATA}/blockA_sarashina_difficult_4spk{SUF}",
             arms=["noedit", "kana", "ours"],
             arms_by_spk={0: ["noedit", "kana", "ours"], 1: ["noedit", "ours"]},
             set="difficult", idx=list(range(50)), spks=[0, 1]),
        # B: both arms are EDITED; the contrast is 15,097 recorded pairs vs zero recordings.
        dict(name="B_cv2_datafree_vs_recorded", layout="explicit",
             arm_dirs={"uttertune_released": f"{CMP}/uttertune_released/ours",
                       "ours": f"{CMP}/ours_cleanroom/ours"},
             set="difficult", idx=list(range(50)), spks=[0]),
        dict(name="C_irodori_difficult", layout="dir_per_spk",
             dir_fmt=IRO + "/difficult_spk%d",
             arms=["noedit", "ours"], set="difficult", idx=list(range(50)), spks=[0, 1]),
        dict(name="D_sarashina_katakana", layout="spk_subdir",
             dir=f"{DATA}/blockB2_sarashina_katakana_4spk{SUF}",
             arms=["noedit", "ours"], set="katakana", idx=list(range(50)), spks=[0, 1]),
        # No E_irodori_katakana (Irodori leak check): it would repeat D's
        # question on a second backbone and detects nothing D does not, at 210 judgments per
        # rater. That budget pays for the speaker replication in A and C instead.
    ]


def _paths(c, spk):
    """(arm, base_dir) pairs for one cell/speaker, whichever layout the cell uses."""
    if c["layout"] == "explicit":
        return list(c["arm_dirs"].items())
    if c["layout"] == "dir_per_spk":
        base = c["dir_fmt"] % spk
    elif c["layout"] == "name_subdir":
        base = f"{c['dir']}/{c['spk_names'][spk]}"
    else:
        base = f"{c['dir']}/spk{spk}"
    arms = c.get("arms_by_spk", {}).get(spk, c["arms"])
    return [(a, f"{base}/{a}") for a in arms]


def cell_entries(c, items):
    """Every wav of one cell as dicts. THE single enumeration: both the align phase and the build
    phase call this, so they cannot drift. Cells with `item_spk` vary the speaker per item."""
    out = []
    if "item_spk" in c:
        for i in c["idx"]:
            for spk in c["item_spk"][i]:
                for arm in c["arms"]:
                    sub = c.get("arm_subdir", {}).get(arm, arm)
                    lay = c.get("arm_layout", {}).get(arm, c["layout"])
                    if lay == "per_item_spk_arms" or lay == "per_item_spk":
                        base = (f'{c["arm_dirs"][arm]}/spk{spk}/{sub}' if "arm_dirs" in c else f'{c["dir"]}/spk{spk}/{sub}')
                    elif lay == "per_item_flat":
                        base = f'{c["arm_dirs"][arm]}/{sub}'
                    elif c["layout"] == "per_item_spk_arms":
                        base = f'{c["arm_dirs"][arm]}/spk{spk}/{sub}'
                    elif c["layout"] == "per_item_flat":
                        # writer put every item in one flat dir; the speaker is still the assigned one
                        base = f'{c["arm_dirs"][arm]}/{sub}'
                    else:
                        base = f'{c["dir"]}/spk{spk}/{sub}'
                    out.append(dict(arm=arm, spk=spk, idx=i, path=f"{base}/{i:02d}.wav", item=items[i]))
        return out
    for spk in c["spks"]:
        for arm, base in _paths(c, spk):
            for i in c["idx"]:
                out.append(dict(arm=arm, spk=spk, idx=i, path=f"{base}/{i:02d}.wav", item=items[i]))
    return out


def wav_list(sets):
    """Every wav to package: (cell, arm, spk, idx, path, item)."""
    out = []
    for c in cells():
        for e in cell_entries(c, sets[c["set"]]):
            out.append((c["name"], e["arm"], e["spk"], e["idx"], e["path"], e["item"]))
    return out


# ---------------------------------------------------------------- phase 1: align (GPU)
def run_align():
    import whisper, torch
    if os.environ.get("DISABLE_CUDNN", "1") == "1":
        # This box's cuDNN backend fails to finalize tensor descriptors for some conv shapes
        # (CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH). The native kernels are numerically
        # equivalent, only slower. Default ON here because alignment is a one-off.
        torch.backends.cudnn.enabled = False
    os.makedirs(ALIGNDIR, exist_ok=True)
    sets = load_sets()
    wl = wav_list(sets)
    # Irodori routes by morpheme-boundary matching; a sentence that does not match falls through
    # to the unedited base and no "ours" wav is written. Prereg §3 fixes the handling: such items
    # are recorded as "edit did not fire", EXCLUDED from the cell (never packaged as if edited),
    # and their count is reported. Any OTHER missing wav is still fatal.
    missing = [t for t in wl if not os.path.exists(t[4])]
    unfired = [t for t in missing if t[1] == "ours"]
    fatal = [t for t in missing if t[1] != "ours"]
    if fatal:
        print(f"FATAL: {len(fatal)} missing wavs, e.g. {[t[4] for t in fatal[:5]]}"); sys.exit(1)
    if unfired:
        skip = {(t[0], t[2], t[3]) for t in unfired}   # (cell, spk, idx): drop BOTH arms so the
        wl = [t for t in wl if (t[0], t[2], t[3]) not in skip]  # comparison stays paired
        rec = sorted((t[0], t[2], t[3]) for t in unfired)
        json.dump({"n_unfired": len(unfired), "items": rec},
                  open(f"{ALIGNDIR}/edit_did_not_fire.json", "w"), ensure_ascii=False, indent=1)
        print(f"edit did not fire on {len(unfired)} item(s); dropped both arms of each "
              f"(paired design) -> {ALIGNDIR}/edit_did_not_fire.json")
    model = whisper.load_model("large-v3", device="cuda" if torch.cuda.is_available() else "cpu")
    aligns = {}
    done_f = f"{ALIGNDIR}/word_align.json"
    if os.path.exists(done_f):
        aligns = json.load(open(done_f))
    for n, (cell, arm, spk, i, p, it) in enumerate(wl):
        if p in aligns:
            continue
        r = model.transcribe(p, language="ja", word_timestamps=True,
                             condition_on_previous_text=False, temperature=0.0)
        words = [dict(w=w["word"].strip(), s=round(w["start"], 3), e=round(w["end"], 3))
                 for seg in r["segments"] for w in seg["words"]]
        aligns[p] = dict(text=r["text"], words=words)
        if (n + 1) % 50 == 0:
            json.dump(aligns, open(done_f, "w"), ensure_ascii=False)
            print(f"aligned {n+1}/{len(wl)}", flush=True)
    json.dump(aligns, open(done_f, "w"), ensure_ascii=False)
    print(f"align done: {len(aligns)} files -> {done_f}")

# ---------------------------------------------------------------- kana matching
def to_kata(s):
    s = unicodedata.normalize("NFKC", s)
    out = []
    for ch in s:
        o = ord(ch)
        if 0x3041 <= o <= 0x3096:
            ch = chr(o + 0x60)
        out.append(ch)
    s = "".join(out)
    return re.sub(r"[^ァ-ヶー]", "", s).translate(str.maketrans({"ヂ": "ジ", "ヅ": "ズ", "ヴ": "ブ"}))

_g2p_cache = {}
def word_kana(w):
    if w not in _g2p_cache:
        k = to_kata(w)
        if not k:
            try:
                import pyopenjtalk
                k = to_kata(pyopenjtalk.g2p(w, kana=True))
            except Exception:
                k = ""
        _g2p_cache[w] = k
    return _g2p_cache[w]

def locate_target(align, item, wav_dur):
    """Return (t0, t1, method). Fuzzy-match target reading/surface against consecutive
    whisper words (kana-normalized); proportional char-offset fallback if score < 0.5."""
    import difflib
    words = align["words"]
    targ_r = to_kata(item["reading"])
    targ_s = item["surface"]
    best = (0.0, None)
    for a in range(len(words)):
        acc_k, acc_s = "", ""
        for b in range(a, min(a + 8, len(words))):
            acc_k += word_kana(words[b]["w"])
            acc_s += words[b]["w"]
            if len(acc_k) > 2 * len(targ_r) + 6:
                break
            sc = max(difflib.SequenceMatcher(None, acc_k, targ_r).ratio(),
                     difflib.SequenceMatcher(None, acc_s, targ_s).ratio(),
                     difflib.SequenceMatcher(None, acc_k, word_kana(targ_s)).ratio() if targ_s else 0)
            if sc > best[0]:
                best = (sc, (words[a]["s"], words[b]["e"]))
    if best[0] >= 0.5 and best[1] is not None:
        return best[1][0], best[1][1], f"whisper({best[0]:.2f})"
    # proportional fallback: char offset in sentence -> time
    sent = item["sentence"]
    c0 = len(item["prefix"]); c1 = c0 + len(item["surface"])
    t0 = wav_dur * c0 / max(len(sent), 1)
    t1 = wav_dur * c1 / max(len(sent), 1)
    return t0, t1, "proportional_fallback"

# ---------------------------------------------------------------- phase 2: build (CPU)
def run_build():
    import soundfile as sf
    import numpy as np
    # Full-sentence stimuli need no word alignment at all (rev.4).
    aligns = {} if FULL_SENTENCE else json.load(open(f"{ALIGNDIR}/word_align.json"))
    sets = load_sets()
    summary = {}
    for c in cells():
        items = sets[c["set"]]
        # Build from the SAME enumeration as wav_list() so the two phases cannot drift: the
        # speaker set is per-cell (not range(4)) and the on-disk layout differs per backbone.
        entries = cell_entries(c, items)  # one per wav; same enumeration as wav_list()
        rng = random.Random(SEED)
        rng.shuffle(entries)
        # 5% duplicate consistency items (precedent: catch = same stimulus twice)
        n_attn = max(1, round(ATTN_FRAC * len(entries)))
        dups = [dict(e, is_attention=True) for e in rng.sample(entries, n_attn)]
        allq = entries + dups
        rng.shuffle(allq)
        kit = f"{OUTROOT}/{c['name']}"
        os.makedirs(f"{kit}/clips", exist_ok=True)
        key_rows, man_rows, fb = [], [], 0
        for qn, e in enumerate(allq, 1):
            bid = f"q{qn:04d}"
            if FULL_SENTENCE:
                # rev.4: the stimulus IS the utterance. No alignment, no cut — see prereg §3
                # ("Presentation: full sentences, not clips").
                shutil.copy2(e["path"], f"{kit}/clips/{bid}.wav")
                key_rows.append(dict(blind_id=bid, cell=c["name"], arm=e["arm"], idx=e["idx"],
                                     speaker=f"spk{e['spk']}", target_word=e["item"]["surface"],
                                     target_reading=e["item"]["reading"],
                                     prescribed_accent=e["item"]["accent"], src_wav=e["path"],
                                     clip_start="", clip_end="", align_method="full_sentence",
                                     is_attention_dup=bool(e.get("is_attention"))))
                man_rows.append(dict(blind_id=bid, target_word=e["item"]["surface"],
                                     target_reading=e["item"]["reading"],
                                     prescribed_accent=e["item"]["accent"]))
                continue
            al = aligns[e["path"]]
            wav, sr = sf.read(e["path"])
            dur = len(wav) / sr
            t0, t1, method = locate_target(al, e["item"], dur)
            if method == "proportional_fallback":
                fb += 1
            s0 = max(0, int((t0 - MARGIN) * sr)); s1 = min(len(wav), int((t1 + MARGIN) * sr))
            sf.write(f"{kit}/clips/{bid}.wav", wav[s0:s1], sr)
            it = e["item"]
            man_rows.append(dict(blind_id=bid, target_word=it["surface"],
                                 target_reading=it["reading"], prescribed_accent=it["accent"]))
            key_rows.append(dict(blind_id=bid, cell=c["name"], arm=e["arm"], idx=e["idx"],
                                 speaker=f"spk{e['spk']}", target_word=it["surface"],
                                 target_reading=it["reading"], prescribed_accent=it["accent"],
                                 src_wav=e["path"], clip_start=round(max(0, t0 - MARGIN), 3),
                                 clip_end=round(min(dur, t1 + MARGIN), 3), align_method=method,
                                 is_attention_dup=bool(e.get("is_attention"))))
        with open(f"{kit}/KEY_DO_NOT_OPEN.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(key_rows[0].keys())); w.writeheader(); w.writerows(key_rows)
        with open(f"{kit}/MANIFEST.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(man_rows[0].keys())); w.writeheader(); w.writerows(man_rows)
        with open(f"{kit}/README_rater.md", "w") as f:
            f.write(f"""# 判定のしかた

音声を順に聴き、画面に示された**対象語**について、そのアクセント型が
画面の**指定アクセント**どおりかを判定してください。1セッションは約95問・20分です。

## 選択肢

- **① 一致している** — 対象語が聞こえ、アクセント型も指定どおり
- **② 対象語だがアクセントが違う** — 語は聞こえるが、下降位置が指定と異なる
- **③ 対象語に聞こえない（読みが違う）** — 別の語に聞こえる
- **判断できない** — 音声が壊れている、無音、雑音など、判定不能なとき

## 判定の基準

- **アクセント型だけを聴いてください。** 音質・自然さ・声の好みは判定対象ではありません。
- 表記は `＼` が下降核（その直後で下がる）、`￣` が平板（下降なし）です。
  例: `シ＼ンシャク` = 1モーラ目の直後で下降 ／ `アイマイ￣` = 下降なし。
- **規範を思い出したり辞書を引いたりする必要はありません。** 目標のアクセント型は
  こちらで事前に確定済みです。「示された型で言えているか」だけを聴いてください。

## 進め方

- 音声は**文全体**です。対象語だけを切り出してはいません。**ヘッドホン**をご使用ください。
- **どの音声がどの手法によるものかは伏せてあります。** 画面にも配信物にも含まれません。
- 回答は自動保存され、途中で閉じても同じ端末・同じブラウザで再開できます。
- 集中状態の確認のため、同一音声を再提示する項目が約5%含まれています。
- 25分ごとに休憩の案内が出ます。無理のない範囲で進めてください。
""")
    json.dump(summary, open(f"{OUTROOT}/build_summary.json", "w"), ensure_ascii=False, indent=1)

RATER_README = """# 盲検判定キット: {cell}

## 判定していただくこと

`clips/qNNNN.wav` を順に聴き、MANIFEST.csv の同じ blind_id 行にある **対象語** について、
**その語のアクセント型が、MANIFEST の `prescribed_accent` 欄に示された型と一致するか**を
二値でご判定ください（**1 = 一致 / 0 = 不一致**、判断できない場合のみ `x`）。合計 {n} 問。

- **アクセント型だけを見てください。** 読み（どの音で読んでいるか）や音質・自然さは判定対象では
  ありません。読みが多少不明瞭でも、アクセント型が指定どおりなら「1」です。
- 表記は `＼` が下降核（その直後で下がる）、`￣` が平板（下降なし）です。
  例: `シ＼ンシャク` = 1モーラ目の直後で下降 ／ `アイマイ￣` = 下降なし。
- **規範を思い出したり辞書を引いたりする必要はありません。** 目標のアクセント型は
  こちらで事前に確定済みです。「示された型で言えているか」だけを聴いてください。

## 進め方

- クリップは対象語の前後0.5秒を切り出した短い音声です。**ヘッドホン**をご使用ください。
- **条件ラベルは意図的に伏せてあります。** どの音声がどの手法によるものかは分かりません。
  `KEY_DO_NOT_OPEN.csv` は開けないでください。
- 25分判定＋5分休憩を1セッション、1日最大4セッションでお願いします。
- 集中状態の確認のため、同一音声を再提示する項目が約5%含まれています。
"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--align", action="store_true")
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    if a.align:
        run_align()
    elif a.build:
        run_build()
    else:
        ap.error("need --align or --build")
