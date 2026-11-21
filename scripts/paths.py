"""Where the scripts find things outside this repository.

REPO is this checkout, found from this file's location. Every other location is an environment
variable, read when a script first asks for it, so a script needs only the variables it uses. An
unset or empty variable stops the script with its name instead of letting it fall back to some
other path (the same rule as ${VAR:?} in scripts/paths.sh).

    DATA           data root: synthesized audio, teacher sets, adapters, logs, rater exports.
                   The round-2/3 work lives under $DATA/v2; earlier shared assets (Sarashina
                   weights, the reference prompt) under $DATA/_handoff.
    ITEM_SOURCE    directory holding the original difficult-50 / katakana-50 / easy-50 test
                   sentence files (test_<set>_{vanilla,lora,yomi,...}.txt).
    COSYVOICE_DIR  CosyVoice checkout, with the CosyVoice 2 model under pretrained_models/.
    T5GEMMA_DIR    T5Gemma-TTS checkout.
    DATASET_DIR    root of the released teacher dataset (UtterTune-SelfDistill-data); paths
                   inside its files are relative to this root.
    SARASHINA_DIR  Sarashina2.2-TTS code (the directory that holds the sarashina_tts package).

The Irodori code is in this repository (irodori/, set up by irodori/setup.sh), so it needs no variable.

Scripts outside scripts/ itself put scripts/ on sys.path and then `import paths`.
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_VARIABLES = ("DATA", "ITEM_SOURCE", "COSYVOICE_DIR", "T5GEMMA_DIR", "DATASET_DIR", "SARASHINA_DIR")


def __getattr__(name):
    if name not in _VARIABLES:
        raise AttributeError(f"module 'paths' has no attribute {name!r}")
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"environment variable {name} is not set (see scripts/paths.py)")
    return value
