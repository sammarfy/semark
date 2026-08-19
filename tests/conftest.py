import os
import sys

import pytest

# Put the repo root on sys.path so `from src import ...` works from anywhere.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# --------------------------------------------------------------------------- #
# Model fixtures. Only instantiated on the GPU host when SEMARK_MODEL_READY=1.  #
# Config + reference clip are passed via environment variables so the same     #
# tests run locally (skipped) and on Colab (active).                           #
# --------------------------------------------------------------------------- #
_MODEL_READY = os.environ.get("SEMARK_MODEL_READY") == "1"


@pytest.fixture(scope="session")
def _cfg():
    import yaml
    path = os.environ.get("SEMARK_CONFIG", os.path.join(REPO_ROOT, "configs", "e0.yaml"))
    return yaml.safe_load(open(path))


@pytest.fixture(scope="session")
def adapter(_cfg):
    if not _MODEL_READY:
        pytest.skip("SEMARK_MODEL_READY != 1")
    from src.adapters.qwen3_tts import SpeechLMAdapter
    return SpeechLMAdapter.load(_cfg, device=os.environ.get("SEMARK_DEVICE", "cuda:0"))


@pytest.fixture(scope="session")
def voice_conditioning(adapter):
    ref_audio = os.environ["SEMARK_REF_AUDIO"]
    ref_text = os.environ["SEMARK_REF_TEXT"]
    return adapter.create_voice_prompt(ref_audio, ref_text)


@pytest.fixture(scope="session")
def sample_text():
    return "The morning train arrived exactly on time."


@pytest.fixture(scope="session")
def sample_waveform(adapter, voice_conditioning):
    tr = adapter.generate_trace(sample_text_value(), seed=101, sample_id="fixture",
                                voice_id="repo_default", voice_clone_prompt=voice_conditioning)
    return tr.waveform


def sample_text_value():
    return "The morning train arrived exactly on time."
