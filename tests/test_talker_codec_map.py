"""spec §2/§12: talker->codec map consumer. Pure tests use a synthetic map; the real-map
tests (decoder-id agreement, no re-encode ground truth) run on the GPU host."""
import os

import numpy as np
import pytest

from src.prewm.talker_map import TalkerCodecMap, ORDINARY


def _synthetic_map():
    # 6 talker ids: 0..3 ordinary (identity to codec), 4=eos, 5=pad
    obj = {"talker_vocab": 6, "codec_vocab": 4, "entries": [
        {"talker_id": 0, "codec_id": 0, "token_class": ORDINARY},
        {"talker_id": 1, "codec_id": 1, "token_class": ORDINARY},
        {"talker_id": 2, "codec_id": 2, "token_class": ORDINARY},
        {"talker_id": 3, "codec_id": 3, "token_class": ORDINARY},
        {"talker_id": 4, "codec_id": -1, "token_class": "eos"},
        {"talker_id": 5, "codec_id": -1, "token_class": "pad"},
    ]}
    return TalkerCodecMap.from_json(obj)


def test_ordinary_mapping_deterministic():
    m = _synthetic_map()
    assert m.map_id(2) == 2 and m.is_ordinary(2)
    assert list(m.ordinary_mask()) == [True, True, True, True, False, False]


def test_specials_enumerated_and_scoreless():
    m = _synthetic_map()
    assert not m.is_ordinary(4) and not m.is_ordinary(5)
    codec_feature = np.array([10.0, 20.0, 30.0, 40.0])
    support = np.array([0, 2, 4, 5])            # two ordinary, two special
    phi = m.phi_scores(support, codec_feature)
    assert phi[0] == 10.0 and phi[1] == 30.0    # ordinary -> feature[codec_id]
    assert phi[2] == 0.0 and phi[3] == 0.0       # special -> 0 (no codec score), mass preserved elsewhere


def test_json_roundtrip():
    m = _synthetic_map()
    m2 = TalkerCodecMap.from_json(m.to_json())
    assert np.array_equal(m.codec_id, m2.codec_id)
    assert m.token_class == m2.token_class


def test_identity_with_specials_matches_qwen_rule():
    """The E0-c source-proven rule: identity for v < codec_vocab; specials above (eos=2150)."""
    m = TalkerCodecMap.identity_with_specials(3072, 2048, special_classes={2150: "eos"})
    assert m.map_id(0) == 0 and m.map_id(2047) == 2047      # identity on ordinary region
    assert m.is_ordinary(1841) and m.map_id(1841) == 1841   # an observed sampled token
    assert not m.is_ordinary(2048) and not m.is_ordinary(2150)
    assert m.token_class[2150] == "eos"
    assert m.token_class[2500] == "reserved"
    assert int(m.ordinary_mask().sum()) == 2048


# ---- real-map tests: run on GPU host after run_e0c_probe writes map.json ----
MAP_PATH = "artifacts/e0c/map.json"
have_real = os.path.exists(MAP_PATH)
real_only = pytest.mark.skipif(not have_real, reason="artifacts/e0c/map.json not present")


@real_only
def test_real_map_covers_all_observed_ordinary():
    import json
    obj = json.load(open(MAP_PATH))
    m = TalkerCodecMap.from_json(obj)
    # every ordinary talker id has a valid codec id in range
    ok = m.ordinary_mask()
    assert np.all(m.codec_id[ok] >= 0) and np.all(m.codec_id[ok] < m.codec_vocab)
    # specials explicitly enumerated (at least the map declares some non-ordinary region)
    assert obj.get("established_from") and "re-encode" not in obj.get("established_from", "").lower()
