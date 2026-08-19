"""spec §12: fit/dev/test text splits are disjoint and never leak."""
import pytest

from src.prewm.splits import text_splits, assert_disjoint


def test_disjoint_and_sizes():
    ids = [f"t{i:03d}" for i in range(100)]
    sp = text_splits(ids, 60, 20, 20, seed=1)
    assert len(sp["fit"]) == 60 and len(sp["dev"]) == 20 and len(sp["test"]) == 20
    assert_disjoint(sp)
    allids = set(sp["fit"]) | set(sp["dev"]) | set(sp["test"])
    assert len(allids) == 100  # cover everything, no dupes


def test_deterministic():
    ids = [f"t{i}" for i in range(100)]
    a = text_splits(ids, 60, 20, 20, seed=42)
    b = text_splits(ids, 60, 20, 20, seed=42)
    assert a == b
    assert a["hash"] == b["hash"]


def test_leak_detected():
    bad = {"fit": ["a", "b"], "dev": ["b", "c"], "test": ["d"]}
    with pytest.raises(AssertionError):
        assert_disjoint(bad)


def test_insufficient_texts():
    with pytest.raises(ValueError):
        text_splits(["a", "b"], 60, 20, 20)
