"""§16.G: manifest/artifact integrity, stable ids, config hashing."""
import os

import numpy as np

from src import io
from src.io import ManifestEntry, make_sample_id, config_hash


def test_sample_id_is_stable_and_unique():
    ch = config_hash({"a": 1, "b": [1, 2, 3]})
    a = make_sample_id("p0", "repo_default", 101, ch)
    b = make_sample_id("p0", "repo_default", 101, ch)
    c = make_sample_id("p0", "repo_default", 202, ch)
    assert a == b            # deterministic
    assert a != c            # seed changes id
    assert a.startswith("p0_repo_default_s101_")


def test_config_hash_is_order_independent():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_manifest_roundtrip_and_integrity(tmp_path):
    root = tmp_path / "samples"
    sid = "p0_repo_default_s101_abcdef012345"
    sample_dir = root / sid
    sample_dir.mkdir(parents=True)
    # create a fake tensor file
    (sample_dir / "semantic_ids.pt").write_bytes(b"fake")

    manifest = str(tmp_path / "manifest.jsonl")
    entry = ManifestEntry(
        sample_id=sid,
        prompt_id="p0",
        text="hello",
        voice_id="repo_default",
        seed=101,
        status="ok",
        tensors={"semantic_ids": "semantic_ids.pt"},
        shapes={"semantic_ids": [42]},
    )
    io.append_manifest(manifest, entry)

    got = io.read_manifest(manifest)
    assert len(got) == 1 and got[0]["sample_id"] == sid

    problems = io.verify_manifest_integrity(manifest, str(root))
    assert problems == []


def test_integrity_flags_missing_tensor(tmp_path):
    root = tmp_path / "samples"
    sid = "p1_repo_default_s202_deadbeef0000"
    (root / sid).mkdir(parents=True)
    manifest = str(tmp_path / "manifest.jsonl")
    io.append_manifest(
        manifest,
        ManifestEntry(
            sample_id=sid, prompt_id="p1", text="x", voice_id="repo_default", seed=202,
            status="ok", tensors={"waveform": "waveform.wav"}, shapes={"waveform": [24000]},
        ),
    )
    problems = io.verify_manifest_integrity(manifest, str(root))
    assert any("missing tensor" in p for p in problems)


def test_failed_sample_preserved_not_dropped(tmp_path):
    """§22: a failed sample stays in the manifest with a reason, never silently omitted."""
    root = tmp_path / "samples"
    root.mkdir(parents=True)
    manifest = str(tmp_path / "manifest.jsonl")
    io.append_manifest(
        manifest,
        ManifestEntry(
            sample_id="p2_repo_default_s303_000000000000", prompt_id="p2", text="y",
            voice_id="repo_default", seed=303, status="failed",
            failure_reason="OOM during generation",
        ),
    )
    problems = io.verify_manifest_integrity(manifest, str(root))
    assert problems == []  # failed-with-reason is valid


def test_failed_without_reason_flagged(tmp_path):
    root = tmp_path / "samples"
    root.mkdir(parents=True)
    manifest = str(tmp_path / "manifest.jsonl")
    io.append_manifest(
        manifest,
        ManifestEntry(
            sample_id="p3_repo_default_s404_111111111111", prompt_id="p3", text="z",
            voice_id="repo_default", seed=404, status="failed", failure_reason=None,
        ),
    )
    problems = io.verify_manifest_integrity(manifest, str(root))
    assert any("without failure_reason" in p for p in problems)
