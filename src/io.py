"""Artifact IO, manifest, hashing, and stable sample ids (Notion §17, §18, §22).

Separation of concerns:
  * raw immutable artifacts  -> artifacts/e0/samples/<sample_id>/...
  * derived metrics          -> artifacts/e0/metrics/...
  * figures / report         -> artifacts/e0/figures, E0_REPORT.md

This module is tensor-agnostic: it handles JSON, manifests, hashing and paths so it
is fully testable without torch. Tensor persistence (.pt) happens in the experiment
scripts, which run on the GPU host.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Optional


# --------------------------------------------------------------------------- #
# Hashing                                                                      #
# --------------------------------------------------------------------------- #
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    """Deterministic JSON string for hashing configs/metadata."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_hash(cfg: dict) -> str:
    return sha256_bytes(canonical_json(cfg).encode())


# --------------------------------------------------------------------------- #
# Stable sample ids (§18): derived from prompt_id, voice_id, seed, config hash  #
# --------------------------------------------------------------------------- #
def make_sample_id(prompt_id: str, voice_id: str, seed: int, config_hash_hex: str) -> str:
    key = f"{prompt_id}|{voice_id}|{seed}|{config_hash_hex}"
    digest = sha256_bytes(key.encode())[:12]
    return f"{prompt_id}_{voice_id}_s{seed}_{digest}"


# --------------------------------------------------------------------------- #
# JSON helpers                                                                 #
# --------------------------------------------------------------------------- #
def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def read_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Manifest (JSON lines). One entry per sample; failed samples are PRESERVED.    #
# --------------------------------------------------------------------------- #
@dataclass
class ManifestEntry:
    sample_id: str
    prompt_id: str
    text: str
    voice_id: str
    seed: int
    status: str = "ok"                 # "ok" | "failed"
    failure_reason: Optional[str] = None
    # relative paths under the sample dir; presence is checked by integrity tests
    tensors: dict = field(default_factory=dict)
    shapes: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def append_manifest(manifest_path: str, entry: ManifestEntry) -> None:
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    with open(manifest_path, "a") as f:
        f.write(entry.to_json() + "\n")


def read_manifest(manifest_path: str) -> list[dict]:
    entries: list[dict] = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def verify_manifest_integrity(manifest_path: str, sample_root: str) -> list[str]:
    """Return a list of integrity problems (empty list == all good).

    Checks (Notion §16.G):
      * every ok entry references tensor paths that EXIST on disk;
      * every referenced tensor has a recorded shape;
      * failed entries are allowed to miss tensors but must carry a reason.
    """
    problems: list[str] = []
    for e in read_manifest(manifest_path):
        sid = e.get("sample_id", "<no-id>")
        if e.get("status") == "failed":
            if not e.get("failure_reason"):
                problems.append(f"{sid}: failed entry without failure_reason")
            continue
        tensors = e.get("tensors", {})
        shapes = e.get("shapes", {})
        if not tensors:
            problems.append(f"{sid}: ok entry has no tensor references")
        for name, rel in tensors.items():
            full = os.path.join(sample_root, sid, rel)
            if not os.path.exists(full):
                problems.append(f"{sid}: missing tensor file {rel}")
            if name not in shapes:
                problems.append(f"{sid}: tensor {name} has no recorded shape")
    return problems
