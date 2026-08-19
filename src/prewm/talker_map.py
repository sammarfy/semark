"""Talker(3072) -> codec-semantic(2048) mapping consumer (spec §2, E0-c).

The map itself is established empirically on the GPU host (run_e0c_probe.py) and saved to
artifacts/e0c/map.json. These are the PURE consumers used everywhere downstream so that no
code silently guesses the mapping. Special/control talker ids get codec_id = -1 and receive
NO codec watermark score (phi = 0), while KEEPING their probability mass in normalization.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ORDINARY = "ordinary_codec"
SPECIAL_CLASSES = {"eos", "bos", "pad", "control", "reserved", "unreachable", "unknown"}


@dataclass
class TalkerCodecMap:
    talker_vocab: int
    codec_vocab: int
    codec_id: np.ndarray      # [talker_vocab] int; -1 for special/control
    token_class: list         # [talker_vocab] str

    def ordinary_mask(self) -> np.ndarray:
        return self.codec_id >= 0

    def map_id(self, talker_id: int) -> int:
        return int(self.codec_id[talker_id])

    def is_ordinary(self, talker_id: int) -> bool:
        return self.codec_id[talker_id] >= 0

    def phi_scores(self, support_ids: np.ndarray, codec_feature: np.ndarray) -> np.ndarray:
        """Candidate scores phi(v) = feature[m(v)] for ordinary v, 0 for special/control.

        support_ids: [K] talker candidate ids at a frame.
        codec_feature: [codec_vocab] precomputed scalar per codec id (e.g. u^T z_codec).
        """
        support_ids = np.asarray(support_ids).ravel()
        out = np.zeros(support_ids.shape, dtype=np.float64)
        cids = self.codec_id[support_ids]
        ok = cids >= 0
        out[ok] = codec_feature[cids[ok]]
        return out

    @classmethod
    def identity_with_specials(cls, talker_vocab: int, codec_vocab: int,
                               special_classes: dict | None = None) -> "TalkerCodecMap":
        """Build the map proven by talker.forward: codec_ids = cat(input_ids, predictor…),
        so ordinary talker id v in [0, codec_vocab) maps to codec id v (identity), and every
        id >= codec_vocab is special/control (codec_id = -1).

        special_classes: {talker_id: class_name} for known specials (e.g. {2150: "eos"});
        remaining ids >= codec_vocab default to "reserved".
        """
        special_classes = special_classes or {}
        cid = np.full(talker_vocab, -1, dtype=np.int64)
        cls_list = ["unknown"] * talker_vocab
        for v in range(talker_vocab):
            if v < codec_vocab:
                cid[v] = v
                cls_list[v] = ORDINARY
            else:
                cls_list[v] = special_classes.get(v, "reserved")
        for v, name in special_classes.items():
            if v < talker_vocab:
                cls_list[v] = name
                cid[v] = -1
        return cls(talker_vocab=talker_vocab, codec_vocab=codec_vocab,
                   codec_id=cid, token_class=cls_list)

    @classmethod
    def from_json(cls, obj: dict) -> "TalkerCodecMap":
        tv = int(obj["talker_vocab"]); cv = int(obj["codec_vocab"])
        cid = np.full(tv, -1, dtype=np.int64)
        cls_list = ["unknown"] * tv
        for row in obj["entries"]:
            t = int(row["talker_id"])
            cid[t] = int(row["codec_id"]) if row["token_class"] == ORDINARY else -1
            cls_list[t] = row["token_class"]
        return cls(talker_vocab=tv, codec_vocab=cv, codec_id=cid, token_class=cls_list)

    def to_json(self) -> dict:
        return {"talker_vocab": self.talker_vocab, "codec_vocab": self.codec_vocab,
                "entries": [{"talker_id": i, "codec_id": int(self.codec_id[i]),
                             "token_class": self.token_class[i]} for i in range(self.talker_vocab)]}
