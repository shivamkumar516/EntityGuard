"""EntityGuard candidate generation (Student 2 component).

This is the canonical blocking implementation used by the application pipeline.
It supports both batch candidate generation and indexed, single-record lookup.

Public interfaces:
    BlockingIndex(dblp, scholar)  # reusable index used by src.pipeline.EntityGuard
    candidate_pairs(dblp, scholar) -> set[(dblp_id, scholar_id)]
    run_blocking(...)             # candidates, per-pass candidates, timings, stats

The four blocking passes are unioned. Duplicate pairs appear once, while
``generate_candidate_details`` preserves every rule that generated each pair.
"""
from __future__ import annotations

import math
import time
from collections import Counter, defaultdict

import numpy as np

from src.matching.features import add_normalized_columns

PASS_NAMES = ("title_prefix", "rare_tokens", "char_prefix", "rare_char6")
PASS_CAPS = {"title_prefix": None, "rare_tokens": 1000, "char_prefix": None, "rare_char6": 500}
PASS_LABELS = {
    "title_prefix": "Title prefix (first significant title token)",
    "rare_tokens": "Rare title tokens (3 rarest)",
    "char_prefix": "Character prefix (first 3 chars of first 2 tokens)",
    "rare_char6": "Rare character 6-grams (title without spaces)",
}


def _normalized_copy(frame):
    """Return a normalized copy; pipeline frames are already normalized."""
    required = {"id", "title"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns: {sorted(missing)}")
    result = frame.copy()
    if not {"t_tokens", "t_squash"}.issubset(result.columns):
        for column in ("authors", "venue", "year"):
            if column not in result:
                result[column] = None
        add_normalized_columns(result)
    return result


def _grams6(squashed):
    text = "" if squashed is None else str(squashed)
    return {text[i:i + 6] for i in range(len(text) - 5)}


class BlockingIndex:
    """Reusable multi-pass index for DBLP × Scholar candidate generation.

    The interface intentionally matches the former class in ``src.pipeline``:
    ``record_keys``, ``candidates``, ``candidates_for_scholar``, ``pair_check``
    and ``run_all`` are retained for application search and explanations.
    """

    def __init__(self, dblp, scholar):
        self.dblp = _normalized_copy(dblp)
        self.scholar = _normalized_copy(scholar)
        d_tok = self.dblp["t_tokens"].tolist()
        s_tok = self.scholar["t_tokens"].tolist()
        d_sq = self.dblp["t_squash"].tolist()
        s_sq = self.scholar["t_squash"].tolist()
        self.n_d, self.n_s = len(d_tok), len(s_tok)
        self.n_docs = self.n_d + self.n_s
        self.tok_df = Counter(t for toks in d_tok + s_tok for t in set(toks))
        self.gram_df = Counter(g for sq in d_sq + s_sq for g in _grams6(sq))
        self.d_keys = {p: [self.keys(p, t, sq) for t, sq in zip(d_tok, d_sq)] for p in PASS_NAMES}
        self.s_keys = {p: [self.keys(p, t, sq) for t, sq in zip(s_tok, s_sq)] for p in PASS_NAMES}
        self.index = {p: self._invert(self.s_keys[p]) for p in PASS_NAMES}
        self.d_index = {p: self._invert(self.d_keys[p]) for p in PASS_NAMES}

    @staticmethod
    def _invert(key_lists):
        blocks = defaultdict(list)
        for row, keys in enumerate(key_lists):
            for key in keys:
                blocks[key].append(row)
        return {key: np.asarray(rows, dtype=np.int64) for key, rows in blocks.items()}

    def rarity(self, token, external=False):
        extra = int(external)
        return math.log((self.n_docs + extra) / (self.tok_df.get(token, 0) + extra + 1))

    def keys(self, pass_name, tokens, squashed, external=False):
        if pass_name == "title_prefix":
            keys = tokens[:1]
        elif pass_name == "rare_tokens":
            keys = sorted(set(tokens), key=lambda token: (-self.rarity(token, external), token))[:3]
        elif pass_name == "char_prefix":
            keys = ["".join(token[:3] for token in tokens[:2])] if tokens else []
        elif pass_name == "rare_char6":
            extra = int(external)
            grams = [g for g in _grams6(squashed) if self.gram_df.get(g, 0) + extra >= 2]
            keys = sorted(grams, key=lambda gram: (self.gram_df.get(gram, 0) + extra, gram))[:5]
        else:
            raise ValueError(f"unknown blocking pass {pass_name!r}; choose from {PASS_NAMES}")
        return sorted(set(keys))

    def record_keys(self, record, external=True):
        return {name: self.keys(name, record["t_tokens"], record["t_squash"], external)
                for name in PASS_NAMES}

    def _block(self, pass_name, key, extra=0):
        rows = self.index[pass_name].get(key)
        if rows is None:
            return None
        cap = PASS_CAPS[pass_name]
        return None if cap and len(rows) + extra > cap else rows

    def candidates(self, keys_by_pass, passes=PASS_NAMES):
        found = defaultdict(list)
        for pass_name in passes:
            rows = set()
            for key in keys_by_pass[pass_name]:
                block = self._block(pass_name, key)
                if block is not None:
                    rows.update(block.tolist())
            for row in rows:
                found[row].append(pass_name)
        return dict(found)

    def candidates_for_scholar(self, keys_by_pass, passes=PASS_NAMES, external=False):
        found = defaultdict(list)
        for pass_name in passes:
            rows = set()
            for key in keys_by_pass[pass_name]:
                cap = PASS_CAPS[pass_name]
                if cap and len(self.index[pass_name].get(key, ())) + int(external) > cap:
                    continue
                block = self.d_index[pass_name].get(key)
                if block is not None:
                    rows.update(block.tolist())
            for row in rows:
                found[row].append(pass_name)
        return dict(found)

    def pair_check(self, d_keys, s_keys, s_external=False):
        result = {}
        for pass_name in PASS_NAMES:
            shared = set(d_keys[pass_name]) & set(s_keys[pass_name])
            valid = []
            for key in sorted(shared):
                size = len(self.index[pass_name].get(key, ())) + int(s_external)
                if not PASS_CAPS[pass_name] or size <= PASS_CAPS[pass_name]:
                    valid.append(key)
            result[pass_name] = valid
        return result

    def run_all(self, passes=PASS_NAMES):
        per_pass, timings = {}, {}
        for pass_name in passes:
            if pass_name not in PASS_NAMES:
                raise ValueError(f"unknown blocking pass {pass_name!r}; choose from {PASS_NAMES}")
            started = time.perf_counter()
            d_blocks = defaultdict(list)
            for d_row, keys in enumerate(self.d_keys[pass_name]):
                for key in keys:
                    d_blocks[key].append(d_row)
            parts = []
            for key, d_rows in d_blocks.items():
                s_rows = self._block(pass_name, key)
                if s_rows is not None:
                    parts.append(np.add.outer(np.asarray(d_rows, dtype=np.int64) * self.n_s,
                                               s_rows).ravel())
            per_pass[pass_name] = (np.unique(np.concatenate(parts)) if parts
                                   else np.array([], dtype=np.int64))
            timings[pass_name] = time.perf_counter() - started
        union = (np.unique(np.concatenate(list(per_pass.values()))) if per_pass
                 else np.array([], dtype=np.int64))
        return union, per_pass, timings


def run_blocking(dblp, scholar, passes=PASS_NAMES):
    """Return candidate pairs, per-pass pair sets, timings and summary statistics."""
    index = BlockingIndex(dblp, scholar)
    union_keys, pass_keys, timings = index.run_all(passes)
    d_ids = index.dblp["id"].astype(str).to_numpy()
    s_ids = index.scholar["id"].astype(str).to_numpy()

    def decode(keys):
        if not len(keys):
            return set()
        d_rows, s_rows = keys // index.n_s, keys % index.n_s
        return set(zip(d_ids[d_rows].tolist(), s_ids[s_rows].tolist()))

    per_pass = {name: decode(keys) for name, keys in pass_keys.items()}
    union = decode(union_keys)
    possible = index.n_d * index.n_s
    stats = {
        "dblp_records": index.n_d,
        "scholar_records": index.n_s,
        "possible_pairs": possible,
        "candidate_pairs": len(union),
        "duplicate_rule_hits": sum(len(pairs) for pairs in per_pass.values()) - len(union),
        "pairs_by_pass": {name: len(pairs) for name, pairs in per_pass.items()},
        "reduction_ratio": 1 - len(union) / possible if possible else 0.0,
        "oversized_blocks_skipped": sum(
            1 for name in passes if PASS_CAPS[name] is not None
            for key, rows in index.index[name].items() if len(rows) > PASS_CAPS[name]
        ),
    }
    return union, per_pass, timings, stats


def generate_candidate_details(dblp, scholar, passes=PASS_NAMES):
    """Return each unique pair with every blocking pass that generated it."""
    _union, per_pass, _timings, _stats = run_blocking(dblp, scholar, passes)
    details = defaultdict(set)
    for rule, pairs in per_pass.items():
        for pair in pairs:
            details[pair].add(rule)
    return dict(details)


def candidate_pairs(dblp, scholar, passes=PASS_NAMES):
    """Pipeline-compatible interface returning {(dblp_id, scholar_id), ...}."""
    union, _per_pass, _timings, _stats = run_blocking(dblp, scholar, passes)
    return union
