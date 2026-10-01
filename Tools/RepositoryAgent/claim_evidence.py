#!/usr/bin/env python3
"""Claim-aware verified evidence ledger for Codex-directed investigations.

This ledger is deliberately separate from evidence.VerifiedEvidenceLedger.  The
production RepositoryAgent ledger remains category-keyed and unchanged; this
ledger keys decisions by the exact proposed claim as well as exact source.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

CLAIM_LEDGER_SCHEMA_VERSION = 1
CLAIM_VERIFIER_SCHEMA_VERSION = 1
CLAIM_VERIFIER_IDENTITY = "expertadvisor_claim_verifier_v1"
CLAIM_LEDGER_CACHE_NAMESPACE = os.environ.get(
    "EXPERTADVISOR_LEDGER_CACHE_NAMESPACE", "ExpertAdvisor"
).strip() or "ExpertAdvisor"

CLAIM_ACCEPTED = "accepted"
CLAIM_REJECTED = "rejected"
CLAIM_INDETERMINATE = "indeterminate"
CLAIM_VERIFIER_ERROR = "verifier_error"

CLAIM_LEDGER_PATH = Path(
    os.environ.get(
        "EXPERTADVISOR_CLAIM_EVIDENCE_LEDGER",
        str(Path(__file__).with_name(".expertadvisor_verified_claims.json")),
    )
)


def normalize_claim(claim: str) -> str:
    """Conservatively normalize only whitespace; preserve case and wording."""
    return re.sub(r"\s+", " ", str(claim).strip())


def claim_hash(claim: str) -> str:
    return hashlib.sha256(normalize_claim(claim).encode("utf-8")).hexdigest()


def source_hash(excerpt: str) -> str:
    return hashlib.sha256(str(excerpt).encode("utf-8")).hexdigest()


class VerifiedClaimLedger:
    """Persistent claim decisions tied to exact claim, source, and verifier identity."""

    def __init__(self, path=CLAIM_LEDGER_PATH):
        self.path = Path(path)
        self.records: dict[str, dict] = {}
        self.bundle_records: dict[str, dict] = {}
        self._load()

    @staticmethod
    def _verification_identity() -> dict:
        return {
            "cache_namespace": CLAIM_LEDGER_CACHE_NAMESPACE,
            "verifier_identity": CLAIM_VERIFIER_IDENTITY,
            "verifier_schema_version": CLAIM_VERIFIER_SCHEMA_VERSION,
        }

    @classmethod
    def _record_identity_matches(cls, record: dict) -> bool:
        expected = cls._verification_identity()
        return all(record.get(k) == v for k, v in expected.items())

    @classmethod
    def _key(cls, topic_id, claim, filename, start, end, excerpt) -> str:
        canonical = json.dumps(
            {
                "topic_id": str(topic_id),
                "claim_sha256": claim_hash(claim),
                "file": str(filename),
                "start": int(start),
                "end": int(end),
                "source_sha256": source_hash(excerpt),
                **cls._verification_identity(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _bundle_members(candidates) -> list[dict]:
        return [
            {
                "file": str(item["file"]),
                "start": int(item["start"]),
                "end": int(item["end"]),
                "source_sha256": source_hash(str(item.get("excerpt", ""))),
            }
            for item in candidates
        ]

    @classmethod
    def _bundle_key(cls, topic_id, claim, members) -> str:
        canonical = json.dumps(
            {
                "topic_id": str(topic_id),
                "claim_sha256": claim_hash(claim),
                "members": members,
                **cls._verification_identity(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _load(self) -> None:
        try:
            obj = json.loads(self.path.read_text())
        except FileNotFoundError:
            return
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(obj, dict) or obj.get("schema_version") != CLAIM_LEDGER_SCHEMA_VERSION:
            return
        records = obj.get("records", {})
        bundles = obj.get("bundle_records", {})
        if isinstance(records, dict):
            self.records = {str(k): v for k, v in records.items() if isinstance(v, dict)}
        if isinstance(bundles, dict):
            self.bundle_records = {str(k): v for k, v in bundles.items() if isinstance(v, dict)}

    def _save(self) -> None:
        payload = {
            "schema_version": CLAIM_LEDGER_SCHEMA_VERSION,
            "verification_identity": self._verification_identity(),
            "records": self.records,
            "bundle_records": self.bundle_records,
        }
        tmp = self.path.with_name(self.path.name + ".tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            os.replace(tmp, self.path)
        except OSError:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    @staticmethod
    def _status(verdict: dict) -> str:
        if verdict.get("verifier_error"):
            return CLAIM_VERIFIER_ERROR
        if verdict.get("supports") is True and str(verdict.get("establishes", "")).strip():
            return CLAIM_ACCEPTED
        if verdict.get("indeterminate"):
            return CLAIM_INDETERMINATE
        return CLAIM_REJECTED

    @staticmethod
    def _cached(record: dict, *, bundle=False):
        status = str(record.get("status", "")).strip().lower()
        if status in {CLAIM_INDETERMINATE, CLAIM_VERIFIER_ERROR}:
            return None
        hit_name = "bundle_ledger_hit" if bundle else "ledger_hit"
        if status == CLAIM_REJECTED:
            return {
                "supports": False,
                "establishes": "",
                "reason": str(record.get("reason", "reused claim rejection")),
                hit_name: True,
                "ledger_status": CLAIM_REJECTED,
            }
        if status != CLAIM_ACCEPTED or not str(record.get("establishes", "")).strip():
            return None
        return {
            "supports": True,
            "establishes": str(record["establishes"]).strip(),
            "reason": "reused claim-ledger decision for identical claim, source, and verifier identity",
            hit_name: True,
            "ledger_status": CLAIM_ACCEPTED,
        }

    def lookup(self, topic_id, claim, filename, start, end, excerpt):
        key = self._key(topic_id, claim, filename, start, end, excerpt)
        record = self.records.get(key)
        if not isinstance(record, dict) or not self._record_identity_matches(record):
            return None
        return self._cached(record)

    def record_decision(self, topic_id, claim, filename, start, end, excerpt, verdict) -> None:
        key = self._key(topic_id, claim, filename, start, end, excerpt)
        status = self._status(verdict)
        existing = self.records.get(key)
        if (
            isinstance(existing, dict)
            and existing.get("status") == CLAIM_ACCEPTED
            and status in {CLAIM_INDETERMINATE, CLAIM_VERIFIER_ERROR}
        ):
            return
        self.records[key] = {
            "topic_id": str(topic_id),
            "claim": normalize_claim(claim),
            "claim_sha256": claim_hash(claim),
            "file": str(filename),
            "start": int(start),
            "end": int(end),
            "source_sha256": source_hash(excerpt),
            "establishes": str(verdict.get("establishes", "")).strip() if status == CLAIM_ACCEPTED else "",
            "reason": str(verdict.get("reason", "")).strip(),
            "status": status,
            **self._verification_identity(),
        }
        self._save()

    def lookup_bundle(self, topic_id, claim, candidates):
        members = self._bundle_members(candidates)
        key = self._bundle_key(topic_id, claim, members)
        record = self.bundle_records.get(key)
        if (
            not isinstance(record, dict)
            or not self._record_identity_matches(record)
            or record.get("members") != members
        ):
            return None
        return self._cached(record, bundle=True)

    def record_bundle_decision(self, topic_id, claim, candidates, verdict) -> None:
        members = self._bundle_members(candidates)
        key = self._bundle_key(topic_id, claim, members)
        status = self._status(verdict)
        existing = self.bundle_records.get(key)
        if (
            isinstance(existing, dict)
            and existing.get("status") == CLAIM_ACCEPTED
            and status in {CLAIM_INDETERMINATE, CLAIM_VERIFIER_ERROR}
        ):
            return
        self.bundle_records[key] = {
            "topic_id": str(topic_id),
            "claim": normalize_claim(claim),
            "claim_sha256": claim_hash(claim),
            "members": members,
            "establishes": str(verdict.get("establishes", "")).strip() if status == CLAIM_ACCEPTED else "",
            "reason": str(verdict.get("reason", "")).strip(),
            "status": status,
            **self._verification_identity(),
        }
        self._save()
