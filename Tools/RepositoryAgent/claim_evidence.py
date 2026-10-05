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
from collections.abc import Mapping
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


def resolve_claim_ledger_path(
    environ: Mapping[str, str] | None = None, *, home: Path | None = None
) -> Path:
    """Select the claim-evidence ledger location without creating filesystem state."""
    environ = os.environ if environ is None else environ
    override = environ.get("EXPERTADVISOR_CLAIM_EVIDENCE_LEDGER")
    if override:
        return Path(override)
    home = Path.home() if home is None else Path(home)
    return home / "Library" / "Caches" / "ExpertAdvisor" / "RepositoryAgent" / "verified_claims.json"


CLAIM_LEDGER_PATH = resolve_claim_ledger_path()


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
        self.relationship_bundle_records: dict[str, dict] = {}
        # Symbol investigations intentionally have a separate namespace from
        # caller-proposed claims.  A controller must never be able to create a
        # claim-ledger entry that aliases a server-owned symbol explanation.
        self.symbol_investigation_records: dict[str, dict] = {}
        # Subsystem investigations have their own identity namespace.  In
        # particular, a symbol result must never be reused for a broader
        # directory-root investigation merely because some evidence overlaps.
        self.subsystem_investigation_records: dict[str, dict] = {}
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
        members = [
            {
                "file": str(item["file"]),
                "start": int(item["start"]),
                "end": int(item["end"]),
                "source_sha256": source_hash(str(item.get("excerpt", ""))),
            }
            for item in candidates
        ]
        # Bundle source ranges are a set of independently provenanced evidence,
        # not a narrative ordering supplied by the caller.  One stable order
        # prevents request-order cache aliases even for direct ledger users.
        return sorted(members, key=lambda member: (
            member["file"], member["start"], member["end"], member["source_sha256"]
        ))

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

    @classmethod
    def _relationship_bundle_identity(cls, topic_id, claim, relationships) -> dict:
        return {
            "operation": "relationship_bundle_claim",
            "topic_id": str(topic_id),
            "claim_sha256": claim_hash(claim),
            "relationships": relationships,
            **cls._verification_identity(),
        }

    @classmethod
    def _relationship_bundle_key(cls, topic_id, claim, relationships) -> str:
        canonical = json.dumps(cls._relationship_bundle_identity(topic_id, claim, relationships), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def _symbol_investigation_identity(
        cls, topic_id, topic, requested_symbol, resolved_symbol,
        semantic_request, members,
    ) -> dict:
        """Return the complete disjoint identity for symbol investigation.

        Topic text is included here because it is part of the semantic prompt.
        This is deliberately stricter than the historical claim ledger, whose
        identity contract is preserved unchanged above.
        """
        normalized_topic = normalize_claim(str(topic))
        normalized_request = normalize_claim(str(semantic_request))
        return {
            "operation": "investigate_symbol",
            "topic_id": str(topic_id),
            "topic_sha256": hashlib.sha256(normalized_topic.encode("utf-8")).hexdigest(),
            "requested_symbol": str(requested_symbol),
            "resolved_symbol": str(resolved_symbol),
            "semantic_request_sha256": hashlib.sha256(
                normalized_request.encode("utf-8")
            ).hexdigest(),
            "members": members,
            **cls._verification_identity(),
        }

    @classmethod
    def _symbol_investigation_key(cls, *args) -> str:
        canonical = json.dumps(
            cls._symbol_investigation_identity(*args),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def _subsystem_investigation_identity(
        cls, topic_id, topic, subsystem, files, semantic_request, members,
    ) -> dict:
        return {
            "operation": "investigate_subsystem",
            "topic_id": str(topic_id),
            "topic_sha256": hashlib.sha256(
                normalize_claim(str(topic)).encode("utf-8")
            ).hexdigest(),
            "subsystem": str(subsystem),
            "files": list(files),
            "semantic_request_sha256": hashlib.sha256(
                normalize_claim(str(semantic_request)).encode("utf-8")
            ).hexdigest(),
            "members": members,
            **cls._verification_identity(),
        }

    @classmethod
    def _subsystem_investigation_key(cls, *args) -> str:
        canonical = json.dumps(
            cls._subsystem_investigation_identity(*args),
            sort_keys=True, separators=(",", ":"),
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
        relationship_bundles = obj.get("relationship_bundle_records", {})
        symbol_investigations = obj.get("symbol_investigation_records", {})
        subsystem_investigations = obj.get("subsystem_investigation_records", {})
        if isinstance(records, dict):
            self.records = {str(k): v for k, v in records.items() if isinstance(v, dict)}
        if isinstance(bundles, dict):
            self.bundle_records = {str(k): v for k, v in bundles.items() if isinstance(v, dict)}
        if isinstance(relationship_bundles, dict):
            self.relationship_bundle_records = {str(k): v for k, v in relationship_bundles.items() if isinstance(v, dict)}
        if isinstance(symbol_investigations, dict):
            self.symbol_investigation_records = {
                str(k): v for k, v in symbol_investigations.items()
                if isinstance(v, dict)
            }
        if isinstance(subsystem_investigations, dict):
            self.subsystem_investigation_records = {
                str(k): v for k, v in subsystem_investigations.items()
                if isinstance(v, dict)
            }

    def _save(self) -> None:
        payload = {
            "schema_version": CLAIM_LEDGER_SCHEMA_VERSION,
            "verification_identity": self._verification_identity(),
            "records": self.records,
            "bundle_records": self.bundle_records,
            "relationship_bundle_records": self.relationship_bundle_records,
            "symbol_investigation_records": self.symbol_investigation_records,
            "subsystem_investigation_records": self.subsystem_investigation_records,
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
                "model_turns": 0,
            }
        if status != CLAIM_ACCEPTED or not str(record.get("establishes", "")).strip():
            return None
        return {
            "supports": True,
            "establishes": str(record["establishes"]).strip(),
            "reason": "reused claim-ledger decision for identical claim, source, and verifier identity",
            hit_name: True,
            "ledger_status": CLAIM_ACCEPTED,
            "model_turns": 0,
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

    def lookup_relationship_bundle(self, topic_id, claim, relationships):
        key = self._relationship_bundle_key(topic_id, claim, relationships)
        record = self.relationship_bundle_records.get(key)
        expected = self._relationship_bundle_identity(topic_id, claim, relationships)
        if not isinstance(record, dict) or not self._record_identity_matches(record):
            return None
        if any(record.get(name) != value for name, value in expected.items()):
            return None
        return self._cached(record, bundle=True)

    def record_relationship_bundle_decision(self, topic_id, claim, relationships, verdict) -> None:
        key = self._relationship_bundle_key(topic_id, claim, relationships)
        status = self._status(verdict)
        existing = self.relationship_bundle_records.get(key)
        if isinstance(existing, dict) and existing.get("status") == CLAIM_ACCEPTED and status in {CLAIM_INDETERMINATE, CLAIM_VERIFIER_ERROR}:
            return
        identity = self._relationship_bundle_identity(topic_id, claim, relationships)
        self.relationship_bundle_records[key] = {
            **identity,
            "claim": normalize_claim(claim),
            "establishes": str(verdict.get("establishes", "")).strip() if status == CLAIM_ACCEPTED else "",
            "reason": str(verdict.get("reason", "")).strip(),
            "status": status,
        }
        self._save()

    def lookup_symbol_investigation(
        self, topic_id, topic, requested_symbol, resolved_symbol,
        semantic_request, candidates,
    ):
        members = self._bundle_members(candidates)
        args = (
            topic_id, topic, requested_symbol, resolved_symbol,
            semantic_request, members,
        )
        key = self._symbol_investigation_key(*args)
        record = self.symbol_investigation_records.get(key)
        expected = self._symbol_investigation_identity(*args)
        if (
            not isinstance(record, dict)
            or not self._record_identity_matches(record)
            or any(record.get(name) != value for name, value in expected.items())
        ):
            return None
        return self._cached(record)

    def record_symbol_investigation_decision(
        self, topic_id, topic, requested_symbol, resolved_symbol,
        semantic_request, candidates, verdict,
    ) -> None:
        members = self._bundle_members(candidates)
        args = (
            topic_id, topic, requested_symbol, resolved_symbol,
            semantic_request, members,
        )
        key = self._symbol_investigation_key(*args)
        status = self._status(verdict)
        existing = self.symbol_investigation_records.get(key)
        if (
            isinstance(existing, dict)
            and existing.get("status") == CLAIM_ACCEPTED
            and status in {CLAIM_INDETERMINATE, CLAIM_VERIFIER_ERROR}
        ):
            return
        identity = self._symbol_investigation_identity(*args)
        self.symbol_investigation_records[key] = {
            **identity,
            "topic": normalize_claim(str(topic)),
            "semantic_request": normalize_claim(str(semantic_request)),
            "establishes": (
                str(verdict.get("establishes", "")).strip()
                if status == CLAIM_ACCEPTED else ""
            ),
            "reason": str(verdict.get("reason", "")).strip(),
            "status": status,
        }
        self._save()

    def lookup_subsystem_investigation(
        self, topic_id, topic, subsystem, files, semantic_request, candidates,
    ):
        members = self._bundle_members(candidates)
        args = (topic_id, topic, subsystem, files, semantic_request, members)
        key = self._subsystem_investigation_key(*args)
        record = self.subsystem_investigation_records.get(key)
        expected = self._subsystem_investigation_identity(*args)
        if (
            not isinstance(record, dict)
            or not self._record_identity_matches(record)
            or any(record.get(name) != value for name, value in expected.items())
        ):
            return None
        return self._cached(record)

    def record_subsystem_investigation_decision(
        self, topic_id, topic, subsystem, files, semantic_request, candidates, verdict,
    ) -> None:
        members = self._bundle_members(candidates)
        args = (topic_id, topic, subsystem, files, semantic_request, members)
        key = self._subsystem_investigation_key(*args)
        status = self._status(verdict)
        existing = self.subsystem_investigation_records.get(key)
        if (
            isinstance(existing, dict)
            and existing.get("status") == CLAIM_ACCEPTED
            and status in {CLAIM_INDETERMINATE, CLAIM_VERIFIER_ERROR}
        ):
            return
        identity = self._subsystem_investigation_identity(*args)
        self.subsystem_investigation_records[key] = {
            **identity,
            "topic": normalize_claim(str(topic)),
            "semantic_request": normalize_claim(str(semantic_request)),
            "establishes": str(verdict.get("establishes", "")).strip()
            if status == CLAIM_ACCEPTED else "",
            "reason": str(verdict.get("reason", "")).strip(),
            "status": status,
        }
        self._save()
