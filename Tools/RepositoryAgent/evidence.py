#!/usr/bin/env python3
"""Verified evidence ledger extracted unchanged from the frozen production agent."""
import hashlib
import json
import os
from pathlib import Path

LEDGER_SCHEMA_VERSION = 2
LEDGER_LEGACY_SCHEMA_VERSION = 1
VERIFIER_SCHEMA_VERSION = 1
VERIFIER_IDENTITY = "expertadvisor_semantic_verifier_v1"
LEDGER_CACHE_NAMESPACE = os.environ.get(
    "EXPERTADVISOR_LEDGER_CACHE_NAMESPACE", "ExpertAdvisor"
).strip() or "ExpertAdvisor"

EVIDENCE_ACCEPTED = "accepted"
EVIDENCE_REJECTED = "rejected"
EVIDENCE_INDETERMINATE = "indeterminate"
EVIDENCE_VERIFIER_ERROR = "verifier_error"

LEDGER_PATH = Path(
    os.environ.get(
        "EXPERTADVISOR_EVIDENCE_LEDGER",
        str(Path(__file__).with_name(".expertadvisor_verified_evidence.json")),
    )
)


def evidence_source_hash(excerpt):
    return hashlib.sha256(excerpt.encode("utf-8")).hexdigest()


class VerifiedEvidenceLedger:
    """Persistent semantic decisions tied to exact source and verifier identity.

    Schema 2 keeps accepted/rejected decisions reusable only for the current
    cache namespace + verifier identity/schema. INDETERMINATE and
    VERIFIER_ERROR are retained for auditability but deliberately retryable.
    Schema-1 ledgers from the immediately preceding production baseline are
    migrated in memory as decisions made by this verifier identity, preserving
    the already-verified evidence while adding explicit identity metadata.
    """

    def __init__(self, path=LEDGER_PATH):
        self.path = Path(path)
        self.records = {}
        self.bundle_records = {}
        self._loaded_legacy_schema = False
        self._load()

    @staticmethod
    def _verification_identity():
        return {
            "cache_namespace": LEDGER_CACHE_NAMESPACE,
            "verifier_identity": VERIFIER_IDENTITY,
            "verifier_schema_version": VERIFIER_SCHEMA_VERSION,
        }

    @classmethod
    def _key(cls, topic_id, category, filename, start, end, source_hash):
        canonical = json.dumps(
            {
                "topic_id": str(topic_id),
                "category": str(category),
                "file": str(filename),
                "start": int(start),
                "end": int(end),
                "source_sha256": str(source_hash),
                **cls._verification_identity(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def _record_identity_matches(cls, record):
        expected = cls._verification_identity()
        return all(record.get(k) == v for k, v in expected.items())

    @classmethod
    def _normalize_legacy_record(cls, record):
        out = dict(record)
        out.update(cls._verification_identity())
        status = str(out.get("status", EVIDENCE_ACCEPTED)).strip().lower()
        if status not in {
            EVIDENCE_ACCEPTED,
            EVIDENCE_REJECTED,
            EVIDENCE_INDETERMINATE,
            EVIDENCE_VERIFIER_ERROR,
        }:
            status = EVIDENCE_INDETERMINATE
        out["status"] = status
        return out

    def _load(self):
        try:
            obj = json.loads(self.path.read_text())
        except FileNotFoundError:
            return
        except (OSError, json.JSONDecodeError) as exc:
            print(f"LEDGER NOTICE: ignoring unreadable ledger {self.path}: {exc}")
            return

        if not isinstance(obj, dict):
            print(f"LEDGER NOTICE: ignoring incompatible ledger {self.path}")
            return

        schema = obj.get("schema_version")
        if schema not in {LEDGER_LEGACY_SCHEMA_VERSION, LEDGER_SCHEMA_VERSION}:
            print(f"LEDGER NOTICE: ignoring incompatible ledger {self.path}")
            return

        records = obj.get("records", {})
        bundles = obj.get("bundle_records", {})
        if schema == LEDGER_LEGACY_SCHEMA_VERSION:
            self._loaded_legacy_schema = True
            # Re-key schema-1 records under the explicit verifier identity.
            if isinstance(records, dict):
                for record in records.values():
                    if not isinstance(record, dict):
                        continue
                    record = self._normalize_legacy_record(record)
                    try:
                        key = self._key(
                            record["topic_id"], record["category"], record["file"],
                            record["start"], record["end"], record["source_sha256"],
                        )
                    except (KeyError, TypeError, ValueError):
                        continue
                    self.records[key] = record
            if isinstance(bundles, dict):
                for record in bundles.values():
                    if not isinstance(record, dict):
                        continue
                    record = self._normalize_legacy_record(record)
                    members = record.get("members")
                    if not isinstance(members, list):
                        continue
                    key = self._bundle_key(record.get("topic_id", ""), record.get("category", ""), members)
                    self.bundle_records[key] = record
            return

        if isinstance(records, dict):
            self.records = {str(k): v for k, v in records.items() if isinstance(v, dict)}
        if isinstance(bundles, dict):
            self.bundle_records = {str(k): v for k, v in bundles.items() if isinstance(v, dict)}

    def _save(self):
        payload = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "verification_identity": self._verification_identity(),
            "records": self.records,
            "bundle_records": self.bundle_records,
        }
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            os.replace(tmp, self.path)
            self._loaded_legacy_schema = False
        except OSError as exc:
            print(f"LEDGER NOTICE: could not persist ledger {self.path}: {exc}")
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    def lookup(self, topic_id, category, filename, start, end, excerpt):
        source_hash = evidence_source_hash(excerpt)
        key = self._key(topic_id, category, filename, start, end, source_hash)
        record = self.records.get(key)
        if not isinstance(record, dict) or not self._record_identity_matches(record):
            return None
        status = str(record.get("status", "")).strip().lower()
        # Errors and indeterminate results are audit records, never cache hits.
        if status in {EVIDENCE_INDETERMINATE, EVIDENCE_VERIFIER_ERROR}:
            return None
        if status == EVIDENCE_REJECTED:
            return {
                "supports": False,
                "establishes": "",
                "reason": str(record.get("reason", "reused semantic rejection")),
                "ledger_hit": True,
                "ledger_status": EVIDENCE_REJECTED,
            }
        if status != EVIDENCE_ACCEPTED:
            return None
        establishes = str(record.get("establishes", "")).strip()
        if not establishes:
            return None
        return {
            "supports": True,
            "establishes": establishes,
            "reason": "reused controller-ledger acceptance for identical source and verifier identity",
            "ledger_hit": True,
            "ledger_status": EVIDENCE_ACCEPTED,
        }

    def record_decision(self, topic_id, category, filename, start, end, excerpt, verdict):
        source_hash = evidence_source_hash(excerpt)
        key = self._key(topic_id, category, filename, start, end, source_hash)
        if verdict.get("verifier_error"):
            status = EVIDENCE_VERIFIER_ERROR
        elif verdict.get("supports") is True and str(verdict.get("establishes", "")).strip():
            status = EVIDENCE_ACCEPTED
        elif verdict.get("indeterminate"):
            status = EVIDENCE_INDETERMINATE
        else:
            status = EVIDENCE_REJECTED
        # A retryable failure must never overwrite a previously accepted
        # decision for the same exact source/verifier identity.
        existing = self.records.get(key)
        if (
            isinstance(existing, dict)
            and existing.get("status") == EVIDENCE_ACCEPTED
            and status in {EVIDENCE_INDETERMINATE, EVIDENCE_VERIFIER_ERROR}
        ):
            return
        self.records[key] = {
            "topic_id": str(topic_id),
            "category": str(category),
            "file": str(filename),
            "start": int(start),
            "end": int(end),
            "source_sha256": source_hash,
            "establishes": str(verdict.get("establishes", "")).strip() if status == EVIDENCE_ACCEPTED else "",
            "reason": str(verdict.get("reason", "")).strip(),
            "status": status,
            **self._verification_identity(),
        }
        self._save()

    def accept(self, topic_id, category, filename, start, end, excerpt, establishes):
        self.record_decision(
            topic_id, category, filename, start, end, excerpt,
            {"supports": True, "establishes": establishes, "reason": "semantic verifier accepted"},
        )

    @staticmethod
    def _bundle_members(candidates):
        members = []
        for item in candidates:
            excerpt = str(item.get("excerpt", ""))
            members.append({
                "file": str(item["file"]),
                "start": int(item["start"]),
                "end": int(item["end"]),
                "source_sha256": evidence_source_hash(excerpt),
            })
        return members

    @classmethod
    def _bundle_key(cls, topic_id, category, members):
        canonical = json.dumps(
            {
                "topic_id": str(topic_id),
                "category": str(category),
                "members": members,
                **cls._verification_identity(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def lookup_bundle(self, topic_id, category, candidates):
        members = self._bundle_members(candidates)
        key = self._bundle_key(topic_id, category, members)
        record = self.bundle_records.get(key)
        if not isinstance(record, dict) or not self._record_identity_matches(record):
            return None
        if record.get("members") != members:
            return None
        status = str(record.get("status", "")).strip().lower()
        if status in {EVIDENCE_INDETERMINATE, EVIDENCE_VERIFIER_ERROR}:
            return None
        if status == EVIDENCE_REJECTED:
            return {
                "supports": False,
                "establishes": "",
                "reason": str(record.get("reason", "reused bundle semantic rejection")),
                "bundle_ledger_hit": True,
                "ledger_status": EVIDENCE_REJECTED,
            }
        if status != EVIDENCE_ACCEPTED:
            return None
        establishes = str(record.get("establishes", "")).strip()
        if not establishes:
            return None
        return {
            "supports": True,
            "establishes": establishes,
            "reason": "reused controller-ledger bundle acceptance for identical source and verifier identity",
            "bundle_ledger_hit": True,
            "ledger_status": EVIDENCE_ACCEPTED,
        }

    def record_bundle_decision(self, topic_id, category, candidates, verdict):
        members = self._bundle_members(candidates)
        key = self._bundle_key(topic_id, category, members)
        if verdict.get("verifier_error"):
            status = EVIDENCE_VERIFIER_ERROR
        elif verdict.get("supports") is True and str(verdict.get("establishes", "")).strip():
            status = EVIDENCE_ACCEPTED
        elif verdict.get("indeterminate"):
            status = EVIDENCE_INDETERMINATE
        else:
            status = EVIDENCE_REJECTED
        existing = self.bundle_records.get(key)
        if (
            isinstance(existing, dict)
            and existing.get("status") == EVIDENCE_ACCEPTED
            and status in {EVIDENCE_INDETERMINATE, EVIDENCE_VERIFIER_ERROR}
        ):
            return
        self.bundle_records[key] = {
            "topic_id": str(topic_id),
            "category": str(category),
            "members": members,
            "establishes": str(verdict.get("establishes", "")).strip() if status == EVIDENCE_ACCEPTED else "",
            "reason": str(verdict.get("reason", "")).strip(),
            "status": status,
            **self._verification_identity(),
        }
        self._save()

    def accept_bundle(self, topic_id, category, candidates, establishes):
        self.record_bundle_decision(
            topic_id, category, candidates,
            {"supports": True, "establishes": establishes, "reason": "bundle semantic verifier accepted"},
        )
