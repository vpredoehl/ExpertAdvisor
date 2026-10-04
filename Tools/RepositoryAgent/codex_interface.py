#!/usr/bin/env python3
"""Repository-read-only machine interface for external controllers.

Repository source remains read-only. Claim verification may persist decisions only
in the dedicated claim-evidence ledger; it cannot write repository source, invoke
a shell, access the database, run builds/tests, or mutate Git.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from expertadvisor_agent import list_files, read_file, search
from .claim_evidence import VerifiedClaimLedger, source_hash
from .claim_verifier import LazyClaimVerifierRuntime, MAX_BUNDLE_RANGES
from .evidence import VerifiedEvidenceLedger
from .repository_index import RepositoryIndex


RELATIONSHIP_CONTEXT_RADIUS = 12


class CodexRepositoryInterface:
    def __init__(self, *, ledger_path: str | None = None, claim_ledger_path: str | None = None,
                 claim_runtime: LazyClaimVerifierRuntime | None = None):
        self._index: RepositoryIndex | None = None
        self._ledger_path = ledger_path
        self._claim_ledger_path = claim_ledger_path
        self._claim_runtime = claim_runtime

    def _idx(self) -> RepositoryIndex:
        if self._index is None:
            self._index = RepositoryIndex().build()
        return self._index

    def _claim_ledger(self) -> VerifiedClaimLedger:
        return VerifiedClaimLedger(Path(self._claim_ledger_path)) if self._claim_ledger_path else VerifiedClaimLedger()

    def _claim_verifier(self) -> LazyClaimVerifierRuntime:
        if self._claim_runtime is None:
            self._claim_runtime = LazyClaimVerifierRuntime()
        return self._claim_runtime

    def capabilities(self) -> dict[str, Any]:
        return {
            "protocol": "expertadvisor.repository.readonly.v1",
            "read_only": True,
            "repository_read_only": True,
            "controlled_state_writes": ["claim_evidence_ledger"],
            "operations": [
                "capabilities", "list_files", "search", "read", "index_stats",
                "resolve_symbol", "function_for_line", "relationship",
                "trace_calls", "source_excerpt", "ledger_records",
                "verify_source_claim", "verify_source_bundle_claim", "verified_claims",
                "investigate_source_claim", "investigate_source_bundle_claim",
                "investigate_relationship_claim",
            ],
            "forbidden_capabilities": [
                "shell", "repository_write", "git_mutation", "database",
                "build", "test_execution", "arbitrary_filesystem",
            ],
        }

    @staticmethod
    def _bounded_int(value: Any, *, default: int, low: int, high: int) -> int:
        try: value = int(value)
        except (TypeError, ValueError): value = default
        return max(low, min(high, value))

    @staticmethod
    def _required_text(request: dict[str, Any], name: str) -> str:
        value = str(request.get(name, "")).strip()
        if not value:
            raise ValueError(f"{name} is required")
        return value

    @staticmethod
    def _required_string(request: dict[str, Any], name: str) -> str:
        value = request.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        return value.strip()

    @staticmethod
    def _require_exact_fields(request: dict[str, Any], allowed: set[str]) -> None:
        unexpected = sorted(set(request) - allowed)
        if unexpected:
            raise ValueError(f"unexpected request fields: {', '.join(unexpected)}")

    @staticmethod
    def _edge_field(edge: Any, name: str) -> Any:
        if isinstance(edge, dict):
            return edge.get(name)
        return getattr(edge, name, None)

    def _claim_item(self, spec: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(spec, dict):
            raise ValueError("source range must be an object")
        file = self._required_text(spec, "file")
        path = Path(file)
        if (
            path.is_absolute() or path.as_posix() != file or "\x00" in file or "\\" in file
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("claim source file must be a repository-relative normalized path")
        try:
            raw_start = spec.get("start", 0)
            raw_end = spec.get("end", 0)
            if isinstance(raw_start, bool) or isinstance(raw_end, bool):
                raise ValueError("claim source range must use integer line bounds")
            start = int(raw_start)
            end = int(raw_end)
        except (TypeError, ValueError) as exc:
            raise ValueError("claim source range must use integer line bounds") from exc
        if start < 1 or end < start or end - start + 1 > 500:
            raise ValueError("claim source range must be 1..500 lines")
        # Retrieve source server-side. The caller cannot inject excerpt text into
        # verification.  Do not initialize the repository-wide structural index
        # simply to obtain one bounded target range.
        try:
            excerpt = read_file(file, start, end)
        except Exception as exc:
            raise ValueError("claim source range could not be read") from exc
        if not isinstance(excerpt, str) or not excerpt.strip():
            raise ValueError("claim source range returned no verifiable source")
        return {"file": file, "start": start, "end": end, "excerpt": excerpt}

    @staticmethod
    def _validate_verdict(verdict: Any) -> dict[str, Any]:
        """Accept only a complete local-verifier state; fail closed otherwise."""
        def rejected(reason: str, turns: int = 0) -> dict[str, Any]:
            return {"supports": False, "establishes": "", "reason": reason,
                    "verifier_error": True, "model_turns": turns}

        if not isinstance(verdict, dict):
            return rejected("claim verifier returned malformed verdict")
        turns = verdict.get("model_turns", 0)
        if isinstance(turns, bool) or not isinstance(turns, int) or not 0 <= turns <= 2:
            return rejected("claim verifier returned invalid model-turn count")
        supports = verdict.get("supports")
        establishes = verdict.get("establishes", "")
        reason = verdict.get("reason", "")
        if not isinstance(supports, bool) or not isinstance(establishes, str) or not isinstance(reason, str):
            return rejected("claim verifier returned malformed verdict", turns)
        if supports and not establishes.strip():
            return rejected("claim verifier returned contradictory supported verdict", turns)
        if not supports and establishes.strip():
            return rejected("claim verifier returned contradictory rejected verdict", turns)
        normalized = {"supports": supports, "establishes": establishes.strip() if supports else "",
                      "reason": reason.strip(), "model_turns": turns}
        if verdict.get("verifier_error") is True:
            normalized["verifier_error"] = True
        return normalized

    def _claim_bundle_items(self, ranges: Any) -> tuple[list[dict[str, Any]], int]:
        """Read an explicit bounded bundle in one stable, unambiguous order."""
        if not isinstance(ranges, list) or not 2 <= len(ranges) <= MAX_BUNDLE_RANGES:
            raise ValueError(f"ranges must contain 2-{MAX_BUNDLE_RANGES} source ranges")
        items = [self._claim_item(spec) for spec in ranges]
        items.sort(key=lambda item: (item["file"], item["start"], item["end"]))
        previous_by_file: dict[str, dict[str, Any]] = {}
        for item in items:
            previous = previous_by_file.get(item["file"])
            if previous is not None:
                if item["start"] == previous["start"] and item["end"] == previous["end"]:
                    raise ValueError("claim source bundle contains a duplicate range")
                if item["start"] <= previous["end"]:
                    raise ValueError("claim source bundle contains overlapping ranges")
            previous_by_file[item["file"]] = item
        return items, len(ranges)

    @staticmethod
    def _render_claim_answer(verdict: dict[str, Any]) -> str:
        if verdict.get("supports") is True:
            return str(verdict.get("establishes", "")).strip()
        reason = str(verdict.get("reason", "claim not established by supplied source")).strip()
        return f"Not established by the supplied source: {reason}"

    def dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise ValueError("request must be a JSON object")
        op = str(request.get("op", "")).strip()
        if op == "capabilities": return self.capabilities()
        if op == "list_files":
            prefix = str(request.get("prefix", ""))
            return {"files": RepositoryIndex._normalize_listing(list_files(prefix))}
        if op == "search":
            pattern = str(request.get("pattern", ""))
            if not pattern: raise ValueError("search requires non-empty pattern")
            limit = self._bounded_int(request.get("limit"), default=50, low=1, high=100)
            raw = search(pattern, max_results=limit)
            return {"pattern": pattern, "result": raw}
        if op == "read":
            file = str(request.get("file", ""))
            start = self._bounded_int(request.get("start"), default=1, low=1, high=10_000_000)
            end = self._bounded_int(request.get("end"), default=start + 199, low=start, high=start + 499)
            return {"file": file, "start": start, "end": end, "content": read_file(file, start, end)}
        if op == "index_stats": return self._idx().stats()
        if op == "resolve_symbol":
            symbol = str(request.get("symbol", "")).strip()
            if not symbol: raise ValueError("resolve_symbol requires symbol")
            return self._idx().resolve_symbol(symbol)
        if op == "function_for_line":
            file = str(request.get("file", "")); line = int(request.get("line", 0))
            fn = self._idx().function_for_line(file, line)
            return {"function": asdict(fn) if fn else None}
        if op == "relationship":
            caller = str(request.get("caller", "")).strip(); callee = str(request.get("callee", "")).strip()
            if not caller or not callee: raise ValueError("relationship requires caller and callee")
            return {"edges": [asdict(x) for x in self._idx().relationship(caller, callee)]}
        if op == "trace_calls":
            symbol = str(request.get("symbol", "")).strip()
            if not symbol: raise ValueError("trace_calls requires symbol")
            depth = self._bounded_int(request.get("max_depth"), default=2, low=1, high=4)
            nodes = self._bounded_int(request.get("max_nodes"), default=50, low=1, high=100)
            return {"trace": self._idx().trace_calls(symbol, max_depth=depth, max_nodes=nodes)}
        if op == "source_excerpt":
            file = str(request.get("file", "")); start = int(request.get("start", 1)); end = int(request.get("end", start))
            if end < start or end - start + 1 > 500: raise ValueError("source_excerpt range must be 1..500 lines")
            return {"file": file, "start": start, "end": end, "content": self._idx().source_excerpt(file, start, end)}
        if op == "ledger_records": return self._ledger_records(request)
        if op == "verify_source_claim": return self._verify_source_claim(request)
        if op == "verify_source_bundle_claim": return self._verify_source_bundle_claim(request)
        if op == "verified_claims": return self._verified_claims(request)
        if op == "investigate_source_claim": return self._investigate_source_claim(request)
        if op == "investigate_source_bundle_claim": return self._investigate_source_bundle_claim(request)
        if op == "investigate_relationship_claim": return self._investigate_relationship_claim(request)
        raise ValueError(f"unsupported operation: {op!r}")

    def _verify_source_claim_result(self, request: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        topic_id = self._required_text(request, "topic_id")
        topic = self._required_text(request, "topic")
        claim = self._required_text(request, "claim")
        item = self._claim_item(request)
        ledger = self._claim_ledger()
        cached = ledger.lookup(topic_id, claim, item["file"], item["start"], item["end"], item["excerpt"])
        if cached is not None:
            return ({"topic_id": topic_id, "claim": claim, "evidence": {k: item[k] for k in ("file","start","end")}, "verdict": cached}, item)
        verdict = self._validate_verdict(self._claim_verifier().verify_claim(topic, claim, item))
        ledger.record_decision(topic_id, claim, item["file"], item["start"], item["end"], item["excerpt"], verdict)
        return ({"topic_id": topic_id, "claim": claim, "evidence": {k: item[k] for k in ("file","start","end")}, "verdict": verdict}, item)

    def _verify_source_claim(self, request: dict[str, Any]) -> dict[str, Any]:
        result, _ = self._verify_source_claim_result(request)
        return result

    def _investigate_source_claim(self, request: dict[str, Any]) -> dict[str, Any]:
        """Run one bounded source-claim investigation without topic fan-out.

        The controller retrieves exactly one caller-specified, repository-allowed
        range server-side, then uses the existing claim verifier and ledger.  The
        rendered answer is deterministic from the verifier verdict; this mode does
        not ask Qwen to plan, browse, or synthesize unrelated architecture topics.
        """
        result, item = self._verify_source_claim_result(request)
        verdict = result["verdict"]
        supports = verdict.get("supports") is True
        manifest = {
            "schema_version": 1,
            "mode": "targeted_source_claim",
            "topic_id": result["topic_id"],
            "topic": self._required_text(request, "topic"),
            "claim": result["claim"],
            "evidence": {
                "file": item["file"],
                "start": item["start"],
                "end": item["end"],
                "source_sha256": source_hash(item["excerpt"]),
            },
            "verification": {
                "supports": supports,
                "ledger_hit": verdict.get("ledger_hit") is True,
                "model_turns": int(verdict.get("model_turns", 0)),
            },
            "metrics": {
                "requested_range_count": 1,
                "effective_range_count": 1,
                "repository_read_count": 1,
                "unrelated_topic_count": 0,
                "model_turn_count": int(verdict.get("model_turns", 0)),
                "ledger_hit": verdict.get("ledger_hit") is True,
            },
        }
        return {
            "manifest": manifest,
            "answer": self._render_claim_answer(verdict),
            "verdict": verdict,
        }

    def _verify_source_bundle_claim_result(self, request: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
        topic_id = self._required_text(request, "topic_id")
        topic = self._required_text(request, "topic")
        claim = self._required_text(request, "claim")
        items, requested_count = self._claim_bundle_items(request.get("ranges"))
        ledger = self._claim_ledger()
        cached = ledger.lookup_bundle(topic_id, claim, items)
        if cached is not None:
            verdict = cached
        else:
            verdict = self._validate_verdict(self._claim_verifier().verify_bundle_claim(topic, claim, items))
            ledger.record_bundle_decision(topic_id, claim, items, verdict)
        return ({
            "topic_id": topic_id, "claim": claim,
            "evidence": [{k: item[k] for k in ("file","start","end")} for item in items],
            "verdict": verdict,
        }, items, requested_count)

    def _verify_source_bundle_claim(self, request: dict[str, Any]) -> dict[str, Any]:
        result, _, _ = self._verify_source_bundle_claim_result(request)
        return result

    def _investigate_source_bundle_claim(self, request: dict[str, Any]) -> dict[str, Any]:
        """Judge one explicit source bundle without controller fan-out or exploration."""
        result, items, requested_count = self._verify_source_bundle_claim_result(request)
        verdict = result["verdict"]
        ledger_hit = verdict.get("bundle_ledger_hit") is True
        manifest = {
            "schema_version": 1,
            "mode": "targeted_source_bundle_claim",
            "topic_id": result["topic_id"],
            "topic": self._required_text(request, "topic"),
            "claim": result["claim"],
            "evidence": [
                {"file": item["file"], "start": item["start"], "end": item["end"],
                 "source_sha256": source_hash(item["excerpt"])}
                for item in items
            ],
            "verification": {
                "supports": verdict.get("supports") is True,
                "ledger_hit": ledger_hit,
                "model_turns": int(verdict.get("model_turns", 0)),
            },
            "metrics": {
                "requested_range_count": requested_count,
                "effective_range_count": len(items),
                "repository_read_count": len(items),
                "unrelated_topic_count": 0,
                "model_turn_count": int(verdict.get("model_turns", 0)),
                "ledger_hit": ledger_hit,
            },
        }
        return {"manifest": manifest, "answer": self._render_claim_answer(verdict), "verdict": verdict}

    def _select_relationship_ranges(self, caller: str, callee: str) -> tuple[dict[str, Any], list[dict[str, int | str]]]:
        """Select bounded relationship evidence from structural navigation only.

        The index is deliberately not evidence: each selected range is later read
        again by the existing claim-investigation path before any semantic call.
        """
        raw_edges = self._idx().relationship(caller, callee)
        edges: dict[tuple[str, int, str, str], tuple[str, int]] = {}
        valid_edge_count = 0
        invalid_edge_count = 0
        qualified_mismatch_edge_count = 0
        for edge in raw_edges:
            file = self._edge_field(edge, "file")
            line = self._edge_field(edge, "line")
            edge_caller = self._edge_field(edge, "caller")
            edge_callee = self._edge_field(edge, "callee")
            try:
                if isinstance(line, bool):
                    raise ValueError
                normalized_line = int(line)
            except (TypeError, ValueError):
                invalid_edge_count += 1
                continue
            if not isinstance(file, str) or not file or normalized_line < 1:
                invalid_edge_count += 1
                continue
            normalized_caller = "" if edge_caller is None else str(edge_caller)
            normalized_callee = "" if edge_callee is None else str(edge_callee)
            if "::" in callee and normalized_callee != callee:
                qualified_mismatch_edge_count += 1
                continue
            valid_edge_count += 1
            key = (file, normalized_line, normalized_caller, normalized_callee)
            edges[key] = (file, normalized_line)

        direct_edges = sorted(edges)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "mode": "direct_relationship_range_selection",
            "caller": caller,
            "callee": callee,
            "context_radius": RELATIONSHIP_CONTEXT_RADIUS,
            "raw_direct_edge_count": len(raw_edges),
            "direct_edge_count": len(direct_edges),
            "candidate_range_count": 0,
            "effective_range_count": 0,
            "normalization": {
                "invalid_direct_edge_count": invalid_edge_count,
                "qualified_mismatch_edge_count": qualified_mismatch_edge_count,
                "duplicate_direct_edge_count": valid_edge_count - len(direct_edges),
                "merged_range_count": 0,
                "unmergeable_overlapping_range_count": 0,
                "maximum_merged_range_lines": 500,
            },
            "selected_ranges": [],
            "route": "not_run",
            "selection_status": "no_direct_edge",
        }
        if invalid_edge_count:
            manifest["selection_status"] = "invalid_direct_edge"
            return manifest, []
        if not direct_edges:
            return manifest, []

        candidates: list[dict[str, Any]] = []
        for file, line, _, _ in direct_edges:
            start = max(1, line - RELATIONSHIP_CONTEXT_RADIUS)
            end = line + RELATIONSHIP_CONTEXT_RADIUS
            boundary = self._idx().function_for_line(file, line)
            function_key: tuple[str, str, int, int] | None = None
            if boundary is not None:
                try:
                    function_start = int(boundary.start_line)
                    function_end = int(boundary.end_line)
                    function_name = str(boundary.name)
                    function_file = str(boundary.file)
                except (AttributeError, TypeError, ValueError):
                    manifest["selection_status"] = "invalid_function_boundary"
                    return manifest, []
                if function_start < 1 or function_end < function_start:
                    manifest["selection_status"] = "invalid_function_boundary"
                    return manifest, []
                start = max(start, function_start)
                end = min(end, function_end)
                function_key = (
                    function_file, function_name, function_start, function_end
                )
            if start < 1 or end < start or end - start + 1 > 500:
                manifest["selection_status"] = "invalid_candidate_range"
                return manifest, []
            candidates.append({
                "file": file,
                "start": start,
                "end": end,
                "_function_key": function_key,
            })

        candidates.sort(key=lambda item: (str(item["file"]), int(item["start"]), int(item["end"])))
        manifest["candidate_range_count"] = len(candidates)
        effective: list[dict[str, int | str]] = []
        merged_range_count = 0
        for candidate in candidates:
            if effective:
                previous = effective[-1]
                same_function = (
                    candidate["_function_key"] == previous["_function_key"]
                )
                same_or_touching = (
                    candidate["file"] == previous["file"]
                    and int(candidate["start"]) <= int(previous["end"]) + 1
                )
                mergeable = same_or_touching and same_function
                merged_size = int(candidate["end"]) - int(previous["start"]) + 1
                if mergeable and merged_size <= 500:
                    previous["end"] = max(int(previous["end"]), int(candidate["end"]))
                    merged_range_count += 1
                    continue
                overlaps = (
                    candidate["file"] == previous["file"]
                    and int(candidate["start"]) <= int(previous["end"])
                )
                if overlaps:
                    # MultiRange forbids overlapping evidence.  Same-function
                    # overlap can reach here only when its union exceeds 500
                    # lines; different-function overlap is never merged.
                    manifest["normalization"]["unmergeable_overlapping_range_count"] = 1
                    manifest["selection_status"] = (
                        "overlapping_ranges_exceed_limit"
                        if same_function
                        else "cross_function_overlap"
                    )
                    return manifest, []
            effective.append(dict(candidate))

        public_effective = [
            {
                "file": str(item["file"]),
                "start": int(item["start"]),
                "end": int(item["end"]),
            }
            for item in effective
        ]
        manifest["effective_range_count"] = len(public_effective)
        manifest["normalization"]["merged_range_count"] = merged_range_count
        manifest["selected_ranges"] = public_effective
        if not public_effective:
            manifest["selection_status"] = "no_effective_range"
        elif len(public_effective) > MAX_BUNDLE_RANGES:
            manifest["selection_status"] = "too_many_ranges"
        else:
            manifest["selection_status"] = "selected"
            manifest["route"] = (
                "single_range" if len(public_effective) == 1 else "multi_range"
            )
        return manifest, public_effective

    @staticmethod
    def _relationship_selection_not_run(manifest: dict[str, Any]) -> dict[str, Any]:
        status = str(manifest["selection_status"])
        return {
            "selection_manifest": manifest,
            "verification": {
                "status": "not_run",
                "reason": status,
                "model_turn_count": 0,
                "ledger_hit": False,
                "repository_read_count": 0,
            },
            "answer": f"No relationship evidence was selected: {status}.",
        }

    def _investigate_relationship_claim(self, request: dict[str, Any]) -> dict[str, Any]:
        """Verify a direct relationship using only server-selected source ranges."""
        self._require_exact_fields(
            request, {"op", "topic_id", "topic", "claim", "caller", "callee"}
        )
        topic_id = self._required_string(request, "topic_id")
        topic = self._required_string(request, "topic")
        claim = self._required_string(request, "claim")
        caller = self._required_string(request, "caller")
        callee = self._required_string(request, "callee")
        selection_manifest, ranges = self._select_relationship_ranges(caller, callee)
        if selection_manifest["selection_status"] != "selected":
            return self._relationship_selection_not_run(selection_manifest)

        verification_request: dict[str, Any] = {
            "topic_id": topic_id,
            "topic": topic,
            "claim": claim,
        }
        if len(ranges) == 1:
            verification_request.update(ranges[0])
            verification = self._investigate_source_claim(verification_request)
        else:
            verification_request["ranges"] = ranges
            verification = self._investigate_source_bundle_claim(verification_request)
        return {
            "selection_manifest": selection_manifest,
            "verification": verification,
            "answer": verification["answer"],
        }

    def _verified_claims(self, request: dict[str, Any]) -> dict[str, Any]:
        ledger = self._claim_ledger()
        status = str(request.get("status", "")).strip().lower()
        topic_id = str(request.get("topic_id", "")).strip()
        claim = str(request.get("claim", "")).strip()
        limit = self._bounded_int(request.get("limit"), default=100, low=1, high=500)
        rows=[]
        for kind, records in (("range", ledger.records), ("bundle", ledger.bundle_records)):
            for key, rec in records.items():
                if status and str(rec.get("status", "")).lower() != status: continue
                if topic_id and str(rec.get("topic_id", "")) != topic_id: continue
                if claim and str(rec.get("claim", "")) != claim: continue
                rows.append({"kind": kind, "key": key, **rec})
        rows.sort(key=lambda r:(str(r.get("topic_id","")), str(r.get("claim","")), r["kind"], r["key"]))
        return {"records": rows[:limit], "returned": min(len(rows), limit), "matched": len(rows)}

    def _ledger_records(self, request: dict[str, Any]) -> dict[str, Any]:
        ledger = VerifiedEvidenceLedger(Path(self._ledger_path)) if self._ledger_path else VerifiedEvidenceLedger()
        status = str(request.get("status", "")).strip().lower()
        topic_id = str(request.get("topic_id", "")).strip()
        category = str(request.get("category", "")).strip()
        limit = self._bounded_int(request.get("limit"), default=100, low=1, high=500)
        rows=[]
        for key, rec in ledger.records.items():
            if status and str(rec.get("status", "")).lower() != status: continue
            if topic_id and str(rec.get("topic_id", "")) != topic_id: continue
            if category and str(rec.get("category", "")) != category: continue
            rows.append({"kind":"range", "key":key, **rec})
        for key, rec in ledger.bundle_records.items():
            if status and str(rec.get("status", "")).lower() != status: continue
            if topic_id and str(rec.get("topic_id", "")) != topic_id: continue
            if category and str(rec.get("category", "")) != category: continue
            rows.append({"kind":"bundle", "key":key, **rec})
        rows.sort(key=lambda r:(str(r.get("topic_id","")), str(r.get("category","")), r["kind"], r["key"]))
        return {"records": rows[:limit], "returned": min(len(rows), limit), "matched": len(rows)}


def main() -> int:
    parser=argparse.ArgumentParser(description="Repository-read-only JSON interface to ExpertAdvisor RepositoryAgent primitives")
    parser.add_argument("--request", help="single JSON request object")
    parser.add_argument("--ledger", help="optional evidence-ledger path")
    parser.add_argument("--claim-ledger", help="optional claim-evidence-ledger path")
    args=parser.parse_args()
    iface=CodexRepositoryInterface(ledger_path=args.ledger, claim_ledger_path=args.claim_ledger)
    try:
        req=json.loads(args.request) if args.request else json.load(__import__('sys').stdin)
        print(json.dumps({"ok":True,"result":iface.dispatch(req)}, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"ok":False,"error":type(exc).__name__,"message":str(exc)}, sort_keys=True))
        return 2

if __name__ == "__main__": raise SystemExit(main())
