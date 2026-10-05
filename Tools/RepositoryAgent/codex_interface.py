#!/usr/bin/env python3
"""Repository-read-only machine interface for external controllers.

Repository source remains read-only. Claim verification may persist decisions only
in the dedicated claim-evidence ledger; it cannot write repository source, invoke
a shell, access the database, run builds/tests, or mutate Git.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
import json
from pathlib import Path
import re
from typing import Any

from expertadvisor_agent import list_files, read_file, search
from .claim_evidence import VerifiedClaimLedger, source_hash
from .claim_verifier import LazyClaimVerifierRuntime, MAX_BUNDLE_RANGES
from .evidence import VerifiedEvidenceLedger
from .repository_index import FunctionBoundary, RepositoryIndex


RELATIONSHIP_CONTEXT_RADIUS = 12
# This is a structural admission limit, not a semantic ranking budget.  The
# operation fails closed once the exact resolved symbol has more direct,
# independently attributable relationships than can be safely packaged.
MAX_SYMBOL_DIRECT_RELATIONSHIPS = 16
# Subsystem investigation is intentionally a small, deterministic view of a
# directory tree, not a license to explore it.  Every limit is an admission
# limit: exceeding one returns no evidence and invokes no semantic model.
MAX_SUBSYSTEM_INVENTORY_SYMBOLS = 512
# A subsystem request is intended to name a deliberately reviewed, small
# source slice.  Sixteen files is enough for a modest implementation unit but
# prevents this operation from becoming a directory or repository inventory.
MAX_SUBSYSTEM_FILES = 16
MAX_SUBSYSTEM_SELECTED_SYMBOLS = 12
MAX_SUBSYSTEM_SELECTED_RELATIONSHIPS = 24
MAX_SUBSYSTEM_EVIDENCE_RANGES = MAX_BUNDLE_RANGES
MAX_SUBSYSTEM_EVIDENCE_LINES = 1200

# Catalog discovery is deliberately smaller than subsystem investigation.  It
# is a deterministic target selector, never a substitute for source evidence.
# Scope resolution examines catalog paths only.  These caps bound the
# directory-tree materialization algorithm, not a particular project folder.
MAX_DISCOVERY_CATALOG_FILES = 100_000
MAX_DISCOVERY_CATALOG_DIRECTORIES = 200_000
# Query matching is separately bounded before candidates are returned.  This
# is deliberately larger than either final candidate class cap so it limits a
# broad metadata query even when its matches span both classes.
MAX_DISCOVERY_MATCHED_CATALOG_ITEMS = 48
MAX_DISCOVERY_CANDIDATE_FILES = 16
MAX_DISCOVERY_CANDIDATE_SYMBOLS = 16
MAX_DISCOVERY_TOTAL_CANDIDATES = 24
MAX_DISCOVERY_QUERY_TERMS = 4

# Relationship-path discovery is intentionally a very small metadata-only
# graph walk.  These are admission limits, not truncation limits: exceeding
# any one raises an error and returns no path result.
MAX_RELATIONSHIP_PATH_HOPS = 4
MAX_RELATIONSHIP_PATH_VISITED_NODES = 64
MAX_RELATIONSHIP_PATH_EXAMINED_EDGES = 512
MAX_RELATIONSHIP_PATH_RESULTS = 4


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
                "investigate_relationship_claim", "investigate_relationship_chain_claim",
                "investigate_relationship_set_claim",
                "discover_catalog_targets",
                "discover_relationship_paths",
                "investigate_symbol",
                "investigate_subsystem",
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
        if op == "investigate_relationship_chain_claim": return self._investigate_relationship_chain_claim(request)
        if op == "investigate_relationship_set_claim": return self._investigate_relationship_set_claim(request)
        if op == "discover_catalog_targets": return self._discover_catalog_targets(request)
        if op == "discover_relationship_paths": return self._discover_relationship_paths(request)
        if op == "investigate_symbol": return self._investigate_symbol(request)
        if op == "investigate_subsystem": return self._investigate_subsystem(request)
        raise ValueError(f"unsupported operation: {op!r}")

    @staticmethod
    def _discovery_scope(value: str) -> str:
        """Validate a logical directory selector without permitting traversal."""
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError("scope must be a non-empty normalized directory selector")
        path = Path(value)
        if (path.is_absolute() or not path.parts or path.as_posix() != value or "\x00" in value
                or "\\" in value or any(part in {"", ".", ".."} for part in path.parts)):
            raise ValueError("scope must be a normalized repository-relative directory selector")
        return value

    @staticmethod
    def _discovery_query_groups(value: Any) -> list[list[str]]:
        if not isinstance(value, list) or not 1 <= len(value) <= MAX_DISCOVERY_QUERY_TERMS:
            raise ValueError(f"query_groups must contain 1-{MAX_DISCOVERY_QUERY_TERMS} groups")
        groups: list[tuple[str, ...]] = []
        for group in value:
            if not isinstance(group, list) or not 1 <= len(group) <= MAX_DISCOVERY_QUERY_TERMS:
                raise ValueError(f"each query_groups entry must contain 1-{MAX_DISCOVERY_QUERY_TERMS} identifier terms")
            terms: list[str] = []
            for term in group:
                if (not isinstance(term, str) or term != term.strip()
                        or not re.fullmatch(r"[A-Za-z][A-Za-z0-9]{2,31}", term)):
                    raise ValueError("each query_groups term must be a 3-32 character identifier term")
                normalized = term.casefold()
                if normalized in terms:
                    raise ValueError("query_groups terms must not contain duplicates")
                terms.append(normalized)
            canonical = tuple(sorted(terms))
            if canonical in groups:
                raise ValueError("query_groups must not contain duplicate groups")
            groups.append(canonical)
        return [list(group) for group in sorted(groups)]

    @staticmethod
    def _catalog_tokens(value: str) -> set[str]:
        """Split only catalog identities; no source text is inspected here."""
        expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
        return {
            token.casefold() for token in re.findall(r"[A-Za-z][A-Za-z0-9]*", expanded)
        }

    def _resolve_discovery_scope(self, requested: str) -> tuple[str, list[str]]:
        """Resolve an exact directory or unique directory basename from index paths."""
        try:
            files = sorted({str(file) for file in self._idx().files})
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("repository catalog is unavailable") from exc
        if len(files) > MAX_DISCOVERY_CATALOG_FILES:
            raise ValueError("repository catalog exceeds the maximum indexed file count")
        directories: set[str] = set()
        for file in files:
            path = Path(file)
            if path.is_absolute() or not path.parent.parts:
                raise ValueError("repository catalog contains malformed file metadata")
            for depth in range(1, len(path.parts)):
                directories.add(Path(*path.parts[:depth]).as_posix())
                if len(directories) > MAX_DISCOVERY_CATALOG_DIRECTORIES:
                    raise ValueError("repository catalog exceeds the maximum indexed directory count")
        if requested in directories:
            matches = [requested]
        else:
            matches = sorted(directory for directory in directories if Path(directory).name == requested)
        if not matches:
            raise ValueError("discovery scope was not found in repository catalog")
        if len(matches) != 1:
            raise ValueError("discovery scope is ambiguous in repository catalog")
        resolved = matches[0]
        # The directory itself is the complete logical boundary.  Descendant
        # directories are separate scopes and are never pulled in implicitly.
        scoped_files = [
            file for file in files
            if Path(file).parent.as_posix() == resolved
        ]
        if not scoped_files:
            raise ValueError("discovery scope contains no indexed source files")
        return resolved, scoped_files

    def _discover_catalog_targets(self, request: dict[str, Any]) -> dict[str, Any]:
        """Return a bounded, non-evidentiary catalog view from structural metadata.

        No source range is selected or reread, no relationship is followed, and
        no semantic verifier is initialized.  This only helps select a later
        bounded investigation target.
        """
        self._require_exact_fields(request, {"op", "scope", "query_groups"})
        requested_scope = self._discovery_scope(request.get("scope"))
        scope, scoped_files = self._resolve_discovery_scope(requested_scope)
        # Validate and apply groups only after the catalog scope is resolved.
        query_groups = self._discovery_query_groups(request.get("query_groups"))

        def matches_any_group(tokens: set[str]) -> bool:
            return any(set(group) <= tokens for group in query_groups)

        matched_files = [
            file for file in scoped_files
            if matches_any_group(self._catalog_tokens(Path(file).stem))
        ]

        try:
            boundaries = list(self._idx().functions)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("repository catalog function metadata is unavailable") from exc
        scoped_file_set = set(scoped_files)
        matched_symbol_extents: list[tuple[str, str, int, int]] = []
        for boundary in boundaries:
            if not isinstance(boundary, FunctionBoundary):
                raise ValueError("repository catalog contains non-function definition metadata")
            try:
                file, name = str(boundary.file), str(boundary.name)
                start, end = int(boundary.start_line), int(boundary.end_line)
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError("repository catalog contains malformed function metadata") from exc
            if file not in scoped_file_set:
                continue
            if not name or start < 1 or end < start:
                raise ValueError("repository catalog contains malformed function metadata")
            if matches_any_group(self._catalog_tokens(name)):
                matched_symbol_extents.append((file, name, start, end))
        matched_symbol_extents.sort()
        if any(left == right for left, right in zip(matched_symbol_extents, matched_symbol_extents[1:])):
            raise ValueError("repository catalog contains duplicate function definition metadata")

        # Function names can legitimately be overloaded inside one file.  Do
        # not disclose source positions; instead retain deterministic ordinal
        # ownership within the canonical container so each indexed definition
        # remains distinguishable in catalog output.
        matched_symbols: list[dict[str, Any]] = []
        offset = 0
        while offset < len(matched_symbol_extents):
            file, name, _, _ = matched_symbol_extents[offset]
            end_offset = offset + 1
            while (end_offset < len(matched_symbol_extents)
                   and matched_symbol_extents[end_offset][:2] == (file, name)):
                end_offset += 1
            definition_count = end_offset - offset
            for ordinal in range(definition_count):
                matched_symbols.append({
                    "identity": name,
                    "kind": "function_definition",
                    "container_file": file,
                    "container_ordinal": ordinal + 1,
                    "container_definition_count": definition_count,
                })
            offset = end_offset
        matched_total = len(matched_files) + len(matched_symbols)
        if matched_total > MAX_DISCOVERY_MATCHED_CATALOG_ITEMS:
            raise ValueError("discovery matched catalog population cap exceeded")

        # There is no ranking or truncation step: every admitted metadata
        # match becomes a candidate only after the match-population admission.
        file_candidates = matched_files
        symbol_rows = matched_symbols
        if len(file_candidates) > MAX_DISCOVERY_CANDIDATE_FILES:
            raise ValueError("discovery candidate file cap exceeded")
        if len(symbol_rows) > MAX_DISCOVERY_CANDIDATE_SYMBOLS:
            raise ValueError("discovery candidate symbol cap exceeded")
        total = len(file_candidates) + len(symbol_rows)
        if total > MAX_DISCOVERY_TOTAL_CANDIDATES:
            raise ValueError("discovery total candidate cap exceeded")

        return {
            "schema_version": 1,
            "mode": "bounded_catalog_discovery",
            "evidentiary_status": "non_evidentiary",
            "required_follow_up": (
                "Use an existing bounded investigation operation before making "
                "repository-derived behavioral claims."
            ),
            "resolved_scope": scope,
            "scope_direct_file_count": len(scoped_files),
            "query_groups": query_groups,
            "matched_catalog_counts": {
                "files": len(matched_files), "symbols": len(matched_symbols),
                "total": matched_total,
            },
            "candidates": {
                "files": file_candidates,
                "symbols": [
                    symbol for symbol in symbol_rows
                ],
            },
            "candidate_counts": {
                "files": len(file_candidates), "symbols": len(symbol_rows), "total": total,
            },
            "caps": {
                "catalog_files": MAX_DISCOVERY_CATALOG_FILES,
                "catalog_directories": MAX_DISCOVERY_CATALOG_DIRECTORIES,
                "matched_catalog_items": MAX_DISCOVERY_MATCHED_CATALOG_ITEMS,
                "candidate_files": MAX_DISCOVERY_CANDIDATE_FILES,
                "candidate_symbols": MAX_DISCOVERY_CANDIDATE_SYMBOLS,
                "total_candidates": MAX_DISCOVERY_TOTAL_CANDIDATES,
            },
        }

    @staticmethod
    def _relationship_path_max_hops(value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("max_hops must be an integer")
        if not 1 <= value <= MAX_RELATIONSHIP_PATH_HOPS:
            raise ValueError(
                f"max_hops must be between 1 and {MAX_RELATIONSHIP_PATH_HOPS}"
            )
        return value

    def _discover_relationship_paths(self, request: dict[str, Any]) -> dict[str, Any]:
        """Discover bounded shortest structural paths without reading source.

        Function boundaries and call sites are index metadata, not evidence.
        In particular, an unqualified call site only contributes an edge when
        its short name identifies exactly one indexed function globally.
        """
        self._require_exact_fields(request, {"op", "scope", "from", "to", "max_hops"})
        requested_scope = self._discovery_scope(request.get("scope"))
        from_requested = self._required_string(request, "from")
        to_requested = self._required_string(request, "to")
        max_hops = self._relationship_path_max_hops(request.get("max_hops"))
        scope, scoped_files = self._resolve_discovery_scope(requested_scope)
        scoped_file_set = set(scoped_files)
        try:
            boundaries = list(self._idx().functions)
            calls_by_caller = self._idx().calls_by_caller
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("repository relationship metadata is unavailable") from exc
        if not isinstance(calls_by_caller, dict):
            raise ValueError("repository relationship metadata is malformed")

        by_identity: dict[str, list[FunctionBoundary]] = {}
        by_short_name: dict[str, list[FunctionBoundary]] = {}
        for boundary in boundaries:
            if not isinstance(boundary, FunctionBoundary):
                raise ValueError("repository relationship metadata contains non-function definitions")
            try:
                name, file = str(boundary.name), str(boundary.file)
                start, end = int(boundary.start_line), int(boundary.end_line)
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError("repository relationship metadata contains malformed function definitions") from exc
            if not name or not file or start < 1 or end < start:
                raise ValueError("repository relationship metadata contains malformed function definitions")
            by_identity.setdefault(name, []).append(boundary)
            by_short_name.setdefault(name.split("::")[-1], []).append(boundary)

        def unique_endpoint(requested: str) -> str:
            exact = by_identity.get(requested, [])
            candidates = exact if exact else (
                by_short_name.get(requested, []) if "::" not in requested else []
            )
            if not candidates:
                raise ValueError("relationship path endpoint was not found")
            if len(candidates) != 1:
                raise ValueError("relationship path endpoint is ambiguous")
            boundary = candidates[0]
            if boundary.file not in scoped_file_set:
                raise ValueError("relationship path endpoint is outside resolved scope")
            return boundary.name

        start = unique_endpoint(from_requested)
        target = unique_endpoint(to_requested)
        if start == target:
            raise ValueError("relationship path endpoints must be distinct")

        examined_edges = 0
        visited_nodes = {start}
        distances: dict[str, int] = {start: 0}
        parents: dict[str, set[str]] = {}
        queue = deque([start])

        def scoped_unique_callees(caller: str) -> list[str]:
            nonlocal examined_edges
            raw_sites = calls_by_caller.get(caller, [])
            if not isinstance(raw_sites, list):
                raise ValueError("repository relationship metadata contains malformed call adjacency")
            resolved: set[str] = set()
            for site in raw_sites:
                examined_edges += 1
                if examined_edges > MAX_RELATIONSHIP_PATH_EXAMINED_EDGES:
                    raise ValueError("relationship path examined-edge cap exceeded")
                try:
                    site_caller, site_callee, site_file = str(site.caller), str(site.callee), str(site.file)
                except (AttributeError, TypeError, ValueError) as exc:
                    raise ValueError("repository relationship metadata contains malformed call edge") from exc
                # The index has short-name convenience aliases; never use them
                # as graph edges, and never let an out-of-scope edge enter.
                if site_caller != caller or site_file not in scoped_file_set or not site_callee:
                    continue
                if "::" in site_callee:
                    candidates = by_identity.get(site_callee, [])
                else:
                    candidates = by_short_name.get(site_callee, [])
                # A short textual call may not pick an arbitrary qualified
                # definition.  Duplicated full identities are also excluded.
                if len(candidates) != 1:
                    continue
                callee = candidates[0]
                if callee.file in scoped_file_set:
                    resolved.add(callee.name)
            return sorted(resolved)

        while queue:
            caller = queue.popleft()
            depth = distances[caller]
            if depth >= max_hops:
                continue
            # Once a shortest target depth is known, nodes at that depth have
            # no role in another shortest path; all earlier levels still run.
            if target in distances and depth >= distances[target]:
                continue
            for callee in scoped_unique_callees(caller):
                next_depth = depth + 1
                known_depth = distances.get(callee)
                if known_depth is None:
                    if len(visited_nodes) >= MAX_RELATIONSHIP_PATH_VISITED_NODES:
                        raise ValueError("relationship path visited-node cap exceeded")
                    visited_nodes.add(callee)
                    distances[callee] = next_depth
                    parents[callee] = {caller}
                    queue.append(callee)
                elif known_depth == next_depth:
                    parents.setdefault(callee, set()).add(caller)

        paths: list[list[str]] = []
        if target in distances:
            def materialize(node: str) -> list[list[str]]:
                if node == start:
                    return [[start]]
                rows: list[list[str]] = []
                for parent in sorted(parents.get(node, ())):
                    for prefix in materialize(parent):
                        rows.append(prefix + [node])
                        if len(rows) > MAX_RELATIONSHIP_PATH_RESULTS:
                            raise ValueError("relationship path result cap exceeded")
                return rows
            paths = sorted(materialize(target))
            if len(paths) > MAX_RELATIONSHIP_PATH_RESULTS:
                raise ValueError("relationship path result cap exceeded")

        return {
            "schema_version": 1,
            "mode": "bounded_relationship_path_discovery",
            "evidentiary_status": "non_evidentiary",
            "required_follow_up": (
                "Use investigate_relationship_chain_claim or another bounded evidentiary "
                "investigation before making repository-derived behavioral claims."
            ),
            "resolved_scope": scope,
            "from": start,
            "to": target,
            "max_hops": max_hops,
            "paths": paths,
            "path_count": len(paths),
            "caps": {
                "max_hops": MAX_RELATIONSHIP_PATH_HOPS,
                "visited_nodes": MAX_RELATIONSHIP_PATH_VISITED_NODES,
                "examined_edges": MAX_RELATIONSHIP_PATH_EXAMINED_EDGES,
                "returned_paths": MAX_RELATIONSHIP_PATH_RESULTS,
            },
        }

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

    def _relationship_direct_edges(self, caller: str, callee: str) -> tuple[dict[str, Any], list[tuple[str, int, str, str]]]:
        """Validate and canonically select direct structural edges for one hop."""
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
        normalization = {
            "invalid_direct_edge_count": invalid_edge_count,
            "qualified_mismatch_edge_count": qualified_mismatch_edge_count,
            "duplicate_direct_edge_count": valid_edge_count - len(direct_edges),
        }
        if invalid_edge_count:
            status = "invalid_direct_edge"
        elif not direct_edges:
            status = "no_direct_edge"
        else:
            status = "selected"
        return ({
            "raw_direct_edge_count": len(raw_edges),
            "direct_edge_count": len(direct_edges),
            "normalization": normalization,
            "selection_status": status,
        }, direct_edges)

    def _relationship_candidate_ranges(
        self, direct_edges: list[tuple[str, int, str, str]]
    ) -> tuple[str, list[dict[str, Any]]]:
        """Construct function-clipped public evidence candidates for direct edges."""
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
                    return "invalid_function_boundary", []
                if function_start < 1 or function_end < function_start:
                    return "invalid_function_boundary", []
                start = max(start, function_start)
                end = min(end, function_end)
                function_key = (
                    function_file, function_name, function_start, function_end
                )
            if start < 1 or end < start or end - start + 1 > 500:
                return "invalid_candidate_range", []
            candidates.append({
                "file": file,
                "start": start,
                "end": end,
                "_function_key": function_key,
            })

        candidates.sort(key=lambda item: (str(item["file"]), int(item["start"]), int(item["end"])))
        return "selected", candidates

    @staticmethod
    def _relationship_normalization_template() -> dict[str, int]:
        return {
            "duplicate_candidate_range_count": 0,
            "merged_range_count": 0,
            "unmergeable_overlapping_range_count": 0,
            "maximum_merged_range_lines": 500,
        }

    def _normalize_relationship_ranges(
        self, candidates: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], list[dict[str, int | str]]]:
        """Globally deduplicate and function-safely normalize candidates.

        Function identity is retained only while selecting ranges.  It is never
        emitted to, or passed through to, the claim verifier.
        """
        candidates = sorted(
            candidates,
            key=lambda item: (
                str(item["file"]), int(item["start"]), int(item["end"]),
                repr(item["_function_key"]),
            ),
        )
        unique_candidates: list[dict[str, Any]] = []
        exact_ranges: set[tuple[str, int, int]] = set()
        exact_duplicate_count = 0
        for candidate in candidates:
            physical_range = (
                str(candidate["file"]), int(candidate["start"]), int(candidate["end"])
            )
            if physical_range in exact_ranges:
                exact_duplicate_count += 1
                continue
            exact_ranges.add(physical_range)
            unique_candidates.append(candidate)

        effective: list[dict[str, int | str]] = []
        merged_range_count = exact_duplicate_count
        normalization = self._relationship_normalization_template()
        for candidate in unique_candidates:
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
                    normalization["unmergeable_overlapping_range_count"] = 1
                    status = (
                        "overlapping_ranges_exceed_limit"
                        if same_function
                        else "cross_function_overlap"
                    )
                    return {"selection_status": status, "normalization": normalization}, []
            effective.append(dict(candidate))

        public_effective = [
            {
                "file": str(item["file"]),
                "start": int(item["start"]),
                "end": int(item["end"]),
            }
            for item in effective
        ]
        normalization["merged_range_count"] = merged_range_count
        normalization["duplicate_candidate_range_count"] = exact_duplicate_count
        if not public_effective:
            status = "no_effective_range"
        elif len(public_effective) > MAX_BUNDLE_RANGES:
            status = "too_many_ranges"
        else:
            status = "selected"
        return {"selection_status": status, "normalization": normalization}, public_effective

    def _select_relationship_ranges(self, caller: str, callee: str) -> tuple[dict[str, Any], list[dict[str, int | str]]]:
        """Select bounded relationship evidence from structural navigation only.

        The index is deliberately not evidence: each selected range is later read
        again by the existing claim-investigation path before any semantic call.
        """
        direct, direct_edges = self._relationship_direct_edges(caller, callee)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "mode": "direct_relationship_range_selection",
            "caller": caller,
            "callee": callee,
            "context_radius": RELATIONSHIP_CONTEXT_RADIUS,
            "raw_direct_edge_count": direct["raw_direct_edge_count"],
            "direct_edge_count": direct["direct_edge_count"],
            "candidate_range_count": 0,
            "effective_range_count": 0,
            "normalization": {
                **direct["normalization"],
                **self._relationship_normalization_template(),
            },
            "selected_ranges": [],
            "route": "not_run",
            "selection_status": direct["selection_status"],
        }
        if direct["selection_status"] != "selected":
            return manifest, []
        candidate_status, candidates = self._relationship_candidate_ranges(direct_edges)
        if candidate_status != "selected":
            manifest["selection_status"] = candidate_status
            return manifest, []
        manifest["candidate_range_count"] = len(candidates)
        normalized, public_effective = self._normalize_relationship_ranges(candidates)
        manifest["normalization"].update(normalized["normalization"])
        manifest["selection_status"] = normalized["selection_status"]
        manifest["effective_range_count"] = len(public_effective)
        manifest["selected_ranges"] = public_effective
        if manifest["selection_status"] == "selected":
            manifest["route"] = "single_range" if len(public_effective) == 1 else "multi_range"
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

    def _required_relationship_chain_path(self, request: dict[str, Any]) -> list[str]:
        """Validate an explicit acyclic chain; this operation never discovers one."""
        path = request.get("path")
        if not isinstance(path, list):
            raise ValueError("path must be an array")
        if not 3 <= len(path) <= 5:
            raise ValueError("path must contain 3-5 symbols")
        normalized: list[str] = []
        for symbol in path:
            if not isinstance(symbol, str) or not symbol.strip():
                raise ValueError("path members must be non-empty strings")
            normalized.append(symbol.strip())
        # Cycles make a caller-supplied chain ambiguous and add no supported
        # investigation use case, so they are rejected before index access.
        if len(set(normalized)) != len(normalized):
            raise ValueError("path must not repeat a symbol")
        return normalized

    @staticmethod
    def _chain_hop_manifest(
        hop_index: int, caller: str, callee: str, direct: dict[str, Any],
        candidates: list[dict[str, Any]] | None = None,
        candidate_status: str | None = None,
    ) -> dict[str, Any]:
        public_candidates = [] if candidates is None else [
            {"file": str(item["file"]), "start": int(item["start"]), "end": int(item["end"])}
            for item in candidates
        ]
        status = candidate_status or str(direct["selection_status"])
        return {
            "hop_index": hop_index,
            "caller": caller,
            "callee": callee,
            "raw_direct_edge_count": direct["raw_direct_edge_count"],
            "direct_edge_count": direct["direct_edge_count"],
            "candidate_range_count": len(public_candidates),
            "candidate_ranges": public_candidates,
            "normalization": {
                **direct["normalization"],
                **CodexRepositoryInterface._relationship_normalization_template(),
            },
            "selection_status": status,
        }

    def _chain_selection_not_run(self, manifest: dict[str, Any]) -> dict[str, Any]:
        return self._relationship_selection_not_run(manifest)

    def _investigate_relationship_chain_claim(self, request: dict[str, Any]) -> dict[str, Any]:
        """Verify a caller-supplied short direct-call chain without path discovery."""
        self._require_exact_fields(request, {"op", "topic_id", "topic", "claim", "path"})
        topic_id = self._required_string(request, "topic_id")
        topic = self._required_string(request, "topic")
        claim = self._required_string(request, "claim")
        path = self._required_relationship_chain_path(request)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "mode": "explicit_relationship_chain_range_selection",
            "context_radius": RELATIONSHIP_CONTEXT_RADIUS,
            "path": path,
            "hop_count": len(path) - 1,
            "hops": [],
            "candidate_range_count": 0,
            "effective_range_count": 0,
            "normalization": self._relationship_normalization_template(),
            "selected_ranges": [],
            "route": "not_run",
            "selection_status": "not_run",
        }
        candidates: list[dict[str, Any]] = []
        first_failure: str | None = None
        for hop_index, (caller, callee) in enumerate(zip(path, path[1:])):
            direct, direct_edges = self._relationship_direct_edges(caller, callee)
            if direct["selection_status"] != "selected":
                manifest["hops"].append(self._chain_hop_manifest(hop_index, caller, callee, direct))
                if first_failure is None:
                    first_failure = str(direct["selection_status"])
                continue
            candidate_status, hop_candidates = self._relationship_candidate_ranges(direct_edges)
            hop = self._chain_hop_manifest(
                hop_index, caller, callee, direct, hop_candidates,
                candidate_status,
            )
            manifest["hops"].append(hop)
            if candidate_status != "selected":
                if first_failure is None:
                    first_failure = candidate_status
            else:
                candidates.extend(hop_candidates)

        if first_failure is not None:
            manifest["selection_status"] = first_failure
            return self._chain_selection_not_run(manifest)

        manifest["candidate_range_count"] = len(candidates)
        normalized, ranges = self._normalize_relationship_ranges(candidates)
        manifest["normalization"].update(normalized["normalization"])
        manifest["effective_range_count"] = len(ranges)
        manifest["selected_ranges"] = ranges
        manifest["selection_status"] = normalized["selection_status"]
        if normalized["selection_status"] != "selected":
            return self._chain_selection_not_run(manifest)
        manifest["route"] = "single_range" if len(ranges) == 1 else "multi_range"

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
            "selection_manifest": manifest,
            "verification": verification,
            "answer": verification["answer"],
        }

    def _required_relationship_set(self, request: dict[str, Any]) -> list[tuple[str, str]]:
        """Validate explicit direct relationships without deriving any new ones."""
        relationships = request.get("relationships")
        if not isinstance(relationships, list):
            raise ValueError("relationships must be an array")
        if not 2 <= len(relationships) <= 5:
            raise ValueError("relationships must contain 2-5 relationship objects")
        normalized: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for relationship in relationships:
            if not isinstance(relationship, dict):
                raise ValueError("relationship entries must be objects")
            self._require_exact_fields(relationship, {"caller", "callee"})
            caller = self._required_string(relationship, "caller")
            callee = self._required_string(relationship, "callee")
            pair = (caller, callee)
            if pair in seen:
                raise ValueError("relationships must not contain duplicate caller/callee pairs")
            seen.add(pair)
            normalized.append(pair)
        return normalized

    @staticmethod
    def _relationship_set_entry_manifest(
        relationship_index: int, caller: str, callee: str, direct: dict[str, Any],
        candidates: list[dict[str, Any]] | None = None,
        candidate_status: str | None = None,
    ) -> dict[str, Any]:
        public_candidates = [] if candidates is None else [
            {"file": str(item["file"]), "start": int(item["start"]), "end": int(item["end"])}
            for item in candidates
        ]
        return {
            "relationship_index": relationship_index,
            "caller": caller,
            "callee": callee,
            "raw_direct_edge_count": direct["raw_direct_edge_count"],
            "direct_edge_count": direct["direct_edge_count"],
            "candidate_range_count": len(public_candidates),
            "candidate_ranges": public_candidates,
            "normalization": {
                **direct["normalization"],
                **CodexRepositoryInterface._relationship_normalization_template(),
            },
            "selection_status": candidate_status or str(direct["selection_status"]),
        }

    def _investigate_relationship_set_claim(self, request: dict[str, Any]) -> dict[str, Any]:
        """Verify a bounded caller-supplied set of direct relationships.

        This deliberately performs no traversal or relationship discovery.  All
        structural assertions must select before any source reread is allowed.
        """
        self._require_exact_fields(
            request, {"op", "topic_id", "topic", "claim", "relationships"}
        )
        topic_id = self._required_string(request, "topic_id")
        topic = self._required_string(request, "topic")
        claim = self._required_string(request, "claim")
        relationships = self._required_relationship_set(request)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "mode": "explicit_relationship_set_range_selection",
            "context_radius": RELATIONSHIP_CONTEXT_RADIUS,
            "relationship_count": len(relationships),
            "relationships": [],
            "candidate_range_count": 0,
            "effective_range_count": 0,
            "normalization": self._relationship_normalization_template(),
            "selected_ranges": [],
            "route": "not_run",
            "selection_status": "not_run",
        }
        candidates: list[dict[str, Any]] = []
        first_failure: str | None = None
        # Complete the structural pass for every caller-supplied pair before
        # entering either source verifier, even if an earlier pair failed.
        for relationship_index, (caller, callee) in enumerate(relationships):
            direct, direct_edges = self._relationship_direct_edges(caller, callee)
            if direct["selection_status"] != "selected":
                manifest["relationships"].append(self._relationship_set_entry_manifest(
                    relationship_index, caller, callee, direct
                ))
                if first_failure is None:
                    first_failure = str(direct["selection_status"])
                continue
            candidate_status, relationship_candidates = self._relationship_candidate_ranges(direct_edges)
            manifest["relationships"].append(self._relationship_set_entry_manifest(
                relationship_index, caller, callee, direct, relationship_candidates,
                candidate_status,
            ))
            if candidate_status != "selected":
                if first_failure is None:
                    first_failure = candidate_status
            else:
                candidates.extend(relationship_candidates)

        if first_failure is not None:
            manifest["selection_status"] = first_failure
            return self._relationship_selection_not_run(manifest)

        manifest["candidate_range_count"] = len(candidates)
        normalized, ranges = self._normalize_relationship_ranges(candidates)
        manifest["normalization"].update(normalized["normalization"])
        manifest["effective_range_count"] = len(ranges)
        manifest["selected_ranges"] = ranges
        manifest["selection_status"] = normalized["selection_status"]
        if manifest["selection_status"] != "selected":
            return self._relationship_selection_not_run(manifest)
        manifest["route"] = "single_range" if len(ranges) == 1 else "multi_range"

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
            "selection_manifest": manifest,
            "verification": verification,
            "answer": verification["answer"],
        }

    @staticmethod
    def _symbol_extent_public(boundary: Any) -> dict[str, int | str]:
        return {
            "file": str(boundary.file),
            "start": int(boundary.start_line),
            "end": int(boundary.end_line),
        }

    def _resolve_symbol_extent(
        self, requested_symbol: str,
    ) -> tuple[str, str | None, Any | None]:
        """Resolve exactly one indexed definition and validate its function extent.

        This intentionally uses no lexical search and no short-name fallback.
        An indexed occurrence that cannot be tied exactly to its enclosing
        function is unusable as evidence and therefore fails closed.
        """
        try:
            definitions = list(self._idx().find_definitions(requested_symbol))
        except (AttributeError, TypeError, ValueError):
            return "malformed_symbol_resolution", None, None
        if not definitions:
            return "missing_symbol", None, None
        if len(definitions) != 1:
            return "ambiguous_symbol", None, None
        definition = definitions[0]
        file = self._edge_field(definition, "file")
        line = self._edge_field(definition, "line")
        occurrence_symbol = self._edge_field(definition, "symbol")
        canonical = self._edge_field(definition, "function")
        try:
            if isinstance(line, bool):
                raise ValueError
            line = int(line)
        except (TypeError, ValueError):
            return "malformed_symbol_definition", None, None
        if (
            not isinstance(file, str) or not file or line < 1
            or not isinstance(occurrence_symbol, str) or occurrence_symbol != requested_symbol
            or not isinstance(canonical, str) or not canonical.strip()
        ):
            return "malformed_symbol_definition", None, None
        path = Path(file)
        if (
            path.is_absolute() or path.as_posix() != file or "\x00" in file or "\\" in file
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            return "malformed_symbol_definition", None, None
        canonical = canonical.strip()
        if "::" in requested_symbol and canonical != requested_symbol:
            return "malformed_symbol_definition", None, None
        try:
            boundary = self._idx().function_for_line(file, line)
        except (AttributeError, TypeError, ValueError):
            return "malformed_symbol_extent", None, None
        if boundary is None:
            return "malformed_symbol_extent", None, None
        try:
            if isinstance(boundary.start_line, bool) or isinstance(boundary.end_line, bool):
                raise ValueError
            boundary_file = str(boundary.file)
            boundary_name = str(boundary.name)
            start = int(boundary.start_line)
            end = int(boundary.end_line)
        except (AttributeError, TypeError, ValueError):
            return "malformed_symbol_extent", None, None
        if (
            boundary_file != file or boundary_name != canonical
            or start < 1 or end < start or line != start
        ):
            return "malformed_symbol_extent", None, None
        if end - start + 1 > 500:
            return "symbol_extent_over_limit", canonical, boundary
        return "selected", canonical, boundary

    def _symbol_direct_relationships(
        self, canonical_symbol: str,
    ) -> tuple[str, list[dict[str, Any]], dict[str, int]]:
        """Select only direct index records exactly attributable to this symbol.

        Incoming and outgoing call records are both admitted only when their
        indexed endpoint equals the canonical symbol.  A source spelling with a
        matching short name is deliberately not treated as the same endpoint.
        """
        try:
            incoming = list(self._idx().callers_of(canonical_symbol))
            outgoing = list(self._idx().callees_of(canonical_symbol))
        except (AttributeError, TypeError, ValueError):
            return "malformed_direct_relationship", [], {
                "raw_direct_relationship_count": 0,
                "invalid_direct_relationship_count": 1,
                "unattributed_direct_relationship_count": 0,
                "duplicate_direct_relationship_count": 0,
                "valid_direct_relationship_count": 0,
            }
        raw_count = len(incoming) + len(outgoing)
        invalid_count = 0
        unattributed_count = 0
        relationships: dict[tuple[str, int, str, str], dict[str, Any]] = {}
        for direction, sites in (("incoming", incoming), ("outgoing", outgoing)):
            for site in sites:
                file = self._edge_field(site, "file")
                line = self._edge_field(site, "line")
                caller = self._edge_field(site, "caller")
                callee = self._edge_field(site, "callee")
                try:
                    if isinstance(line, bool):
                        raise ValueError
                    line = int(line)
                except (TypeError, ValueError):
                    invalid_count += 1
                    continue
                if (
                    not isinstance(file, str) or not file or line < 1
                    or not isinstance(caller, str) or not caller
                    or not isinstance(callee, str) or not callee
                ):
                    invalid_count += 1
                    continue
                if (
                    (direction == "incoming" and callee != canonical_symbol)
                    or (direction == "outgoing" and caller != canonical_symbol)
                ):
                    unattributed_count += 1
                    continue
                key = (file, line, caller, callee)
                row = relationships.setdefault(key, {
                    "caller": caller,
                    "callee": callee,
                    "file": file,
                    "line": line,
                    "directions": [],
                })
                row["directions"].append(direction)
        admitted = []
        for key in sorted(relationships):
            row = relationships[key]
            row["directions"] = sorted(set(row["directions"]))
            admitted.append(row)
        accounting = {
            "raw_direct_relationship_count": raw_count,
            "invalid_direct_relationship_count": invalid_count,
            "unattributed_direct_relationship_count": unattributed_count,
            "duplicate_direct_relationship_count": raw_count - invalid_count - unattributed_count - len(admitted),
            "valid_direct_relationship_count": len(admitted),
        }
        if invalid_count:
            return "malformed_direct_relationship", [], accounting
        if len(admitted) > MAX_SYMBOL_DIRECT_RELATIONSHIPS:
            return "too_many_direct_relationships", [], accounting
        return "selected", admitted, accounting

    @staticmethod
    def _symbol_semantic_request(requested_symbol: str, resolved_symbol: str) -> str:
        return (
            "Explain, using only the supplied evidence, the narrow source-visible "
            f"responsibility or behavior of resolved symbol {resolved_symbol!r} "
            f"(requested as {requested_symbol!r}) and any represented direct "
            "source-visible interactions. Do not infer unseen implementation, "
            "runtime behavior, transitive effects, unrepresented callers or "
            "callees, database state, scheduler state, or other repository content."
        )

    def _symbol_selection_not_run(self, manifest: dict[str, Any]) -> dict[str, Any]:
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
            "answer": f"No symbol evidence was selected: {status}.",
        }

    def _investigate_symbol(self, request: dict[str, Any]) -> dict[str, Any]:
        """Investigate one exact indexed symbol with server-owned evidence only."""
        self._require_exact_fields(request, {"op", "topic_id", "topic", "symbol"})
        topic_id = self._required_string(request, "topic_id")
        topic = self._required_string(request, "topic")
        requested_symbol = self._required_string(request, "symbol")
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "mode": "bounded_symbol_investigation_selection",
            "requested_symbol": requested_symbol,
            "resolved_canonical_symbol": None,
            "resolution_status": "not_run",
            "definition_source_extent": None,
            "direct_structural_relationship_count": 0,
            "direct_relationship_accounting": {
                "raw_direct_relationship_count": 0,
                "invalid_direct_relationship_count": 0,
                "unattributed_direct_relationship_count": 0,
                "duplicate_direct_relationship_count": 0,
                "valid_direct_relationship_count": 0,
                "maximum_admitted_direct_relationships": MAX_SYMBOL_DIRECT_RELATIONSHIPS,
            },
            "admitted_direct_relationships": [],
            "candidate_range_count": 0,
            "effective_range_count": 0,
            "normalization": self._relationship_normalization_template(),
            "selected_ranges": [],
            "route": "not_run",
            "selection_status": "not_run",
        }
        resolution_status, canonical, boundary = self._resolve_symbol_extent(requested_symbol)
        manifest["resolution_status"] = resolution_status
        manifest["resolved_canonical_symbol"] = canonical
        if boundary is not None:
            manifest["definition_source_extent"] = self._symbol_extent_public(boundary)
        if resolution_status != "selected":
            manifest["selection_status"] = resolution_status
            return self._symbol_selection_not_run(manifest)

        assert canonical is not None and boundary is not None
        relationship_status, admitted, accounting = self._symbol_direct_relationships(canonical)
        manifest["direct_relationship_accounting"].update(accounting)
        manifest["direct_structural_relationship_count"] = int(
            accounting["valid_direct_relationship_count"]
        )
        manifest["admitted_direct_relationships"] = admitted
        if relationship_status != "selected":
            manifest["selection_status"] = relationship_status
            return self._symbol_selection_not_run(manifest)

        function_key = (
            str(boundary.file), str(boundary.name),
            int(boundary.start_line), int(boundary.end_line),
        )
        candidates: list[dict[str, Any]] = [{
            **self._symbol_extent_public(boundary),
            "_function_key": function_key,
        }]
        direct_edges = [
            (row["file"], row["line"], row["caller"], row["callee"])
            for row in admitted
        ]
        candidate_status, direct_candidates = self._relationship_candidate_ranges(direct_edges)
        if candidate_status != "selected":
            manifest["selection_status"] = candidate_status
            return self._symbol_selection_not_run(manifest)
        candidates.extend(direct_candidates)
        manifest["candidate_range_count"] = len(candidates)
        normalized, ranges = self._normalize_relationship_ranges(candidates)
        manifest["normalization"].update(normalized["normalization"])
        manifest["effective_range_count"] = len(ranges)
        manifest["selected_ranges"] = ranges
        manifest["selection_status"] = normalized["selection_status"]
        if manifest["selection_status"] != "selected":
            return self._symbol_selection_not_run(manifest)
        manifest["route"] = "single_range" if len(ranges) == 1 else "multi_range"

        # All selected ranges are independently reread before ledger lookup so
        # a cache hit can never validate stale index content.
        items = [self._claim_item(source_range) for source_range in ranges]
        semantic_request = self._symbol_semantic_request(requested_symbol, canonical)
        ledger = self._claim_ledger()
        verdict = ledger.lookup_symbol_investigation(
            topic_id, topic, requested_symbol, canonical, semantic_request, items
        )
        if verdict is None:
            if len(items) == 1:
                raw_verdict = self._claim_verifier().verify_claim(topic, semantic_request, items[0])
            else:
                raw_verdict = self._claim_verifier().verify_bundle_claim(topic, semantic_request, items)
            verdict = self._validate_verdict(raw_verdict)
            ledger.record_symbol_investigation_decision(
                topic_id, topic, requested_symbol, canonical, semantic_request, items, verdict
            )
        ledger_hit = verdict.get("ledger_hit") is True
        semantic_manifest = {
            "schema_version": 1,
            "mode": "symbol_investigation_semantic_verification",
            "topic_id": topic_id,
            "topic": topic,
            "requested_symbol": requested_symbol,
            "resolved_canonical_symbol": canonical,
            "semantic_request": semantic_request,
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
                "requested_range_count": len(ranges),
                "effective_range_count": len(items),
                "repository_read_count": len(items),
                "unrelated_topic_count": 0,
                "model_turn_count": int(verdict.get("model_turns", 0)),
                "ledger_hit": ledger_hit,
            },
        }
        verification = {
            "manifest": semantic_manifest,
            "answer": self._render_claim_answer(verdict),
            "verdict": verdict,
        }
        return {
            "selection_manifest": manifest,
            "verification": verification,
            "answer": verification["answer"],
        }

    @staticmethod
    def _subsystem_root(value: str) -> str:
        """Accept only a normalized, repository-relative directory root."""
        path = Path(value)
        if (
            path.is_absolute() or path.as_posix() != value or "\x00" in value
            or "\\" in value or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("subsystem must be a repository-relative normalized directory root")
        return value.rstrip("/")

    @staticmethod
    def _path_in_file_set(file: str, admitted_files: set[str]) -> bool:
        return file in admitted_files

    def _subsystem_exact_file_listing(self, file: str) -> list[str]:
        """List one exact repository path without discovering a directory tree."""
        return RepositoryIndex._normalize_listing(list_files(file))

    def _subsystem_files(self, subsystem: str, raw_files: Any) -> list[str]:
        """Validate and canonically order an explicit, non-recursive file set.

        File existence is checked through the repository file-list capability,
        never by a source read.  Passing an exact file path gives no directory
        traversal or search fallback to this operation.
        """
        if not isinstance(raw_files, list) or not raw_files:
            raise ValueError("files must be a non-empty array")
        if len(raw_files) > MAX_SUBSYSTEM_FILES:
            raise ValueError(
                f"files must contain at most {MAX_SUBSYSTEM_FILES} entries"
            )
        admitted: list[str] = []
        seen: set[str] = set()
        for value in raw_files:
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError("each files entry must be a non-empty normalized relative path")
            path = Path(value)
            if (
                path.is_absolute() or path.as_posix() != value or "\x00" in value
                or "\\" in value or any(part in {"", ".", ".."} for part in path.parts)
            ):
                raise ValueError("files entries must be normalized paths relative to subsystem")
            file = f"{subsystem}/{value}"
            if not file.startswith(subsystem + "/"):
                raise ValueError("files entry escapes subsystem")
            if file in seen:
                raise ValueError("duplicate files entry")
            try:
                listing = self._subsystem_exact_file_listing(file)
            except Exception as exc:
                raise ValueError("files entry could not be validated") from exc
            if file not in listing:
                # This covers nonexistent paths, directories, disallowed paths,
                # and any listing that cannot attest to this exact source file.
                raise ValueError("files entry must name an existing source file within subsystem")
            seen.add(file)
            admitted.append(file)
        return sorted(admitted)

    @staticmethod
    def _subsystem_topic_tokens(topic: str) -> set[str]:
        # This is an intentionally mechanical selector, not a semantic search:
        # it cannot read source or discover paths, and Qwen never influences it.
        return {
            token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9]*", topic)
            if len(token) >= 3
        }

    @staticmethod
    def _subsystem_symbol_tokens(symbol: str) -> set[str]:
        expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", symbol)
        return {
            token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9]*", expanded)
            if len(token) >= 3
        }

    def _subsystem_inventory(self, admitted_files: set[str]) -> tuple[str, list[Any], dict[str, Any]]:
        """Inventory only complete indexed function extents in the admitted files."""
        try:
            functions = list(self._idx().functions)
        except (AttributeError, TypeError, ValueError):
            return "malformed_subsystem_inventory", [], {}
        inventory = []
        for boundary in functions:
            try:
                file = str(boundary.file)
                name = str(boundary.name)
                start, end = int(boundary.start_line), int(boundary.end_line)
            except (AttributeError, TypeError, ValueError):
                return "malformed_subsystem_inventory", [], {}
            if not self._path_in_file_set(file, admitted_files):
                continue
            if not name or start < 1 or end < start:
                return "malformed_subsystem_inventory", [], {}
            try:
                enclosing = self._idx().function_for_line(file, start)
            except (AttributeError, TypeError, ValueError):
                return "malformed_subsystem_inventory", [], {}
            if enclosing != boundary:
                return "malformed_subsystem_inventory", [], {}
            inventory.append(boundary)
        inventory.sort(key=lambda b: (str(b.file), int(b.start_line), int(b.end_line), str(b.name)))
        if len(inventory) > MAX_SUBSYSTEM_INVENTORY_SYMBOLS:
            return "subsystem_inventory_over_limit", [], {"inventory_count": len(inventory)}
        if not inventory:
            return "no_subsystem_symbols", [], {"inventory_count": 0}
        names = [str(boundary.name) for boundary in inventory]
        duplicate_count = len(names) - len(set(names))
        # Duplicate short/heuristic names elsewhere in a large directory do not
        # make an otherwise exact topic match unusable.  They remain accounted
        # for, and become fail-closed only if the selection would admit one.
        return "selected", inventory, {
            "inventory_count": len(inventory),
            "ambiguous_inventory_symbol_count": duplicate_count,
        }

    def _subsystem_selected_symbols(self, topic: str, inventory: list[Any]) -> tuple[str, list[Any]]:
        tokens = self._subsystem_topic_tokens(topic)
        scored = []
        for boundary in inventory:
            score = len(tokens & self._subsystem_symbol_tokens(str(boundary.name)))
            if score:
                scored.append((score, str(boundary.file), int(boundary.start_line), str(boundary.name), boundary))
        if not scored:
            return "no_topic_matched_symbols", []
        scored.sort(key=lambda row: (-row[0], row[1], row[2], row[3]))
        # Select only the strongest deterministic lexical-match tier.  This is
        # a declared selection rule, not a truncation of weaker matches, and
        # keeps a broad directory inventory from becoming an evidence dump.
        highest_score = scored[0][0]
        selected_rows = [row for row in scored if row[0] == highest_score]
        if len(selected_rows) > MAX_SUBSYSTEM_SELECTED_SYMBOLS:
            return "too_many_topic_matched_symbols", []
        selected = [row[-1] for row in selected_rows]
        names = [str(row.name) for row in selected]
        if len(set(names)) != len(names):
            return "ambiguous_topic_matched_symbols", []
        if any(int(row.end_line) - int(row.start_line) + 1 > 500 for row in selected):
            return "selected_symbol_extent_over_limit", []
        return "selected", selected

    def _subsystem_relationships(
        self, selected: list[Any], inventory: list[Any], admitted_files: set[str],
    ) -> tuple[str, list[dict[str, Any]], dict[str, int]]:
        """Admit only direct edges whose site and endpoint extents are admitted."""
        by_name: dict[str, list[Any]] = {}
        for boundary in inventory:
            by_name.setdefault(str(boundary.name), []).append(boundary)
        inventory_by_name = {
            name: boundaries[0] for name, boundaries in by_name.items()
            if len(boundaries) == 1
        }
        selected_names = {str(boundary.name) for boundary in selected}
        admitted: dict[tuple[str, int, str, str], dict[str, Any]] = {}
        raw_count = invalid_count = outside_count = unattributed_count = 0
        for name in sorted(selected_names):
            try:
                incoming = list(self._idx().callers_of(name))
                outgoing = list(self._idx().callees_of(name))
            except (AttributeError, TypeError, ValueError):
                return "malformed_subsystem_relationship", [], {"raw_relationship_count": raw_count, "invalid_relationship_count": 1, "unattributed_relationship_count": unattributed_count, "external_relationship_count": outside_count, "admitted_relationship_count": 0}
            for direction, sites in (("incoming", incoming), ("outgoing", outgoing)):
                for site in sites:
                    raw_count += 1
                    file = self._edge_field(site, "file")
                    line = self._edge_field(site, "line")
                    caller = self._edge_field(site, "caller")
                    callee = self._edge_field(site, "callee")
                    try:
                        if isinstance(line, bool): raise ValueError
                        line = int(line)
                    except (TypeError, ValueError):
                        invalid_count += 1
                        continue
                    if not isinstance(file, str) or not file or line < 1:
                        invalid_count += 1
                        continue
                    if not isinstance(caller, str) or not caller or not isinstance(callee, str) or not callee:
                        # Lightweight indexing can surface a declaration-like
                        # call with no enclosing function.  It is neither
                        # evidence nor an edge to follow, so exclude it rather
                        # than treating unrelated index noise as evidence.
                        unattributed_count += 1
                        continue
                    if (direction == "incoming" and callee != name) or (direction == "outgoing" and caller != name):
                        invalid_count += 1
                        continue
                    if not self._path_in_file_set(file, admitted_files):
                        outside_count += 1
                        continue
                    caller_boundary = inventory_by_name.get(caller)
                    callee_boundary = inventory_by_name.get(callee)
                    if (
                        caller_boundary is None or callee_boundary is None
                        or not self._path_in_file_set(str(caller_boundary.file), admitted_files)
                        or not self._path_in_file_set(str(callee_boundary.file), admitted_files)
                    ):
                        outside_count += 1
                        continue
                    try:
                        site_boundary = self._idx().function_for_line(file, line)
                    except (AttributeError, TypeError, ValueError):
                        invalid_count += 1
                        continue
                    if site_boundary != caller_boundary:
                        invalid_count += 1
                        continue
                    key = (file, line, caller, callee)
                    row = admitted.setdefault(key, {"file": file, "line": line, "caller": caller, "callee": callee, "directions": []})
                    row["directions"].append(direction)
        rows = []
        for key in sorted(admitted):
            row = admitted[key]
            row["directions"] = sorted(set(row["directions"]))
            rows.append(row)
        accounting = {"raw_relationship_count": raw_count, "invalid_relationship_count": invalid_count, "unattributed_relationship_count": unattributed_count, "external_relationship_count": outside_count, "admitted_relationship_count": len(rows), "maximum_admitted_relationships": MAX_SUBSYSTEM_SELECTED_RELATIONSHIPS}
        if invalid_count:
            return "malformed_subsystem_relationship", [], accounting
        if len(rows) > MAX_SUBSYSTEM_SELECTED_RELATIONSHIPS:
            return "too_many_subsystem_relationships", [], accounting
        return "selected", rows, accounting

    @staticmethod
    def _subsystem_semantic_request(subsystem: str, files: list[str]) -> str:
        return (
            "Explain only the source-visible behavior relevant to the supplied topic "
            f"within directory-root subsystem {subsystem!r} and admitted files {files!r}, using only the supplied "
            "server-selected evidence. Do not infer unseen files, symbols, runtime "
            "behavior, transitive effects, external relationships, database state, or "
            "other repository content."
        )

    def _subsystem_selection_not_run(self, manifest: dict[str, Any]) -> dict[str, Any]:
        status = str(manifest["selection_status"])
        return {"selection_manifest": manifest, "verification": {"status": "not_run", "reason": status, "model_turn_count": 0, "ledger_hit": False, "repository_read_count": 0}, "answer": f"No subsystem evidence was selected: {status}."}

    def _investigate_subsystem(self, request: dict[str, Any]) -> dict[str, Any]:
        self._require_exact_fields(request, {"op", "topic_id", "topic", "subsystem", "files"})
        topic_id = self._required_string(request, "topic_id")
        topic = self._required_string(request, "topic")
        subsystem = self._subsystem_root(self._required_string(request, "subsystem"))
        files = self._subsystem_files(subsystem, request.get("files"))
        admitted_files = set(files)
        manifest: dict[str, Any] = {
            "schema_version": 2, "mode": "bounded_subsystem_investigation_selection",
            "subsystem": subsystem, "files": files, "admitted_file_count": len(files),
            "maximum_files": MAX_SUBSYSTEM_FILES, "inventory_count": 0,
            "ambiguous_inventory_symbol_count": 0,
            "maximum_inventory_symbols": MAX_SUBSYSTEM_INVENTORY_SYMBOLS,
            "maximum_selected_symbols": MAX_SUBSYSTEM_SELECTED_SYMBOLS,
            "maximum_selected_relationships": MAX_SUBSYSTEM_SELECTED_RELATIONSHIPS,
            "maximum_evidence_ranges": MAX_SUBSYSTEM_EVIDENCE_RANGES,
            "maximum_evidence_lines": MAX_SUBSYSTEM_EVIDENCE_LINES,
            "selected_symbols": [], "relationships": [], "relationship_accounting": {},
            "candidate_range_count": 0, "effective_range_count": 0, "effective_evidence_lines": 0,
            "normalization": self._relationship_normalization_template(), "selected_ranges": [],
            "route": "not_run", "selection_status": "not_run",
        }
        inventory_status, inventory, inventory_meta = self._subsystem_inventory(admitted_files)
        manifest.update(inventory_meta)
        if inventory_status != "selected":
            manifest["selection_status"] = inventory_status
            return self._subsystem_selection_not_run(manifest)
        selected_status, selected = self._subsystem_selected_symbols(topic, inventory)
        if selected_status != "selected":
            manifest["selection_status"] = selected_status
            return self._subsystem_selection_not_run(manifest)
        manifest["selected_symbols"] = [self._symbol_extent_public(boundary) | {"symbol": str(boundary.name)} for boundary in selected]
        relationship_status, relationships, accounting = self._subsystem_relationships(selected, inventory, admitted_files)
        manifest["relationship_accounting"] = accounting
        manifest["relationships"] = relationships
        if relationship_status != "selected":
            manifest["selection_status"] = relationship_status
            return self._subsystem_selection_not_run(manifest)
        candidates = [{**self._symbol_extent_public(boundary), "_function_key": (str(boundary.file), str(boundary.name), int(boundary.start_line), int(boundary.end_line))} for boundary in selected]
        relationship_edges = [(row["file"], row["line"], row["caller"], row["callee"]) for row in relationships]
        edge_status, edge_candidates = self._relationship_candidate_ranges(relationship_edges)
        if edge_status != "selected":
            manifest["selection_status"] = edge_status
            return self._subsystem_selection_not_run(manifest)
        candidates.extend(edge_candidates)
        manifest["candidate_range_count"] = len(candidates)
        normalized, ranges = self._normalize_relationship_ranges(candidates)
        manifest["normalization"].update(normalized["normalization"])
        manifest["effective_range_count"] = len(ranges)
        manifest["selected_ranges"] = ranges
        manifest["selection_status"] = normalized["selection_status"]
        if manifest["selection_status"] != "selected":
            return self._subsystem_selection_not_run(manifest)
        lines = sum(int(row["end"]) - int(row["start"]) + 1 for row in ranges)
        manifest["effective_evidence_lines"] = lines
        if len(ranges) > MAX_SUBSYSTEM_EVIDENCE_RANGES:
            manifest["selection_status"] = "too_many_subsystem_ranges"
            return self._subsystem_selection_not_run(manifest)
        if lines > MAX_SUBSYSTEM_EVIDENCE_LINES:
            manifest["selection_status"] = "too_many_subsystem_evidence_lines"
            return self._subsystem_selection_not_run(manifest)
        manifest["route"] = "single_range" if len(ranges) == 1 else "multi_range"
        items = [self._claim_item(source_range) for source_range in ranges]
        semantic_request = self._subsystem_semantic_request(subsystem, files)
        ledger = self._claim_ledger()
        verdict = ledger.lookup_subsystem_investigation(topic_id, topic, subsystem, files, semantic_request, items)
        if verdict is None:
            raw_verdict = self._claim_verifier().verify_claim(topic, semantic_request, items[0]) if len(items) == 1 else self._claim_verifier().verify_bundle_claim(topic, semantic_request, items)
            verdict = self._validate_verdict(raw_verdict)
            ledger.record_subsystem_investigation_decision(topic_id, topic, subsystem, files, semantic_request, items, verdict)
        ledger_hit = verdict.get("ledger_hit") is True
        semantic_manifest = {"schema_version": 2, "mode": "subsystem_investigation_semantic_verification", "topic_id": topic_id, "topic": topic, "subsystem": subsystem, "files": files, "semantic_request": semantic_request, "evidence": [{"file": item["file"], "start": item["start"], "end": item["end"], "source_sha256": source_hash(item["excerpt"])} for item in items], "verification": {"supports": verdict.get("supports") is True, "ledger_hit": ledger_hit, "model_turns": int(verdict.get("model_turns", 0))}, "metrics": {"requested_range_count": len(ranges), "effective_range_count": len(items), "effective_evidence_lines": lines, "repository_read_count": len(items), "unrelated_topic_count": 0, "model_turn_count": int(verdict.get("model_turns", 0)), "ledger_hit": ledger_hit}}
        verification = {"manifest": semantic_manifest, "answer": self._render_claim_answer(verdict), "verdict": verdict}
        return {"selection_manifest": manifest, "verification": verification, "answer": verification["answer"]}

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
