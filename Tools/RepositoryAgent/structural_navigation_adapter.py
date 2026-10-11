"""Opt-in, CPU-only structural navigation for the investigation tool protocol.

The underlying interface is private and lazy. Only the four operations below
can reach it; navigation never verifies evidence or persists a decision.
"""
from __future__ import annotations

import json
import os
import re

from .codex_interface import CodexRepositoryInterface


OPERATIONS = frozenset({
    "discover_catalog_targets", "list_catalog_children", "resolve_symbol", "trace_calls",
})
NAVIGATION_MARKER = "NON-EVIDENTIARY STRUCTURAL NAVIGATION"
MAX_NAVIGATION_OUTPUT = 30000


def structural_navigation_enabled() -> bool:
    """Enable only with the explicit experiment setting, never by default."""
    return os.environ.get("EA_STRUCTURAL_NAVIGATION") == "1"


def _text(value, name, maximum):
    if (not isinstance(value, str) or not value.strip() or len(value) > maximum
            or any(ord(character) < 32 or ord(character) == 127 for character in value)):
        raise ValueError(f"{name} must be a non-empty string of at most {maximum} characters without controls")
    return value


def _integer(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer between {low} and {high}")
    return value


def validate_call(call) -> dict:
    """Return the complete canonical tool request without initializing an index."""
    if not isinstance(call, dict) or any(not isinstance(key, str) for key in call):
        raise ValueError("request must be an object with string fields")
    tool = call.get("tool")
    if not isinstance(tool, str) or tool not in OPERATIONS:
        raise ValueError("unsupported structural-navigation tool")
    fields = {
        "discover_catalog_targets": {"scope", "query_groups"},
        "list_catalog_children": {"scope", "cursor", "limit"},
        "resolve_symbol": {"symbol"},
        "trace_calls": {"symbol", "max_depth", "max_nodes"},
    }[tool] | {"tool"}
    if set(call) - fields:
        raise ValueError("unexpected request fields: " + ", ".join(sorted(set(call) - fields)))
    validated = {"tool": tool}
    if tool in {"discover_catalog_targets", "list_catalog_children"}:
        scope = _text(call.get("scope"), "scope", 512)
        validated["scope"] = CodexRepositoryInterface._discovery_scope(scope)
    else:
        validated["symbol"] = _text(call.get("symbol"), "symbol", 256).strip()
    if tool == "discover_catalog_targets":
        validated["query_groups"] = CodexRepositoryInterface._discovery_query_groups(call.get("query_groups"))
    elif tool == "list_catalog_children":
        validated["limit"] = _integer(call.get("limit", 16), "limit", 1, 32)
        if "cursor" in call:
            cursor = _text(call["cursor"], "cursor", 4096)
            if not re.fullmatch(r"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", cursor):
                raise ValueError("cursor must be an opaque catalog cursor")
            validated["cursor"] = cursor
    elif tool == "trace_calls":
        validated["max_depth"] = _integer(call.get("max_depth", 2), "max_depth", 1, 4)
        validated["max_nodes"] = _integer(call.get("max_nodes", 50), "max_nodes", 1, 100)
    return validated


def investigation_prompt(base: str) -> str:
    """Extend controller policy without editing any benchmark-owned prompt."""
    if not structural_navigation_enabled():
        return base
    return base + """

EA_STRUCTURAL_NAVIGATION=1 adds four tools to the legacy tools listed above.
Use the same protocol: exactly one JSON object with "tool", never "op".
Only these fields are supported (optional fields have the defaults shown):
- {"tool":"discover_catalog_targets","scope":"Sources","query_groups":[["catalog"]]}
  scope: normalized repository-relative directory selector, 1-512 characters;
  a unique directory basename is also accepted. Discovery uses immediate files
  in that scope, not descendants. query_groups: 1-4 groups of 1-4 identifier
  terms, each 3-32 ASCII letters/digits starting with a letter. Terms in a group
  are ANDed; groups are ORed; matching is case-insensitive; no duplicates.
- {"tool":"list_catalog_children","scope":"Sources","limit":16}
  scope: same selector; limit: integer 1-32 (default 16); optional cursor:
  opaque next_cursor string from an earlier page, at most 4096 characters.
  Lists immediate children only. Use next_cursor with the same scope and this
  controller session; null means the final page. Stale cursors fail closed.
- {"tool":"resolve_symbol","symbol":"Example"}
  symbol: non-empty string, at most 256 characters, no control characters.
- {"tool":"trace_calls","symbol":"Example","max_depth":2,"max_nodes":50}
  symbol: same string; max_depth: integer 1-4 (default 2); max_nodes:
  integer 1-100 (default 50), bounding returned trace rows, not unique symbols.
All outputs are NON-EVIDENTIARY STRUCTURAL NAVIGATION. The structural index is
heuristic; names and call traces do not establish semantics or execution.
Trace views are depth/row bounded and may include callback bindings; completeness
is never asserted. Read the indicated source with the legacy read tool before
citing it. Navigation cannot satisfy source-line provenance or verification.
Oversized responses fail closed with no partial result. Narrow the symbol/query
or reduce the catalog page size. Unsupported fields and operations are rejected.
"""


class StructuralNavigationAdapter:
    def __init__(self):
        self._interface = None

    def execute_tool(self, call) -> str:
        if not structural_navigation_enabled():
            return f"TOOL ERROR: {NAVIGATION_MARKER}: disabled (requires EA_STRUCTURAL_NAVIGATION=1)"
        try:
            validated = validate_call(call)
            if self._interface is None:
                self._interface = CodexRepositoryInterface()
            request = {"op": validated["tool"], **{key: value for key, value in validated.items() if key != "tool"}}
            data = self._interface.dispatch(request)
            response = {
                "evidentiary_status": "non_evidentiary",
                "request": validated,
                "qualification": "Heuristic structural metadata only; source reads and existing verification are required.",
                "data": data,
            }
            if validated["tool"] == "trace_calls":
                response["trace_limits"] = {
                    "max_depth": validated["max_depth"],
                    "max_nodes": validated["max_nodes"],
                    "row_limit_reached": len(data["trace"]) >= validated["max_nodes"],
                    "completeness": "not_asserted",
                }
                response["qualification"] += (
                    " Trace rows omit relationship kinds and may include callback bindings rather than direct invocations;"
                    " they do not prove execution. The depth/row-bounded view is not a complete relationship set."
                )
            # One JSON line escapes even source-shaped metadata. Preserve the
            # entire backend result (including qualifiers and cursor), or none.
            rendered = NAVIGATION_MARKER + "\n" + json.dumps(
                response, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
            )
            if len(rendered) > MAX_NAVIGATION_OUTPUT:
                raise ValueError("response exceeds 30000 characters; no partial result returned; narrow the request")
            return rendered
        except (ValueError, TypeError, KeyError, OSError) as exc:
            # Do not echo untrusted metadata/exception text as numbered source.
            detail = json.dumps(str(exc), ensure_ascii=True)
            return f"TOOL ERROR: {NAVIGATION_MARKER}: " + detail[:1000]
        except Exception as exc:
            # Unexpected index/interface failures must not abort an investigation.
            # Do not expose internal exception messages or filesystem paths.
            return (
                f"TOOL ERROR: {NAVIGATION_MARKER}: "
                f"internal navigation failure ({type(exc).__name__})"
            )
