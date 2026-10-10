#!/usr/bin/env python3
"""Dependency-free stdio MCP adapter for the repository-read-only RepositoryAgent interface."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "expertadvisor-repository-agent", "version": "1.12.0"}
PROFILE_ENVIRONMENT_VARIABLE = "EXPERTADVISOR_REPOSITORY_AGENT_MCP_PROFILE"
DEFAULT_TOOL_PROFILE = "full"

TOOLS = [
    {
        "name": "capabilities",
        "description": "Return RepositoryAgent protocol capabilities and forbidden capabilities.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "list_files",
        "description": "List allowed ExpertAdvisor source files under an optional repository-relative prefix.",
        "inputSchema": {
            "type": "object",
            "properties": {"prefix": {"type": "string"}},
            "additionalProperties": False,
        },
    },
    {
        "name": "search",
        "description": "Search allowed ExpertAdvisor source files for a textual pattern.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
    },
    {
        "name": "read",
        "description": "Read an exact line range from an allowed ExpertAdvisor source file (maximum 500 lines).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file": {"type": "string"},
                "start": {"type": "integer", "minimum": 1},
                "end": {"type": "integer", "minimum": 1},
            },
            "required": ["file"],
            "additionalProperties": False,
        },
    },
    {
        "name": "index_stats",
        "description": "Return structural RepositoryIndex statistics.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "resolve_symbol",
        "description": "Resolve a source symbol to definitions, callers, callees, and references in the structural index.",
        "inputSchema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
            "additionalProperties": False,
        },
    },
    {
        "name": "function_for_line",
        "description": "Return the indexed enclosing function for a repository source file and line.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file": {"type": "string"},
                "line": {"type": "integer", "minimum": 1},
            },
            "required": ["file", "line"],
            "additionalProperties": False,
        },
    },
    {
        "name": "relationship",
        "description": "Return indexed call edges matching a caller and callee.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "caller": {"type": "string"},
                "callee": {"type": "string"},
            },
            "required": ["caller", "callee"],
            "additionalProperties": False,
        },
    },
    {
        "name": "trace_calls",
        "description": "Trace bounded structural call relationships from a symbol.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "max_depth": {"type": "integer", "minimum": 1, "maximum": 4},
                "max_nodes": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            "required": ["symbol"],
            "additionalProperties": False,
        },
    },
    {
        "name": "source_excerpt",
        "description": "Return exact source text for a repository file line range, maximum 500 lines.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file": {"type": "string"},
                "start": {"type": "integer", "minimum": 1},
                "end": {"type": "integer", "minimum": 1},
            },
            "required": ["file", "start", "end"],
            "additionalProperties": False,
        },
    },
    {
        "name": "ledger_records",
        "description": "Read legacy category-based verified-evidence ledger records with optional filters.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "topic_id": {"type": "string"},
                "category": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 500},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "verify_source_claim",
        "description": "Semantically verify whether one exact server-retrieved source range establishes a proposed claim. The first uncached call lazily loads Qwen; the decision is stored only in the controlled claim ledger.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic_id": {"type": "string"},
                "topic": {"type": "string"},
                "claim": {"type": "string"},
                "file": {"type": "string"},
                "start": {"type": "integer", "minimum": 1},
                "end": {"type": "integer", "minimum": 1},
            },
            "required": ["topic_id", "topic", "claim", "file", "start", "end"],
            "additionalProperties": False,
        },
    },
    {
        "name": "investigate_source_claim",
        "description": "Run one bounded, source-grounded claim investigation. It retrieves only the requested source range server-side, verifies the claim with Qwen when uncached, records only the claim-ledger decision, and returns a deterministic manifest and answer.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic_id": {"type": "string"},
                "topic": {"type": "string"},
                "claim": {"type": "string"},
                "file": {"type": "string"},
                "start": {"type": "integer", "minimum": 1},
                "end": {"type": "integer", "minimum": 1},
            },
            "required": ["topic_id", "topic", "claim", "file", "start", "end"],
            "additionalProperties": False,
        },
    },
    {
        "name": "verify_source_bundle_claim",
        "description": "Semantically verify whether 2-8 exact server-retrieved source ranges form a source-visible path establishing a proposed claim.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic_id": {"type": "string"},
                "topic": {"type": "string"},
                "claim": {"type": "string"},
                "ranges": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "properties": {
                            "file": {"type": "string"},
                            "start": {"type": "integer", "minimum": 1},
                            "end": {"type": "integer", "minimum": 1},
                        },
                        "required": ["file", "start", "end"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["topic_id", "topic", "claim", "ranges"],
            "additionalProperties": False,
        },
    },
    {
        "name": "investigate_source_bundle_claim",
        "description": "Run one bounded multi-range source-claim investigation. The server reads exactly 2-8 explicit non-duplicate, non-overlapping ranges, canonicalizes their order, verifies only their combined evidence with Qwen when uncached, and returns a deterministic manifest and answer.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic_id": {"type": "string"},
                "topic": {"type": "string"},
                "claim": {"type": "string"},
                "ranges": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "properties": {
                            "file": {"type": "string"},
                            "start": {"type": "integer", "minimum": 1},
                            "end": {"type": "integer", "minimum": 1},
                        },
                        "required": ["file", "start", "end"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["topic_id", "topic", "claim", "ranges"],
            "additionalProperties": False,
        },
    },
    {
        "name": "investigate_relationship_claim",
        "description": "Select bounded direct caller-to-callee source ranges server-side from the structural index, then semantically verify only that exact reread evidence. Qwen cannot search, select ranges, or navigate the repository.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic_id": {"type": "string"},
                "topic": {"type": "string"},
                "claim": {"type": "string"},
                "caller": {"type": "string"},
                "callee": {"type": "string"},
            },
            "required": ["topic_id", "topic", "claim", "caller", "callee"],
            "additionalProperties": False,
        },
    },
    {
        "name": "investigate_operation_relationship_claim",
        "description": "Select and verify one indexed operation binding, field invocation, or operation_implementation_call. Implementation claims require an exact operation selector and reread the complete assigned callback, independently of the owner's symbol-wide relationship count. No callback relationship is an unconditional direct call.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic_id": {"type": "string"}, "topic": {"type": "string"},
                "claim": {"type": "string"}, "caller": {"type": "string"},
                "callee": {"type": "string"},
                "relationship_kind": {"type": "string", "enum": ["operation_binding", "operation_invocation", "operation_implementation_call"]},
                "operation": {"type": "string", "minLength": 1},
            },
            "required": ["topic_id", "topic", "claim", "caller", "callee", "relationship_kind"],
            "additionalProperties": False,
        },
    },
    {
        "name": "investigate_relationship_chain_claim",
        "description": "Structurally validate every adjacent hop of a caller-supplied acyclic 3-5 symbol direct-call chain, then semantically verify only the server-selected exact reread evidence. Repeated symbols are rejected; Qwen cannot discover paths, search, select ranges, or navigate the repository.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic_id": {"type": "string"},
                "topic": {"type": "string"},
                "claim": {"type": "string"},
                "path": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 5,
                    "items": {"type": "string"},
                },
            },
            "required": ["topic_id", "topic", "claim", "path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "investigate_relationship_set_claim",
        "description": "Structurally validate a caller-supplied set of 2-5 explicit direct relationships, then semantically verify only globally normalized server-selected exact reread evidence. Qwen cannot discover relationships, search, select ranges, or navigate the repository.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic_id": {"type": "string"},
                "topic": {"type": "string"},
                "claim": {"type": "string"},
                "relationships": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 5,
                    "items": {
                        "type": "object",
                        "properties": {
                            "caller": {"type": "string"},
                            "callee": {"type": "string"},
                        },
                        "required": ["caller", "callee"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["topic_id", "topic", "claim", "relationships"],
            "additionalProperties": False,
        },
    },
    {
        "name": "discover_catalog_targets",
        "description": "Return non-evidentiary canonical file and function identities from one bounded indexed directory scope. This deterministic catalog operation never returns source text, line contents, semantic conclusions, or search matches; use a bounded investigation before making repository-derived behavioral claims.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "scope": {"type": "string", "minLength": 1},
                "query_groups": {
                    "type": "array", "minItems": 1, "maxItems": 4,
                    "items": {
                        "type": "array", "minItems": 1, "maxItems": 4,
                        "items": {"type": "string", "minLength": 3, "maxLength": 32},
                    },
                },
            },
            "required": ["scope", "query_groups"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_catalog_children",
        "description": "Return one deterministic, metadata-only page of direct indexed catalog child scopes and source files. This operation never returns source text, excerpts, search results, or semantic conclusions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "scope": {"type": "string", "minLength": 1},
                "cursor": {"type": "string", "minLength": 1},
                "limit": {"type": "integer", "minimum": 1, "maximum": 32},
            },
            "required": ["scope"],
            "additionalProperties": False,
        },
    },
    {
        "name": "discover_relationship_paths",
        "description": "Return deterministic shortest structural call paths within one direct indexed directory scope. This metadata-only operation returns no source text or behavioral evidence; use investigate_relationship_chain_claim or another bounded evidentiary investigation before behavioral claims.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "scope": {"type": "string", "minLength": 1},
                "from": {"type": "string", "minLength": 1},
                "to": {"type": "string", "minLength": 1},
                "max_hops": {"type": "integer", "minimum": 1, "maximum": 4},
            },
            "required": ["scope", "from", "to", "max_hops"],
            "additionalProperties": False,
        },
    },
    {
        "name": "discover_operation_relationship_paths",
        "description": "Discover metadata-only operation bindings, field invocations, or operation_implementation_call relationships. Implementation discovery requires operation. An invocation can traverse a three-edge static-slot-compatible callback context with explicit binding_owner and operation; this does not establish runtime object wiring or unconditional execution. All existing scope and traversal limits apply.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "scope": {"type": "string", "minLength": 1}, "from": {"type": "string", "minLength": 1},
                "to": {"type": "string", "minLength": 1},
                "max_hops": {"type": "integer", "minimum": 1, "maximum": 4},
                "relationship_kind": {"type": "string", "enum": ["operation_binding", "operation_invocation", "operation_implementation_call"]},
                "operation": {"type": "string", "minLength": 1},
                "binding_owner": {"type": "string", "minLength": 1},
            },
            "required": ["scope", "from", "to", "max_hops", "relationship_kind"],
            "additionalProperties": False,
        },
    },
    {
        "name": "investigate_symbol",
        "description": "Resolve one exact indexed symbol and run a bounded server-owned source investigation. The server selects and rereads all evidence; Qwen cannot select files, ranges, or relationships.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic_id": {"type": "string", "minLength": 1},
                "topic": {"type": "string", "minLength": 1},
                "symbol": {"type": "string", "minLength": 1},
            },
            "required": ["topic_id", "topic", "symbol"],
            "additionalProperties": False,
        },
    },
    {
        "name": "investigate_subsystem",
        "description": "Investigate one explicit repository-relative directory root and required bounded file set using deterministic structural inventory and server-selected exact reread evidence. File entries are normalized relative paths under subsystem; Qwen cannot search, select paths, symbols, relationships, or evidence.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic_id": {"type": "string", "minLength": 1},
                "topic": {"type": "string", "minLength": 1},
                "subsystem": {"type": "string", "minLength": 1},
                "files": {
                    "type": "array", "minItems": 1, "maxItems": 16,
                    "items": {"type": "string", "minLength": 1},
                },
            },
            "required": ["topic_id", "topic", "subsystem", "files"],
            "additionalProperties": False,
        },
    },
    {
        "name": "verified_claims",
        "description": "Read claim-verification ledger records with optional status/topic/claim filters.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "topic_id": {"type": "string"},
                "claim": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 500},
            },
            "additionalProperties": False,
        },
    },
]

# This is the sole MCP admission policy.  The active profile is selected once
# when the stdio server starts; both tools/list and tools/call consult the same
# selected tuple, so a client cannot use an unadvertised operation by calling it
# directly.  The assisted profile contains only planning-facing investigation
# workflows.  Their lower-level verification helpers remain in the full
# development/debugging profile because the investigation workflows add the
# deterministic manifests and server-owned evidence-selection boundary.
TOOL_PROFILES: dict[str, tuple[str, ...]] = {
    "full": tuple(tool["name"] for tool in TOOLS),
    "codex_assisted": (
        "list_catalog_children",
        "discover_catalog_targets",
        "discover_relationship_paths",
        "discover_operation_relationship_paths",
        "investigate_source_claim",
        "investigate_source_bundle_claim",
        "investigate_relationship_claim",
        "investigate_operation_relationship_claim",
        "investigate_relationship_chain_claim",
        "investigate_relationship_set_claim",
        "investigate_symbol",
        "investigate_subsystem",
    ),
}
_TOOLS_BY_NAME = {tool["name"]: tool for tool in TOOLS}
_UNKNOWN_PROFILE_TOOLS = {
    profile: sorted(set(names) - set(_TOOLS_BY_NAME))
    for profile, names in TOOL_PROFILES.items()
    if set(names) - set(_TOOLS_BY_NAME)
}
if _UNKNOWN_PROFILE_TOOLS:
    raise RuntimeError(f"MCP tool profile contains unknown tools: {_UNKNOWN_PROFILE_TOOLS}")


def resolve_tool_profile(profile: str | None = None) -> str:
    """Resolve and validate the immutable MCP profile selected at startup."""
    selected = profile
    if selected is None:
        selected = os.environ.get(PROFILE_ENVIRONMENT_VARIABLE, DEFAULT_TOOL_PROFILE)
    if not isinstance(selected, str) or selected not in TOOL_PROFILES:
        allowed = ", ".join(sorted(TOOL_PROFILES))
        raise ValueError(f"invalid MCP tool profile {selected!r}; expected one of: {allowed}")
    return selected


class StdioMCPServer:
    def __init__(self, *, profile: str | None = None) -> None:
        self.tool_profile = resolve_tool_profile(profile)
        self._tools = tuple(_TOOLS_BY_NAME[name] for name in TOOL_PROFILES[self.tool_profile])
        self._tools_by_name = {tool["name"]: tool for tool in self._tools}
        # Validate profile admission before importing the runtime dependency, so
        # an invalid startup profile always fails closed at this boundary.
        from .codex_interface import CodexRepositoryInterface
        self.iface = CodexRepositoryInterface()

    @staticmethod
    def _read_message() -> dict[str, Any] | None:
        # MCP stdio transport uses one UTF-8 JSON-RPC message per line.
        # stdout must contain no non-MCP output.
        while True:
            line = sys.stdin.buffer.readline()
            if not line:
                return None
            if not line.strip():
                continue
            value = json.loads(line.decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("MCP message must be a JSON object")
            return value

    @staticmethod
    def _write_message(message: dict[str, Any]) -> None:
        body = json.dumps(message, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        sys.stdout.buffer.write(body + b"\n")
        sys.stdout.buffer.flush()

    @staticmethod
    def _result(req_id: Any, result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    @staticmethod
    def _error(req_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}

    @staticmethod
    def _validate_schema(value: Any, schema: dict[str, Any], *, location: str) -> None:
        """Enforce the closed subset of JSON Schema advertised by MCP tools."""
        expected_type = schema.get("type")
        if expected_type == "object":
            if not isinstance(value, dict):
                raise ValueError(f"{location} must be an object")
            properties = schema.get("properties", {})
            if schema.get("additionalProperties") is False:
                unexpected = sorted(set(value) - set(properties))
                if unexpected:
                    raise ValueError(f"unexpected {location} fields: {', '.join(unexpected)}")
            missing = [name for name in schema.get("required", []) if name not in value]
            if missing:
                raise ValueError(f"missing required {location} fields: {', '.join(missing)}")
            for name, item_schema in properties.items():
                if name in value:
                    StdioMCPServer._validate_schema(
                        value[name], item_schema, location=f"{location}.{name}"
                    )
            return
        if expected_type == "array":
            if not isinstance(value, list):
                raise ValueError(f"{location} must be an array")
            minimum = schema.get("minItems")
            maximum = schema.get("maxItems")
            if minimum is not None and len(value) < minimum:
                raise ValueError(f"{location} must contain at least {minimum} items")
            if maximum is not None and len(value) > maximum:
                raise ValueError(f"{location} must contain at most {maximum} items")
            item_schema = schema.get("items")
            if isinstance(item_schema, dict):
                for index, item in enumerate(value):
                    StdioMCPServer._validate_schema(item, item_schema, location=f"{location}[{index}]")
            return
        if expected_type == "string":
            if not isinstance(value, str):
                raise ValueError(f"{location} must be a string")
            minimum = schema.get("minLength")
            if minimum is not None and len(value) < minimum:
                raise ValueError(f"{location} must contain at least {minimum} characters")
            maximum = schema.get("maxLength")
            if maximum is not None and len(value) > maximum:
                raise ValueError(f"{location} must contain at most {maximum} characters")
        if expected_type == "integer" and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError(f"{location} must be an integer")

    @staticmethod
    def _validate_tool_arguments(tool: dict[str, Any], arguments: dict[str, Any]) -> None:
        """Enforce the advertised closed tool schemas before dispatching.

        The interface remains responsible for operation-specific semantic
        validation; this prevents unadvertised fields from altering dispatch.
        """
        schema = tool["inputSchema"]
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            unexpected = sorted(set(arguments) - set(properties))
            if unexpected:
                raise ValueError(f"unexpected tool arguments: {', '.join(unexpected)}")
        missing = [name for name in schema.get("required", []) if name not in arguments]
        if missing:
            raise ValueError(f"missing required tool arguments: {', '.join(missing)}")
        StdioMCPServer._validate_schema(arguments, schema, location="tool arguments")

    def _handle_request(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        method = msg.get("method")
        req_id = msg.get("id")

        # Notifications have no id and receive no response.
        if req_id is None:
            return None

        if method == "initialize":
            return self._result(req_id, {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": SERVER_INFO,
            })

        if method == "ping":
            return self._result(req_id, {})

        if method == "tools/list":
            return self._result(req_id, {"tools": list(self._tools)})

        if method == "tools/call":
            params = msg.get("params") or {}
            if not isinstance(params, dict):
                return self._error(req_id, -32602, "tools/call params must be an object")
            name = str(params.get("name", "")).strip()
            arguments = params.get("arguments") or {}
            if not isinstance(arguments, dict):
                return self._error(req_id, -32602, "tool arguments must be an object")
            tool = self._tools_by_name.get(name)
            if tool is None:
                return self._result(req_id, {
                    "content": [{"type": "text", "text": f"unsupported tool: {name}"}],
                    "isError": True,
                })
            try:
                self._validate_tool_arguments(tool, arguments)
                result = self.iface.dispatch({"op": name, **arguments})
                return self._result(req_id, {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(result, sort_keys=True, ensure_ascii=False),
                    }],
                    "structuredContent": result,
                    "isError": False,
                })
            except Exception as exc:
                return self._result(req_id, {
                    "content": [{
                        "type": "text",
                        "text": f"{type(exc).__name__}: {exc}",
                    }],
                    "isError": True,
                })

        return self._error(req_id, -32601, f"method not found: {method}")

    def run(self) -> int:
        while True:
            try:
                msg = self._read_message()
                if msg is None:
                    return 0
                response = self._handle_request(msg)
                if response is not None:
                    self._write_message(response)
            except EOFError:
                return 0
            except Exception as exc:
                # stderr is safe for diagnostics; stdout is reserved for MCP framing.
                print(f"repository_agent_mcp: {type(exc).__name__}: {exc}", file=sys.stderr)
                return 2


def main() -> int:
    parser = argparse.ArgumentParser(
        description="RepositoryAgent stdio MCP server with a startup-fixed tool profile."
    )
    parser.add_argument(
        "--profile",
        help=("MCP tool profile (full or codex_assisted). Defaults to "
              f"${PROFILE_ENVIRONMENT_VARIABLE} or {DEFAULT_TOOL_PROFILE}."),
    )
    args = parser.parse_args()
    try:
        return StdioMCPServer(profile=args.profile).run()
    except ValueError as exc:
        print(f"repository_agent_mcp: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
