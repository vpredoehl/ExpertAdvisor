#!/usr/bin/env python3
"""Dependency-free stdio MCP adapter for the repository-read-only RepositoryAgent interface."""
from __future__ import annotations

import json
import sys
from typing import Any

from .codex_interface import CodexRepositoryInterface

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "expertadvisor-repository-agent", "version": "1.6.0"}

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


class StdioMCPServer:
    def __init__(self) -> None:
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
        if expected_type == "string" and not isinstance(value, str):
            raise ValueError(f"{location} must be a string")
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
            return self._result(req_id, {"tools": TOOLS})

        if method == "tools/call":
            params = msg.get("params") or {}
            if not isinstance(params, dict):
                return self._error(req_id, -32602, "tools/call params must be an object")
            name = str(params.get("name", "")).strip()
            arguments = params.get("arguments") or {}
            if not isinstance(arguments, dict):
                return self._error(req_id, -32602, "tool arguments must be an object")
            if name not in {tool["name"] for tool in TOOLS}:
                return self._result(req_id, {
                    "content": [{"type": "text", "text": f"unsupported tool: {name}"}],
                    "isError": True,
                })
            try:
                tool = next(tool for tool in TOOLS if tool["name"] == name)
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
    return StdioMCPServer().run()


if __name__ == "__main__":
    raise SystemExit(main())
