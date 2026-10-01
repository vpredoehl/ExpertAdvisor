#!/usr/bin/env python3
"""Read-only machine interface for external RepositoryAgent controllers.

This module intentionally exposes repository facts and verified-ledger state,
not benchmark answers. It performs no writes to the ExpertAdvisor repository,
no shell execution, no database access, and no Git mutation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from expertadvisor_agent import list_files, read_file, search
from .evidence import VerifiedEvidenceLedger
from .repository_index import RepositoryIndex


class CodexRepositoryInterface:
    def __init__(self, *, ledger_path: str | None = None):
        self._index: RepositoryIndex | None = None
        self._ledger_path = ledger_path

    def _idx(self) -> RepositoryIndex:
        if self._index is None:
            self._index = RepositoryIndex().build()
        return self._index

    def capabilities(self) -> dict[str, Any]:
        return {
            "protocol": "expertadvisor.repository.readonly.v1",
            "read_only": True,
            "operations": [
                "capabilities", "list_files", "search", "read", "index_stats",
                "resolve_symbol", "function_for_line", "relationship",
                "trace_calls", "source_excerpt", "ledger_records",
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
        raise ValueError(f"unsupported operation: {op!r}")

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
    parser=argparse.ArgumentParser(description="Read-only JSON interface to ExpertAdvisor RepositoryAgent primitives")
    parser.add_argument("--request", help="single JSON request object")
    parser.add_argument("--ledger", help="optional evidence-ledger path")
    args=parser.parse_args()
    iface=CodexRepositoryInterface(ledger_path=args.ledger)
    try:
        req=json.loads(args.request) if args.request else json.load(__import__('sys').stdin)
        print(json.dumps({"ok":True,"result":iface.dispatch(req)}, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"ok":False,"error":type(exc).__name__,"message":str(exc)}, sort_keys=True))
        return 2

if __name__ == "__main__": raise SystemExit(main())
