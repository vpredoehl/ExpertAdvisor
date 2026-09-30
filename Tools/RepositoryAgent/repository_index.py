#!/usr/bin/env python3
"""
General read-only source index for the ExpertAdvisor repository agent.

Phase 2 goals:
- controller-owned symbol definitions and references
- approximate C/C++ function boundaries
- call-site lookup
- bounded caller -> callee traversal
- no embeddings and no repository writes

The module consumes the existing expertadvisor_agent.py API:
    list_files(prefix="")
    read_file(name, start=1, end=200)

It keeps the index in memory by default.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import hashlib
import json
import re
from typing import Iterable, Optional

from expertadvisor_agent import list_files, read_file


_ALLOWED_SUFFIXES = (".cpp", ".cc", ".cxx", ".h", ".hpp", ".metal")
_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_CONTROL_WORDS = {
    "if", "for", "while", "switch", "catch", "return", "sizeof",
    "alignof", "decltype", "static_cast", "dynamic_cast",
    "reinterpret_cast", "const_cast", "new", "delete",
}
_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_:]*)\s*\(")

# Deliberately heuristic: this recognizes ordinary C/C++ definitions while
# rejecting common control-flow constructs. Exact source remains authoritative.
_FUNCTION_HEAD_RE = re.compile(
    r"""(?x)
    ^\s*
    (?!if\b|for\b|while\b|switch\b|catch\b)
    (?:
        (?:template\s*<[^;{}]*>\s*)?
        (?:[\w:<>,~*&\[\]\s]+\s+)?
    )
    (?P<name>
        (?:[A-Za-z_][A-Za-z0-9_]*::)*
        ~?[A-Za-z_][A-Za-z0-9_]*
        |operator\s*[^\s(]+
    )
    \s*\([^;{}]*\)
    (?:\s*(?:const|noexcept|override|final|->\s*[^{}]+))*\s*
    \{
    """
)


@dataclass(frozen=True)
class SourceLocation:
    file: str
    line: int


@dataclass(frozen=True)
class FunctionBoundary:
    file: str
    name: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class SymbolOccurrence:
    file: str
    line: int
    symbol: str
    kind: str
    function: Optional[str] = None


@dataclass(frozen=True)
class CallSite:
    file: str
    line: int
    caller: Optional[str]
    callee: str


class RepositoryIndex:
    def __init__(self) -> None:
        self.files: list[str] = []
        self.file_hashes: dict[str, str] = {}
        self.functions: list[FunctionBoundary] = []
        self.definitions: dict[str, list[SymbolOccurrence]] = defaultdict(list)
        self.references: dict[str, list[SymbolOccurrence]] = defaultdict(list)
        self.calls_by_callee: dict[str, list[CallSite]] = defaultdict(list)
        self.calls_by_caller: dict[str, list[CallSite]] = defaultdict(list)
        self._lines: dict[str, list[str]] = {}
        self._functions_by_file: dict[str, list[FunctionBoundary]] = defaultdict(list)

    @staticmethod
    def _normalize_listing(raw) -> list[str]:
        """Accept the existing tool's list, JSON, or newline-oriented output."""
        if isinstance(raw, list):
            return [str(x) for x in raw]
        if isinstance(raw, dict):
            for key in ("files", "results", "items"):
                value = raw.get(key)
                if isinstance(value, list):
                    out = []
                    for item in value:
                        if isinstance(item, str):
                            out.append(item)
                        elif isinstance(item, dict):
                            name = item.get("name") or item.get("path") or item.get("file")
                            if name:
                                out.append(str(name))
                    return out
        text = str(raw)
        try:
            decoded = json.loads(text)
            if decoded is not raw:
                return RepositoryIndex._normalize_listing(decoded)
        except Exception:
            pass
        out = []
        for line in text.splitlines():
            candidate = line.strip().lstrip("-* ").strip()
            if candidate.endswith(_ALLOWED_SUFFIXES):
                out.append(candidate)
        return out

    @staticmethod
    def _normalize_read(raw) -> str:
        if isinstance(raw, str):
            return raw
        if isinstance(raw, dict):
            for key in ("content", "text", "output"):
                if isinstance(raw.get(key), str):
                    return raw[key]
        return str(raw)

    @staticmethod
    def _strip_number_prefix(line: str) -> str:
        # Existing repo reader may return "123: source" or "123 | source".
        return re.sub(r"^\s*\d+\s*(?::|\|)\s?", "", line)

    def _read_whole_file(self, name: str, chunk_lines: int = 500) -> list[str]:
        lines: list[str] = []
        start = 1
        while True:
            raw = self._normalize_read(read_file(name, start, start + chunk_lines - 1))
            chunk = raw.splitlines()
            if not chunk:
                break
            cleaned = [self._strip_number_prefix(x) for x in chunk]
            lines.extend(cleaned)
            if len(chunk) < chunk_lines:
                break
            start += chunk_lines
        return lines

    @staticmethod
    def _brace_delta(line: str) -> int:
        # Sufficient for navigation; semantic evidence still comes from exact source.
        line = re.sub(r'"(?:\\.|[^"\\])*"', '""', line)
        line = re.sub(r"'(?:\\.|[^'\\])*'", "''", line)
        line = re.sub(r"//.*$", "", line)
        return line.count("{") - line.count("}")

    @staticmethod
    def _candidate_function_name(signature: str) -> Optional[str]:
        """Return the function name immediately associated with the parameter list.

        This is intentionally syntax-oriented rather than scheduler-aware.  It
        tolerates attributes, namespaces/classes, destructors, operators,
        multiline signatures, trailing return types, and constructor initializer
        lists.  It rejects ordinary control-flow heads.
        """
        compact = re.sub(r"\s+", " ", signature).strip()
        compact = re.sub(r"\[\[[^\]]*\]\]\s*", "", compact)

        # Locate each opening parenthesis and keep the last plausible callable
        # name immediately before it.  The last plausible one handles return
        # types such as decltype(...) without confusing them with the function.
        candidates = []
        for m in re.finditer(
            r"(?P<name>(?:[A-Za-z_][A-Za-z0-9_]*::)*~?[A-Za-z_][A-Za-z0-9_]*"
            r"|operator\s*(?:\[\]|\(\)|[^\s(]+))\s*\(",
            compact,
        ):
            name = re.sub(r"\s+", " ", m.group("name")).strip()
            if name.split("::")[-1] not in _CONTROL_WORDS:
                candidates.append(name)
        return candidates[-1] if candidates else None

    def _discover_functions(self, file: str, lines: list[str]) -> list[FunctionBoundary]:
        """Discover C/C++/Metal function bodies with brace-balanced boundaries.

        This is a lightweight navigation index, not a parser.  Exact retrieved
        source remains authoritative for evidence.
        """
        result: list[FunctionBoundary] = []
        i = 0
        n = len(lines)

        while i < n:
            matched_name = None
            open_line = None

            # Function heads in this repository can span several lines.  Stop at
            # an obvious declaration terminator before a body begins.
            for width in range(1, 13):
                if i + width > n:
                    break
                window = lines[i:i + width]
                candidate = " ".join(x.strip() for x in window)

                # Ignore preprocessor records and obvious type/control bodies.
                first = lines[i].lstrip()
                if first.startswith("#"):
                    break

                brace_pos = candidate.find("{")
                semi_pos = candidate.find(";")
                if semi_pos >= 0 and (brace_pos < 0 or semi_pos < brace_pos):
                    break

                if brace_pos >= 0:
                    head = candidate[:brace_pos + 1]
                    name = self._candidate_function_name(head)
                    if name:
                        # Reject class/struct/enum/namespace declarations whose
                        # body happens to contain a constructor-like token.
                        prefix = head.split("(", 1)[0]
                        if not re.search(r"\b(class|struct|enum|namespace)\b", prefix):
                            matched_name = name
                            open_line = i + width - 1
                    break

            if not matched_name or open_line is None:
                i += 1
                continue

            depth = 0
            seen_open = False
            end_line = open_line
            for j in range(i, n):
                delta = self._brace_delta(lines[j])
                if "{" in lines[j]:
                    seen_open = True
                depth += delta
                if seen_open and depth <= 0:
                    end_line = j
                    break

            result.append(
                FunctionBoundary(file, matched_name, i + 1, end_line + 1)
            )
            i = max(i + 1, end_line + 1)

        return result

    @staticmethod
    def _function_at(functions: list[FunctionBoundary], line: int) -> Optional[FunctionBoundary]:
        for fn in functions:
            if fn.start_line <= line <= fn.end_line:
                return fn
        return None

    def build(self, prefixes: Iterable[str] = ("Headers", "Sources", "LSTM")) -> "RepositoryIndex":
        names: set[str] = set()
        for prefix in prefixes:
            names.update(self._normalize_listing(list_files(prefix)))
        self.files = sorted(x for x in names if x.endswith(_ALLOWED_SUFFIXES))

        for file in self.files:
            lines = self._read_whole_file(file)
            self._lines[file] = lines
            self.file_hashes[file] = hashlib.sha256(
                "\n".join(lines).encode("utf-8", "replace")
            ).hexdigest()
            funcs = self._discover_functions(file, lines)
            self.functions.extend(funcs)
            self._functions_by_file[file].extend(funcs)

        for fn in self.functions:
            short = fn.name.split("::")[-1]
            occ = SymbolOccurrence(fn.file, fn.start_line, short, "definition", fn.name)
            self.definitions[short].append(occ)
            if fn.name != short:
                self.definitions[fn.name].append(
                    SymbolOccurrence(fn.file, fn.start_line, fn.name, "definition", fn.name)
                )

        for file, lines in self._lines.items():
            funcs = self._functions_by_file[file]
            for line_no, line in enumerate(lines, 1):
                owner = self._function_at(funcs, line_no)
                owner_name = owner.name if owner else None

                for ident in set(_IDENTIFIER_RE.findall(line)):
                    self.references[ident].append(
                        SymbolOccurrence(file, line_no, ident, "reference", owner_name)
                    )

                for match in _CALL_RE.finditer(line):
                    callee = match.group(1)
                    short = callee.split("::")[-1]
                    if short in _CONTROL_WORDS:
                        continue
                    call = CallSite(file, line_no, owner_name, callee)
                    self.calls_by_callee[short].append(call)
                    if callee != short:
                        self.calls_by_callee[callee].append(call)
                    if owner_name:
                        self.calls_by_caller[owner_name].append(call)
                        owner_short = owner_name.split("::")[-1]
                        if owner_short != owner_name:
                            self.calls_by_caller[owner_short].append(call)
        return self

    def find_definitions(self, symbol: str) -> list[SymbolOccurrence]:
        return list(self.definitions.get(symbol, ()))

    def find_references(self, symbol: str, limit: int = 100) -> list[SymbolOccurrence]:
        return list(self.references.get(symbol, ()))[:limit]

    def function_for_line(self, file: str, line: int) -> Optional[FunctionBoundary]:
        return self._function_at(self._functions_by_file.get(file, ()), line)

    def callers_of(self, symbol: str) -> list[CallSite]:
        return list(self.calls_by_callee.get(symbol, ()))

    def callees_of(self, function: str) -> list[CallSite]:
        return list(self.calls_by_caller.get(function, ()))

    def resolve_symbol(self, symbol: str) -> dict:
        """Return controller-owned structural facts for a symbol."""
        return {
            "symbol": symbol,
            "definitions": [asdict(x) for x in self.find_definitions(symbol)],
            "references": [asdict(x) for x in self.find_references(symbol)],
            "callers": [asdict(x) for x in self.callers_of(symbol)],
            "callees": [asdict(x) for x in self.callees_of(symbol)],
        }

    def relationship(self, caller: str, callee: str) -> list[CallSite]:
        """Return direct source-visible call edges from caller to callee."""
        target_short = callee.split("::")[-1]
        seen = set()
        out = []
        for site in self.callees_of(caller):
            if site.callee == callee or site.callee.split("::")[-1] == target_short:
                key = (site.file, site.line, site.caller, site.callee)
                if key not in seen:
                    seen.add(key)
                    out.append(site)
        return out

    def trace_calls(self, start: str, max_depth: int = 4, max_nodes: int = 100) -> list[dict]:
        """Breadth-first source-visible call traversal; no semantic claims are inferred."""
        out: list[dict] = []
        q = deque([(start, 0)])
        seen = {start}
        while q and len(out) < max_nodes:
            caller, depth = q.popleft()
            if depth >= max_depth:
                continue
            for site in self.callees_of(caller):
                callee = site.callee
                row = {
                    "depth": depth + 1,
                    "caller": caller,
                    "callee": callee,
                    "file": site.file,
                    "line": site.line,
                }
                out.append(row)
                short = callee.split("::")[-1]
                if short not in seen:
                    seen.add(short)
                    q.append((short, depth + 1))
                if len(out) >= max_nodes:
                    break
        return out

    def source_excerpt(self, file: str, start_line: int, end_line: int) -> str:
        lines = self._lines[file]
        start = max(1, start_line)
        end = min(len(lines), end_line)
        return "\n".join(f"{i}: {lines[i-1]}" for i in range(start, end + 1))

    def stats(self) -> dict:
        return {
            "files": len(self.files),
            "functions": len(self.functions),
            "definition_keys": len(self.definitions),
            "reference_keys": len(self.references),
            "call_edges": len({
                (site.file, site.line, site.caller, site.callee)
                for sites in self.calls_by_caller.values()
                for site in sites
            }),
        }


def build_repository_index() -> RepositoryIndex:
    return RepositoryIndex().build()


if __name__ == "__main__":
    idx = build_repository_index()
    print(json.dumps(idx.stats(), indent=2, sort_keys=True))
