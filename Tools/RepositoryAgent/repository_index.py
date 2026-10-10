#!/usr/bin/env python3
"""
General read-only source index for the ExpertAdvisor repository agent.

Phase 2 goals:
- controller-owned symbol definitions and references
- approximate C/C++ function boundaries
- call-site lookup
- bounded caller -> callee traversal
- no embeddings and no repository writes

The module consumes the startup-selected source_reader API:
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

from .source_reader import list_files, read_file


_ALLOWED_SUFFIXES = (".cpp", ".cc", ".cxx", ".h", ".hpp", ".metal")
_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_CONTROL_WORDS = {
    "if", "for", "while", "switch", "catch", "return", "sizeof",
    "alignof", "decltype", "static_cast", "dynamic_cast",
    "reinterpret_cast", "const_cast", "new", "delete",
}
_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_:]*)\s*\(")
# A call in a lambda body is not a direct invocation by the named function
# that lexically encloses the lambda.  The index remains deliberately
# lightweight, but this common operation/callback form has a structural
# distinction that must survive into relationship traversal.
_LAMBDA_OPEN_RE = re.compile(
    r"\[[^\[\]\n]*\]\s*(?:\([^{};]*\)\s*)?"
    r"(?:mutable\s*)?(?:noexcept(?:\s*\([^{};]*\))?\s*)?"
    r"(?:->\s*[^{};]+\s*)?\{"
)
# Scheduler composition uses named operation fields rather than direct calls:
# `operations.runCycle = [&] { return RunSchedulerOnce(...); };`.  The
# production-shaped singular local `operation.runCheckpointAnalysis = ...` is
# also an explicit operation receiver. Record only the deliberately narrow
# one-call lambda form. A more complex lambda is not an operation binding in
# this lightweight index (fail closed).
_OPERATION_BINDING_RE = re.compile(
    r"\b(?P<operation>(?:operation|operations_?|[A-Za-z_][A-Za-z0-9_]*Operations)(?:\s*\.\s*[A-Za-z_][A-Za-z0-9_]*)+)\s*="
    r"\s*\[[^\[\]\n]*\]\s*(?:\([^{};]*\)\s*)?"
    r"(?:mutable\s*)?(?:noexcept(?:\s*\([^{};]*\))?\s*)?"
    r"(?:->\s*[^{};]+\s*)?\{"
)
_OPERATION_INVOKE_RE = re.compile(
    r"\b(?P<operation>operations_?|[A-Za-z_][A-Za-z0-9_]*Operations)\s*\.\s*"
    r"(?P<field>[A-Za-z_][A-Za-z0-9_]*)\s*\("
)
# A dotted call is not a direct function edge until its receiver's static type
# is established by one of the deliberately narrow forms below.
_MEMBER_CALL_RE = re.compile(
    r"\b(?P<receiver>[A-Za-z_][A-Za-z0-9_]*)\s*\.\s*"
    r"(?P<method>[A-Za-z_][A-Za-z0-9_]*)\s*\("
)
# `EA::SchedulerCore::SchedulerEngine().run(...)`.  Arguments are deliberately
# limited to one flat parenthesized expression; complex construction fails
# closed rather than attempting C++ parsing.
_EXPLICIT_TEMPORARY_MEMBER_CALL_RE = re.compile(
    r"\b(?P<type>(?:[A-Za-z_][A-Za-z0-9_]*::)*[A-Za-z_][A-Za-z0-9_]*)"
    r"\s*\([^(){};]*\)\s*\.\s*"
    r"(?P<method>[A-Za-z_][A-Za-z0-9_]*)\s*\("
)
# `SchedulerCycleService service{...}; service.runOnce()`.  `auto`, aliases,
# pointers, references, and inferred/dynamic receivers are intentionally out
# of scope.
_EXPLICIT_LOCAL_OBJECT_RE = re.compile(
    r"\b(?P<type>(?:[A-Za-z_][A-Za-z0-9_]*::)*[A-Za-z_][A-Za-z0-9_]*)"
    r"\s+(?P<receiver>[A-Za-z_][A-Za-z0-9_]*)\s*\{"
)
_NAMESPACE_OPEN_RE = re.compile(
    r"\bnamespace\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*)\s*\{"
)
# This is deliberately a schema, not a general aggregate-initializer rule.
# Production installs these exact callbacks positionally in the declared
# CheckpointAnalysisOperations order.  Recording any other aggregate as an
# operation binding would be speculative.
_OPERATION_AGGREGATE_FIELDS = {
    "CheckpointAnalysisOperations": (
        "claim", "afterClaim", "execute", "afterWork", "finalize",
        "generateReports",
    ),
}
_OPERATION_AGGREGATE_OPEN_RE = re.compile(
    r"\b(?:(?:[A-Za-z_][A-Za-z0-9_]*::)*)"
    r"(?P<type>CheckpointAnalysisOperations)\s+"
    r"(?P<receiver>operation|operations_?)\s*\{"
)

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
    relationship_kind: str = "direct_invocation"
    operation: Optional[str] = None


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
                    # A candidate window must not cross a closing brace from a
                    # preceding scope/function into the next function head.
                    if "}" in candidate[:brace_pos]:
                        break
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

    @staticmethod
    def _callback_body_offsets(lines: list[str]) -> list[tuple[int, int]]:
        """Return source offsets occupied by syntactically apparent lambdas.

        This is intentionally conservative metadata: a call matched within one
        of these spans is retained as a callback invocation, never promoted to
        a direct edge of its lexical named-function owner.  If braces cannot be
        balanced, the remainder is treated as callback code (fail closed).
        """
        text = "\n".join(lines)
        spans: list[tuple[int, int]] = []
        for match in _LAMBDA_OPEN_RE.finditer(text):
            open_at = match.end() - 1
            depth = 0
            close_at = len(text)
            for offset in range(open_at, len(text)):
                char = text[offset]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        close_at = offset + 1
                        break
            spans.append((open_at, close_at))
        return spans

    @staticmethod
    def _operation_binding_spans(lines: list[str]) -> list[tuple[int, int, str, int | None]]:
        """Return conservative operation-field lambda bindings.

        The final member is the sole callable target offset when the lambda has
        exactly one apparent call; otherwise it is None.  No lexical lambda is
        promoted to a binding merely because it mentions a function name.
        """
        text = "\n".join(lines)
        spans = []
        for match in _OPERATION_BINDING_RE.finditer(text):
            open_at = match.end() - 1
            depth, close_at = 0, None
            for offset in range(open_at, len(text)):
                if text[offset] == "{": depth += 1
                elif text[offset] == "}":
                    depth -= 1
                    if depth == 0:
                        close_at = offset + 1
                        break
            if close_at is None:
                continue
            calls = [m.start() for m in _CALL_RE.finditer(text, open_at + 1, close_at - 1)
                     if m.group(1).split("::")[-1] not in _CONTROL_WORDS]
            spans.append((open_at, close_at, re.sub(r"\s+", "", match.group("operation")),
                          calls[0] if len(calls) == 1 else None))

        # The production checkpoint-analysis adapter uses a complete, ordered
        # aggregate rather than member assignments.  Admit only this declared
        # operation type, only its exact field count/order, and only a slot
        # whose lambda has one apparent callable target.
        for match in _OPERATION_AGGREGATE_OPEN_RE.finditer(text):
            fields = _OPERATION_AGGREGATE_FIELDS[match.group("type")]
            aggregate_open = match.end() - 1
            depth, aggregate_close = 0, None
            for offset in range(aggregate_open, len(text)):
                if text[offset] == "{": depth += 1
                elif text[offset] == "}":
                    depth -= 1
                    if depth == 0:
                        aggregate_close = offset
                        break
            if aggregate_close is None:
                continue
            slots: list[tuple[int, int]] = []
            slot_start, brace_depth, paren_depth, bracket_depth = aggregate_open + 1, 0, 0, 0
            for offset in range(slot_start, aggregate_close):
                char = text[offset]
                if char == "{": brace_depth += 1
                elif char == "}": brace_depth -= 1
                elif char == "(": paren_depth += 1
                elif char == ")": paren_depth -= 1
                elif char == "[": bracket_depth += 1
                elif char == "]": bracket_depth -= 1
                elif (char == "," and brace_depth == 0 and paren_depth == 0
                      and bracket_depth == 0):
                    slots.append((slot_start, offset))
                    slot_start = offset + 1
            slots.append((slot_start, aggregate_close))
            if len(slots) != len(fields):
                continue
            receiver = re.sub(r"\s+", "", match.group("receiver"))
            for field, (start, end) in zip(fields, slots):
                lambda_match = _LAMBDA_OPEN_RE.search(text, start, end)
                if lambda_match is None or text[start:lambda_match.start()].strip():
                    continue
                lambda_open = lambda_match.end() - 1
                lambda_depth, lambda_close = 0, None
                for offset in range(lambda_open, end):
                    if text[offset] == "{": lambda_depth += 1
                    elif text[offset] == "}":
                        lambda_depth -= 1
                        if lambda_depth == 0:
                            lambda_close = offset + 1
                            break
                if lambda_close is None or text[lambda_close:end].strip():
                    continue
                calls = [candidate.start() for candidate in
                         _CALL_RE.finditer(text, lambda_open + 1, lambda_close - 1)
                         if candidate.group(1).split("::")[-1] not in _CONTROL_WORDS]
                spans.append((lambda_open, lambda_close, receiver + "." + field,
                              calls[0] if len(calls) == 1 else None))
        return spans

    @staticmethod
    def _offset_line(lines: list[str], offset: int) -> int:
        """Translate a joined-source offset to its one-based source line."""
        if offset < 0:
            raise ValueError("source offset must be non-negative")
        return "\n".join(lines)[:offset].count("\n") + 1

    @staticmethod
    def _balanced_close(text: str, open_at: int) -> int | None:
        """Return the matching brace offset for one already-open brace."""
        depth = 0
        for offset in range(open_at, len(text)):
            if text[offset] == "{":
                depth += 1
            elif text[offset] == "}":
                depth -= 1
                if depth == 0:
                    return offset
        return None

    def positional_operation_binding_proof(
        self, caller: str, callee: str
    ) -> dict | None:
        """Return source proof ranges for one already-admitted positional binding.

        This does not classify edges or infer a new relationship.  It merely
        recovers the declaration and complete initializer required to verify an
        existing ``operation_binding`` edge whose provenance is the deliberately
        admitted positional aggregate schema.
        """
        edges = [edge for edge in self.callees_of(caller)
                 if edge.relationship_kind == "operation_binding"
                 and edge.callee == callee
                 and isinstance(edge.operation, str)]
        if len(edges) != 1:
            return None
        edge = edges[0]
        lines = self._lines.get(edge.file)
        if not lines:
            return None
        text = "\n".join(lines)
        line_offsets: list[int] = []
        offset = 0
        for source_line in lines:
            line_offsets.append(offset)
            offset += len(source_line) + 1
        if not 1 <= edge.line <= len(line_offsets):
            return None
        call_offset = line_offsets[edge.line - 1] + lines[edge.line - 1].find(callee)
        if call_offset < line_offsets[edge.line - 1]:
            return None

        for match in _OPERATION_AGGREGATE_OPEN_RE.finditer(text):
            aggregate_type = match.group("type")
            fields = _OPERATION_AGGREGATE_FIELDS[aggregate_type]
            open_at = match.end() - 1
            close_at = self._balanced_close(text, open_at)
            if close_at is None or not open_at < call_offset < close_at:
                continue
            # Reuse the same admitted slot grammar as classification, then
            # require its complete field count and canonical slot identity.
            slots: list[tuple[int, int]] = []
            slot_start, brace_depth, paren_depth, bracket_depth = open_at + 1, 0, 0, 0
            for position in range(slot_start, close_at):
                char = text[position]
                if char == "{": brace_depth += 1
                elif char == "}": brace_depth -= 1
                elif char == "(": paren_depth += 1
                elif char == ")": paren_depth -= 1
                elif char == "[": bracket_depth += 1
                elif char == "]": bracket_depth -= 1
                elif char == "," and brace_depth == paren_depth == bracket_depth == 0:
                    slots.append((slot_start, position)); slot_start = position + 1
            slots.append((slot_start, close_at))
            if len(slots) != len(fields):
                return None
            # A trailing comma can otherwise manufacture an empty final slot.
            # The semantic proof is available only for the same complete
            # lambda-per-field aggregate shape admitted by the index.
            for candidate_start, candidate_end in slots:
                candidate_lambda = _LAMBDA_OPEN_RE.search(text, candidate_start, candidate_end)
                if candidate_lambda is None or text[candidate_start:candidate_lambda.start()].strip():
                    return None
                candidate_close = self._balanced_close(text, candidate_lambda.end() - 1)
                if candidate_close is None or text[candidate_close + 1:candidate_end].strip():
                    return None
            receiver = re.sub(r"\s+", "", match.group("receiver"))
            for field, (slot_start, slot_end) in zip(fields, slots):
                if receiver + "." + field != edge.operation:
                    continue
                lambda_match = _LAMBDA_OPEN_RE.search(text, slot_start, slot_end)
                if lambda_match is None:
                    return None
                lambda_close = self._balanced_close(text, lambda_match.end() - 1)
                if lambda_close is None or not (lambda_match.end() - 1 < call_offset < lambda_close):
                    return None
                # The schema definition itself is source evidence.  Require a
                # unique declaration whose declared field order exactly matches
                # the admitted schema; otherwise semantic proof is unavailable.
                declaration = self._positional_operation_schema_declaration(
                    aggregate_type, fields
                )
                if declaration is None:
                    return None
                initializer_end = close_at + 1
                while initializer_end < len(text) and text[initializer_end] in " \t":
                    initializer_end += 1
                if initializer_end < len(text) and text[initializer_end] == ";":
                    initializer_end += 1
                initializer_start = self._offset_line(lines, match.start())
                initializer_end_line = self._offset_line(lines, initializer_end - 1)
                return {
                    "relationship_kind": "operation_binding",
                    "caller": caller,
                    "callee": callee,
                    "operation": edge.operation,
                    "aggregate_type": aggregate_type,
                    "aggregate_fields": list(fields),
                    "declaration": declaration,
                    "initializer": {
                        "file": edge.file,
                        "start": initializer_start,
                        "end": initializer_end_line,
                        "selection_source_sha256": hashlib.sha256(
                            "\n".join(lines[initializer_start - 1:initializer_end_line]).encode("utf-8", "replace")
                        ).hexdigest(),
                    },
                }
        return None

    def _positional_operation_schema_declaration(
        self, aggregate_type: str, fields: tuple[str, ...]
    ) -> dict | None:
        matches: list[dict] = []
        declaration_re = re.compile(r"\bstruct\s+" + re.escape(aggregate_type) + r"\s*\{")
        for file, lines in self._lines.items():
            text = "\n".join(lines)
            for match in declaration_re.finditer(text):
                close_at = self._balanced_close(text, match.end() - 1)
                if close_at is None:
                    continue
                names = tuple(re.findall(r">\s*([A-Za-z_][A-Za-z0-9_]*)\s*;", text[match.end():close_at]))
                if names != fields:
                    continue
                start = self._offset_line(lines, match.start())
                end = self._offset_line(lines, close_at)
                matches.append({
                    "file": file,
                    "start": start,
                    "end": end,
                    "selection_source_sha256": hashlib.sha256(
                        "\n".join(lines[start - 1:end]).encode("utf-8", "replace")
                    ).hexdigest(),
                })
        return matches[0] if len(matches) == 1 else None

    def _qualified_owner(self, boundary: FunctionBoundary) -> str:
        """Recover the enclosing named namespace for exact receiver matching."""
        lines = self._lines.get(boundary.file, [])
        if not lines:
            return boundary.name.rsplit("::", 1)[0]
        text = "\n".join(lines)
        offset = sum(len(line) + 1 for line in lines[:boundary.start_line - 1])
        enclosing: list[tuple[int, str]] = []
        for match in _NAMESPACE_OPEN_RE.finditer(text):
            open_at = match.end() - 1
            depth, close_at = 0, None
            for position in range(open_at, len(text)):
                if text[position] == "{": depth += 1
                elif text[position] == "}":
                    depth -= 1
                    if depth == 0:
                        close_at = position
                        break
            if close_at is not None and open_at < offset < close_at:
                enclosing.append((open_at, match.group("name")))
        namespace = ""
        for _, name in sorted(enclosing):
            namespace = name if not namespace or "::" in name else namespace + "::" + name
        owner = boundary.name.rsplit("::", 1)[0]
        return namespace + "::" + owner if namespace else owner

    def _canonical_member_target(self, type_name: str, method: str) -> str | None:
        """Resolve a syntactically explicit receiver type to one unique method."""
        type_short = type_name.split("::")[-1]
        candidates = {
            boundary.name for boundary in self.functions
            if "::" in boundary.name
            and boundary.name.split("::")[-1] == method
            and boundary.name.rsplit("::", 1)[0].split("::")[-1] == type_short
            and (
                "::" not in type_name
                or self._qualified_owner(boundary) == type_name
            )
        }
        return next(iter(candidates)) if len(candidates) == 1 else None

    def _member_call_targets(
        self, lines: list[str], funcs: list[FunctionBoundary], line_offsets: list[int]
    ) -> dict[int, str]:
        """Return canonical direct targets for two explicit receiver forms only."""
        text = "\n".join(lines)
        targets: dict[int, str] = {}
        for function in funcs:
            start = line_offsets[function.start_line - 1]
            end = line_offsets[function.end_line - 1] + len(lines[function.end_line - 1])
            body = text[start:end]
            for match in _EXPLICIT_TEMPORARY_MEMBER_CALL_RE.finditer(body):
                target = self._canonical_member_target(
                    match.group("type"), match.group("method")
                )
                if target is not None:
                    targets[start + match.start("method")] = target

            declarations: dict[str, tuple[str, int]] = {}
            ambiguous_receivers: set[str] = set()
            for declaration in _EXPLICIT_LOCAL_OBJECT_RE.finditer(body):
                receiver = declaration.group("receiver")
                if receiver in declarations:
                    ambiguous_receivers.add(receiver)
                else:
                    declarations[receiver] = (
                        declaration.group("type"), declaration.end()
                    )
            for receiver, (type_name, declaration_end) in declarations.items():
                if receiver in ambiguous_receivers:
                    continue
                receiver_call = re.compile(
                    rf"\b{re.escape(receiver)}\s*\.\s*"
                    r"(?P<method>[A-Za-z_][A-Za-z0-9_]*)\s*\("
                )
                for call in receiver_call.finditer(body, declaration_end):
                    # A subsequent assignment means the original local type no
                    # longer establishes this receiver's identity.
                    intervening = body[declaration_end:call.start()]
                    if re.search(rf"\b{re.escape(receiver)}\s*=", intervening):
                        break
                    target = self._canonical_member_target(
                        type_name, call.group("method")
                    )
                    if target is not None:
                        targets[start + call.start("method")] = target
        return targets

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
            line_offsets: list[int] = []
            offset = 0
            for source_line in lines:
                line_offsets.append(offset)
                offset += len(source_line) + 1
            callback_spans = self._callback_body_offsets(lines)
            operation_binding_spans = self._operation_binding_spans(lines)
            member_call_targets = self._member_call_targets(lines, funcs, line_offsets)
            member_call_offsets = {
                match.start("method") for match in _MEMBER_CALL_RE.finditer("\n".join(lines))
            }
            operation_invocations = {
                match.start("field"): re.sub(r"\s+", "", match.group("operation")) + "." + match.group("field")
                for match in _OPERATION_INVOKE_RE.finditer("\n".join(lines))
            }
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
                    # A function definition signature looks call-like to the
                    # lightweight regex. Suppress that synthetic self-edge
                    # anywhere from the indexed signature start through the
                    # line containing its opening body brace. Genuine recursive
                    # calls later in the body remain indexed.
                    if owner is not None and short == owner.name.split("::")[-1]:
                        signature_has_open_brace = any(
                            "{" in lines[k - 1]
                            for k in range(owner.start_line, line_no + 1)
                        )
                        if not signature_has_open_brace:
                            continue
                        if line_no == owner.start_line and "{" in line:
                            continue
                        if "{" in line:
                            prefix = line.split("{", 1)[0]
                            if match.start() < len(prefix):
                                continue
                    absolute_offset = line_offsets[line_no - 1] + match.start()
                    binding = next((span for span in operation_binding_spans
                                    if span[0] <= absolute_offset < span[1] and span[3] == absolute_offset), None)
                    invocation = operation_invocations.get(absolute_offset)
                    member_target = member_call_targets.get(absolute_offset)
                    if binding is not None:
                        relationship_kind, operation = "operation_binding", binding[2]
                    elif invocation is not None:
                        # The selector is a callable operation identity, not a
                        # guessed function target.  It can never enter direct
                        # call discovery.
                        relationship_kind, operation, callee = "operation_invocation", invocation, invocation
                    elif member_target is not None:
                        relationship_kind, operation, callee = "direct_invocation", None, member_target
                    elif absolute_offset in member_call_offsets:
                        # A raw `foo.bar()` is not direct-call evidence unless
                        # a safe receiver-type form above resolved it.
                        relationship_kind, operation = "member_invocation", None
                    elif any(start <= absolute_offset < end for start, end in callback_spans):
                        relationship_kind, operation = "callback_invocation", None
                    else:
                        relationship_kind, operation = "direct_invocation", None
                    call = CallSite(file, line_no, owner_name, callee, relationship_kind, operation)
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
            if site.relationship_kind != "direct_invocation":
                continue
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
