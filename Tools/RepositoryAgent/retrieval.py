#!/usr/bin/env python3
"""Frozen retrieval, provenance, and generic relationship assembly."""
import json
import os
import re
from pathlib import Path
from .repository_index import build_repository_index
TOPIC_NAVIGATION = {}

def configure_topic_navigation(navigation):
    """Install benchmark-owned navigation text without importing a benchmark."""
    global TOPIC_NAVIGATION
    TOPIC_NAVIGATION = dict(navigation or {})
from expertadvisor_agent import list_files, search, read_file

MAX_TOOL_OUTPUT = 30000
MAX_READ_LINES = 500

_REPOSITORY_INDEX = None

def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}

GENERIC_INDEX_NAVIGATION_ENABLED = _env_flag("EA_GENERIC_INDEX_NAVIGATION", True)

def get_repository_index():
    """Build the generic read-only repository index once, lazily."""
    global _REPOSITORY_INDEX
    if _REPOSITORY_INDEX is None:
        print("REPOSITORY INDEX: building generic symbol/call index...")
        _REPOSITORY_INDEX = build_repository_index()
        print(
            "REPOSITORY INDEX: ready "
            + json.dumps(_REPOSITORY_INDEX.stats(), sort_keys=True)
        )
    return _REPOSITORY_INDEX

def _navigation_symbol_score(token, resolved):
    """Rank topic-derived identifiers by how likely they are to be useful code symbols.

    The score is repository-generic: it uses identifier shape and index evidence,
    never scheduler-specific symbol names or source locations.
    """
    definitions = resolved.get("definitions", [])
    callers = resolved.get("callers", [])
    callees = resolved.get("callees", [])
    score = 0
    if definitions:
        score += 12
    if callers:
        score += 5
    if callees:
        score += 5
    if re.search(r"[a-z][A-Z]|[A-Z][a-z].*[A-Z]|_", token):
        score += 8
    if token[:1].isupper():
        score += 3
    if len(token) >= 8:
        score += 2
    # Very common prose-like lowercase words are weak navigation anchors even
    # when a heuristic index happens to find a same-named variable/call token.
    if token.islower() and "_" not in token:
        score -= 8
    # Prefer symbols with a compact repository footprint over ubiquitous names.
    footprint = len(definitions) + len(callers) + len(callees)
    if 0 < footprint <= 12:
        score += 4
    elif footprint > 40:
        score -= 6
    return score

def topic_index_navigation(topic, max_symbols=14, max_rows=72):
    """Produce generic controller-owned symbol and call-chain navigation facts.

    Candidate identifiers come only from the assigned topic text. The generic
    repository index ranks those identifiers, then emits navigation in rounds
    across symbols so one noisy symbol cannot consume the entire row budget.
    Caller/callee relationships and same-function neighborhoods are navigation
    only; exact source must still pass provenance and semantic verification.
    """
    idx = get_repository_index()
    text = " ".join([
        str(topic.get("question", "")),
        str(topic.get("hints", "")),
        str(TOPIC_NAVIGATION.get(topic.get("id"), "")),
    ])
    tokens = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", text)

    seen = set()
    ranked = []
    for position, token in enumerate(tokens):
        if token in seen:
            continue
        seen.add(token)
        resolved = idx.resolve_symbol(token)
        if not (resolved["definitions"] or resolved["callers"] or resolved["callees"]):
            continue
        score = _navigation_symbol_score(token, resolved)
        ranked.append((-score, position, token, resolved))

    ranked.sort(key=lambda item: (item[0], item[1]))
    symbols = [(token, resolved) for _, _, token, resolved in ranked[:max_symbols]]

    rows = []
    emitted = set()

    def add(row):
        if row not in emitted and len(rows) < max_rows:
            emitted.add(row)
            rows.append(row)

    # Round 1: give every selected symbol a compact direct footprint before any
    # expansion. This prevents an ambiguous but highly connected identifier from
    # starving more useful topic-derived code symbols later in the ranking.
    for symbol, resolved in symbols:
        for d in resolved["definitions"][:1]:
            add(f"definition {symbol}: {d['file']}:{d['line']} function={d.get('function') or '-'}")
        for c in resolved["callers"][:2]:
            add(f"caller -> {symbol}: {c['file']}:{c['line']} caller={c.get('caller') or '-'} callee={c['callee']}")
        for c in resolved["callees"][:2]:
            add(f"{symbol} -> callee: {c['file']}:{c['line']} callee={c['callee']}")

    # Round 2: same-function neighborhoods for call sites. Nearby calls in the
    # same enclosing function frequently reveal the next control/data handoff.
    # Keep a small per-symbol quota so expansion remains balanced and generic.
    for symbol, resolved in symbols:
        neighborhood = []
        neighborhood_seen = set()
        for site in resolved["callers"][:6]:
            filename = site.get("file")
            line = site.get("line")
            if not filename or not isinstance(line, int):
                continue
            boundary = idx.function_for_line(filename, line)
            if boundary is None:
                continue
            for edge in idx.callees_of(boundary.name):
                if edge.file != filename or edge.line == line:
                    continue
                distance = edge.line - line
                if abs(distance) > 160:
                    continue
                key = (edge.file, edge.line, boundary.name, edge.callee)
                if key in neighborhood_seen:
                    continue
                neighborhood_seen.add(key)
                neighborhood.append((abs(distance), edge.line, edge.callee, boundary.name, distance, edge.file))
        neighborhood.sort(key=lambda x: (x[0], x[1], x[2]))
        for _, line, callee, function, distance, filename in neighborhood[:8]:
            add(f"same-function near {symbol}: {filename}:{line} function={function} callee={callee} distance={distance:+d}")

    # Round 3: one additional graph hop from direct callers/callees. Again use a
    # bounded per-symbol quota rather than allowing the first symbol to dominate.
    for symbol, resolved in symbols:
        adjacent = []
        for c in resolved["callers"][:4]:
            caller = c.get("caller")
            if caller and caller != "-":
                adjacent.append(caller)
        for c in resolved["callees"][:4]:
            callee = c.get("callee")
            if callee:
                adjacent.append(callee)

        adjacent_seen = set()
        emitted_for_symbol = 0
        for neighbor in adjacent:
            short = neighbor.split("::")[-1]
            if not short or short in adjacent_seen:
                continue
            adjacent_seen.add(short)
            nr = idx.resolve_symbol(short)
            for edge in nr.get("callers", [])[:2]:
                add(f"2-hop caller -> {short}: {edge['file']}:{edge['line']} caller={edge.get('caller') or '-'} callee={edge['callee']}")
                emitted_for_symbol += 1
                if emitted_for_symbol >= 6:
                    break
            if emitted_for_symbol >= 6:
                break
            for edge in nr.get("callees", [])[:2]:
                add(f"2-hop {short} -> callee: {edge['file']}:{edge['line']} callee={edge['callee']}")
                emitted_for_symbol += 1
                if emitted_for_symbol >= 6:
                    break
            if emitted_for_symbol >= 6:
                break

    if not rows:
        return "No generic index relationships resolved from this topic text."
    return "\n".join(rows[:max_rows])

def execute_tool(call):
    tool = call.get("tool")

    if tool == "list_files":
        prefix = str(call.get("prefix", ""))
        result = list_files(prefix)

    elif tool == "search":
        pattern = str(call.get("pattern", ""))

        if not pattern:
            return "TOOL ERROR: search pattern is empty"

        result = search(pattern)

    elif tool == "read":
        filename = str(call.get("file", ""))

        try:
            start = int(call.get("start", 1))
            end = int(call.get("end", start + 199))
        except (TypeError, ValueError):
            return "TOOL ERROR: start/end must be integers"

        if not filename:
            return "TOOL ERROR: filename is empty"

        if start < 1:
            return "TOOL ERROR: start must be >= 1"

        if end < start:
            return "TOOL ERROR: end must be >= start"

        if end - start + 1 > MAX_READ_LINES:
            return (
                "TOOL ERROR: read exceeds maximum range of "
                f"{MAX_READ_LINES} lines"
            )

        try:
            result = read_file(filename, start, end)
        except Exception as exc:
            return f"TOOL ERROR: {exc}"

    else:
        return f"TOOL ERROR: unknown tool {tool!r}"

    if not result:
        return "TOOL RESULT: no matches"

    if len(result) > MAX_TOOL_OUTPUT:
        result = result[:MAX_TOOL_OUTPUT]
        result += "\n[TOOL OUTPUT TRUNCATED]"

    return result

def normalize_call(call):
    tool = call.get("tool")

    if tool == "list_files":
        return (
            "list_files",
            str(call.get("prefix", "")),
        )

    if tool == "search":
        return (
            "search",
            str(call.get("pattern", "")),
        )

    if tool == "read":
        try:
            start = int(call.get("start", 1))
            end = int(call.get("end", start + 199))
        except (TypeError, ValueError):
            start = call.get("start")
            end = call.get("end")

        return (
            "read",
            str(call.get("file", "")),
            start,
            end,
        )

    return ("unknown", str(tool))

RETRIEVED_LINE_RE = re.compile(r"^\s*(\d+)\s+\|\s?(.*)$")


def record_retrieved_lines(store, filename, result):
    """Record only exact numbered source lines actually returned by read_file."""
    file_lines = store.setdefault(filename, {})

    for raw in result.splitlines():
        match = RETRIEVED_LINE_RE.match(raw)

        if match:
            file_lines[int(match.group(1))] = match.group(2)

def retrieved_evidence_excerpt(store, filename, start, end):
    """Return evidence only if every claimed line was actually retrieved."""
    if start < 1 or end < start:
        return None

    # Keep semantic verification focused and prevent oversized evidence claims.
    if end - start + 1 > 120:
        return None

    file_lines = store.get(filename, {})

    if any(
        line_number not in file_lines
        for line_number in range(start, end + 1)
    ):
        return None

    return "\n".join(
        f"{line_number:6d} | {file_lines[line_number]}"
        for line_number in range(start, end + 1)
    )

def _topic_resolved_symbols(topic, max_symbols=18):
    """Return repository-resolved identifiers derived only from topic text.

    This is a generic relationship-assembly primitive. It contains no
    repository-specific symbol names or source locations.
    """
    idx = get_repository_index()
    text = " ".join([
        str(topic.get("question", "")),
        str(topic.get("hints", "")),
        str(TOPIC_NAVIGATION.get(topic.get("id"), "")),
    ])
    tokens = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", text)
    seen = set()
    ranked = []
    for position, token in enumerate(tokens):
        if token in seen:
            continue
        seen.add(token)
        resolved = idx.resolve_symbol(token)
        if not (resolved["definitions"] or resolved["callers"] or resolved["callees"]):
            continue
        ranked.append((-_navigation_symbol_score(token, resolved), position, token, resolved))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [(token, resolved) for _, _, token, resolved in ranked[:max_symbols]]

def _semantic_identifier_terms(text):
    """Return lightweight lexical terms for generic source-symbol ranking.

    This is deliberately mechanical: split prose/identifiers, split camelCase,
    and normalize a few common English suffixes. It does not encode repository
    or benchmark-specific vocabulary.
    """
    terms = set()
    for raw in re.findall(r"[A-Za-z][A-Za-z0-9_]*", str(text or "")):
        pieces = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", raw).replace("_", " ").split()
        for piece in pieces:
            word = piece.lower()
            if len(word) < 3:
                continue
            terms.add(word)
            for suffix in ("ing", "tion", "ed", "er", "s"):
                if word.endswith(suffix) and len(word) - len(suffix) >= 4:
                    terms.add(word[:-len(suffix)])
    return terms


def _retrieved_category_symbols(topic, category, retrieved_lines, max_symbols=16):
    """Rank resolvable call symbols already present in retrieved source.

    Phase 5C could close a relationship when the extractor proposed a caller,
    but it could miss a stronger implementation that had already been read.
    This helper never broadens retrieval: it considers only callable symbols on
    exact lines already in ``retrieved_lines`` and uses the repository index only
    to resolve structural facts. Ranking favors lexical alignment with the
    missing category/topic and compact, concrete repository footprints.
    """
    idx = get_repository_index()
    target_text = " ".join([
        str(category),
        str(topic.get("title", "")),
        str(topic.get("question", "")),
        str(topic.get("hints", "")),
        str(TOPIC_NAVIGATION.get(topic.get("id"), "")),
    ])
    target_terms = _semantic_identifier_terms(target_text)
    seen = set()
    ranked = []
    position = 0
    for filename in sorted(retrieved_lines):
        for line in sorted(retrieved_lines[filename]):
            text = retrieved_lines[filename][line]
            for qualified in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*)\s*\(", text):
                symbol = qualified.split("::")[-1]
                if symbol in seen:
                    continue
                seen.add(symbol)
                resolved = idx.resolve_symbol(symbol)
                if not (resolved.get("definitions") or resolved.get("callers") or resolved.get("callees")):
                    continue
                symbol_terms = _semantic_identifier_terms(symbol)
                overlap = len(symbol_terms & target_terms)
                score = overlap * 20 + _navigation_symbol_score(symbol, resolved)
                # A symbol whose implementation or use is itself already read is
                # more useful for an exact-provenance bundle than an index-only hit.
                retrieved_structural_hits = 0
                for row in resolved.get("definitions", []) + resolved.get("callers", []):
                    f = row.get("file")
                    n = row.get("line")
                    if f and isinstance(n, int) and n in retrieved_lines.get(f, {}):
                        retrieved_structural_hits += 1
                score += min(retrieved_structural_hits, 4) * 3
                ranked.append((-score, -overlap, position, symbol, resolved))
                position += 1
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    return [(symbol, resolved) for _, _, _, symbol, resolved in ranked[:max_symbols]]


def _retrieved_window(retrieved_lines, filename, line, radius=6):
    """Return a narrow exact-provenance window around an already retrieved line."""
    file_lines = retrieved_lines.get(filename, {})
    if line not in file_lines:
        return None
    start = line
    end = line
    for n in range(line - 1, max(0, line - radius) - 1, -1):
        if n not in file_lines:
            break
        start = n
    for n in range(line + 1, line + radius + 1):
        if n not in file_lines:
            break
        end = n
    excerpt = retrieved_evidence_excerpt(retrieved_lines, filename, start, end)
    if excerpt is None:
        return None
    return {"file": filename, "start": start, "end": end, "excerpt": excerpt}

def generic_relationship_bundle_candidates(
    topic,
    category,
    retrieved_lines,
    provenance_valid_by_category,
    max_items=8,
):
    """Assemble a small multi-range relationship bundle from generic index facts.

    The assembler does not decide semantics. It proposes exact-provenance ranges
    that expose source-visible call/data/control relationships. The independent
    semantic bundle verifier remains authoritative. Existing extractor ranges are
    retained first, then topic-derived call sites and same-function neighbors are
    added. No scheduler-specific symbol or line knowledge lives here.
    """
    idx = get_repository_index()
    out = []
    seen = set()

    def add(item, origin):
        if not item:
            return
        key = (item["file"], item["start"], item["end"])
        if key in seen or len(out) >= max_items:
            return
        seen.add(key)
        row = dict(item)
        row["origin"] = origin
        out.append(row)

    # Preserve category-local evidence first, then allow already-provenanced
    # ranges from sibling categories to participate in a relationship proof.
    for item in provenance_valid_by_category.get(category, []):
        add(item, "category_evidence")
    for other_category, items in provenance_valid_by_category.items():
        if other_category == category:
            continue
        for item in items[:2]:
            add(item, f"sibling_category:{other_category}")

    # Follow concrete symbols that appeared in category-local retrieved source.
    # This closes relationships the investigator already discovered (for example
    # caller -> callee implementation) without relying on benchmark vocabulary.
    # Only definitions/call sites whose exact lines were already retrieved can
    # enter a bundle; semantic acceptance is still delegated to the verifier.
    evidence_symbols = []
    evidence_symbol_seen = set()
    for item in provenance_valid_by_category.get(category, []):
        text = item.get("excerpt", "")
        for qualified in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*)\s*\(", text):
            symbol = qualified.split("::")[-1]
            if symbol in evidence_symbol_seen:
                continue
            evidence_symbol_seen.add(symbol)
            resolved = idx.resolve_symbol(symbol)
            if resolved.get("definitions") or resolved.get("callers"):
                evidence_symbols.append((symbol, resolved))

    for symbol, resolved in evidence_symbols[:12]:
        for definition in resolved.get("definitions", [])[:4]:
            filename = definition.get("file")
            line = definition.get("line")
            if filename and isinstance(line, int) and line in retrieved_lines.get(filename, {}):
                add(_retrieved_window(retrieved_lines, filename, line, radius=12),
                    f"evidence_definition:{symbol}")
        for site in resolved.get("callers", [])[:6]:
            filename = site.get("file")
            line = site.get("line")
            if filename and isinstance(line, int) and line in retrieved_lines.get(filename, {}):
                add(_retrieved_window(retrieved_lines, filename, line),
                    f"evidence_call:{symbol}")

    # Next, prefer concrete resolvable call symbols from source that was already
    # retrieved for this investigation. This is category-directed completion,
    # not broader search: no new file or line enters the bundle unless its exact
    # source was already read. It lets a missing category move outward from a
    # nearby producer/caller toward a more relevant consumer/load/store path.
    for symbol, resolved in _retrieved_category_symbols(topic, category, retrieved_lines):
        for definition in resolved.get("definitions", [])[:4]:
            filename = definition.get("file")
            line = definition.get("line")
            if filename and isinstance(line, int) and line in retrieved_lines.get(filename, {}):
                add(
                    _retrieved_window(retrieved_lines, filename, line, radius=12),
                    f"category_definition:{symbol}",
                )
        for site in resolved.get("callers", [])[:6]:
            filename = site.get("file")
            line = site.get("line")
            if filename and isinstance(line, int) and line in retrieved_lines.get(filename, {}):
                add(
                    _retrieved_window(retrieved_lines, filename, line),
                    f"category_call:{symbol}",
                )
        if len(out) >= max_items:
            break

    # Add exact retrieved windows around topic-derived call sites. These are
    # structural candidates only; they cannot become evidence without semantic
    # bundle acceptance.
    call_sites = []
    for symbol, resolved in _topic_resolved_symbols(topic):
        for site in resolved.get("callers", [])[:8]:
            filename = site.get("file")
            line = site.get("line")
            if filename and isinstance(line, int) and line in retrieved_lines.get(filename, {}):
                call_sites.append((symbol, filename, line, site.get("caller") or ""))

    # Direct topic-symbol call sites before neighborhood expansion.
    for symbol, filename, line, _ in call_sites:
        add(_retrieved_window(retrieved_lines, filename, line), f"topic_call:{symbol}")

    # If room remains, expose nearby calls in the same enclosing function.
    # This is the generic bridge for producer -> stored value/callback -> consumer
    # and candidate-load -> selection/claim -> gate relationships.
    for symbol, filename, line, caller in call_sites:
        if len(out) >= max_items:
            break
        boundary = idx.function_for_line(filename, line)
        function_name = boundary.name if boundary is not None else caller
        if not function_name:
            continue
        neighbors = []
        for edge in idx.callees_of(function_name):
            if edge.file != filename or edge.line == line:
                continue
            distance = edge.line - line
            if abs(distance) > 180:
                continue
            if edge.line not in retrieved_lines.get(filename, {}):
                continue
            neighbors.append((abs(distance), edge.line, edge.callee))
        neighbors.sort(key=lambda x: (x[0], x[1], x[2]))
        for _, neighbor_line, callee in neighbors[:6]:
            add(
                _retrieved_window(retrieved_lines, filename, neighbor_line),
                f"same_function:{symbol}->{callee}",
            )
            if len(out) >= max_items:
                break

    return out
