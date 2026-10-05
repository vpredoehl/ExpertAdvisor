#!/usr/bin/env python3
"""Deterministic, non-evidentiary coverage for bounded catalog navigation."""
from __future__ import annotations

import json
import sys
import types

READ_CALLS = []
stub = types.ModuleType("expertadvisor_agent")
stub.list_files = lambda prefix="": []
stub.search = lambda pattern, max_results=100: ""
def forbidden_read(name, start=1, end=200):
    READ_CALLS.append((name, start, end))
    return "SOURCE_SENTINEL_MUST_NOT_LEAK"
stub.read_file = forbidden_read
sys.modules.setdefault("expertadvisor_agent", stub)

from Tools.RepositoryAgent.codex_interface import CodexRepositoryInterface


class FakeIndex:
    def __init__(self, files):
        self.files = list(files)
        self.functions = []


class FakeInterface(CodexRepositoryInterface):
    def __init__(self, index):
        super().__init__()
        self.index = index

    def _idx(self):
        return self.index


def request(**extra):
    return {"op": "list_catalog_children", "scope": "Sources", **extra}


def rejected(action, text):
    try:
        action()
    except ValueError as exc:
        assert text in str(exc), str(exc)
        return
    raise AssertionError("request must fail closed")


def main():
    files = [
        "Sources/Root.cpp",
        "Sources/Alpha/One.cpp",
        "Sources/Alpha/Nested/Two.cpp",
        "Sources/Beta/Three.cpp",
        "Sources/Beta/Four.cpp",
    ]
    iface = FakeInterface(FakeIndex(list(reversed(files))))

    first = iface.dispatch(request(limit=2))
    assert first["mode"] == "bounded_catalog_children"
    assert first["resolved_scope"] == "Sources"
    assert first["children"] == [
        {"kind": "scope", "identity": "Sources/Alpha"},
        {"kind": "scope", "identity": "Sources/Beta"},
    ]
    assert first["next_cursor"]
    assert "SOURCE_SENTINEL_MUST_NOT_LEAK" not in json.dumps(first, sort_keys=True)
    assert "excerpt" not in json.dumps(first, sort_keys=True)
    assert not READ_CALLS and iface._claim_runtime is None

    second = iface.dispatch(request(limit=2, cursor=first["next_cursor"]))
    assert second["children"] == [{"kind": "source_file", "identity": "Sources/Root.cpp"}]
    assert second["next_cursor"] is None
    assert [*first["children"], *second["children"]] == [
        {"kind": "scope", "identity": "Sources/Alpha"},
        {"kind": "scope", "identity": "Sources/Beta"},
        {"kind": "source_file", "identity": "Sources/Root.cpp"},
    ]

    nested = iface.dispatch({"op": "list_catalog_children", "scope": "Sources/Alpha"})
    assert nested["children"] == [
        {"kind": "scope", "identity": "Sources/Alpha/Nested"},
        {"kind": "source_file", "identity": "Sources/Alpha/One.cpp"},
    ]
    assert "Sources/Alpha/Nested/Two.cpp" not in json.dumps(nested)

    # Stable ordering does not depend on catalog enumeration order.
    reordered = FakeInterface(FakeIndex(files)).dispatch(request(limit=32))
    assert reordered["children"] == [*first["children"], *second["children"]]

    rejected(lambda: iface.dispatch(request(scope="Missing")), "not found")
    rejected(lambda: iface.dispatch(request(cursor="bad.cursor")), "cursor")
    rejected(lambda: iface.dispatch({"op": "list_catalog_children", "scope": "Sources/Alpha", "cursor": first["next_cursor"]}), "stale or incompatible")
    rejected(lambda: iface.dispatch(request(limit=33)), "limit")

    changed = FakeInterface(FakeIndex(files + ["Sources/Gamma/Five.cpp"]))
    # Reuse the original interface secret to isolate catalog-identity staleness.
    changed._catalog_cursor_secret = iface._catalog_cursor_secret
    rejected(lambda: changed.dispatch(request(cursor=first["next_cursor"])), "stale or incompatible")

    print("test_codex_catalog_navigation: PASS")


if __name__ == "__main__":
    main()
