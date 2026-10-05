#!/usr/bin/env python3
"""Fail-closed direct member-call canonicalization coverage."""
from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path

stub = types.ModuleType("expertadvisor_agent")
stub.list_files = lambda prefix="": []
stub.search = lambda pattern, max_results=100: ""
stub.read_file = lambda name, start=1, end=200: "forbidden"
sys.modules.setdefault("expertadvisor_agent", stub)

from Tools.RepositoryAgent import repository_index as ri
from Tools.RepositoryAgent.codex_interface import CodexRepositoryInterface


SOURCE = """\
namespace EA::SchedulerCore
{
int SchedulerEngine::run() { return 0; }
int SchedulerCycleService::runOnce() { return 0; }
}
int A::Foo::run() { return 0; }
int B::Foo::run() { return 0; }

int TemporaryBridge()
{
    return EA::SchedulerCore::SchedulerEngine().run();
}

int LocalBridge()
{
    EA::SchedulerCore::SchedulerCycleService service{};
    return service.runOnce();
}

int UnknownReceiver()
{
    Unknown receiver{};
    return receiver.run();
}

int WrongQualifiedReceiver()
{
    Wrong::SchedulerEngine receiver{};
    return receiver.run();
}

int InferredReceiver()
{
    auto service = BuildService();
    return service.runOnce();
}

int AmbiguousReceiver()
{
    Foo receiver{};
    return receiver.run();
}
"""


class Runtime:
    def verify_claim(self, topic, claim, item):
        return {"supports": True, "establishes": "test", "reason": "test", "model_turns": 0}


class Interface(CodexRepositoryInterface):
    def __init__(self, index, ledger):
        super().__init__(claim_runtime=Runtime(), claim_ledger_path=str(ledger))
        self.index = index

    def _idx(self): return self.index
    def _claim_item(self, spec): return {**spec, "excerpt": "selected source"}


def main():
    old_list_files, old_read_file = ri.list_files, ri.read_file
    try:
        lines = SOURCE.splitlines()
        ri.list_files = lambda prefix="": ["Sources/SchedulerCore/Member.cpp"]
        ri.read_file = lambda name, start=1, end=200: "\n".join(
            f"{number}: {line}" for number, line in enumerate(lines[start - 1:end], start)
        )
        index = ri.RepositoryIndex().build(prefixes=("Sources/SchedulerCore",))
        with tempfile.TemporaryDirectory() as directory:
            interface = Interface(index, Path(directory) / "claims.json")
            for caller, callee in (("TemporaryBridge", "SchedulerEngine::run"),
                                   ("LocalBridge", "SchedulerCycleService::runOnce")):
                discovery = interface.dispatch({
                    "op": "discover_relationship_paths", "scope": "SchedulerCore",
                    "from": caller, "to": callee, "max_hops": 1,
                })
                assert discovery["paths"] == [[caller, callee]], discovery
                claim = interface.dispatch({
                    "op": "investigate_relationship_claim", "topic_id": "member",
                    "topic": "member calls", "claim": "direct member call",
                    "caller": caller, "callee": callee,
                })
                assert claim["selection_manifest"]["selection_status"] == "selected", claim

            # Neither unknown/static-inferred receivers nor an ambiguous type
            # method can acquire a direct edge.
            for caller, callee in (("UnknownReceiver", "SchedulerEngine::run"),
                                   ("WrongQualifiedReceiver", "SchedulerEngine::run"),
                                   ("InferredReceiver", "SchedulerCycleService::runOnce"),
                                   ("AmbiguousReceiver", "A::Foo::run"),
                                   ("AmbiguousReceiver", "B::Foo::run")):
                assert index.relationship(caller, callee) == [], (caller, callee)
                discovery = interface.dispatch({
                    "op": "discover_relationship_paths", "scope": "SchedulerCore",
                    "from": caller, "to": callee, "max_hops": 1,
                })
                assert discovery["paths"] == [], discovery
    finally:
        ri.list_files, ri.read_file = old_list_files, old_read_file
    print("test_member_call_canonicalization: PASS")


if __name__ == "__main__":
    main()
