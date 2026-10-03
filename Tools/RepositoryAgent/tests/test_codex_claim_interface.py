#!/usr/bin/env python3
import sys
import tempfile
import types
from pathlib import Path

stub = types.ModuleType("expertadvisor_agent")
stub.list_files = lambda prefix="": []
stub.search = lambda pattern, max_results=100: ""
stub.read_file = lambda name, start=1, end=200: "stub source"
sys.modules.setdefault("expertadvisor_agent", stub)

from ..codex_interface import CodexRepositoryInterface


class FakeRuntime:
    def __init__(self): self.calls=[]

    def verify_claim(self, topic, claim, item):
        self.calls.append(("single", topic, claim, item))
        if claim == "Unsupported claim":
            return {"supports": False, "establishes": "", "reason": "source does not establish it", "model_turns": 1}
        return {"supports": True, "establishes": "source establishes claim", "reason": "test", "model_turns": 1}

    def verify_bundle_claim(self, topic, claim, items):
        self.calls.append(("bundle", topic, claim, items))
        if claim == "Cross-file relationship":
            assert [item["file"] for item in items] == ["admission.cpp", "worker.cpp"]
            return {"supports": True, "establishes": "admission forwards the selected work to worker", "reason": "visible handoff", "model_turns": 1}
        if claim == "Contradictory bundle":
            assert any("SUPPORTIVE" in item["excerpt"] for item in items)
            assert any("CONTRADICTORY" in item["excerpt"] for item in items)
            return {"supports": False, "establishes": "", "reason": "supplied evidence contradicts the claim", "model_turns": 1}
        if claim in {"Unsupported multi-range", "Insufficient bundle"}:
            return {"supports": False, "establishes": "", "reason": "bundle does not establish the relationship", "model_turns": 1}
        if claim == "Contradictory protocol state":
            return {"supports": True, "establishes": "", "reason": "invalid", "model_turns": 1}
        return {"supports": True, "establishes": "bundle establishes claim", "reason": "test", "model_turns": 1}


class FakeInterface(CodexRepositoryInterface):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.revision = 1

    def _claim_item(self, spec):
        file = self._required_text(spec, "file")
        start = int(spec["start"])
        end = int(spec["end"])
        if start < 1 or end < start or end-start+1 > 500:
            raise ValueError("claim source range must be 1..500 lines")
        text = f"{file}:{start}-{end}: exact source revision={self.revision}"
        if file == "support.cpp": text += " SUPPORTIVE"
        if file == "contradiction.cpp": text += " CONTRADICTORY"
        return {"file": file, "start": start, "end": end, "excerpt": text}


def expect_value_error(action):
    try: action()
    except ValueError: return
    raise AssertionError("invalid bundle was accepted")


def main():
    with tempfile.TemporaryDirectory() as td:
        runtime = FakeRuntime(); ledger = Path(td) / "claims.json"
        iface = FakeInterface(claim_ledger_path=str(ledger), claim_runtime=runtime)
        caps = iface.dispatch({"op": "capabilities"})
        assert caps["repository_read_only"] is True
        assert "claim_evidence_ledger" in caps["controlled_state_writes"]
        assert "investigate_source_bundle_claim" in caps["operations"]

        req = {"op": "verify_source_claim", "topic_id": "t1", "topic": "Topic", "claim": "A claim", "file": "LSTM/x.cpp", "start": 10, "end": 12}
        first = iface.dispatch(req); assert first["verdict"]["supports"] is True
        second = iface.dispatch(req); assert second["verdict"]["ledger_hit"] is True
        assert len(runtime.calls) == 1

        targeted = {**req, "op": "investigate_source_claim", "claim": "A separately bounded claim"}
        first_targeted = iface.dispatch(targeted)
        assert first_targeted["answer"] == "source establishes claim"
        assert first_targeted["manifest"]["metrics"] == {
            "requested_range_count": 1, "effective_range_count": 1,
            "repository_read_count": 1, "unrelated_topic_count": 0,
            "model_turn_count": 1, "ledger_hit": False,
        }
        second_targeted = iface.dispatch(targeted)
        assert second_targeted["manifest"]["verification"] == {"supports": True, "ledger_hit": True, "model_turns": 0}
        assert len(runtime.calls) == 2
        unsupported = iface.dispatch({**targeted, "claim": "Unsupported claim"})
        assert unsupported["answer"] == "Not established by the supplied source: source does not establish it"
        assert unsupported["manifest"]["verification"]["supports"] is False
        assert len(runtime.calls) == 3

        bundle_ranges = [{"file": "b.cpp", "start": 3, "end": 4}, {"file": "a.cpp", "start": 1, "end": 2}]
        breq = {"op": "verify_source_bundle_claim", "topic_id": "t1", "topic": "Topic", "claim": "A path", "ranges": bundle_ranges}
        b1 = iface.dispatch(breq)
        assert [item["file"] for item in b1["evidence"]] == ["a.cpp", "b.cpp"]
        b2 = iface.dispatch({**breq, "ranges": list(reversed(bundle_ranges))})
        assert b2["verdict"]["bundle_ledger_hit"] is True
        assert len(runtime.calls) == 4

        multi = {"op": "investigate_source_bundle_claim", "topic_id": "t1", "topic": "Admission routing", "claim": "Cross-file relationship", "ranges": [{"file": "worker.cpp", "start": 30, "end": 32}, {"file": "admission.cpp", "start": 10, "end": 12}]}
        multi_first = iface.dispatch(multi)
        assert multi_first["answer"] == "admission forwards the selected work to worker"
        assert multi_first["manifest"]["mode"] == "targeted_source_bundle_claim"
        assert multi_first["manifest"]["metrics"] == {
            "requested_range_count": 2, "effective_range_count": 2,
            "repository_read_count": 2, "unrelated_topic_count": 0,
            "model_turn_count": 1, "ledger_hit": False,
        }
        assert all(item["source_sha256"] for item in multi_first["manifest"]["evidence"])
        multi_repeat = iface.dispatch({**multi, "ranges": list(reversed(multi["ranges"]))})
        assert multi_repeat["manifest"]["verification"] == {"supports": True, "ledger_hit": True, "model_turns": 0}
        assert len(runtime.calls) == 5

        for claim in ("Unsupported multi-range", "Insufficient bundle"):
            result = iface.dispatch({**multi, "claim": claim})
            assert result["verdict"]["supports"] is False
        contradictory = iface.dispatch({**multi, "claim": "Contradictory bundle", "ranges": [{"file": "support.cpp", "start": 1, "end": 1}, {"file": "contradiction.cpp", "start": 2, "end": 2}]})
        assert contradictory["answer"] == "Not established by the supplied source: supplied evidence contradicts the claim"
        malformed = iface.dispatch({**multi, "claim": "Contradictory protocol state"})
        assert malformed["verdict"]["verifier_error"] is True
        assert malformed["verdict"]["supports"] is False

        maximum = {**multi, "claim": "Maximum bundle", "ranges": [{"file": f"f{i}.cpp", "start": 1, "end": 1} for i in range(8)]}
        assert iface.dispatch(maximum)["manifest"]["metrics"]["effective_range_count"] == 8
        expect_value_error(lambda: iface.dispatch({**maximum, "ranges": maximum["ranges"] + [{"file": "f8.cpp", "start": 1, "end": 1}]}))
        expect_value_error(lambda: iface.dispatch({**multi, "ranges": [{"file": "x.cpp", "start": 1, "end": 1}, {"file": "x.cpp", "start": 1, "end": 1}]}))
        expect_value_error(lambda: iface.dispatch({**multi, "ranges": [{"file": "x.cpp", "start": 1, "end": 3}, {"file": "x.cpp", "start": 3, "end": 5}]}))

        mutable = {**multi, "claim": "Mutable bundle", "ranges": [{"file": "a.cpp", "start": 1, "end": 1}, {"file": "b.cpp", "start": 2, "end": 2}]}
        iface.dispatch(mutable); before_change = len(runtime.calls)
        assert iface.dispatch(mutable)["manifest"]["verification"]["ledger_hit"] is True
        iface.revision = 2
        changed = iface.dispatch(mutable)
        assert changed["manifest"]["verification"]["ledger_hit"] is False
        assert len(runtime.calls) == before_change + 1

        rows = iface.dispatch({"op": "verified_claims", "topic_id": "t1"})
        assert rows["matched"] >= 8
    print("test_codex_claim_interface: PASS")


if __name__ == "__main__": main()
