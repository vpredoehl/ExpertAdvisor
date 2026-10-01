#!/usr/bin/env python3
import tempfile
from pathlib import Path

from ..codex_interface import CodexRepositoryInterface


class FakeRuntime:
    def __init__(self): self.calls=[]
    def verify_claim(self, topic, claim, item):
        self.calls.append(("single",topic,claim,item))
        return {"supports":True,"establishes":"source establishes claim","reason":"test"}
    def verify_bundle_claim(self, topic, claim, items):
        self.calls.append(("bundle",topic,claim,items))
        return {"supports":True,"establishes":"bundle establishes claim","reason":"test"}


class FakeInterface(CodexRepositoryInterface):
    def _claim_item(self, spec):
        file=self._required_text(spec,"file"); start=int(spec["start"]); end=int(spec["end"])
        if start < 1 or end < start or end-start+1 > 500: raise ValueError("claim source range must be 1..500 lines")
        return {"file":file,"start":start,"end":end,"excerpt":f"{file}:{start}-{end}: exact source"}


def main():
    with tempfile.TemporaryDirectory() as td:
        runtime=FakeRuntime(); ledger=Path(td)/"claims.json"
        iface=FakeInterface(claim_ledger_path=str(ledger),claim_runtime=runtime)
        caps=iface.dispatch({"op":"capabilities"})
        assert caps["repository_read_only"] is True
        assert "claim_evidence_ledger" in caps["controlled_state_writes"]
        req={"op":"verify_source_claim","topic_id":"t1","topic":"Topic","claim":"A claim","file":"LSTM/x.cpp","start":10,"end":12}
        first=iface.dispatch(req); assert first["verdict"]["supports"] is True
        second=iface.dispatch(req); assert second["verdict"]["ledger_hit"] is True
        assert len(runtime.calls)==1
        breq={"op":"verify_source_bundle_claim","topic_id":"t1","topic":"Topic","claim":"A path", "ranges":[{"file":"a.cpp","start":1,"end":2},{"file":"b.cpp","start":3,"end":4}]}
        b1=iface.dispatch(breq); assert b1["verdict"]["supports"] is True
        b2=iface.dispatch(breq); assert b2["verdict"]["bundle_ledger_hit"] is True
        assert len(runtime.calls)==2
        rows=iface.dispatch({"op":"verified_claims","topic_id":"t1"})
        assert rows["matched"]==2
        try: iface.dispatch({**breq,"ranges":[{"file":"a","start":1,"end":1}]})
        except ValueError: pass
        else: raise AssertionError("one-range bundle accepted")
    print("test_codex_claim_interface: PASS")

if __name__ == "__main__": main()
