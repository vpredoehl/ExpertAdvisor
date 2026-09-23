---
title: "TG4 Layout 8 Width 80 Semantic Worker Controlled Rollover"
document_type: "architecture review"
status: "final"
generated_from: "TG4_Layout8_Width80_SemanticWorker_Controlled_Rollover_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# TG4 Layout 8 Width 80 Semantic Worker Controlled Rollover

Implemented and validated the atomic layout-8 rollover mechanism. Production registry was not touched; no scheduler or experiments were changed.

Files changed/added:

- [RollSemanticWorkerLayout.py](</Volumes/Developer SSD/ExpertAdvisor/Scripts/RollSemanticWorkerLayout.py>)
- [SemanticWorkerRolloverTests.py](</Volumes/Developer SSD/ExpertAdvisor/Tests/SemanticWorkerRolloverTests.py>)
- [SemanticWorkerRolloverTests.sh](</Volumes/Developer SSD/ExpertAdvisor/Tests/SemanticWorkerRolloverTests.sh>)
- [SemanticWorkerRegistryTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/SemanticWorkerRegistryTests.cpp>)
- [semantic-layout-inference-worker-routing.md](</Volumes/Developer SSD/ExpertAdvisor/docs/semantic-workers/semantic-layout-inference-worker-routing.md>)
- [layout-8-width-80-controlled-rollover.md](</Volumes/Developer SSD/ExpertAdvisor/docs/semantic-workers/layout-8-width-80-controlled-rollover.md>)

Implementation: a separate `RollSemanticWorkerLayout.py` command requires both `LSTM_Release` and `lstm-infer-worker`. It derives layout/width from clean source, verifies provenance, stages both immutable artifacts/runtime links, validates the complete prospective registry, then performs one atomic registry replacement. Existing infer-only publication remains unchanged and still rejects a layout advance. There is no train-only rollover operation.

Validated disposable post-rollover routing:

| Layout / role | Result |
| --- | --- |
| 6 / infer | historical layout-6 artifact |
| 7 / train | historical layout-7 training/reference artifact |
| 7 / infer | historical layout-7 infer artifact |
| 8 / train | current layout-8 `LSTM_Release` |
| 8 / infer | current layout-8 `lstm-infer-worker` |

Source contract: layout 8 / width 80.

Clean isolated Release validation at `66b2613fb7df0bf8064df02e95a62cbf0afdad91`:

- `LSTM_Release`: `5718d86eff7a0ca4b0ad9b3bac5fc5cfecdbd119b8f387870f55949308bb2b07`
- `lstm-infer-worker`: `a5f5f4dc6c753879ab339c3c1c4b0c5d2b130dde2b181754f06e7b912acc76d6`
- Infer identity reported the exact commit and SHA.
- Disposable real rollover succeeded and retained the existing runtime identity `6c8d208a…defee34e`.

Tests passed:

- `Tests/SemanticWorkerRolloverTests.sh` — 11 tests, including registry replacement failure, corrupt registry/artifact/runtime, staged conflict, missing/wrong-role candidate, provenance failure, and C++ exact routing.
- `Tests/SemanticWorkerPublisherTests.sh`
- `Tests/SemanticWorkerRegistryTests.sh`
- `Tests/SchedulerSemanticAdmissionTests.sh`
- All six requested TG4/layout-8 compatibility suites.
- Isolated clean `LSTM Release` and `LSTM Infer Worker` Release builds.
- `git diff --check`

Operational audit:

- Starting/final HEAD: `66b2613fb7df0bf8064df02e95a62cbf0afdad91`
- Initial status: clean.
- Final status: intended uncommitted implementation/docs/tests only.
- Operational registry remains byte-identical: SHA `78ac5c4e906c43555f6d36c21ff3fe7b4f76e7952550c051efc4654340745c69`, current layout 7.
- Existing live scheduler remained running and untouched.
- No production DB or experiment state was mutated.

Production publication is intentionally blocked pending review, commit, and an approved clean build from that future committed HEAD. After that, the operator command is:

```bash
xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Release" \
  -configuration Release -derivedDataPath <clean-derived-data> build

xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Infer Worker" \
  -configuration Release -derivedDataPath <clean-derived-data> build

/usr/bin/python3 Scripts/RollSemanticWorkerLayout.py \
  --repository-root /Volumes/Developer\ SSD/ExpertAdvisor \
  --training-executable <clean-derived-data>/Build/Products/Release/LSTM_Release \
  --inference-executable <clean-derived-data>/Build/Products/Release/lstm-infer-worker \
  --source-commit <approved-clean-HEAD>
```

Expected pre-state is current layout 7 with its two current bindings; expected post-state is current layout 8 with exactly two current bindings, while layouts 6 and 7 remain historical. Any failure before registry replacement leaves layout 7 authoritative; staged artifacts may remain unreachable.

`git diff --stat` reports the tracked changes; newly added files are visible in `git status --short`.

Recommended commit message:

```text
Add atomic semantic worker layout rollover publication
```