---
title: "LSTM Phase 22E Standalone Inference Worker and Role-Aware Artifacts"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_Phase22E_StandaloneInferenceWorker_RoleAwareArtifacts_Prompt.txt"
reasoning_effort: "medium"
model: "gpt-5.6-terra"
---

# LSTM Phase 22E Standalone Inference Worker and Role-Aware Artifacts

Phase 22E is **NO-GO**: I completed and tested the role-aware registry/publisher foundation, but did not add the required real `lstm-infer-worker` target or scheduler INFER cutover.

Implemented:
- Registry schema v3, distinct from manifest schema v2.
- Legacy registry v2 / manifest v1 loading.
- Explicit legacy layout-6 immutable `LSTM_Release` preservation.
- Role-aware publisher path: `layout<layout>/infer/<commit>/<sha256>/lstm-infer-worker`.
- Atomic registry replacement/rollback coverage.

Validated:
- `Tests/SemanticWorkerRegistryTests.sh` — passed
- `Tests/SemanticWorkerPublisherTests.sh` — passed
- `python3 -m py_compile Scripts/PublishSemanticWorker.py` — passed
- `git diff --check` — passed

No production artifacts, registry, scheduler, database, experiments, or workers were modified.

The full status, limitations, next required increment, and disposition are in [Phase 22E report](/Volumes/Developer%20SSD/ExpertAdvisor/docs/Phase22/LSTM_Phase22E_StandaloneInferenceWorker_RoleAwareArtifacts_Output.md).

`git status --short` contains only the five intended Phase 22E files; `git diff --stat` is 216 insertions / 69 deletions.

## Continuation review (2026-09-19)

This continuation started on `lstm-feature-development` at
`2760c8237525ea73e6a38f257437252399697524`, over the expected dirty Phase
22E foundation.  The actual baseline has four modified implementation/test
files (shown in the status below); the report itself was already tracked and
unchanged when the continuation began.

### Preserved foundation

- The semantic-worker registry remains schema **v3**.  Legacy registry v2
  loading remains supported.
- Legacy artifact manifests remain schema **v1**; role-aware artifact
  manifests remain schema **v2**.  These versions are intentionally distinct
  from registry v3.
- Layout 6 remains an explicit compatibility binding to immutable
  `LSTM_Release` at commit
  `7645265bca0c2529523e1d2cdb37e7d023dfd559` and SHA-256
  `945225dd2a42f87a2a8dfbfe47b006708e3d90c88a858d25787e5a2237c62dd7`.
  It was not rebuilt, renamed, moved, republished, or modified.
- The role-aware immutable inference artifact convention remains
  `layout<layout>/infer/<commit>/<sha256>/lstm-infer-worker`.
- Final and checkpoint dispatch already select an inference artifact through
  `SelectInferenceWorker`, reserve the exact canonical executable, inject the
  attempt id only after canonical identity validation, and persist/reconcile
  that exact identity.  ANALYZE remains `lstm-analyze-worker` and outside the
  semantic registry; TRAIN routing remains unchanged.

### Stopping boundary

No `lstm-infer-worker` target/product was added.  The existing inference
composition is embedded in `LSTM/main.cpp`; its `LSTM Release` target directly
compiles the monolithic application source list.  A new executable cannot link
the existing implementation without either copying that whole target or first
extracting a substantial shared application/library composition.  Neither is
the requested narrow inference boundary, and both exceed this continuation's
explicit prohibition on wholesale target copying and broad refactoring.

Creating a wrapper, symlink, or copied `LSTM_Release` artifact would also be
incorrect: it would not have a thin, restricted infer-worker CLI and would
either weaken executable-identity enforcement or fail worker-attempt identity
validation.  No database migration is indicated by the existing lifecycle
flow; the blocker is solely executable composition/target topology.

### Validation and safety

Passed during this continuation:

- `Tests/SemanticWorkerRegistryTests.sh`
- `Tests/SemanticWorkerPublisherTests.sh`
- `python3 -m py_compile Scripts/PublishSemanticWorker.py`
- `git diff --check`

Focused Release validation was intentionally not attempted.  The required
preflight found an active production scheduler and two active training workers;
no scheduler command, worker executable, build, publication, cutover,
database operation, or experiment operation was launched.  Therefore no live
artifact/registry publication occurred and no production scheduler, database,
or experiment state was modified.

Final continuation status at the stopping boundary:

```
 M Scripts/PublishSemanticWorker.py
 M Sources/SchedulerCore/SemanticWorkerRegistry.cpp
 M Sources/SchedulerCore/SemanticWorkerRegistry.hpp
 M Tests/SemanticWorkerPublisherTests.py
```

`git diff --stat` remains 4 files changed, 135 insertions, 55 deletions.

The post-commit full Release validation remains required, but only after the
missing composition boundary is implemented and human review/commit is
complete.  The recommended future sequence is: extract the shared inference
composition into a properly linked library or narrow shared target; add and
validate the restricted worker target; review and commit; run the
provenance-valid full Release build; then obtain explicit publication/cutover
authorization.

**Phase 22E standalone inference worker implementation: NO-GO**
