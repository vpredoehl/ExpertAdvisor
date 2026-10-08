# Phase 24D — Semantic-Worker Publication Contract Hardening

## Outcome and scope

The three confirmed Phase 24C gaps are corrected in the Rollover worktree: new dedicated INFER candidates must report their compiled semantic layout and input width; same-layout refresh qualifies both dedicated roles; and explicit current INFER CLI arguments must agree with the checkout source contract.

**GO for implementation review. NO-GO for deployment/publication pending a separately authorized build and qualification of fresh candidates.** Python and lightweight structural validation passed. No Xcode build or actual worker identity invocation was performed, so the modified native INFER identity emission has not been verified in a compiled executable.

Work was confined to `/Volumes/Developer SSD/ExpertAdvisor-Rollover`. The isolated `expertadvisor-repository-rollover` MCP connection supplied read-only capabilities/source evidence, including the pre-change INFER entrypoint. Publication scripts were inspected directly because they are outside that connection's source allowlist. No production RepositoryAgent connection or semantic/model-backed MCP operation was used. No RepositoryAgent implementation was modified.

No production checkout files, authoritative production registries, worker binaries, database data, schedulers, or running experiments were touched. Tests operated on disposable fixtures. No Qwen weights or Ollama were loaded or started.

## Phase 24A/C documentation closure

Both requested reports were read and checked against the relevant source and existing tests. They remain complete historical audit snapshots: Phase 24A records the architectural gaps at its audit baseline; Phase 24C records the publication paths, surrounding compatibility workflow, confirmed gaps, coverage, and implementation recommendation before this repair. Phase 24D supersedes the publication-gap findings without rewriting those historical conclusions.

Committed **only**:

- `docs/phases/Phase24/LSTM_Phase24A_FinalRefactoringGapAnalysis_Output.md`
- `docs/phases/Phase24/LSTM_Phase24C_SemanticWorkerPublicationContractAudit_Output.md`

Commit: `89be36dcfcb9bcda71d76dae18cef6fb6e6647ea` — `Phase 24A/C: Document final refactoring and publication audits`.

That commit contains two files and 470 insertions. Phase 24B remains committed as `7222021f`. No Phase 24D implementation or report was staged or committed.

## Files modified for Phase 24D

| File | Change |
| --- | --- |
| `LSTM/InferWorkerMain.cpp` | Adds authoritative compiled layout/width constants to the existing version-1 INFER identity record. |
| `Scripts/PublishSemanticWorker.py` | Shared strict identity reader and semantic qualification; INFER qualification; unconditional current source-contract comparison; current TRAIN width compatibility gate; prospective registry validation; delays directory creation until qualification. |
| `Scripts/RollSemanticWorkerLayout.py` | Dedicated TRAIN and INFER candidates use the shared full qualification; removes the second TRAIN identity invocation. |
| `Scripts/RefreshSemanticWorkerGeneration.py` | Both dedicated candidates must match the intended same-layout contract. |
| `Scripts/tests/test_dedicated_train_rollover.py` | Updates identity fixtures and mocks for the shared qualification contract. |
| `Tests/SemanticWorkerPublisherTests.py` | Updates existing identity regression fixtures/calls with expected layout and width. |
| `Tests/SemanticWorkerPublicationContractTests.py` | New 14-test offline suite exercising real publication orchestration with mocked candidate identity subprocesses and disposable immutable artifacts. |
| `docs/semantic-workers/semantic-layout-inference-worker-routing.md` | Documents new candidate requirements, historical compatibility, source/registry checks, independent commits, refresh, and failure boundaries. |
| This report | Records implementation, evidence, validation, and limitations. |

## Contract changes and source evidence

### Dedicated identity qualification

`LSTM/InferWorkerMain.cpp`, the `--build-identity` branch (lines 60–79), emits `semantic_layout` using `EA::kModelInputSemanticLayoutVersion` and `model_input_width` using `EA::kCurrentModelInputWidth`, with the header, role, source commit, canonical executable, and executable SHA-256 retained. No identity version or semantic version was advanced. The checkout contract remains layout 13 / width 171.

`Scripts/PublishSemanticWorker.py:211–264`, `read_worker_build_identity`, `validate_worker_semantic_contract`, `verify_worker_build_identity`, and `verify_inference_build_identity`, consume one successful version-1 identity record per candidate. Qualification rejects multiple records, incorrect headers, malformed/empty fields, duplicate keys, and unsupported/missing versions. Well-formed additional fields remain allowed. Role, source commit, self-reported hash, layout, and width must all agree with the intended publication contract and independently computed artifact hash.

`publish` (`Scripts/PublishSemanticWorker.py:726–789`) now passes the intended layout/width to INFER qualification. `rollover` (`Scripts/RollSemanticWorkerLayout.py:183–224`) applies the same check to dedicated TRAIN and to INFER in both dedicated and legacy TRAIN modes. The legacy TRAIN embedded-commit contract is retained. `verify_train_semantic_contract` remains a thin shared semantic-reader adapter rather than a separate parser.

`refresh` (`Scripts/RefreshSemanticWorkerGeneration.py:69–105`) qualifies both dedicated candidates against the requested layout/width before artifact staging. The prior/current layout gate, runtime-resource agreement, retained historical candidates, selection priorities, independently supplied role commits, immutable storage, and atomic paired registry replacement are preserved.

### INFER source and registry consistency

`main` (`Scripts/PublishSemanticWorker.py:905–940`) always derives the current source contract, including when both layout and width were explicitly supplied. Each omitted value defaults independently using `is None`; supplied zero or contradictory values cannot bypass comparison. The existing clean-HEAD/current source-commit check remains mandatory. Historical import still requires explicit historical layout, width, and source commit; it deliberately does not compare those values with today's source checkout.

Under the publication lock, `publish` requires the current layout binding and exactly one current TRAIN binding whose width matches the proposed current INFER width (`Scripts/PublishSemanticWorker.py:795–809`). This prevents an identity-qualified candidate from being labeled with a contract inconsistent with the authoritative current training/reference binding. Before replacing the registry, the complete prospective registry is validated (`Scripts/PublishSemanticWorker.py:883–884`), using the existing authoritative validator.

Artifact-root resolution still precedes construction of immutable paths. Directory creation now follows candidate/runtime qualification. Review found and corrected an initial relative-root normalization regression; the new relative/symlink-root test checks canonical absolute return paths and idempotent reuse.

## Compatibility analysis

- Registered historical workers are loaded and dispatched using their existing manifest/registry contracts. Registry loading does not execute them or demand the new compiled identity fields. No registry schema, loader, dispatcher, historical publisher, or selection algorithm was changed.
- A previously published dedicated INFER artifact lacking semantic fields remains usable through its existing registry entry. It cannot be relabeled as a newly qualified modern candidate. New dedicated publications require the new proof regardless of the proposed current/historical rule.
- Legacy `LSTM_Release` TRAIN rollover remains supported with its existing provenance check; the accompanying newly published dedicated INFER must satisfy the stronger contract.
- Historical import remains governed by explicit historical metadata rather than the current checkout. Existing historical TRAIN and INFER publisher tests pass.
- Dedicated TRAIN and INFER may retain independent source commits where already supported. Qualification compares each candidate with its own intended commit, not necessarily with the other candidate.
- The Python verifier's internal call signature now requires expected layout/width. All repository callers and relevant identity fixtures were updated. Existing `check_embedded_commit=False` fixture seams remain internal callable options; the CLI provides no qualification-bypass flag.

Evidence: unchanged registry-loading/validation and historical publication workflows in `Scripts/PublishSemanticWorker.py`, existing rollover/refresh generation assembly, and the historical/registry test suites listed below. The new suite also loads an existing registry containing legacy TRAIN and old dedicated INFER bytes without invoking any identity subprocess.

## Failure atomicity and rollback analysis

Candidate identity, role, commit, hash, and semantic rejection happens before runtime or worker staging in all three modern publication routes. New tests snapshot existing registry bytes, manifests, runtime files, immutable executable bytes/modes, and symlink targets before rejection, and require identical snapshots afterward. They also assert no staging/registry-write/current-link calls and no residual `.stage` entries. Invalid INFER qualification cannot create a previously absent artifact root.

The additional current TRAIN width gate runs under the publisher lock before staging. Creating/opening `.publish.lock` is bookkeeping and does not change authoritative publication state. Runtime and binary content hashing, immutable-directory conflict checks, staged rehashing, advisory locking, atomic registry replacement, and post-commit convenience-link behavior remain in the established pipeline (`publish`, `rollover`, `refresh`, `atomic_write_json`, and `update_current_link_after_registry_commit`).

There is no claim that every filesystem error can roll back a completed registry replacement. Existing generic failures after successful immutable staging may leave unregistered immutable artifacts; they do not authorize a partially updated registry. A convenience-link error after registry commit is explicitly reported as committed. A directory-fsync error after registry replacement also remains an existing ambiguous committed/durability boundary in `atomic_write_json` (`Scripts/PublishSemanticWorker.py:126–139`). This patch does not redesign those boundaries. The new qualification failures occur before them and preserve existing authority.

## Security and trust review

The identity record is executable-provided qualification evidence, not cryptographic attestation of honest semantic behavior. Existing embedded-commit inspection, independent SHA-256 calculation, self-reported identity comparison, runtime package validation, staged rehashing, immutable paths, registry validation, and lock/replace boundaries are preserved. Parsing one strict record avoids duplicated/conflicting fields and multiple identity responses. Qualification no longer combines two independently obtained TRAIN identity records.

No new external dependency, publication endpoint, schema, bypass CLI option, scheduler operation, database write, or production path was introduced. Existing execution trust for operator-supplied candidates remains: publication scripts normally invoke `--build-identity`; this task's regression suite mocks that invocation and never launches workers. Sandboxing/signing candidate execution is outside this increment.

## Tests and results

All commands ran from the Rollover root. **90 Python test methods passed**, including parameterized negative cases, plus **six lightweight shell structural/configuration checks**. No failing checks remain.

| Command | Result |
| --- | --- |
| `PYTHONDONTWRITEBYTECODE=1 python3 Tests/SemanticWorkerPublicationContractTests.py` | 14 passed; final run includes relative/symlink-root regression. |
| `PYTHONDONTWRITEBYTECODE=1 TMPDIR=/private/tmp python3 Tests/SemanticWorkerPublisherTests.py` | 12 passed; rerun after canonical-path correction. |
| `PYTHONDONTWRITEBYTECODE=1 TMPDIR=/private/tmp python3 -m unittest discover -s Scripts/tests -p 'test_dedicated_train_rollover.py' -v` | 12 passed. |
| `PYTHONDONTWRITEBYTECODE=1 TMPDIR=/private/tmp python3 Tests/SemanticWorkerGenerationRefreshTests.py` | 13 passed. |
| `env -u EA_SEMANTIC_REGISTRY_UNDER_TEST -u EA_TRAINING_SELECTION_REGISTRY_UNDER_TEST -u EA_SEMANTIC_REGISTRY_ROLLOVER_UNDER_TEST PYTHONDONTWRITEBYTECODE=1 TMPDIR=/private/tmp python3 Tests/SemanticWorkerRolloverTests.py` | 13 passed. Optional external registry inputs were removed; the suite uses its own temporary registry. |
| `PYTHONDONTWRITEBYTECODE=1 TMPDIR=/private/tmp python3 Tests/SemanticWorkerHistoricalTrainingCandidatePublisherTests.py` | 6 passed. |
| `PYTHONDONTWRITEBYTECODE=1 TMPDIR=/private/tmp python3 Tests/SemanticWorkerHistoricalInferenceWorkerPublisherTests.py` | 6 passed. |
| `PYTHONDONTWRITEBYTECODE=1 python3 Tests/DedicatedTrainingWorkerArchitectureGuardTests.py` | 14 passed. |
| `bash Tests/DedicatedTrainingWorkerArchitectureTests.sh` | Passed. |
| `bash Tests/ReleaseWorkerBuildConfigurationTests.sh` | Passed. |
| `bash Tests/SchedulerTrainingWorkerRoutingTests.sh` | Passed. |
| `bash Tests/LSTMPhase22Z3InferenceRuntimeCompositionBoundaryTests.sh` | Passed. |
| `bash Tests/LSTMPhase22Z4ManagedInferenceApplicationBoundaryTests.sh` | Passed. |
| `bash Tests/RuntimeFoundationStructuralTests.sh` | Passed. |
| Python `ast.parse` for modified/new Python sources | Passed; no bytecode generated. |
| `git diff --check` | Passed. |

The existing rollover suite compiles and executes a disposable C++ registry-parser test fixture with `clang++ -std=c++20 -Wall -Wextra -Werror`. This is not an Xcode project build or a worker launch. No native TRAIN/INFER executable was built or executed.

The new 14-test suite covers accepted matching layout/width; wrong/missing layout and width; malformed identity; wrong role, commit, and artifact hash; both refresh roles and dedicated rollover roles; fully/partly explicit CLI contradictions and defaults; TRAIN-reference width mismatch; preservation of existing authority; historical loading; independent commits; and canonical artifact paths. Its negative matrices exercise 20 semantic mismatch/missing-field cases, 15 role/commit/hash cases, and 40 malformed-record cases across the modern routes/roles.

## Remaining risks and recommendation

1. Native identity compilation/output remains unverified. Structural tests confirm the INFER source uses authoritative constants, but a separately authorized isolated build and fresh-candidate qualification are needed before publication.
2. Already-built INFER candidates without the new fields will intentionally fail new publication. Existing registered workers continue to load; no historical artifact should be edited to fabricate identity fields.
3. Executable identity is an operator trust contract; a malicious candidate can misrepresent its semantics. The patch closes deterministic metadata qualification gaps, not arbitrary executable trust.
4. Existing post-registry-replace fsync/convenience-link failure semantics remain. Qualification rejection is verified atomic; broader crash recovery was not redesigned.

**GO:** review the bounded implementation and its passing offline regression coverage. **NO-GO:** publish/deploy workers until fresh binaries are independently built and identity-qualified under separately authorized operational procedures. Stop here; do not commit Phase 24D or begin another implementation task.

## Worktree review state

The documentation closure commit is the current HEAD. The index is empty. Phase 24D changes remain unstaged/uncommitted.

`git status --short`:

```text
 M LSTM/InferWorkerMain.cpp
 M Scripts/PublishSemanticWorker.py
 M Scripts/RefreshSemanticWorkerGeneration.py
 M Scripts/RollSemanticWorkerLayout.py
 M Scripts/tests/test_dedicated_train_rollover.py
 M Tests/SemanticWorkerPublisherTests.py
 M docs/semantic-workers/semantic-layout-inference-worker-routing.md
?? Tests/SemanticWorkerPublicationContractTests.py
?? docs/phases/Phase24/LSTM_Phase24D_SemanticWorkerPublicationHardening_Output.md
```

`git diff --stat` (tracked changes only; the new regression suite and this report are untracked):

```text
 LSTM/InferWorkerMain.cpp                           |  3 +
 Scripts/PublishSemanticWorker.py                   | 77 +++++++++++++-----
 Scripts/RefreshSemanticWorkerGeneration.py         |  5 +-
 Scripts/RollSemanticWorkerLayout.py                | 25 ++----
 Scripts/tests/test_dedicated_train_rollover.py     |  7 +-
 Tests/SemanticWorkerPublisherTests.py              |  7 +-
 .../semantic-layout-inference-worker-routing.md    | 95 +++++++++++++++-------
 7 files changed, 144 insertions(+), 75 deletions(-)
```
