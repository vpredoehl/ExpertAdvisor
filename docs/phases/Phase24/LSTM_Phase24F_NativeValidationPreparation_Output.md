# Phase 24F — Commit Publication Hardening and Prepare Native Validation

Phase 24H documentation review: execution results and worktree status below are the historical Phase 24F snapshot. Phase 24G subsequently confirmed that `MetaNN/MetaNN` is a valid symlink and that `MetaNN/MetaNN/MetaNNXC.xcodeproj` is accessible. The missing path is the parent project's reference, not the shared project itself. The Phase 24F proposal for a new independent dependency checkout and explicit output overrides is superseded by the user's Phase 24G/H instruction to preserve the existing dependency arrangement and use Rollover DerivedData without a `CONFIGURATION_BUILD_DIR` override. Historical proposed commands below were never executed and are not the current build procedure.

## Outcome

Phase 24D was verified and committed as **`817a674044beb59a9703aaa0960f6817183f9529`**:

`Phase 24D: Harden semantic-worker publication contracts`

The commit contains exactly nine reviewed Phase 24D implementation, regression-test, and documentation files: 585 insertions and 75 deletions. Offline validation passed: **90 Python test methods and six lightweight shell checks**. The Phase 24E GO-for-development / NO-GO-for-production decision was supplied in the task context; no separate Phase 24E report was present in the Phase24 directory inspected here.

**GO for the completed isolated development commit and native-validation preparation. NO-GO for a native build in the dependency state observed during Phase 24F, for native execution without explicit approval, or for production deployment/publication.** Rollover reports an uninitialized Git-submodule entry; its parent-referenced Xcode subproject path is absent, while a nested dependency symlink points into production. This does not establish that the shared project is missing or the symlink is invalid. These were inspected, not repaired. This Phase 24F report was uncommitted at that phase's completion.

No Xcode build, worker launch, publication, binary replacement, PostgreSQL access, scheduler operation, MetaNN initialization/modification, production dependency modification, RepositoryAgent modification, Qwen loading, or Ollama operation occurred. Commands in the native preparation sections below are proposals and were **not executed**.

## Final Phase 24D verification and commit scope

The worktree initially contained only the following Phase 24D changes. All tracked diffs and the two new files were reviewed before staging; the staged path list and whitespace checks were verified before committing.

| Committed file | Reviewed behavior |
| --- | --- |
| `LSTM/InferWorkerMain.cpp` | Existing INFER identity gains authoritative compiled layout and input-width fields. |
| `Scripts/PublishSemanticWorker.py` | Strict single version-1 identity record; role/commit/hash/semantic qualification; current source-contract enforcement even for explicit CLI arguments; current TRAIN width compatibility; prospective registry validation. |
| `Scripts/RollSemanticWorkerLayout.py` | Shared qualification for new dedicated candidates, using one identity invocation per candidate; legacy TRAIN mode retained. |
| `Scripts/RefreshSemanticWorkerGeneration.py` | Both dedicated candidates must prove the intended same-layout semantic contract. |
| `Scripts/tests/test_dedicated_train_rollover.py` | Updated identity fixtures and shared-verifier mocks. |
| `Tests/SemanticWorkerPublisherTests.py` | Updated expected semantic fields and verifier arguments. |
| `Tests/SemanticWorkerPublicationContractTests.py` | Fourteen offline tests, including negative matrices, source-argument bypass, failure atomicity, historical loading, independent commits, and canonical artifact paths. |
| `docs/semantic-workers/semantic-layout-inference-worker-routing.md` | Publication contract, historical compatibility, refresh, and failure-boundary documentation. |
| `docs/phases/Phase24/LSTM_Phase24D_SemanticWorkerPublicationHardening_Output.md` | Historical implementation/validation report, preserved as its Phase 24D snapshot. |

No registry schema, historical dispatch algorithm, scheduler workflow, source semantic version, or persisted worker artifact was changed. AST comparisons against the pre-commit baseline confirmed that `load_registry`, `validate_existing_registry`, `atomic_write_json`, and `update_current_link_after_registry_commit` in `Scripts/PublishSemanticWorker.py` were unchanged. The new call to prospective registry validation changes publication admission, not the validator's historical compatibility behavior. Native registry/dispatch source was also unchanged; historical publisher files are absent from the commit's path list.

Registered workers without the expanded compiled identity remain loadable without execution; the new proof applies to newly published dedicated candidates. Existing legacy TRAIN rollover/import remains supported. Qualification rejection precedes artifact staging and registry replacement; tests compare prior registry/artifact bytes, modes, and symlink targets. Existing post-replacement fsync/link failure semantics remain unchanged.

Source evidence: `Scripts/PublishSemanticWorker.py:211–264` (shared identity qualification), `:726–884` (`publish`), `:905–940` (`main`); `Scripts/RollSemanticWorkerLayout.py:183–224` (`rollover`); `Scripts/RefreshSemanticWorkerGeneration.py:69–105` (`refresh`); `LSTM/InferWorkerMain.cpp:60–92` (identity and early dispatch).

## Final regression results

All commands ran in `/Volumes/Developer SSD/ExpertAdvisor-Rollover`. Candidate subprocesses in the new contract suite are mocked; publication fixtures are disposable. The existing rollover suite builds/runs only a disposable C++ registry-parser test fixture with `clang++`, not a worker or an Xcode project. Optional external registry environment inputs were removed for that suite.

| Executed command | Result |
| --- | --- |
| `PYTHONDONTWRITEBYTECODE=1 python3 Tests/SemanticWorkerPublicationContractTests.py` | 14 passed. |
| `PYTHONDONTWRITEBYTECODE=1 TMPDIR=/private/tmp python3 Tests/SemanticWorkerPublisherTests.py` | 12 passed. |
| `PYTHONDONTWRITEBYTECODE=1 TMPDIR=/private/tmp python3 -m unittest discover -s Scripts/tests -p 'test_dedicated_train_rollover.py' -v` | 12 passed. |
| `PYTHONDONTWRITEBYTECODE=1 TMPDIR=/private/tmp python3 Tests/SemanticWorkerGenerationRefreshTests.py` | 13 passed. |
| `env -u EA_SEMANTIC_REGISTRY_UNDER_TEST -u EA_TRAINING_SELECTION_REGISTRY_UNDER_TEST -u EA_SEMANTIC_REGISTRY_ROLLOVER_UNDER_TEST PYTHONDONTWRITEBYTECODE=1 TMPDIR=/private/tmp python3 Tests/SemanticWorkerRolloverTests.py` | 13 passed. |
| `PYTHONDONTWRITEBYTECODE=1 TMPDIR=/private/tmp python3 Tests/SemanticWorkerHistoricalTrainingCandidatePublisherTests.py` | 6 passed. |
| `PYTHONDONTWRITEBYTECODE=1 TMPDIR=/private/tmp python3 Tests/SemanticWorkerHistoricalInferenceWorkerPublisherTests.py` | 6 passed. |
| `PYTHONDONTWRITEBYTECODE=1 python3 Tests/DedicatedTrainingWorkerArchitectureGuardTests.py` | 14 passed. |
| `bash Tests/DedicatedTrainingWorkerArchitectureTests.sh` | Passed. |
| `bash Tests/ReleaseWorkerBuildConfigurationTests.sh` | Passed. |
| `bash Tests/SchedulerTrainingWorkerRoutingTests.sh` | Passed. |
| `bash Tests/LSTMPhase22Z3InferenceRuntimeCompositionBoundaryTests.sh` | Passed. |
| `bash Tests/LSTMPhase22Z4ManagedInferenceApplicationBoundaryTests.sh` | Passed. |
| `bash Tests/RuntimeFoundationStructuralTests.sh` | Passed. |
| Python AST/syntax, reviewed-file whitespace/newline checks, and disposable-fixture cleanup checks | Passed. |
| `git diff --check` and `git diff --cached --check` before commit | Passed. |

Only the isolated `expertadvisor-repository-rollover` MCP was used: read-only capability inspection and a deterministic read of the updated INFER entrypoint. No semantic/model-backed claim verification was invoked. Local project/plist parsing and source inspection supplied build evidence; `xcodebuild` was not invoked, including for project listing/build settings.

Production `git status --short`, with `GIT_OPTIONAL_LOCKS=0`, was empty before and after tests/commit. The commit path list includes no build products, registry artifacts, dependency changes, or production paths. Test fixtures use temporary directories and never select the production registry. Production ignored artifacts were not rewritten or used for test publication. This is an operation/scope verification, not a cryptographic audit of every ignored production file. The linked worktree shares Git object/ref storage under production's `.git`; the authorized commit updates the Rollover branch and its worktree index, not production checkout content or its active branch.

## Native target and prerequisites

The required shared scheme is **`LSTM Infer Worker`**, in `ExpertAdvisor.xcodeproj/xcshareddata/xcschemes/LSTM Infer Worker.xcscheme:1–28`. It builds target **`lstm-infer-worker`**, identifier `0F8000043800000100AAA001`, product `lstm-infer-worker`. Use the Release configuration for a real source commit; Debug provenance is intentionally unavailable. Do not use `LSTM Release`, the scheduler bundle, Train Worker, scheme Run, or a Clean action.

Project evidence:

- `ExpertAdvisor.xcodeproj/project.pbxproj:1943–1964`: INFER target owns provenance, Sources, and Frameworks phases. Its direct dependencies are SchedulerCore, ProfitabilityCore, StrategyEvaluationCore, ModelInputPreparation, MarketDataCore, and the external `libMetaNN` target. ModelInputPreparation also depends on MarketDataCore; StrategyEvaluationCore depends on ProfitabilityCore. These are libraries, not executable worker dependencies.
- `:1185–1201`: links those local libraries, `libMetaNN.a`, `libMetalBuffer.a`, Foundation, Metal, and MetalPerformanceShaders. Production archives must not be copied in as substitutes for independently built dependency products.
- `:3790–3830`: C++20, macOS deployment target 26.2, libpqxx 7.10.1, libpq, and Release OpenMP/libomp headers/libraries. INFER's Release build uses `NDEBUG`, `LSTM_BUILD`, and `LSTM_TARGET_RELEASE` definitions and writes a linker map under `TARGET_TEMP_DIR`.
- `:2175–2191`: its only shell phase invokes `Scripts/GenerateBuildProvenance.py` and writes `GeneratedBuildProvenance.hpp` beneath `DERIVED_FILE_DIR`. Local library dependencies have no shell phases. The canonical LSTM publication phase belongs to a different target and is not in this dependency graph.
- `Scripts/GenerateBuildProvenance.py:40–75`: Release requires an entirely clean `git status --porcelain`, records exact HEAD, and writes/fsyncs the declared derived header. An untracked Phase 24F report blocks Release provenance in this worktree.

Read-only environment observations: selected developer directory `/Applications/Xcode.app/Contents/Developer`; installed Xcode bundle version **27.0**; host macOS **27.0.1**, architecture **arm64**. AGENTS names Xcode 26.5, so exact toolchain reproduction is not established. Resolve that difference explicitly before treating a native result as the accepted toolchain result. No SDK/compiler invocation was used to build the worker.

The configured Homebrew dependencies exist and resolve to:

- `/Volumes/Darwin/homebrew/Cellar/libpqxx@7.10.1/7.10.1`
- `/Volumes/Darwin/homebrew/Cellar/libpq/18.6`
- `/Volumes/Darwin/homebrew/Cellar/libomp/23.1.3`

Their expected dylib paths exist. They are shared read-only prerequisites, not fully independent dependency installations; no Homebrew update/install/relink was performed. Existence does not prove ABI/architecture compatibility. A later approved native phase must inspect actual linkage and retain versions without modifying these dependencies.

## MetaNN and dependency isolation assessment

`.gitmodules:1–3` names `MetaNN` with URL `git@github.com:vpredoehl/MetaNN.git`. The superproject gitlink is pinned to **`a270e7a5dd239b524fd7d34ad3bb73b646dd7fd6`**. `git submodule status` returned the leading `-` marker for that commit: the Rollover submodule is uninitialized.

The filesystem inspection found:

```text
MetaNN/                            local directory
MetaNN/MetaNN                      symlink -> /Volumes/Developer SSD/ExpertAdvisor/MetaNN
MetaNN/MetaNNXC.xcodeproj           absent
```

The Xcode project references `MetaNN/MetaNNXC.xcodeproj` (`project.pbxproj:712`) and external target `libMetaNN` through the cross-project proxy. The nested symlink provides the project at `MetaNN/MetaNN/MetaNNXC.xcodeproj`, as confirmed in Phase 24G. It does not establish an independent dependency checkout, but shared read-only source consumption can be compatible with isolated outputs. Recursive `$(PROJECT_DIR)/MetaNN/**` header search paths alone do not prove a write into production. `Headers/LSTM.hpp:31` and `LSTM/LSTM.cpp:35–36` require MetaNN headers. Safety depends on the resolved dependency graph, output settings, and build phases.

**Phase 24F did not establish build readiness.** Do not initialize it, retarget the symlink, or borrow production build products in this phase. The original Phase 24F proposal was a separate clean validation source checkout at the Phase 24D commit, with its own exact-revision MetaNN checkout. That proposal is superseded by the user's later instruction to preserve the valid shared dependency configuration and repair only the demonstrably incorrect parent reference. Do not clone the latest MetaNN revision as a substitute.

Before any future build, inspect that independent MetaNN project's dependency graph, build phases, absolute paths, output settings, resource generation, and static initialization. Confirm how it produces `libMetaNN.a`, `libMetalBuffer.a`, `MetaNN_metal.metallib`, and `default.metallib`. The parent publication contract requires the two metallibs (`Scripts/PublishSemanticWorker.py:31–34`); the missing subproject prevents confirming their generation/copy behavior now. No dependency scripts or outputs may resolve into production. Dependency acquisition and clean-checkout setup require separate authorization and were not performed.

## DerivedData and product isolation strategy

Use a fresh independent source checkout under:

`/Volumes/Developer SSD/ExpertAdvisor-Rollover/DerivedData/Phase24Native/Source`

Use a separate fresh output root under:

`/Volumes/Developer SSD/ExpertAdvisor-Rollover/DerivedData/Phase24Native/Run-817a6740`

`DerivedData/` is ignored by the parent `.gitignore:2`; the validation checkout must itself remain clean and pinned to `817a674044beb59a9703aaa0960f6817183f9529`. This leaves the requested uncommitted report in the original worktree while avoiding a dirty-provenance bypass. No such checkout/output directory was created during Phase 24F.

Route derived headers, intermediates, products, linker maps, module cache, and precompiled headers into the fresh output root. Parent project configuration parsing found no explicit `SYMROOT`, `OBJROOT`, `CONFIGURATION_BUILD_DIR`, `DSTROOT`, or `SHARED_PRECOMPS_DIR` overrides. Global command-line overrides are a proposed strategy, not proof of isolation for the absent MetaNN project. Audit its scripts and inspect effective per-target settings before building. Shared Xcode/SDK/Homebrew reads and host CPU/memory resources remain shared; never use a production DerivedData directory or `Builds/SemanticWorkers` as output.

## Proposed build commands — not executed

These commands require explicit native approval, the independent source/dependency setup above, an accepted toolchain, and a completed transitive dependency audit. The preflight must reject any dirty source status, unexpected HEAD/submodule revision, production symlink, missing target/project, or output outside the validation root. Never weaken the clean-provenance generator to accommodate this report.

```bash
phase24_source='/Volumes/Developer SSD/ExpertAdvisor-Rollover/DerivedData/Phase24Native/Source'
phase24_output='/Volumes/Developer SSD/ExpertAdvisor-Rollover/DerivedData/Phase24Native/Run-817a6740'
phase24_commit='817a674044beb59a9703aaa0960f6817183f9529'
test "$(git -C "$phase24_source" rev-parse HEAD)" = "$phase24_commit"
test -z "$(git -C "$phase24_source" status --porcelain)"
test "$(git -C "$phase24_source/MetaNN" rev-parse HEAD)" = \
  'a270e7a5dd239b524fd7d34ad3bb73b646dd7fd6'
test -f "$phase24_source/MetaNN/MetaNNXC.xcodeproj/project.pbxproj"
```

After dependency/script/path review, first inspect effective build settings using the same output overrides as the following command with `-showBuildSettings` in place of `build`. Confirm all per-target output paths, dependency products, generated-header paths, and selected toolchain. The settings inspection itself is reserved for the approved native phase.

```bash
xcodebuild \
  -project "$phase24_source/ExpertAdvisor.xcodeproj" \
  -scheme 'LSTM Infer Worker' \
  -configuration Release \
  -derivedDataPath "$phase24_output" \
  -jobs 2 \
  SYMROOT="$phase24_output/Build/Products" \
  OBJROOT="$phase24_output/Build/Intermediates.noindex" \
  CONFIGURATION_BUILD_DIR="$phase24_output/Build/Products/Release" \
  SHARED_PRECOMPS_DIR="$phase24_output/Build/PrecompiledHeaders" \
  CLANG_MODULE_CACHE_PATH="$phase24_output/ModuleCache.noindex" \
  DSTROOT="$phase24_output/DstRoot" \
  PUBLISH_CANONICAL_LSTM_RELEASE=NO \
  build
```

Retain build logs and the linker map inside the validation output root; verify clean source/submodule state again after build. Resolve compiler warnings/errors before accepting native validation. Do not add a clean step, launch action, publication action, scheduler command, database connectivity test, or dependency update.

## Proposed identity/provenance validation — not executed

`LSTM/InferWorkerMain.cpp:87–91` recognizes exactly `--build-identity` before calling the managed inference CLI. `PrintBuildIdentity` (`:60–82`) reads its canonical executable and invokes `/usr/bin/shasum` through `ExecutableSha256` (`:29–48`); it does not intentionally enter model loading, database access, inference, or training. Linked-library initializers still run before `main`, so the future dependency audit must examine that boundary before execution.

After the separately approved isolated build, inspect linkage without execution, then invoke only the identity branch:

```bash
phase24_binary="$phase24_output/Build/Products/Release/lstm-infer-worker"
test -x "$phase24_binary"
/usr/bin/file "$phase24_binary"
/usr/bin/otool -L "$phase24_binary"
/usr/bin/shasum -a 256 -- "$phase24_binary"
"$phase24_binary" --build-identity
```

Require exit status zero and one version-1 `INFER_WORKER_BUILD_IDENTITY` line, role `lstm-infer-worker`, exact built source commit, canonical validation executable path, compiled layout/width matching the source probe, and a self-hash matching the independent SHA-256. Current source evidence is layout **13** (`Headers/ModelInputExpansion.hpp:23`) and width **171** (`Headers/ModelInputContract.hpp:67–89`); compare authoritative compiled constants rather than relying only on textual integer extraction.

The following proposal uses the same qualification functions as publication without creating artifacts, a lock, a registry, or a convenience link. `current_semantic_contract` compiles/runs a small temporary C++ constant probe (`Scripts/PublishSemanticWorker.py:69–98`), not a worker. This probe and the real identity invocation are reserved for future approval as well.

```bash
PYTHONDONTWRITEBYTECODE=1 TMPDIR=/private/tmp python3 - \
  "$phase24_source" "$phase24_binary" "$phase24_commit" <<'PY'
import importlib.util
from pathlib import Path
import subprocess
import sys

source, binary = (Path(value).resolve(strict=True) for value in sys.argv[1:3])
commit = sys.argv[3]
spec = importlib.util.spec_from_file_location(
    'phase24_qualification', source / 'Scripts/PublishSemanticWorker.py')
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)
assert publisher.clean_source_commit(source) == commit
layout, width = publisher.current_semantic_contract(source)
assert (layout, width) == (13, 171)
before = publisher.sha256(binary)
publisher.verify_embedded_commit(binary, commit)
fields = publisher.read_worker_build_identity(binary, 'infer')
assert fields.get('artifact_role') == 'lstm-infer-worker'
assert fields.get('source_commit') == commit
assert fields.get('executable_sha256') == 'sha256:' + before
assert fields.get('canonical_executable') == str(binary)
publisher.validate_worker_semantic_contract(fields, 'infer', layout, width)
assert publisher.sha256(binary) == before
independent = subprocess.check_output(
    ['/usr/bin/shasum', '-a', '256', '--', str(binary)], text=True).split()[0]
assert independent == before
print('Qualified INFER identity without publication:', layout, width, commit, before)
PY
```

Also retain the generated provenance header and inspect the build log/linker map to ensure local archives/resources were generated in the validation root. Embedded commit plus hash proves candidate identity agreement, not a signed attestation of every dependency. Do not invoke `PublishSemanticWorker.py` as a CLI for this proof: its `main` performs publication.

## Disposable qualification/registry fixture strategy

Re-run the already-passing `Tests/SemanticWorkerPublicationContractTests.py` offline suite from the pinned validation source tree. Its `setUp`, `publish`, `reject_without_publication`, and `qualified_publication` seed disposable old TRAIN/INFER registry artifacts, route through real publisher/rollover/refresh orchestration, mock candidate identity subprocesses, and compare authority snapshots. It requires no live registry and verifies missing/mismatched semantic metadata, role/commit/hash rejection, CLI contradictions, and historical compatibility.

Combine that orchestration coverage with the proposed read-only native proof above. The fixture suite alone does not validate native emitted output, and the native proof alone does not exercise registry replacement. If a later phase authorizes a native fixture integration, seed a fresh registry from synthetic historical fixtures with the native contract; use the qualified native INFER candidate and only its isolated runtime products as inputs; exercise acceptance/rejection strictly within a new disposable artifact root. Preserve snapshot/cleanup assertions and require exact source commit/hash. Do not point the harness at the real Rollover or production registry, seed it by copying a live registry, or run a scheduler against it. Such integration was neither implemented nor executed here.

## GPU/resource impact assessment and remaining risks

Building the INFER target and static libraries is compilation/linking, not a TRAIN/INFER launch. The parent dependency graph includes no historical worker executable and its only relevant script generates provenance. With audited isolated dependency scripts/outputs, a build should not replace files used by running historical workers. A two-job cap reduces, but does not eliminate, shared CPU, memory, disk I/O, and thermal contention on the same machine.

Metal shader compilation should not itself perform model inference/training dispatch. That is an inference from the build purpose, not a guarantee about the absent MetaNN project's scripts. Linking Metal frameworks does not prove that pre-main initializers avoid GPU device allocation. `Tests/MetalRuntimeResolutionTests.sh` explicitly runs a Metal test and is excluded from this preparation and proposed minimum identity validation. No GPU/resource probe or interaction with active workers was performed.

Remaining gates:

1. Acquire and audit an independent exact-revision MetaNN checkout; the current nested production link cannot be used as an isolation shortcut.
2. Provide a clean pinned source checkout without altering or committing this report; provenance must remain fail-closed.
3. Resolve Xcode 27.0 versus the documented 26.5 toolchain expectation, and confirm SDK, architecture, dependency ABI, resources, and effective output paths.
4. Audit transitive scripts, library initializers, and actual linked paths before claiming filesystem/GPU independence. Shared host resource contention remains even with complete output isolation.
5. Obtain explicit approval for the native build and identity execution. Native INFER identity is still unverified by compilation/execution; no deployment or publication is authorized by this development commit.

## Final repository state

Phase 24D is committed. Only this Phase 24F preparation report is untracked; the index is empty.

`git status --short`:

```text
?? docs/phases/Phase24/LSTM_Phase24F_NativeValidationPreparation_Output.md
```

`git diff --stat`: empty (the report is untracked). `git diff --check`: passed. Phase 24D commit stat: **9 files changed, 585 insertions(+), 75 deletions(-)**.

Stop after reporting. No proposed native command was executed, and this report is not committed.
