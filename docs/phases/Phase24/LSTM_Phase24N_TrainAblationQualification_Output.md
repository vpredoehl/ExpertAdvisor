# Phase 24N — TRAIN logging correction and isolated ablation routing qualification

**GO for completion of the logging correction and source/CPU contract qualification of scheduler-managed TRAIN ablation. NO-GO for production publication or deployment.** The existing `train_feature_ablation_v1` semantics are supported by the dedicated TRAIN implementation: canonical persisted masks reach the shared input-materialization hook, controls preserve input values, treatments zero only their named projected columns, and continuation rejects incompatible lineage. Behavioral tests and source review establish this bounded equivalence; they do not establish a successful native database/GPU training or continuation run.

The narrowly scoped correction and regression tests are committed as **`12e31c0299193e4198907701182ccbaad0dcf67a`**. The rebuilt TRAIN candidate passes native identity and publication preflight qualification. INFER remains byte-for-byte unchanged. No capability was published, no production registry was changed, and no production work was launched or interrupted. This report remains **uncommitted**. Stop after reporting.

## 1. Baseline commit and protection evidence

| Check | Evidence |
| --- | --- |
| Repository / branch | `/Volumes/Developer SSD/ExpertAdvisor-Rollover` / `dedicated-train-layout-rollover-squashed-v1` |
| Baseline HEAD | **`90e5a943bb002c1e884338dce0b233ebf89e8943`** |
| Initial worktree | Clean; `git status --short` and `git diff --stat` empty |
| Phase 24M report | Tracked and committed by baseline `90e5a943`, message `Phase 24M: Document dedicated worker qualification and publication blockers`; that commit changes only the report, 254 inserted lines |
| Required reading | AGENTS.md; Phase 24J remediation/Qwen restoration; Phase 24K compatibility; Phase 24L alignment; Phase 24M qualification reports reviewed |
| Existing TRAIN | SHA-256 **`b04e7d084203207ad06aec209ae996127b3ea59fb1668a18ceb17c06ec6c8b16`**, independently verified before the build; qualified source `ff260a5131fe94910f60fe77addd64ad576c12cb` |
| Existing INFER | SHA-256 **`a34cb34d111f746e1757366af8dc36a7ffa49d811db17b779ef68c3c8a2e1d35`**, independently verified before and after; qualified source `d99449d44f4473639c7fab1b831f058d6ac4386b` |
| Nested MetaNN link | **`MetaNN/MetaNN -> /Volumes/Developer SSD/ExpertAdvisor/MetaNN`**; top-level MetaNN is the containing directory |
| Production HEAD | **`b8cdfef03ccdb073caccbf93b0a4282070c0c4d3`**, unchanged |
| Production worktree | Clean in baseline, post-test, post-commit and final read-only snapshots |
| Production registry | SHA-256 **`af1121e80d42af13b3933628954348550214b790f85cc9823b71c156b26e8e79`**, unchanged in all protection snapshots |
| Shared MetaNN | All **265** tracked shared file/symlink entries and both shared Xcode project hashes unchanged |
| RepositoryAgent / Qwen | Preexisting RepositoryAgent worktree status preserved; no source/configuration edit, model query, reload, restart or dependency installation |

Read-only `ps -axo pid,command` inspection identified four active production legacy TRAIN workers (PIDs **21660, 21666, 55503, 55801**), production INFER **5916**, and scheduler **30886**, with screen/login wrappers **30884/30885**. This records observed activity, not a claim that naturally progressing production processes or experiment state were frozen. No process was signaled, no scheduler command was executed, and no PostgreSQL connection or write was made by this phase.

The old development TRAIN executable was retained unchanged under ignored `DerivedData/ExpertAdvisor/Phase24N/baseline/lstm-train-worker`, with its baseline hash verified, before the authorized build replaced the stable development product. No historical production artifact was copied over or modified. The linked Git worktree's shared administrative index required an approved metadata write for the authorized Rollover commit; production HEAD and checkout content remained unchanged.

## 2. Logging correction and reviewed commit

The Phase 24M report correctly locates the definition at **`LSTM/LSTM.cpp:331–332`**, rather than in `Sources/TrainingWorkerApplication.cpp`. The application declares and calls `PrintAndResetDistribution`; its definition is live in TRAIN and absent from the retained INFER symbol table.

The only application-source change is:

```diff
-        printf("row%d: %zu %zu %zu\n",
+        printf("row%zu: %zu %zu %zu\n",
                i,
```

`i` is `size_t`. `%zu` is its supported, type-correct conversion on the preserved Apple compiler/runtime. Row labels, counts, distributions, accuracy output, reset behavior, loop bounds and control flow remain identical. No training calculation, feature, semantic layout, width, experiment selection, project setting or capability-selection policy changed.

Commit **`12e31c0299193e4198907701182ccbaad0dcf67a`**, message **`Phase 24N: Correct TRAIN logging and add ablation qualification regressions`**, contains only:

- `LSTM/LSTM.cpp`: the one format conversion.
- `Tests/TrainingDistributionLoggingTests.py`: compile and execute the actual extracted logger with fixture counters; verify exact output, reset and empty-epoch silence; prove the original `%d` mismatch fails compilation.
- `Tests/TrainingWorkerFeatureAblationTests.cpp`: extend existing mask/materialization and expansion tests.
- `Tests/LaunchArgumentsFreshInitializationSeedTests.cpp`: reject duplicate worker mask options using the real launch parser.
- `Tests/TrainingWorkerPersistedAblationTests.cpp` and `.sh`: compile the real persistence component and actual resume type; exercise its pure mask/continuation validators without connecting to PostgreSQL.
- `Tests/DedicatedTrainAblationRoutingTests.py`: compile the real C++ selector and use production-shaped, disposable synthetic publication/registry fixtures.

Seven committed files, **357 insertions / two deletions**; no report, publisher, registry, project or configuration file included. All tests passed before the commit; `git diff --cached --check` passed. Source was clean before and after the build and native qualification.

## 3. Regression coverage and ablation semantic-contract review

The semantic contract is broader than worker selection. It is implemented across these established boundaries:

| Boundary | Source review / behavioral evidence |
| --- | --- |
| Queue CLI | `ExperimentScheduler.cpp::ParseSchedulerArgs` accepts both `--ablate-features VALUE` and `--ablate-features=VALUE`; syntax is canonicalized through the shared `FeatureAblationMask` implementation. This queue parser was source-reviewed, not executed against a database |
| Persisted identity | `ProductionSchedulerDaemon.cpp::ResolveQueuedFeatureAblationMask` resolves requested names/wildcards against the queued semantic layout and persists concrete canonical names. Resume merge validates the requested mask against source lineage; expansion preserves old treatment and permits only newly appended treatment |
| Scheduler admission | Persisted layout/width and mask determine required capabilities; the reserved attempt retains selected executable/source/hash/runtime identity. Existing routing structural and registry/admission tests pass |
| Managed worker CLI | `TrainingWorkerApplication` requires exact experiment/attempt ownership and TRAIN mode. Scheduler TRAIN argv deliberately omits a duplicate mask override. Actual `ParseLaunchArgs` rejects empty, valid and invalid `--ablate-features=...` options and separated spelling |
| Worker persisted mask | `LoadSchedulerFeatureAblationMask` reads the authoritative experiment mask. `LoadResumeCheckpointConfig` carries the persisted model mask. Ordinary continuation requires canonical equality; expansion calls the shared expansion validator |
| Model construction | Dedicated TRAIN passes the selected persisted mask and input width through `LstmRuntimeConstruction::CreateLstmForRuntimeLogLevel` into the same `LSTM` implementation used by legacy TRAIN |
| Materialization | `LSTM::CalculateBatch` calls `CopyTensorFeaturesForModelInput(..., featureAblationMask)` before appending returns. The constructor validates the mask against the projected feature count. The same shared hook is used by prediction paths |
| Historical compatibility | Projection copies only the model's historical tensor prefix; masks for absent columns fail closed. Historical layout/width bindings remain selector authority rather than being inferred from current source age |

Direct worker rejection of `--ablate-features` is **intentional**, not a discovered implementation defect: `SchedulerTrainingWorkerRoutingTests.sh` explicitly forbids forwarding this queue option to managed TRAIN. Matching and mismatching worker overrides are both rejected before a persisted value can be superseded. The capability's canonical parser is the shared mask implementation used by queue resolution and worker persisted-mask decoding. No new capability name or changed CLI/registry semantics is needed.

The new behavioral fixtures exercise the **actual shared implementation**, not a detached zeroing algorithm:

- Empty fresh and persisted controls produce identical projected values.
- Every one of the **115 registry entries plus two separately named directional features** is parsed, applied, and checked column-by-column. Fresh and persisted canonical masks produce identical results.
- Named masks, duplicates, whitespace and family wildcards canonicalize consistently; unknown names, malformed/unresolvable wildcards and trailing empty tokens reject.
- Masked columns become zero; all other projected columns remain unchanged; source tensor values and the four appended-return destination slots remain untouched. Existing parity tests separately verify return construction.
- Historical projection rejects a mask whose column is absent from the learned input prefix.
- The real persisted validator accepts controls and reordered/canonical equivalent masks; mismatches reject with `FEATURE_ABLATION_MASK_LINEAGE_MISMATCH`.
- Ordinary continuation preserves the same mask and rejects widened or removed treatment. Explicit expansion allows newly appended feature treatment while rejecting removed source treatment, changed historical treatment and insufficient target width.

The persisted validator test compiles `Sources/PersistedModelRuntimeConfig.cpp` itself, links the existing libpqxx 7 for header type-name initialization, and dead-strips unused database workflows. Its executable calls only pure validators; it does not instantiate a database connection or execute training. Suppressions cover existing shared-header/libpqxx diagnostics, not the logging format defect. Fixture setup initially needed the complete shared include paths and pinned libpqxx 7 headers ahead of incidental `/usr/local` headers; final checked-in harness passes. These setup failures were not defects in the native Xcode worker build and caused no dependency changes.

Source comparison confirms the legacy and dedicated applications use the same persisted-mask/continuation components and LSTM materialization hook. **Verified equivalence is the canonical mask, projected feature vector and lineage contract.** Full PostgreSQL queue/persistence execution, masked native GPU training, checkpoint persistence, native continuation and model-result equivalence remain unexecuted; they are not reported as passing.

## 4. Isolated routing fixture results

Schema-5 fixtures retain a layout-13 / width-171 legacy TRAIN with `train,infer,analyze,train_feature_ablation_v1`, priority 0; append current dedicated TRAIN, priority 1; retain dedicated INFER; and include a historical layout-9 / width-103 TRAIN. Only synthetic bytes/resources are staged under temporary fixture roots. No native candidate is published or executed. The real registry loader, selector, canonical-path/runtime verifier and actual publisher fixture helpers are used.

| Case | Plain dedicated `train` | Hypothetically qualified `train,train_feature_ablation_v1` |
| --- | --- | --- |
| Fresh control without model-bearing identity | Dedicated TRAIN | Dedicated TRAIN |
| Persisted 13/171 control | Retained legacy TRAIN | Dedicated TRAIN |
| Fresh ablation without model-bearing identity | Reject: current TRAIN lacks required capability | Dedicated TRAIN |
| Persisted 13/171 ablation | Retained legacy TRAIN | Dedicated TRAIN |
| Canonical family-wildcard ablation | Retained legacy TRAIN | Dedicated TRAIN |
| Unknown mask | Reject before selection | Reject before selection |
| Control continuation, 13/171 | Retained legacy TRAIN | Dedicated TRAIN |
| Masked continuation, 13/171 | Retained legacy TRAIN | Dedicated TRAIN |
| Historical 9/103 control / ablation | Historical TRAIN | Historical TRAIN |
| Unsupported historical lifecycle mask | Reject before selection | Reject before selection |
| Missing continuation identity | Reject | Reject |
| Incompatible layout/width pair | Reject | Reject |

All **26 routing subcases** pass. Fixture bytes remain unchanged by probing. Existing registry tests also preserve INFER routing, role/capability separation, historical identity, command identity and deterministic capability/priority selection. No selection implementation, priority, historical binding or registry semantics was edited. A hypothetical capability claim is a routing test input, not evidence by itself of implementation conformance.

## 5. TRAIN build results

Effective settings for TRAIN Debug and Release plus its eight dependency schemes were read using `xcodebuild -showBuildSettings`. Both TRAIN configurations remain **ARM64 / macOS 27.0**. Products, intermediates, derived sources, caches and temporary files resolve under the prescribed stable Rollover DerivedData; `DEPLOYMENT_LOCATION=NO`.

The five in-project static libraries retain 15.0 deployment settings; MetaNN/Metal targets retain 26.2. The four validation-bearing shared libraries retain `-O3`. No project object, shared-source setting, Homebrew path, Apple compiler selection or Metal toolchain setting changed.

Exact native build command:

```bash
TMPDIR='/Volumes/Developer SSD/ExpertAdvisor-Rollover/DerivedData/ExpertAdvisor/TemporaryFiles' \
PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 \
/usr/bin/xcodebuild \
  -project '/Volumes/Developer SSD/ExpertAdvisor-Rollover/ExpertAdvisor.xcodeproj' \
  -scheme 'LSTM Train Worker' -configuration Release \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath '/Volumes/Developer SSD/ExpertAdvisor-Rollover/DerivedData/ExpertAdvisor' \
  -jobs 2 \
  'CCHROOT=/Volumes/Developer SSD/ExpertAdvisor-Rollover/DerivedData/ExpertAdvisor/Caches.noindex' \
  'CACHE_ROOT=/Volumes/Developer SSD/ExpertAdvisor-Rollover/DerivedData/ExpertAdvisor/Caches.noindex' \
  -resultBundlePath '/Volumes/Developer SSD/ExpertAdvisor-Rollover/DerivedData/ExpertAdvisor/Phase24N/NativeBuildARM64.xcresult' \
  build
```

**BUILD SUCCEEDED, exit 0, 11.34 seconds**, from clean committed source. This incremental build compiles two units, `LSTM.cpp` and `TrainWorkerMain.cpp`, and reuses unchanged dependencies. No Clean, `CONFIGURATION_BUILD_DIR` override, INFER build, scheduler/analyzer executable build or production build ran. Actual toolchain remains XcodeDefault **Apple clang 21.0.0 (`clang-2100.3.34.2`)**, Xcode **27.0 (`27A266a`)**, SDK 27.0. Native worker target is `arm64-apple-macos27.0`.

For settings inspections, the same project/DerivedData/cache/temp arguments were used with `-scheme '<scheme>' -configuration '<Debug-or-Release>' -arch arm64 -showBuildSettings` instead of destination/result-bundle/build arguments. All ten inspections passed.

## 6. Artifact identity, SHA-256 and publication qualification

| Property | New TRAIN result |
| --- | --- |
| Stable artifact | `DerivedData/ExpertAdvisor/Build/Products/Release/lstm-train-worker` |
| Architecture | Thin ARM64 Mach-O executable; `file` and `lipo -archs` agree |
| Size / UUID | **2,197,568 bytes** / **`430E1799-F9BC-3553-ACBE-7A523F1E6FA2`** |
| `LC_BUILD_VERSION` | Minimum macOS **27.0**, SDK **27.0** |
| Identity version / role | **1 / lstm-train-worker** |
| Embedded source commit | **`12e31c0299193e4198907701182ccbaad0dcf67a`** |
| Layout / model input width | **13 / 171**, also checked through a compiled source-contract probe |
| Final SHA-256 | **`018333a5694971e1dcd97a97d8f4a2018243f3a7ec96ee728dd74f4954a577d9`** |
| Provenance | Generated header, binary strings, native identity and clean build commit agree |
| Native identity query | Exactly one `--build-identity` invocation from `/`; exit 0, empty stderr; early dispatch precedes worker/database entry |
| Native validation | Role-specific identity validator accepts independent source/hash/layout/width and canonical executable path |

All **12 TRAIN negative cases** reject before staging: wrong layout, width, role, source commit or hash; omission of each of the six required identity fields; and omission of both semantic fields. Fixture registry authority, immutable files/modes and current links remain unchanged. Twelve INFER negatives were also revalidated against the retained Phase 24L native-derived identity response, with **zero new INFER executions**.

Real paired same-layout refresh preflight accepts the independently qualified TRAIN and retained INFER records, actual executable hashes and actual shared runtime resources. A separate hypothetical `feature_ablation_qualified=True` preflight also accepts the metadata contract. Both stop at an intercepted first runtime-staging call: no native worker staging, registry write or current-link update occurs. Identity has **no capability-attestation field**; semantic conformance comes from the tests/source review, not this preflight.

INFER remains source `d99449d44f4473639c7fab1b831f058d6ac4386b`, hash `a34cb34d111f746e1757366af8dc36a7ffa49d811db17b779ef68c3c8a2e1d35`, ARM64 / 27.0 / role INFER / 13 / 171. Unlike the Phase 24L report-only amendment, the current source delta includes a real logging fix and regression tests. Explicit compatibility review finds the sole application change confined to `PrintAndResetDistribution`: native `nm` confirms it is present in TRAIN and absent from the retained INFER executable. No INFER runtime/feature/semantic configuration changed, so rebuilding INFER is unnecessary. Independent embedded commits are valid under the paired contract.

## 7. Runtime dependencies and signing

Both candidates retain identical direct load paths and the same recursively inspected **11 non-system path references / 10 distinct resolved dylib files**. Every required non-system library remains ARM64, matches Phase 24K's hash/resolution and passes strict signature verification. `otool -D` confirms preserved absolute Homebrew install names; inspected images have no rpaths or dependency load names into production.

| Preserved component | Version | Minimum macOS / SDK |
| --- | --- | --- |
| libpq / libpqxx | 18.6 / 7.10.1 | 26.0 / 26.5 |
| libomp | 23.1.3 | **27.0 / 27.0** |
| OpenSSL libssl / libcrypto | 3.6.5 | **27.0 / 27.0** |
| MIT Kerberos/GSSAPI family | 1.22.2 | 26.0 / 26.2 |
| Conditional PostgreSQL OAuth bundle | PostgreSQL 18 | 26.0 / 26.5 |
| Conditional OpenSSL legacy provider and three engines | OpenSSL 3.6.5 | **27.0 / 27.0** |

All five known conditional modules retain baseline hashes, ARM64 architecture and valid strict signatures. No inspected required image/module minimum exceeds 27.0. Apple system dependency edges remain unchanged; arbitrary configuration-selected/computed plugins and full authentication flows remain unverified. No Homebrew dependency was installed, replaced, patched or relinked.

TRAIN signature remains **ad-hoc**, strict verification passes, TeamIdentifier is unset, application identifier is `47QSNDWH88.`, and `get-task-allow=true`. INFER's unchanged ad-hoc signature also passes. Signature integrity is verified; acceptance/signing/notarization policy on intended rollout hosts is an outstanding operational requirement, not an assumed failure.

Actual paired resources remain unchanged: `MetaNN_metal.metallib` SHA-256 `9c894e9e02b3dfafb69d639535ebc064c7b5ef30ce72edd5cf628f220e16f759`; `default.metallib` SHA-256 `a13694e6940e8287c1b3ca696edbcc291e85de2fd514ade52e431054f1d537d5`; deterministic runtime identity **`6c8d208a0aae281f3fbb1a2a38fe2c7d6deb92811b4a41d3615fac67defee34e`**. Resource/package integrity is verified; a native GPU workload was not run.

## 8. Regression commands, results and remaining warnings

All **31 recorded regression command invocations / 30 unique commands** pass. The existing Python suites account for **100 tests**, with two new unittest methods additionally exercising logging and 26 routing subcases. Structural, standalone C++ persistence/ablation/continuation/registry suites also pass. Finite-value validation retains **91 passing checks**, the expected **28 original `-Ofast` failures** as a required negative control, and byte parity across **26 registered widths** at `-O3` and `-O0`.

Commands run with the preserved Apple driver (`CXX=/usr/bin/clang++` where supported), `PYTHONDONTWRITEBYTECODE=1`, isolated temporary output, and all three optional external-registry test overrides removed:

```bash
python3 Tests/TrainingDistributionLoggingTests.py
bash Tests/TrainingWorkerFeatureAblationTests.sh
bash Tests/LaunchArgumentsFreshInitializationSeedTests.sh
python3 Tests/DedicatedTrainAblationRoutingTests.py
bash Tests/TrainingWorkerPersistedAblationTests.sh
bash Tests/TrainingRuntimeConfigTests.sh
bash Tests/PersistedModelRuntimeConfigStructuralTests.sh
bash Tests/SchedulerRuntimeConfigValidationStructuralTests.sh
bash Tests/LSTMModelInputCompatibilityTests.sh
bash Tests/LSTMPhase22UResumeDirectInferenceMigrationTests.sh
bash Tests/ContinuationOrchestrationServiceTests.sh
bash Tests/CorrectedCausalSurpriseReplicationContinuationTests.sh
bash Tests/FeatureAblationPairEvaluationTests.sh
bash Tests/FeatureAblationReplicationEvaluationTests.sh
bash Tests/ReleaseWorkerBuildConfigurationTests.sh
bash Tests/DedicatedTrainingWorkerArchitectureTests.sh
bash Tests/SchedulerTrainingWorkerRoutingTests.sh
bash Tests/LSTMPhase22Z3InferenceRuntimeCompositionBoundaryTests.sh
bash Tests/LSTMPhase22Z4ManagedInferenceApplicationBoundaryTests.sh
python3 Tests/DedicatedTrainingWorkerArchitectureGuardTests.py
python3 Tests/SemanticWorkerPublicationContractTests.py
python3 Tests/SemanticWorkerPublisherTests.py
python3 Tests/SemanticWorkerGenerationRefreshTests.py
python3 -m unittest discover -s Scripts/tests -p test_dedicated_train_rollover.py -v
python3 Tests/SemanticWorkerRolloverTests.py
python3 Tests/SemanticWorkerHistoricalTrainingCandidatePublisherTests.py
python3 Tests/SemanticWorkerHistoricalInferenceWorkerPublisherTests.py
bash Tests/SemanticWorkerRegistryTests.sh
bash Tests/SchedulerSemanticAdmissionTests.sh
bash Tests/ReleaseFiniteValueValidationTests.sh
```

The existing ablation command appears twice in the recorded runner. Exact argument arrays, exit statuses and per-command logs are retained in `DerivedData/ExpertAdvisor/Phase24N/regression-results.json`. Database integration/SQL migration tests were not run. No test queued/resumed experiments or launched native production workers/schedulers.

The incremental native build has **eight warning occurrences / eight distinct messages**: six existing unused variables, one existing unused function and one discovery warning for the unselected LLVM23 installation. The `%d` / `size_t` warning is absent. No format, deployment-minimum or build-error diagnostic remains in this log. Unchanged translation units were reused, so these counts are **not** a full-build warning-debt reduction claim. Phase 24M's broader audited libpqxx/unused/loop/archive warnings remain historical debt; no general warning cleanup or compiler switch is required by this qualification.

## 9. Capability qualification decision and future publication contract

| Classification | Decision |
| --- | --- |
| A. Verified functional capability | The canonical v1 mask, control/treatment projection, persisted canonical identity, continuation lineage and selection contracts qualify at the authorized **source/CPU regression** scope. No functional gap was demonstrated in the supported managed TRAIN path |
| B. Partially verified operational behavior | Database queue/materialization transactions and full native masked/control GPU training/continuation were source-reviewed but not executed end-to-end. No native training success or model-result equivalence is claimed |
| C. Unsupported surfaces | Direct worker `--ablate-features` override and unmanaged TRAIN are intentionally unsupported; INFER/ANALYZE are not TRAIN capability claims. Invalid/absent historical masks and incomplete continuation identity fail closed |
| D. Outstanding operational requirements | Isolated native workflow smoke evidence, supported-host/signing/dependency acceptance, scheduler active-attempt coordination and explicit rollback/publication authorization |

**The dedicated TRAIN candidate can be proposed with the existing immutable metadata `capabilities: ["train", "train_feature_ablation_v1"]` for a future controlled rollout.** This is a bounded semantic qualification decision, not an operational pass or authorization to advertise it in production now. No actual manifest or registry claim was changed in this phase.

The future reviewed proposal must bind role `train`, manifest schema **2**, registry schema **5**, layout **13**, width **171**, exact TRAIN commit/hash above, retained independent INFER commit/hash, and runtime manifest schema **1** / identity above. The supported same-layout paired route is `RefreshSemanticWorkerGeneration.py` with independently supplied role commits and its existing **`--train-feature-ablation-qualified`** opt-in (`feature_ablation_qualified=True` API). Default `train` only would retain the observed persisted-job routing constraint. Generic INFER publishing or the historical legacy-TRAIN publisher is not a substitute.

Do not add `infer`/`analyze`, change priorities, remove historical claims or substitute identity validation for ablation evidence to force selection. Append-only priority and retained historical TRAIN remain governed by the established publisher. Exact native candidates and dependencies must be rehashed/revalidated immediately before any separately authorized publication; a changed Homebrew resolution requires renewed qualification.

## 10. Remaining blockers and recommended next phase

The confirmed logging defect is fixed. No deployment-floor mismatch, identity rejection, dependency closure mismatch or required substantial architectural change was found. The plain-capability routing constraint is understood and the required existing capability's functional contract is verified; no selector redesign is proposed.

**Operational readiness remains unverified**, including native scheduler-managed control/treatment training, checkpoint/model persistence and continuation on isolated development data; exact-floor macOS 27.0 host behavior (current host is 27.0.1); deployment-host signing and absolute Homebrew availability; and rollback execution. The source/CPU qualification must not be presented as those operations passing. No capability metadata was written to production to test these requirements.

Recommended next phase: a separately scoped **isolated native workflow canary and controlled publication/rollback preparation**. Reuse the qualified candidates; use only a development database/fixtures and bounded control/treatment/continuation cases; review active-worker resource contention, expected model identities, current-layout paired refresh, immutable historical retention and rollback authority before any publication proposal. This recommendation does not authorize a database setup, native training run, registry write, scheduler action or deployment now. No substantive implementation fix is requested because none was demonstrated.

**GO** for accepting the scoped logging correction and functional ablation qualification evidence. **NO-GO** for production publication/deployment in this phase; operational requirements and explicit authorization remain outstanding.

## 11. Evidence and final Git state

Ignored evidence is under `DerivedData/ExpertAdvisor/Phase24N`: baseline/report hashes; production/process/shared-source protection snapshots; retained baseline TRAIN; regression harness/logs/results; ten effective-settings records; exact native command/log/result/XCResult and warning summary; TRAIN identity/Mach-O/signature/entitlement/provenance/symbol checks; retained INFER identity replay and negatives; paired positive/negative preflights and intercepted capability preflight; dependency/install-name comparison; runtime manifest; and final qualification summary. These snapshots cover inspected tracked shared/protection files and the registry, not every ignored production artifact.

Final HEAD: **`12e31c0299193e4198907701182ccbaad0dcf67a`**.

`git status --short`:

```text
?? docs/phases/Phase24/LSTM_Phase24N_TrainAblationQualification_Output.md
```

`git diff --stat`: **empty** because this report is untracked. No staged changes; `git diff --check` passes. The correction commit's separate stat is seven files, 357 insertions and two deletions. No push, merge, publication or deployment occurred. Stop after qualification and this uncommitted report.
