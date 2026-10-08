# Phase 24M — Dedicated TRAIN/INFER qualification and publication readiness

**GO for the completed development artifact qualification and paired publication preflight. NO-GO for actual production publication/deployment in this phase.** Dedicated TRAIN builds successfully under macOS **27.0**, reports the correct version-1 TRAIN identity and **layout 13 / input width 171**, and passes native identity, provenance, hash, signing, dependency and negative qualification checks. Existing Phase 24L INFER is reused unchanged and passes requalification without rebuilding or executing it.

Combined publisher support exists and accepts the independent source commits. Two concrete readiness findings remain: a plain-TRAIN refresh would not reroute persisted controls/ablation jobs from the retained ablation-capable legacy TRAIN worker, and the known `%d`/`size_t` format defect is live in TRAIN (dead-stripped in INFER). Operational training/inference, exact-floor host behavior and deployment/rollback execution were not tested. No production publication, deployment, scheduler action or experiment change is authorized or performed.

## 1. Baseline and protection evidence

| Check | Result |
| --- | --- |
| Repository / branch | `/Volumes/Developer SSD/ExpertAdvisor-Rollover` / `dedicated-train-layout-rollover-squashed-v1` |
| Initial and final HEAD | **`ff260a5131fe94910f60fe77addd64ad576c12cb`**, matching the requested HEAD |
| Initial worktree | Clean; `git status --short` and `git diff --stat` empty |
| Required reading | AGENTS.md, Phase 24J remediation, Phase 24K compatibility and Phase 24L alignment reports reviewed; Phase 24K system appendix contains 686 retained path entries |
| Existing INFER SHA-256 | **`a34cb34d111f746e1757366af8dc36a7ffa49d811db17b779ef68c3c8a2e1d35`**, independently verified before and after qualification/build |
| INFER source provenance | **`d99449d44f4473639c7fab1b831f058d6ac4386b`**, predating the report amendment |
| Amendment comparison | `git diff --name-status d99449d4 ff260a51` contains only the addition of `docs/phases/Phase24/LSTM_Phase24L_DeploymentTargetAlignment_Output.md` (185 lines); source/project/test content is identical |
| Production HEAD | **`b8cdfef03ccdb073caccbf93b0a4282070c0c4d3`** |
| Production worktree | Clean before and after; no production source/artifact/database write |
| MetaNN | **`MetaNN/MetaNN -> /Volumes/Developer SSD/ExpertAdvisor/MetaNN`**; top-level MetaNN is the containing directory, not the symlink |
| Shared-source protection | All **265** tracked shared MetaNN file/symlink snapshot entries and both shared project hashes match before/after snapshots |
| RepositoryAgent / Qwen | Preexisting RepositoryAgent status preserved; no source/configuration edit, model query, reload or service restart |

Read-only `ps -axo pid,command` inspection found six active legacy production TRAIN workers (PIDs 19535, 19541, 21660, 21666, 55503, 55801), one production INFER worker (PID 5916), and the active scheduler (PID 30886, with screen/login wrappers 30884/30885). No process was signaled or executed by this task. No production scheduler status/control command or PostgreSQL command ran.

The source worktree stayed clean through the provenance-sensitive build and native qualification. Helpers, caches, result bundles and evidence are ignored under Rollover DerivedData. This report was created afterward and is the only remaining untracked deliverable. No commit, push or merge occurred.

## 2. Existing INFER qualification

The Phase 24L artifact was **not rebuilt, replaced, patched, signed again or executed**. Its retained actual native identity response was matched to the unchanged executable hash, replayed through the unchanged validator, and compared with the retained Phase 24L qualification record.

| Property | Reverified result |
| --- | --- |
| Artifact | `DerivedData/ExpertAdvisor/Build/Products/Release/lstm-infer-worker` |
| Architecture / size | Thin ARM64 Mach-O executable / 1,704,752 bytes |
| `LC_BUILD_VERSION` | Minimum macOS **27.0**, SDK **27.0** |
| Identity version / role | **1 / lstm-infer-worker** |
| Semantic contract | **13 / 171** |
| Embedded source commit | **`d99449d44f4473639c7fab1b831f058d6ac4386b`**, independently verified with binary `strings` |
| SHA-256 | **`a34cb34d111f746e1757366af8dc36a7ffa49d811db17b779ef68c3c8a2e1d35`** |
| Signing | Ad-hoc; strict signature verification passes; no TeamIdentifier; `get-task-allow=true` retained |
| Existing native evidence | Phase 24L captured exit-0 response, canonical path, exact source/hash/role/layout/width and accepted validator record remain consistent |

All **12 INFER negative cases** were rerun using native-derived identity fixtures: wrong layout, width, role, source commit or hash; omission of each of identity version, role, commit, hash, layout or width; and omission of both semantic fields. All reject without native artifact staging, registry writes or current-link updates. Disposable authority snapshots remain identical. Runtime dependencies and known conditional modules were refreshed separately; see section 5.

The current report amendment does not invalidate this INFER provenance. Identical embedded commits are neither required nor asserted for the dedicated paired contract.

## 3. Dedicated TRAIN scheme, settings and native build

`LSTM Train Worker.xcscheme` selects PBXNativeTarget **`0FA000043A00000100AAA001`**, buildable **lstm-train-worker**. The target owns its dedicated Sources phase and provenance phase; provenance precedes compilation. `TrainWorkerMain.cpp` dispatches `--build-identity` before `RunTrainingWorkerApplication`. Existing structural guards pass. No monolithic entry point is substituted.

Effective settings were inspected with `xcodebuild -showBuildSettings` for TRAIN Debug and Release and all eight Release dependency schemes. Both TRAIN configurations resolve to **macOS 27.0 / ARM64**. Products, intermediates, derived sources, caches and temporary output paths are under the prescribed Rollover DerivedData; `DEPLOYMENT_LOCATION=NO`.

Dependency graph: ModelInputPreparation, MarketDataCore, ProfitabilityCore, StrategyEvaluationCore, SchedulerCore, MetaNN, MetaNN_metal and MetalBuffer. The five in-project shared libraries retain 15.0 floors; MetaNN/Metal targets retain 26.2. No shared project or setting change was needed. The four corrected validation-bearing libraries and TRAIN retain `-O3`; actual compiler response files contain no `-Ofast`, `-ffast-math` or finite-only override. MarketDataCore's existing separate fast policy is not broadened into a new remediation here.

Exact successful native command:

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
  -resultBundlePath '/Volumes/Developer SSD/ExpertAdvisor-Rollover/DerivedData/ExpertAdvisor/Phase24M/NativeBuildARM64.xcresult' \
  build
```

**BUILD SUCCEEDED, exit 0, 52.34 seconds.** No build defect occurred and no configuration/source fix was implemented. No Clean or `CONFIGURATION_BUILD_DIR` override was used. Actual graph has nine targets; actual compilation contains **67 translation units**: 25 TRAIN, 27 SchedulerCore, seven StrategyEvaluationCore, four MetaNN, and one each for ModelInputPreparation, ProfitabilityCore, MarketDataCore and MetalBuffer. Required shared products were rebuilt/reused inside Rollover DerivedData without changing shared source. MetaNN_metal's existing resource was reused. INFER, scheduler/analyzer executables, monolithic LSTM and production targets were not built.

Toolchain remains XcodeDefault **Apple clang 21.0.0 (`clang-2100.3.34.2`)**, **Xcode 27.0 (`27A266a`)**, SDK **27.0**. Actual TRAIN compile/link target is `arm64-apple-macos27.0`. The installed Apple baseline differs from historical AGENTS.md Xcode 26.5 wording; no compiler, Metal toolchain or Homebrew installation was switched.

For effective-setting inspections, the same project/DerivedData/cache/temp arguments were used with `-scheme '<dependency-or-LSTM Train Worker>' -configuration '<Debug-or-Release>' -arch arm64 -showBuildSettings` instead of the build destination/result-bundle/action. All ten settings inspections succeeded.

## 4. TRAIN identity and artifact qualification

| Property | Verified result |
| --- | --- |
| Artifact | `DerivedData/ExpertAdvisor/Build/Products/Release/lstm-train-worker` |
| Format / architecture | Native thin ARM64 Mach-O executable; `file` and `lipo -archs` agree |
| Size / UUID | **2,197,568 bytes** / **`AE51DFDF-098C-36DF-89B8-F177F1788B8C`** |
| `LC_BUILD_VERSION` | Minimum macOS **27.0**, SDK **27.0** |
| Identity version / role | **1 / lstm-train-worker** |
| Embedded source commit | **`ff260a5131fe94910f60fe77addd64ad576c12cb`** |
| Semantic layout / model input width | **13 / 171**, matching an independent compiled source-contract probe |
| Final SHA-256 | **`b04e7d084203207ad06aec209ae996127b3ea59fb1668a18ceb17c06ec6c8b16`** |
| Provenance | Exact clean source commit in generated TRAIN `GeneratedBuildProvenance.hpp`, binary `strings` and native identity |
| Native identity execution | Exactly **one** `--build-identity` invocation, from `/`; exit 0, empty stderr; no managed work or PostgreSQL access |
| Canonical executable / self-hash | Actual Rollover candidate path and independently computed executable SHA agree |
| Signing / entitlements | Ad-hoc, strict verification passes, no TeamIdentifier; application identifier `47QSNDWH88.`, `get-task-allow=true` |
| Production runtime paths | No Mach-O dylib/rpath dependency on production paths; intentional shared source/debug paths are not runtime worker dependencies |

`verify_worker_build_identity(..., "train", ff260a51…, hash, 13, 171)` accepts the actual captured TRAIN response. TRAIN qualification is independent of INFER, with the TRAIN parser/header, role and expected source/hash. Twelve TRAIN-derived mutations, covering the same wrong/missing fields as INFER, all reject through the paired refresh before staging. Existing registry authority, immutable fixture bytes/modes and current links remain unchanged.

The actual native TRAIN and retained actual INFER records also pass the **real paired refresh preflight** with separate commits. That run uses a disposable current-layout registry, replays captured identity/embedded-commit responses, verifies actual native hashes and actual shared runtime resources, and intentionally throws at the first intercepted runtime-staging call. No worker-staging, atomic registry write or current-link update occurs. Native binaries are never copied into a publication root. Both candidate hashes remain unchanged after all checks.

## 5. Runtime dependency comparison and signing boundaries

Both candidates have the same ten direct strong load paths and the same recursively inspected **11 non-system path references / 10 distinct resolved dylib files**. The opt/Cellar libcrypto aliases resolve to one file. Every required non-system image remains ARM64, retains the Phase 24K hash/canonical resolution, has an absolute Homebrew install name and no inspected rpath, and passes strict signature verification. Explicit `otool -D` inspection confirms install names; no runtime load path points into production.

| Preserved library / module group | Version | Minimum macOS / SDK |
| --- | --- | --- |
| libpq | 18.6 | 26.0 / 26.5 |
| libpqxx | 7.10.1 | 26.0 / 26.5 |
| libomp | 23.1.3 | **27.0 / 27.0** |
| OpenSSL libssl / libcrypto | 3.6.5 | **27.0 / 27.0** |
| MIT GSSAPI, krb5, k5crypto, com_err, krb5support | 1.22.2 | 26.0 / 26.2 |
| Conditional libpq OAuth helper | PostgreSQL 18 | 26.0 / 26.5 |
| Conditional OpenSSL legacy provider and padlock/capi/loader_attic engines | OpenSSL 3.6.5 | **27.0 / 27.0** |

All five known conditional module hashes remain unchanged and signatures pass. No inspected required library/module minimum exceeds 27.0. No dependency was installed, replaced, relinked, packaged or patched. Apple system edges are unchanged into the OS-provided runtime/frameworks; the full historical Phase 24K system-cache inventory was reviewed, not reenacted as a claim of runtime coverage.

The paired runtime-resource verifier accepts the actual stable Release resources:

- `MetaNN_metal.metallib`, published name `MetaNN.metallib`: SHA-256 `9c894e9e02b3dfafb69d639535ebc064c7b5ef30ce72edd5cf628f220e16f759`.
- `default.metallib`: SHA-256 `a13694e6940e8287c1b3ca696edbcc291e85de2fd514ade52e431054f1d537d5`.
- Deterministic runtime manifest identity: **`6c8d208a0aae281f3fbb1a2a38fe2c7d6deb92811b4a41d3615fac67defee34e`**.

These hashes match the runtime identity referenced by the observed production registry, but no production resource file or GPU workload was executed. Both build candidates share the stable product directory; resource agreement proves package-contract agreement, not independently exercised GPU correctness. Shared Metal code resolves resources via the main bundle rather than a hard-coded production directory; no resource-loader redesign occurred.

Host is **macOS 27.0.1 (`26A434`)**. TRAIN identity loaded successfully here; INFER retains its earlier successful identity evidence. Exact macOS 27.0 runtime behavior, configured/computed plugins, TLS/GSSAPI/OAuth flows and complete model/GPU/database behavior remain unverified. Absolute mutable Homebrew dependencies must remain available and requalified after changes. Ad-hoc signature integrity is verified; acceptance on intended deployment hosts and any external-distribution signing/notarization policy remain separate, untested requirements rather than assumed failures.

## 6. Publisher, registry, scheduler, historical compatibility and rollback

**Dedicated TRAIN publisher support is present.** Shared version-1 identity validation accepts role-specific commits/layout/width/hash. `RefreshSemanticWorkerGeneration.py` supports paired same-layout dedicated refresh; `RollSemanticWorkerLayout.py --dedicated-training` supports a genuine layout transition. Both accept independently supplied role commits. The generic `PublishSemanticWorker.py` CLI intentionally publishes INFER only; it is not an unsupported-TRAIN gap in the paired workflows. `PublishHistoricalTrainingCandidate.py` remains a legacy `LSTM_Release` append route; it is not a dedicated-TRAIN publication substitute.

Registry contract is schema **5**, role-aware immutable artifact manifests schema **2**, and runtime manifest schema **1**. Content-addressed paths bind layout, role, source commit and final hash; exact capability claims and selection priorities remain authoritative. Current bindings must include exactly TRAIN and INFER with compatible widths. Advisory publication locking, pre-staging qualification, staged hashing, immutable collision checks, atomic paired registry replacement and post-commit convenience-link handling are preserved and covered by offline tests. No validation bypass switch was used for native qualification.

A **read-only shape/metadata inspection** of the active production registry found schema 5, current layout 13, 18 worker records and one runtime. Current INFER is manifest schema 2 / width 171 / `infer`; current legacy TRAIN is manifest schema 1 / width 171 / `train,infer,analyze,train_feature_ablation_v1`, priority 0. Both current records use source `b3e5d5b68cde0abbbcb951e1d0eccf93b111cd4b`. The recorded registry SHA-256 is `af1121e80d42af13b3933628954348550214b790f85cc9823b71c156b26e8e79` and remains stable through final verification. This is metadata inspection, not full live artifact/admission validation or a registry write. It points to a same-layout refresh contract, not a layout rollover.

### Verified routing constraint

Scheduler selection uses persisted layout/width, authoritative capabilities, minimum excess capabilities, then priority. When any exact-layout TRAIN candidate advertises the canonical ablation capability, persisted empty-mask controls use that same qualified selection domain. This established policy is preserved.

A production-shaped synthetic schema-5 fixture retained an outgoing legacy TRAIN with the observed four capabilities and appended a current dedicated TRAIN. The real C++ registry loader, runtime verifier and selector produced:

| Synthetic prospective current TRAIN claim | Fresh training with no model identity | Persisted 13/171 empty-mask control | Persisted 13/171 ablation job |
| --- | --- | --- | --- |
| `train` only | New dedicated fixture | Retained legacy fixture | Retained legacy fixture |
| `train,train_feature_ablation_v1` | New dedicated fixture | New dedicated fixture | New dedicated fixture |

The second row tests a hypothetical, explicitly qualified claim; **this phase does not declare the native TRAIN ablation capability qualified**. Unit-level ablation regressions pass, but masked native training/materialization was not exercised. The native identity itself has no capability-attestation field. Do not set `--train-feature-ablation-qualified` merely to force selection, remove historical claims, alter priorities or redesign the selector. Resolve the intended routing/capability scope through separate bounded qualification before a publication intended to move persisted TRAIN work onto this candidate.

Historical schema-v1 legacy workers and role-aware generations remain supported by existing publisher/registry fixtures. Refresh retains outgoing TRAIN as historical, preserves immutable bytes/manifests/runtime links, and retires the outgoing INFER binding while retaining its immutable artifact. Historical compatibility tests pass; no actual historical artifact was modified or executed.

Failure tests verify pre-staging rejection and prior registry preservation on replacement failure. A link failure after successful registry replacement is reported as **committed**, not rolled back; directory-fsync failure can leave a committed/durability ambiguity. Publication may leave staged but unregistered immutable artifacts after later failure. Before any deployment, separately review the exact rollback authority/workflow, retained TRAIN/INFER artifacts and runtimes, scheduler/active-attempt handling, and prior registry/current-link recovery. No production rollback rehearsal, manual registry restoration, SQL edit or destructive cleanup occurred. Passing failure-atomicity fixtures is not a tested operational rollback procedure.

## 7. Offline regressions and exact commands

All **17 final suites pass**: five structural shell checks, **100 Python tests**, three standalone C++ suites (registry, semantic admission, ablation), and finite-value/parity validation.

```bash
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
CXX=/usr/bin/clang++ bash Tests/TrainingWorkerFeatureAblationTests.sh
CXX=/usr/bin/clang++ bash Tests/ReleaseFiniteValueValidationTests.sh
```

Python counts: architecture guard 14, publication contract 14, publisher 12, refresh 13, dedicated rollover 12, legacy rollover 13, historical TRAIN append six, historical INFER six. Finite-value validation passes **91 checks** at `-O3 -DNDEBUG`; original `-Ofast` negative control detects the known defect; all registered feature-row parity checks match optimized/reference behavior, including layout 13 / width 171. Ordinary standalone source compiles use `-Werror`.

Environment: `PYTHONDONTWRITEBYTECODE=1`, temporary fixtures under `DerivedData/ExpertAdvisor/Phase24M/tmp` where supported. Optional external-registry environment overrides (`EA_SEMANTIC_REGISTRY_UNDER_TEST`, `EA_SEMANTIC_REGISTRY_ROLLOVER_UNDER_TEST`, `EA_TRAINING_SELECTION_REGISTRY_UNDER_TEST`) were removed from the regression runner so registry tests only exercise disposable fixtures. Shell scripts with fixed `/tmp` locations retain their existing disposable cleanup. Any subprocess executable in registry fixtures is synthetic, not a production worker.

One initial ablation invocation failed to find `cassert`/`cstddef`: the explicitly selected bare Xcode toolchain clang++ lacked default SDK header discovery for that script. Retrying the unchanged test with Apple's `/usr/bin/clang++` passed. This is a corrected test-driver invocation, not a worker build failure or source fix; the failed log is retained. The finite test then passed with the same Apple driver and its existing explicit SDK arguments.

Additional exact qualification commands are retained as executable evidence helpers:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 DerivedData/ExpertAdvisor/Phase24M/verify_infer.py
TMPDIR='/Volumes/Developer SSD/ExpertAdvisor-Rollover/DerivedData/ExpertAdvisor/Phase24M/tmp' \
  PYTHONDONTWRITEBYTECODE=1 python3 DerivedData/ExpertAdvisor/Phase24M/qualify_train.py
TMPDIR='/Volumes/Developer SSD/ExpertAdvisor-Rollover/DerivedData/ExpertAdvisor/Phase24M/tmp' \
  PYTHONDONTWRITEBYTECODE=1 python3 DerivedData/ExpertAdvisor/Phase24M/combined_qualification.py
PYTHONDONTWRITEBYTECODE=1 python3 DerivedData/ExpertAdvisor/Phase24M/inspect_train_dependencies.py
PYTHONDONTWRITEBYTECODE=1 python3 DerivedData/ExpertAdvisor/Phase24M/inspect_infer_dependencies.py
/usr/bin/clang++ -std=c++20 -Wall -Wextra -Werror -IHeaders -ISources \
  DerivedData/ExpertAdvisor/Phase24M/routing_probe.cpp \
  Sources/SchedulerCore/SemanticWorkerRegistry.cpp \
  -o DerivedData/ExpertAdvisor/Phase24M/routing_probe
PYTHONDONTWRITEBYTECODE=1 python3 DerivedData/ExpertAdvisor/Phase24M/routing_fixtures.py
```

No live production TRAIN/INFER, PostgreSQL integration test, scheduler execution, experiment queue/resume/state action or native publication was run.

## 8. Remaining warnings and precise TRAIN finding

TRAIN build: **679 warning occurrences / 153 distinct message bodies**:

| Category | Occurrences |
| --- | --- |
| libpqxx deprecated APIs | 662 |
| Unused functions / variables | 3 / 6 |
| Existing MarketDataCore `-Ofast` deprecation | 1 |
| Loop increment unreachable | 1 |
| Unneeded internal declarations | 2 |
| `%d` / `size_t` format defect | 1 |
| Unselected LLVM23 discovery | 1 |
| Comments-only `metal_copy.o` empty archive member | 2 |

No macOS deployment-floor warning occurs. Fourteen exact message bodies were not in the earlier INFER logs: eleven additional libpqxx template instantiations, two unneeded TRAIN application helpers, and the TRAIN single-iteration loop. The two helper definitions are not emitted; the loop ends in an explicit existing `break` at `TrainingWorkerApplication.cpp:1829`. These are newly observed TRAIN compilation diagnostics, not newly introduced source changes or demonstrated build failures. No general warning-remediation effort is required solely because the prior warning debt remains.

A relevant distinction is independently confirmed: **`LSTM/LSTM.cpp:331–332` passes `size_t i` to `printf("row%d: …")`, and `PrintAndResetDistribution` is live in the new TRAIN link map**. Phase 24L's INFER link map dead-stripped it. This is a known concrete source format defect with a different live-code status for TRAIN, not evidence that training model outputs failed in this phase. The narrow proposed follow-up is changing only the row index conversion **`row%d` to `row%zu`**, followed by a clean-provenance TRAIN rebuild and requalification. No patch was applied, and no broad libpqxx/compiler/architecture migration is proposed.

## 9. Readiness classification, blockers and next phase

| Classification | Finding |
| --- | --- |
| **A — Verified** | Native TRAIN build, both ARM64/macOS-27.0 artifacts, distinct version-1 roles, matching 13/171 contracts, exact independent provenance/hashes, strict signatures, unchanged compatible Homebrew closure and runtime-resource agreement |
| **A — Verified** | Dedicated paired publisher preflight accepts separate role commits; both sets of 12 native-derived negative cases reject without staging/authority changes; isolated publication/registry/historical/failure tests and real selector fixture checks pass |
| **B — Unverified** | Exact macOS 27.0 host behavior; native managed TRAIN/resume/checkpoint/materialization and INFER model/GPU behavior; configured authentication/plugins; intended-host signing acceptance; operational scheduler canary and rollback procedure |
| **B — Unverified** | Native artifact's `train_feature_ablation_v1` claim; standalone ablation helper regressions do not establish full masked native materialization behavior |
| **C — Concrete routing constraint** | A plain-TRAIN same-layout refresh does not select the new worker for persisted controls/ablations in the observed capability domain; explicit qualification/scope decision is needed before claiming that routing outcome |
| **C — Concrete source defect** | Live TRAIN `%d`/`size_t` logging mismatch; recommend the single-conversion correction and requalification, not broad warning cleanup |
| **C — Authorization boundary** | Production publication/deployment is expressly unauthorized; no qualification result overrides that boundary |

There is **no demonstrated build failure, missing dedicated TRAIN identity validator, incompatible declared dependency minimum, semantic-width mismatch or source-provenance mismatch**. Independent commits are valid because their diff is documentation-only. Untested operations are not counted as passing. Existing deprecation/unused/archive/discovery warnings alone are not a blanket publication blocker.

Recommended next phase: **Phase 24N — bounded TRAIN capability/routing qualification and targeted logging correction**, separately authorized and still isolated from production. Review the one-line format correction, rebuild/requalify TRAIN from clean committed source, and establish the exact candidate's required ablation/materialization behavior using isolated fixtures before any capability claim. Include the tested intended scheduler-selection domain and an operational rollback/readiness plan; retain immutable INFER identity with an explicit source-compatibility review if code changes in the follow-up. Do not alter scheduler selection, registry priorities, historical artifacts or production rows to force the result. Broader runtime canaries/production publication need their own explicit scope and authorization.

**GO** for completion of Phase 24M development qualification and contract-level readiness evidence. **NO-GO** for actual publication/deployment now, and for claiming operational dedicated TRAIN replacement of persisted jobs without resolving the concrete routing/capability requirement. The report and proposed bounded follow-up are ready for review; no implementation or next phase begins here.

## 10. Deliverable, evidence and final Git state

Only requested source-tree deliverable:

`docs/phases/Phase24/LSTM_Phase24M_DedicatedWorkerQualification_Output.md` — **uncommitted**.

No project, test, application, shared source, dependency, registry, RepositoryAgent or Qwen configuration file changed. The new TRAIN candidate and all ignored qualification evidence reside under the prescribed Rollover DerivedData. Evidence includes baseline/amendment comparison; protection/process snapshots; ten effective-settings logs/JSON; native build command/log/result/bundle and warning/task audit; TRAIN/INFER identities and inspection logs; paired acceptance/negative fixtures; dependency/install-name comparison; runtime manifest; read-only production registry shape; synthetic routing probe; regression commands/results and retry; and final artifact/source checks. Source/status snapshots cover the inspected files, not every ignored production artifact; executed command/output scope establishes the remaining production isolation.

Final HEAD: `ff260a5131fe94910f60fe77addd64ad576c12cb`.

`git status --short`:

```text
?? docs/phases/Phase24/LSTM_Phase24M_DedicatedWorkerQualification_Output.md
```

`git diff --stat`: **empty** (report untracked). `git diff --check`: passes. No staged changes. Stop after this report; no publishing, deployment, merge or push.
