# Phase 24G — Isolated Native INFER Validation

Phase 24H documentation review: this report accurately records the pre-repair Phase 24G run. Its HEAD, unresolved graph, missing referenced path, and untracked-report status are historical observations, not the post-repair state. Phase 24H's parent-reference repair and verification are documented separately; no native build or identity result is retroactively claimed here.

## Outcome and recommendation

**NO-GO for native build/identity acceptance in this run. Stopped before compilation because the complete INFER dependency graph could not be resolved.** The existing nested MetaNN symlink is valid, its shared project is accessible through that symlink, and the shared checkout remains at the pinned revision. Those observations correct any implication in Phase 24F that the nested symlink itself was broken. However, the parent project currently references a different, absent project path; Xcode's resolved graph contains an unresolved external `libMetaNN` dependency rather than loaded MetaNN/MetalBuffer targets.

No symlink, project, build script, shared dependency, production source, registry, worker artifact, or Git commit was changed. No native worker was built or executed. The requested mocked negative validation was completed: **14 tests passed**, including candidate rejection that preserves registry authority and current-worker links. Native acceptance remains unverified.

This report and the earlier Phase 24F report remain uncommitted. No PostgreSQL access, scheduler interaction, TRAIN/INFER operation, publication, binary replacement, Qwen loading, Ollama operation, RepositoryAgent modification, merge, clean, or destructive Git operation occurred.

## 1. Pre-build safety findings

### HEAD and worktree

HEAD before and after inspection:

`817a674044beb59a9703aaa0960f6817183f9529`

Initial worktree status contained only:

```text
?? docs/phases/Phase24/LSTM_Phase24F_NativeValidationPreparation_Output.md
```

The untracked report is also an independent Release provenance gate: `Scripts/GenerateBuildProvenance.py:40–49`, `release_commit`, rejects any nonempty `git status --porcelain`. No report was hidden/moved, no clean-status check was bypassed, and no provenance script was changed. Dependency preflight stopped this run before that build phase could execute.

### Existing dependency configuration preserved

Observed paths:

| Path | Observation |
| --- | --- |
| `MetaNN/MetaNN` | Valid symlink to `/Volumes/Developer SSD/ExpertAdvisor/MetaNN`. |
| `MetaNN/MetaNN/MetaNNXC.xcodeproj/project.pbxproj` | Accessible through the symlink; read-only inspection succeeded. |
| `MetaNN/MetaNN/MetalSwift/MetalSwift.xcodeproj/project.pbxproj` | Accessible through the shared MetaNN source tree; read-only inspection succeeded. |
| `MetaNN/MetaNNXC.xcodeproj` | Absent. This is the parent project's referenced path. |

Shared MetaNN HEAD is `a270e7a5dd239b524fd7d34ad3bb73b646dd7fd6`, matching the superproject gitlink. Read-only shared MetaNN and production worktree status checks were empty. No submodule initialization or dependency modification was attempted.

The discrepancy is concretely in `ExpertAdvisor.xcodeproj/project.pbxproj:712`: the `PBXFileReference` uses `path = "MetaNN/MetaNNXC.xcodeproj"` and `sourceTree = "<group>"`. Its enclosing main group has no path prefix, and the project has empty `projectDirPath`/`projectRoot`; the extra nested `MetaNN` directory is not supplied by a parent group. The accessible nested project therefore does not demonstrate resolution of that reference. The user's historical successful build is recorded as context, but this inspection does not reproduce that historical build configuration/cache state.

### Target, scripts, and resolved graph

Shared scheme `LSTM Infer Worker` selects native target `lstm-infer-worker` (`ExpertAdvisor.xcodeproj/xcshareddata/xcschemes/LSTM Infer Worker.xcscheme:13–26`). Target `0F8000043800000100AAA001` owns provenance, Sources, and Frameworks phases (`project.pbxproj:1943–1964`). Its required direct dependencies are SchedulerCore, ProfitabilityCore, StrategyEvaluationCore, ModelInputPreparation, MarketDataCore, and external `libMetaNN`.

The INFER target's only shell phase is `Generate infer worker build provenance` (`project.pbxproj:2175–2191`). It invokes `Scripts/GenerateBuildProvenance.py` with Rollover `PROJECT_DIR` and writes/fsyncs the declared `DERIVED_FILE_DIR/GeneratedBuildProvenance.hpp` (`GenerateBuildProvenance.py:60–75`). The local library dependencies have no shell phases. The canonical publication script is attached to another target and is not an INFER dependency.

Read-only inspection of the accessible shared MetaNN and MetalSwift projects found no shell-script phases or custom build rules writing to shared sources. Relevant targets are MetaNN (the cross-project `libMetaNN` identity), MetaNN_metal, and MetalBuffer. Inspected copy phases were empty and restricted to deployment postprocessing. No explicit dependency output-directory override was found in the inspected configurations. Absolute include paths remain build inputs, not demonstrated output destinations. This source audit supports read-only consumption in principle; it is not a substitute for a fully resolved dependency/output graph.

The successful metadata-only Xcode inspection generated a PIF graph containing **only the parent ExpertAdvisor project**. The INFER dependency record for the five local libraries has target GUIDs; the external dependency is only:

```json
{"name": "libMetaNN (from MetaNNXC.xcodeproj)"}
```

No target GUID is present for it, and no MetaNN or MetalBuffer target/project is loaded. The parent Frameworks phase links `libMetaNN.a` and `libMetalBuffer.a` (`project.pbxproj:1185–1201`), so omitting this dependency is not a valid INFER build strategy. No existing production archives were copied or substituted. **Part A did not pass the complete dependency-resolution/isolation gate; Part B was not started.**

### DerivedData and output boundaries

No repository-local `DerivedData` directory existed at the start of this run. The default user Xcode DerivedData location had no `ExpertAdvisor*` directory discovered. Therefore no existing Rollover DerivedData directory was identified/reused. A stable Rollover-specific path was created for metadata inspection and retained evidence:

`/Volumes/Developer SSD/ExpertAdvisor-Rollover/DerivedData/ExpertAdvisor`

Effective parent-target settings from Xcode:

| Setting | Resolved value beneath that DerivedData root |
| --- | --- |
| `BUILD_DIR`, `SYMROOT` | `Build/Products` |
| `CONFIGURATION_BUILD_DIR`, `TARGET_BUILD_DIR` | `Build/Products/Release` |
| `OBJROOT` | `Build/Intermediates.noindex` |
| `DERIVED_FILE_DIR` | `Build/Intermediates.noindex/ExpertAdvisor.build/Release/lstm-infer-worker.build/DerivedSources` |
| `SHARED_PRECOMPS_DIR` | `Build/Intermediates.noindex/PrecompiledHeaders` |
| `MODULE_CACHE_DIR` | `ModuleCache.noindex` |

`DEPLOYMENT_LOCATION = NO`. No `CONFIGURATION_BUILD_DIR` override was supplied. No parent-target build output resolves into production or historical semantic-worker storage. Complete transitive output isolation could not be verified because the external project was unresolved. No binary or semantic-worker artifact was overwritten.

SHA-256 snapshots of the parent project and accessible shared MetaNN/MetalSwift project files were identical before/after metadata inspection. Production/shared MetaNN Git status remained empty. This verifies the inspected project inputs and operation scope, not a cryptographic audit of every ignored production artifact.

## 2. Exact commands and retained inspection evidence

Executed **metadata inspection only**, initially in the sandbox and then with filesystem escalation because the sandbox blocked Xcode's Rollover PIF-cache write:

```bash
/usr/bin/xcodebuild \
  -project '/Volumes/Developer SSD/ExpertAdvisor-Rollover/ExpertAdvisor.xcodeproj' \
  -scheme 'LSTM Infer Worker' \
  -configuration Release \
  -derivedDataPath '/Volumes/Developer SSD/ExpertAdvisor-Rollover/DerivedData/ExpertAdvisor' \
  -showBuildSettings
```

Both process exit codes were 0, but the first output contained `Could not get build settings` due to a denied cache write and was not accepted as successful settings evidence. The escalated metadata retry provided the resolved parent settings. This was a sandbox execution limitation, not an automatic approval-review rejection. No compilation/build action was requested in either invocation.

Retained files under `DerivedData/ExpertAdvisor/Phase24G/`:

- `build-settings.log` and `build-settings-exit-status.txt`: initial attempt, including sandbox/cache and simulator-service diagnostics.
- `build-settings-unsandboxed.log` and `build-settings-unsandboxed-exit-status.txt`: successful metadata retry.
- `resolved-build-graph-summary.json`: resolved projects, target names, and INFER dependency records.
- `project-input-sha256-before.json`: project-input hashes used for the unchanged-input comparison.

The associated PIF cache is under `DerivedData/ExpertAdvisor/Build/Intermediates.noindex/XCBuildData/PIFCache/`. These are ignored development metadata/evidence, not worker build products.

The intended Release **build command was not executed**:

```bash
/usr/bin/xcodebuild \
  -project '/Volumes/Developer SSD/ExpertAdvisor-Rollover/ExpertAdvisor.xcodeproj' \
  -scheme 'LSTM Infer Worker' \
  -configuration Release \
  -derivedDataPath '/Volumes/Developer SSD/ExpertAdvisor-Rollover/DerivedData/ExpertAdvisor' \
  -jobs 2 \
  build
```

## 3–7. Native build, identity, semantic contract, hash, and qualification

| Required result | Status |
| --- | --- |
| Native build exit status / build log | Not applicable: build not attempted; metadata logs are not a build log. |
| Resulting isolated executable | None. Expected product would be `DerivedData/ExpertAdvisor/Build/Products/Release/lstm-infer-worker`; it does not exist. |
| Complete native `--build-identity` output | Not obtained; no executable was run. |
| Identity version, role, commit, and self-hash acceptance | Unverified natively. |
| Compiled semantic layout / model input width | Unverified natively. Source contract is layout 13 (`Headers/ModelInputExpansion.hpp:23`) / width 171 (`Headers/ModelInputContract.hpp:67–89`). |
| Executable SHA-256 | Unavailable; project-input snapshots must not be mistaken for binary hashes. |
| Generated build provenance | None generated. Expected source commit is current HEAD; no claim is made that a binary embeds it. |
| Publication validator accepting the actual native candidate | Not performed. No publication occurred. |

The native identity branch remains source-visible at `LSTM/InferWorkerMain.cpp:60–92`; it reports semantic constants and bypasses managed inference CLI dispatch for exactly `--build-identity`. Shared identity qualification remains in `Scripts/PublishSemanticWorker.py:211–264`. Neither source evidence nor mocked fixture acceptance establishes actual compiled output.

## 8. Negative regression results and authority preservation

Executed:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 Tests/SemanticWorkerPublicationContractTests.py
```

Result: **14 tests passed**, exit 0. No actual candidate executable or production worker ran; identity/embedded-commit subprocesses were mocked. The suite creates and removes disposable registry/artifact roots under Rollover.

`test_semantic_rejection_is_atomic_for_all_modern_routes` covers wrong layout, wrong width, and missing layout/width for INFER-only, dedicated rollover, and both refresh roles. `test_role_commit_and_hash_rejection_is_atomic` covers incorrect role, source commit, and self-reported artifact hash across those routes/roles. `test_malformed_identity_rejection_is_atomic` additionally covers empty/malformed/multiple records, duplicate/empty fields, and missing/unsupported versions.

`reject_without_publication` snapshots existing registry bytes, immutable artifacts/runtime files, file modes, and symlink targets; asserts no staging, registry-write, or current-link update; and compares unchanged authority after rejection. The semantic matrix has 20 negative subcases, role/commit/hash matrix 15, and malformed matrix 40. Other passing tests include valid mocked candidates, source-argument enforcement, current TRAIN-reference width consistency, historical loading without identity execution, independent commits, and canonical artifact-root behavior. Disposable fixture cleanup passed.

## 9. Resource impact

Only project/plist inspection, metadata-only Xcode processing, hashing of three project inputs, and mocked Python tests ran. No C++/Metal worker compilation, native identity execution, model initialization, GPU kernel dispatch, database connection, scheduler operation, or running-worker interaction was requested. Xcode metadata generated ordinary cache/simulator-service diagnostics; no training/inference workload was launched. Shared host CPU/memory/I/O impact was limited to these inspection/tests and was not quantitatively measured.

## 10–11. Remaining risks and GO / NO-GO

1. The shared dependency is accessible and unchanged, but the current parent-project reference is not resolved by the inspected Xcode graph. The historical successful build does not explain this discrepancy. Resolve the actual previously successful project/path/cache configuration before another native attempt; this run makes no configuration repair.
2. The untracked Phase 24F report would independently fail the existing clean-Release-provenance check. Any later authorized attempt must preserve documentation and the clean-source rule, not fabricate provenance or weaken scripts.
3. Parent output paths are demonstrably isolated; dependency output paths must be checked again once the real external targets are loaded. Do not supply production binary archives to compensate for missing targets.
4. Native output, hash, provenance, and validator acceptance remain unverified. Passing mocked regressions does not authorize publication/deployment.

**GO:** retain the unchanged dependency configuration and passing offline rejection evidence. **NO-GO:** native validation acceptance or publication. The explicit pre-build stop condition was honored; no build was attempted.

## Final review state

Files changed in this phase: this new report plus ignored Xcode metadata/evidence under Rollover DerivedData. No tracked source change; index empty; HEAD unchanged.

`git status --short`:

```text
?? docs/phases/Phase24/LSTM_Phase24F_NativeValidationPreparation_Output.md
?? docs/phases/Phase24/LSTM_Phase24G_IsolatedNativeInferenceValidation_Output.md
```

`git diff --stat`: empty. `git diff --check`: passed. Neither report was committed. Stopped after reporting.
