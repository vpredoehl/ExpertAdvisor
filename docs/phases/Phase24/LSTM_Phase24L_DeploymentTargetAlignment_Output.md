# Phase 24L — macOS 27 deployment target alignment

**GO for completion of the authorized deployment configuration alignment and isolated native INFER identity qualification. NO-GO for publication/deployment.** Both dedicated workers now explicitly require macOS **27.0** in Debug and Release. The new native ARM64 INFER candidate declares minimum macOS 27.0, embeds the configuration commit, and passes identity qualification for semantic layout **13** and model input width **171**. Its preserved Homebrew dependency closure has no observed deployment minimum above 27.0.

Only INFER was built and queried for identity. TRAIN was neither built nor executed. No production worker was built or executed, and no binary was published or deployed. Stop after this validation/report; no merge or subsequent phase is authorized.

## 1. Baseline and Phase 24K report commit

| Item | Result |
| --- | --- |
| Repository | `/Volumes/Developer SSD/ExpertAdvisor-Rollover` |
| Branch | `dedicated-train-layout-rollover-squashed-v1` |
| Initial HEAD | `c0aea80a834589469389935f6fe1356214480940`, matching expected `c0aea80a` |
| Initial worktree | Only the requested Phase 24K report was untracked; no tracked changes |
| Phase 24K report review | Reviewed policy, dependency evidence, target scope, acceptance boundaries and retained system-path appendix; historical findings preserved without amendment |
| Report-only commit | **`e4fcaba5e39b06d2f639a8c7812da76851d0a5a8`** |
| Commit message | `Phase 24K: Document macOS 27 deployment policy` |
| Committed file | `docs/phases/Phase24/LSTM_Phase24K_RuntimeDependencyCompatibility_Output.md` only; 992 insertions |

This is a Git worktree whose index is held in the shared Git administrative directory. The requested commits required permission to write that worktree's Git metadata; neither commit changed the production branch, worktree files or production HEAD.

## 2. Deployment configuration changes and configuration commit

Exactly four build-setting values changed in `ExpertAdvisor.xcodeproj/project.pbxproj`:

| Target | Configuration / object ID | Before | After / effective setting |
| --- | --- | --- | --- |
| lstm-infer-worker | Debug / `0F8000503800000100AAA001` | 26.2 | **27.0** |
| lstm-infer-worker | Release / `0F8000513800000100AAA001` | 26.2 | **27.0** |
| lstm-train-worker | Debug / `0FA000063A00000100AAA001` | 26.2 | **27.0** |
| lstm-train-worker | Release / `0FA000073A00000100AAA001` | 26.2 | **27.0** |

`Tests/ReleaseWorkerBuildConfigurationTests.sh` now resolves both target-owned configuration lists and requires explicit 27.0 values in Debug and Release. Existing map-path, scheme, dedicated TRAIN and deployment graph checks remain intact. Four disposable negative fixtures individually reverted each setting and were rejected by this guard.

**Configuration commit: `d99449d44f4473639c7fab1b831f058d6ac4386b`.** Message: `Phase 24L: Align dedicated workers with macOS 27 deployment policy`. It contains only the project and regression-test files: 29 insertions and four deletions. The source worktree was clean after this commit, before/after the native build, and during provenance/identity qualification. This report was created afterward and is deliberately uncommitted.

The present user authorization explicitly extends alignment to TRAIN, superseding Phase 24K's narrower implementation recommendation. TRAIN's declared deployment and compile-time availability floor rises from 26.2 to 27.0 in both configurations. This does not qualify a TRAIN executable or authorize replacing an existing TRAIN worker; its separate build/hash/identity/runtime validation remains outstanding.

Parsed before/after project comparison verified that every other project object and setting is identical. Project defaults and unrelated executables remain at 26.2; the five in-project shared static libraries remain at 15.0. Shared MetaNN, MetaNN_metal and MetalBuffer remain at 26.2. No shared-library floor change was necessary. Apple Clang, C++ settings, Homebrew paths, optimization, signing, Metal toolchain settings, schemes and sources were preserved. Semantic constants and feature widths were not edited.

## 3. Configuration and regression validation

The smallest structural configuration test ran first, followed by these existing offline checks:

```bash
bash Tests/ReleaseWorkerBuildConfigurationTests.sh
bash Tests/DedicatedTrainingWorkerArchitectureTests.sh
bash Tests/LSTMPhase22Z3InferenceRuntimeCompositionBoundaryTests.sh
bash Tests/LSTMPhase22Z4ManagedInferenceApplicationBoundaryTests.sh
python3 Tests/SemanticWorkerPublicationContractTests.py
python3 Tests/SemanticWorkerPublisherTests.py
python3 Tests/SemanticWorkerGenerationRefreshTests.py
python3 -m unittest discover -s Scripts/tests -p test_dedicated_train_rollover.py -v
bash Tests/ReleaseFiniteValueValidationTests.sh
```

All exit **0**: four structural shell checks, **51 Python tests** (14 + 12 + 13 + 12), and finite-value/feature-parity validation. The latter preserves `-O3`, tests non-finite rejection with `NDEBUG`, detects the original `-Ofast` defect as an expected negative control, and compares feature-row parity against `-O0` at the registered widths, including layout 13 / width 171. Normal source compiles use `-Werror`; the negative control deliberately retains its expected fast-math diagnostics. These tests use disposable fixtures and do not launch scheduler, TRAIN or INFER workloads or access production PostgreSQL.

Effective settings were read using the following command template for INFER and TRAIN, each with Debug and Release, and the eight INFER dependency schemes with Release:

```bash
TMPDIR='/Volumes/Developer SSD/ExpertAdvisor-Rollover/DerivedData/ExpertAdvisor/TemporaryFiles' \
PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 \
/usr/bin/xcodebuild \
  -project '/Volumes/Developer SSD/ExpertAdvisor-Rollover/ExpertAdvisor.xcodeproj' \
  -scheme '<scheme>' -configuration '<Debug-or-Release>' -arch arm64 \
  -derivedDataPath '/Volumes/Developer SSD/ExpertAdvisor-Rollover/DerivedData/ExpertAdvisor' \
  'CCHROOT=/Volumes/Developer SSD/ExpertAdvisor-Rollover/DerivedData/ExpertAdvisor/Caches.noindex' \
  'CACHE_ROOT=/Volumes/Developer SSD/ExpertAdvisor-Rollover/DerivedData/ExpertAdvisor/Caches.noindex' \
  -showBuildSettings
```

All **12 successful settings inspections** confirm ARM64 and isolated product/intermediate/cache/temporary paths; `DEPLOYMENT_LOCATION=NO`. Both worker configurations resolve to 27.0. Dependency schemes: libMetaNN, MetaNN_metal and MetalBuffer resolve to 26.2; SchedulerCore, ProfitabilityCore, StrategyEvaluationCore, ModelInputPreparation and MarketDataCore resolve to 15.0. The four finite-validation-bearing libraries still resolve to optimization level 3. No `CONFIGURATION_BUILD_DIR` override was used.

The first sandboxed settings attempt returned an Xcode settings-cache permission error and supplied no usable target settings despite exit 0. It was rejected by the validation harness, retained as evidence, and rerun successfully with permission. No failed settings output was accepted as validation.

## 4. Isolated native INFER Release build

Actual successful command:

```bash
TMPDIR='/Volumes/Developer SSD/ExpertAdvisor-Rollover/DerivedData/ExpertAdvisor/TemporaryFiles' \
PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 \
/usr/bin/xcodebuild \
  -project '/Volumes/Developer SSD/ExpertAdvisor-Rollover/ExpertAdvisor.xcodeproj' \
  -scheme 'LSTM Infer Worker' -configuration Release \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath '/Volumes/Developer SSD/ExpertAdvisor-Rollover/DerivedData/ExpertAdvisor' \
  -jobs 2 \
  'CCHROOT=/Volumes/Developer SSD/ExpertAdvisor-Rollover/DerivedData/ExpertAdvisor/Caches.noindex' \
  'CACHE_ROOT=/Volumes/Developer SSD/ExpertAdvisor-Rollover/DerivedData/ExpertAdvisor/Caches.noindex' \
  -resultBundlePath '/Volumes/Developer SSD/ExpertAdvisor-Rollover/DerivedData/ExpertAdvisor/Phase24L/NativeBuildARM64.xcresult' \
  build
```

**BUILD SUCCEEDED, exit 0, 23.17 seconds.** The destination selects ARM64; no competing `-arch` option is passed. An initial invocation combining `-arch arm64` and a destination was rejected by Xcode with exit 70 before compilation; its command/log/result bundle are retained separately. No Clean ran.

The dependency graph contains INFER and its eight required libraries/resources. Actual compile/provenance/link/sign tasks ran only for **lstm-infer-worker**: 20 C/C++/Objective-C++ translation-unit compiles, provenance generation, link and signing. Existing shared products were reused. No dedicated TRAIN, scheduler, analyzer or monolithic LSTM executable was built. All output paths remained under the prescribed Rollover DerivedData.

Observed toolchain: **Xcode 27.0 (`27A266a`)**, SDK **27.0**, XcodeDefault **Apple clang 21.0.0 (`clang-2100.3.34.2`)**. Actual worker compile/link target: `arm64-apple-macos27.0`. Historical AGENTS.md wording names Xcode 26.5; this task retains and explicitly records the installed Apple toolchain rather than switching compilers. No Metal toolchain selection or Homebrew dependency changed.

### Retained warnings

The native build is **not warning-free**: **270 occurrences / 80 distinct message bodies**. Categories: 260 libpqxx deprecations, six unused variables, two unused functions, one `%d`/`size_t` format defect, and one preexisting unselected LLVM23 toolchain-discovery warning. Every message matches the prior full native audit or Phase 24J remediation; comparison with only the incremental remediation log would incorrectly label recompiled INFER warnings as new. No new warning message or macOS dependency-floor warning appears. The former libomp 27.0-versus-26.2 linker diagnostic is absent.

These retained defects are explicitly documented under the user's bounded deployment-setting scope; no unrelated source/API migration or toolchain cleanup was added. The format-warning function `PrintAndResetDistribution` and its confusion-matrix string are again marked `<<dead>>` in this candidate's link map. It remains a reusable-source defect requiring separate correction. Deprecations/unused declarations do not establish an observed deployment-floor failure. Existing warning debt is not represented as resolved or as general release acceptance.

## 5. Embedded identity, final artifact hash and qualification

| Item | Verified result |
| --- | --- |
| Candidate | `DerivedData/ExpertAdvisor/Build/Products/Release/lstm-infer-worker` |
| Format / architecture | Mach-O thin ARM64 executable; `file` and `lipo -archs` agree |
| Size / UUID | 1,704,752 bytes / `8B0E765A-F934-3B12-A502-8F9F43111FA1` |
| Mach-O `LC_BUILD_VERSION` | Minimum macOS **27.0**, SDK **27.0** |
| Embedded source commit | **`d99449d44f4473639c7fab1b831f058d6ac4386b`** |
| Identity contract / role | **1 / lstm-infer-worker** |
| Semantic layout / model input width | **13 / 171**, matching the source contract probe |
| SHA-256 | **`a34cb34d111f746e1757366af8dc36a7ffa49d811db17b779ef68c3c8a2e1d35`** |
| Identity self-hash | Matches independently recomputed Python and `/usr/bin/shasum` hashes |
| Provenance | Generated `GeneratedBuildProvenance.hpp`, binary `strings` and native identity contain the exact clean configuration commit |
| Canonical executable | Matches the actual Rollover DerivedData candidate path |
| Signature | Ad-hoc; `codesign --verify --strict` passes; no TeamIdentifier; `get-task-allow=true` retained |

The sole candidate execution was its early-exit `--build-identity` branch, invoked from `/` after reviewing its dispatch. It precedes managed inference CLI dispatch, does not connect to PostgreSQL, and starts no inference/training work. Exit **0**; stderr empty. No production executable was queried.

The unchanged publication validator `verify_inference_build_identity` accepted the captured actual native response against the independently computed hash, clean source commit and source-probed 13/171 contract. Subsequent checks replayed this capture rather than reexecuting the worker. One matching native identity was accepted and **12 negative cases** were rejected: wrong layout, width, role, commit or hash; missing each of identity version, role, commit, hash, layout or width; and both semantic fields missing together. Disposable authority snapshots/current links remained unchanged; staging, atomic registry writes and current-link updates were mocked and asserted absent. No successful native publish operation or publisher CLI ran.

The executable hash was rechecked after qualification and remained unchanged. Passing identity admission is not full runtime-package, model/GPU, database or operational publication qualification.

## 6. Runtime dependency compatibility

The new candidate's direct/transitive non-system Mach-O loads were recursively reinspected. Each image's declared linked paths match Phase 24K; all **11 path references / 10 distinct resolved dylib files** retain the same canonical paths and SHA-256 hashes. Duplicate opt/Cellar crypto references identify one file. All are ARM64, have no inspected rpaths, and pass strict signature verification.

| Preserved dependency | Installed version | Minimum macOS / SDK |
| --- | --- | --- |
| libpq | 18.6 | 26.0 / 26.5 |
| libpqxx | 7.10.1 | 26.0 / 26.5 |
| libomp | 23.1.3 | **27.0 / 27.0** |
| OpenSSL libssl / libcrypto | 3.6.5 | **27.0 / 27.0** |
| MIT GSSAPI / krb5 / k5crypto / com_err / krb5support | 1.22.2 | 26.0 / 26.2 |

Five known conditional ARM64 modules were separately rechecked against their Phase 24K hashes: libpq OAuth helper (minimum 26.0 / SDK 26.5), OpenSSL legacy provider and padlock/capi/loader_attic engines (minimum 27.0 / SDK 27.0). All hashes are unchanged and strict signatures pass. No dependency minimum above **27.0** was found in this inspected closure/module set. The previous 26.2-policy mismatch is resolved by the authorized worker-floor alignment, without replacement dependencies or load-command edits.

Apple system loads remain the same declared edges into the host's OS-supplied frameworks/runtime. The retained Phase 24K report contains the full historical 686-system-path cache inventory; this phase refreshed the new executable and non-system/module closure, not every system-cache image. Host identity loading succeeded on **macOS 27.0.1 (`26A434`)**. This does not establish exact macOS 27.0 runtime behavior, every imported API, or every externally configured/computed `dlopen` plugin. Absolute mutable Homebrew opt/Cellar dependencies remain an operational availability/requalification requirement; the worker hash alone does not pin them.

## 7. Protection, remaining publication blockers and recommendation

Production HEAD remains `b8cdfef03ccdb073caccbf93b0a4282070c0c4d3`; production worktree status remains empty. The intact symlink is **`MetaNN/MetaNN -> /Volumes/Developer SSD/ExpertAdvisor/MetaNN`**. The initial preflight mistakenly checked the containing directory and was corrected without changing any symlink/source. All **265** tracked shared MetaNN file/symlink snapshot entries and both shared Xcode project hashes matched after build and after qualification. RepositoryAgent's preexisting modified/untracked status also matched; no RepositoryAgent or Qwen configuration was altered.

Read-only process inspection observed active production training/inference workers and the scheduler. No process was signaled or restarted; no scheduler control/status executable, experiment command or production PostgreSQL command ran. No registry was modified, no binary was copied to semantic-worker storage, and no branch was merged. Source/status snapshots cover the inspected files, not every ignored production artifact; production protection is also established by the isolated executed commands and output settings.

| Remaining boundary / blocker | Assessment |
| --- | --- |
| Publication/deployment authorization | **Not granted**; no publishing or deploying in this phase |
| TRAIN native qualification | **Outstanding**; configuration is aligned but no new TRAIN artifact was built, hashed or runtime-qualified |
| Exact minimum host | **Unverified on macOS 27.0**; current-host identity evidence is from 27.0.1 |
| Full inference / GPU / model / database / authentication behavior | **Unverified here**; no operational workload or production fixture was used |
| Homebrew availability and configured plugins | Preserve recorded closure; requalify changed dependencies and required isolated plugin/authentication paths |
| Signing/distribution | Ad-hoc integrity passes; external-distribution signing, entitlement and notarization acceptance remain separate |
| Existing compiler-warning debt | Retained and audited, including dead-stripped format defect; requires bounded maintenance/release acceptance outside this four-setting change |
| Toolchain reproduction | Record actual Xcode 27.0/Apple Clang baseline; historical Xcode 26.5 wording does not reproduce this candidate |

**GO** for Phase 24L's configuration alignment, build and identity/dependency-floor validation. **NO-GO** for publication/deployment or for claiming complete TRAIN/INFER runtime qualification. Stop here.

## 8. Files, evidence and final Git state

Files changed during this task:

1. `docs/phases/Phase24/LSTM_Phase24K_RuntimeDependencyCompatibility_Output.md` — committed unchanged as the report-only commit.
2. `ExpertAdvisor.xcodeproj/project.pbxproj` — four dedicated-worker deployment values only, committed separately.
3. `Tests/ReleaseWorkerBuildConfigurationTests.sh` — target-owned 27.0 deployment regression guard, included in the configuration commit.
4. `docs/phases/Phase24/LSTM_Phase24L_DeploymentTargetAlignment_Output.md` — this uncommitted report.

Ignored validation evidence is retained under `DerivedData/ExpertAdvisor/Phase24L`: configuration-scope comparison, negative guard cases, 12 effective-settings logs/JSON, regression commands/logs/results, successful and rejected build command/log/result records, result bundles, native identity/qualification and negative fixtures, binary inspection/signature/provenance evidence, dependency/module inspection, build-warning/task audit, dead-strip evidence, protection snapshots and final validation. No validation helper is committed.

Final HEAD: `d99449d44f4473639c7fab1b831f058d6ac4386b`.

`git status --short`:

```text
?? docs/phases/Phase24/LSTM_Phase24L_DeploymentTargetAlignment_Output.md
```

`git diff --stat`: **empty** (the remaining report is untracked). `git diff --check`: passes. No staged changes remain. The Phase 24L report is intentionally not committed.
