# Phase 24J — Native Release Build Warning Audit

**NO-GO for Phase 24K if it entails production publication or deployment of this candidate.** The retained build and native identity qualification are valid within their original scope, but a confirmed floating-point compilation/source-contract conflict remains in live inference input preparation code. The declared macOS 26.2 deployment floor is also inconsistent with a required macOS 27.0 library. No remediation, rebuild, publication, worker execution, or production operation was performed.

The complete log contains **655 warning occurrences**, rather than 654: the earlier count omitted Xcode's uppercase destination warning. There are **142 exact distinct message bodies**, **17 message bodies after normalizing `exec_params` template arguments**, and **10 diagnostic categories**. These counts are not counts of independent defects.

## 1. Baseline and build identity

| Item | Verified evidence |
| --- | --- |
| Audit checkout | `/Volumes/Developer SSD/ExpertAdvisor-Rollover` |
| Audit HEAD | `2f71ef0fa877e77f03d0659ec62275c4a0abbd01`, matching requested `2f71ef0f` |
| Initial working tree | `git status --short` empty; no tracked or untracked changes |
| Phase 24I report | `docs/phases/Phase24/LSTM_Phase24I_NativeInferenceIdentityValidation_Output.md`, read in full |
| Audited log | `DerivedData/ExpertAdvisor/Phase24I/native-build.log`: 8,337 lines; SHA-256 `f97dc8e43b86a4d249c1c608fa7c7b0f77fec6bddd3f09eefaf0b5daf75dd9b9` |
| Build identity | Release / `LSTM Infer Worker` / `lstm-infer-worker` / arm64 |
| Recorded build result | `native-build-result.json`: exit 0, 51.27 seconds; log ends `** BUILD SUCCEEDED **` |
| Independent result-bundle check | `xcresulttool get build-results`: `Build "LSTM Infer Worker"`, succeeded, zero errors; destination arm64 Mac Studio, macOS 27.0.1 |
| Build interval | Result bundle: 2026-10-08 03:53:36.985–03:54:27.507 UTC (October 7 local time) |
| Actual compiler / SDK | Xcode 27.0 (27A266a); Apple clang 21.0.0 (`clang-2100.3.34.2`), XcodeDefault; macOS SDK 27.0 |
| Built source commit | `71c8f88df9bfc46f2d383ab2c64121fb3cecf8a8` |
| Shared MetaNN HEAD | `a270e7a5dd239b524fd7d34ad3bb73b646dd7fd6`, matching Phase 24I |
| Retained native identity | Version 1; role `lstm-infer-worker`; layout 13; input width 171; source commit `71c8f88d…` |
| Executable SHA-256 | `c93c5657dc84ad61e246117bbf655bae2de8f73210e7e61faa30cb2d4eedcbdb`, rehashed during this audit and unchanged |

`git diff 71c8f88d HEAD --stat` shows only the committed Phase 24I report. The executable correctly identifies the source commit at which it was built, not the later documentation commit used for this audit. Its generated provenance header and embedded commit string agree with retained identity evidence. No warning indicates corruption of layout, width, identity generation, or executable provenance.

The log's invocation, successful result, exact nine-target closure, result-bundle title/destination, generated provenance, and unchanged executable hash establish that this is the successful Phase 24I build. No new build was used to establish the baseline.

## 2. Extraction methodology and reproducible evidence

A case-insensitive `\bwarning:` scan of the complete log captured source diagnostics, clang driver warnings, archive/linker warnings, and Xcode build-system warnings. Every matching line was reviewed by diagnostic form. None was an echoed compiler command, note, source quotation, or progress message. The extractor preserves original text, one-based log line number, warning option/category, source location where present, and the enclosing Xcode build-task/target/project context.

Target attribution uses the recorded `(in target '…' from project '…')` task boundaries; global destination/toolchain discovery diagnostics are attributed to the build system. This log groups diagnostic output under those task blocks despite its two-job build. Header inclusion context identifies the infinity diagnostics with `ModelInputPreparation.cpp`, not the header name used by the result-bundle issue navigator.

Excluded from occurrence counts: 616 note lines, 3,023 `In file included from` lines, nine `N warnings generated.` summaries, caret/source context, command invocations, dependency edges, progress lines, and the script-phase dependency-analysis note. The compiler's summaries and notes do not add independent warnings.

Definitions and reconciliation:

| Measure | Count / meaning |
| --- | --- |
| Actual warning occurrences | **655**: 651 clang compiler/driver + one Apple libtool + one Apple linker + two Xcode build-system diagnostics |
| Previous Phase 24I count | 654 lowercase `warning:` diagnostics; add log line 12, `xcodebuild: WARNING: Using the first of multiple matching destinations` |
| Exact unique messages | **142**, warning body only, retaining template specialization text and warning option; strip emitting path/location/prefix |
| Normalized unique messages | **17**, additionally normalize `'exec_params<…>'` to `'exec_params'`; do not merge other messages |
| Unique diagnostic keys | **585**, `(source path, line, column, exact message)`; diagnostics without a source location use a null location |
| Source warning sites | **296** distinct line/column sites in **11 ExpertAdvisor source/header files** |
| Diagnostic categories | **10** (six named clang options, four build/archive/linker categories) |
| Result-bundle warning entries | **608**, an aggregated issue list, not the raw log occurrence count |

The result bundle lists one `-Ofast` message versus 37 log occurrences, 592 `exec_params` issues versus 602 log occurrences, and omits the destination-selection warning. The remaining 14 entries reconcile exactly (nine unused, two infinity, one format, one archive, one linker), plus its toolchain entry. Thus `655 - 36 - 10 - 1 = 608`. Its issue URLs also misattribute both infinity locations to `TG4ProductionStreamingPulseAdapter.hpp`; the raw diagnostic locations and include stacks are authoritative for this audit.

Disposable, ignored evidence is retained under `DerivedData/ExpertAdvisor/Phase24J/`:

- `extract_warnings.py`, `warning-lines.txt`, and `warning-occurrences.json`: all 655 occurrences with location/context.
- `unique-messages.json`: all 142 distinct exact messages, occurrence counts, log lines, and target sets.
- `warning-summary.json` and `affected-build-tasks.json`: category, source, target, and build-task inventories. These include the 37 translation units with `-Ofast` warnings.
- `xcresult-build-results.json`: independent build result/issue reconciliation.
- `binary-inspection.json`, `dependency-inspection.json`, object symbol/disassembly logs, complete executable disassembly, and focused `tg2-`, `tg3-`, and `prepare-binary-disassembly.log` evidence.
- `embedded-provenance.json`: read-only embedded commit and executable digest check.

The inventory preserves every emitted template diagnostic. All 602 deprecated-declaration warnings name `exec_params`, representing 284 unique call sites and 568 distinct location/message keys. Generic and specialized diagnostics are emitted for the same calls; repeated header use contributes another 34 occurrences. The 126 exact distinct deprecated messages therefore describe one API migration family, not 126 independent incompatibilities.

## 3. Warning classification

“Unique” below means exact distinct message bodies. Parenthetical values are distinct location/message keys where repetition materially changes the count. The two floating-point rows share an underlying optimization-policy problem; their occurrences are counted separately without adding the later disassembly findings as new log warnings.

| Root cause / suggested class | Occurrences | Unique | Representative diagnostic / location | Affected targets | Severity and recommended action |
| --- | ---: | ---: | --- | --- | --- |
| Deprecated libpqxx API, A / external-library API | 602 | 126 (568 keys) | `'exec_params' is deprecated: Use exec(zview, params) instead.`; `SchedulerOwnershipRepository.hpp:167` | SchedulerCore 342; INFER 260 | **INFORMATIONAL** for demonstrated runtime/ABI risk. Existing wrapper remains implemented. Plan a bounded API migration or explicitly document acceptance; preserve transaction/query behavior. |
| Deprecated optimization spelling, A / I | 37 | 1 | `argument '-Ofast' is deprecated`; representative `ModelInputPreparation.cpp` compile task | StrategyEvaluationCore 7; SchedulerCore 27; ProfitabilityCore 1; ModelInputPreparation 1; MarketDataCore 1 | **ACTION REQUIRED**. Establish the intended floating-point policy. Merely replacing the spelling with `-O3 -ffast-math` preserves the serious assumptions described below. |
| Unused local variables, B | 6 | 6 | `unused variable 'run_cpu_reference'`; `LSTM.cpp:2037` | INFER | **INFORMATIONAL**: no demonstrated inference defect from non-use. Review conditional diagnostics and dead counters; clarify the unused initialization `limit` before any separate cleanup. |
| Unused functions, B | 3 | 3 | `unused function 'CommandContainsOptionValue'`; `ExperimentScheduler.cpp:9244` | SchedulerCore 1; INFER 2 | **INFORMATIONAL**: unreferenced helpers; no missing authoritative call established. Remove or justify in a bounded cleanup. |
| Infinity versus finite-only floating-point compilation, F / inference correctness | 2 | 1 (2 keys) | `use of infinity is undefined behavior due to the currently enabled floating-point options`; TG2 header:635 and TG3 header:1012 | ModelInputPreparation | **BLOCKER** for the associated confirmed live source-contract conflict. Resolve the numerical compilation contract before publication. Actual inference-output effects and the warned sentinel paths are separately bounded/UNKNOWN below. |
| Variadic format/type mismatch, C / F | 1 | 1 | `%d` expects `int`, argument `size_t` / `unsigned long`; `LSTM.cpp:332` | INFER | **ACTION REQUIRED** for reusable source. Confirmed undefined behavior if invoked, but containing function is dead-stripped from this executable; no demonstrated runtime risk in the exact candidate. Future fix should use the size-type format. |
| Invalid discovered LLVM23 installation, I | 1 | 1 | `failed to load toolchain: could not find Info.plist` | Global Xcode preparation | **INFORMATIONAL** for this build: not selected. Repair/remove the stale installation reference later to prevent environment ambiguity. |
| Higher deployment floor in libomp, G / H | 1 | 1 | `building for macOS-26.2 … libomp.dylib … newer version 27.0` | INFER link | **ACTION REQUIRED** before deployment; **BLOCKER for any claimed 26.2-compatible release** until the dependency floor is reconciled. Accepting a documented 27.0+ floor would still require deployment validation. |
| Empty MetaNN archive member, J / K | 1 | 1 | `libtool: warning: 'metal_copy.o' has no symbols` | MetaNN archive | **INFORMATIONAL**: comments-only source yields no externally linkable code. This is not a shader/compiler failure. Remove empty membership or document it in a later dependency-maintenance change. |
| Ambiguous matching destination, L / Xcode configuration | 1 | 1 | `Using the first of multiple matching destinations` | Global xcodebuild invocation | **INFORMATIONAL**: arm64 explicitly requested, selected destination and output independently confirmed. Make destination selection explicit in a future invocation if desired. |
| **Total** | **655** | **142** | | | |

No other category occurs. In particular, there are **zero warnings** for dangling references, uninitialized values, ordinary narrowing/conversion options, missing/undefined symbols, duplicate symbols, or incompatible architectures. The only format warning is a variadic type mismatch, not a lossy conversion diagnostic. The only explicit undefined-behavior warning text concerns infinity; the format warning also has undefined-behavior semantics. There are no direct concurrency/thread-safety or lifetime warning diagnostics. Absence of warnings is not a runtime safety or race-freedom proof: compilation includes `-Wno-conversion`, `-Wno-float-conversion`, `-Wno-sign-conversion`, and `-Wno-unused-parameter`, and linking includes `-no_warn_duplicate_libraries`.

### Ownership and affected files

Diagnostic emission and ownership are different: Apple clang emits application diagnostics, while the libpqxx deprecation annotation is in external headers.

| Disjoint source/configuration ownership | Occurrences | Explanation |
| --- | ---: | --- |
| ExpertAdvisor source and target settings | 651 | 614 source-location warnings plus 37 `-Ofast` target-setting warnings; 602 source warnings originate from calls to a deprecated external API |
| Shared MetaNN source/archive membership | 1 | Empty `metal_copy.mm` archive member; no shared-source compiler warning |
| External library binary | 1 | Homebrew libomp deployment floor |
| Apple/Xcode discovery and selection | 2 | LLVM23 discovery and matching destinations |

External libpqxx contributes the root annotation for 602 warnings, but all those primary diagnostic locations are application call sites. No external-header implementation or shared MetaNN template diagnostic is present. “No warnings” for those dependencies is limited by their `-isystem` inclusion and the compiler options actually used.

| Primary diagnostic source file (relative to repository) | Occurrences | Distinct location/message keys |
| --- | ---: | ---: |
| `Sources/SchedulerOwnershipRepository.hpp` | 40 | 8 |
| `Sources/SchedulerCore/SchedulerWorkerRegistration.cpp` | 4 | 4 |
| `Sources/SchedulerCore/SchedulerPhasePriorityRepository.hpp` | 2 | 2 |
| `Sources/SchedulerCore/ProductionSchedulerDaemon.cpp` | 226 | 224 |
| `Sources/SchedulerCore/ExperimentScheduler.cpp` | 79 | 79 |
| `Headers/CausalTrendLineBreakRetestBehavior.hpp` | 1 | 1 |
| `Headers/CausalFibonacciConfluenceIntegration.hpp` | 1 | 1 |
| `Sources/RunMetadata.cpp` | 4 | 4 |
| `LSTM/LSTM.cpp` | 8 | 8 |
| `Sources/GlobalExperimentControl.cpp` | 241 | 241 |
| `Sources/ContinuationPolicyPersistence.cpp` | 8 | 8 |
| **Source-location total** | **614** | **580** |

Target totals, including driver/archive/linker warnings: SchedulerCore **370**, INFER **270**, StrategyEvaluationCore **7**, ModelInputPreparation **3**, ProfitabilityCore **1**, MarketDataCore **1**, MetaNN **1**, and global build system **2**. MetaNN_metal and MetalBuffer have **zero** warning occurrences. The nine-target graph has no additional executable build. `InferWorkerMain.cpp` has no primary warning location; that does not qualify its linked dependencies as warning-free.

## 4. Critical investigations

### LLVM23 discovery versus selection

Log line 19 records Xcode failing to discover `/Users/vjp/Library/Developer/Toolchains/LLVM23.xctoolchain`. The current installation is a stale symlink to `/opt/homebrew/Cellar/llvm/23.1.1/Toolchains/LLVM23.xctoolchain`; that destination and its `Info.plist` are absent.

The successful build's **62 CompileC command invocations** explicitly use `/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang`. Its one link command uses that toolchain's `clang++`; seven archives use its `libtool`. Retained settings identify the same `TOOLCHAIN_DIR`/`DT_TOOLCHAIN_DIR`, and retained `compiler.log` records Apple clang 21.0.0. Read-only current `xcrun --find clang` and explicit compiler `--version` corroborate this. **LLVM23 was discovered unsuccessfully, not selected for native compilation.** There is no demonstrated ABI or provenance contamination from it.

Metal compilation uses the separate Apple-provided tool at `/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v27.1.266.1.4W6Gm5/Metal.xctoolchain/usr/bin/metal`; it is unrelated to the stale LLVM23 installation. Actual Xcode 27.0 differs from AGENTS.md's Xcode 26.5 baseline: **ACTION REQUIRED** to explicitly accept/document the release toolchain or requalify the intended one in a separately authorized phase.

### libomp and deployment compatibility

The linker names a **dynamic library**, not an application object: `/opt/homebrew/opt/libomp/lib/libomp.dylib`. Current resolution is `/Volumes/Darwin/homebrew/Cellar/libomp/23.1.3/lib/libomp.dylib`, arm64, SHA-256 `cb679440b0af57131274b6c0bcc11b8c18ef7ad45e2a6625ef57e90379fc2ae4`.

`otool -l` confirms `LC_BUILD_VERSION` platform macOS, **libomp minos 27.0 / SDK 27.0**; executable **minos 26.2 / SDK 27.0**. The executable has a required `LC_LOAD_DYLIB` entry for that absolute Homebrew path, not a weak load or optional plugin. Libraries load before the worker can dispatch its identity/inference CLI branch.

This could affect runtime loading or API availability on 26.2–26.x; declaring a lower executable floor does not lower a dependency's requirement. This audit does **not** claim that this specific dyld must reject the image solely because of its minos field: no older-OS loader experiment or dependency API availability analysis was performed. It does establish that 26.2 compatibility is not supported by the linked dependency contract. The prior identity success on macOS 27.0.1 only proves that host's identity invocation succeeded.

The current transitive dependency inspection also found macOS **27.0** floors in `libssl.3.dylib` and `libcrypto.3.dylib`, reached through libpq. They produce no additional diagnostics in the native log. These are **current runtime dependency findings**, not proof of their exact bytes at Phase 24I build time: Homebrew paths are mutable, and the retained executable hash does not cover them. Address the whole dependency closure when establishing a deployment floor, not only libomp.

### `metal_copy.o` and Metal products

The archived `metal_copy.o` came from shared `MetaNN/MetaNN/data_copy/metal_copy.mm`, which contains comments only. Its existing object is Mach-O arm64. `nm` shows only assembler-local `ltmp0`/`ltmp1` labels; `otool -tvV` shows an empty text section. No callable/global definition is supplied. Its companion header declares a `CopyBuffer` template, but the inspected shared source tree has no use of that name outside the declaration.

Thus the empty archive-member warning is expected from the actual source content for this Release ARM64 build, rather than an architecture-specific exclusion or a failed `.metal` kernel. The final link map loads `continuous_memory_metal.o` and `metal_matmul.o`, not `metal_copy.o`; there is no missing-symbol warning or link failure. This explains this warning without claiming all Metal paths are implemented or runtime-qualified.

Both `MetaNN_metal.metallib` and `default.metallib` were generated successfully in the retained log. No Metal compiler/linker warning occurs. GPU loading, kernel execution, and numerical parity remain **UNKNOWN/unverified** because this audit neither executed inference nor loaded Metal resources.

### Floating-point options, infinity, and confirmed live validation loss

Log lines 4360/4368 identify:

- TG2 `TrendLineBehaviorTracker::PairOuter`, header line 635: initialize `bestDistance` with positive infinity, then choose the nearest eligible outer line with deterministic tie breaking.
- TG3 `FibonacciConfluenceTracker::ClassifyConfluence`, header line 1012: initialize `minimumDistance` with positive infinity, then accumulate minimum level distance and possibly normalize by ATR.

The enclosing `ModelInputPreparation.cpp` compile uses a retained response file containing `-Ofast`; settings report `GCC_OPTIMIZATION_LEVEL = fast`. No later conforming optimization override or infinity/NaN-preservation option appears in that compile task. The dedicated executable's own translation units use `-O3`; their setting does not retroactively govern a static library's compilation.

Clang documents that fast-math permits assumptions excluding NaN/infinite operands and results; `-Ofast` adds potentially nonconforming optimizations. The diagnostic is consistent with the explicitly infinite source sentinel. Changing the deprecated spelling while preserving fast-math would preserve this conflict. [Clang floating-point options](https://clang.llvm.org/docs/UsersManual.html#cmdoption-ffast-math), [Clang optimization options](https://clang.llvm.org/docs/CommandGuide/clang.html#code-generation-options).

**Important reachability boundary:** the existing `ModelInputPreparation.o` symbol inventory does not emit `PairOuter` or `ClassifyConfluence`. The final link map selects their live definitions from `Tensor.o` (object 11), compiled with `-O3`. The warnings arise while parsing included definitions and therefore do **not**, by themselves, prove that these two particular sentinel operations execute with fast-math in this final executable. Their resulting inference impact must not be invented from warning text.

However, the same object **does** supply live `Tensor` construction and TG2/TG3 configuration validation. Source and final executable disassembly establish a concrete contract difference:

| Contract | Source | Existing executable evidence |
| --- | --- | --- |
| TG2 tolerances must be finite and nonnegative | `CausalTrendLineBreakRetestBehavior.hpp:396–407`: `std::isfinite(value) && value >= 0.0` | Live validator at `0x1000D5AD4`, originating from object 24 (`ModelInputPreparation.o`), checks each double with `fcmp …, #0.0` / `b.lt`; there is no finite-value test. Positive infinity passes those tolerance checks when the remaining configuration is valid. |
| TG3 confluence tolerance must be finite and nonnegative | `CausalFibonacciConfluenceIntegration.hpp:833–836`: explicitly rejects non-finite tolerance | Live validator at `0x1000D5DE4`, also object 24, loads the tolerance and uses only a negative-value branch; its finite-value check is absent. |

The source call chain is `ManagedInferenceApplication.cpp:283` → `PrepareInferenceInput` → `ModelInputPreparation::Prepare` → `Tensor` construction and configuration validation. The final link map retains that chain and the affected validators. Focused final-binary disassembly matches the preexisting object's disassembly. This is **confirmed loss of an authoritative validation condition in live input preparation**, inferred directly from machine instructions, without executing a worker or constructing a test input. It reflects the configured optimization assumptions, not evidence that Apple clang disobeyed its flags.

**BLOCKER:** reconcile the finite-value source contract with compilation policy before production publication. Fixed production defaults inspected in source are finite; this finding does not show an observed bad production input, corrupt prediction, or an inference failure. Numerical behavior across real data, finite-value overflow, all inline/coalesced definitions, and all fast-math-compiled dependencies remains **UNKNOWN**. A later remediation phase must establish targeted numerical and non-finite-input evidence under the intended Release policy; no such build or test was authorized here.

### Variadic format mismatch and unused code

`PrintAndResetDistribution` passes a `size_t` loop index to `printf("row%d: …", i, …)` at `LSTM.cpp:332`. The warning is a genuine argument-type mismatch; the formatting contract defines incorrect conversion-argument types as undefined behavior. [POSIX formatted-output specification](https://pubs.opengroup.org/onlinepubs/9799919799/functions/fprintf.html).

The final link map explicitly lists `PrintAndResetDistribution` as `<<dead>>` under `# Dead Stripped Symbols` (line 7765); source search finds its definition without a call. Therefore this source defect is **ACTION REQUIRED**, but not an execution blocker for this exact binary. Do not extend that conclusion to a differently linked LSTM executable. No repair was attempted.

The nine unused diagnostics name six locals (`run_cpu_reference`, `limit`, `isFirstMiniBatch`, and three prediction counters) and three helpers (`CommandContainsOptionValue`, `PrintPhase3CompareStats`, `CanonicalizeExecutablePath`). CPU-reference/compare and first-batch diagnostics have conditional call sites excluded in this Release configuration. Counters have no active consumers. The unused `limit` is accompanied by a Xavier/Glorot comment while initialization actually uses fixed `uniform_symmetric(0.01f)`; this deserves source-intent clarification in later cleanup but does not prove an inference-checkpoint initialization error. No unused diagnostic establishes a lost lifetime, uninitialized read, or missing safety operation.

### Deprecated API compatibility

All 602 deprecated-declaration occurrences concern `pqxx::transaction_base::exec_params`. Installed `transaction_base.hxx:726–730` still defines it as a template forwarding to `exec(query, params{args...})`; the deprecation asks callers to use the new API. The compile include root and linked library both identify libpqxx 7.10. The deprecation is not evidence of an incompatible C++ ABI, missing implementation, or changed persisted workflow.

No database query was executed. Any migration must preserve the existing Repository/workflow boundaries, transaction lifetime, binding order, and persisted behavior. The completed link and Phase 24I identity run are not database workflow regression evidence.

## 5. Existing executable inspection

Inspected only `DerivedData/ExpertAdvisor/Build/Products/Release/lstm-infer-worker`; no invocation of that executable occurred in this phase.

| Property | Read-only finding |
| --- | --- |
| Architecture / type | Thin Mach-O 64-bit arm64 executable; 1,704,656 bytes |
| UUID | `5097145F-1870-3395-A3CF-BDE465A05483`; matching arm64 dSYM UUID |
| Platform / deployment / SDK | `LC_BUILD_VERSION`: macOS, minos **26.2**, SDK **27.0**, linker tool version **27037.1** |
| Runtime loading | Ten direct `LC_LOAD_DYLIB` entries; no `LC_RPATH` or weak-dylib load entries |
| Signing | Valid on disk; strict codesign verification exit 0; **ad-hoc**, `TeamIdentifier=not set`, no Developer ID authority shown |
| Entitlements | `com.apple.security.get-task-allow = true`; application identifier `47QSNDWH88.` |
| Integrity/provenance | SHA-256 unchanged from Phase 24I; expected historical source commit still embedded |

Direct dependencies recorded by `otool -L`:

- `/opt/homebrew/opt/libpq/lib/libpq.5.dylib` (compatibility 5.0.0, current 5.18.0).
- `/opt/homebrew/opt/libpqxx@7.10.1/lib/libpqxx-7.10.dylib` (compatibility/current 0.0.0).
- `/opt/homebrew/opt/libomp/lib/libomp.dylib` (compatibility/current 5.0.0).
- Foundation, Metal, MetalPerformanceShaders, CoreFoundation, `/usr/lib/libc++.1.dylib`, `/usr/lib/libSystem.B.dylib`, and `/usr/lib/libobjc.A.dylib`.

All inspected non-system direct/transitive Mach-O dependencies contain arm64. Current libpq/libpqxx and Kerberos components declare macOS 26.0; current libomp/OpenSSL declare 27.0. All discovered absolute non-system dependency paths resolve on this host. System-framework/shared-cache availability on another OS and runtime symbol binding were not tested. Read-only linkage inspection neither opens PostgreSQL nor initializes a worker.

These absolute Homebrew runtime paths are deployment assumptions. Publisher `RUNTIME_RESOURCE_SPECS` packages the two Metal resources; it does not package these dylibs. A valid worker SHA and Metal runtime manifest therefore do not attest the dylib dependency closure or guarantee installation on a deployment host. **ACTION REQUIRED:** explicitly establish dependency provisioning/version policy and signing/debug-entitlement acceptance before deployment. Ad-hoc integrity verification is not Developer ID distribution or notarization qualification.

The initial entitlement read used the deprecated `codesign --entitlements :-` display syntax and generated a tool warning during this audit. It was repeated with supported `--entitlements -`; both were display-only. This audit-generated warning is excluded from the original native-build warning count.

## 6. Release risk assessment and required remediation

| Qualification layer | Assessment |
| --- | --- |
| Native build qualification | **PASS within scope**: existing Release ARM64 build succeeded, correct compiler/architecture verified, unchanged executable/provenance retained. It was not warning-free and does not establish inference correctness. |
| Phase 24I publication identity qualification | **PASS within original scope**: retained version/role/commit/hash/layout/width acceptance, twelve negative cases, 51 Python tests, and four structural checks. These results were read, not rerun here. |
| Current-HEAD publication admission | **Not qualified**: candidate embeds `71c8f88d…`, audit HEAD is `2f71ef0f…`; current-rule publisher requires exact clean HEAD and exact candidate commit. The committed difference being documentation-only does not relax that gate. The uncommitted report also makes the working tree dirty as required by this task. |
| Production publication/deployment readiness | **NO-GO**: confirmed floating-point validation-contract loss; unsupported claimed 26.2 dependency floor; host dependency/signing/toolchain decisions and inference/Metal numerical behavior remain unqualified. |

Required work belongs to a separately authorized phase; none was performed here:

1. **BLOCKER — floating-point/source contract:** correct the compilation-policy/source-contract conflict and requalify the affected Release input preparation and numerical behavior. Explicitly preserve rejection of non-finite values and deterministic feature decisions. Suppressing warnings or renaming `-Ofast` alone is not a resolution.
2. **ACTION REQUIRED; conditional BLOCKER — deployment floor:** reconcile executable and entire required dylib closure. A 26.2 release must use a compatible dependency closure; alternatively, formally choose and validate an appropriate higher deployment floor. Exact loading behavior on older macOS is **UNKNOWN**, not accepted as safe.
3. **ACTION REQUIRED — format defect:** repair the `%d`/`size_t` mismatch in a later source phase; present candidate's dead stripping explains its limited runtime scope.
4. **ACTION REQUIRED — release configuration:** accept/document Xcode 27.0 versus the stated 26.5 baseline; establish runtime dependency provisioning and the intended signing/debug-entitlement contract.
5. **Admission gate — provenance:** any future candidate must satisfy the publisher's clean-tree/exact-source-commit contract. Changes to binary bytes, including rebuilding or signing, require a new digest/identity qualification. Do not relabel this Phase 24I artifact as having been built at audit HEAD or bypass the gate.
6. **INFORMATIONAL maintenance:** later API migration, stale-toolchain cleanup, empty archive-member cleanup, and unused-code clarification can be bounded independently. Preserve completed phases and shared dependency authority.

No evidence establishes a compiler implementation bug, corrupted semantic identity, ABI architecture mismatch, dangling lifetime, or uninitialized read. Confirmed contract loss, genuine source-level variadic UB, dependency floors, and the remaining UNKNOWN numerical/runtime scope prevent turning successful compilation into production approval.

**Phase 24K recommendation: NO-GO for publication/deployment of this executable.** A separately authorized remediation/qualification increment can address these findings. This audit does not start Phase 24K, modify registries, or grant deployment approval.

## 7. Commands, results, and final review state

No build, application regression suite, compiler probe, inference/identity worker query, scheduler command, PostgreSQL operation, publisher CLI, Qwen/Ollama action, source/configuration repair, commit, or merge was run. The Phase 24I test/build results above are retained historical evidence, not new test results.

Audit commands included:

```text
git rev-parse HEAD
git status --short
git show --stat --oneline HEAD
git diff 71c8f88d HEAD --stat
GIT_OPTIONAL_LOCKS=0 git -C '/Volumes/Developer SSD/ExpertAdvisor/MetaNN' rev-parse HEAD
PYTHONDONTWRITEBYTECODE=1 python3 DerivedData/ExpertAdvisor/Phase24J/extract_warnings.py
/usr/bin/xcrun xcresulttool get build-results --path DerivedData/ExpertAdvisor/Phase24I/NativeBuild.xcresult
/usr/bin/file <existing-executable-and-dependency/object-paths>
/usr/bin/otool -L <existing-executable-and-dependency-paths>
/usr/bin/otool -l <existing-executable-and-dependency-paths>
/usr/bin/otool -tvV <existing-executable-and-object-paths>
/usr/bin/nm <existing-metal_copy.o-and-ModelInputPreparation.o>
/usr/bin/dwarfdump --uuid <existing-executable-and-dSYM>
/usr/bin/codesign -d --verbose=4 <existing-executable>
/usr/bin/codesign --verify --strict --verbose=4 <existing-executable>
/usr/bin/codesign -d --entitlements - <existing-executable>
/usr/bin/xcrun --find clang
/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang --version
/usr/bin/xcodebuild -version
/usr/bin/sw_vers
/usr/bin/strings <existing-executable>
git diff --check
git status --short
git diff --stat
```

`<…>` denotes the paths recorded in the JSON command evidence, not shell commands that were executed with placeholders. Version/metadata queries do not rebuild targets. Python file hashing and inventory-reconciliation assertions were used for read-only integrity/count verification. Targeted source, response-file, link-map, installed-header, and build-evidence reads used `rg`, `sed`, and Python. Clang/POSIX primary documentation informed the interpretation cited above.

Files changed: only this new report, plus ignored disposable analysis files under Rollover DerivedData. **No application behavior, source, compiler flags, project settings, shared MetaNN data, or existing executable bytes changed.** Extraction reconciled all 655 diagnostics; executable/signature/UUID inspection succeeded. Required remediation remains outstanding, and older-OS loading, live inference numerical correctness, Metal runtime behavior, and deployment policy remain unverified.

Final HEAD remains `2f71ef0fa877e77f03d0659ec62275c4a0abbd01`. `git diff --check` passes; tracked diff/index are empty.

`git status --short`:

```text
?? docs/phases/Phase24/LSTM_Phase24J_ReleaseBuildWarningAudit_Output.md
```

`git diff --stat`: empty, because the report is untracked. Nothing was committed or published.
