# Phase 24A — Final refactoring gap analysis

Audit date: 2026-10-07, America/Chicago. Assignment: repository inspection and this report only.

Disposition: the standalone process boundaries are implemented, including managed TRAIN. Phase 23 remains closed. Remaining work is a failed architecture guard, publication qualification gaps, and measurable build coupling. A new training engine, general runtime context, scheduler redesign, or wholesale deletion of legacy main is not justified.

## 1. Verified baseline and operational safety

| Check | Observed result |
| --- | --- |
| Inspection/report directory | `/Volumes/Developer SSD/ExpertAdvisor-Rollover` |
| Checkout branch | `dedicated-train-layout-rollover-squashed-v1`; preserved throughout |
| Checkout HEAD | `d9dd89fc29abd05bffa843bb37f182ceafdde65c` |
| Initial status | `git status --short` empty |
| Initial target tracking ref | `origin/lstm-feature-development` was `b8cdfef03ccdb073caccbf93b0a4282070c0c4d3`; requested merge object initially unavailable |
| Refresh | `git fetch origin lstm-feature-development`, from the isolated directory only; succeeded |
| Verified merged audit baseline | `ac12bfcdbdc989c8926d60d366078293960d4cd4`, PR #1; parents `b8cdfef0` and `d9dd89fc` |
| Tree comparison | HEAD and merged baseline both have tree `485a8be6407aa40ecdbf081f0ceeb154debc078b`; `git diff --name-status HEAD origin/lstm-feature-development` empty; `git diff --quiet` returned 0 |
| Original qualified source | `e67a5ce94ebe2ef0163867cb090a7b5b1cba54f8` exists locally; its tracked-tree diff against `d9dd89fc` is empty |
| Build isolation obstruction | `git submodule status`: `-a270e7a5dd239b524fd7d34ad3bb73b646dd7fd6 MetaNN` (uninitialized). `MetaNN/MetaNN` is a symlink to `/Volumes/Developer SSD/ExpertAdvisor/MetaNN`; the expected isolated `MetaNN/MetaNNXC.xcodeproj` is absent |
| Process inspection | `ps -axo pid,comm` denied by sandbox (`Operation not permitted`). No live process inventory or scheduler-status validation claimed |

File/line evidence below is from the isolated checkout, whose tracked tree is identical to the merged baseline. No checkout, reset, clean, new branch, submodule initialization, production-directory traversal, or production build was performed. Fetch updated shared Git tracking metadata; it did not move the checked-out production branch or change its files. `git worktree list` was metadata inspection only.

Production activity is an assignment constraint, not a newly measured result. No LSTM worker executable, scheduler, database command, production registry reader/writer, publisher CLI against real artifacts, or GPU workload was launched. Tests inspected first use source checks, mocks, disposable directories, or small CPU fixtures. The frozen TRAIN identity `c8e74e26f913c31bf73e6bc168c213a1710be098` was never rebuilt. The MetaNN symlink was not followed or altered.

Historical closure reports under `docs/phases` are often short summaries linking detailed reports in the production worktree. Those links were not followed. Their historical build/test claims are distinguished from checks performed in this audit.

## 2. Executable and build inventory

Counts were obtained by parsing the project with `plutil -convert json -o -` and resolving sources, framework phases, target dependencies, and configurations. “Closure TUs” means source-phase compilation memberships across reachable targets in this project, per architecture/configuration, including duplicate compilation of a file in different targets. It excludes the unavailable MetaNN subproject, headers, generated files, and Metal compilation. These are static counts, not timings or observed incremental build events.

| Executable | Entrypoint → composition | Direct TUs / project closure TUs | Core libraries and major dependencies |
| --- | --- | --- | --- |
| `LSTM_Release` | `LSTM/main.cpp:5530` → `RunLegacyTrainingWorkerApplication` (`:4026`); managed INFER delegates at `:4087` | 172 / 345 | ModelInputPreparation, MarketDataCore, SchedulerCore, StrategyEvaluationCore, ProfitabilityCore; MetaNN and libMetaNN; MetalBuffer; Foundation, Metal, MetalPerformanceShaders; pq/pqxx, OpenMP. Explicit build dependency on `lstm-analyze-worker` |
| `lstm-train-worker` | `LSTM/TrainWorkerMain.cpp:94` → `RunDedicatedTrainWorkerMain` (`:87`) → `RunTrainingWorkerApplication`, `Sources/TrainingWorkerApplication.cpp:997` | 25 / 62 | Same five core libraries; MetaNN and libMetaNN; MetalBuffer and three Apple frameworks; pq/pqxx, OpenMP. No dependency on `LSTM Release`, INFER, or ANALYZE executables |
| `lstm-infer-worker` | `LSTM/InferWorkerMain.cpp:84` → shared managed worker CLI → `RunManagedInference`, `Sources/ManagedInferenceApplication.cpp:247` → `RunInferenceRuntime` | 20 / 57 | Same five core libraries; libMetaNN, MetalBuffer, three Apple frameworks; pq/pqxx and Release OpenMP. No legacy main or TRAIN application membership |
| `lstm-analyze-worker` | `LSTM/AnalyzeWorkerMain.cpp:3` → `RunStandaloneAnalyzeWorkerCli`, `Sources/SchedulerCore/ExperimentScheduler.cpp:12395` → `AnalyzeExperimentById`, `:8887` | 136 / 171 | SchedulerCore, StrategyEvaluationCore, ProfitabilityCore, libMetaNN; substantial campaign/recommendation/reporting implementations; Tensor and LSTM sources; MetalBuffer, Foundation and Metal; pq/pqxx |
| `lstm-scheduler` | `LSTM/SchedulerMain.cpp:116` → `RunStandaloneSchedulerDaemonCli`; explicit reconciliation/recovery adapters → production scheduler composition and SchedulerEngine | 10 / 38 | SchedulerCore, ProfitabilityCore, libMetaNN; MetalBuffer, Foundation and Metal; pq/pqxx. No direct Tensor/LSTM or campaign service source membership |
| `lstm-observer` | `LSTM/ObserverMain.cpp:5` → `RunSchedulerObserverCli`, `Sources/SchedulerCore/SchedulerObserverCli.cpp:40` | 1 / 28 | SchedulerCore, pq/pqxx. No explicit MetaNN or GPU framework link inputs |

Project evidence: `ExpertAdvisor.xcodeproj/project.pbxproj:1815` (Release), `:1887` (scheduler), `:1907` (observer), `:1924` (ANALYZE), `:1943` (INFER), `:1965` (TRAIN). Source phases: `:2227` (TRAIN), `:2435` (Release), `:2636` (SchedulerCore), `:2670` (scheduler), `:2687` (observer), `:2695` (ANALYZE), `:2839` (INFER). Exact phases were resolved by ID rather than inferred from comments.

Core library memberships: SchedulerCore 27 TUs; StrategyEvaluationCore 7; ProfitabilityCore 1; MarketDataCore 1; ModelInputPreparation 1. ModelInputPreparation depends on MarketDataCore; StrategyEvaluationCore depends on ProfitabilityCore. SchedulerCore compiles both `ExperimentScheduler.cpp` and `ProductionSchedulerDaemon.cpp` (`project.pbxproj:2640`, `:2645`). Archive compilation and object extraction at link time are different facts.

Build phases generate provenance for Release, scheduler, INFER, and TRAIN. TRAIN provenance is first and `alwaysOutOfDate = 1` (`project.pbxproj:1969`, `:2194`), with a declared generated header (`:2203`). TRAIN/INFER have no executable-local copy phase; their runtime resources must come from the dependency products and publication package. The Scheduler Bundle aggregate depends on scheduler and ANALYZE only (`project.pbxproj:23`). This is its established scope, not a missing TRAIN dependency.

`Scripts/build-release.sh:4` invokes the normal Release build. Ordinary publication belongs to the separate `Publish LSTM Canonical` aggregate (`project.pbxproj:9`, `:2217`), gated by `PUBLISH_CANONICAL_LSTM_RELEASE`. Ordinary Release build is not authority to roll the semantic registry. The explicit Release→ANALYZE dependency is protected as an operational packaging contract by `Tests/ReleaseWorkerBuildConfigurationTests.sh:25`.

### Remaining implementation weight and fan-out

| Measured fact | Architectural implication |
| --- | --- |
| `LSTM/main.cpp`: 5,533 lines | Still a compatibility application, not merely a launcher |
| `Sources/TrainingWorkerApplication.cpp`: 1,866 lines | Dedicated managed TRAIN is extracted but still composes parsing, startup validation, diagnostics, training, checkpoint policy, and final persistence |
| `LSTM/LSTM.cpp`: 7,571 lines; direct membership in Debug, Release, TRAIN, INFER, ANALYZE | Shared numerical/model implementation is compiled separately in five executable targets; process separation does not imply model-code separation |
| `Sources/SchedulerCore/ExperimentScheduler.cpp`: 12,493 lines | Compatibility CLI, analysis and administrative workflows remain in one archive object/source |
| `Sources/SchedulerCore/ProductionSchedulerDaemon.cpp`: 11,088 lines | Established production adapter composition remains large; size alone does not justify reopening the extracted authoritative engine |
| `Sources/GlobalExperimentControl.cpp`: 6,933 lines | Worker registration/control helpers share a TU with broad operational control; retained functionality requires dependency analysis before partitioning |
| `Headers/PgModelIO.hpp`: 1,639 lines; includes `LSTM.hpp` at `:15` | Persistence and materialization interfaces expose the model/Metal type graph to their consumers |
| TRAIN/INFER have 10 identical direct-source memberships | Includes LSTM, Tensor, PricePoint, db_cursor, HistoricalFxTimestamp, economic features/repository, logging, GlobalExperimentControl and CheckpointPolicy; these are shared foundations, not evidence TRAIN calls managed inference |
| ANALYZE/Release have 135 identical direct-source memberships | Normal Release builds both executable targets, so a fresh build compiles these memberships twice; 345 project closure TUs total |

Release dead-code stripping is enabled for workers (`project.pbxproj:3773`, `:3814`, `:3859`); it cannot avoid compiling source-phase memberships. No current link map, binary size, elapsed build time, external subproject fan-out, or actual object extraction was measured. The initial broad inventory script encountered an unresolved build-file ID in an unrelated target; the corrected parser restricted dependency accounting to the audited targets and produced the counts above. Project plist lint passed.

## 3. Completed versus remaining objectives

| Objective | Finding | Repository evidence |
| --- | --- | --- |
| Authoritative scheduler engine and standalone scheduler | Complete; preserve | `Sources/SchedulerCore/SchedulerDaemonCli.cpp:69`; `LSTM/SchedulerMain.cpp:144`; Phase20T report describes production link closure without extracting `ExperimentScheduler.o`; Phase20U records standalone target acceptance |
| Standalone managed INFER and shared compatibility adapter | Complete; preserve | `LSTM/InferWorkerMain.cpp:84`; `LSTM/main.cpp:4075`; `Sources/ManagedInferenceApplication.cpp:247`; Phase22Z5 and Phase23C reports |
| Detached model materialization and per-instance geometry | Complete for accepted inference boundaries | `Headers/LSTM.hpp:51` immutable `hiddenSize_`; `Headers/PgModelIO.hpp:741` owned materialization; `Sources/InferenceRuntime.cpp:74` construction and `:82` detached application; no new generic lifecycle extraction recommended |
| Short RR/RO read followed by independent computation and fresh result transaction | Complete managed-INFER boundary | `Sources/ManagedInferenceApplication.cpp:263`–`:310`: committed RR/RO materialization, runtime, write revalidation, result plus profitability commit; Phase22X closure supersedes earlier Phase22K/Q NO-GO findings |
| Read-only market input and reusable input preparation | Complete boundary | `Sources/MarketDataCore/MarketDataCore.cpp`; `Sources/ModelInputPreparation/ModelInputPreparation.hpp:14`–`:46`; Phase21B; structural boundary test passed |
| Standalone ANALYZE routing and observer | Complete operational boundaries | `LSTM/AnalyzeWorkerMain.cpp:3`; `LSTM/ObserverMain.cpp:5`; `Sources/SchedulerCore/ExperimentScheduler.cpp:12395`; routing guard passed; build minimality is a separate issue |
| Phase23 status correction and ordinary canonical publisher | Closed; do not repeat deployment | Phase23C report `:12`; Phase23E report `:12`, `:21`–`:25`; ordinary publisher's three offline contract tests passed now |
| Dedicated managed TRAIN process/application | Complete | `LSTM/TrainWorkerMain.cpp:87`; `Sources/TrainingWorkerApplication.cpp:997`–`:1021`; distinct 25-TU source phase `project.pbxproj:2227`; PR #1 tree verified |
| Entire legacy TRAIN application removed from Release | Deliberately incomplete; removal not required | `LSTM/main.cpp:4026`, `:5251`–`:5478`, `:5532`; `Tests/DedicatedTrainingWorkerArchitectureTests.sh:66`–`:70` explicitly requires Strategy B compatibility body |
| TRAIN application is a small typed service independent of CLI/persistence | Not achieved; not a prerequisite for existing standalone managed TRAIN | `Sources/TrainingWorkerApplication.cpp:1003` parses LaunchArgs; `:827`, `:860` model linkage; `:1605` startup read commit; `:1729`–`:1816` final model save/producer binding |
| Standalone executables compile only relevant application code | Partial | Source/closure inventory above; SchedulerCore compiles its compatibility source for every consumer; ANALYZE directly compiles campaign/import/model code |
| TRAIN architecture regression guard remains usable after provenance phase | Failed | `Tests/DedicatedTrainingWorkerArchitectureTests.sh:23` requires Sources immediately after `buildPhases = (`; actual first phase is provenance, `project.pbxproj:1969` |
| Dedicated rollover proves TRAIN provenance and compiled semantic contract | Implemented | `LSTM/TrainWorkerMain.cpp:60`–`:76`; `Scripts/RollSemanticWorkerLayout.py:223`–`:232`; 12 dedicated tests passed |
| Paired publication/refresh proves both executable semantic contracts | Incomplete | INFER identity lacks layout/width (`LSTM/InferWorkerMain.cpp:60`–`:72`); validator checks only role/commit/hash (`Scripts/PublishSemanticWorker.py:211`–`:229`); refresh also omits TRAIN semantic verification (`Scripts/RefreshSemanticWorkerGeneration.py:89`–`:97`) |
| Dedicated operation documentation matches merged CLI | Incomplete | `docs/semantic-workers/semantic-layout-inference-worker-routing.md:229`–`:246` describes legacy TRAIN basename, same source commit and broad capabilities only; dedicated flags are implemented at `Scripts/RollSemanticWorkerLayout.py:334`–`:337` |

TRAIN is therefore standalone for scheduler-managed work, not a replacement for every historical/ad hoc training mode. Its admission requires an experiment and attempt ID, rejects checkpoint-inference binding and inference mode, and returns 125 on registration failure before work. It uses the accepted shared runtime foundation and commits startup reads before the long epoch loop. Final save and producer binding remain in one transaction. There is no managed inference runtime/application in its source phase.

Release still owns diagnostic/research/import/provenance/admin dispatch (`LSTM/main.cpp:4032`–`:4048`), direct selected-model inference, infer-all (`:3770`), inference presentation, and legacy training/checkpoint/model-save behavior. Managed INFER already delegates. Diagnostic and model-link helpers are duplicated: classification proof (`main.cpp:246`, TRAIN `:173`), tensor diagnostics (`main.cpp:403`, TRAIN `:330`), experiment linkage (`main.cpp:1569`, TRAIN `:827`) and parent linkage (`main.cpp:1602`, TRAIN `:860`). This is maintenance coupling; it is not authority to combine the two application paths or change their transaction boundaries.

## 4. Provenance, semantics, packaging and compatibility findings

**Source identity is not tree identity.** The original real TRAIN executable built from `e67a5ce94ebe2ef0163867cb090a7b5b1cba54f8` must retain that commit in all provenance, manifests and attempt records. Equal trees do not make it a `d9dd89fc` or `ac12bfc` build. A newly built future executable has its actual build commit and a new hash. No real artifact was inspected here, so its existence, digest or deployment state is not certified by this report.

The clean-tree generator binds Release builds to actual HEAD (`Scripts/GenerateBuildProvenance.py:40`–`:49`). Dedicated rollover intentionally permits explicit TRAIN and independent INFER source commits (`Scripts/RollSemanticWorkerLayout.py:301`–`:324`) while checking their embedded/runtime identities. Keep that supported reuse of retained artifacts; do not force retained artifacts to report the workflow checkout's commit. The untracked report will itself prevent clean-tree Release provenance in this checkout; do not bypass the gate.

**Verified semantic gap:** new dedicated rollover checks TRAIN's reported role/layout/width but INFER only reports role/source/hash. An offline mocked identity with matching role/source/hash and extra `semantic_layout=12,model_input_width=127` was accepted by `verify_inference_build_identity`. This proves the helper does not validate those fields; it does not prove a real mismatched pair has been published. Compiled baseline contract is layout 13 / width 171 (`Headers/ModelInputExpansion.hpp:23`; `Headers/ModelInputContract.hpp:88`). Test fixtures using layout 14 do not advance the source contract.

Same-layout refresh derives manifest layout/width from its controlling checkout, allows independent source identities, and verifies executable identity without checking even the already-available TRAIN compiled layout/width (`Scripts/RefreshSemanticWorkerGeneration.py:89`–`:97`, `:181`–`:188`). Equal metallib hashes prove resource equality, not feature interpretation or input width. These checks should be hardened for new dedicated publication operations. Existing historical artifacts and schema-v1 readers must retain their accepted contracts; absence of new fields in an already registered immutable artifact is not grounds to rewrite or invalidate it.

**Existing safety mechanisms to preserve:** exact source string and role/self-hash checks; basename validation; narrow dedicated TRAIN capabilities; atomic complete-registry replacement under publication lock; immutable conflict rejection; separation of registry commit from convenience-link failure. Evidence: `Scripts/PublishSemanticWorker.py:171`, `:196`, `:211`, `:329`; `Scripts/RollSemanticWorkerLayout.py:42`, `:242`, `:253`–`:297`. Dedicated tests cover previous-generation retention and pre/post-commit failure handling. Keep `check_embedded_commit=False` confined to disposable fixture calls; it is not an exposed production CLI option.

**Packaging limit:** the immutable runtime manifest covers exactly two resources, `MetaNN_metal.metallib` → `MetaNN.metallib` and `default.metallib` (`Scripts/PublishSemanticWorker.py:30`, `:501`). It hashes resources, verifies shared TRAIN/INFER resources and installs links (`Scripts/RollSemanticWorkerLayout.py:77`; `Scripts/PublishSemanticWorker.py:679`). That is an implemented Metal packaging contract, not complete machine portability. pq/pqxx and OpenMP rely on external link inputs/Homebrew paths (`project.pbxproj:3865`–`:3871`). No dynamic-loader inspection or Metal resolution test was performed; do not assert missing dylibs, bad shader lookup or GPU parity without a fresh isolated artifact. Qualification should record `otool -L`, applicable architectures, external runtime versions, and run the existing Metal resolution fixture on development resources only.

**Compatibility obligations:** preserve schema-v1 `LSTM_Release` artifacts and role-aware `lstm-train-worker`/`lstm-infer-worker` paths (`Sources/SchedulerCore/SemanticWorkerRegistry.cpp:596`–`:617`), registry schema upgrades (`Scripts/PublishSemanticWorker.py:300`–`:325`), exact persisted role/layout/width/capability selection (`SemanticWorkerRegistry.cpp:983`–`:1045`), fresh legacy NULL/NULL identity admission (`:1104`–`:1139`), ablation-qualified control/treatment selection (`:1141`–`:1182`), and selection priorities. Scheduler attempts must continue to copy the selected artifact's actual identity and launch its reserved canonical executable; `Tests/SchedulerTrainingWorkerRoutingTests.sh` guards both. Do not replace historical candidates with “latest” code or change experiment rows to fit new workers. ANALYZE remains scheduler-routed, outside semantic-worker selection.

## 5. Ranked remaining tasks and justified milestones

### Rank 1 / Phase 24B — repair the TRAIN architecture guard

Motivation: the current guard returns 1 before testing its substantive application assertions. This is a concrete regression after provenance insertion, not evidence the dedicated application failed extraction.

Scope: `Tests/DedicatedTrainingWorkerArchitectureTests.sh` only. Resolve the TRAIN target's own build-phase references, allow provenance ahead of Sources, and assert provenance precedes that target's Sources phase. Assert exactly its own root/application memberships and absence of legacy main/inference application; retain every existing policy/compatibility check. Avoid an unbounded regex that could accidentally match another target.

Prerequisites: none beyond this clean baseline; no MetaNN, DB, executable, or project modification needed. Acceptance: repaired script returns 0; controlled temporary malformed project fixtures demonstrate rejection when TRAIN loses provenance, puts it after Sources, references legacy main, or borrows another target's phase. Run ReleaseWorkerBuildConfiguration and dedicated rollover suite after the focused guard. Build fan-out effect: zero. Operational risk: negligible, test only.

### Rank 2 / Phase 24C — qualify semantic contracts for new dedicated pairs

Motivation: the demonstrated INFER check gap and missing TRAIN check in refresh allow a manifest contract to be asserted without proving it matches both compiled candidates.

Exact scope: `LSTM/InferWorkerMain.cpp`, shared identity validation in `Scripts/PublishSemanticWorker.py`, `Scripts/RollSemanticWorkerLayout.py`, `Scripts/RefreshSemanticWorkerGeneration.py`, focused publisher/rollover/refresh tests, and `docs/semantic-workers/semantic-layout-inference-worker-routing.md`. Append compiled layout/width to the INFER identity output; require both candidates' compiled contracts to agree with the requested/source-derived pair in new dedicated rollover and refresh. Reuse the existing TRAIN semantic check. Validate identity record version/shape consistently where new checks rely on it. Preserve distinct commits and all legacy import/registry paths.

Prerequisites: define how a deliberately retained older INFER candidate lacking these fields is qualified before new publication. Prefer a fresh development candidate with the expanded identity; any retained-candidate exception needs explicit verifiable provenance/contract evidence, not inferred equality from metallib hashes or a renamed commit. Existing registry loads must remain unchanged. Phase24B repaired guard is the qualification prerequisite.

Acceptance: offline tests reject wrong/missing layout and width for each role, stale independent INFER identity, and incorrect TRAIN identity during refresh, before registry/staging changes; matching distinct commits succeed; source relabeling fails; failure injection retains prior authority. Re-run dedicated rollover (12), semantic publisher (12), rollover (13), refresh (13), and historical candidate/import compatibility suites. Fresh identity binaries are checked only after independent build isolation is ready. Fan-out: one INFER entrypoint recompilation per affected configuration/architecture; publication changes affect no numerical/source-phase graph. Operational risk: low during development, material at eventual publication; there is no deployment in this milestone.

### Rank 3 / Phase 24D — independent development build and packaging qualification

Motivation: the present MetaNN setup is not a usable isolated build input. Current audit establishes graph membership, not fresh binary qualification.

Scope: a separate development checkout/worktree and its own pinned MetaNN checkout at gitlink `a270e7a5dd239b524fd7d34ad3bb73b646dd7fd6`, new DerivedData, retained development outputs, and qualification evidence. Do not repair the symlink by writing into or building through production. No application redesign, schema change or publication needed.

Prerequisites: all transitive project/source references resolve within development roots; no shared writable products; inspect executable/build scripts and active-worker status before any worker CLI tests. Commit state must be clean for Release provenance; do not weaken provenance to accommodate this report or pending changes.

Acceptance: independent target builds for TRAIN, INFER, scheduler, ANALYZE and observer; ordinary Release build with publication disabled; capture per-architecture link maps and compile commands; inspect real embedded source/hash/layout/width, `otool -L`, and the two runtime resources. Repeat a no-change build and measure CompileC/Libtool/Ld; account for always-run provenance generation. Existing standalone CLI tests may be used only after verifying their rejection paths cannot register work. Database parity/resume/checkpoint tests require explicitly isolated test DBs and no route to production. Metal fixture uses new development metallibs. Fan-out reduction: zero; this provides the evidence needed for reductions. Operational risk: low with independent inputs; unacceptable through the current production-pointing setup.

### Rank 4 / Phase 24E — one proven build-edge reduction, conditional

Motivation: TRAIN's framework/dependency phase reuses Release's library set; its sources have no direct StrategyEvaluationCore/InferenceProfitability calls in inspected paths. A fresh TRAIN build currently reaches StrategyEvaluationCore's 7 TUs and ProfitabilityCore's 1. This makes those edges candidates, not proven removable dependencies.

Exact initial scope: only TRAIN's `StrategyEvaluationCore` and `ProfitabilityCore` target dependencies/link inputs in `ExpertAdvisor.xcodeproj/project.pbxproj`, and focused build-boundary tests. First inspect TRAIN link maps for extracted objects and symbols; remove only an edge proven unused, with no new stub or relocated application policy. If actual symbol closure requires them, stop and mark the reduction unnecessary. Leave Release→ANALYZE packaging intact.

Prerequisites: Phase24D independently built baseline and compile/link evidence, including Debug and Release differences. Acceptance: focused TRAIN builds/link maps unchanged for required symbols; guard and worker routing/feature-ablation tests pass; isolated fresh/resume/checkpoint/final-save behavior is unchanged. Expected static closure improvement if both libraries prove unneeded: 62 → 54 project TU memberships for a fresh standalone TRAIN build, excluding MetaNN. No claim of elapsed savings, and no reduction in shared-library work when building all other consumers. Operational risk: low development-only project change, medium if a symbol/runtime dependency was misclassified.

### Rank 5 — optional later compile deduplication; no committed milestone yet

The 135 common ANALYZE/Release source memberships and five LSTM.cpp memberships justify measuring duplicate compile cost. Candidate scope is build-only reuse of existing implementation units with identical compilation contracts, not reopening accepted analyzer/scheduler application extraction. Release/ANALYZE and TRAIN/INFER Debug settings differ in OpenMP and other target configuration (`project.pbxproj:3773`, `:3817`, `:3838`), so a single library cannot be assumed interchangeable.

Prerequisite: actual compile-command comparison and link-map/fixture parity from Phase24D. Acceptance before proposing an implementation: enumerate a concrete compatible subset, its new owners, unchanged macro/ABI/runtime behavior, measured compile reduction, and regression set. The theoretical upper bound for sharing the entire common ANALYZE/Release set is 135 avoided duplicate compilations per architecture; it is not a promised or presently safe reduction. Operational risk rises sharply if numerical/model or campaign workflow ownership moves. Defer until evidence supports one small subset.

### Work explicitly not recommended

- No new generic TrainingEngine/ModelCore/runtime-context framework to make main shorter. Keep numerical lifecycle and transaction policy intact.
- No wholesale deletion or delegation of Release's legacy TRAIN implementation to the managed-only application: ad hoc and historical compatibility differ, and existing tests require the retained body.
- No reimplementation of scheduler ownership, priority, retries, persisted admission, or the accepted detached-INFER boundaries.
- No analyzer extraction or observer-core split based only on archive source counts. SchedulerCore's archive compilation is measurable overhead; prior Phase20T/20U link closure is evidence that unrelated object extraction can already be avoided. Obtain current maps before changing its target ownership.
- No automatic new semantic layout. Baseline remains layout 13 / width 171. Code-only replacement belongs to the separate generation-refresh workflow; layout rollover requires a real semantic change.
- No deletion/replacement of historical artifacts, mutation of persisted experiment identity, or rebuild of frozen production TRAIN source.

## 6. Development gates versus eventual deployment

Development may proceed while production TRAIN runs if it has independent source/submodule trees, DerivedData, temporary CPU fixtures, artifact roots and explicitly isolated test databases. Gates for every increment: no production-path traversal by build inputs; no production executable launch; no registry/current-link operation; no signals to production processes; no real experiment mutation; preserved exact source identity; focused tests before broader regression; compiler warnings/errors resolved for any implemented change.

During this audit the production registry was not accessed, so “no registry change” is established by the executed operations' scope, not a before/after production hash comparison. For future build qualification use development artifact directories only and do not run either publication aggregate or publisher CLI with default real artifact roots. The ordinary Release build command remains:

```bash
set +e
xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Release" -configuration Release -derivedDataPath DerivedData/ExpertAdvisor build
```

That is a future development command, **not executed here**, and it requires a fully isolated, clean checkout first. Do not run Clean.

Production deployment is a distinct later assignment: choose qualified immutable candidates with their actual commits/hashes; establish their compiled semantic and runtime compatibility; review historical route/attempt obligations; then use the authoritative publication/refresh/rollover workflow under explicit deployment authority. Development completion does not require production cutover. No scheduler restart, registry change, or replacement of active worker files is part of Phases24B–24E as proposed.

## 7. Concrete next-session increment

Implement Phase24B alone: repair `Tests/DedicatedTrainingWorkerArchitectureTests.sh:23` to verify the actual target-owned source/provenance phase relationship, preserving all substantive checks, and add temporary malformed-project checks if needed to show the repaired guard cannot match another target. Change no runtime source, project settings, DB schema, registry or artifact. Run the repaired guard, `ReleaseWorkerBuildConfigurationTests.sh`, and the 12-test dedicated rollover suite. Document the result and stop before Phase24C. This is useful and testable even with the current MetaNN build obstruction.

## 8. Commands, results, and exact changes

Material setup/verification commands executed from the isolated root:

```text
pwd
git status --short
git branch --show-current
git rev-parse HEAD
git log -5 --oneline --decorate
git branch -vv
git remote -v
git worktree list
git rev-parse origin/lstm-feature-development
git show --no-patch --format=fuller ac12bfcdbdc989c8926d60d366078293960d4cd4
git fetch origin lstm-feature-development
git show --no-patch --format=fuller origin/lstm-feature-development
git diff --stat HEAD origin/lstm-feature-development
git diff --name-status HEAD origin/lstm-feature-development
git diff --quiet HEAD origin/lstm-feature-development
git rev-parse HEAD^{tree} origin/lstm-feature-development^{tree}
git cat-file -t e67a5ce94ebe2ef0163867cb090a7b5b1cba54f8
git show --no-patch --format=fuller e67a5ce94ebe2ef0163867cb090a7b5b1cba54f8
git diff --stat e67a5ce94ebe2ef0163867cb090a7b5b1cba54f8 d9dd89fc29abd05bffa843bb37f182ceafdde65c
git submodule status
git show HEAD:.gitmodules
ls -la MetaNN
ps -axo pid,comm | rg '(LSTM_Release|lstm-scheduler|lstm-train-worker|lstm-infer-worker|lstm-analyze-worker|lstm-observer)$'
plutil -lint ExpertAdvisor.xcodeproj/project.pbxproj
git diff --check
git diff --stat
```

The initial `git show` for the merge failed because the object was absent; fetch resolved that. `ps` was denied. Exploration also used `rg --files`, focused `rg -n` searches, `wc -l`, and `cat`/`nl -ba` with `sed -n` ranges on the evidence files identified throughout this report and Phase20/21/22/23 reports. Missing local MetaNN subproject/library probes confirmed the isolation prerequisite, not a compiler failure. Read-only inline Python parsed `plutil` output to count source/target closure and inspect compile settings. Another inline Python check mocked `subprocess.run` for `verify_inference_build_identity`; no real executable/path was opened by that probe.

Tests actually run (Python suites used `PYTHONDONTWRITEBYTECODE=1 TMPDIR=/private/tmp`; temporary C++ fixtures used no production application):

| Command | Result |
| --- | --- |
| `python3 -m unittest discover -s Scripts/tests -p 'test_dedicated_train_rollover.py' -v` | PASS, 12 tests |
| `bash Tests/DedicatedTrainingWorkerArchitectureTests.sh` | FAIL, exit 1 at line 23 |
| `bash -x Tests/DedicatedTrainingWorkerArchitectureTests.sh` | Same failure localized to phase-order regex; not a full suite pass |
| `bash Tests/ReleaseWorkerBuildConfigurationTests.sh` | PASS |
| `bash Tests/MarketDataCoreBoundaryTests.sh` | PASS, exit 0 (silent script) |
| `bash Tests/RuntimeFoundationStructuralTests.sh` | PASS |
| `bash Tests/ModelRuntimeValidationStructuralTests.sh` | PASS |
| `bash Tests/PersistedModelRuntimeConfigStructuralTests.sh` | PASS |
| `bash Tests/SchedulerAnalyzeWorkerRoutingTests.sh` | PASS |
| `bash Tests/SchedulerTrainingWorkerRoutingTests.sh` | PASS |
| `bash Tests/LSTMPhase22Z3InferenceRuntimeCompositionBoundaryTests.sh` | PASS |
| `bash Tests/LSTMPhase22Z4ManagedInferenceApplicationBoundaryTests.sh` | PASS |
| `python3 Tests/SemanticWorkerPublisherTests.py` | PASS, 12 tests |
| `python3 Tests/SemanticWorkerRolloverTests.py` | PASS, 13 tests, including CPU C++ registry fixture compiled with `clang++ -std=c++20 -Wall -Wextra -Werror` |
| `python3 Tests/SemanticWorkerGenerationRefreshTests.py` | PASS, 13 tests |
| `python3 Tests/CanonicalLSTMReleasePublisherTests.py` | PASS, 3 tests; tiny C fixtures and temporary Git repositories only |
| `TMPDIR=/private/tmp bash Tests/SchedulerCoreBoundaryTests.sh` | PASS; small C++ engine/policy fixture compiled with `clang++ -std=c++20 -Wall -Wextra -Werror` |

The nine source-only passing scripts were also rerun through a small subprocess harness to capture each exit code independently; all returned 0. Total Python tests: 53 passed. `SemanticWorkerRolloverTests.py` compiler discovery emitted filesystem-event/user-cache diagnostics from Xcode tooling; it ran no project build and finished successfully. No application compiler-warning claim is made because no application was compiled.

No Xcode project build, Clean, standalone worker CLI suite, GPU/Metal test, DB integration, model numerical parity test or real artifact publication was run. Historical report claims of passing Release builds are not fresh results. The architecture guard failure is deliberately left for the next implementation assignment because this assignment prohibits test/source edits.

Exact files changed: this report only, `docs/phases/Phase24/LSTM_Phase24A_FinalRefactoringGapAnalysis_Output.md`. Behavioral change: none. No source/project/test/schema modification, commit, merge or push.

Final `git status --short`:

```text
?? docs/phases/Phase24/
```

Final `git diff --stat`: empty; Git's ordinary diff does not include the untracked report. `git diff --check` passes for tracked changes; the new report was separately checked for trailing whitespace and conflict markers.

Remaining limitations: production processes/status and real artifact identities were not measured; dependency subproject is not available as an isolated checkout; current binary/link maps, build timings, external-loader dependencies, Metal behavior and database parity remain unqualified. These limits constrain build/deployment recommendations, not the verified source boundaries or offline findings.
