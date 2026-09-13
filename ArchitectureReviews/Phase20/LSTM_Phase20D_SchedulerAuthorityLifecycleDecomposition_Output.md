# LSTM Phase 20D — Scheduler Authority and Lifecycle Decomposition

## Status

Phase 20D is complete.

Commit:

`af62890 Extract scheduler authority service`

The phase extracted scheduler authority/ownership/lease/fencing responsibilities into a dedicated authority service and repository seam while preserving the existing scheduler executable architecture and behavior.

## Implemented structure

New files:

- `Sources/SchedulerCore/SchedulerAuthorityRepository.hpp`
- `Sources/SchedulerCore/SchedulerAuthorityService.cpp`
- `Sources/SchedulerCore/SchedulerAuthorityService.hpp`
- `Tests/SchedulerAuthorityServiceTests.cpp`
- `Tests/SchedulerAuthorityServiceTests.sh`

Modified files include the scheduler monolith, PostgreSQL repository implementation/seams, ownership-policy compatibility header, repository tests, and the Xcode project.

`SchedulerOwnershipPolicy.hpp` now acts primarily as a compatibility-facing layer while authority/takeover logic resides in SchedulerCore.

## Scope and behavior

The extraction covers scheduler authority responsibilities including:

- scheduler invocation authority
- ownership validation
- lease acquisition/renewal/release
- fencing-token validation
- protocol generation/cutover interaction
- stale-owner/displacement protection
- takeover decision behavior
- authority-loss diagnostics and failure behavior

No standalone scheduler executable was introduced in Phase 20D.

No database schema changes were introduced.

No intended changes were made to training, inference, strategy evaluation, model behavior, scheduler CLI semantics, worker process signaling, or continuation semantics.

## Validation

Focused tests passed:

- `SchedulerAuthorityServiceTests`
- `SchedulerCoreBoundaryTests`
- `SchedulerInternalSeamTests`
- `SchedulerOrchestrationServiceTests`
- `PostgresSchedulerRepositoryTests`
- `SchedulerSemanticAdmissionTests`
- `SchedulerOwnershipPolicyTests`
- `SchedulerChildStatusTests`

`git diff --check` passed before commit.

Required Release builds all passed with return code 0:

- `SchedulerCore`
- `StrategyEvaluationCore`
- `LSTM Release`

The initial SchedulerCore Release build completed successfully with existing libpqxx deprecation warnings only.

## Incremental-build result

Using the project-standard stable DerivedData path:

`DerivedData/Release`

and repeating the exact same `LSTM Release` build command produced:

- `CompileC = 1`
- `Libtool = 0`
- `Ld = 1`

The only recompiled source was:

`Sources/CampaignOperationsProductionAdmissionService.cpp`

This matches the previously observed build-provenance dependency behavior. The static libraries were not rebuilt, confirming that the stable DerivedData policy is preserving incremental compilation across the modularized targets.

The project should continue to avoid varying `CONFIGURATION_BUILD_DIR` merely to retain/version builds. Completed products should instead be copied from the normal Release product directory into a retained `Builds/<version>/` directory after a successful build.

## Monolith size

Before Phase 20D:

`Sources/SchedulerCore/ExperimentScheduler.cpp` = 29,133 lines

After Phase 20D:

`Sources/SchedulerCore/ExperimentScheduler.cpp` = 28,837 lines

The line-count reduction is modest because Phase 20D primarily moved a coherent authority responsibility while preserving compatibility and orchestration behavior. More importantly, the authority boundary is now explicit and testable.

## Phase 20E recommendation

Do not split out a standalone `lstm-scheduler` executable yet.

`ExperimentScheduler.cpp` remains approximately 28.8k lines, so the scheduler monolith still contains substantial internal responsibility beyond composition and compatibility glue. Phase 20E should therefore continue internal SchedulerCore decomposition before introducing a separate scheduler executable.

The next phase should inventory the remaining responsibilities in `ExperimentScheduler.cpp` and extract the next largest cohesive service boundary. Good candidates are scheduler lifecycle/invocation management, continuation/checkpoint orchestration, or worker-attempt lifecycle orchestration, depending on the current ownership map after Phase 20D.

A dedicated scheduler executable should become the next step only when `ExperimentScheduler.cpp` is predominantly composition, command dispatch, and compatibility glue rather than domain logic.

## Conclusion

Phase 20D successfully established a tested scheduler-authority boundary, preserved Release-build compatibility, and retained effective incremental compilation under the stable DerivedData build policy.

Recommended next step: Phase 20E internal SchedulerCore decomposition, not executable separation yet.
