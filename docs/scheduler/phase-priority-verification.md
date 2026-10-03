# Phase priority implementation verification

Base: `lstm-feature-development`, commit `3a6c97b99794b228a6afe577c0b0d7311c0e4131`.

Implemented persisted phase ordering, live set/show controls, admission gating
for final and checkpoint jobs, draining on phase changes, and concurrent-mode
compatibility. Job ordering and job-priority preemption code are unchanged.

Changed files:

- `Database/README.md` and migration `099_scheduler_phase_priority.sql`.
- `Sources/SchedulerCore/SchedulerPhasePriority.hpp` (pure policy and admission).
- `Sources/SchedulerCore/SchedulerPhasePriorityService.hpp` (service/repository seam).
- `Sources/SchedulerCore/SchedulerPhasePriorityRepository.hpp` (PostgreSQL adapter).
- `SchedulerDaemonConfiguration.hpp/.cpp`, `SchedulerCycleService.hpp/.cpp`,
  `SchedulerRuntimeContext.hpp`, `ProductionSchedulerRuntimeInternal.hpp`, and
  `ProductionSchedulerDaemon.cpp` under `Sources/SchedulerCore`.
- Configuration/cycle tests; new portable and PostgreSQL phase-policy tests.
- `docs/scheduler/phase-priority.md` and this verification record.

## Checks run in the Linux workspace

Passed:

1. `CXX=g++ bash Tests/SchedulerPhasePriorityTests.sh`.
2. Cycle tests built and run with:
   `g++ -std=c++20 -Wall -Wextra -Werror -IHeaders -ISources Sources/SchedulerCore/SchedulerCycleService.cpp Tests/SchedulerCycleServiceTests.cpp -o /tmp/SchedulerCycleServiceTests`.
3. Final-dispatch regression tests built and run with:
   `g++ -std=c++20 -Wall -Wextra -Werror -IHeaders -ISources Sources/SchedulerCore/FinalExperimentDispatchService.cpp Tests/FinalExperimentDispatchServiceTests.cpp -o /tmp/FinalExperimentDispatchServiceTests`.
4. Configuration tests compiled with GCC, linking OpenSSL through a temporary
   CommonCrypto SHA256 compatibility header, then executed successfully.
   Flags: `-std=c++20 -Wall -Wextra -Werror -Wno-dangling-reference -IHeaders
   -ISources -I/tmp/phase-shims`, sources `Tests/SchedulerDaemonConfigurationTests.cpp`,
   `Sources/SchedulerCore/SchedulerDaemonConfiguration.cpp`, and
   `Sources/SchedulerCore/SemanticWorkerRegistry.cpp`; linked with `-lcrypto`.
   The dangling-reference suppression addresses existing GCC diagnostics in
   the unmodified registry implementation.
5. New PostgreSQL tests compiled with libpqxx 7.10.3 and executed against a
   disposable PostgreSQL 16.15 cluster/database. Both `unmigrated` and `migrated`
   test modes passed. Migration applied and replayed successfully; runtime role
   SELECT/UPDATE privileges checked. Final policy remained `infer:analyze:train`,
   revision 2 after replay. No production database was accessed.
6. `ProductionSchedulerDaemon.cpp` and legacy `ExperimentScheduler.cpp` passed
   GCC syntax checks with temporary CommonCrypto and mach-o declaration headers,
   MetaNN headers, and libpqxx 7.10.3. Flags included `-std=c++20 -Wall -Wextra
   -Werror -fsyntax-only`, with existing platform/compiler warning classes
   suppressed: deprecated declarations, dangling references, unused functions,
   unknown pragmas, unused parameters, missing field initializers, sequence-point
   warnings, and (legacy file only) sign comparison. None of the temporary
   compatibility headers or downloaded dependencies is included in this change.
7. `git diff --check` passed.

The checked-in shell tests retain the project's native macOS/clang workflow.
The new portable/PostgreSQL scripts also support `CXX`.

## Remaining verification

The macOS Xcode Release build and real scheduler/worker process integration
were not run: this workspace has no access to the user's Mac Studio, production
PostgreSQL instance, canonical binaries, or active workers. On the Mac, run the
Release build and relevant tests before replacing the running scheduler. Apply
migration 099 through the normal migration runner. The first upgrade requires a
scheduler restart; subsequent phase-order changes do not.

Worker limits are still startup options. A scheduler with an inference/analyze
limit of zero needs one restart with those classes enabled. Continuous demand
in an earlier phase may delay later phases, as documented. Global cancellation
keeps its existing exceptional checkpoint execution path.
