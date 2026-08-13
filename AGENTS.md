# ExpertAdvisor Development Instructions

## Project

- C++20 project built with Xcode 26.5 on macOS.
- Primary executable: LSTM_Release.
- Avoid Swift.
- Prefer existing C++ and Objective-C++ test patterns.
- PostgreSQL access uses libpqxx.
- Treat the database schema and persisted workflows as authoritative contracts.

## Build

From the repository root:

xcodebuild   -project ExpertAdvisor.xcodeproj   -scheme "LSTM Release"   -configuration Release   -derivedDataPath DerivedData/ExpertAdvisor   build

Do not run Clean unless explicitly requested.

## Tests

Run the smallest relevant tests first, then broader regression tests.

Do not launch tests or executables that may interfere with active scheduler,
training, inference, or analysis workers without checking first.

## Scheduler Safety

Before running LSTM_Release commands that could start work:

1. Inspect running LSTM_Release processes.
2. Check scheduler status.
3. Avoid interrupting active experiments unless explicitly requested.
4. Never alter production experiment rows merely to make tests pass.

## Architecture

Preserve the established layering:

CLI -> Service / Workflow -> Repository -> PostgreSQL

Do not bypass authoritative workflows with direct repository or SQL writes.

Preserve:

- deterministic identities
- transactional boundaries
- concurrency safety
- immutable persisted decisions and materializations
- explicit phase boundaries
- idempotent retry behavior

## Development Style

- Make the smallest change that satisfies the requested increment.
- Do not redesign adjacent phases.
- Do not implement future roadmap items.
- Prefer reuse of existing authoritative workflows.
- Add tests for concrete behavior and regressions.
- Update documentation when the public or operational contract changes.
- Treat compiler warnings as defects unless clearly justified.
- Compiler and tests are authoritative.
- Resolve compiler warnings, build errors, and failing tests before considering an implementation complete.

## Review Output

At completion, report:

1. Files changed.
2. Exact behavioral change.
3. Tests and build commands run.
4. Results.
5. Remaining risks or unverified assumptions.
6. `git status --short`
7. `git diff --stat`

## Current Development Status

Current branch:
- phase6

Completed major phases:
- Phase 1: Checkpoint inference
- Phase 2: Evaluation persistence
- Phase 3: Continuation framework
- Phase 4: Recommendation campaigns
- Phase 5: Campaign execution
- Phase 6A: ...
- Phase 6B: ...
- Phase 6C: ...

Always preserve compatibility with completed phases.
Do not redesign previously accepted architecture.

## Never

- Never run `xcodebuild clean` unless explicitly requested.
- Never modify production experiment data to simplify testing.
- Never bypass authoritative workflows.
- Never redesign completed phases during an incremental implementation.

