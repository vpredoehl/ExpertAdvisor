---
title: "Campaign Operations Phase H H1 Protected Function Preflight Targeted Correction"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_PhaseH_H1_ProtectedFunctionPreflight_TargetedCorrection_Prompt.txt"
reasoning_effort: "high"
model: "default"
---

# Campaign Operations Phase H H1 Protected Function Preflight Targeted Correction

## 1. Executive summary

Migration 055 now classifies the complete protected-function contract before any DDL or ACL normalization. Incompatible return metadata, object kind, leakproof state, ownership, and ACL state fail with frozen H1A diagnostics instead of reaching PostgreSQL DDL errors or being silently repaired.

The supported schema-054 upgrade, migration replay, deployment audit, manifest validation, and relevant H1 regression suite pass.

No H1/H2/H3/H4 boundaries or unrelated scheduler, recovery, identity, lock-order, or deployment semantics were changed.

## 2. Exact root cause corrected

The original preflight validated only part of the protected catalog tuple. Return metadata, `prokind`, `proleakproof`, and expanded ACL semantics were either absent or checked only after mutation.

Consequently:

- incompatible return types could reach `CREATE OR REPLACE`;
- incompatible ACLs could be normalized by `REVOKE`/`GRANT`;
- procedures or aggregates could evade complete preflight classification;
- post-DDL validation was acting as the first authoritative check for some frozen tuple components.

Migration 055 now completes those checks before mutation begins.

## 3. Files changed

Correction-specific files:

- [055_campaign_operations_production_admission_foundation.sql](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql>)
  - Complete protected-function preflight.
  - Exact ACL expansion and comparison.
  - Return, OUT/TABLE, object-kind, and leakproof validation.
  - Corresponding post-DDL audit consistency checks.

- [CampaignOperationsProductionAdmission.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsProductionAdmission.hpp>)
  - Updated embedded migration SHA-256.

- [CampaignOperationsPhaseH1ProtectedFunctionPreflightTests.sh](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1ProtectedFunctionPreflightTests.sh>)
  - Added authentic disposable-cluster mutation and rollback tests.

## 4. Protected-function tuple comparison

| Tuple component | Before | After |
|---|---|---|
| Schema/name/signature | Preflighted | Preserved; identity arguments now also compared literally |
| Protected signature uniqueness | Preflighted | Preserved |
| Owner | Allowed phase checked | Preserved; failure now identifies exact signature |
| `prokind` | Incomplete/late | Exact `f` required before mutation |
| Language | Preflighted | Preserved |
| SECURITY DEFINER | Preflighted | Preserved |
| Volatility | Preflighted | Preserved |
| Parallel mode | Preflighted | Preserved |
| `proleakproof` | Incomplete/late | Exact `false` required before mutation |
| Defaults/variadic | Preflighted | Preserved |
| `proconfig`/search path | Preflighted | Preserved, including accepted schema-054 compatibility |
| Return type | Not completely preflighted | Exact namespace/type and set-returning state checked |
| OUT/INOUT/TABLE metadata | Not completely preflighted | Exact `proallargtypes`, modes, and output names checked |
| ACL origin | Not authoritative pre-DDL | Exact accepted NULL/explicit origin checked |
| Expanded ACL | Not authoritative pre-DDL | Bidirectional exact set comparison |
| PUBLIC EXECUTE | Could be normalized | Explicitly classified and rejected when unsupported |
| Grantees/privilege/grant option | Could be normalized | All entries compared before mutation |
| Post-DDL audit | Defense-in-depth | Retained and strengthened with the same literal contract |

## 5. Pre-mutation validation order

The effective protected-object order is now:

1. Validate migration prerequisites and persisted compatibility state.
2. Resolve protected role identities and validate protected object inventory.
3. Search all schemas for protected names and identities.
4. Enforce exact schema, `proname`, signature uniqueness, and overload/default/variadic exclusions.
5. Validate the accepted current or schema-054 legacy owner phase.
6. Validate `prokind`, language, SECURITY DEFINER, volatility, parallel mode, leakproof state, defaults, variadic type, and `proconfig`.
7. Validate literal `pg_get_function_identity_arguments`.
8. Validate return namespace/type, `proretset`, `proallargtypes`, argument modes, and OUT/TABLE names.
9. Expand NULL ACLs using PostgreSQL `acldefault('f', owner)` semantics.
10. Compare ACL origin and the complete expanded ACL set bidirectionally, including PUBLIC, every grantee, privilege type, and grant option.
11. Complete wrapper/dependency and remaining role/default-ACL preflights.
12. Only after successful classification can role creation, `CREATE OR REPLACE`, ownership changes, `REVOKE`, `GRANT`, or search-path normalization begin.

## 6. Stable H1A diagnostic mapping

| Negative branch | SQLSTATE | H1A code |
|---|---:|---|
| Incompatible return type | `55000` | `H1A008` |
| Procedure/object-kind mismatch | `55000` | `H1A008` |
| Leakproof mismatch | `55000` | `H1A008` |
| OUT/INOUT/TABLE identity or mode mismatch | `55000` | `H1A008` |
| Unexpected explicit ACL grantee | `42501` | `H1A006` |
| Unexpected PUBLIC EXECUTE | `42501` | `H1A006` |
| Unexpected grant option | `42501` | `H1A006` |
| Unsupported owner phase | `42501` | `H1A004` |
| Existing signature/overload/default classification | `42501` | `H1A005` |

No new diagnostic namespace was introduced.

## 7. Test additions and non-vacuous proof

The new test creates a disposable PostgreSQL 17 cluster from the repository dump plus migrations 050–054.

It first proves that the exact supported schema-054 state upgrades successfully, replays idempotently, and passes the deployment audit. Each negative fixture then starts from that independently valid state.

Fixtures cover:

- incompatible return type;
- protected identity occupied by a procedure;
- leakproof function;
- unexpected explicit grantee;
- unexpected PUBLIC EXECUTE;
- grant option on an otherwise expected grantee;
- incompatible OUT metadata while retaining the expected return type;
- owner accepted by neither current nor legacy phase.

For every rejection, the test:

- validates the intended precondition in PostgreSQL catalogs;
- verifies the requested attribute is the relevant mismatch;
- runs migration 055 as the failing statement;
- verifies SQLSTATE, H1A code, and exact signature;
- verifies the error originates in protected-function preflight;
- rejects known raw PostgreSQL DDL mismatch errors;
- compares complete before/after catalog snapshots byte-for-byte;
- verifies no admission transition object appeared.

A static consistency assertion also compares the literal preflight and post-DDL protected-function contracts.

## 8. Commands run and results

- `Tests/CampaignOperationsPhaseH1ProtectedFunctionPreflightTests.sh`
  - Passed schema-054 upgrade, replay, audit, all eight negative branches, and all rollback/no-normalization checks.

- `Tests/CampaignOperationsPhaseH1MigrationTests.sh`
  - Passed the relevant H1 migration, audit, lock, repository/service, Phase 1–5 regression, evidence, and traceability checks.

- `Scripts/CampaignOperationsH1ManifestValidator.sh`
  - Passed:
    - digest: `ec6e34b1dd3ee68bf5315f193baee233222822eb045437d154d4b29e9444fb9f`
    - inventory rows: 92
    - explicit ACL rows: 70
    - default ACL states: 27
    - column ACL rows: 7

- Strict checksum-header compile:
  ```text
  clang++ -std=c++20 -Wall -Wextra -Werror -fsyntax-only \
    -I Sources -I Headers \
    -I /opt/homebrew/opt/libpqxx@7.10.1/include \
    -I /opt/homebrew/opt/libpq/include \
    Sources/CampaignOperationsProductionAdmission.cpp
  ```
  Passed without warnings.

- `bash -n Tests/CampaignOperationsPhaseH1ProtectedFunctionPreflightTests.sh`
  - Passed.

- Correction-scoped `git diff --cached --check`
  - Passed for all three correction files.

The full staged-index check still reports pre-existing trailing-tab warnings in `Tests/fixtures/CampaignOperationsH1Artifacts.tsv`; those TSV empty-field delimiters belong to the staged H1 baseline and were not changed.

The Release `xcodebuild` was not run. Process inspection found an active scheduler and multiple training workers using the shared `DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release`, so rebuilding could have replaced their live executable.

## 9. Migration checksum and manifest

Migration SHA-256:

```text
e696c8512f3652860dbc042bea06012e159b14cabab01952d9ae37ce7a18fdbb
```

The embedded C++ checksum matches exactly.

The ACL/object manifest content did not change, so its manifest-set digest correctly remains:

```text
ec6e34b1dd3ee68bf5315f193baee233222822eb045437d154d4b29e9444fb9f
```

No unrelated historical reports or CEE output Markdown files were staged.

## 10. Remaining risks or unverified checks

- The shared Release Xcode build remains unverified in this pass because live scheduler/training workers were using the target binary.
- Full final-assurance orchestration was not rerun because it requires an artifact root, run ID, and completed Xcode build log. The focused migration suite and broader H1 regression suite passed.
- The preflight and post-DDL audit require separate literal arrays for independent execution; the new test compares them to prevent semantic drift.

## 11. `git status --short`

The correction files are staged within the existing staged H1 candidate. The existing staged baseline contains 114 files. Correction-specific entries are:

```text
A  Database/migrations/055_campaign_operations_production_admission_foundation.sql
A  Sources/CampaignOperationsProductionAdmission.hpp
A  Tests/CampaignOperationsPhaseH1ProtectedFunctionPreflightTests.sh
```

The repository also retains the pre-existing staged H1 candidate files and untracked CEE/report outputs shown by `git status --short`; none of those output Markdown files were added by this pass.

## 12. `git diff --stat`

Unstaged diff:

```text
(no unstaged changes)
```

Complete staged H1 candidate:

```text
114 files changed, 30122 insertions(+), 20 deletions(-)
```

## 13. Final disposition

READY_FOR_FOCUSED_INDEPENDENT_REVERIFICATION