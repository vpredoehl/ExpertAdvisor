# Campaign Operations Phase H Final Integration Closure Assurance Rerun — Independent Findings

Date: 2026-08-10

## Review basis

This independent assessment reviews the output of the Phase H Final Integration & Closure Assurance rerun performed against branch `campaign-operations` at HEAD `6042a2dba69791ee13d3c62d00cbe27e54c5b785`.

The rerun concluded:

```text
PHASE_H_CLOSURE_BLOCKED
READY_FOR_PHASE_H_FINAL_TARGETED_CORRECTION
```

I agree with that disposition.

## Executive assessment

The rerun did not uncover a new cross-phase architecture failure. H1, H2, H3, and H4 continue to compose with the intended authority boundaries:

- H1 remains authoritative for production enablement/readiness.
- H2 remains authoritative for production dispatch, transaction, replay, and concurrency correctness.
- H3 remains the bounded run-once Campaign Manager.
- H4 remains external deployment-owned continuous supervision with no database or scheduler ownership expansion.

The previous final-assurance defects were substantially closed:

1. the migration-058 function-name defect was corrected in place;
2. current Phase H/H4 documentation was updated to distinguish accepted external H4 supervision from still-prohibited in-process continuous Manager operation.

The rerun identified three remaining MEDIUM closure issues and one INFORMATIONAL evidence item.

## Finding H-001 — MEDIUM — H4 persisted state/log directory trust

### Finding

H4 validates the JSON configuration and connection environment file, but existing `state_directory` and `log_directory` paths were only required to be absolute.

The implementation did not yet prove that an existing directory:

- is an actual directory rather than a symlink;
- is owned by the deployment execution identity;
- has sufficiently restrictive permissions.

### Why this matters

H4 treats persisted deployment state as authoritative during restart. In particular, a complete persisted STOP must remain a STOP and must not silently become a normal or retry action.

If another principal can modify the state directory, it can potentially replace a valid STOP record with another syntactically valid record. Schema/shape validation alone cannot detect an intentional rewrite of otherwise valid JSON.

This undermines ADR-0020's persisted-state authority and no-automatic-retry requirements.

### Smallest correction

Fail closed when an existing state/log path is:

- a symlink;
- not a directory;
- not owned by the deployment execution identity/effective deployment UID;
- group/world accessible beyond the reviewed owner-only policy.

Add focused adversarial tests for insecure directory permissions and symlink paths. Preserve the existing H4 architecture; do not introduce database state, a database singleton, signing infrastructure, lease authority, PID authority, or another control plane.

## Finding H-002 — MEDIUM — Volume XII database status is stale

### Finding

`docs/architecture/Volume_XII_Database.md` still describes Phase H database implementation as complete only through H1 / migration 055.

The repository now contains the implemented H2/H3 database chain through migrations 056, 057, and 058.

### Why this matters

Volume XII is a canonical database-architecture surface. Leaving its implementation status at migration 055 can mislead an operator or future recovery effort about the database state required for Phase H.

### Smallest correction

Update Volume XII to state that:

- H1 is represented by migration 055;
- H2 database/privilege contracts are represented by migrations 056 and 057;
- H3 database support is represented by migration 058;
- H4, under ADR-0020, adds no migration, schema, ACL, role, database scheduler, singleton, heartbeat, lease, or leader-election authority.

Update the status/version/revision history and relevant references consistently.

## Finding H-003 — MEDIUM — H3 documentation retains stale “H4 exclusion” wording

### Finding

`docs/CampaignOperationsPhaseH3.rst` still states that the H4 exclusion remains unchanged.

That wording predates accepted ADR-0020.

### Why this matters

The current architecture intentionally distinguishes two different statements:

- H3 itself remains bounded and has no continuous execution mode.
- ADR-0020 separately authorizes external deployment-owned H4 supervision that repeatedly invokes bounded H3.

The stale wording obscures that distinction.

### Smallest correction

Replace the stale H4-exclusion sentence with explicit wording that H3 itself remains bounded/non-continuous while ADR-0020 authorizes only external deployment-owned H4 supervision.

Do not change H3 runtime authority or add a continuous `LSTM_Release` command.

## Finding H-004 — INFORMATIONAL — migration-058 execution evidence must be retained

### Finding

The new executable migration-058 regression was untracked at the time of the assurance rerun, and the rerun correctly did not execute migration tests because of its non-mutating assurance constraints and active workers.

### Assessment

This is not an additional implementation defect.

The disposable regression is the correct evidence mechanism because it executes migration 058 against an isolated predecessor state and contains a negative control for the former misspelled function name.

### Required closure evidence

After the remaining corrections:

1. execute `Tests/CampaignOperationsPhaseH3Migration058ExecutionTests.sh`;
2. confirm the corrected migration installs successfully in the disposable environment;
3. retain the result in the Phase H review-artifact hierarchy;
4. include the regression in the closure commit.

The authoritative production LSTM database should not be used for this disposable verification.

## Cross-phase conclusion

The rerun continues to support the Phase H architecture:

```text
H1 authority/readiness          PRESERVED
H2 dispatch/transaction         PRESERVED
H3 bounded orchestration        PRESERVED
H4 external supervision         PRESERVED
Scheduler ownership boundary    PRESERVED
Database ownership boundary     PRESERVED
Retry/STOP semantics            PRESERVED
H3/H4 machine contract          PRESERVED
```

No new HIGH-severity issue or architectural redesign is indicated.

## Recommended correction scope

Proceed with one final narrowly scoped correction:

1. harden H4 existing state/log directory trust validation and add focused tests;
2. align `Volume_XII_Database.md` through H3 migrations 055–058 and state that H4 is database-neutral;
3. correct stale H3 documentation wording;
4. rerun and retain the disposable migration-058 execution regression;
5. independently reverify only these residual issues before the final closure assurance.

## Independent disposition

```text
PHASE_H_CLOSURE_BLOCKED

H-001 OPEN — MEDIUM
H-002 OPEN — MEDIUM
H-003 OPEN — MEDIUM
H-004 INFORMATIONAL / EVIDENCE STEP

READY_FOR_PHASE_H_FINAL_TARGETED_CORRECTION
```

This review does not recommend reopening ADR-0020, changing H1–H3 runtime semantics, adding database coordination authority, or broadening the H4 architecture.
