# Return Feature Exact-Attempt Reconciliation
## Final Independent Reverification Findings

**Reviewed source:** `ReturnFeature_ExactAttemptReconciliation_FinalIndependentReverification_Input.txt`
**Reviewed HEAD:** `6fe16942ba2294daad8d8a0f99ed4d10d767eb44` on `lstm-feature-development`

## Executive verdict

**VERDICT: PASS**

**Blocking findings: NONE.**

The final implementation is narrowly scoped and fail-closed. `--reconcile-worker-attempt=ID` is bound to one exact immutable worker-attempt ID, re-verifies the exact attempt and exact lifecycle binding under lock, retains historical scheduler invocation/fence identity as exact predicates, distinguishes positive process absence from inspection failure, and commits the absent-process attempt/lifecycle changes atomically before emitting `outcome=applied`.

The captured production state shows that both target attempts have already been terminalized successfully:

- attempt 623 / experiment 549: failed and unbound;
- attempt 629 / experiment 554: failed and unbound.

Therefore **do not run either reconciliation command again**. The terminal-replay/source-state fence is designed to reject them now.

The old scheduler PID 42864 remains SIGSTOP-frozen, worker PIDs 68338 and 32973 are absent, no checkpoint evaluation is running, and experiments 552/553 remain paused and unchanged.

## Findings

### 1. Exact-attempt scope — PASS

The command selects and locks one `worker_attempt_id`. Verification includes experiment/checkpoint identity, worker kind, lifecycle phase, capacity class, historical scheduler invocation ID/fencing token, required `identity_ambiguous` source state, complete process identity, and the exact active lifecycle binding. It does not scan attempts and is not generic orphan recovery.

**Classification: NONE.**

### 2. Identity and fencing — PASS

The guarded path includes exact PID, PGID, start identity, canonical executable, full command line, command identity, active experiment-to-attempt binding, scheduler invocation ID and fencing token. The administrative command does not acquire scheduler dispatch authority or refresh the scheduler lease; this is appropriate because it cannot dispatch or signal work.

**Classification: NONE.**

### 3. Live-process branch — PASS

A live worker cannot enter the absent-process branch. It must pass `ValidateManagedWorker`, including exact worker-attempt option, PID/PGID/start identity, executable, command, and phase/experiment identity. A valid live process may transition only from `identity_ambiguous` to `observed`; incomplete or mismatched identity rejects.

**Classification: NONE.**

### 4. Positive absence vs. inspection failure — PASS

Absent-process handling is admitted only for `ProcessMissing` with successful inspection and no permission denial. Permission errors, incomplete inspection, path/canonicalization failures, or start-identity instability fail closed.

**Classification: NONE.**

### 5. Atomic absent-process mutation — PASS

The absent branch updates the exact worker-attempt and its exact experiment lifecycle in the same `pqxx::work` transaction. Each mutation has exact identity/binding predicates and requires exactly one affected row.

Expected durable semantics are:

- attempt: `failed`, exit `-1`, `process_missing_no_result`, diagnostic `exact_process_identity_absent_no_result`, completion set;
- experiment: `failed`, original phase retained, PID/PGID cleared, active attempt cleared, exit `-1`, `worker_process_missing_no_result`, completion set.

**Classification: NONE.**

### 6. Dry-run — PASS

Dry-run performs validation and reports the proposed result without persistence mutation. The focused test verifies unchanged state from a fresh database connection.

**Classification: NONE.**

### 7. Stale/replaced binding rejection — PASS

The focused test changes the lifecycle command identity and confirms rejection with no partial mutation and no `outcome=applied`. Final UPDATE predicates repeat exact identity checks and use exactly-one-row enforcement.

**Classification: NONE.**

### 8. Commit failure / applied ordering — PASS

The corrected order is:

1. lock and verify;
2. classify process;
3. print `outcome=applying`;
4. perform both guarded updates;
5. require exactly one row from each;
6. commit;
7. print `outcome=applied`.

The focused integration test forces a deferred commit failure and proves: exit status 2, reconciliation error, no `outcome=applied`, and fresh-connection rollback of both rows.

**Classification: NONE.**

### 9. Test adequacy — PASS

The dedicated integration test covers dry-run, absent process, committed terminalization, terminal replay rejection, stale binding rejection, matching live process, forced commit failure, and fresh-connection verification. Observer/process tests additionally cover PID reuse, PGID/executable/command mismatch, worker-attempt mismatch, and fail-closed fallback behavior.

A broader scheduler ownership suite previously encountered an unrelated fixture/schema mismatch. That is not a blocker for this narrow correction.

**Classification: LOW test-environment debt only.**

### 10. Observe() ENOENT fallback — PASS

The `proc_pidpath()` ENOENT correction does not weaken identity. Only ENOENT triggers `KERN_PROCARGS2`; the kernel-recorded executable path is canonicalized with `realpath`, process start identity is read before and after observation, and the full identity checks remain required. Other failures remain fail-closed.

**Classification: NONE.**

## Production state

The current captured database state shows:

- experiment 549: `failed/train`, current epoch 61, PID/PGID and active attempt cleared, exit `-1`, `worker_process_missing_no_result`;
- experiment 554: `failed/train`, current epoch 79, PID/PGID and active attempt cleared, exit `-1`, `worker_process_missing_no_result`;
- attempt 623: `failed`, exit `-1`, `process_missing_no_result`, diagnostic `exact_process_identity_absent_no_result`;
- attempt 629: same expected terminal state;
- experiments 552 and 553: still `paused/train`, unchanged;
- scheduler PID 42864: still `T+`;
- old worker PIDs 68338 and 32973: absent;
- running checkpoint evaluations: zero.

So attempts 623 and 629 are **already complete**. Re-running `--reconcile-worker-attempt` is neither necessary nor appropriate.

## Remaining cutover blockers

There is **no exact-attempt reconciliation blocker** before retiring old scheduler PID 42864.

Retirement should be a separate guarded operation that proves the old scheduler identity/frozen state, proves 623/629 and 549/554 remain terminal/unbound, proves old workers remain absent and checkpoint evaluations remain quiescent, then kills only the old scheduler and proves it is gone.

After that, start the corrected scheduler as a separate checkpoint and verify its ownership/authority and process status before resuming experiments 552/553. Do not resume 552/553 in the scheduler-retirement operation.

## Repository hygiene

The captured repository is seven commits ahead of its remote tracking branch and has one untracked review artifact:

`ReturnFeature_ExactAttemptReconciliation_ApplyCommitPersistence_FinalCorrection_Output.md`

This is not a runtime-safety blocker, but it should be archived/staged/committed, and the local commits pushed when appropriate, before archival closure.

**Classification: LOW.**

## Final classification

- BLOCKER: 0
- HIGH: 0
- MEDIUM: 0
- LOW: 1 (repository/archive hygiene only)
- NONE: all exact-attempt reconciliation safety questions

**VERDICT: PASS**
