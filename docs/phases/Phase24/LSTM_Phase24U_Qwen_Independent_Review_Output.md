# Phase 24U — Qwen independent review and controlled quiescence

## Result

Production was paused once through the supported global experiment-control
workflow and restored before this review was closed. It was not paused again.
No production source, database row, registry, binary, scheduler setting, or
process was modified outside those authorized pause/resume controls. Qwen was
actually invoked through the configured MLX MCP. The review is **NO-GO for
production TRAIN expansion** and **NO-GO for treating the candidate rollback
guard as qualified** until its private regression can run successfully. The
existing Phase 24T scope remains the only qualified scheduler change.

## Baseline and evidence

Development was branch `dedicated-train-layout-rollover-squashed-v1`, HEAD
`577c20b0e1f567caa7724ebcf9e790aab8aca694` (Phase 24T), with the existing
Phase 24T work preserved. Production was read-only except for the authorized
supported pause/resume operation; its preflight branch/HEAD were recorded in
`DerivedData/ExpertAdvisor/Phase24U/preflight.json`. All operational evidence
and credentials remain under the ignored, mode-0700 Phase24U directory.

The preflight recorded one production scheduler (PID 58062), protocol
generation 52 complete, active lease/fencing token 200, and the configured
`TRAIN=0, INFER=1, ANALYZE=0` command with continuation queueing disabled.
The live operator pauses were exactly 732, 733, 748 and 749. Workers 714 and
716 retained their exact PID, attempt, kernel start identity, executable and
command-line bindings. No active TRAIN or ANALYZE worker was present.

The supported retained production controller performed the pause with its
coordination lock and verified worker identities. The quiescent snapshot shows
the global gate paused and both executing INFER workers OS-stopped. Restoration
returned the gate to `running`, cleared the request, preserved the owner and
fencing token, retained the operator-paused rows, and left the registry and
qualified artifact hashes unchanged. The first bounded post-restore admission
check recorded one INFER, zero TRAIN and zero ANALYZE. The production scheduler
was never restarted or replaced by development code.

## Why one Qwen bundle was rejected

The configured repository MCP is a read-only source-review interface. Its
source reader admits production `Sources/` and `Headers/` paths plus a small
explicit allowlist of test paths; it does not admit the general
`Tests/GlobalExperimentControlTests.cpp` path used by the first native-identity
bundle. The MCP therefore returned `ValueError: claim source range could not
be read` before model invocation. This was an admission-policy failure, not a
Qwen verdict and not a source finding. No credentials, connection strings,
production records or secrets were sent to the model. The bundle was not
retried with a configuration change.

## Actual Qwen review

The model identifier was
`mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit`. Five uncached model turns
completed through `investigate_source_bundle_claim`; manifests record
`model_turn_count=1` for each and `ledger_hit=false`. The admitted bundles
covered native identity, rollback compensation, reservation/child-exec gate,
and priority/operator-pause behavior. The MCP does not run tests and could not
admit the general test source, so this is a bounded source review rather than a
full independent review of every Phase 24T regression.

Qwen's first rollback answer was not accepted: it was based on an incomplete
range and incorrectly suggested that the old compensation path respected an
operator pause. The corrected range showed the opposite: the pre-abort
`ManagedWorker` snapshot was used after transaction abort without a fresh
authority/lifecycle/attempt check. Qwen returned `supports:false` for the
claimed safety property. Codex independently confirmed the concrete
interleaving: scheduler A SIGSTOPs a victim, aborts, scheduler/control action B
commits a newer pause or ownership change, and A's old snapshot can SIGCONT it.
This is a confirmed source-level race in the pre-existing compensation helper
as reused by Phase 24T; it is not evidence that caused production observations.

The native-identity and priority/operator-pause answers were narrow positive
findings consistent with the source and Phase 24T tests. The reservation-gate
answer was a non-finding caused by the supplied bundle not establishing the
whole external-resume chain; it did not contradict the Phase 24T native result.
No Qwen answer established severity, production causality, or a new test result.

## Development follow-up

The old behavior was characterized in a private SCRAM cluster using inert,
positively identified fixture workers. The retained result records a supported
global pause committing after the scheduler transaction abort and the old
compensation subsequently reporting `restored=1`, reproducing the unsafe
abort-to-SIGCONT interleaving. The fixture cleanup receipt is PASS and no test
worker or private PostgreSQL service remains.

A narrowly scoped guard was drafted in the worktree. It reacquires scheduler
authority and the global running gate, locks and rechecks the exact attempt,
PID/PGID/start identity, executable, command, lifecycle and observing owner
before compensation. On uncertainty it withholds SIGCONT and leaves the exact
attempt for normal reconciliation. The targeted Phase24U build and link pass
with the same two pre-existing Apple-Clang warnings seen in the Phase 24T
source build. `py_compile` also passes for the private regression harness.

The post-fix private regression could not be executed: the sandbox denied
PostgreSQL's `shmget` during `initdb` (`Operation not permitted`). The required
escalated retry was rejected by the session credit limit, so the fix and its
new harness are intentionally **not committed or declared qualified**. No
additional retry was made. The modified source and untracked harness remain
visible for a later authorized development run; the Phase 24T baseline commit
was not amended.

## Memory and restoration observations

The retained host snapshots showed approximately 52–57% free memory. Swap
usage rose from about 11.2 GiB at preflight/quiescence to about 14.9 GiB after
restoration while the host also ran Qwen. Swap used is cumulative/occupancy
evidence, not a pressure rate and cannot attribute memory to production
workers. No model training or inference was launched for this review.

## Decisions and remaining coverage

* Scheduler correctness: **NO-GO for the unvalidated compensation guard**;
  the Phase 24T qualified external-resume reconciliation remains unchanged.
* Production TRAIN=1 / INFER=1: **NO-GO** pending representative memory
  qualification and a separately authorized activation.
* Unreviewed or only partially reviewed: general test-source behavior through
  Qwen, real production-scale memory, real PID reuse, concurrent result writers,
  and the cause of historical production displacement.
* Confirmed defect: abort-time compensation can act on a stale worker snapshot
  after a newer pause/authority transition. Hypotheses about production
  causality remain unproven.

`git diff --check` and the targeted Phase24U build completed. Production was
paused: **yes, once**; restored: **yes**; Qwen used: **yes, five model turns**.
No merge, push, publication, deployment or production TRAIN activation was
performed.
