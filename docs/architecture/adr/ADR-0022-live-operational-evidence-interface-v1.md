# ADR-0022: Live operational evidence interface V1, amended for V2

Status: Accepted
Date: 2026-10-01
Deciders: Project architecture
Affected volumes: Volume XI §§3–9; Volume XII §6
Supersedes: None
Superseded by: None

## 1. Context and problem statement

ADR-0021 extracted a dedicated read-only scheduler observation boundary, but
its `lstm-observer` commands still render human-oriented status reports. AI
and tooling need a small machine-readable live evidence surface without
acquiring scheduler authority or treating rendered text as data.

## 2. Decision

`lstm-observer` exposes two closed V1 JSON requests:

```text
lstm-observer evidence scheduler
lstm-observer evidence experiment EXPERIMENT_ID
```

Both emit `expertadvisor-operational-evidence-v1`. They construct typed
observation records inside the existing `REPEATABLE READ, READ ONLY`
SchedulerOperationalReadModel snapshot and serialize those records directly.
The JSON has distinct `durable` and `process_observation` objects. Durable
PostgreSQL scheduler, lifecycle, and exact-attempt data remain authoritative;
operating-system process data is diagnostic only.

The interface is a closed evidence surface. It has no generic SQL operation,
scheduler control, process signaling, worker publication, registry mutation,
or experiment mutation. It preserves the existing human observer commands and
does not merge `Scripts/ExpertAdvisorInvestigator.py` with live evidence.

### 2.1 V2 amendment: inference and profitability evidence

V2 adds two more closed, read-only requests without changing the V1 schema or
V1 commands:

```text
lstm-observer evidence inference EXPERIMENT_ID
lstm-observer evidence profitability EXPERIMENT_ID
```

They emit `expertadvisor-operational-evidence-v2` with an explicit `kind` and
`requested_experiment_id`. Their `durable` arrays contain typed PostgreSQL
records captured in the same `REPEATABLE READ, READ ONLY` snapshot before JSON
serialization. An existing experiment with no qualifying evidence yields an
empty array; an unknown experiment is an error.

Final inference association reconstructs the repository's established exact
final inference context from the experiment, final model, model configuration,
target type, and inference range. Checkpoint association requires the durable
`parent_experiment_id`, `checkpoint_eval_id`, checkpoint epoch, and checkpoint
model relationship. The response returns every qualifying row in deterministic
order instead of selecting a latest row. Profitability observations additionally
validate their persisted result ID, model, scope, checkpoint identity, and
inference range against that associated inference evidence.

Profitability values are immutable terminal-horizon directional log-return
observations, not money, portfolio return, or realized P&L. V2 intentionally
returns the persisted identity hashes but omits the potentially large canonical
strings. It is neither generic SQL nor experiment comparison.

## 3. Rationale and decision drivers

Typed capture prevents formatting changes from becoming an unversioned API.
Explicit schema and kind fields let an investigator consume the evidence
deterministically, while narrow request grammar prevents a diagnostic tool
from becoming an operational-control interface.

## 4. Consequences

Consumers can combine repository manifests and live evidence, but those are
separate evidence producers with different authority boundaries. Evidence is
not authorization. A V2 may add narrowly justified evidence fields, but it
must preserve the durable-versus-diagnostic distinction and ADR-0018
exact-attempt authority.

## 5. Compatibility and migration

This is additive. `LSTM_Release --status`, `LSTM_Release --scheduler-status`,
`lstm-observer scheduler`, and `lstm-observer experiment` retain their
existing output and transaction behavior. No database schema, privilege,
scheduler, worker, registry, or experiment state changes are required.

## 6. Verification and operational evidence

Tests cover the closed evidence grammar, invalid identifiers, private
read-only snapshot boundary, typed JSON serializer, structural durable/process
separation, and absence of control/signaling dependencies. Committed-tree
observer smoke tests parse both JSON results using a deterministic JSON parser.

## 7. Alternatives considered

Parsing the existing text reports was rejected because the reports are human
renderers, not a stable typed authority. Extending the repository investigator
was rejected because it must remain checked-in-only and cannot acquire live
PostgreSQL or process evidence. A generic query endpoint was rejected because
it would broaden observation into an unsafe authority surface.

## 8. References

- [ADR-0018](ADR-0018-scheduler-generation-52-exact-attempt-authority.md)
- [ADR-0021](ADR-0021-read-only-scheduler-operational-observation.md)
- [Volume XI](../Volume_XI_Scheduler.md)
- [Volume XII](../Volume_XII_Database.md)

## 9. Revision history

| Date | Change |
|---|---|
| 2026-10-01 | Accepted the narrow typed live operational evidence interface. |
| 2026-10-01 | Amended with additive V2 typed inference and terminal-horizon profitability evidence; V1 remains unchanged. |
