# Campaign-materialization scheduler control

The administrative selector is the immutable recommendation campaign
materialization, not the separate operational campaign identity:

```text
LSTM_Release --pause-campaign-materialization=ID --dry-run
LSTM_Release --pause-campaign-materialization=ID --yes
LSTM_Release --resume-campaign-materialization=ID --dry-run
LSTM_Release --resume-campaign-materialization=ID --yes
```

Mutating commands require `--yes`. Dry-run performs no writes or signals.

Pause resolves the exact ordered materialization manifest through its frozen
conversion proposals and durable conversion execution/activation evidence. It
records an immutable operation, exact member snapshots, and per-member
outcomes. Pending train/infer members become paused. Running train/infer
members use the existing exact worker-attempt and process-identity checks
before SIGSTOP; stopped attempts consume no scheduler execution capacity.
Already paused, terminal, analyze, unresolved, and identity-failed members are
classified without granting release ownership.

Only `changed_by_group_pause` outcomes acquire active group ownership. An
individual experiment pause or resume changes that ownership to `superseded`.
A campaign-materialization resume can therefore release only still-active
ownership from a prior frozen pause population. It changes those experiments
to `status=pending,resume_requested=true`, preserves `scheduler_priority`, and
consumes the ownership in the same transaction.

Resume never sends SIGCONT. A retained stopped process remains stopped until
the ordinary scheduler admits it within compatible capacity. At admission the
existing exact-attempt path either safely sends SIGCONT or applies its existing
process-disappearance checkpoint/restart fallback. Ordering remains:

1. `resume_requested=true`;
2. persistent priority `high`, `normal`, `low`;
3. existing deterministic tie breakers.

Migration `087_campaign_materialization_pause_resume.sql` must be applied in a
separate reviewed cutover before using these commands. The application binary
must not be deployed before that schema is present. Migration and binary
deployment do not require or authorize changing existing experiment rows.
