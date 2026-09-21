Forced FINAL Inference Reruns
=============================

``--requeue-inference=EXPERIMENT_ID`` is the operator control for rerunning
FINAL inference through the scheduler.  Migration 078 adds the durable
``experiment.operator_forced_final_inference_rerun_requested`` flag.  Apply
the migration before deploying a scheduler binary that requires this column.

The confirmed control command sets the flag and moves the experiment to
``pending/infer``::

  "$CANONICAL_BIN" --requeue-inference=547 --yes

``--scheduler-status`` exposes the flag as
``operator_forced_final_inference_rerun_requested=1`` in machine output and as
``Forced FINAL Infer Rerun: requested`` in compact human output.

Lifecycle
---------

The flag is durable and is not cleared when a worker is merely reserved or
launched.  The existing scheduler lease, fencing token, active worker-attempt
link, and process-identity reconciliation therefore prevent a restart or
ownership handoff from launching a second worker for the same request.

The flag bypasses only the scheduler's completed-FINAL-result dispatch skip.
It does not alter checkpoint inference and does not alter the unique semantic
identity or upsert behavior of ``inference_eval_result``.

The scheduler clears the flag in the same transaction that advances the exact
running inference attempt to analysis, and only after finding a matching
completed FINAL result whose ``completed_at`` is at or after that attempt's
reservation time.  A failed attempt without this evidence becomes
``failed/infer`` with an error beginning
``forced_final_inference_rerun_missing_attempt_result;`` and retains the flag.
The operator may inspect the failure and issue ``--requeue-inference`` again.

Verification
------------

Capture the semantic result identity before requeueing::

  psql -d LSTM -v experiment_id=547 <<'SQL'
  SELECT e.experiment_id,
         e.status,
         e.phase,
         e.operator_forced_final_inference_rerun_requested,
         r.id AS inference_eval_result_id,
         r.completed_at,
         p.profitability_observation_id
  FROM experiment e
  JOIN inference_eval_result r
    ON r.model_id = e.last_model_id
   AND r.symbol = e.symbol
   AND r.prediction_horizon = e.prediction_horizon
   AND r.threshold_logret = e.c_next_threshold
   AND r.from_date = e.infer_start::date::text
   AND r.to_date = e.infer_end::date::text
   AND r.status = 'completed'
   AND r.inference_scope = 'final'
   AND r.checkpoint_eval_id IS NULL
  LEFT JOIN inference_profitability_observation p
    ON p.inference_eval_result_id = r.id
  WHERE e.experiment_id = :experiment_id;
  SQL

After the command, monitor the durable state and exact worker attempt::

  "$CANONICAL_BIN" --scheduler-status

  psql -d LSTM -v experiment_id=547 <<'SQL'
  SELECT e.experiment_id, e.status, e.phase,
         e.operator_forced_final_inference_rerun_requested,
         e.active_scheduler_worker_attempt_id,
         a.lifecycle_state, a.worker_pid, a.reserved_at
  FROM experiment e
  LEFT JOIN experiment_scheduler_worker_attempt a
    ON a.worker_attempt_id = e.active_scheduler_worker_attempt_id
  WHERE e.experiment_id = :experiment_id;
  SQL

After completion, rerun the first query and confirm that the original result
ID remains, ``completed_at`` is refreshed, a profitability observation is
present, and the force flag is false.  Confirm semantic uniqueness explicitly::

  psql -d LSTM -v experiment_id=547 <<'SQL'
  SELECT r.model_id, r.symbol, r.prediction_horizon, r.threshold_logret,
         r.from_date, r.to_date, count(*) AS completed_final_rows
  FROM experiment e
  JOIN inference_eval_result r
    ON r.model_id = e.last_model_id
   AND r.symbol = e.symbol
   AND r.prediction_horizon = e.prediction_horizon
   AND r.threshold_logret = e.c_next_threshold
   AND r.from_date = e.infer_start::date::text
   AND r.to_date = e.infer_end::date::text
   AND r.status = 'completed'
   AND r.inference_scope = 'final'
   AND r.checkpoint_eval_id IS NULL
  WHERE e.experiment_id = :experiment_id
  GROUP BY r.model_id, r.symbol, r.prediction_horizon,
           r.threshold_logret, r.from_date, r.to_date;
  SQL

The count must be exactly one.
