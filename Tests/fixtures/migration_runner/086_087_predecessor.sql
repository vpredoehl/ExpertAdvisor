CREATE TABLE experiment (
    experiment_id bigint PRIMARY KEY,
    status text NOT NULL DEFAULT 'pending',
    phase text NOT NULL DEFAULT 'train',
    updated_at timestamptz NOT NULL DEFAULT now(),
    active_scheduler_worker_attempt_id bigint
);

CREATE TABLE experiment_scheduler_worker_attempt (
    worker_attempt_id bigint PRIMARY KEY,
    worker_kind text NOT NULL,
    experiment_id bigint REFERENCES experiment(experiment_id),
    checkpoint_eval_id bigint,
    lifecycle_phase text NOT NULL,
    capacity_class text NOT NULL,
    lifecycle_state text NOT NULL,
    CONSTRAINT experiment_scheduler_worker_attempt_lifecycle_state_check
        CHECK (lifecycle_state IN (
            'reserved', 'spawned', 'running', 'observed',
            'identity_ambiguous', 'completed', 'failed',
            'launch_failed', 'abandoned'
        ))
);

CREATE UNIQUE INDEX experiment_scheduler_worker_attempt_active_experiment_uidx
    ON experiment_scheduler_worker_attempt(experiment_id, lifecycle_phase)
    WHERE worker_kind = 'experiment'
      AND lifecycle_state IN (
          'reserved', 'spawned', 'running', 'observed',
          'identity_ambiguous'
      );

CREATE UNIQUE INDEX experiment_scheduler_worker_attempt_active_checkpoint_uidx
    ON experiment_scheduler_worker_attempt(checkpoint_eval_id)
    WHERE worker_kind = 'checkpoint_infer'
      AND lifecycle_state IN (
          'reserved', 'spawned', 'running', 'observed',
          'identity_ambiguous'
      );

CREATE UNIQUE INDEX scheduler_worker_attempt_active_checkpoint_analyze_uidx
    ON experiment_scheduler_worker_attempt(checkpoint_eval_id)
    WHERE worker_kind = 'checkpoint_analyze'
      AND lifecycle_state IN (
          'reserved', 'spawned', 'running', 'observed',
          'identity_ambiguous'
      );

CREATE TABLE experiment_recommendation_campaign_materialization (
    recommendation_campaign_materialization_id bigint PRIMARY KEY
);

CREATE TABLE experiment_recommendation_campaign_materialization_member (
    recommendation_campaign_materialization_member_id bigint PRIMARY KEY
);

CREATE TABLE experiment_recommendation_conversion_execution (
    recommendation_conversion_execution_id bigint PRIMARY KEY
);

CREATE TABLE experiment_recommendation_conversion_activation (
    recommendation_conversion_activation_id bigint PRIMARY KEY
);
