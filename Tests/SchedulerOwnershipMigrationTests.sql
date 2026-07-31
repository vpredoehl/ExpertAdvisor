\set ON_ERROR_STOP on

DO $$
BEGIN
    IF to_regclass('experiment_scheduler_invocation') IS NULL
       OR to_regclass('experiment_scheduler_lease') IS NULL
       OR to_regclass('experiment_scheduler_worker_attempt') IS NULL
       OR to_regclass('experiment_scheduler_protocol') IS NULL THEN
        RAISE EXCEPTION 'scheduler ownership/protocol tables missing';
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'experiment'
          AND column_name = 'active_scheduler_worker_attempt_id'
    ) THEN
        RAISE EXCEPTION 'experiment active attempt fence missing';
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'experiment_checkpoint_eval'
          AND column_name = 'active_scheduler_worker_attempt_id'
    ) THEN
        RAISE EXCEPTION 'checkpoint active attempt fence missing';
    END IF;
END $$;

BEGIN;
SELECT set_config(
    'expertadvisor.scheduler_protocol_generation',
    '52',
    true
);

INSERT INTO experiment (
    experiment_id,symbol,prediction_horizon,c_next_threshold,
    target_epochs,checkpoint_interval,train_start,train_end,
    status,phase,duplicate_nonce
) VALUES (
    910051,'schedulerownershiptest',4,0.0008,
    20,20,'2020-01-01','2021-01-01',
    'pending','train',910051
);

INSERT INTO experiment_scheduler_invocation (
    scheduler_invocation_id,process_pid,process_group_id,
    process_start_identity,canonical_executable_path,command_line,
    invocation_nonce,status
) VALUES
    ('scheduler:test-owner-a',910051,910051,'start-a',
     '/tmp/LSTM_Release','/tmp/LSTM_Release --schedule-experiments',
     '000000000000000000000000000000a1','owner'),
    ('scheduler:test-owner-b',910052,910052,'start-b',
     '/tmp/LSTM_Release','/tmp/LSTM_Release --schedule-experiments',
     '000000000000000000000000000000b2','starting');

UPDATE experiment_scheduler_invocation
SET protocol_generation=52
WHERE scheduler_invocation_id IN (
    'scheduler:test-owner-a',
    'scheduler:test-owner-b'
);

UPDATE experiment_scheduler_lease
SET owner_scheduler_invocation_id='scheduler:test-owner-a',
    fencing_token=51,
    authority_state='active',
    acquired_at=clock_timestamp(),
    heartbeat_at=clock_timestamp(),
    expires_at=clock_timestamp()+interval '90 seconds',
    released_at=NULL,
    transition_reason='migration_test'
WHERE singleton;

DO $$
DECLARE
    changed integer;
BEGIN
    UPDATE experiment_scheduler_lease
    SET heartbeat_at=clock_timestamp()
    WHERE singleton
      AND owner_scheduler_invocation_id='scheduler:test-owner-b'
      AND fencing_token=51;
    GET DIAGNOSTICS changed = ROW_COUNT;
    IF changed <> 0 THEN
        RAISE EXCEPTION 'foreign scheduler refreshed owner lease';
    END IF;

    UPDATE experiment_scheduler_lease
    SET authority_state='released'
    WHERE singleton
      AND owner_scheduler_invocation_id='scheduler:test-owner-b'
      AND fencing_token=51;
    GET DIAGNOSTICS changed = ROW_COUNT;
    IF changed <> 0 THEN
        RAISE EXCEPTION 'foreign scheduler released owner lease';
    END IF;
END $$;

INSERT INTO experiment_scheduler_worker_attempt (
    launch_attempt_identity,scheduler_invocation_id,
    scheduler_fencing_token,experiment_id,worker_kind,
    lifecycle_phase,capacity_class,ownership_origin,lifecycle_state,
    canonical_executable_path,command_identity
) VALUES (
    'migration-test-attempt-0000000000000001',
    'scheduler:test-owner-a',51,910051,'experiment',
    'train','train','scheduler_launch','reserved',
    '/tmp/LSTM_Release','experiment:910051:train'
);

UPDATE experiment
SET status='running',
    active_scheduler_worker_attempt_id=(
        SELECT worker_attempt_id
        FROM experiment_scheduler_worker_attempt
        WHERE launch_attempt_identity=
              'migration-test-attempt-0000000000000001'
    )
WHERE experiment_id=910051;

DO $$
DECLARE
    consuming integer;
BEGIN
    SELECT count(*) INTO consuming
    FROM experiment_scheduler_worker_attempt
    WHERE capacity_class='train'
      AND lifecycle_state IN (
          'reserved','spawned','running','observed',
          'identity_ambiguous'
      );
    IF consuming <> 1 THEN
        RAISE EXCEPTION 'reserved attempt did not consume capacity';
    END IF;
END $$;

INSERT INTO model(model_id) VALUES (910053);
INSERT INTO experiment_checkpoint_eval (
    checkpoint_eval_id,experiment_id,parent_experiment_id,
    checkpoint_epoch,checkpoint_model_id,status,phase
) VALUES (
    910053,910051,910051,20,910053,'pending','infer'
);
WITH attempt AS (
    INSERT INTO experiment_scheduler_worker_attempt (
        launch_attempt_identity,scheduler_invocation_id,
        scheduler_fencing_token,experiment_id,checkpoint_eval_id,
        worker_kind,lifecycle_phase,capacity_class,ownership_origin,
        lifecycle_state,canonical_executable_path,command_identity
    ) VALUES (
        'migration-test-checkpoint-attempt-000001',
        'scheduler:test-owner-a',51,910051,910053,
        'checkpoint_infer','infer','infer',
        'scheduler_launch','reserved',
        '/tmp/LSTM_Release','checkpoint_infer:910053'
    )
    RETURNING worker_attempt_id
)
UPDATE experiment_checkpoint_eval
SET status='running',
    active_scheduler_worker_attempt_id=(
        SELECT worker_attempt_id FROM attempt
    )
WHERE checkpoint_eval_id=910053;

DO $$
BEGIN
    BEGIN
        INSERT INTO experiment_scheduler_worker_attempt (
            launch_attempt_identity,scheduler_invocation_id,
            scheduler_fencing_token,experiment_id,checkpoint_eval_id,
            worker_kind,lifecycle_phase,capacity_class,
            ownership_origin,lifecycle_state,
            canonical_executable_path,command_identity
        ) VALUES (
            'migration-test-checkpoint-attempt-000002',
            'scheduler:test-owner-a',51,910051,910053,
            'checkpoint_infer','infer','infer',
            'scheduler_launch','reserved',
            '/tmp/LSTM_Release','checkpoint_infer:910053'
        );
        RAISE EXCEPTION
            'duplicate active checkpoint-infer attempt accepted';
    EXCEPTION
        WHEN unique_violation THEN
            NULL;
    END;
END $$;

DO $$
BEGIN
    BEGIN
        INSERT INTO experiment_scheduler_worker_attempt (
            launch_attempt_identity,scheduler_invocation_id,
            scheduler_fencing_token,experiment_id,worker_kind,
            lifecycle_phase,capacity_class,ownership_origin,
            lifecycle_state,canonical_executable_path,command_identity
        ) VALUES (
            'migration-test-attempt-0000000000000002',
            'scheduler:test-owner-a',51,910051,'experiment',
            'train','train','scheduler_launch','reserved',
            '/tmp/LSTM_Release','experiment:910051:train'
        );
        RAISE EXCEPTION 'duplicate active experiment attempt accepted';
    EXCEPTION
        WHEN unique_violation THEN
            NULL;
    END;
END $$;

UPDATE experiment_scheduler_worker_attempt
SET lifecycle_state='failed',completed_at=clock_timestamp()
WHERE launch_attempt_identity=
      'migration-test-attempt-0000000000000001';

DO $$
DECLARE
    consuming integer;
BEGIN
    SELECT count(*) INTO consuming
    FROM experiment_scheduler_worker_attempt
    WHERE capacity_class='train'
      AND lifecycle_state IN (
          'reserved','spawned','running','observed',
          'identity_ambiguous'
      );
    IF consuming <> 0 THEN
        RAISE EXCEPTION 'terminal attempt continued consuming capacity';
    END IF;
END $$;

INSERT INTO model(model_id) VALUES (910052);
INSERT INTO experiment_checkpoint_eval (
    checkpoint_eval_id,experiment_id,parent_experiment_id,
    checkpoint_epoch,checkpoint_model_id,status,phase
) VALUES (
    910052,910051,910051,20,910052,'pending','analyze'
);

WITH attempt AS (
    INSERT INTO experiment_scheduler_worker_attempt (
        launch_attempt_identity,scheduler_invocation_id,
        scheduler_fencing_token,experiment_id,checkpoint_eval_id,
        worker_kind,lifecycle_phase,capacity_class,ownership_origin,
        lifecycle_state,canonical_executable_path,command_identity
    ) VALUES (
        'migration-test-analyze-attempt-00000001',
        'scheduler:test-owner-a',51,910051,910052,
        'checkpoint_analyze','analyze','analyze',
        'scheduler_in_process','running',
        '/tmp/LSTM_Release','checkpoint_analyze:910052'
    )
    RETURNING worker_attempt_id
)
UPDATE experiment_checkpoint_eval
SET status='running',
    active_scheduler_worker_attempt_id=(
        SELECT worker_attempt_id FROM attempt
    )
WHERE checkpoint_eval_id=910052;

DO $$
DECLARE
    analyze_attempt_id bigint;
    foreign_changed boolean;
BEGIN
    SELECT active_scheduler_worker_attempt_id
    INTO analyze_attempt_id
    FROM experiment_checkpoint_eval
    WHERE checkpoint_eval_id=910052;

    BEGIN
        INSERT INTO experiment_scheduler_worker_attempt (
            launch_attempt_identity,scheduler_invocation_id,
            scheduler_fencing_token,experiment_id,checkpoint_eval_id,
            worker_kind,lifecycle_phase,capacity_class,
            ownership_origin,lifecycle_state,
            canonical_executable_path,command_identity
        ) VALUES (
            'migration-test-analyze-attempt-00000002',
            'scheduler:test-owner-a',51,910051,910052,
            'checkpoint_analyze','analyze','analyze',
            'scheduler_in_process','reserved',
            '/tmp/LSTM_Release','checkpoint_analyze:910052'
        );
        RAISE EXCEPTION 'duplicate active analyze attempt accepted';
    EXCEPTION
        WHEN unique_violation THEN
            NULL;
    END;

    BEGIN
        UPDATE experiment
        SET active_scheduler_worker_attempt_id=analyze_attempt_id
        WHERE experiment_id=910051;
        RAISE EXCEPTION
            'checkpoint analyze attempt bound to experiment lifecycle';
    EXCEPTION
        WHEN check_violation THEN
            NULL;
    END;

    WITH changed AS (
        UPDATE experiment_scheduler_worker_attempt a
        SET lifecycle_state='completed',
            completed_at=clock_timestamp()
        WHERE a.worker_attempt_id=analyze_attempt_id + 100000
          AND a.scheduler_invocation_id='scheduler:test-owner-a'
          AND a.scheduler_fencing_token=51
          AND EXISTS (
              SELECT 1
              FROM experiment_checkpoint_eval ce
              WHERE ce.checkpoint_eval_id=910052
                AND ce.active_scheduler_worker_attempt_id=
                    a.worker_attempt_id
          )
        RETURNING 1
    )
    SELECT EXISTS(SELECT 1 FROM changed)
    INTO foreign_changed;
    IF foreign_changed THEN
        RAISE EXCEPTION 'foreign attempt released analyze capacity';
    END IF;

    UPDATE experiment_scheduler_worker_attempt a
    SET lifecycle_state='completed',
        completed_at=clock_timestamp()
    WHERE a.worker_attempt_id=analyze_attempt_id
      AND a.scheduler_invocation_id='scheduler:test-owner-a'
      AND a.scheduler_fencing_token=51
      AND EXISTS (
          SELECT 1
          FROM experiment_checkpoint_eval ce
          WHERE ce.checkpoint_eval_id=910052
            AND ce.active_scheduler_worker_attempt_id=a.worker_attempt_id
      );
    IF NOT FOUND THEN
        RAISE EXCEPTION 'exact analyze attempt did not terminalize';
    END IF;
    UPDATE experiment_checkpoint_eval
    SET active_scheduler_worker_attempt_id=NULL,
        status='completed',
        phase='done'
    WHERE checkpoint_eval_id=910052
      AND active_scheduler_worker_attempt_id=analyze_attempt_id;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'exact analyze lifecycle did not clear';
    END IF;
END $$;

DO $$
BEGIN
    BEGIN
        UPDATE experiment_scheduler_protocol
        SET cutover_state='failed'
        WHERE singleton=true;
        RAISE EXCEPTION
            'completed cutover could be rolled back without clearing proof';
    EXCEPTION
        WHEN check_violation THEN
            NULL;
    END;
END $$;

ROLLBACK;

\echo 'SchedulerOwnershipMigrationTests passed'
