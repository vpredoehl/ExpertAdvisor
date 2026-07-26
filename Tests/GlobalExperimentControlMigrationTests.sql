DO $migration_test$
BEGIN
    IF to_regclass('experiment_global_control') IS NULL OR
       to_regclass('experiment_admin_request') IS NULL OR
       to_regclass('experiment_admin_worker_outcome') IS NULL THEN
        RAISE EXCEPTION 'global experiment control tables missing';
    END IF;

    IF (SELECT count(*) FROM experiment_global_control) <> 1 OR
       NOT EXISTS (
           SELECT 1 FROM experiment_global_control WHERE singleton=true
       ) THEN
        RAISE EXCEPTION 'global control singleton row missing or duplicated';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid='experiment_global_control'::regclass
          AND conname='experiment_global_control_active_request_fkey'
    ) THEN
        RAISE EXCEPTION 'active request foreign key missing';
    END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema=current_schema()
          AND table_name='experiment'
          AND column_name='worker_control_state'
          AND is_nullable='YES'
    ) OR NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema=current_schema()
          AND table_name='experiment'
          AND column_name IN (
              'worker_process_group_id',
              'worker_executable',
              'worker_command_line',
              'worker_process_start_identity',
              'worker_control_state',
              'cancellation_request_id',
              'cancel_after_checkpoint_epoch',
              'last_checkpoint_stop_decision_epoch',
              'cancel_infer_before',
              'cancellation_completed_at')
        GROUP BY table_name HAVING count(*)=10
    ) THEN
        RAISE EXCEPTION 'experiment control/identity columns incomplete';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema=current_schema()
          AND table_name='experiment_checkpoint_eval'
          AND column_name='cancellation_request_id'
    ) OR NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema=current_schema()
          AND table_name='experiment_checkpoint_eval'
          AND column_name='worker_process_group_id'
    ) OR NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema=current_schema()
          AND table_name='experiment_checkpoint_eval'
          AND column_name='worker_process_start_identity'
    ) OR NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema=current_schema()
          AND table_name='experiment_admin_worker_outcome'
          AND column_name='worker_process_start_identity'
    ) THEN
        RAISE EXCEPTION 'checkpoint worker control columns incomplete';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema=current_schema()
          AND table_name='experiment_global_control'
          AND column_name='current_pause_request_id'
    ) OR NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema=current_schema()
          AND table_name='experiment'
          AND column_name='worker_global_pause_request_id'
    ) OR NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema=current_schema()
          AND table_name='experiment_checkpoint_eval'
          AND column_name='worker_global_pause_request_id'
    ) OR NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema=current_schema()
          AND table_name='experiment_admin_request'
          AND column_name='target_experiment_id'
    ) OR NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema=current_schema()
          AND table_name='experiment_admin_worker_outcome'
          AND column_name IN (
              'worker_executable',
              'worker_command_line',
              'source_pause_request_id')
        GROUP BY table_name HAVING count(*)=3
    ) THEN
        RAISE EXCEPTION 'selective resume control columns incomplete';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid='experiment_admin_request'::regclass
          AND conname='experiment_admin_request_target_shape_check'
    ) THEN
        RAISE EXCEPTION 'selective resume request shape constraint missing';
    END IF;

    IF NOT has_table_privilege(
            'pqxx','experiment_global_control','SELECT') OR
       NOT has_table_privilege(
            'pqxx','experiment_global_control','UPDATE') OR
       NOT has_table_privilege(
            'pqxx','experiment_admin_request','INSERT') THEN
        RAISE EXCEPTION 'runtime global control privileges incomplete';
    END IF;

    IF has_column_privilege(
            'pqxx','experiment_admin_request','action','UPDATE') OR
       NOT has_column_privilege(
            'pqxx','experiment_admin_request','status','UPDATE') THEN
        RAISE EXCEPTION 'administrative request audit update privileges unsafe';
    END IF;

    IF has_column_privilege(
            'pqxx','experiment_admin_worker_outcome','experiment_id','UPDATE') OR
       has_column_privilege(
            'pqxx','experiment_admin_worker_outcome','worker_executable','UPDATE') OR
       has_column_privilege(
            'pqxx','experiment_admin_worker_outcome','worker_command_line','UPDATE') OR
       has_column_privilege(
            'pqxx','experiment_admin_worker_outcome','source_pause_request_id','UPDATE') OR
       NOT has_column_privilege(
            'pqxx','experiment_admin_worker_outcome','outcome_status','UPDATE') THEN
        RAISE EXCEPTION 'worker outcome audit update privileges unsafe';
    END IF;
END
$migration_test$;
