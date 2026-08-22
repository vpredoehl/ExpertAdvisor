DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'experiment'
          AND column_name =
              'continuation_policy_min_profit_actionable_count'
    ) THEN
        RAISE EXCEPTION 'missing profitability actionable-count policy column';
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'experiment'
          AND column_name =
              'continuation_policy_min_profit_aggregate_log_return_sum'
    ) THEN
        RAISE EXCEPTION 'missing aggregate profitability policy column';
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'experiment'
          AND column_name =
              'continuation_policy_min_profit_average_log_return'
    ) THEN
        RAISE EXCEPTION 'missing average profitability policy column';
    END IF;
END $$;

INSERT INTO experiment (experiment_id) VALUES (1);

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM experiment
        WHERE experiment_id = 1
          AND (
              continuation_policy_min_profit_actionable_count IS NOT NULL
              OR continuation_policy_min_profit_aggregate_log_return_sum IS NOT NULL
              OR continuation_policy_min_profit_average_log_return IS NOT NULL
          )
    ) THEN
        RAISE EXCEPTION 'legacy/default policy fields were not null';
    END IF;
END $$;

DO $$
BEGIN
    BEGIN
        UPDATE experiment
        SET continuation_policy_min_profit_actionable_count = 0
        WHERE experiment_id = 1;
        RAISE EXCEPTION 'zero actionable minimum was accepted';
    EXCEPTION WHEN check_violation THEN
        NULL;
    END;

    BEGIN
        UPDATE experiment
        SET continuation_policy_min_profit_aggregate_log_return_sum =
            'NaN'::double precision
        WHERE experiment_id = 1;
        RAISE EXCEPTION 'NaN aggregate minimum was accepted';
    EXCEPTION WHEN check_violation THEN
        NULL;
    END;

    BEGIN
        UPDATE experiment
        SET continuation_policy_min_profit_average_log_return =
            'Infinity'::double precision
        WHERE experiment_id = 1;
        RAISE EXCEPTION 'infinite average minimum was accepted';
    EXCEPTION WHEN check_violation THEN
        NULL;
    END;
END $$;

UPDATE experiment
SET continuation_policy_last_decision = 'rejected_profitability',
    continuation_policy_min_profit_actionable_count = 10,
    continuation_policy_min_profit_aggregate_log_return_sum =
        -0.01,
    continuation_policy_min_profit_average_log_return =
        0.001
WHERE experiment_id = 1;

INSERT INTO experiment_continuation_decision (decision)
VALUES ('rejected_profitability');

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM experiment
        WHERE experiment_id = 1
          AND continuation_policy_last_decision = 'rejected_profitability'
          AND continuation_policy_min_profit_actionable_count = 10
    ) THEN
        RAISE EXCEPTION 'valid profitability policy did not round trip';
    END IF;
END $$;
