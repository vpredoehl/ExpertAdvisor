-- Phase 4A: make the only historically active LSTM training objective an
-- explicit, immutable-by-identity experiment contract. This migration does
-- not activate an auxiliary loss or alter any model/training data.

CREATE OR REPLACE FUNCTION training_objective_tagged_fnv1a64(
    canonical_value text)
RETURNS text
LANGUAGE plpgsql
IMMUTABLE
STRICT
PARALLEL SAFE
SECURITY INVOKER
AS $$
DECLARE
    canonical_bytes bytea := convert_to(canonical_value, 'UTF8');
    hash_value numeric := 14695981039346656037;
    byte_index integer;
    low_byte integer;
    high_word bigint;
    low_word bigint;
BEGIN
    IF octet_length(canonical_bytes) > 0 THEN
        FOR byte_index IN 0..octet_length(canonical_bytes) - 1 LOOP
            low_byte := mod(hash_value, 256)::integer #
                get_byte(canonical_bytes, byte_index);
            hash_value := hash_value - mod(hash_value, 256) + low_byte;
            hash_value := mod(
                hash_value * 1099511628211, 18446744073709551616);
        END LOOP;
    END IF;
    high_word := trunc(hash_value / 4294967296)::bigint;
    low_word := mod(hash_value, 4294967296)::bigint;
    RETURN 'fnv1a64:' || lpad(to_hex(high_word), 8, '0') ||
        lpad(to_hex(low_word), 8, '0');
END;
$$;

ALTER TABLE experiment
    ADD COLUMN IF NOT EXISTS training_objective_id text NOT NULL
        DEFAULT 'legacy_first_hit_weighted_ce_v1',
    ADD COLUMN IF NOT EXISTS training_objective_version integer NOT NULL
        DEFAULT 1,
    ADD COLUMN IF NOT EXISTS loss_definition_version integer NOT NULL
        DEFAULT 1,
    ADD COLUMN IF NOT EXISTS training_objective_canonical text NOT NULL
        DEFAULT 'training_objective_configuration_v1;schema_version=1;objective_id=legacy_first_hit_weighted_ce_v1;objective_family=up_neutral_down_first_hit_classification;objective_version=1;loss_definition_version=1;mode=legacy_first_hit_classification;classification_loss=true_class_weighted_softmax_cross_entropy_v1;classification_target=up_neutral_down_return_high_low_first_hit_strict_threshold_up_tie_v1;class_index_order=down_0_neutral_1_up_2;class_weight_semantics=true_class_weight_multiplies_loss_and_all_logit_components_v1;class_weight_down=1;class_weight_neutral=1;class_weight_up=1;softmax_loss_probability_floor=1e-12;classification_logit_gradient_scale=0.1;shared_core_classification_gradient_scale=4;internal_loss_normalization=weighted_loss_sum_divided_by_true_class_weight_sum_v1;calculate_batch_return_normalization=weighted_loss_sum_divided_by_example_count_v1;gradient_normalization=all_calculate_batch_gradients_divided_by_true_class_weight_sum_v1;batch_window_boundary=overlapping_windows_do_not_cross_outer_tensor_batch_v1;optimizer_family=sgd;optimizer_update=parameter_minus_learning_rate_times_gradient_v1;learning_rate_contract=base_rate_and_parameter_group_multipliers_persisted_in_training_config_v1;gradient_clipping_mode=componentwise_after_normalization_before_update;gradient_clip_threshold=10;nonfinite_gradient_policy=skip_parameter_update_v1;weight_decay=none;gradient_accumulation_precision=core_gradient_accumulation_double_head_gradient_accumulation_float_loss_accumulation_double_v1;auxiliary_loss_mode=disabled;auxiliary_loss_coefficient=0;regression_target_definition=NULL;regression_normalization_identity=NULL;robust_loss_definition=NULL;robust_loss_delta=NULL;target_clipping_definition=none;shared_gradient_combination=classification_only_v1;',
    ADD COLUMN IF NOT EXISTS training_objective_hash text NOT NULL
        DEFAULT 'fnv1a64:65818f2e1fa1a324',
    ADD COLUMN IF NOT EXISTS auxiliary_loss_mode text NOT NULL
        DEFAULT 'disabled',
    ADD COLUMN IF NOT EXISTS auxiliary_loss_coefficient double precision NOT NULL
        DEFAULT 0.0,
    ADD COLUMN IF NOT EXISTS regression_target_definition text,
    ADD COLUMN IF NOT EXISTS regression_normalization_identity text,
    ADD COLUMN IF NOT EXISTS robust_loss_definition text,
    ADD COLUMN IF NOT EXISTS robust_loss_delta double precision,
    ADD COLUMN IF NOT EXISTS target_clipping_definition text NOT NULL
        DEFAULT 'none',
    ADD COLUMN IF NOT EXISTS objective_normalization_identity text NOT NULL
        DEFAULT 'weighted_loss_sum_by_weight_sum_gradients__calculate_batch_return_by_example_count_v1';

ALTER TABLE experiment
    DROP CONSTRAINT IF EXISTS experiment_training_objective_provenance_check;
ALTER TABLE experiment
    ADD CONSTRAINT experiment_training_objective_provenance_check CHECK (
        training_objective_id <> '' AND
        training_objective_version > 0 AND
        loss_definition_version > 0 AND
        training_objective_hash =
            training_objective_tagged_fnv1a64(training_objective_canonical) AND
        auxiliary_loss_mode <> '' AND
        objective_normalization_identity <> '' AND
        target_clipping_definition <> '' AND
        (auxiliary_loss_mode <> 'disabled' OR (
            auxiliary_loss_coefficient = 0.0 AND
            regression_target_definition IS NULL AND
            regression_normalization_identity IS NULL AND
            robust_loss_definition IS NULL AND
            robust_loss_delta IS NULL
        ))
    );

COMMENT ON COLUMN experiment.training_objective_id IS
    'Stable human-readable objective family/version. Historical rows resolve to legacy_first_hit_weighted_ce_v1.';
COMMENT ON COLUMN experiment.training_objective_version IS
    'Version of the named training objective; historical and Phase 4A legacy rows use 1.';
COMMENT ON COLUMN experiment.loss_definition_version IS
    'Version of loss/gradient mathematics within the objective; historical and Phase 4A legacy rows use 1.';
COMMENT ON COLUMN experiment.training_objective_canonical IS
    'Authoritative complete objective/loss/gradient/optimizer semantic contract; exact equality is required for resume.';
COMMENT ON COLUMN experiment.training_objective_hash IS
    'FNV-1a-64 accelerator over training_objective_canonical; canonical text remains authoritative.';
COMMENT ON COLUMN experiment.auxiliary_loss_mode IS
    'Phase 4A default is disabled. A non-disabled mode requires a later migration and runtime implementation.';
COMMENT ON COLUMN experiment.auxiliary_loss_coefficient IS
    'Auxiliary loss coefficient; exactly zero for every Phase 4A legacy experiment.';
COMMENT ON COLUMN experiment.regression_target_definition IS
    'Nullable future auxiliary regression target identity; NULL means no regression objective.';
COMMENT ON COLUMN experiment.regression_normalization_identity IS
    'Nullable future regression-target normalization identity; NULL for the legacy objective.';
COMMENT ON COLUMN experiment.robust_loss_definition IS
    'Nullable future robust auxiliary loss identity; NULL for the legacy objective.';
COMMENT ON COLUMN experiment.robust_loss_delta IS
    'Nullable future robust-loss delta; NULL for the legacy objective.';
COMMENT ON COLUMN experiment.target_clipping_definition IS
    'Training-target clipping or winsorization identity; none for the legacy objective.';
COMMENT ON COLUMN experiment.objective_normalization_identity IS
    'Explicit legacy distinction between internally weight-normalized loss/gradients and CalculateBatch returned loss.';

-- The objective hash joins the existing practical experiment identity. All
-- historical rows receive the same legacy default, so their prior collision
-- behavior is retained. Canonical equality is also checked by application
-- duplicate detection after hash filtering.
DROP INDEX IF EXISTS experiment_unique_identity_uidx;

CREATE UNIQUE INDEX experiment_unique_identity_uidx
    ON experiment (
        symbol, prediction_horizon, c_next_threshold,
        COALESCE(core_lr_mult, '-infinity'::double precision),
        COALESCE(head_lr_mult, '-infinity'::double precision),
        target_epochs, checkpoint_interval, train_start, train_end,
        COALESCE(infer_start, '-infinity'::timestamptz),
        COALESCE(infer_end, '-infinity'::timestamptz),
        COALESCE(resume_model_id, -1), donchian20_mode,
        donchian_lookback, feature_warmup_scope, feature_ablation_mask,
        resume_expand_input_width, training_objective_hash, duplicate_nonce
    ) WHERE status <> 'cancelled';

GRANT EXECUTE ON FUNCTION training_objective_tagged_fnv1a64(text) TO pqxx;
GRANT SELECT, INSERT, UPDATE, DELETE ON experiment TO pqxx;
