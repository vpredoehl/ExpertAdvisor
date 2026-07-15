\set ON_ERROR_STOP on

BEGIN;
SET TRANSACTION READ WRITE;

CREATE SCHEMA step4_scoring_migration_test;
SET LOCAL search_path TO step4_scoring_migration_test, public;

-- Minimal pre-Step-4 dependency shape. The test executes the production
-- migrations verbatim inside an isolated schema and rolls everything back.
CREATE TABLE experiment (
    experiment_id bigserial PRIMARY KEY
);
CREATE TABLE experiment_recommendation_scan (
    recommendation_scan_id bigserial PRIMARY KEY
);
CREATE TABLE experiment_recommendation (
    recommendation_id bigserial PRIMARY KEY,
    recommendation_scan_id bigint
        REFERENCES experiment_recommendation_scan(recommendation_scan_id),
    note text
);

\ir ../Database/migrations/028_experiment_recommendation_scoring.sql

INSERT INTO experiment DEFAULT VALUES;
INSERT INTO experiment_recommendation_scan DEFAULT VALUES;
INSERT INTO experiment_recommendation_scan DEFAULT VALUES;
INSERT INTO experiment_recommendation_scan DEFAULT VALUES;

-- This is the exact legacy state migration 029 must preserve. Migration 028's
-- NOT VALID CHECK accepts the NULL result and must not be mistaken for new-row
-- enforcement.
INSERT INTO experiment_recommendation (
    recommendation_scan_id, source_rank, note)
VALUES
    (NULL, NULL, 'legacy_unrelated'),
    (NULL, NULL, 'legacy_transition'),
    (NULL, NULL, 'legacy_untouched');

\ir ../Database/migrations/029_experiment_recommendation_source_rank_enforcement.sql
\ir ../Database/migrations/030_experiment_recommendation_source_rank_transition_enforcement.sql
\ir ../Database/migrations/031_experiment_recommendation_scan_provenance_enforcement.sql

UPDATE experiment_recommendation
SET note = 'legacy_updated'
WHERE note = 'legacy_unrelated';

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM experiment_recommendation
        WHERE note = 'legacy_updated'
          AND recommendation_scan_id IS NULL
          AND source_rank IS NULL
    ) THEN
        RAISE EXCEPTION 'legacy_source_rank_row_was_not_preserved';
    END IF;
END $$;

CREATE FUNCTION expect_source_rank_check_violation(command text, label text)
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    observed_sqlstate text;
BEGIN
    BEGIN
        EXECUTE command;
    EXCEPTION WHEN OTHERS THEN
        GET STACKED DIAGNOSTICS observed_sqlstate = RETURNED_SQLSTATE;
        IF observed_sqlstate = '23514' THEN
            RETURN;
        END IF;
        RAISE EXCEPTION '% returned SQLSTATE %, expected 23514',
            label, observed_sqlstate;
    END;
    RAISE EXCEPTION '% was accepted, expected SQLSTATE 23514', label;
END;
$$;

CREATE FUNCTION assert_source_rank_fixture_state(
    fixture_note text, expected_scan_id bigint, expected_source_rank integer)
RETURNS void
LANGUAGE plpgsql
AS $$
BEGIN
    IF (SELECT count(*) FROM experiment_recommendation
        WHERE note = fixture_note
          AND recommendation_scan_id IS NOT DISTINCT FROM expected_scan_id
          AND source_rank IS NOT DISTINCT FROM expected_source_rank) <> 1 THEN
        RAISE EXCEPTION '% state changed; expected scan %, rank %',
            fixture_note, expected_scan_id, expected_source_rank;
    END IF;
END;
$$;

SELECT expect_source_rank_check_violation(
    $sql$INSERT INTO experiment_recommendation
        (recommendation_scan_id, source_rank, note)
        VALUES (1, NULL, 'missing')$sql$,
    'null_source_rank_insert');
SELECT expect_source_rank_check_violation(
    $sql$INSERT INTO experiment_recommendation
        (recommendation_scan_id, source_rank, note)
        VALUES (1, 0, 'zero')$sql$,
    'zero_source_rank_insert');
SELECT expect_source_rank_check_violation(
    $sql$INSERT INTO experiment_recommendation
        (recommendation_scan_id, source_rank, note)
        VALUES (1, -1, 'negative')$sql$,
    'negative_source_rank_insert');

INSERT INTO experiment_recommendation (
    recommendation_scan_id, source_rank, note)
VALUES (1, 1, 'positive');

SELECT expect_source_rank_check_violation(
    $sql$UPDATE experiment_recommendation
        SET recommendation_scan_id = 1
        WHERE note = 'legacy_transition'$sql$,
    'null_to_nonnull_with_null_rank');
SELECT assert_source_rank_fixture_state('legacy_transition', NULL, NULL);
SELECT expect_source_rank_check_violation(
    $sql$UPDATE experiment_recommendation
        SET recommendation_scan_id = 1, source_rank = 0
        WHERE note = 'legacy_transition'$sql$,
    'null_to_nonnull_with_zero_rank');
SELECT assert_source_rank_fixture_state('legacy_transition', NULL, NULL);
SELECT expect_source_rank_check_violation(
    $sql$UPDATE experiment_recommendation
        SET recommendation_scan_id = 1, source_rank = -1
        WHERE note = 'legacy_transition'$sql$,
    'null_to_nonnull_with_negative_rank');
SELECT assert_source_rank_fixture_state('legacy_transition', NULL, NULL);

UPDATE experiment_recommendation
SET recommendation_scan_id = 1, source_rank = 7
WHERE note = 'legacy_transition';
SELECT assert_source_rank_fixture_state('legacy_transition', 1, 7);

INSERT INTO experiment_recommendation (
    recommendation_scan_id, source_rank, note)
VALUES (1, 5, 'reassociation');

UPDATE experiment_recommendation
SET recommendation_scan_id = 2
WHERE note = 'reassociation';
SELECT assert_source_rank_fixture_state('reassociation', 2, 5);

UPDATE experiment_recommendation
SET recommendation_scan_id = 3, source_rank = 6
WHERE note = 'reassociation';
SELECT assert_source_rank_fixture_state('reassociation', 3, 6);

SELECT expect_source_rank_check_violation(
    $sql$UPDATE experiment_recommendation
        SET recommendation_scan_id = 1, source_rank = NULL
        WHERE note = 'reassociation'$sql$,
    'nonnull_to_different_nonnull_with_null_rank');
SELECT assert_source_rank_fixture_state('reassociation', 3, 6);
SELECT expect_source_rank_check_violation(
    $sql$UPDATE experiment_recommendation
        SET recommendation_scan_id = 1, source_rank = 0
        WHERE note = 'reassociation'$sql$,
    'nonnull_to_different_nonnull_with_zero_rank');
SELECT assert_source_rank_fixture_state('reassociation', 3, 6);
SELECT expect_source_rank_check_violation(
    $sql$UPDATE experiment_recommendation
        SET recommendation_scan_id = 1, source_rank = -1
        WHERE note = 'reassociation'$sql$,
    'nonnull_to_different_nonnull_with_negative_rank');
SELECT assert_source_rank_fixture_state('reassociation', 3, 6);

SELECT expect_source_rank_check_violation(
    $sql$UPDATE experiment_recommendation
        SET recommendation_scan_id = NULL
        WHERE note = 'reassociation'$sql$,
    'nonnull_to_null_retaining_positive_rank');
SELECT assert_source_rank_fixture_state('reassociation', 3, 6);
SELECT expect_source_rank_check_violation(
    $sql$UPDATE experiment_recommendation
        SET recommendation_scan_id = NULL, source_rank = NULL
        WHERE note = 'reassociation'$sql$,
    'nonnull_to_null_with_null_rank');
SELECT assert_source_rank_fixture_state('reassociation', 3, 6);
SELECT expect_source_rank_check_violation(
    $sql$UPDATE experiment_recommendation
        SET recommendation_scan_id = NULL, source_rank = 0
        WHERE note = 'reassociation'$sql$,
    'nonnull_to_null_with_zero_rank');
SELECT assert_source_rank_fixture_state('reassociation', 3, 6);
SELECT expect_source_rank_check_violation(
    $sql$UPDATE experiment_recommendation
        SET recommendation_scan_id = NULL, source_rank = -1
        WHERE note = 'reassociation'$sql$,
    'nonnull_to_null_with_negative_rank');
SELECT assert_source_rank_fixture_state('reassociation', 3, 6);

SELECT expect_source_rank_check_violation(
    $sql$UPDATE experiment_recommendation
        SET source_rank = NULL
        WHERE note = 'positive'$sql$,
    'persisted_source_rank_clear');
SELECT assert_source_rank_fixture_state('positive', 1, 1);
SELECT expect_source_rank_check_violation(
    $sql$UPDATE experiment_recommendation
        SET source_rank = 0
        WHERE note = 'positive'$sql$,
    'persisted_source_rank_zero');
SELECT assert_source_rank_fixture_state('positive', 1, 1);
SELECT expect_source_rank_check_violation(
    $sql$UPDATE experiment_recommendation
        SET source_rank = -1
        WHERE note = 'positive'$sql$,
    'persisted_source_rank_negative');
SELECT assert_source_rank_fixture_state('positive', 1, 1);

UPDATE experiment_recommendation
SET recommendation_scan_id = 2
WHERE note = 'positive';

-- Direct SQL rerun is idempotent in addition to migration-runner checksum
-- idempotency.
\ir ../Database/migrations/031_experiment_recommendation_scan_provenance_enforcement.sql

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM experiment_recommendation
        WHERE note = 'positive'
          AND recommendation_scan_id = 2
          AND source_rank = 1
    ) THEN
        RAISE EXCEPTION 'valid_scan_reassociation_failed';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM experiment_recommendation
        WHERE note = 'legacy_untouched'
          AND recommendation_scan_id IS NULL
          AND source_rank IS NULL
    ) THEN
        RAISE EXCEPTION 'legacy_untouched_row_changed';
    END IF;
    IF (SELECT count(*) FROM pg_trigger
        WHERE tgname = 'experiment_recommendation_source_rank_enforce_trigger'
          AND tgrelid = 'experiment_recommendation'::regclass
          AND NOT tgisinternal) <> 1 THEN
        RAISE EXCEPTION 'source_rank_trigger_not_idempotent';
    END IF;
END $$;

ROLLBACK;
