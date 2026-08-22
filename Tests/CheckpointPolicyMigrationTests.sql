DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM experiment_checkpoint_decision
        WHERE checkpoint_eval_id=100
          AND identity_status='legacy'
          AND policy_revision IS NULL
          AND policy_hash IS NULL
          AND evidence_watermark IS NULL
          AND reason='legacy preserved'
    ) THEN
        RAISE EXCEPTION 'historical decision was not preserved as legacy';
    END IF;
    IF (SELECT checkpoint_policy_revision FROM experiment WHERE experiment_id=1) <> 1 THEN
        RAISE EXCEPTION 'legacy experiment did not receive safe revision 1';
    END IF;
END $$;

INSERT INTO experiment_checkpoint_decision(
    checkpoint_eval_id,parent_experiment_id,checkpoint_epoch,
    checkpoint_model_id,analysis_id,inference_eval_result_id,
    policy_revision,policy_hash,evidence_watermark,
    rank_population_watermark,identity_status,decision,reason,
    requested_stop_epoch)
VALUES
    (100,1,20,10,200,300,1,'1111111111111111','aaaaaaaaaaaaaaaa',
     'rank-one','active','stop_requested','version one',40),
    (100,1,20,10,201,301,2,'2222222222222222','bbbbbbbbbbbbbbbb',
     'rank-two','active','continue','version two',NULL);

DO $$
DECLARE
    row_count integer;
BEGIN
    SELECT count(*) INTO row_count
    FROM experiment_checkpoint_decision WHERE checkpoint_eval_id=100;
    IF row_count <> 3 THEN
        RAISE EXCEPTION 'append-only identities were not preserved';
    END IF;

    BEGIN
        INSERT INTO experiment_checkpoint_decision(
            checkpoint_eval_id,parent_experiment_id,checkpoint_epoch,
            checkpoint_model_id,analysis_id,inference_eval_result_id,
            policy_revision,policy_hash,evidence_watermark,
            identity_status,decision,reason)
        VALUES (100,1,20,10,201,301,2,'2222222222222222',
                'bbbbbbbbbbbbbbbb','active','continue','duplicate');
        RAISE EXCEPTION 'duplicate semantic decision identity was accepted';
    EXCEPTION WHEN unique_violation THEN
        NULL;
    END;

    BEGIN
        UPDATE experiment_checkpoint_decision SET decision='continue'
        WHERE policy_revision=1;
        RAISE EXCEPTION 'immutable decision meaning was overwritten';
    EXCEPTION WHEN SQLSTATE '55000' THEN
        NULL;
    END;
END $$;

UPDATE experiment_checkpoint_decision
SET identity_status='superseded', superseded_at=clock_timestamp(),
    superseded_reason='policy_or_evidence_changed'
WHERE policy_revision=2;

UPDATE experiment
SET checkpoint_policy_hash='1111111111111111',
    checkpoint_policy_last_decision_id=(
        SELECT checkpoint_decision_id
        FROM experiment_checkpoint_decision WHERE policy_revision=1)
WHERE experiment_id=1;

BEGIN;
UPDATE experiment
SET stop_after_checkpoint_epoch=40,
    checkpoint_policy_stop_decision_id=(
        SELECT checkpoint_decision_id
        FROM experiment_checkpoint_decision WHERE policy_revision=1)
WHERE experiment_id=1;
UPDATE experiment_checkpoint_decision
SET identity_status='action_applied', stop_request_applied=true,
    stop_request_applied_at=clock_timestamp(),
    stop_action_worker_attempt_id=400
WHERE policy_revision=1;
COMMIT;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM experiment e
        JOIN experiment_checkpoint_decision d
          ON d.checkpoint_decision_id=e.checkpoint_policy_stop_decision_id
        WHERE e.experiment_id=1
          AND e.stop_after_checkpoint_epoch=40
          AND d.stop_request_applied
          AND d.identity_status='action_applied'
          AND d.stop_action_worker_attempt_id=400
    ) THEN
        RAISE EXCEPTION 'atomic stop attribution did not round trip';
    END IF;

    BEGIN
        UPDATE experiment_checkpoint_decision
        SET stop_request_applied=false,
            stop_request_applied_at=NULL,
            stop_action_worker_attempt_id=NULL,
            identity_status='active'
        WHERE policy_revision=1;
        RAISE EXCEPTION 'terminal stop attribution was reversible';
    EXCEPTION WHEN SQLSTATE '55000' THEN
        NULL;
    END;

    BEGIN
        UPDATE experiment_checkpoint_decision
        SET identity_status='active', superseded_at=NULL,
            superseded_reason=NULL, superseded_by_decision_id=NULL
        WHERE policy_revision=2;
        RAISE EXCEPTION 'superseded decision was revived';
    EXCEPTION WHEN SQLSTATE '55000' THEN
        NULL;
    END;

    BEGIN
        UPDATE experiment_checkpoint_decision
        SET identity_status='action_applied', stop_request_applied=true,
            stop_request_applied_at=clock_timestamp(),
            stop_action_worker_attempt_id=400,
            superseded_at=NULL, superseded_reason=NULL,
            superseded_by_decision_id=NULL
        WHERE policy_revision=2;
        RAISE EXCEPTION 'superseded decision became action_applied';
    EXCEPTION WHEN SQLSTATE '55000' THEN
        NULL;
    END;

    BEGIN
        UPDATE experiment_checkpoint_decision
        SET superseded_reason='rewritten terminal metadata'
        WHERE policy_revision=2;
        RAISE EXCEPTION 'superseded terminal metadata was mutable';
    EXCEPTION WHEN SQLSTATE '55000' THEN
        NULL;
    END;

    BEGIN
        UPDATE experiment_checkpoint_decision
        SET identity_status='superseded', stop_request_applied=false,
            stop_request_applied_at=NULL, stop_action_worker_attempt_id=NULL,
            superseded_at=clock_timestamp(), superseded_reason='demoted'
        WHERE policy_revision=1;
        RAISE EXCEPTION 'action_applied decision was demoted';
    EXCEPTION WHEN SQLSTATE '55000' THEN
        NULL;
    END;

    BEGIN
        UPDATE experiment_checkpoint_decision
        SET stop_request_applied_at=clock_timestamp()
        WHERE policy_revision=1;
        RAISE EXCEPTION 'action_applied terminal metadata was mutable';
    EXCEPTION WHEN SQLSTATE '55000' THEN
        NULL;
    END;

    BEGIN
        UPDATE experiment_checkpoint_decision
        SET identity_status='active'
        WHERE identity_status='legacy';
        RAISE EXCEPTION 'legacy decision became active';
    EXCEPTION WHEN SQLSTATE '55000' THEN
        NULL;
    END;
END $$;

-- Persistence authority uses checkpoint epoch first and checkpoint_eval_id as
-- the deterministic equal-epoch tie-breaker.  A changed semantic identity for
-- the same evaluation is also superseded even though its authority key ties.
INSERT INTO experiment(experiment_id) VALUES (2), (3);
INSERT INTO experiment_checkpoint_eval VALUES
    (101, 1, 40, 11),
    (200, 2, 20, 10),
    (201, 2, 40, 10),
    (202, 2, 40, 11),
    (300, 3, 40, 10),
    (301, 3, 40, 11);
INSERT INTO experiment_analysis_result VALUES
    (202), (203), (204), (205), (206), (207), (208);
INSERT INTO inference_eval_result VALUES
    (302), (303), (304), (305), (306), (307), (308);

INSERT INTO experiment_checkpoint_decision(
    checkpoint_eval_id,parent_experiment_id,checkpoint_epoch,
    checkpoint_model_id,analysis_id,inference_eval_result_id,
    policy_revision,policy_hash,evidence_watermark,
    rank_population_watermark,identity_status,decision,reason,
    requested_stop_epoch)
VALUES
    (200,2,20,10,202,302,1,'3333333333333333','cccccccccccccccc',
     'epoch-20','active','continue','epoch 20',NULL),
    (201,2,40,10,203,303,1,'3333333333333333','dddddddddddddddd',
     'epoch-40-a','active','continue','epoch 40 eval 201',NULL),
    (300,3,40,10,206,306,1,'5555555555555555','1111111111111111',
     'fence-old','active','stop_requested','older fence candidate',60);

UPDATE experiment_checkpoint_decision SET
    identity_status='superseded', superseded_at=clock_timestamp(),
    superseded_reason=CASE
      WHEN checkpoint_eval_id=201 THEN 'policy_or_evidence_changed'
      ELSE 'newer_checkpoint_decision_became_authoritative' END,
    superseded_by_decision_id=(
      SELECT checkpoint_decision_id FROM experiment_checkpoint_decision
      WHERE checkpoint_eval_id=201 AND evidence_watermark='dddddddddddddddd')
WHERE parent_experiment_id=2
  AND checkpoint_decision_id<>(
      SELECT checkpoint_decision_id FROM experiment_checkpoint_decision
      WHERE checkpoint_eval_id=201 AND evidence_watermark='dddddddddddddddd')
  AND identity_status='active'
  AND (checkpoint_epoch<40
       OR (checkpoint_epoch=40 AND checkpoint_eval_id<201)
       OR checkpoint_eval_id=201);

DO $$
BEGIN
    IF (SELECT identity_status FROM experiment_checkpoint_decision
        WHERE checkpoint_eval_id=200) <> 'superseded'
       OR (SELECT identity_status FROM experiment_checkpoint_decision
           WHERE checkpoint_eval_id=201) <> 'active'
       OR (SELECT identity_status FROM experiment_checkpoint_decision
           WHERE checkpoint_eval_id=300) <> 'active'
    THEN
        RAISE EXCEPTION 'different-epoch authority supersession failed';
    END IF;
END $$;

INSERT INTO experiment_checkpoint_decision(
    checkpoint_eval_id,parent_experiment_id,checkpoint_epoch,
    checkpoint_model_id,analysis_id,inference_eval_result_id,
    policy_revision,policy_hash,evidence_watermark,
    rank_population_watermark,identity_status,decision,reason)
VALUES
    (202,2,40,11,204,304,1,'3333333333333333','eeeeeeeeeeeeeeee',
     'epoch-40-b','active','continue','epoch 40 eval 202');

UPDATE experiment_checkpoint_decision SET
    identity_status='superseded', superseded_at=clock_timestamp(),
    superseded_reason=CASE
      WHEN checkpoint_eval_id=202 THEN 'policy_or_evidence_changed'
      ELSE 'newer_checkpoint_decision_became_authoritative' END,
    superseded_by_decision_id=(
      SELECT checkpoint_decision_id FROM experiment_checkpoint_decision
      WHERE checkpoint_eval_id=202 AND evidence_watermark='eeeeeeeeeeeeeeee')
WHERE parent_experiment_id=2
  AND checkpoint_decision_id<>(
      SELECT checkpoint_decision_id FROM experiment_checkpoint_decision
      WHERE checkpoint_eval_id=202 AND evidence_watermark='eeeeeeeeeeeeeeee')
  AND identity_status='active'
  AND (checkpoint_epoch<40
       OR (checkpoint_epoch=40 AND checkpoint_eval_id<202)
       OR checkpoint_eval_id=202);

DO $$
BEGIN
    IF (SELECT identity_status FROM experiment_checkpoint_decision
        WHERE checkpoint_eval_id=201) <> 'superseded'
       OR (SELECT identity_status FROM experiment_checkpoint_decision
           WHERE checkpoint_eval_id=202
             AND evidence_watermark='eeeeeeeeeeeeeeee') <> 'active'
       OR (SELECT count(*) FROM experiment_checkpoint_decision
           WHERE parent_experiment_id=2 AND identity_status='active') <> 1
    THEN
        RAISE EXCEPTION 'equal-epoch eval-id authority supersession failed';
    END IF;
END $$;

INSERT INTO experiment_checkpoint_decision(
    checkpoint_eval_id,parent_experiment_id,checkpoint_epoch,
    checkpoint_model_id,analysis_id,inference_eval_result_id,
    policy_revision,policy_hash,evidence_watermark,
    rank_population_watermark,identity_status,decision,reason)
VALUES
    (202,2,40,11,205,305,2,'4444444444444444','ffffffffffffffff',
     'epoch-40-b-revised','active','continue','changed semantic identity');

UPDATE experiment_checkpoint_decision SET
    identity_status='superseded', superseded_at=clock_timestamp(),
    superseded_reason=CASE
      WHEN checkpoint_eval_id=202 THEN 'policy_or_evidence_changed'
      ELSE 'newer_checkpoint_decision_became_authoritative' END,
    superseded_by_decision_id=(
      SELECT checkpoint_decision_id FROM experiment_checkpoint_decision
      WHERE checkpoint_eval_id=202 AND evidence_watermark='ffffffffffffffff')
WHERE parent_experiment_id=2
  AND checkpoint_decision_id<>(
      SELECT checkpoint_decision_id FROM experiment_checkpoint_decision
      WHERE checkpoint_eval_id=202 AND evidence_watermark='ffffffffffffffff')
  AND identity_status='active'
  AND (checkpoint_epoch<40
       OR (checkpoint_epoch=40 AND checkpoint_eval_id<202)
       OR checkpoint_eval_id=202);

INSERT INTO experiment_checkpoint_decision(
    checkpoint_eval_id,parent_experiment_id,checkpoint_epoch,
    checkpoint_model_id,analysis_id,inference_eval_result_id,
    policy_revision,policy_hash,evidence_watermark,
    rank_population_watermark,identity_status,decision,reason)
VALUES
    (202,2,40,11,205,305,2,'4444444444444444','ffffffffffffffff',
     'epoch-40-b-revised','active','continue','changed semantic identity')
ON CONFLICT (checkpoint_eval_id,policy_revision,policy_hash,evidence_watermark)
WHERE policy_revision IS NOT NULL
  AND policy_hash IS NOT NULL
  AND evidence_watermark IS NOT NULL
DO NOTHING;

DO $$
BEGIN
    IF (SELECT identity_status FROM experiment_checkpoint_decision
        WHERE checkpoint_eval_id=202
          AND evidence_watermark='eeeeeeeeeeeeeeee') <> 'superseded'
       OR (SELECT identity_status FROM experiment_checkpoint_decision
           WHERE checkpoint_eval_id=202
             AND evidence_watermark='ffffffffffffffff') <> 'active'
       OR (SELECT count(*) FROM experiment_checkpoint_decision
           WHERE checkpoint_eval_id=202) <> 2
       OR (SELECT count(*) FROM experiment_checkpoint_decision
           WHERE parent_experiment_id=2 AND identity_status='active') <> 1
    THEN
        RAISE EXCEPTION 'semantic supersession or idempotent reuse failed';
    END IF;
END $$;

-- The stop fence independently rejects an older checkpoint whenever a newer
-- active/action-applied authority key exists, even if anomalous rows coexist.
INSERT INTO experiment_checkpoint_decision(
    checkpoint_eval_id,parent_experiment_id,checkpoint_epoch,
    checkpoint_model_id,analysis_id,inference_eval_result_id,
    policy_revision,policy_hash,evidence_watermark,
    rank_population_watermark,identity_status,decision,reason,
    requested_stop_epoch)
VALUES
    (301,3,40,11,207,307,1,'5555555555555555','2222222222222222',
     'fence-new','active','continue','newer authority',NULL);

DO $$
DECLARE
    old_decision_id bigint;
BEGIN
    SELECT checkpoint_decision_id INTO old_decision_id
    FROM experiment_checkpoint_decision WHERE checkpoint_eval_id=300;
    IF EXISTS (
        SELECT 1
        FROM experiment_checkpoint_decision current_decision
        WHERE current_decision.checkpoint_decision_id=old_decision_id
          AND current_decision.identity_status='active'
          AND NOT EXISTS (
              SELECT 1 FROM experiment_checkpoint_decision newer
              WHERE newer.parent_experiment_id=3
                AND newer.identity_status IN ('active','action_applied')
                AND (newer.checkpoint_epoch>current_decision.checkpoint_epoch
                     OR (newer.checkpoint_epoch=current_decision.checkpoint_epoch
                         AND newer.checkpoint_eval_id>
                             current_decision.checkpoint_eval_id)))
    ) THEN
        RAISE EXCEPTION 'newer same-epoch decision did not fence old stop';
    END IF;
END $$;

-- A later evaluation after terminal stop attribution remains observational and
-- cannot replace or mutate the applied stop decision.
INSERT INTO experiment_checkpoint_decision(
    checkpoint_eval_id,parent_experiment_id,checkpoint_epoch,
    checkpoint_model_id,analysis_id,inference_eval_result_id,
    policy_revision,policy_hash,evidence_watermark,
    rank_population_watermark,identity_status,superseded_at,
    superseded_reason,decision,reason)
VALUES
    (101,1,40,11,208,308,1,'1111111111111111','3333333333333333',
     'post-stop', 'superseded',clock_timestamp(),
     'stop_action_already_applied','continue','later observation');

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM experiment e
        JOIN experiment_checkpoint_decision d
          ON d.checkpoint_decision_id=e.checkpoint_policy_stop_decision_id
        WHERE e.experiment_id=1
          AND e.stop_after_checkpoint_epoch=40
          AND d.checkpoint_eval_id=100
          AND d.identity_status='action_applied'
          AND d.stop_request_applied
          AND d.stop_action_worker_attempt_id=400)
       OR (SELECT identity_status FROM experiment_checkpoint_decision
           WHERE checkpoint_eval_id=101) <> 'superseded'
       OR (SELECT identity_status FROM experiment_checkpoint_decision
           WHERE checkpoint_eval_id=100 AND policy_revision IS NULL) <> 'legacy'
    THEN
        RAISE EXCEPTION 'terminal stop attribution or legacy history changed';
    END IF;
END $$;
