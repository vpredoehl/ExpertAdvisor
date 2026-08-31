BEGIN;

CREATE TABLE migration_runner_pre_ledger_effect (
    effect_id integer PRIMARY KEY
);

ALTER TABLE schema_migrations
    ADD CONSTRAINT migration_runner_reject_040_check
    CHECK (version <> '040');

COMMIT;
