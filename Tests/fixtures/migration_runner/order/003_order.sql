CREATE TABLE migration_runner_order (
    order_id bigserial PRIMARY KEY,
    migration_version text NOT NULL
);

INSERT INTO migration_runner_order(migration_version) VALUES ('003');
