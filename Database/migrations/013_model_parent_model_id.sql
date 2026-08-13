ALTER TABLE model
    ADD COLUMN IF NOT EXISTS parent_model_id BIGINT NULL;

CREATE INDEX IF NOT EXISTS model_parent_model_id_idx
    ON model (parent_model_id);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'model_parent_model_id_fkey'
    ) THEN
        ALTER TABLE model
            ADD CONSTRAINT model_parent_model_id_fkey
            FOREIGN KEY (parent_model_id)
            REFERENCES model(model_id)
            NOT VALID;
    END IF;
END $$;

GRANT SELECT, INSERT, UPDATE, DELETE ON model TO pqxx;
