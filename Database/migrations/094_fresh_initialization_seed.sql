-- Historical fresh models used the LSTM seed-42 initialization contract.
ALTER TABLE experiment
    ADD COLUMN IF NOT EXISTS fresh_initialization_seed bigint NOT NULL DEFAULT 42
    CHECK (fresh_initialization_seed > 0 AND fresh_initialization_seed <= 4294967295);
