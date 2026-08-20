\set ON_ERROR_STOP on

DO $$
DECLARE
    identity_index text;
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema='public' AND table_name='experiment'
          AND column_name='resume_expand_input_width'
          AND data_type='boolean' AND is_nullable='NO'
          AND column_default='false'
    ) THEN
        RAISE EXCEPTION
            'resume_expand_input_width must be boolean NOT NULL DEFAULT false';
    END IF;

    SELECT pg_get_indexdef(indexrelid) INTO identity_index
    FROM pg_index
    WHERE indexrelid='experiment_unique_identity_uidx'::regclass;
    IF identity_index IS NULL OR
       identity_index NOT LIKE '%resume_expand_input_width%' THEN
        RAISE EXCEPTION
            'experiment identity index is missing expansion mode: %',
            identity_index;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid='experiment'::regclass
          AND conname='experiment_resume_expand_input_width_source_check'
    ) THEN
        RAISE EXCEPTION 'expansion source constraint is missing';
    END IF;
END $$;
