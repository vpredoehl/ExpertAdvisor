BEGIN;

-- Independent scheduler admission policy; does not change experiment ordering.
CREATE TABLE IF NOT EXISTS public.experiment_scheduler_phase_policy (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    phase_priority text NOT NULL DEFAULT 'concurrent' CHECK (phase_priority IN (
        'concurrent', 'train:infer:analyze', 'train:analyze:infer',
        'infer:train:analyze', 'infer:analyze:train',
        'analyze:train:infer', 'analyze:infer:train')),
    revision bigint NOT NULL DEFAULT 0 CHECK (revision >= 0),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
INSERT INTO public.experiment_scheduler_phase_policy(singleton)
VALUES (true) ON CONFLICT (singleton) DO NOTHING;

GRANT SELECT, UPDATE ON public.experiment_scheduler_phase_policy TO pqxx;

COMMIT;
