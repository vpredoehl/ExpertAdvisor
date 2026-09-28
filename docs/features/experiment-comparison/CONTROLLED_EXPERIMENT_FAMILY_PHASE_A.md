# Controlled experiment family: Phase A foundation

Migration `098_controlled_experiment_family_foundation.sql` implements the
additive relational foundation described in
`CONTROLLED_EXPERIMENT_FAMILY_ARCHITECTURE.md`.  It creates the immutable
family-version declaration graph (version, arms, cells, members, and required
execution identities), plus the durable provenance relations that later phases
will populate for review, approval, materialization, and authorization.

The C++ `ControlledExperimentFamilyRepository` intentionally has a narrow
surface: it creates one complete declaration graph in a caller-owned database
transaction and loads that graph by version or canonical plan identity.  It
does not expose arbitrary updates and it does not create experiments.

Scientific contracts are relational and typed: time ranges, target epoch
budget, layout/width, warmup and Donchian policy, objective and calendar
identities, continuation declaration, execution requirements, and arm masks
are stored independently of the rendered JSON specification snapshot.  The
snapshot is audit rendering only.  Canonical/hash pairs are stored for every
identity that must survive a later lookup or collision check.

All durable scientific and future-provenance rows use restrictive foreign
keys.  This deliberately prevents deletion from erasing declared science,
worker/resource provenance, or eventual experiment evidence.  Historical
`experiment` rows are unaffected because their nullable
`controlled_experiment_family_member_id` remains `NULL`.

The migration installs immutable-row triggers for the declaration and
provenance tables.  Phase D will replace the member-link portion with the
strict, materialization-transaction-only transition specified by the
architecture; Phase A has no path that can attach an experiment.  The root
family record alone permits a future retirement annotation without rewriting
the frozen version.

The current runtime-role model does not establish a separately tested
least-privilege writer role for these new tables.  The triggers therefore give
database-level protection in the current ownership model, while Phase B/D
must add and qualify the exact runtime grants and the materialization-only
member-link trigger before any operator lifecycle is exposed.  Likewise,
manifest SHA-256 remains optional only for historical worker identities whose
registry entry lacks it; new Phase-B bindings must require it when the selected
registry entry provides it.

Phase A deliberately does not add a CLI, canonicalizer, review or approval
workflow, experiment materialization, scheduler admission, execution
authorization, continuation enforcement, or any production controlled-family
data.
