# Semantic-worker registry v5

Schema v5 permits multiple immutable `train` candidates sharing a semantic
layout and model-input width. `infer` remains exactly one binding per layout.

Each v5 worker entry has a non-negative `selection_priority`, unique within
the exact `(semantic_layout, model_input_width, worker_role)` candidate group.
This stronger invariant prevents a priority tie from depending on registry
entry order. Lower priority is considered only after exact layout/width/role
matching, required-capability filtering, and minimal capability-superset
filtering. Publication appends historical TRAIN candidates with a greater
priority than every existing candidate in that exact group.

Minimal capability-superset filtering counts every advertised capability not
required by the request plus its selected role capability. For example, during
a `train` selection, advertised `infer` and `analyze` capabilities count as
excess before `selection_priority` is considered. This is the current V5
behavior; the registry does not apply a role-scoped excess calculation.

For an exact layout/width TRAIN group that contains an explicitly
`train_feature_ablation_v1`-qualified candidate, an otherwise empty-capability
TRAIN request is evaluated with that capability too. This general superset
rule intentionally sends a no-mask control through the same deterministic
selection domain as a feature-ablated treatment. It prevents later append-only
plain-TRAIN candidates from separating a matched control/treatment pair onto
different artifacts. A layout/width with no ablation-qualified candidate keeps
the established empty-mask routing behavior.

Capabilities remain immutable manifest properties and must exactly agree with
the registry entry. `selection_priority` is registry routing policy and is not
written into an artifact manifest.

V5 readers accept v4 registries as singleton candidates with implicit priority
zero. V4 schedulers do not read v5 registries, so scheduler deployment must
precede registry publication. No database schema change is required because
worker attempts already record immutable artifact identity and manifest
provenance.
