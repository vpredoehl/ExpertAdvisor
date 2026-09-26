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

Capabilities remain immutable manifest properties and must exactly agree with
the registry entry. `selection_priority` is registry routing policy and is not
written into an artifact manifest.

V5 readers accept v4 registries as singleton candidates with implicit priority
zero. V4 schedulers do not read v5 registries, so scheduler deployment must
precede registry publication. No database schema change is required because
worker attempts already record immutable artifact identity and manifest
provenance.
