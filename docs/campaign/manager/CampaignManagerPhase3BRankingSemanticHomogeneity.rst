Campaign Manager Phase 3B Ranking Semantic Homogeneity
======================================================

Phase 3B makes recommendation ranking fail closed unless the complete source
population has one verified scoring semantic identity and one verified
evaluation semantic identity.  Validation occurs before sorting, limiting,
membership construction, or snapshot persistence.  Every supported scope is
covered: evaluation run, recommendation scan, symbol, horizon, family,
symbol/horizon, and global.

Semantic identities
-------------------

The scoring semantic identity is version 1.  Its authoritative canonical text
contains the complete authoritative scoring-policy canonical plus an explicit
algorithm contract: component names and order, normalization, weighted
aggregation and clamping, missing-value behavior, structural-distance behavior,
and source-metric behavior.  The evaluation semantic identity is version 1 and
contains the evaluator algorithm, disposition precedence, eligibility rule,
evaluation and evaluator versions, and the complete scoring semantic canonical.

Canonical text is authoritative.  Each hash must be the FNV-1a hash of its own
canonical text; a matching hash never substitutes for canonical equality.
Unsupported versions, malformed canonical text, incorrect hashes, or a mismatch
between run and result provenance make the population unverified and ranking is
rejected.

Snapshot and legacy behavior
----------------------------

New snapshot identities use
``experiment_recommendation_ranking_snapshot_identity_v2`` and commit to the
full-population semantic state and both common semantic canonicals.  Non-empty
new snapshots must be ``verified_homogeneous``.  Empty populations preserve the
existing empty-snapshot behavior and use the explicit ``empty`` state with no
fabricated scoring or evaluation identity.

Migration 077 classifies historical snapshots from their complete persisted
``source_membership_canonical``, not from top-N ranking members.  A snapshot is
classified as verified homogeneous, empty, legacy heterogeneous, or legacy
unverified.  Historical rank, score, membership, evaluation, and materialized
campaign decisions are not rewritten.  Campaign planning accepts verified
homogeneous or well-formed empty input and rejects heterogeneous or unverified
input.

Profitability boundary
----------------------

Final profitability provenance remains observational.  Profitability is absent
from scoring input, score components, evaluation disposition, ranking semantics,
membership, ordering, tie-breaks, snapshot identity, and campaign selection.
``profitability_weight`` and ``profitability_score_contribution`` remain zero.
Migration 076 and exact FINAL profitability binding are unchanged.  Phase 3C is
not implemented.
