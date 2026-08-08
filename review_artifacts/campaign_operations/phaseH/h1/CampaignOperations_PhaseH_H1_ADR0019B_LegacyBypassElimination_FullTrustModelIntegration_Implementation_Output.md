---
title: "Campaign Operations Phase H H1 ADR-0019B Legacy-Bypass Elimination and Full Trust-Model Integration"
document_type: "implementation output"
status: "final"
date: "2026-08-02"
---

# Campaign Operations Phase H H1 ADR-0019B Legacy-Bypass Elimination and Full Trust-Model Integration

Disposition: `READY_FOR_FOCUSED_INDEPENDENT_REVERIFICATION`.

## Implementation result

The missing disposable PostgreSQL ACL/default v3 generator is integrated. It
queries PostgreSQL directly without an expected-manifest input, records the
query process and raw catalog bytes, binds both outputs to snapshots, emits a
trusted generator receipt and one v2 envelope per ACL/default obligation, and
passes the observed rows to the existing comparator only after capture.

All registered production evidence classes now materialize through the trusted
generator runner. Parent-captured snapshot bytes cross the process boundary on
standard input; generators cannot reopen the original input pathname. The
materialized evidence includes generator receipts, raw v2 envelopes, output
snapshots, runtime records, trusted validator receipts/results, reports, and
complete provenance.

Validation-mode graph reconciliation consumes and authenticates the persisted
validator execution rather than replacing it with a second process identity.
Idempotent regeneration reuses a receipt only when the exact declared input
artifact IDs and digests still match; changed inputs execute the affected
validator again.

## Provenance

Every reviewed clause/obligation derivation is materialized. Multiple governing
clauses produce multiple authoritative chains; no preferred clause is inferred.
Each adjacent edge is stored in both directions.

```text
governing clause
  -> evidence obligation
  -> generator execution
  -> generator receipt
  -> raw evidence envelope
  -> snapshot
  -> runtime record
  -> validator execution
  -> validator receipt
  -> validator result
  -> report entry
  -> generated report
```

The disposable result contains 1,160 clause/obligation chains and 5,910 unique
directed provenance edges. Mutation tests reject removal of any node, any
adjacent forward/reverse edge, generator receipt, validator receipt, or
snapshot.

## Readiness proof

Run `h1-20260802T185411Z-3999` produced:

| Materialization | Count |
|---|---:|
| Requirements/runtime/results/reports | 287 each |
| Trusted generator executions | 247 |
| Raw v2 envelopes | 287 |
| Generator output snapshots | 248 |
| Trusted validator executions | 7 |
| Final-assurance controls | 12/12 |
| Mutation controls | 92/92 |
| Clause/obligation provenance chains | 1,160 |
| Directed provenance edges | 5,910 |
| Graph defects | 0 |

The exact readiness equation evaluated:

```text
authority_complete=1
trusted_generator_execution_complete=1
raw_envelopes_complete=1
snapshots_complete=1
runtime_records_complete=1
trusted_validator_execution_complete=1
validator_results_complete=1
acl_catalog_independent=1
provenance_graph_complete=1
report_complete=1
no_legacy_path_reachable=1
ready=1
```

All fourteen raw contract classes are present, including the dedicated
exclusion envelope. The `H1ACL501` production stop and
`trusted-v3-catalog-generator-not-integrated` diagnostic are removed.

## Legacy inventory

`Scripts/CampaignOperationsH1LegacyInventory.py --check` reports 18
machine-probed entries and zero reachable paths. It covers legacy authority,
validator, validator-receipt, raw evidence, runtime-v1, report generation,
ACL/default generation, restore validation, path-reopen, graph, and readiness
classes. The TSV is generated; it is not manually maintained.

## Tests and build

- Complete disposable migration/replay, restore A–J, scheduler ownership,
  locks, ACL origin, ACL/default, repository/service, Phase 1–5 regression,
  trusted generation, graph generation, and graph freshness validation passed.
- `CampaignOperationsPhaseH1FinalAssuranceTests.sh` passed `results=12`.
- All 92 mutation cases passed, including reference graph, registry, record
  delta, forged build, forged restore, ACL, lock, manifest, and parser cases.
- Evidence-authority: 41/41 passed.
- Trusted runner: 12/12 passed.
- ACL independence: 13/13 passed.
- Provenance integration: 3/3 passed.
- Trust model integration: 8/8 passed.
- Exact Release `xcodebuild` command succeeded.
- `git diff --check` succeeded.

The Release invocation emitted only the environment destination-selection and
missing optional `LLVM22.xctoolchain/Info.plist` warnings; it emitted no new
source diagnostic in the incremental build.

## Architecture and operational safety

No Campaign Operations database authority, identity, migration, transaction,
workflow, scheduler, or H1–H4 boundary was redesigned. No production row was
changed, no production worker was launched, no clean build was run, and no
commit was created.

The detailed architecture/readiness diagrams are in
`docs/architecture/CampaignOperations_PhaseH_Legacy_Bypass_Elimination.md`.
The unified working-tree changes are available with `git diff HEAD`; untracked
new Phase H files can be reviewed with `git diff --no-index /dev/null <file>`.

## Remaining risks

- The worktree contained a large pre-existing staged/untracked Phase H change
  set. This pass preserved it and did not attempt to split or commit it.
- The two Xcode environment warnings remain outside the Phase H source change.
- The successful run is disposable/offline evidence; no production cutover or
  live scheduler interaction was attempted.
