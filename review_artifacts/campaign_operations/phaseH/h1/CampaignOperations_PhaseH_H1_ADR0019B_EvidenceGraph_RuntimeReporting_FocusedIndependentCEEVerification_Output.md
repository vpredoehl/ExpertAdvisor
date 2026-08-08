---
title: "Campaign Operations Phase H H1 ADR-0019B Evidence Graph and Runtime-Derived Reporting Focused Independent CEE Verification"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_PhaseH_H1_ADR0019B_EvidenceGraph_RuntimeReporting_FocusedIndependentCEEVerification_Prompt.txt"
reasoning_effort: "high"
model: "default"
---

# Campaign Operations Phase H H1 ADR-0019B Evidence Graph and Runtime-Derived Reporting Focused Independent CEE Verification

# Campaign Operations Phase H H1 focused independent verification

## 1. Executive verdict

**NOT_VERIFIED**

The two primary claims were falsified:

1. The evidence graph is not complete, explicitly bidirectional, or exclusively runtime-backed.
2. Reports are not generated exclusively from semantically reconciled runtime records.

The implementation can emit `READY_FOR_FOCUSED_INDEPENDENT_REVERIFICATION` after accepting forged generator/validator metadata, semantically false evidence artifacts, filename-inferred edges, unindexed files, syntactically valid stale restore data, and a build log containing `** BUILD FAILED **`.

No repository files, production database, roles, scheduler state, or experiment state were modified.

## 2. Findings ordered by severity

### BLOCKER — Graph-health and readiness values are hard-coded

`orphan`, `unresolved`, `duplicate`, and `stale` are assigned literal zero values in [CampaignOperationsH1EvidenceGraph.py](</Volumes/Developer SSD/ExpertAdvisor/Scripts/CampaignOperationsH1EvidenceGraph.py:313>). `Reconciliation: PASS` is unconditional, and readiness depends on those literal zeros at [line 353](</Volumes/Developer SSD/ExpertAdvisor/Scripts/CampaignOperationsH1EvidenceGraph.py:353>).

The claimed `0/0/0/0` is therefore not independently computed.

### BLOCKER — Validator results are synthesized

All 195 validator results are constructed with `status="VALIDATED"` at [lines 173–176](</Volumes/Developer SSD/ExpertAdvisor/Scripts/CampaignOperationsH1EvidenceGraph.py:173>); no validator-result records containing parsed values, comparisons, or diagnostics are consumed.

The 12 final-assurance validators are synthesized the same way at [lines 212–215](</Volumes/Developer SSD/ExpertAdvisor/Scripts/CampaignOperationsH1EvidenceGraph.py:212>).

### BLOCKER — Semantically false evidence is accepted

Independent mutations accepted by the complete reference-graph wrapper included:

- Replacing the H1 H2/H3/H4-exclusion artifact with `H2 H3 H4 mutation entry points are present`, updating its digest, and retaining `SUCCESS`.
- Replacing `RELEASE_BUILD.log` with `** BUILD FAILED **`, updating its digest, and still generating `READY_FOR_FOCUSED_INDEPENDENT_REVERIFICATION` with `PASS`.
- Changing restore scenario A to a forged but internally matching diagnostic plus another valid 64-hex digest.
- Changing the runtime generator source for H1ROLE001 to `forged-generator-source`.

### BLOCKER — Registry semantic drift is ignored

One mutation changed all of the following and still passed the full wrapper:

- Requirement section, description, runtime cardinality `1→99`, required validator, report cardinality `1→88`, and status policy.
- Fixture source to `nonexistent.tsv:999`, classification, and cardinality.
- Generator implementation to `NonexistentGenerator.sh`, input/output registries, and policy.
- Runtime record type, classification, and cardinality.
- Validator implementation to `NonexistentValidator.sh`, input/output type, and duplicate/stale policies.
- Report cardinality.

Only fixture cardinality and selected identifier equality are enforced.

### BLOCKER — Filename and prefix reconstruction is authoritative

Edges are manufactured from:

- `RT- + fixture`
- `REP- + fixture`
- filename stems under `raw-lock` and `raw-acl-origin`
- `H1LOCK`, `H1RESTORE`, and `H1REG` prefixes
- shell/Python globbing and sorted directory scans

See [lines 230–264](</Volumes/Developer SSD/ExpertAdvisor/Scripts/CampaignOperationsH1EvidenceGraph.py:230>).

Adding an invalid `raw-lock/H1ROLE001.tsv` caused the tool to create:

```text
ARTFILE-0063 -> RT-H1ROLE001 -> REP-H1ROLE001
```

The full wrapper accepted it solely because the stem matched a known runtime ID.

### BLOCKER — Checked-in final report is not the generated report

The checked-in [implementation output](</Volumes/Developer SSD/ExpertAdvisor/CampaignOperations_PhaseH_H1_ADR0019B_FinalEvidenceModelCorrection_Implementation_Output.md:1>) is 51 manually authored lines. Fresh generation produced a different 438-line report.

Validation returned:

```text
H1R006 key=CampaignOperations_PhaseH_H1_ADR0019B_FinalEvidenceModelCorrection_Implementation_Output.md stage=reference-graph-reconciliation detail=stale-generated-report
```

### HIGH — “Independent” requirement registry is exactly a union

The 183 requirements are exactly:

```text
142 traceability + 38 ACL-origin + 2 pre-enablement + 1 invariant = 183
```

All 183 descriptions are mechanically generic: `Evidence contract for <ID>`. The registry is therefore the prohibited union rather than an independently maintained architectural requirements authority.

### HIGH — Final-assurance records are outside the declared graph

The final 195 records include 12 runtime records referring to:

- 12 requirement IDs absent from the requirement registry
- 12 fixture IDs absent from the fixture registry
- `GEN-FINAL-ASSURANCE`, absent from the generator registry
- `VAL-FINAL-ASSURANCE`, absent from the validator registry
- 12 report entries absent from the authoritative report-entry registry

These nodes are appended ad hoc at [lines 205–221](</Volumes/Developer SSD/ExpertAdvisor/Scripts/CampaignOperationsH1EvidenceGraph.py:205>).

### HIGH — Artifact stable IDs do not join

All runtime/report rows use IDs such as `ART:h1-lock-runtime.tsv`. The artifact index uses `ARTFILE-0001`, etc.

Independent recomputation found:

```text
runtime artifact IDs matching artifact-index IDs: 0 / 195
report artifact IDs matching artifact-index IDs:  0 / 195
```

Path/basename inference substitutes for the absent ID edge.

### HIGH — One-record changes do not have one-entry effects

- One H1LOCK record change altered all 15 H1LOCK report entries.
- One ACL-origin record change altered all 38 ACL-origin report entries.
- One generic runtime generator-source change altered no report entry; only the aggregate runtime-file digest in the artifact index changed.
- One restore normalized-record change altered no report entry.

### HIGH — Reverse lock direction is normalized incorrectly

Both possible blocker directions are mapped to the configured permitted direction at [CampaignOperationsH1LockEvidence.py:138](</Volumes/Developer SSD/ExpertAdvisor/Scripts/CampaignOperationsH1LockEvidence.py:138>). A raw reverse wait therefore cannot become the prohibited direction through this normalizer.

### HIGH — Restore validation is shape-only

[CampaignOperationsPhaseH1RestoreArtifactTests.sh](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1RestoreArtifactTests.sh:8>) checks row shape, A–J presence, expected/actual equality, and 64-hex syntax. It does not locate the referenced artifacts or recompute their digests. A forged restore record passed.

## 3. Registry inventory

| Registry/path | Version/key | Status | Digest/validation | Principal deficiency |
|---|---|---|---|---|
| `CampaignOperationsH1Requirements.tsv` | No version; `requirement_id` | Hand-maintained union | Whole-file SHA-256 | No explicit fixture reverse IDs; semantic fields ignored |
| `CampaignOperationsH1Fixtures.tsv` | v1; `fixture_id` | Hand-maintained | Whole-file SHA-256 | Source path/version/classification/cardinality not validated |
| `CampaignOperationsH1Generators.tsv` | v1; `generator_id` | Hand-maintained | Whole-file SHA-256 | No fixture list, emitted runtime IDs, run ID, or generator version |
| `CampaignOperationsH1RuntimeRecords.tsv` | v1; `runtime_record_id` | Hand-maintained expected-node registry | Whole-file SHA-256 | Not actual runtime data; semantic fields ignored |
| `CampaignOperationsH1Validators.tsv` | v1; `validator_id` | Hand-maintained definitions | Whole-file SHA-256 | No parsed runtime IDs, comparison results, diagnostic, or report reverse edge |
| `CampaignOperationsH1ReportEntries.tsv` | v1; `report_entry_id` | Hand-maintained | Whole-file SHA-256 | No validator-result ID; cardinality ignored |
| `h1-runtime-results.tsv` | v1; graph keys by fixture | Generated | Per-artifact digest | Generator ID/version absent; several factual fields ignored |
| `h1-lock-runtime.tsv` | Header-only format; `h1lock_id` | Generated | Raw-file digests recomputed | No run ID/version; aggregate digest fans out to 15 entries |
| `h1-acl-origin-runtime.tsv` | v2; `fixture_id` | Generated | Raw and expansion digests | Aggregate digest fans out to 38 entries |
| `h1-restore-runtime.tsv` | v1; scenario A–J | Generated | Only digest syntax checked | Artifact digests not recomputed by restore validator |
| `h1-pre-enablement-runtime.tsv` | v1; `evidence_id` | Generated from static TSV | Aggregate digest | Not an observation of a callable runtime workflow |
| `h1-uniqueness-invariant-runtime.tsv` | v1; `invariant_id` | Generated from catalog query | Aggregate digest | One accepted non-executable invariant |
| `h1-final-assurance-results.tsv` | v1; `evidence_id` | Generated | Log digest only | Log semantics never parsed |
| `h1-validator-results.tsv` | v1; `validator_result_id` | Synthesized | Exact regeneration | Not authentic validator output |
| `h1-report-entry-registry.tsv` | v1; `report_entry_id` | Synthesized | Exact regeneration | Built from expected static mappings |
| `h1-artifact-index.tsv` | v1; `ARTFILE-*` | Generated | Exact regeneration | Does not index itself or eight other evidence files; ID namespace does not join |
| Lock/ACL/restore/pre/invariant fixtures | Mixed | Mostly hand-maintained | Mixed | Not incorporated into one explicit bidirectional graph |

## 4. Requirement-registry verdict

**NOT VERIFIED**

The registry has 183 unique rows and no duplicate keys, but:

- It is exactly a union of subordinate evidence registries.
- All 183 descriptions are placeholders.
- Architectural source references are broad section labels rather than exact clauses.
- Required runtime, validator, report cardinality, and status-policy fields can be forged without failure.
- There is no explicit requirement→fixture reverse edge.

The number **183 is verified only as a file-row count**, not as an independently established authoritative-requirement count.

## 5. Fixture/generator-edge verdict

**NOT VERIFIED**

The 183 fixture→generator IDs are internally consistent. Reverse generator→fixture mappings do not exist. Generator implementation/source/cardinality metadata can drift undetected.

`GEN-ACL-MANIFEST` claims `CampaignOperationsH1ManifestValidator.sh` as its generator, but the 41 rows are actually emitted by [CampaignOperationsPhaseH1MigrationTests.sh:2316](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1MigrationTests.sh:2316>).

## 6. Runtime-record-edge verdict

**NOT VERIFIED**

There are 183 declared expected runtime nodes and 195 generated reconciled rows. The extra 12 assurance rows are outside the authoritative graph.

Runtime generator/validator IDs are copied from the expected static registry at [lines 168–172](</Volumes/Developer SSD/ExpertAdvisor/Scripts/CampaignOperationsH1EvidenceGraph.py:168>), not authenticated from runtime emissions.

## 7. Validator-result-edge verdict

**NOT VERIFIED**

All 195 validator results are fabricated by the report generator with `VALIDATED`. They do not contain:

- parsed actual values
- expected values
- comparison result
- validator diagnostic
- report-entry reverse reference
- authentic validator execution identity

## 8. Report-entry/artifact-edge verdict

**NOT VERIFIED**

Counts are 195 report entries and 153 indexed artifacts, but:

- 12 report entries are undeclared.
- 0/195 report artifact IDs resolve to artifact-index stable IDs.
- Nine files are unindexed.
- The report generator accepts semantically false but correctly digested artifacts.
- Generated registries are omitted from the artifact index.

## 9. Forward/reverse cardinality verdict

**NOT VERIFIED**

| Edge | Available direction | Missing/invalid direction |
|---|---:|---:|
| Requirement ↔ Fixture | 183 fixture→requirement | Requirement→fixture absent; 12 assurance fixtures undeclared |
| Fixture ↔ Generator | 183 fixture→generator | Generator→fixture absent; assurance generator undeclared |
| Generator ↔ Runtime | 195 runtime→generator strings | No emitted-runtime reverse lists; 12 use undeclared generator |
| Runtime ↔ Validator result | 195 validator-result→runtime | Runtime→validator-result absent |
| Validator result ↔ Report | 195 report→validator-result | Validator-result→report absent |
| Runtime ↔ Artifact | Path/inferred reverse sets | 0/195 stable artifact IDs resolve |
| Report ↔ Artifact | Path/inferred reverse sets | 0/195 stable artifact IDs resolve |

At least 1,365 edge instances lack the required explicit reverse or stable-ID-resolved direction.

## 10. Independently computed graph health

| Measure | Independent result |
|---|---:|
| Static requirement/fixture/runtime/report duplicate keys | 0 |
| Generated runtime/validator/report/artifact duplicate keys | 0 |
| Indexed-artifact digest mismatches | 0 |
| Indexed-artifact stale run IDs | 0 |
| Generated-registry stale versions | 0 |
| Unregistered assurance requirement IDs | 12 |
| Unregistered assurance fixture IDs | 12 |
| Unregistered assurance generator IDs | 1 |
| Unregistered assurance validator IDs | 1 |
| Generated report entries absent static registry | 12 |
| Unindexed files | 9 |
| Runtime artifact IDs unresolved against artifact-index IDs | 195 |
| Report artifact IDs unresolved against artifact-index IDs | 195 |
| Demonstrated semantically stale artifacts accepted | ≥3 |
| Missing explicit reverse/stable-ID edge instances | ≥1,365 |

The nine unindexed files were:

```text
default.tsv
explicit.tsv
h1-artifact-index.tsv
h1-reconciled-runtime-records.tsv
h1-reference-graph.log
h1-report-entry-registry.tsv
h1-traceability-validation.log
h1-validator-results.tsv
run-id
```

## 11. Former 41-orphan requirement table

All rows shared:

- Generator: `GEN-ACL-MANIFEST`
- Validator: `VAL-TRACE`
- Artifact: `runtime-artifacts/h1-acl-requirement-evidence.tsv`
- Fresh digest prefix: `f6498594ff02`
- Generated status: `SUCCESS`
- Independent verdict: **NOT RUNTIME-VERIFIED**

The artifact contains static manifest rows and file digests generated at [migration-test lines 2310–2356](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1MigrationTests.sh:2310>), not 41 independently indexed production-audit results.

| Requirement | Fixture | Runtime | Report |
|---|---|---|---|
| H1-ACL-ADMISSION | H1REG001 | RT-H1REG001 | REP-H1REG001 |
| H1-ACL-ATTEMPT | H1REG002 | RT-H1REG002 | REP-H1REG002 |
| H1-ACL-ATTEMPT-V1 | H1REG003 | RT-H1REG003 | REP-H1REG003 |
| H1-ACL-CANONICAL | H1REG004 | RT-H1REG004 | REP-H1REG004 |
| H1-ACL-COMPLETION | H1REG005 | RT-H1REG005 | REP-H1REG005 |
| H1-ACL-COMPLETION-AUDIT | H1REG006 | RT-H1REG006 | REP-H1REG006 |
| H1-ACL-COMPLETION-AUDIT-COLUMNS | H1REG007 | RT-H1REG007 | REP-H1REG007 |
| H1-ACL-COMPLETION-COLUMNS | H1REG008 | RT-H1REG008 | REP-H1REG008 |
| H1-ACL-CONTEXT | H1REG009 | RT-H1REG009 | REP-H1REG009 |
| H1-ACL-COUPLED-TYPES | H1REG010 | RT-H1REG010 | REP-H1REG010 |
| H1-ACL-DEPLOYMENT-AUDIT | H1REG011 | RT-H1REG011 | REP-H1REG011 |
| H1-ACL-DISPATCH-AUDIT | H1REG012 | RT-H1REG012 | REP-H1REG012 |
| H1-ACL-DISPATCH-AUDIT-COLUMNS | H1REG013 | RT-H1REG013 | REP-H1REG013 |
| H1-ACL-ENABLEMENT | H1REG014 | RT-H1REG014 | REP-H1REG014 |
| H1-ACL-ENABLEMENT-AUDIT | H1REG015 | RT-H1REG015 | REP-H1REG015 |
| H1-ACL-FIXED-TRANSITIONS | H1REG016 | RT-H1REG016 | REP-H1REG016 |
| H1-ACL-LEDGER | H1REG017 | RT-H1REG017 | REP-H1REG017 |
| H1-ACL-LOCK-1 | H1REG018 | RT-H1REG018 | REP-H1REG018 |
| H1-ACL-LOCK-2 | H1REG019 | RT-H1REG019 | REP-H1REG019 |
| H1-ACL-LOCK-3 | H1REG020 | RT-H1REG020 | REP-H1REG020 |
| H1-ACL-LOCK-4 | H1REG021 | RT-H1REG021 | REP-H1REG021 |
| H1-ACL-LOCK-5 | H1REG022 | RT-H1REG022 | REP-H1REG022 |
| H1-ACL-READINESS | H1REG023 | RT-H1REG023 | REP-H1REG023 |
| H1-ACL-REPLAY | H1REG024 | RT-H1REG024 | REP-H1REG024 |
| H1-ACL-REQUEST | H1REG025 | RT-H1REG025 | REP-H1REG025 |
| H1-ACL-REQUEST-COLUMNS | H1REG026 | RT-H1REG026 | REP-H1REG026 |
| H1-ACL-SCHEDULER-COLUMNS | H1REG027 | RT-H1REG027 | REP-H1REG027 |
| H1-ACL-SCHEDULER-EVIDENCE | H1REG028 | RT-H1REG028 | REP-H1REG028 |
| H1-ACL-SCHEMA-PUBLIC | H1REG029 | RT-H1REG029 | REP-H1REG029 |
| H1-ACL-SEQUENCES | H1REG030 | RT-H1REG030 | REP-H1REG030 |
| H1-ACL-STATUS | H1REG031 | RT-H1REG031 | REP-H1REG031 |
| H1-ACL-SUPPORTING-DEFINERS | H1REG032 | RT-H1REG032 | REP-H1REG032 |
| H1-DEFAULT-GLOBAL-FUNCTION | H1REG033 | RT-H1REG033 | REP-H1REG033 |
| H1-DEFAULT-GLOBAL-SCHEMA | H1REG034 | RT-H1REG034 | REP-H1REG034 |
| H1-DEFAULT-GLOBAL-SEQUENCE | H1REG035 | RT-H1REG035 | REP-H1REG035 |
| H1-DEFAULT-GLOBAL-TABLE | H1REG036 | RT-H1REG036 | REP-H1REG036 |
| H1-DEFAULT-GLOBAL-TYPE | H1REG037 | RT-H1REG037 | REP-H1REG037 |
| H1-DEFAULT-PUBLIC-FUNCTION | H1REG038 | RT-H1REG038 | REP-H1REG038 |
| H1-DEFAULT-PUBLIC-SEQUENCE | H1REG039 | RT-H1REG039 | REP-H1REG039 |
| H1-DEFAULT-PUBLIC-TABLE | H1REG040 | RT-H1REG040 | REP-H1REG040 |
| H1-DEFAULT-PUBLIC-TYPE | H1REG041 | RT-H1REG041 | REP-H1REG041 |

## 12. Graph mutation verdict

**NOT VERIFIED**

The supplied suites passed their intended 63 aggregate cases:

- Lock: 10
- ACL-origin: 14
- Reference graph: 20
- Manifest: 19

Representative correct rejection diagnostics included:

```text
H1R005 key=RT-H1ROLE001 stage=reference-graph-reconciliation detail=missing-actual-runtime
H1R005 key=final-assurance stage=reference-graph-reconciliation detail=missing-or-extra-assurance-result
H1R005 key=H1ROLE001 stage=reference-graph-reconciliation detail=stale-digest
H1R006 key=h1-report-entry-registry.tsv stage=reference-graph-reconciliation detail=stale-generated-registry
H1R006 key=...CampaignOperationsH1Traceability.md stage=reference-graph-reconciliation detail=stale-generated-report
```

However, the independent accepted-negative cases above invalidate the suite’s completeness.

## 13. Silent-reconstruction verdict

**FAILED**

- Generator identity is copied from static expectations.
- Artifact and report edges are reconstructed from prefixes and stems.
- A bogus raw-lock file with a known stem was accepted and indexed.
- Arbitrary `rogue.tsv` files at top level and in an unscanned nested directory were accepted and ignored.

## 14. Registry-drift verdict

**FAILED**

A syntactically valid, re-digested mutation across requirement, fixture, generator, runtime, validator, and report semantic fields passed the full wrapper. Digest agreement proves only that the altered copies agree with their digest ledger.

## 15. Report-generator hard-code audit

**FAILED**

Hard-coded or synthesized report content includes:

- orphan/unresolved/duplicate/stale zeros
- `Reconciliation: PASS`
- readiness disposition logic
- H1LOCK007/008/009/012 prose
- closure of 41 former orphans
- restore/historical claims
- H2/H3/H4 exclusion
- skipped-suite classification
- residual-risk prose

See [the `facts` dictionary](</Volumes/Developer SSD/ExpertAdvisor/Scripts/CampaignOperationsH1EvidenceGraph.py:365>).

## 16. Per-evidence report-entry verdict

**NOT VERIFIED**

There are 195 generated entries, but:

- 12 lack authoritative requirement/fixture/generator/validator/report declarations.
- Validator results are synthesized.
- The 63 mutations exist as four aggregate PASS logs, not 63 individual runtime/validator/report records.
- Build, compile, checksum, worktree, and determinism logs are not semantically parsed.
- The checked-in final report is not the generated report.

## 17. Runtime-to-report one-change verdict

**FAILED**

| Changed evidence | Expected | Observed |
|---|---:|---:|
| One H1LOCK record | 1 report entry | 15 entries changed |
| One ACL-origin record | 1 report entry | 38 entries changed |
| One generic generator-source field | 1 report entry | 0 entries changed |
| One restore normalized field | 1 report entry | 0 entries changed |

The H1LOCK and ACL-origin fan-out occurs because every entry uses the digest of one aggregate normalized TSV.

## 18. Runtime deletion verdict

**PASS for simple deletion**

Deleting H1ROLE001 failed before readiness:

```text
H1R005 key=RT-H1ROLE001 stage=reference-graph-reconciliation detail=missing-actual-runtime
```

Deleting `FULL_PIPELINE` from final assurance failed:

```text
H1R005 key=final-assurance stage=reference-graph-reconciliation detail=missing-or-extra-assurance-result
```

This does not cure the other provenance failures.

## 19. Report-entry deletion/addition verdict

**PASS for generated-registry tampering**

Both deletion and addition failed:

```text
H1R006 key=h1-report-entry-registry.tsv stage=reference-graph-reconciliation detail=stale-generated-registry
```

Authoritative report-entry semantic fields and undeclared assurance entries remain unresolved.

## 20. Stale validly formatted artifact verdict

**FAILED**

| Artifact class | Result |
|---|---|
| Lock raw artifact/digest mutations in supplied suite | Rejected |
| ACL raw/expansion mutations in supplied suite | Rejected |
| Generic runtime semantic replacement with updated digest | Accepted |
| Restore normalized record with valid forged digest | Accepted |
| Final-assurance Release build log saying BUILD FAILED | Accepted as PASS/READY |
| Generated graph registry modification | Rejected as stale |
| Artifact-index modification | Rejected as stale |
| Traceability report modification | Rejected as stale |
| Generated final report modification | Rejected as stale |
| Checked-in final implementation report | Already stale/not generated |

## 21. Stale-report verdict

**PARTIAL PASS, overall NOT VERIFIED**

Generated report mutation was detected with `stale-generated-report`, and identical-input regeneration was deterministic.

However, the checked-in final implementation report itself failed freshness validation, and false upstream evidence can be regenerated into a current PASS/READY report.

## 22. Independent summary recomputation

| Claim | Recomputed | Verdict |
|---|---:|---|
| Requirements | 183 rows | Row count verified; authority not verified |
| Fixtures | 183 | Verified |
| Generators | 6 declarations | Authentic runtime generators not verified |
| Runtime records | 195 | Count verified; 12 undeclared |
| Validator results | 195 | Count verified; all synthesized |
| Report entries | 195 | Count verified; 12 undeclared |
| Indexed artifacts | 153 | Count verified; 162 files exist |
| Executable H1LOCK | 15 | Verified |
| Pre-enablement | 2 | Verified as records |
| Uniqueness invariants | 1 | Verified from disposable catalog |
| Authentic ACL-origin fixtures | 38 | Runtime suite passed |
| ACL-origin evidence overall | 46 | 38 authentic + 8 legacy |
| Restore scenarios | 10 | Runtime suite passed; normalized validator inadequate |
| Restore/historical requirements | 11 | 10 A–J + historical bytes |
| Accepted mutation aggregate | 63 | Verified as 10+14+20+19 |
| Synthetic parser cases | 18 | Correctly excluded |
| Orphan/unresolved/duplicate/stale | Not `0/0/0/0` | Claim false |

## 23. Final implementation-report provenance verdict

**FAILED**

The checked-in report at [lines 20–52](</Volumes/Developer SSD/ExpertAdvisor/CampaignOperations_PhaseH_H1_ADR0019B_FinalEvidenceModelCorrection_Implementation_Output.md:20>) contains manually authored factual claims without per-claim source record, validator result, artifact, digest, and report-entry references.

It differs from the generator output and fails freshness validation.

The generated version also uses hard-coded factual prose and accepted a forged failed build as PASS/READY.

## 24. Determinism-versus-correctness verdict

**Determinism verified; correctness not verified**

Two independent regenerations of:

- both Markdown reports
- reconciled runtime TSV
- validator TSV
- report-entry TSV
- artifact index

were byte-identical.

The same deterministic generator also deterministically accepted forged evidence, so determinism provides no acceptance conclusion.

## 25. Lock/ACL regression verdict

**PARTIALLY VERIFIED, overall NOT VERIFIED**

Verified:

- H1LOCK008 calls `AdministerCampaignBudget` in [CampaignOperationsPhaseH1WorkflowLockTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1WorkflowLockTests.cpp:287>).
- H1LOCK007 and H1LOCK009 are excluded from the executable 15-row matrix.
- H1LOCK012 is represented as the uniqueness invariant.
- Fifteen fresh lock cases and ten supplied lock mutations passed.
- All 38 ACL-origin fixtures invoked the production deployment audit.
- Raw ACL evidence and expansion digests were recomputed.

Not verified:

- Reverse wait direction is incorrectly normalized.
- Lock records lack an explicit run ID/version.
- One lock change modifies all 15 report entries.
- One ACL change modifies all 38 report entries.

## 26. Migration/restore/build verdict

**Product regression checks passed; evidence-reporting acceptance failed**

Fresh isolated results:

- Disposable PostgreSQL cluster: passed.
- Repository-owned schema seed and 050→055 path: passed.
- Transactional failing 054→055 upgrade cases: passed.
- 055 replay: passed.
- Restore A–J: passed.
- Historical Attempt V1/Completion V1 bytes: passed.
- Phase 1–5 repository/service/completion regression: passed.
- Focused H1 repository/service tests: passed.
- Strict `-Wall -Wextra -Werror -fsyntax-only`: passed, two translation units.
- H1 CLI parser against isolated build: passed.
- Migration checksum: `3b9b75f289197e89c8cc6fbd5d9bee4aca8996b54efc092c1854f939938fb37d`.
- Manifest digest: `ec6e34b1dd3ee68bf5315f193baee233222822eb045437d154d4b29e9444fb9f`.
- Isolated Release build: `** BUILD SUCCEEDED **`.

The Release build emitted 660 warning lines, including libpqxx deprecations, precision-loss warnings, missing toolchain metadata, and a no-symbols archive warning. The focused Werror compilation remained clean.

## 27. Residual risks and deferred evidence

- The requirement count has not been independently derived clause-by-clause from ADR-0019/19A/19B.
- The 41 former-orphan rows remain static manifest evidence rather than per-requirement runtime audit outputs.
- Reverse lock-direction behavior needs correction and a raw reverse-direction mutation.
- Final-assurance log semantics must be parsed, not accepted by status/digest alone.
- Generated registry files and validation logs need first-class artifact nodes.
- The pure `CampaignOperationsPhaseH1Tests.cpp` executable has no declared project/test harness and was not independently run.
- Shared scheduler/worker integration suites were not launched while the scheduler and seven training workers were active.
- No production migration or role change was attempted.

## 28. Skipped-suite classification

**Safety-required deferment**

Skipped:

- Any suite invoking the shared `DerivedData/ExpertAdvisor/.../LSTM_Release` against production/shared scheduler state.
- Process integration capable of scheduler, training, inference, analysis, or experiment mutation.
- Production migration/cutover.
- H2/H3/H4 suites.

Not skipped:

- Disposable PostgreSQL migration/restore tests.
- Isolated lock and ACL workflows.
- Isolated Release build.
- H1 CLI parser using the isolated binary.
- Static, mutation, compile, checksum, and report checks.

## 29. Acceptance/readiness decisions

| Decision | Result |
|---|---|
| Ready for final H1 acceptance review | **No** |
| Safe to commit | **No** |
| Safe to migrate while disabled | **No acceptance recommendation**; migration runtime passed, but required evidence provenance is invalid |
| Ready for H2 | **No** |

## 30. Exact correction requirements

Before re-verification:

1. Replace hard-coded health values with graph-walk results.
2. Add explicit reverse adjacency for all seven edge types.
3. Use one artifact stable-ID namespace end-to-end.
4. Register final-assurance requirements, fixtures, generator, validator, and report entries.
5. Consume real validator-result records containing parsed/expected/actual/diagnostic fields.
6. Reject unknown files across the complete evidence root.
7. Eliminate filename/prefix edge reconstruction.
8. Validate every registry semantic field and implementation path.
9. Produce 41 authentic per-requirement catalog-audit results.
10. Use per-record artifacts/digests or record-level digests so one change affects one entry.
11. Parse build/compile/checksum/worktree/mutation log semantics.
12. Correct lock wait-direction normalization.
13. Make the checked-in final report the exact generated report and freshness-check it.

## 31. Exact re-verification commands

After correction:

```bash
cd "/Volumes/Developer SSD/ExpertAdvisor"

capture_root="$(mktemp -d /tmp/ea-h1-independent-reverify.XXXXXX)"
H1_EVIDENCE_CAPTURE_DIR="$capture_root" \
  Tests/CampaignOperationsPhaseH1MigrationTests.sh

run_id="$(tr -d '\r\n' < "$capture_root/run-id")"

derived_root="$(mktemp -d /tmp/ea-h1-independent-derived.XXXXXX)"
build_log="$(mktemp /tmp/ea-h1-independent-xcodebuild.XXXXXX)"

xcodebuild \
  -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Release" \
  -configuration Release \
  -derivedDataPath "$derived_root" \
  build 2>&1 | tee "$build_log"

Tests/CampaignOperationsPhaseH1CliTests.sh \
  "$derived_root/Build/Products/Release/LSTM_Release"

Tests/CampaignOperationsPhaseH1FinalAssuranceTests.sh \
  "$capture_root" "$run_id" "$build_log"

python3 Scripts/CampaignOperationsH1EvidenceGraph.py \
  validate "$capture_root" "$run_id" \
  "$capture_root/CampaignOperationsH1Traceability.md"

python3 Scripts/CampaignOperationsH1EvidenceGraph.py \
  validate "$capture_root" "$run_id" \
  "$capture_root/CampaignOperations_PhaseH_H1_ADR0019B_FinalEvidenceModelCorrection_Implementation_Output.md"

git diff --check
git status --short
git diff --stat
```

The independent accepted-negative cases must also be converted into permanent tests: forged generator/validator paths, ignored top-level/nested files, filename-inferred raw artifacts, semantic generic/restore/build replacements, and exact one-record report-delta assertions.

## 32. Repository review output

Files changed by this verification: **none**.

Behavioral change: **none; independent review only**.

`git diff --check`: passed.

`git diff --stat`:

```text
25 files changed, 10786 insertions(+), 20 deletions(-)
```

`git status --short` remains the pre-existing dirty H1 worktree: 25 tracked modifications/additions plus the untracked H1 reports, scripts, manifests, tests, fixture registries, generated traceability report, and ADR-0019A/B files. No verification artifact was added to the repository.

PHASE_H_H1_EVIDENCE_GRAPH_REPORTING_NOT_VERIFIED