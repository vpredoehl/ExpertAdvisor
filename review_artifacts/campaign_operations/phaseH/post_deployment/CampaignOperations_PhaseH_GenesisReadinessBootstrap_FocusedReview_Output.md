# Campaign Operations Phase H — Genesis Readiness Bootstrap Focused Review

Review type: read-only focused architectural/code review
Reviewed source commit: `fa2bf5ee8a3267a618d3980c36cc48f31ca6a254`
Reviewed Release executable SHA-256: `ca90330e60479921c6be8b4d8e47181d90f6672fdbef2fbf419fcbbffeba8e00`
Live state supplied by request: first valid enablement event only; zero production Admission V1 rows and zero production Attempt V2 rows.

## 1. Executive disposition

GENESIS_READINESS_DEFECT_CONFIRMED

The `canonical_contract_versions` blocker is intentionally fail-closed for existing production evidence with a wrong or mixed version, but its current implementation also treats the absence of all production Admission V1 and Attempt V2 evidence as a version mismatch. That makes the sanctioned first-dispatch path unreachable immediately after a valid genesis enablement.

The database transition function can be invoked directly by the dispatcher capability and can create the first evidence without evaluating the application readiness snapshot. That is an out-of-band readiness bypass, not a compliant genesis bootstrap mechanism. It does not cure the defect in the production CLI/Manager state machine.

## 2. Review scope and safety

No source, documentation, test, migration, manifest, or database file was edited except for creation of this requested report. No live SQL, production enable/disable, explicit dispatch, Manager run-once, supervisor, executable, or test harness was run. The repository was inspected at the supplied commit; initial `git status --short` was empty.

The review used static source, migration, test, documentation, and existing assurance-artifact inspection only.

## 3. Live-state interpretation

The current fields are the direct result of the readiness query and not evidence of a malformed row:

```text
admission_contract_version=missing
production_attempt_contract_version=missing
```

Migration 055 defines the readiness view as `string_agg` over the admission table and over attempts carrying production admission or enablement references ([055, lines 4042–4051](</Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:4042>)). With zero qualifying rows, PostgreSQL returns `NULL`; the C++ snapshot loader preserves `NULL` as an unset optional ([repository, lines 876–912](</Volumes/Developer%20SSD/ExpertAdvisor/Sources/CampaignOperationsProductionAdmissionRepository.cpp:876>)). Rendering converts an unset optional to `missing` ([service, lines 570–610](</Volumes/Developer%20SSD/ExpertAdvisor/Sources/CampaignOperationsProductionAdmissionService.cpp:570>).

The first production acquisition is the producer of both evidence families. The authoritative SQL transition creates an Admission V1 with `admission_contract_version := 1` and then an Attempt V2 with `attempt_contract_version := 2` in the same acquisition transaction ([055, lines 3834–3866](</Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:3834>), [055, lines 3895–3933](</Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:3895>)). Therefore missing values are expected before the first acquisition.

## 4. Readiness CLI trace

The executable routes `--campaign-operations-production-readiness` through the dedicated Manager DB principal to `RunProductionReadinessCommand` ([scheduler, lines 22820–22828](</Volumes/Developer%20SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:22820>)). The command opens a connection, calls `LoadProductionReadiness`, renders the result, and returns exit code 2 when blocked ([service, lines 682–693](</Volumes/Developer%20SSD/ExpertAdvisor/Sources/CampaignOperationsProductionAdmissionService.cpp:682>).

`LoadProductionReadiness` uses a repeatable-read, read-only transaction and calls `LoadProductionReadinessSnapshot`; it then calls `EvaluateProductionReadiness` ([service, lines 530–552](</Volumes/Developer%20SSD/ExpertAdvisor/Sources/CampaignOperationsProductionAdmissionService.cpp:530>). The snapshot loader reads the readiness view or the H2 read-only wrapper and maps the two aggregate columns to optional strings ([repository, lines 876–918](</Volumes/Developer%20SSD/ExpertAdvisor/Sources/CampaignOperationsProductionAdmissionRepository.cpp:876>).

The evaluator compares all five values unconditionally: scheduler, Manager build, enablement, admission, and production Attempt V2. A missing optional fails `ObservedVersionMatches`, and any failure adds exactly `canonical_contract_versions` ([service, lines 453–490](</Volumes/Developer%20SSD/ExpertAdvisor/Sources/CampaignOperationsProductionAdmissionService.cpp:453>). Thus the supplied live state deterministically yields `ready=false` with that blocker.

Repository hydration is stricter than the aggregate display. Existing evidence rows are structurally reconstructed, canonical/hash/audit/linkage checked, and their observed versions joined into `1`, `2`, or a mixed set ([repository, lines 951–997](</Volumes/Developer%20SSD/ExpertAdvisor/Sources/CampaignOperationsProductionAdmissionRepository.cpp:951>). This is the correct basis for rejecting existing bad evidence; it is not a valid reason to reject an empty evidence family.

## 5. Explicit production dispatch trace

The CLI constructs a `ProductionDispatchRequest`, captures and validates the running Release build, and calls `DispatchOneRequestForProduction` ([scheduler, lines 22876–22897](</Volumes/Developer%20SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:22876>).

The production service validates acknowledgement, operation key, request version, supplied build, and actual executable build. It then loads production readiness and throws `campaign_operations_production_readiness_blocked:<blockers>` before entering the dispatch adapter when `ready` is false ([dispatch service, lines 818–863](</Volumes/Developer%20SSD/ExpertAdvisor/Sources/CampaignOperationsDispatchService.cpp:818>). With the current genesis snapshot, this path stops at `canonical_contract_versions`; no acquisition transaction runs.

If readiness were true, the common adapter would:

1. perform exact binding/lease recovery lookup;
2. acquire a production lease under `campaign_operations_production_dispatcher` through `AcquireProductionDispatchLeaseInTransaction`;
3. call the protected `transition_campaign_operations_request_dispatch_production_v2` function;
4. hydrate the resulting Attempt V2 and its immutable Admission V1;
5. perform the established Phase E handoff under the production Phase 5 role.

The adapter separates acquisition and handoff transactions and has fresh-connection recovery for uncertain commits ([dispatch service, lines 296–389](</Volumes/Developer%20SSD/ExpertAdvisor/Sources/CampaignOperationsDispatchService.cpp:296>)). None of those recovery branches is entered when the readiness check rejects first.

## 6. Manager run-once trace

`RunCampaignOperationsManagerOnce` takes one optimistic candidate snapshot, creates a deterministic Manager operation identity, and processes candidates sequentially ([Manager service, lines 127–185](</Volumes/Developer%20SSD/ExpertAdvisor/Sources/CampaignOperationsManagerService.cpp:127>). The production branch calls `DispatchOneRequestForProductionManager`, not a special genesis path.

The Manager production adapter repeats the acknowledgement, operation-key, build, and readiness checks and rejects on any blocker ([dispatch service, lines 865–905](</Volumes/Developer%20SSD/ExpertAdvisor/Sources/CampaignOperationsDispatchService.cpp:865>). Consequently, the first eligible Manager candidate cannot create Admission V1 or Attempt V2 in the supplied live state. Manager classification treats `production_readiness_blocked` as a global Manager stop (`manager_build_not_ready`) ([Manager service, lines 24–40](</Volumes/Developer%20SSD/ExpertAdvisor/Sources/CampaignOperationsManagerService.cpp:24>).

There is no Manager-specific bootstrap exception in the production branch.

## 7. Replay, recovery, and SQL bootstrap routes

### Replay/recovery

The replay helper only selects an already existing Attempt V2 by request ID, operation key, and `attempt_contract_version=2`; if none exists it returns without creating anything ([055, lines 3236–3249](</Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:3236>). The C++ exact lookup similarly selects only an existing Attempt V2 ([repository, lines 861–874](</Volumes/Developer%20SSD/ExpertAdvisor/Sources/CampaignOperationsProductionAdmissionRepository.cpp:861>).

Therefore replay, uncertain-commit recovery, recovered-lease lookup, and exact binding replay cannot bootstrap genesis. They can only acknowledge or recover evidence that already exists. The architecture explicitly requires recovery to load the stored attempt/admission/enable chain and says lookup alone cannot bypass the current readiness gate ([architecture, lines 248–258](</Volumes/Developer%20SSD/ExpertAdvisor/docs/architecture/CampaignOperations_PhaseH_Production_Dispatch_Admission_and_Manager.md:248>).

### Protected SQL transition

`transition_campaign_operations_request_dispatch_production_v2` is the only SQL function authorized to establish the first admission, Boolean production witness, Attempt V2, and production audit atomically ([architecture, lines 684–710](</Volumes/Developer%20SSD/ExpertAdvisor/docs/architecture/CampaignOperations_PhaseH_Production_Dispatch_Admission_and_Manager.md:684>). Its creation path checks scheduler evidence, the current effective enablement, request locks, request state/version, and the compare-and-set predicate, then inserts the Admission V1 and Attempt V2 ([055, lines 3762–3823](</Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:3762>), [055, lines 3834–3950](</Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:3834>).

It does not call `LoadProductionReadiness`, inspect `canonical_contract_versions`, validate the actual Release executable, validate the full role/readiness report, or check reconciliation counts. Migration 056 deliberately grants its EXECUTE privilege to `campaign_operations_production_dispatcher` ([056, lines 117–153](</Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/056_campaign_operations_h2_privilege_deployment_contract.sql:117>), [ACL manifest](</Volumes/Developer%20SSD/ExpertAdvisor/Database/manifests/056_campaign_operations_h2_explicit_acl.tsv:4>)). The live Manager login is a member of that capability, so an out-of-band SQL caller can reach the transition from the current role graph.

This is a reachable readiness bypass, not an intended genesis mechanism: it can create the first Admission V1/Attempt V2 evidence without the application readiness gate, but it does not provide the sanctioned full CLI/Manager dispatch contract. It must not be used to activate production and was not invoked during this review.

## 8. Circularity analysis

For the sanctioned production state machine, the cycle exists:

```text
genesis enablement
  -> admission/Attempt V2 aggregates are NULL
  -> C++ adds canonical_contract_versions
  -> explicit CLI and Manager readiness gates reject
  -> transition that creates first admission/Attempt V2 is never called
```

The raw SQL transition breaks the cycle only by bypassing the application readiness contract. That is not a valid answer to “production dispatch requires readiness,” and it weakens the fail-closed boundary rather than proving intentional bootstrap semantics.

The first production Admission V1 / Attempt V2 is therefore not reachable from the current enabled state through a compliant explicit-dispatch or Manager run-once path. It is reachable only through the separately exposed SQL capability, which is an additional architectural exposure.

## 9. Normative architecture analysis

The documents support interpretation B:

> every persisted instance that exists must have the required contract version, while an uninstantiated contract family is valid during genesis.

Evidence:

- The architecture defines Admission V1 as the “first request production admission,” says there is exactly one per request, and says it uses the first production dispatch key ([architecture, lines 199–217](</Volumes/Developer%20SSD/ExpertAdvisor/docs/architecture/CampaignOperations_PhaseH_Production_Dispatch_Admission_and_Manager.md:199>)). It cannot simultaneously require an admission before the first dispatch and define that admission as the output of the first dispatch.
- The architecture defines Attempt V2 as the production attempt shape with contract version 2 ([architecture, lines 219–244](</Volumes/Developer%20SSD/ExpertAdvisor/docs/architecture/CampaignOperations_PhaseH_Production_Dispatch_Admission_and_Manager.md:219>)). The SQL producer sets version 2 at first acquisition; it is not a pre-enable deployment marker.
- The normative rollout order is readiness/status, explicit enablement, then one exact caller-keyed canary ([architecture, lines 832–846](</Volumes/Developer%20SSD/ExpertAdvisor/docs/architecture/CampaignOperations_PhaseH_Production_Dispatch_Admission_and_Manager.md:832>). This requires a valid post-enable readiness state before the first canary can be dispatched.
- The architecture says readiness is true only when every required item is exact, current, and complete ([architecture, lines 615–638](</Volumes/Developer%20SSD/ExpertAdvisor/docs/architecture/CampaignOperations_PhaseH_Production_Dispatch_Admission_and_Manager.md:615>). “Complete” for an uninstantiated first-admission/attempt family must mean no invalid persisted evidence exists; otherwise the documented rollout is unreachable.
- The H1 documentation explicitly calls admission and Attempt V2 values aggregates across relevant persisted evidence and says readable wrong or mixed observed values block ([H1 docs, lines 161–172](</Volumes/Developer%20SSD/ExpertAdvisor/docs/CampaignOperationsPhaseH1.rst:161>). It does not define an empty production history as wrong evidence.
- The same readiness design already recognizes an empty family for Completion: when no completed campaign exists, it uses deployed Completion V1 column-contract evidence instead of inventing a persisted completion row ([055, lines 3974–3989](</Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:3974>). That is consistent with distinguishing “not instantiated” from “instantiated with an invalid version.”

The phrase “all H1 contract versions” therefore cannot reasonably mean that every evidence family must already have a persisted instance before first dispatch. The implementation currently applies that stronger interpretation to Admission and Attempt V2 only, creating the defect.

## 10. Test coverage analysis

Existing tests prove important non-genesis behavior but do not prove the required genesis readiness state:

- `Tests/CampaignOperationsPhaseH1Tests.cpp` constructs a ready snapshot with `admission_contract_version=1` and `production_attempt_contract_version=2`, and verifies a wrong observed version produces `canonical_contract_versions` ([lines 208–306](</Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1Tests.cpp:208>). It has no ready snapshot with both families absent.
- `Tests/CampaignOperationsPhaseH1RepositoryTests.cpp` mutates existing Admission and Attempt evidence to versions 2/3 and verifies readable wrong and mixed values block ([lines 766–818](</Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1RepositoryTests.cpp:766>). It also exercises corruption failures, but it does not assert that a valid enablement with zero production rows yields readiness true.
- `Tests/CampaignOperationsPhaseH2WorkflowTests.cpp` enables a disposable database and dispatches through `DispatchOneRequestForProductionForTest` ([lines 133–180](</Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH2WorkflowTests.cpp:133>). That test-only adapter bypasses the production readiness evaluation, so it proves the SQL acquisition/recovery workflow, not the real post-enable production readiness gate.
- `Tests/CampaignOperationsPhaseH3RuntimeConcurrencyTests.cpp` runs actual Manager logic in disposable cloned databases, but its fixture adapter intentionally skips only `canonical_contract_versions` because historical Attempt V1 rows pollute the aggregate summary ([source, lines 907–947](</Volumes/Developer%20SSD/ExpertAdvisor/Sources/CampaignOperationsDispatchService.cpp:907>)). The harness artifact explicitly records that production Manager/CLI readiness is unchanged ([H3 artifact, lines 40–45](</Volumes/Developer%20SSD/ExpertAdvisor/review_artifacts/campaign_operations/phaseH/h3/CampaignOperations_PhaseH_H3_RuntimeConcurrency_Harness_Output.md:40>).
- The H1/H2 migration tests directly invoke the protected SQL transition in isolated databases, including raw acquisition and replay checks ([migration test, lines 1058–1075](</Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1MigrationTests.sql:1058>). This confirms the transition’s independent SQL reachability, not compliant application readiness.

The scripts’ isolation mechanisms are disposable PostgreSQL clusters/databases under `/tmp`, with H2/H3 cloning the H1 fixture; they do not use the live LSTM database. They were reviewed but not run for this read-only review.

Missing targeted coverage:

1. genesis enablement with zero Admission V1 rows and zero production Attempt V2 rows;
2. readiness immediately after that enablement with all healthy fields and no version blocker;
3. real `DispatchOneRequestForProduction` creating the first admission/Attempt V2;
4. real `RunCampaignOperationsManagerOnce` creating the first admission/Attempt V2;
5. a paired negative case where at least one existing row has a wrong/mixed version and readiness still blocks;
6. a paired malformed-evidence case proving malformed rows are not mistaken for an empty genesis family.

## 11. H3 fixture special-case significance

The H3 fixture’s `if (blocker == "canonical_contract_versions") continue;` is explicitly scoped to `DispatchOneRequestForProductionManagerWithFixture`, compiled only under H3 testing. Its comment says the repository-native H1 fixture retains immutable historical Attempt V1 materializations, so only the aggregate version summary is non-ready; runtime authority and other readiness blockers remain enforced ([dispatch service, lines 907–947](</Volumes/Developer%20SSD/ExpertAdvisor/Sources/CampaignOperationsDispatchService.cpp:907>)).

This establishes two semantics:

- `canonical_contract_versions` is intended to be an aggregate evidence-quality blocker, not an independent authorization source; and
- historical non-production Attempt V1 materializations must not prevent the disposable fixture from exercising H3 runtime behavior.

It does not prove that production should ignore the blocker. It also does not provide a production genesis exception: the production adapters at lines 844–858 and 885–900 enforce every blocker.

## 12. Defect classification

Severity: High operational correctness defect, with a separate high-risk readiness-bypass exposure.

Safety impact:

- The normal production CLI and real Manager path fail closed, so they do not dispatch unsafely from the current genesis state.
- The same deployment exposes the authoritative SQL acquisition transition to the dispatcher capability without an in-function readiness check. A caller with that capability can create production acquisition evidence while build, role, completion-proof, reconciliation, or aggregate readiness conditions are not satisfied. This is not a safe bootstrap path and should be treated as a separate boundary finding.

Does it block first production activation? Yes for the sanctioned explicit CLI and Manager run-once paths. The live enablement event alone cannot make either path reach first acquisition.

Do migration/schema bytes need to change? No schema or data migration is required for the genesis semantic correction. The narrow implementation can use read-only evidence-presence information from the same repeatable-read snapshot and adjust readiness evaluation. The immutable 055 migration should not be rewritten.

Does only readiness interpretation need to change? For the genesis blocker, yes. The SQL producer already writes Admission V1=1 and Attempt V2=2 atomically. The separate direct-SQL readiness bypass is not needed to solve genesis and should be handled as a distinct access/workflow review rather than by weakening readiness.

Documentation/tests: yes. Documentation should state the empty-genesis exception explicitly, and tests should cover empty, valid, wrong, mixed, malformed, and missing-link cases separately.

## 13. Narrow correction recommendation — not implemented

Change only the readiness interpretation so that an evidence family with zero relevant persisted production rows is treated as “not yet instantiated,” not as a wrong version:

- zero Admission V1 rows: accept the admission family as genesis-empty;
- zero production-linked Attempt rows: accept the Attempt V2 family as genesis-empty;
- any existing Admission row: require aggregate exactly `1`;
- any existing production Attempt row: require aggregate exactly `2`;
- any mixed, wrong, malformed, stale-canonical, broken-link, missing-required-link, or structurally unreadable row: retain fail-closed behavior and preserve the existing `canonical_contract_versions` or integrity failure classification.

The empty check must be based on explicit row/evidence presence, not on `NULL` alone. Otherwise a malformed row whose version is NULL could be misclassified as genesis. The same repeatable-read snapshot must distinguish:

```text
no relevant rows                 => genesis-empty, allowed
rows exist and aggregate = 1/2   => allowed for that family
rows exist and aggregate wrong/mixed => canonical_contract_versions
row exists but cannot hydrate     => integrity failure
```

All other blockers remain unchanged: enablement/build mismatch, role mismatch, scheduler evidence mismatch, missing links, malformed canonical evidence, mixed versions, blocked leases, and reconciliation blockers must continue to reject production readiness. No test-only exception should be added to production adapters.

The direct dispatcher-capability SQL route should not be used as the correction. If the normative contract requires every production acquisition to be readiness-gated at the database boundary, its separate reachability should receive a narrowly scoped follow-up review for enforcement/ACL design; that question is distinct from the genesis aggregate interpretation.

## 14. Explicit next action

READY_FOR_GENESIS_READINESS_TARGETED_CORRECTION
