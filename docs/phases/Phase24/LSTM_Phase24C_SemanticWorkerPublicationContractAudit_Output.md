# Phase 24C — Semantic-worker publication contract audit

Audit date: 2026-10-07, America/Chicago. Repository: `/Volumes/Developer SSD/ExpertAdvisor-Rollover`. Audited HEAD after Phase 24B closure: `7222021fd41ad71f6b7118914613b550f66405df`, branch `dedicated-train-layout-rollover-squashed-v1`.

Phase 24C is inspection only, with this report as its sole file output. No publication script, executable source, project, registry, or artifact was changed. No real candidate was executed or published. No Xcode build, PostgreSQL operation, scheduler/worker operation, Qwen loading, Ollama startup, production checkout modification, or Phase 24C commit occurred.

## 1. Executive findings

**GO for a narrow implementation assignment; NO-GO for treating the current checks as complete qualification of a new dedicated worker pair.** The existing publication architecture should remain intact. Its content-addressed artifacts, runtime verification, publication lock, and registry commit boundary are substantive safeguards. The missing link is proof that each candidate's compiled semantic contract matches the contract being recorded.

1. **Dedicated rollover qualifies TRAIN's compiled layout and width, but not INFER's.** TRAIN prints both constants; rollover compares them with the source-derived contract. INFER prints role, source commit and self-hash but no semantic fields. Its shared verifier checks those three identity values only. Evidence: `LSTM/TrainWorkerMain.cpp:60–76`, `LSTM/InferWorkerMain.cpp:59–73`, `Scripts/RollSemanticWorkerLayout.py:223–232`, `Scripts/PublishSemanticWorker.py:211–234`.
2. **Same-layout refresh qualifies neither candidate's compiled layout/width.** It verifies both role/commit/hash identities and checks that the prior current registry matches the source layout/width. It then assigns that contract to both new manifests. TRAIN's already-available semantic fields are unused. Evidence: `Scripts/RefreshSemanticWorkerGeneration.py:_validate_refresh_prestate:45–65`, `refresh:82–112`, `refresh_from_repository:180–188`.
3. **The infer-only CLI has an additional source-contract escape in its argument handling.** When both layout and width are supplied, even a current publication skips `current_semantic_contract` and the explicit-versus-source comparison. The current commit must still equal clean HEAD, and current layout cannot advance. Those protections do not check compiled semantics or prevent a wrong explicit width. Evidence: `Scripts/PublishSemanticWorker.py:871–901`, `publish:716–723,765–770`.
4. **Registry/hash/runtime checks do not close these qualification gaps.** They establish consistent metadata and exact artifact bytes, not the meaning of those bytes. The C++ consumer checks current TRAIN/INFER width agreement, but never obtains either binary's compiled identity during registry parsing. Selection deliberately trusts the validated registry's layout/width contract. Evidence: `Scripts/PublishSemanticWorker.py:329–498,612–631`, `Sources/SchedulerCore/SemanticWorkerRegistry.cpp:632–665,786–798,1081–1093`.
5. **Legacy historical workflows intentionally use a different evidence contract.** They accept archived `LSTM_Release`, explicit historical layout/width/commit, embedded commit evidence and immutable packaging, without requiring a modern dedicated-worker identity. That is documented and covered by compatibility tests; absence of the modern check alone is not a reason to invalidate historical artifacts. Evidence: `Scripts/PublishHistoricalInferenceWorker.py:35–56`, `Scripts/PublishHistoricalTrainingCandidate.py:40–66`, `docs/semantic-workers/semantic-layout-inference-worker-routing.md:186–210`, `Tests/SemanticWorkerHistoricalInferenceWorkerPublisherTests.py:103–143`.

The current source baseline is layout 13 / width 171, from `Headers/ModelInputExpansion.hpp:23` and `Headers/ModelInputContract.hpp:88–90`. No real production candidate or registry was inspected; this report establishes source-visible acceptance gaps, not an observed incompatible deployment.

### Evidence method and MCP isolation

Only `expertadvisor-repository-rollover` was used. `capabilities` reported protocol `expertadvisor.repository.readonly.v1`, read-only repository access, and forbidden shell/repository-write/Git/database/build/test/arbitrary-filesystem capabilities. No semantic verification or investigation tool was called.

The configured source root and its resolved path are `/Volumes/Developer SSD/ExpertAdvisor-Rollover`; the cache namespace is `ExpertAdvisor-Rollover`; the configured and resolved claim-ledger path is `/Users/vjp/Library/Caches/ExpertAdvisor-Rollover/RepositoryAgent/verified_claims.json`. No claim-ledger write was requested. Source-root/configuration verification used a selective read of the rollover connection in `/Users/vjp/.codex/config.toml`, without accessing the production connection.

Deterministic `index_stats` returned 534 files, 5,487 functions, 4,367 definition keys, 33,328 reference keys and 50,783 call edges. `resolve_symbol` returned `EA::Training::RunDedicatedTrainWorkerMain`, its identity/application callees, and its caller. Exact MCP reads covered TRAIN identity (`LSTM/TrainWorkerMain.cpp:60–97`), INFER identity (`LSTM/InferWorkerMain.cpp:1–89`) and the registry selection boundary (`Sources/SchedulerCore/SemanticWorkerRegistry.cpp:1078–1093`), consistent with direct checkout reads.

MCP rejected `Scripts/PublishSemanticWorker.py:211–234` as outside its read-only source boundary. Publication scripts, tests, project configuration and documentation were therefore inspected directly in the authorized rollover checkout with bounded line reads and textual/AST searches. The MCP rejection was a coverage limitation, not a publication failure. No Qwen-backed tool or fallback model service was used.

## 2. TRAIN publication contract

### Publication entrypoint inventory

The checkout contains five semantic-worker publication CLIs. Each delegates to the existing shared publisher/staging machinery; there is no automatic candidate scanner or build step in these publication CLIs.

| Entrypoint | Candidate/contract inputs | Authority changed | Concrete evidence |
| --- | --- | --- | --- |
| `Scripts/PublishSemanticWorker.py:main` → `publish` | Explicit `--built-executable`; INFER only; current or historical rule | One layout's INFER binding; cannot advance current layout | `parse_arguments:852–863`, `main:866–902`, `validate_inputs:260–262`, `publish:765–770,825–848` |
| `Scripts/RollSemanticWorkerLayout.py:main` → `rollover_from_repository` → `rollover` | Explicit TRAIN/INFER paths; source-derived contract; legacy TRAIN by default or `--dedicated-training`; optional independent commits in dedicated mode | Both roles and new current layout in one replacement | `301–349`, `211–251`, `279–297` |
| `Scripts/RefreshSemanticWorkerGeneration.py:main` → `refresh_from_repository` → `refresh` | Explicit dedicated TRAIN/INFER paths; source-derived contract; optional per-role commits | Both current bindings at unchanged layout | `171–211`, `82–112`, `139–167` |
| `Scripts/PublishHistoricalTrainingCandidate.py:main` → `append_historical_training_candidate` | Explicit legacy `LSTM_Release`, historical contract, capabilities and priority | Append one historical TRAIN candidate | `26–66`, `79–100`, `117–148` |
| `Scripts/PublishHistoricalInferenceWorker.py:main` → `publish_historical_inference_worker` | Explicit legacy `LSTM_Release` and historical contract | Add one historical INFER binding | `26–58`, `65–99`, `103–121` |

`Scripts/PublishCanonicalLSTMRelease.py` is an adjacent ordinary-release publisher, not a semantic-worker entrypoint: its documented scope explicitly excludes semantic-worker artifacts and registry (`:2–7`). `GenerateBuildProvenance.py` supplies provenance, not registry publication (`:79–85`). Searches of the available Scripts and Xcode project did not identify another semantic publication wrapper. This inventory is limited to the checkout, not external operator tooling.

### Dedicated TRAIN, end to end

1. **Candidate selection:** the operator supplies the exact path. `_resolve_executable` requires an absolute path, resolves it strictly, and requires an executable regular file whose resolved basename is `lstm-train-worker` in dedicated mode. It does not discover a binary by newest modification time or search DerivedData. Evidence: `Scripts/RollSemanticWorkerLayout.py:42–54,212–214,328–337`.
2. **Expected contract and Git provenance:** `rollover_from_repository` requires clean source, derives layout/width by compiling and running a small source-header probe, and delegates to `rollover`. This probe measures the checkout's constants, not the supplied worker's compiled constants. In dedicated mode explicit TRAIN and INFER commits may differ from each other and from the workflow checkout; each is format-checked and bound to its own binary. Legacy mode requires clean HEAD and prohibits independent INFER commits. Evidence: `Scripts/PublishSemanticWorker.py:59–92`, `Scripts/RollSemanticWorkerLayout.py:215–222,301–324`.
3. **Build identity:** Release provenance generation requires a clean checkout, reads exact HEAD and emits `EXPERTADVISOR_SOURCE_COMMIT`; non-Release generation emits an empty marker. The TRAIN entrypoint returns its identity before entering the application. Its record contains version 1, role, source commit, compiled semantic layout, compiled width, canonical path and self-hash. Evidence: `Scripts/GenerateBuildProvenance.py:40–56,79–85`, `LSTM/TrainWorkerMain.cpp:29–48,51–76,87–97`, Xcode provenance phase `ExpertAdvisor.xcodeproj/project.pbxproj:2192–2210`.
4. **Commit, SHA and semantic validation:** `verify_embedded_commit` requires an exact output line from `strings`; Python independently hashes the candidate; `verify_worker_build_identity` compares reported role, commit and `sha256:<digest>`; `verify_train_semantic_contract` separately compares reported role/layout/width with the expected contract. Thus dedicated rollover performs two TRAIN identity invocations. Explicit commit validation proves format, embedded text and identity agreement; it does not resolve every explicit retained commit as a Git object or prove reproducibility. Evidence: `Scripts/PublishSemanticWorker.py:95–100,171–182,211–229`, `Scripts/RollSemanticWorkerLayout.py:57–74,223–228`.
5. **Runtime and manifest:** both candidates must have identical hashes for `MetaNN_metal.metallib`/`MetaNN.metallib` and `default.metallib`. A deterministic runtime manifest supplies a content address. Dedicated TRAIN capabilities are `[train]`, or `[train,train_feature_ablation_v1]` only with the explicit qualification flag. The manifest and registry record use the validated requested layout/width, actual commit and independently computed hash. Evidence: `Scripts/RollSemanticWorkerLayout.py:77–97,100–139,239–251`, `Scripts/PublishSemanticWorker.py:196–208,501–529`.
6. **Immutable storage:** the dedicated address is `layout<N>/train/<commit>/<sha256>/lstm-train-worker`, with schema-v2 manifest. `_stage_worker` checks conflicts, copies and re-hashes bytes, applies executable mode 0555 and manifest mode 0444, fsyncs and renames staging, then installs verified runtime links. Existing addresses require exactly matching bytes and manifest; they are not overwritten. Evidence: `Scripts/RollSemanticWorkerLayout.py:100–175`, `Scripts/PublishSemanticWorker.py:612–676,679–693`.
7. **Registry publication:** under `.publish.lock`, load and validate the prior registry; reject the already-current/already-registered target layout; require prior current TRAIN and INFER bindings; stage both candidates; retain the outgoing current bindings as historical; validate the full prospective registry; replace `registry.json` once; then update the non-authoritative `current` link to INFER. Evidence: `Scripts/RollSemanticWorkerLayout.py:178–194,253–297`.
8. **Rollback/failure boundary:** before registry replacement succeeds, failed staging or prospective validation leaves the prior registry authoritative. Finalized but unregistered immutable artifacts can remain and be reused after a precommit failure. A failure of the later convenience-link update is explicitly committed, not a rollback. A successful rollover repeated against the registered target is rejected by prestate validation, so retry is not an unconditional successful no-op. Evidence: `Scripts/RollSemanticWorkerLayout.py:148–174,183–186,290–297`, `Scripts/PublishSemanticWorker.py:126–164`; failure tests in `Scripts/tests/test_dedicated_train_rollover.py:170–361`.

### Other TRAIN routes

Legacy rollover accepts `LSTM_Release`, checks embedded commit and hash, uses schema-v1 paths/manifests and broad legacy capabilities, and omits dedicated identity/semantic checks (`Scripts/RollSemanticWorkerLayout.py:212–244`). Historical TRAIN append also accepts only `LSTM_Release`; it requires a pre-existing historical layout/width binding, explicit ablation qualification, a new artifact identity and an appended selection priority. It preserves current layout/rules and validates prospective state before replacement (`Scripts/PublishHistoricalTrainingCandidate.py:40–66,79–124`). These are compatibility/attestation routes, not evidence that a new dedicated TRAIN binary was semantically qualified.

## 3. INFER publication contract

### Dedicated paired publication

Rollover and refresh receive an explicit `lstm-infer-worker` path through the same strict resolver. They accept an independent INFER source commit where supported; verify its embedded commit, compute its SHA-256, compare reported role/commit/self-hash, and match its two runtime resources to TRAIN's. The INFER manifest is schema v2, capabilities exactly `[infer]`, stored at `layout<N>/infer/<commit>/<sha256>/lstm-infer-worker`. Paired registry staging, prospective validation and atomic replacement are shared with TRAIN. Evidence: `Scripts/RollSemanticWorkerLayout.py:42–54,214–232,248–297`; `Scripts/RefreshSemanticWorkerGeneration.py:82–112,119–167`.

However, `LSTM/InferWorkerMain.cpp:59–73` emits no `semantic_layout` or `model_input_width`. `verify_inference_build_identity` is only a compatibility wrapper around the role/commit/hash validator (`Scripts/PublishSemanticWorker.py:211–234`). Both INFER manifests therefore record the workflow's requested contract without measuring the candidate's compiled contract.

### Infer-only publication

`PublishSemanticWorker.py:main` supports current and historical dedicated INFER publication. Current mode requires clean HEAD and rejects a differing explicit commit; historical mode requires explicit commit, layout and width. With either layout/width absent, current mode probes source and checks supplied values against it. With both present, it accepts the explicit pair without that probe/comparison (`:871–890`).

`publish` validates path, executable permission, positive contract fields, commit format, rule and infer-only capabilities; verifies embedded commit; hashes the bytes and checks dedicated identity; hashes and stages runtime resources; loads a valid prior registry under the lock; requires existing training/reference authority and forbids changing current layout. It stages a schema-v2 artifact, replaces the INFER binding for the requested layout, preserves immutable disk artifacts and TRAIN binding, then replaces registry JSON and updates `current` only for current rule. Evidence: `Scripts/PublishSemanticWorker.py:237–262,696–848`.

Two limits matter: `publish` never compares compiled semantic fields; it also lacks the full prospective-state `validate_existing_registry` call present in paired workflows (`:825–845`). Existing-registry validation before staging is real, but not final validation of the newly constructed state. The current CLI also does not compare a supplied width with the current TRAIN binding. The C++ reader subsequently rejects current-role width disagreement (`Sources/SchedulerCore/SemanticWorkerRegistry.cpp:786–798,845–855`), which protects scheduler admission but does not prevent the publisher from replacing the registry with that metadata.

### Historical INFER compatibility

The separate historical bootstrap intentionally accepts legacy `LSTM_Release` and embedded commit evidence without `--build-identity`. It stages schema-v1 `[infer]` artifacts, rejects a duplicate target binding, validates the prospective registry and preserves current layout/rules/link. Evidence: `Scripts/PublishHistoricalInferenceWorker.py:35–99`; documented contract at `docs/semantic-workers/semantic-layout-inference-worker-routing.md:186–210`. Do not impose the modern dedicated identity on existing historical archives as part of this repair.

## 4. Contract comparison

| Check | Dedicated TRAIN rollover | Dedicated INFER rollover | Dedicated same-layout refresh | Infer-only publisher |
| --- | --- | --- | --- | --- |
| Candidate selection | Explicit absolute path; strict basename | Same | Same for both roles | Explicit candidate, strict resolution/file/permission; no basename requirement in `validate_inputs` |
| Expected layout/width | Checkout-header probe | Same expected pair | Checkout-header probe plus prior current registry agreement | Source probe only when one field is omitted; otherwise explicit values |
| Compiled layout | **Compared** | **Not emitted or compared** | **Neither role compared** | **Not compared** |
| Compiled input width | **Compared** | **Not emitted or compared** | **Neither role compared** | **Not compared** |
| Worker role | Reported TRAIN role checked | Reported INFER role checked | Both reported roles checked | Reported INFER role checked |
| Source commit | Embedded exact line plus identity equality | Same, independently supplied commit allowed | Same for each role | Same; current commit also clean HEAD |
| Artifact SHA-256 | Independent hash plus reported self-hash; staged bytes re-hashed | Same | Same | Same |
| Runtime package | Both candidates' resource hashes equal | Same | Same | Hash/manifest/link verification, without paired resource comparison |
| Registry compatibility | Prior and full prospective validation; new layout with both roles | Shared atomic state | Prior/current contract and full prospective validation; unchanged layout | Prior validation and current-layout gate; no full prospective validation |
| Identity version/record shape | Version/header/duplicate keys not enforced by validators | Same | Same | Same |

Table evidence: `Scripts/PublishSemanticWorker.py:211–262,612–631,696–848,871–901`; `Scripts/RollSemanticWorkerLayout.py:42–97,219–232,253–324`; `Scripts/RefreshSemanticWorkerGeneration.py:45–65,82–112,119–188`.

**Direct answers for INFER:** role, source commit and artifact hash are checked for modern dedicated candidates. Compiled semantic layout and input width are not. Registry compatibility is checked against stored metadata, artifacts and runtimes; paired workflows additionally validate the complete prospective registry. These checks do not derive a binary's semantic constants.

The identity parser accepts any comma-separated output containing the three matching fields: it does not require the record header or `identity_contract_version`, rejects no duplicate key, and uses the last occurrence of a key. TRAIN's separate semantic parser has the same shape permissiveness. This is a confirmed parser limitation, not proof of malicious candidates or corrupted production state (`Scripts/PublishSemanticWorker.py:221–229`, `Scripts/RollSemanticWorkerLayout.py:65–74`).

## 5. Same-layout refresh analysis

Refresh is explicitly a code-only generation replacement: source layout/width and prior current registry must agree (`Scripts/RefreshSemanticWorkerGeneration.py:2–6,45–65`). It cannot advance layout. The wrapper permits explicit independent per-role commits even though it first requires a clean workflow checkout (`:171–188`). This is an intended retained-artifact feature, exercised by `Tests/SemanticWorkerGenerationRefreshTests.py:271–287,367–376`.

The missing candidate-semantic comparison matters precisely at that boundary. A workflow checkout and prior registry can both say layout 13 / width 171, while a retained INFER binary from a different approved commit is compiled for layout 12 / width 127. If its exact role/commit/hash identity and runtime-resource equality pass, `refresh` creates manifests labeling both candidates 13 / 171; prospective validation sees consistent labels and bytes. The same issue applies to TRAIN refresh, because its reported semantic fields are ignored. This is a source-derived possible execution path, not a live publication performed during the audit (`:89–112,161–162`).

Retention is asymmetric by design: the outgoing current TRAIN is retained as historical, the outgoing INFER singleton binding is retired from the registry but its immutable directory remains on disk, and the new TRAIN gets a priority above the existing candidates at the same layout/width. Both new bindings are then committed together (`:139–167`). This preserves historical TRAIN candidate selection; it does not prove numerical parity or semantic compatibility.

Runtime admission provides additional defense after publication. `SemanticWorkerRegistry::Load` compares bytes, manifest metadata and current-role widths, and `SelectWorkerForRole` selects on persisted identity and declared registry contract (`Sources/SchedulerCore/SemanticWorkerRegistry.cpp:632–665,786–798,983–1093`). Scheduler startup deliberately supplies no current-layout/width expectation tied to the scheduler's own compilation (`Sources/SchedulerCore/ProductionSchedulerDaemon.cpp:10871–10876`). Managed inference reads model materialization and validates semantic markers before numerical work (`Sources/ManagedInferenceApplication.cpp:155–163,263–275`; `Headers/PgModelIO.hpp:429–458,1109–1112`; `Sources/InferenceRuntime.cpp:64–82`). These guards may reject incompatible work later; they neither prevent mislabeled publication nor justify claiming that incorrect outputs will occur.

Failure semantics are also precise: failures before `os.replace(registry)` preserve old authority; later convenience-link failures retain committed authority and raise a distinct error. `atomic_write_json` performs directory fsync after replacement, so a failure at that fsync cannot truthfully guarantee restoration of old registry bytes. Current precommit failure tests mock the whole writer, while link-fsync tests cover a later boundary. This is an untested durability/error-reporting case, separate from candidate semantics (`Scripts/PublishSemanticWorker.py:126–164`; `Tests/SemanticWorkerGenerationRefreshTests.py:191–249`). No rollback API or artifact deletion is part of these publication workflows.

## 6. Existing regression coverage

The six publication-related suites were reviewed by source/AST; their 62 test methods were **not run during this read-only Phase 24C audit**. Previous Phase 24A/24B test results remain historical evidence, not fresh executions here.

| Suite | Methods | Existing coverage and limits |
| --- | --- | --- |
| `Tests/SemanticWorkerPublisherTests.py` | 12 | Schema upgrades, capabilities, TRAIN role/commit/hash helper behavior, immutable artifacts, registry failure, alias/hash/runtime rejection. Publication fixtures bypass candidate identity with `check_embedded_commit=False` (`:86–95`); helper tests do not require compiled semantic fields (`:164–210`). |
| `Tests/SemanticWorkerRolloverTests.py` | 13 | Role/layout matrix, ablation qualification, historical retention, candidate/prestate failures, source-cleanliness and explicit-commit checks, C++ registry consumer fixture (`:110–315`). Most staging tests bypass candidate identity (`:103–108`); INFER preflight failure is mocked (`:254–271`). |
| `Scripts/tests/test_dedicated_train_rollover.py` | 12 | Dedicated schema/capabilities, basename/commit rejection, previous generation retention, immutable conflicts and pre/postcommit failures (`:27–361`). TRAIN semantic helper rejects wrong/missing contract (`:363–388`). Distinct-commit test stops before INFER identity validation and mocks TRAIN semantic validation (`:390–414`). |
| `Tests/SemanticWorkerGenerationRefreshTests.py` | 13 | Current prestate/layout/width checks, paired replacement, retained TRAIN priority, independent commits, corruption and link failures (`:112–376`). Main publication helper bypasses identity (`:103–110`); layout/width rejection tests compare requested values with registry, not a compiled candidate (`:271–294`); generic identity failure is mocked (`:313–340`). |
| `Tests/SemanticWorkerHistoricalTrainingCandidatePublisherTests.py` | 6 | Explicit ablation qualification, append-only priority/identity checks, staging retry/conflict safety and v4 upgrade compatibility (`:126–324`). Legacy schema contract should remain supported. |
| `Tests/SemanticWorkerHistoricalInferenceWorkerPublisherTests.py` | 6 | Explicit legacy identity-interface compatibility, duplicate-binding/invalid-input rejection, commit/runtime/registry failure and CLI summary (`:103–253`). |

Missing regression coverage: modern INFER missing/wrong layout or width; refresh TRAIN missing/wrong compiled contract; candidate contracts differing from each other while manifests would agree; full explicit infer-only CLI fields disagreeing with source; current INFER width disagreeing with TRAIN binding before replacement; malformed identity headers/versions/duplicate required fields; positive identity preflight with valid independent commits through complete paired workflows; assertions that semantic rejection occurs before staging/registry/link changes; and registry-directory fsync failure after replacement.

### Read-only diagnostic probes actually run

A transient inline Python probe imported the unchanged rollover/publisher modules with bytecode writes disabled, replaced `subprocess.run` with fixed `CompletedProcess` results, and called only identity validators. No binary, `strings`, filesystem publication function, build, or registry was executed. All candidate paths were unused mock paths under the rollover root.

| Probe | Observed result |
| --- | --- |
| Generic INFER verifier, matching role/commit/hash, semantic fields absent | ACCEPTED |
| Generic INFER verifier, matching identity, layout 12 / width 127 | ACCEPTED |
| Generic INFER verifier, identity version 99 | ACCEPTED |
| Generic INFER verifier, wrong role / commit / hash, separately | REJECTED in all three cases |
| Generic TRAIN verifier with the same six cases | Same results |
| TRAIN semantic helper expecting 13 / 171, missing fields or 12 / 127 | REJECTED |
| TRAIN semantic helper expecting 13 / 171, matching fields | ACCEPTED |

These 15 mocked invocations establish validator behavior, not executable or registry qualification. A read-only AST call inventory also confirmed: `rollover` calls `verify_train_semantic_contract` and final registry validation; `refresh` omits the semantic helper; `publish` omits both a semantic comparison and final registry validation. Functions and line evidence: `Scripts/PublishSemanticWorker.py:211–234,696–848`, `Scripts/RollSemanticWorkerLayout.py:57–74,197–298`, `Scripts/RefreshSemanticWorkerGeneration.py:68–168`.

## 7. Confirmed defects versus hypothetical risks

| Classification | Finding | Evidence / scope limit |
| --- | --- | --- |
| Confirmed modern qualification defect | INFER compiled layout/width are unavailable in identity and never compared before dedicated publication | INFER entrypoint `:59–73`; shared verifier `PublishSemanticWorker.py:211–234`; rollover `:229–251`; refresh `:93–112`; mocked acceptance probes |
| Confirmed modern qualification defect | Refresh ignores TRAIN's already-reported compiled contract | TRAIN entrypoint `:60–76`; refresh `:89–112`; generic TRAIN verifier acceptance probes |
| Confirmed current infer-only contract gap | Fully explicit layout/width skip current-source comparison; wrong width can reach registry replacement without a paired-width/final prospective gate | Publisher `:871–885,765–770,825–845`; C++ current-width rejection `SemanticWorkerRegistry.cpp:786–798`. Source-visible path; no registry written here. |
| Confirmed parsing limitation | Identity version/header and duplicate required keys are not enforced | Publisher `:221–229`; rollover `:65–74`; version-99 acceptance probe. Harden when requiring semantic fields; do not claim an observed attack. |
| Intentional compatibility contract | Archived legacy TRAIN/INFER publication lacks modern compiled-field validation | Historical publisher functions and documented legacy bootstrap cited above; explicit legacy compatibility tests. No retrospective artifact rewrite justified. |
| Hypothetical operational consequence | Retained old-layout INFER with valid role/commit/hash and matching resources could be labeled as a new/current contract | Independent commits are allowed in rollover/refresh; manifests are generated from requested fields. No real incompatible candidate or deployment was observed. |
| Hypothetical numerical consequence | Mislabeled publication could cause incorrect feature interpretation or rejected work | Equal width is not sufficient for semantics: layouts 6 and 7 have the same physical width with a corrected interpretation (`Headers/ModelInputExpansion.hpp:48–53`). Worker admission/materialization also rejects some mismatches. Neither erroneous results nor production interruption was demonstrated. |
| Separate untested failure case | Directory fsync failure after registry replace has committed/uncertain-durability semantics without the distinct link error | `PublishSemanticWorker.py:126–164`; existing failure-test boundaries above. Preserve this qualification in operational descriptions; it is not grounds for a publication-engine redesign in the semantic increment. |

SHA-256 verifies exact bytes; a source commit identifies claimed provenance; identical metallibs verify packaged resources. None independently proves that recorded layout/width equals the supplied executable's compiled constants. Conversely, a missing modern identity on an approved historical archive is not itself proof of incompatibility.

## 8. Recommended implementation scope

Proceed only under a subsequent implementation assignment. The smallest coherent repair covers **new dedicated candidates across all modern publication routes**, reusing the current workflows:

1. Add INFER compiled `semantic_layout` and `model_input_width` to its identity record, using the same authoritative constants as TRAIN (`LSTM/InferWorkerMain.cpp:59–73`, `LSTM/TrainWorkerMain.cpp:60–76`). Do not invent a semantic layout increment for this code-only repair.
2. Reuse shared role-aware identity parsing/validation to check required record version/shape and candidate semantics alongside role/commit/hash. Preserve independent immutable commits. Avoid duplicating parsers or making two inconsistent identity reads the normal qualification mechanism (`Scripts/PublishSemanticWorker.py:211–234`, `Scripts/RollSemanticWorkerLayout.py:57–74`).
3. Require both dedicated candidates' measured contracts to match the intended pair in dedicated rollover and same-layout refresh; preserve existing lock, artifact, capability, prospective-registry and link boundaries (`RollSemanticWorkerLayout.py:223–232`, `RefreshSemanticWorkerGeneration.py:89–112,161–167`).
4. Close the same bypass in standalone **new dedicated INFER** publication: compare candidate semantics with requested contract; always compare explicit current contract with the source contract and current TRAIN width; validate prospective state before registry replacement. This belongs in the already-audited `PublishSemanticWorker.py`, not a new publishing subsystem (`:716–723,765–770,825–845,871–885`).
5. Add focused regression coverage and update the operational documentation to describe dedicated mode, independent commits, measured candidate contracts and precise failure boundaries. Existing rollover documentation still describes only legacy TRAIN, identical commits and prior v4 state (`docs/semantic-workers/semantic-layout-inference-worker-routing.md:229–253`), while implementation supports dedicated TRAIN and schema v5 (`RollSemanticWorkerLayout.py:178–194,301–337`).

Define the retained-INFER qualification policy before implementation acceptance. Older dedicated INFER candidates presently lack semantic fields. Prefer a fresh independently built candidate that reports them. If retained candidates must be used for a new publication, any exception needs explicit, reviewable evidence of their compiled contract; commit-name equality, equal source trees, metallib hashes, or inferred metadata are insufficient substitutes. A new publication gate must not retroactively rewrite stored manifests, invalidate running historical attempts, or require legacy archives to implement a new interface.

Excluded: registry/schema redesign; changes to selection, scheduler independence, model materialization, PostgreSQL or active attempts; legacy artifact deletion/relabeling; new build-edge refactoring; production cutover. The fsync reporting case should be recorded as a separate bounded follow-up unless expressly included in the next assignment.

## 9. Validation strategy

For a future implementation:

1. Start with deterministic identity fixtures: correct contract succeeds; missing/wrong layout, missing/wrong width, wrong role/commit/self-hash, unavailable identity, wrong record version/header and duplicate required fields fail. Verify valid independent source commits remain supported.
2. Exercise dedicated rollover and refresh through their real preflight paths using isolated candidate fixtures and mocked identity output, with embedded-commit checks enabled. Include TRAIN-only and INFER-only semantic mismatches and mismatched candidate pairs. Ensure rejection precedes runtime/artifact staging and registry/current-link mutation. Do not merely mock the new verifier to fail.
3. Exercise infer-only CLI default, partly explicit and fully explicit current arguments; reject source-contract disagreement and current TRAIN/INFER width disagreement before registry replacement. Test dedicated historical INFER contract mismatch without changing legacy bootstrap compatibility.
4. Run the smallest affected suites first, then the six reviewed suites listed in section 6. Re-run Phase 24B architecture/guard fixtures and Release build configuration checks. Retain tests for previous-generation preservation, immutable conflicts, priority/schema upgrades, and committed-link-failure behavior. Use development-only temporary roots; never invoke publisher defaults against real artifacts.
5. Fresh real identity/output and build qualification require a later expressly authorized independent build task: isolated submodule inputs, development DerivedData/artifact roots and clean provenance. The Phase 24A MetaNN isolation obstruction remains an unqualified prerequisite; no build readiness was established by this audit. Check source paths and active-work safety before any worker CLI run. No DB/GPU/model-weight workload is needed to validate the proposed Python contract gates.

Report-level source tests and mocks cannot prove actual binary provenance, numerical parity, loader behavior or registry deployment correctness. Those remain separate qualification obligations.

## 10. Explicit GO / NO-GO recommendation

**GO:** authorize the narrow implementation scope in section 8, after deciding how retained dedicated INFER candidates without semantic fields will be qualified. The source and mocked-validator evidence justify that work; Phase 24B is closed.

**NO-GO:** consider present modern publication checks sufficient to certify compiled semantic compatibility; publish a newly labeled dedicated pair based only on role/commit/hash/resource agreement; perform production cutover; rebuild/relabel frozen historical TRAIN; or expand this increment into scheduler, numerical-runtime, registry or build-graph redesign.

This is a development/qualification recommendation, not an assertion that currently running historical workers are incompatible or authority to interrupt them. This task stops at the audit report; no Phase 24C implementation has begun.

## Appendix — Phase 24B closure and session results

Initial status was the modified architecture guard, the untracked regression file, and the pre-existing untracked Phase24 report directory. Review confirmed target-owned source/provenance IDs; exactly one TRAIN source phase; provenance before Sources without requiring either first; unshared phases; exactly one root/application membership; legacy/checkpoint-persistence/inference exclusions; Release's own Sources membership; and preserved managed-only/runtime/persistence/legacy-body checks (`Tests/DedicatedTrainingWorkerArchitectureTests.sh:15–117`). Regression fixtures cover ordering, ownership, phase types, missing/duplicate required source memberships and forbidden contamination (`Tests/DedicatedTrainingWorkerArchitectureGuardTests.py:63–141`).

Commands actually run for closure:

```text
bash Tests/DedicatedTrainingWorkerArchitectureTests.sh
PYTHONDONTWRITEBYTECODE=1 python3 Tests/DedicatedTrainingWorkerArchitectureGuardTests.py
git diff --check
bash -n Tests/DedicatedTrainingWorkerArchitectureTests.sh
git diff --cached --check
git add -- Tests/DedicatedTrainingWorkerArchitectureTests.sh Tests/DedicatedTrainingWorkerArchitectureGuardTests.py
git commit -m 'Phase 24B: Repair dedicated TRAIN architecture guard' -- Tests/DedicatedTrainingWorkerArchitectureTests.sh Tests/DedicatedTrainingWorkerArchitectureGuardTests.py
```

Results: guard PASS; 14 regression tests PASS; shell syntax and unstaged/staged diff checks PASS. The index was empty before staging and was checked to contain only the two requested files. Commit `7222021fd41ad71f6b7118914613b550f66405df` uses exactly the requested message and changes only those files (199 insertions, 8 deletions). No report was committed. No additional production/publication tests or build was run during closure.

The worktree's Git directory is shared metadata at `/Volumes/Developer SSD/ExpertAdvisor/.git/worktrees/ExpertAdvisor-Rollover`. Initial sandboxed staging could not create its index lock. The authorized stage/commit then succeeded with the required filesystem escalation. These operations updated rollover-branch Git metadata; no production checkout source or artifact was changed. There was no automatic-approval rejection.

Phase 24C operations were selective configuration reads, rollover MCP deterministic operations, direct `rg`/numbered source reads, source AST inventory, and 15 fully mocked identity-validator calls with `PYTHONDONTWRITEBYTECODE=1`. No publication or worker CLI was run. The only new Phase 24C file is this report; the Phase 24A report remains untracked and unchanged.

Final `git status --short`:

```text
?? docs/phases/Phase24/
```

Final `git diff --stat`: empty (ordinary Git diff excludes untracked reports). `git diff --check` passes; this new report is separately checked for trailing whitespace/conflict markers. Remaining unverified assumptions: actual production process/registry state, real candidate compiled identities and compatibility, independent build readiness, loader/Metal/numerical behavior, and the post-registry-replace fsync failure case. None was exercised or altered in this task.
