# EconomicEventFeatures Phase 4 / Migration 083 Review Findings

**Review date:** 2026-08-26
**Bundle reviewed:** `EconomicEventFeatures_Phase4_083_Review_20260826T185212Z.tar.gz`
**Scope:** migration 083, Phase 4 consensus/surprise feature implementation, repository integration, feature layout, model-input expansion, and included validation evidence.

## Executive verdict

**GO for commit after minor cleanup; NO-GO for production deployment as-is until one scientific-causality issue is resolved or surprise is explicitly disabled.**

Migration 083 itself is narrow and structurally safe: it only replaces the selected-consensus view to append selected-row actual semantics, without inserting, updating, or deleting consensus observations. The Phase 4 feature layout is correctly append-only: existing Tensor columns 0–58 remain in place, new economic-consensus fields occupy 59–66, the physical Tensor grows from 59 to 67, and model input grows from 63 to 71 while width 63 remains a registered predecessor under the existing explicit input-width expansion mechanism.

The main blocker is the **provenance of the `actual` used to compute surprise**. The feature engine correctly prevents `actual-consensus` surprise from appearing before the authoritative event timestamp. However, the reviewed artifacts do not establish that the persisted OANDA `actual` is the original value known at release time rather than a later revised value. Immutability of the database row is not sufficient to prove historical causal availability. If an OANDA historical observation contains a later revision, the current implementation would inject future information into all post-release bars for that historical event. This conflicts directly with the Phase 4 requirement that future revisions never leak backward.

A safe deployment can proceed in either of two ways:

1. **Consensus-only deployment:** retain the 8-column layout if desired but force `released_event_has_surprise=0` and all surprise values to zero until release-time actual provenance is established; or
2. **Provenance-gated surprise:** add an explicit contract proving that the actual is the original release-time observation (or source authoritative first-release actuals independently), and compute surprise only for rows satisfying that contract.

## Severity-ranked findings

### F1 — HIGH / deployment blocker: release-time provenance of `actual` is not established

**Location:**
- `Database/migrations/083_economic_event_selected_consensus_release_semantics.sql`
- `Sources/EconomicEventRepository.cpp`
- `Sources/EconomicEventFeatures.cpp`

Migration 083 exposes these fields from the selected provider observation:

- `actual_parse_status`
- `actual_canonical_value_low/high`
- `actual_value_kind`
- `actual_unit`
- `actual_scale`
- `actual_qualifier`

The repository maps those fields into `EconomicEventSelectedConsensus::actual`, and `SetSurprise()` computes:

`actual.canonicalValueLow - forecast.canonicalValueLow`

once the event is strictly before the completed-bar cutoff.

That establishes **time gating of use**, but not **time provenance of the value**. The reviewed evidence shows that 1,379 of 1,405 selected OANDA observations have parsed actual values, while all 116 Myfxbook observations have no actual. What is missing is proof that those OANDA actuals represent the first-release value as it existed at the event timestamp.

The Phase 4 prompt explicitly required that revised future actual values never leak backward. No field such as `actual_known_at`, `actual_revision_status`, `initial_release_actual`, or equivalent provenance contract exists in migration 083 or in the feature gate. The semantic contract `oanda_economic_consensus_candidate_v1` proves source/shape semantics, but the reviewed bundle does not show that it proves first-release revision status.

**Impact:** potentially material look-ahead bias in 1,379 historical OANDA surprise observations.

**Required disposition before production:** either disable surprise features or establish/restrict to release-time actual provenance.

### F2 — MEDIUM / deployment hardening: 083 lacks a dedicated production-state migration-clone validation

The included `EconomicEventFeatureRangeRepositoryTests.sh` does exercise migrations 072 → 081 → 082 → 083 in a disposable synthetic database, which is useful and passed. However, there is no dedicated migration test in the review bundle that clones the actual 082 production consensus state and asserts that 083 preserves:

- 1,532 physical consensus rows;
- 1,521 selected rows;
- 1,405 selected OANDA rows;
- 116 selected Myfxbook rows;
- unchanged provider-selection precedence;
- exact existing view columns plus the seven appended actual-semantic columns;
- `pqxx` SELECT privilege;
- zero stored-row mutation.

Because 083 is a view-only migration, risk is low, but a production-clone disposable test is appropriate before deployment and would align with how 082 was rolled out.

### F3 — MEDIUM / scientific comparability: surprise availability is provider-correlated

Production evidence in the transcript shows:

- OANDA selected: 1,405; parsed actual: 1,379
- Myfxbook selected: 116; parsed actual: 0

Thus `released_event_has_surprise` is structurally much more likely to be 1 for OANDA-backed events than Myfxbook-backed events. The feature code does not feed provider identity directly, which is good, but the missingness pattern can indirectly encode provider/source-era coverage.

This is not a correctness defect if intentional, and the validity bit is necessary to distinguish missing surprise from true zero. It should nevertheless be treated as a known modeling confound and measured in ablation/diagnostics.

### F4 — LOW / scientific information loss: one global “relevant event” is selected at simultaneous timestamps

`MostRelevantAtTimestamp()` picks one event using:

1. higher `eventImportance`;
2. lower positive `economicEventId` as the deterministic tie-break.

This is deterministic and causal, but if multiple economically distinct releases share the same timestamp, only one consensus tuple is emitted into columns 59–62. Existing event-family indicator features can still show multiple simultaneous families, but consensus details for the non-selected simultaneous events are discarded.

This is acceptable for the stated minimal Phase 4 scope, but should be documented as a modeling choice rather than interpreted as full event coverage.

### F5 — LOW / documentation: Database README does not appear to document migration 083

The bundle includes `Database/README.md`, but Phase 4's diff stat does not show it as modified. Migration 083 should be added to the migration documentation before commit/deployment so the selected-view release semantics are discoverable.

### F6 — EXPECTED deployment prerequisite: Release build is still unverified on a clean worktree

The included report states that the exact Release build reached provenance enforcement and exited 65 because the worktree was dirty. Debug compiled and linked, focused tests passed, and provenance enforcement was not bypassed. This is the correct behavior.

A clean-worktree Release build remains mandatory after commit and before scheduler cutover.

## Migration 083 review

### What is correct

Migration 083 uses `CREATE OR REPLACE VIEW economic_event_selected_consensus` and preserves all existing selected-consensus columns in their existing order, then appends seven actual-semantic columns. It keeps the same selection predicate:

`WHERE c.forecast_parse_status = 'parsed'`

Therefore it does not alter provider precedence or selected-row cardinality by design.

It does not mutate `economic_event_consensus` and does not alter authoritative `economic_event` rows. It reasserts the view comment and grants SELECT to `pqxx`.

### Deployment compatibility

The new repository code expects the 083 columns unconditionally, so **083 must be applied before any Phase 4 binary is allowed to query production**. Deploying the binary first against schema 082 would fail at query time.

Recommended ordering:

1. backup production at 082;
2. apply 083;
3. register 083 in `schema_migrations` through the normal migration procedure;
4. verify row counts/view contract read-only;
5. commit/clean worktree;
6. clean Release build;
7. controlled scheduler cutover;
8. queue only new-width experiments intentionally using width 71.

## Causality review

### Consensus timing

The implementation is conservative and defensible given the absence of historical consensus `known_at` timestamps:

- future event: no consensus;
- event exactly at completed-bar cutoff: consensus available, actual/surprise withheld;
- event strictly before cutoff: consensus and compatible surprise may be available.

This avoids projecting the final persisted historical consensus into arbitrarily early pre-release bars.

The feature-range repository intentionally changes its in-range upper bound from `< endUtc` to `<= endUtc` so that an event exactly at the final bar cutoff can be loaded for consensus-only boundary exposure. The engine still uses strict `< informationCutoff` to decide whether the event is released, which preserves actual/surprise causality at the boundary.

### Surprise timing

The **engine timing rule is correct**. The unresolved issue is whether the **actual value itself** is historically first-release causal. Those are separate questions. Phase 4 solves the former, but the bundle does not prove the latter.

## Feature-layout and model-input review

### Layout

The appended economic-event indices are clean and ordered:

- 10 `relevantEventHasConsensus`
- 11 `relevantEventConsensusLow`
- 12 `relevantEventConsensusHigh`
- 13 `relevantEventConsensusIsRange`
- 14 `releasedEventHasSurprise`
- 15 `releasedEventSurprise`
- 16 `releasedEventSurpriseAbs`
- 17 `releasedEventSurpriseDirection`

These map to Tensor columns 59–66 because the pre-consensus Tensor width is 59.

### Width ancestry

The reviewed contracts support:

- previous physical Tensor width: 59
- current physical Tensor width: 67
- previous model input width: 63
- current model input width: 71

The four return features remain a stable semantic suffix. Expansion logic copies the old Tensor prefix, inserts zero-initialized new Tensor rows before the return suffix, relocates the four return-feature rows, and then copies recurrent hidden-state rows. This is the correct shape-preserving strategy.

The tests include width 63 and 71 in the registered-width set and exercise append-only expansion/provenance. No silent reinterpretation of the old 63-wide model input was found.

## Repository and performance review

Both event loaders use one `LEFT JOIN economic_event_selected_consensus` and map optional consensus/actual fields in one result set. There is no per-bar or N+1 SQL introduced.

Feature computation remains an in-memory chronological scan. The additional cost is a wider range query and eight float outputs per bar. No scheduler, training, or inference lifecycle behavior is changed by the feature implementation itself.

## Normalization review

The static divisors are causal and dataset-independent:

- percent-like families: ÷ 10
- employment counts: ÷ 1,000,000
- JOLTS counts: ÷ 10,000,000

The code validates expected units and rejects unsupported family/unit combinations. Surprise requires scalar forecast/actual values with identical unit, scale, and qualifier. FOMC range forecasts preserve low/high endpoints and do not invent a midpoint or range surprise.

This is a good first-phase normalization contract. It should be evaluated empirically rather than optimized before first controlled experiments.

## Test review

The evidence reports successful execution of:

- `EconomicEventFeaturesTests.sh`
- `EconomicEventFeatureRangeRepositoryTests.sh`
- `EconomicEventTensorIntegrationTests.sh`
- `LSTMFeatureVectorParityTests.sh`
- `LSTMInputWidthExpansionTests.sh`
- `LSTMModelInputCompatibilityTests.sh`
- 14 width-sensitive causal suites
- strict compilation of real-input integration source
- Debug Xcode build using `DerivedData/Development`
- `git diff --check`

The focused tests cover provider neutrality, missing consensus, emergency FOMC behavior, exact-boundary timing, scalar/range handling, surprise sign/zero, incompatible semantics, normalization, append-only layout, parity, and width ancestry.

The test suite is strong. The missing test is not ordinary code coverage; it is a **provenance assertion** that historical actuals are first-release causal values.

## Recommended correction path

### Preferred minimal correction

Keep the current 71-wide layout and 083 view if desired, but gate surprise on a new explicit provenance predicate. Until such provenance exists, set:

- `released_event_has_surprise = 0`
- `released_event_surprise = 0`
- `released_event_surprise_abs = 0`
- `released_event_surprise_direction = 0`

This allows deployment and experimentation with the four consensus fields without introducing questionable historical actual information, while avoiding another immediate input-width change later.

### Stronger long-term correction

Persist or source authoritative first-release actual observations with explicit provenance, for example:

- original release value;
- release timestamp/known-at timestamp;
- revision status or revision sequence;
- source artifact and immutable hash;
- semantic unit/scale/qualifier.

Then enable surprise only when the repository can prove that the actual was available at the event timestamp and semantically matches the selected consensus.

## Go/no-go checklist

### Safe now

- [x] Phase 4 core implementation can be committed after documentation cleanup.
- [x] Width/layout changes are append-only.
- [x] Old width 63 remains explicitly supported.
- [x] Repository access is range-based.
- [x] Consensus exact-boundary timing is conservative.
- [x] FOMC range handling is safe.
- [x] Missing consensus/surprise have explicit validity bits.

### Required before production

- [ ] Resolve F1: prove first-release actual provenance or disable surprise.
- [ ] Add/run a disposable production-082-clone test for migration 083.
- [ ] Update `Database/README.md` for migration 083.
- [ ] Apply/register 083 before deploying the Phase 4 binary.
- [ ] Obtain a clean-worktree Release build.
- [ ] Perform controlled scheduler cutover without disturbing existing old-width workers.
- [ ] Start with intentionally new width-71 experiments; do not silently convert existing models.

## Final recommendation

**Do not deploy the current surprise feature path unchanged.** The consensus portion of Phase 4 is well designed and production-worthy after migration 083 validation. The surprise portion needs one additional provenance guarantee because timing the use of an actual after release is not enough if the stored actual itself may be a later revision.

Once surprise is either provenance-gated or disabled, I would approve migration 083 + Phase 4 for controlled production deployment, followed by a clean Release build and deliberate scheduler cutover.
