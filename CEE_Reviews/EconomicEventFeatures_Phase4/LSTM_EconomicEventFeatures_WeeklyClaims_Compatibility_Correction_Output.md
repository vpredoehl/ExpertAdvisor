---
title: "LSTM EconomicEventFeatures DOL ETA Weekly Claims Compatibility Correction"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_EconomicEventFeatures_WeeklyClaims_Compatibility_Correction_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM EconomicEventFeatures DOL ETA Weekly Claims Compatibility Correction

Implemented and validated the targeted compatibility correction. `DOL_ETA/WEEKLY_CLAIMS` now maps explicitly to the existing employment model family, and the complete production calendar loads successfully.

## Root cause and history

The feature mapper was created against the original 1,741-row authoritative universe, which excluded DOL/ETA. It registered ten canonical source/family pairs and deliberately failed closed on everything else.

History showed:

- `299abd1` added first-party DOL/ETA Weekly Claims ingestion on August 24, 2026.
- `c77e41b` exposed its shared import CLI.
- `203aeba` completed the validated 2010-present corpus and the narrow 2011–2012 timezone-label reconstruction contract.
- `20dd78f` incorporated DOL/ETA into the authoritative coverage audit, but the LSTM mapper still retained its earlier ten-family universe.
- The original DOL ingestion phase prohibited adding model mappings only because that phase was calendar-ingestion scoped. It did not establish permanent feature exclusion.

The provider-neutral range loader therefore correctly returned all production events, after which the fail-closed mapper rejected the newly authoritative eleventh pair.

## Production investigation

Production was queried using both:

```bash
PGOPTIONS='-c default_transaction_read_only=on'
BEGIN READ ONLY;
```

The transaction reported `transaction_read_only=on`. No production writes or migrations occurred.

| Finding | Result |
|---|---:|
| Total rows | 860 |
| Earliest source release date | 2010-01-07 |
| Latest source release date | 2026-08-20 |
| Earliest event timestamp | 2010-01-07 13:30:00Z |
| Latest event timestamp | 2026-08-20 12:30:00Z |
| Exact timestamps | 815 |
| Reconstructed timestamps | 45 |
| Source timezone | America/New_York: 860 |
| Source release time | 08:30:00: 860 |
| UTC release times | 12:30Z: 562; 13:30Z: 298 |
| Importance 3 | 860 |
| Source event ID populated | 860 |
| Reference period populated | 860 |
| Duplicate canonical timestamps | 0 |
| Duplicate source IDs | 0 |
| Selected consensus | 0 |

The 45 reconstructed rows run from 2011-11-10 through 2012-09-13. They are not reconstructed from a modern schedule: each retained artifact publishes the 08:30 local embargo clock, but uses a demonstrably incorrect EST/EDT abbreviation. The importer preserves that clock, resolves it using historical America/New_York rules, and downgrades confidence to `reconstructed`. UTC/local date and time reconciliation found zero mismatches.

Representative rows were inspected:

- Beginning: `dol_eta:usdl-10-02-nat`, 2010-01-07 13:30Z.
- Middle: `dol_eta:usdl-18-494-nat`, 2018-03-29 12:30Z.
- Recent: `dol_eta:artifact-2026-082026-pdf`, 2026-08-20 12:30Z.

There were no duplicate patterns. Release intervals were predominantly seven days, with audited holiday Wednesday releases and two authoritative archive gaps: 14 days in 2019 and 56 days in 2025.

## Decision and correction

The final mapping is:

```text
DOL_ETA / WEEKLY_CLAIMS
    -> EconomicEventModelFamily::employment
```

This is preferable because Weekly Claims is an authoritative, importance-3 unemployment-insurance labor-market release with a causal release instant.

Alternatives were rejected:

- Explicit ignore would discard a validated employment release.
- Repository filtering would violate the provider/event-neutral loader contract.
- A new model family or Tensor column is not justified for one labor-market stream.
- No catch-all behavior was added; unrelated unknown families still fail closed.

The code correction is confined to [EconomicEventFeatures.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/EconomicEventFeatures.cpp>). Repository SQL did not change.

## Feature and compatibility contract

| Contract | Before | After |
|---|---:|---:|
| Tensor width | 67 | 67 |
| Model input width | 71 | 71 |
| Semantic layout | v4 | v4 |
| Production schema | 082 | 082 |
| Migration required | None | None |

Weekly Claims affects only:

- `employment_event`
- `employment_recency_decay`

It does not set inflation, growth, fed-policy, or consumer-demand bits.

Production has no width-63 or width-71 models. Consequently, the incomplete mapper was never consumed by an already-trained production economic-feature model, and v4 can be completed before deployment without advancing the layout.

Compatibility details:

- Width-53 models project only Tensor columns 0–48 and are unaffected.
- Existing running workers are older-width workers and remain on their mapped binary.
- Production contains zero width-63 and zero width-71 models.
- A hypothetical non-production width-63/71 model trained under the incomplete mapper would receive changed employment values and must not be assumed semantically interchangeable with corrected inputs.

No consensus was manufactured. Weekly Claims continues to produce zero in columns 59–62 because production has no selected consensus. Reserved surprise columns 63–66 remain zero.

## Files changed

- [EconomicEventFeatures.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/EconomicEventFeatures.cpp>)
- [EconomicEventFeaturesTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/EconomicEventFeaturesTests.cpp>)
- [EconomicEventFeatureRangeRepositoryTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/EconomicEventFeatureRangeRepositoryTests.cpp>)
- [EconomicEventFeaturesRealInputIntegrationTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/EconomicEventFeaturesRealInputIntegrationTests.cpp>)
- [EconomicEventTensorIntegrationTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/EconomicEventTensorIntegrationTests.cpp>)
- [LSTM_EconomicEventConsensusFeatures.md](</Volumes/Developer SSD/ExpertAdvisor/EconomicCalendar/LSTM_EconomicEventConsensusFeatures.md>)

Tests cover explicit mapping, no unsupported exception, inside-bar activation, causal decay, isolated family bits, gaps, exact boundaries, missing consensus, reserved surprise, unchanged BLS/other mappings, widths, parity, schema 082, and production real input.

## Validation

All required tests passed:

```text
Tests/EconomicEventFeaturesTests.sh
ECONOMIC_EVENT_FEATURES_TEST_PASS,canonical_mappings=11,feature_width=18

Tests/EconomicEventFeatureRangeRepositoryTests.sh
DISPOSABLE_SCHEMA_END=082
MIGRATION_083_APPLIED=false
SELECTED_CONSENSUS_VIEW_082_ONLY=true
DISPOSABLE_DATABASE_DROPPED=ea_economic_event_feature_range_93538

Tests/EconomicEventTensorIntegrationTests.sh
PASS

PGOPTIONS='-c default_transaction_read_only=on' \
Tests/EconomicEventFeaturesRealInputIntegrationTests.sh
ECONOMIC_EVENT_FEATURES_REAL_INPUT_INTEGRATION_PASS,bars=486,events=2601,weekly_claims=860,reconstructed_weekly_claims=45,consumed=6

Tests/LSTMFeatureVectorParityTests.sh
PASS

bash Tests/LSTMInputWidthExpansionTests.sh
PASS

Tests/LSTMModelInputCompatibilityTests.sh
PASS
```

Additional validation:

```text
Tests/DolEtaWeeklyClaimsAdapterTests.sh
PASS

Tests/EconomicEventBarAlignmentTests.sh
ECONOMIC_EVENT_BAR_ALIGNMENT_TEST_PASS

git diff --check
PASS
```

Incremental Debug link build:

```bash
xcodebuild \
  -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Release" \
  -configuration Debug \
  -derivedDataPath DerivedData/Development \
  ENABLE_USER_SCRIPT_SANDBOXING=NO \
  build
```

Result: `** BUILD SUCCEEDED **`.

Affected standalone compilation used `-Wall -Wextra -Werror` without warnings. Xcode retained one pre-existing unrelated unreachable-code warning in `CampaignOperationsProductionAdmissionService.cpp`.

## Safety and readiness

- Production accessed: yes.
- All production access read-only: yes.
- Production mutated: no.
- Scheduler/workers altered or signaled: no.
- Experiments queued or started: no.
- Disposable databases created: one, `ea_economic_event_feature_range_93538`.
- Disposable database destroyed and absence verified: yes.
- Newly discovered unsupported authoritative families: none.
- DOL/ETA deployment blocker: resolved.
- Database migration required: no.
- Next step: review/commit, then clean Release build using `DerivedData/Development`.
- Controlled scheduler cutover: unblocked by this defect after that clean Release build and normal active-worker safety checks.
- Release build and cutover were not performed.

`git status --short`:

```text
 M EconomicCalendar/LSTM_EconomicEventConsensusFeatures.md
 M Sources/EconomicEventFeatures.cpp
 M Tests/EconomicEventFeatureRangeRepositoryTests.cpp
 M Tests/EconomicEventFeaturesRealInputIntegrationTests.cpp
 M Tests/EconomicEventFeaturesTests.cpp
 M Tests/EconomicEventTensorIntegrationTests.cpp
```

`git diff --stat`:

```text
 .../LSTM_EconomicEventConsensusFeatures.md         | 36 +++++++++++
 Sources/EconomicEventFeatures.cpp                  |  6 ++
 Tests/EconomicEventFeatureRangeRepositoryTests.cpp | 25 ++++----
 ...nomicEventFeaturesRealInputIntegrationTests.cpp | 25 +++++++-
 Tests/EconomicEventFeaturesTests.cpp               | 69 +++++++++++++++++++++-
 Tests/EconomicEventTensorIntegrationTests.cpp      | 12 ++--
 6 files changed, 155 insertions(+), 18 deletions(-)
```