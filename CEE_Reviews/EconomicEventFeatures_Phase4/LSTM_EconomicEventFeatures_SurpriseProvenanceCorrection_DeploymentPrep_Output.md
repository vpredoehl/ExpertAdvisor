---
title: "LSTM EconomicEventFeatures Surprise Provenance Correction and Deployment Prep"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_EconomicEventFeatures_SurpriseProvenanceCorrection_DeploymentPrep_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM EconomicEventFeatures Surprise Provenance Correction and Deployment Prep

Implemented the production-safe Phase‑4 correction. Surprise derivation is disabled, columns 63–66 remain reserved zero channels, and runtime now works against migration 082 without 083.

## Correction and behavior

- Root cause: the persisted provider `actual` values have no explicit first-release/revision provenance. Immutability, provider identity, timestamps, import dates, and value presence do not prove historical causality.
- Current persisted actuals do **not** have proven first-release provenance.
- Removed the surprise calculation path entirely. Even populated OANDA or Myfxbook fixture actuals cannot activate surprise.
- Repository queries no longer select any provider actual fields.
- Deleted un-applied migration 083 because its sole purpose was exposing those unsafe actual fields.
- Production schema requirement is now exactly **082**. No 083 or 084 is required.
- No migration, ingestion, scheduler restart, or production mutation was performed.

Current contract:

- Tensor 59–62: active provider-neutral consensus features.
- Tensor 63–66: reserved and always zero.
- Tensor width: **67**.
- Model input width: **71**.
- Semantic layout: **v4**.
- Width 63 remains the append-only v3 predecessor.
- Expansion from 63 to 71 still requires `--resume-expand-input-width`.

Consensus behavior remains unchanged: static family normalization, exact release-boundary exposure, no pre-release projection, no future leakage, zero-consensus presence bit, and FOMC low/high range preservation without a midpoint.

Both repository loaders remain one-query `LEFT JOIN` loaders with no per-bar or N+1 SQL.

## Files changed

- Runtime: [EconomicEventFeatures.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/EconomicEventFeatures.cpp>), [EconomicEventFeatures.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/EconomicEventFeatures.hpp>), [EconomicEventRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/EconomicEventRepository.cpp>), [EconomicEventRepository.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/EconomicEventRepository.hpp>)
- Database: deleted [083_economic_event_selected_consensus_release_semantics.sql](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/083_economic_event_selected_consensus_release_semantics.sql>); updated [Database/README.md](</Volumes/Developer SSD/ExpertAdvisor/Database/README.md>)
- Documentation: [LSTM_EconomicEventConsensusFeatures.md](</Volumes/Developer SSD/ExpertAdvisor/EconomicCalendar/LSTM_EconomicEventConsensusFeatures.md>)
- Tests: [EconomicEventFeaturesTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/EconomicEventFeaturesTests.cpp>), [EconomicEventFeatureRangeRepositoryTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/EconomicEventFeatureRangeRepositoryTests.cpp>), [EconomicEventFeatureRangeRepositoryTests.sh](</Volumes/Developer SSD/ExpertAdvisor/Tests/EconomicEventFeatureRangeRepositoryTests.sh>), [EconomicEventTensorIntegrationTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/EconomicEventTensorIntegrationTests.cpp>), [EconomicEventFeaturesRealInputIntegrationTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/EconomicEventFeaturesRealInputIntegrationTests.cpp>)

## Validation

Passed:

```text
Tests/EconomicEventFeaturesTests.sh
ECONOMIC_EVENT_FEATURES_TEST_PASS,canonical_mappings=10,feature_width=18

Tests/EconomicEventFeatureRangeRepositoryTests.sh
DISPOSABLE_SCHEMA_END=082
MIGRATION_083_APPLIED=false
SELECTED_CONSENSUS_VIEW_082_ONLY=true
DISPOSABLE_DATABASE_DROPPED=ea_economic_event_feature_range_89805

Tests/EconomicEventTensorIntegrationTests.sh
exit 0

Tests/LSTMFeatureVectorParityTests.sh
exit 0

bash Tests/LSTMInputWidthExpansionTests.sh
exit 0

Tests/LSTMModelInputCompatibilityTests.sh
exit 0

git diff --check
exit 0
```

The width-expansion script is not executable, so direct invocation returned permission denied; explicit `bash` execution passed.

Strict affected compilation used `-Wall -Wextra -Werror`.

Debug link validation:

```bash
xcodebuild -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Release" \
  -configuration Debug \
  -derivedDataPath DerivedData/Development \
  ENABLE_USER_SCRIPT_SANDBOXING=NO \
  build
```

Result: `** BUILD SUCCEEDED **`. The override was needed because Xcode’s script sandbox blocked the provenance generator’s atomic temporary file. Existing unrelated project warnings remain; no affected strict-test warning occurred.

The real-input integration compiled strictly but aborted during production execution on:

```text
unsupported_authoritative_economic_event_family:DOL_ETA:WEEKLY_CLAIMS
```

Production currently has 860 such rows. This is independent of the consensus/surprise correction, but it is a deployment blocker for a full production runtime cutover because the existing feature mapper supports only the established ten families.

## Database verification

One disposable database was created: `ea_economic_event_feature_range_89805`. It was destroyed, and its absence was verified afterward.

Production was accessed read-only only, using `PGOPTIONS='-c default_transaction_read_only=on'`, `BEGIN READ ONLY`, and the integration test’s `pqxx::read_transaction`.

Verified:

- Migration ledger ends at 082.
- Physical consensus rows: 1,532.
- Selected OANDA: 1,405.
- Selected Myfxbook: 116.
- Migration-082 view has only its original 20-column projection.
- No explicit actual-known-at/revision/first-release provenance column exists.
- Representative OANDA and Myfxbook scalar consensus loaded.
- Representative FOMC range retained low `3.50`, high `3.75`.
- Emergency FOMC releases on 2020-03-03 and 2020-03-15 remain missing consensus.
- A genuine Myfxbook zero consensus remained present through its validity bit.

## Release build and cutover

After review/commit, the exact clean-worktree Release build is:

```bash
cd "/Volumes/Developer SSD/ExpertAdvisor"
test -z "$(git status --porcelain)"

xcodebuild \
  -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Release" \
  -configuration Release \
  -derivedDataPath DerivedData/Development \
  ENABLE_USER_SCRIPT_SANDBOXING=NO \
  build
```

Post-build verification:

```bash
BIN="/Volumes/Developer SSD/ExpertAdvisor/DerivedData/Development/Build/Products/Release/LSTM_Release"

test -x "$BIN"
codesign --verify --strict "$BIN"
shasum -a 256 "$BIN"
git status --short
```

Do not cut over until the `DOL_ETA:WEEKLY_CLAIMS` compatibility issue is resolved. After that, the planned procedure is:

1. Recheck processes and scheduler status.
2. Send `SIGTERM` only to the scheduler PID. Per the scheduler contract, graceful shutdown preserves active worker attempts.
3. Confirm the scheduler exited and width-63 workers remain alive.
4. Start the new scheduler from the Development Release product with the existing capacity/options:

```bash
screen -dmS expertadvisor_scheduler bash -lc '
cd "/Volumes/Developer SSD/ExpertAdvisor"
exec "/Volumes/Developer SSD/ExpertAdvisor/DerivedData/Development/Build/Products/Release/LSTM_Release" \
  --schedule-experiments \
  --max-train-procs=2 \
  --max-infer-procs=0 \
  --max-analyze-procs=1 \
  --scheduler-poll-seconds=30 \
  --scheduler-log-dir=experiment_logs \
  --auto-evaluate-continuations \
  --auto-queue-continuations \
  --continuation-scan-seconds=300 \
  --continuation-max-queues-per-scan=1
'
```

Post-cutover:

```bash
PGOPTIONS='-c default_transaction_read_only=on' \
  "/Volumes/Developer SSD/ExpertAdvisor/DerivedData/Development/Build/Products/Release/LSTM_Release" \
  --scheduler-status

pgrep -alf LSTM_Release
```

Currently running width-63 workers must remain on their existing mapped binary and model width until normal completion. The replacement scheduler should observe them and count their capacity without mutation.

Fresh experiments launched after cutover default to width 71. Width-63 models:

- ordinary resume/inference stays width 63;
- expansion to width 71 requires `--resume-expand-input-width`;
- expansion zero-initializes the eight new input rows and preserves the old Tensor prefix, return rows, and recurrent state.

Example:

```bash
"$BIN" --queue-experiment \
  --resume-model-id=MODEL_ID \
  --resume-expand-input-width \
  --target-epochs=TARGET_EPOCH
```

Read-only schema verification after cutover:

```bash
PGOPTIONS='-c default_transaction_read_only=on' \
psql -X --host=localhost --username=pqxx --dbname=LSTM \
  -v ON_ERROR_STOP=1 -c "
BEGIN READ ONLY;
SELECT version, filename
FROM schema_migrations
ORDER BY version::integer DESC
LIMIT 1;

SELECT consensus_source, count(*)
FROM economic_event_selected_consensus
GROUP BY consensus_source
ORDER BY consensus_source;
COMMIT;"
```

## Remaining blockers

- Resolve or explicitly filter/map production `DOL_ETA:WEEKLY_CLAIMS` rows before scheduler cutover.
- Commit/review these changes and obtain the clean Release build.
- The original surprise-provenance blocker itself is resolved.

No commit or push was performed.

`git status --short`:

```text
 M Database/README.md
 D Database/migrations/083_economic_event_selected_consensus_release_semantics.sql
 M EconomicCalendar/LSTM_EconomicEventConsensusFeatures.md
 M Sources/EconomicEventFeatures.cpp
 M Sources/EconomicEventFeatures.hpp
 M Sources/EconomicEventRepository.cpp
 M Sources/EconomicEventRepository.hpp
 M Tests/EconomicEventFeatureRangeRepositoryTests.cpp
 M Tests/EconomicEventFeatureRangeRepositoryTests.sh
 M Tests/EconomicEventFeaturesRealInputIntegrationTests.cpp
 M Tests/EconomicEventFeaturesTests.cpp
 M Tests/EconomicEventTensorIntegrationTests.cpp
```

`git diff --stat`:

```text
12 files changed, 157 insertions(+), 200 deletions(-)
```