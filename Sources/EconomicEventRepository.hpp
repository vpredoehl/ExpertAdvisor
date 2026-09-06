#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <pqxx/pqxx>

namespace EA::EconomicCalendar
{

inline constexpr int kEconomicCalendarSnapshotHashContractVersion = 1;

struct EconomicCalendarSnapshotIdentity
{
    long long snapshotId = 0;
    std::string contentHash;
};

struct EconomicCalendarSnapshotReport
{
    std::optional<long long> snapshotId;
    std::string contentHash;
    long long canonicalEventCount = 0;
    long long selectedConsensusCount = 0;
    long long releaseActualCount = 0;
    long long provenFirstReleaseActualCount = 0;
    long long provenanceUnavailableCount = 0;
    long long ambiguousFirstReleaseCount = 0;
    std::string sourceFamilyCountsJson;
    bool reused = false;
    bool dryRun = false;
};

bool SameEconomicCalendarSnapshotIdentity(
    const std::optional<EconomicCalendarSnapshotIdentity>& lhs,
    const std::optional<EconomicCalendarSnapshotIdentity>& rhs);

struct EconomicEventConsensusValue
{
    std::string valueKind;
    double canonicalValueLow = 0.0;
    std::optional<double> canonicalValueHigh;
    std::string unit;
    double scale = 0.0;
    std::optional<std::string> qualifier;
};

struct EconomicEventSelectedConsensus
{
    EconomicEventConsensusValue forecast;

    // A provider artifact may contain an actual value, but the current
    // persistence contract does not prove that it is the original value known
    // at release time rather than a later revision. Repository loaders ending
    // at schema 082 therefore never populate this field, and feature
    // computation must ignore it. It exists only so tests can prove that mere
    // actual-value presence cannot activate the reserved surprise channels.
    std::optional<EconomicEventConsensusValue> unprovenProviderActual;

    // Retained for diagnostics and repository tests only. Feature computation
    // deliberately ignores provider identity and all provider provenance.
    std::string provider;
};

// One provenance-certified authoritative initial-release actual.  Secondary
// provider actual fields are deliberately not accepted here.  Later revisions
// remain in economic_event_release_actual but are excluded from the feature
// view so they cannot rewrite historical release surprise.
struct EconomicEventReleaseActual
{
    EconomicEventConsensusValue actual;
    std::int64_t availableAtUnixMicros = 0;
    std::string sourceAgency;
    std::string sourceObservationId;
    std::string sourceArtifactPath;
    std::string sourceArtifactSha256;
    std::string semanticContract;
};

enum class EconomicEventFirstReleaseActualState
{
    provenanceUnavailable,
    ambiguous,
    notYetAvailable,
    provenFirstRelease,
};

// Phase-1 point-in-time result. The value and proven availability instant are
// populated exclusively from economic_event_first_release_actual_at(cutoff).
// The cutoff-independent assessment view supplies only state/reason metadata.
struct EconomicEventFirstReleaseActual
{
    EconomicEventConsensusValue actual;
    std::int64_t provenAvailableAtUnixMicros = 0;
    std::string sourceName;
    std::optional<std::string> sourceNativeEventId;
    std::string sourceObservationId;
    std::string evidenceKey;
};

struct EconomicEvent
{
    long long economicEventId = 0;

    std::string currency;
    std::string eventFamily;

    // Canonical instant, independent of PostgreSQL session timezone.
    std::int64_t eventTimestampUnixMicros = 0;

    // Canonical UTC representation intended for diagnostics/tests.
    std::string eventTimestampUtc;

    std::string sourceAgency;
    std::optional<std::string> sourceEventId;
    std::string sourceUrl;

    std::optional<std::string> referencePeriod;

    int eventImportance = 0;
    std::string historicalTimeConfidence;

    std::optional<std::string> sourceReleaseDate;
    std::optional<std::string> sourceReleaseTime;
    std::optional<std::string> sourceTimezone;

    // Populated only by the provider-neutral selected-consensus abstraction.
    // Events without a selected forecast remain std::nullopt.
    std::optional<EconomicEventSelectedConsensus> selectedConsensus;

    // Populated only from economic_event_feature_release_actual.  This is the
    // authoritative initial observation with an explicit causal known-at
    // instant; persisted revisions are never selected into model features.
    std::optional<EconomicEventReleaseActual> releaseActual;

    // Point-in-time state for the Phase-2 causal surprise channels. Missing,
    // not-yet-available, and ambiguous provenance never carry an actual.
    EconomicEventFirstReleaseActualState firstReleaseActualState =
        EconomicEventFirstReleaseActualState::provenanceUnavailable;
    std::string firstReleaseActualSelectionReason;
    std::optional<EconomicEventFirstReleaseActual> firstReleaseActual;
};

bool EconomicEventSchemaExists(
    pqxx::transaction_base& transaction);

std::vector<EconomicEvent> LoadEconomicEvents(
    pqxx::transaction_base& transaction,
    const std::string& currency,
    const std::string& startUtc,
    const std::string& endUtc);

// Load every event in [startUtc, endUtc], plus the latest event before
// startUtc for each authoritative (agency, canonical-family) stream. The
// chronological feature engine then deterministically reduces those prior
// rows to the latest state of each model-facing family. This exact seed query
// avoids an arbitrary recency lookback and remains one ordered database query.
// The inclusive end boundary supplies consensus to the final completed bar
// when a release occurs exactly at that bar's information cutoff.
// Legacy migration-088 authoritative initials are projected only when
// available_at is strictly before endUtc, preserving width-75 behavior.
// Phase-2 actuals are bulk-loaded exclusively through
// economic_event_first_release_actual_at(endUtc); per-bar evaluation repeats
// its inclusive proven_available_at boundary in memory.
std::vector<EconomicEvent> LoadEconomicEventsForFeatureRange(
    pqxx::transaction_base& transaction,
    const std::string& currency,
    const std::string& startUtc,
    const std::string& endUtc,
    const std::optional<EconomicCalendarSnapshotIdentity>& snapshot =
        std::nullopt);

// Capture and finalize the current model-facing corpus inside the caller's
// transaction. Equal canonical content reuses the same finalized snapshot.
// Dry-run computes the complete report without writing.
EconomicCalendarSnapshotReport CreateOrReuseEconomicCalendarSnapshot(
    pqxx::work& transaction,
    const std::string& createdBy,
    const std::optional<std::string>& creationNote,
    bool dryRun);

EconomicCalendarSnapshotReport InspectEconomicCalendarSnapshot(
    pqxx::transaction_base& transaction,
    const EconomicCalendarSnapshotIdentity& identity);

std::optional<EconomicCalendarSnapshotIdentity>
LoadExperimentEconomicCalendarSnapshot(
    pqxx::transaction_base& transaction,
    long long experimentId);

std::optional<EconomicCalendarSnapshotIdentity>
LoadModelEconomicCalendarSnapshot(
    pqxx::transaction_base& transaction,
    long long modelId);

} // namespace EA::EconomicCalendar
