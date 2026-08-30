#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <pqxx/pqxx>

namespace EA::EconomicCalendar
{

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
std::vector<EconomicEvent> LoadEconomicEventsForFeatureRange(
    pqxx::transaction_base& transaction,
    const std::string& currency,
    const std::string& startUtc,
    const std::string& endUtc);

} // namespace EA::EconomicCalendar
