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
    std::optional<EconomicEventConsensusValue> actual;

    // Retained for diagnostics and repository tests only. Feature computation
    // deliberately ignores provider identity and all provider provenance.
    std::string provider;
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
// when a release occurs exactly at that bar's information cutoff; the strict
// feature-engine release test still prevents actual/surprise leakage.
std::vector<EconomicEvent> LoadEconomicEventsForFeatureRange(
    pqxx::transaction_base& transaction,
    const std::string& currency,
    const std::string& startUtc,
    const std::string& endUtc);

} // namespace EA::EconomicCalendar
