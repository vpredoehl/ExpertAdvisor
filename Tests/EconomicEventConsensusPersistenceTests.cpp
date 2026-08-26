#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <pqxx/pqxx>

#include "../Sources/EconomicEventConsensusImport.hpp"

using namespace EA::EconomicCalendar;

namespace
{

std::string RequiredEnvironment(const char* name)
{
    const char* value = std::getenv(name);
    if (!value || !*value)
        throw std::runtime_error(std::string{"missing environment: "} + name);
    return value;
}

const EconomicEventConsensusCandidate& Find(
    const std::vector<EconomicEventConsensusCandidate>& candidates,
    std::int64_t economicEventId)
{
    const auto iterator = std::find_if(
        candidates.begin(), candidates.end(),
        [=](const auto& candidate)
        {
            return candidate.economicEventId == economicEventId;
        });
    if (iterator == candidates.end())
        throw std::runtime_error("fixture_candidate_not_found");
    return *iterator;
}

void SeedOfficial(
    pqxx::connection& connection,
    const EconomicEventConsensusCandidate& candidate)
{
    pqxx::work transaction{connection};
    transaction.exec(
        "INSERT INTO economic_event (economic_event_id, currency, "
        "event_family, event_timestamp_utc, source_agency, source_event_id, "
        "source_url, reference_period, event_importance, "
        "historical_time_confidence, source_release_date) VALUES ("
        "$1, 'USD', $2, $3::timestamptz, $4, $5, $6, $7, 3, 'exact', "
        "$8::date);",
        pqxx::params{
            candidate.economicEventId,
            candidate.eventFamily,
            candidate.eventTimestampUtc,
            candidate.sourceAgency,
            candidate.sourceEventId,
            "https://example.invalid/fixture/" +
                std::to_string(candidate.economicEventId),
            candidate.referencePeriod,
            candidate.sourceReleaseDate});
    transaction.commit();
}

template <typename Function>
void ExpectInvalidArgument(Function function)
{
    bool rejected = false;
    try
    {
        function();
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }
    assert(rejected);
}

long long Count(pqxx::connection& connection, const std::string& predicate)
{
    pqxx::read_transaction transaction{connection};
    return transaction.query_value<long long>(
        "SELECT count(*) FROM economic_event_consensus WHERE " + predicate);
}

} // namespace


int main()
{
    const std::string database = RequiredEnvironment("LSTM_DB_NAME");
    if (database.rfind("ea_consensus_phase1_test_", 0) != 0)
        throw std::runtime_error("refusing_non_disposable_database");
    const std::filesystem::path input = RequiredEnvironment("CONSENSUS_INPUT");
    const auto all = LoadAndValidateOandaEconomicConsensusCsv(input);
    assert(all.size() == 1416);
    assert(std::count_if(all.begin(), all.end(), [](const auto& candidate)
    {
        return candidate.forecast.parseStatus == "missing";
    }) == 11);

    std::vector<EconomicEventConsensusCandidate> selected{
        Find(all, 78),    // normal CPI
        Find(all, 79),    // employment/count
        Find(all, 1636),  // GDP percent
        Find(all, 2154),  // FOMC range
        Find(all, 2146),  // FOMC scalar
        Find(all, 685),   // verified CPI OANDA blank
        Find(all, 779),   // CPI report-698 y/y exception
        Find(all, 1979)   // legitimate multi-month PCE release
    };

    const std::string connectionString =
        "host=" + RequiredEnvironment("LSTM_DB_HOST") +
        " user=" + RequiredEnvironment("LSTM_DB_USER") +
        " dbname=" + database;
    pqxx::connection connection{connectionString};
    for (const auto& candidate : selected)
        SeedOfficial(connection, candidate);

    const auto dryRun = RunEconomicEventConsensusImport(
        connection, selected, EconomicEventConsensusImportMode::dryRun);
    assert(dryRun.inserted == selected.size());
    assert(dryRun.unchanged == 0);
    assert(dryRun.rejected == 0);
    assert(Count(connection, "true") == 0);

    const auto firstApply = RunEconomicEventConsensusImport(
        connection, selected, EconomicEventConsensusImportMode::apply);
    assert(firstApply.inserted == selected.size());
    assert(firstApply.unchanged == 0);
    assert(firstApply.rejected == 0);
    assert(Count(connection, "true") ==
           static_cast<long long>(selected.size()));

    assert(Count(connection,
        "economic_event_id = 78 AND source_report_id = 699 AND "
        "forecast_value_low = -0.1 AND forecast_canonical_value_low = -0.1 "
        "AND forecast_unit = 'percent' AND forecast_qualifier = 'm/m'") == 1);
    assert(Count(connection,
        "economic_event_id = 79 AND forecast_value_low = 90 AND "
        "forecast_canonical_value_low = 90000 AND forecast_scale = 1000 "
        "AND forecast_unit = 'count'") == 1);
    assert(Count(connection,
        "economic_event_id = 1636 AND forecast_value_low = 1.9 AND "
        "forecast_unit = 'percent' AND forecast_qualifier IS NULL") == 1);
    assert(Count(connection,
        "economic_event_id = 2154 AND forecast_value_kind = 'range' AND "
        "forecast_value_low = 0 AND forecast_value_high = 0.25") == 1);
    assert(Count(connection,
        "economic_event_id = 2146 AND forecast_value_kind = 'scalar' AND "
        "forecast_value_low = 0.25 AND forecast_value_high IS NULL") == 1);
    assert(Count(connection,
        "economic_event_id = 685 AND forecast_raw IS NULL AND "
        "forecast_parse_status = 'missing' AND forecast_value_low IS NULL "
        "AND forecast_canonical_value_low IS NULL") == 1);
    assert(Count(connection,
        "economic_event_id = 779 AND source_report_id = 698 AND "
        "forecast_raw = '3,1% y/y' AND forecast_value_low = 3.1 AND "
        "forecast_qualifier = 'y/y'") == 1);
    assert(Count(connection,
        "economic_event_id = 1979 AND source_report_id = 694 AND "
        "source_period = 'November' AND "
        "match_rule = 'authoritative_reference_period'") == 1);

    const auto secondApply = RunEconomicEventConsensusImport(
        connection, selected, EconomicEventConsensusImportMode::apply);
    assert(secondApply.inserted == 0);
    assert(secondApply.unchanged == selected.size());
    assert(secondApply.rejected == 0);

    const auto makeMyfxbook = [](EconomicEventConsensusCandidate candidate,
                                 const std::string& consensus,
                                 const std::string& observation)
    {
        candidate.consensusSource = "MYFXBOOK";
        candidate.sourceReportId.reset();
        candidate.secondarySourceEventId = -265326876;
        candidate.secondarySourceObservationId = observation;
        candidate.secondarySourceEventName = "CPI_MOM";
        candidate.secondarySourcePeriod.reset();
        candidate.secondarySourcePriority.reset();
        candidate.secondarySourceTimestampEpoch.reset();
        candidate.secondarySourceDate.reset();
        candidate.secondarySourceArtifactPath =
            "EconomicCalendar/raw/myfxbook/myfxbook_consensus_history.json";
        candidate.secondarySourceArtifactSha256 =
            "1538055386838d8ec50ab9cce40153f69aa8bf841c2ae50f67751b2c465432cb";
        candidate.candidateClassification = "myfxbook_oanda_blank_fill";
        candidate.matchRule = "oanda_blank_official_identity_unique";
        candidate.semanticContract = "myfxbook_consensus_observation_v1";
        candidate.providerProvenance =
            R"({"provider":"MYFXBOOK","source_observation_ordinal":1})";
        candidate.forecast.raw = consensus;
        candidate.forecast.parseStatus = "parsed";
        candidate.forecast.valueKind = "scalar";
        candidate.forecast.valueLow = consensus;
        candidate.forecast.valueHigh.reset();
        candidate.forecast.canonicalValueLow = consensus;
        candidate.forecast.canonicalValueHigh.reset();
        candidate.forecast.unit = "percent";
        candidate.forecast.scale = "1";
        candidate.forecast.qualifier = "m/m";
        candidate.previous = {};
        candidate.previous.parseStatus = "missing";
        candidate.actual = {};
        candidate.actual.parseStatus = "missing";
        return candidate;
    };

    auto myfxbookFill = makeMyfxbook(
        Find(all, 685), "0.0",
        "myfxbook:event:-265326876:date:2023-12-12:series:1:observation:1");
    const auto fill = RunEconomicEventConsensusImport(
        connection, {myfxbookFill}, EconomicEventConsensusImportMode::apply);
    assert(fill.inserted == 1 && fill.unchanged == 0 && fill.rejected == 0);
    const auto fillRetry = RunEconomicEventConsensusImport(
        connection, {myfxbookFill}, EconomicEventConsensusImportMode::apply);
    assert(fillRetry.inserted == 0 && fillRetry.unchanged == 1 &&
           fillRetry.rejected == 0);
    assert(Count(connection,
        "economic_event_id = 685 AND consensus_source = 'MYFXBOOK' AND "
        "forecast_canonical_value_low = 0.0 AND "
        "candidate_classification = 'myfxbook_oanda_blank_fill' AND "
        "source_artifact_sha256 IS NOT NULL AND "
        "provider_provenance ? 'source_observation_ordinal'") == 1);

    auto conflictingFill = myfxbookFill;
    conflictingFill.forecast.raw = "0.1";
    conflictingFill.forecast.valueLow = "0.1";
    conflictingFill.forecast.canonicalValueLow = "0.1";
    const auto fillConflict = RunEconomicEventConsensusImport(
        connection, {conflictingFill}, EconomicEventConsensusImportMode::apply);
    assert(fillConflict.rejected == 1);
    assert(fillConflict.items.front().diagnostic ==
           "immutable_persisted_payload_conflict");

    auto overlap = makeMyfxbook(
        Find(all, 78), "-0.1",
        "myfxbook:event:-265326876:date:2011-08-18:series:1:observation:2");
    const auto overlapResult = RunEconomicEventConsensusImport(
        connection, {overlap}, EconomicEventConsensusImportMode::apply);
    assert(overlapResult.rejected == 1);
    assert(overlapResult.items.front().diagnostic ==
           "populated_consensus_precedence_conflict");
    assert(Count(connection,
        "economic_event_id = 78 AND consensus_source = 'OANDA'") == 1);

    auto conflicting = selected.front();
    conflicting.forecast.raw = "-0.10% m/m";
    conflicting.forecast.valueLow = "-0.10";
    const auto conflictResult = RunEconomicEventConsensusImport(
        connection, {conflicting}, EconomicEventConsensusImportMode::apply);
    assert(conflictResult.inserted == 0);
    assert(conflictResult.unchanged == 0);
    assert(conflictResult.rejected == 1);
    assert(conflictResult.items.front().diagnostic ==
           "immutable_persisted_payload_conflict");

    auto unapprovedCpi = selected.front();
    unapprovedCpi.sourceReportId = 698;
    ExpectInvalidArgument([&]
    {
        RunEconomicEventConsensusImport(
            connection, {unapprovedCpi},
            EconomicEventConsensusImportMode::dryRun);
    });
    ExpectInvalidArgument([&]
    {
        RunEconomicEventConsensusImport(
            connection, {selected.front(), selected.front()},
            EconomicEventConsensusImportMode::dryRun);
    });
    auto duplicateSourceEvent = selected[1];
    duplicateSourceEvent.secondarySourceObservationId =
        selected.front().secondarySourceObservationId;
    ExpectInvalidArgument([&]
    {
        RunEconomicEventConsensusImport(
            connection, {selected.front(), duplicateSourceEvent},
            EconomicEventConsensusImportMode::dryRun);
    });

    bool updateRejected = false;
    try
    {
        pqxx::work transaction{connection};
        transaction.exec(
            "UPDATE economic_event_consensus SET match_rule = 'changed' "
            "WHERE economic_event_id = 78;");
        transaction.commit();
    }
    catch (const pqxx::sql_error&)
    {
        updateRejected = true;
    }
    assert(updateRejected);
    assert(Count(connection, "economic_event_id = 78 AND match_rule <> 'changed'")
           == 1);

    return 0;
}
