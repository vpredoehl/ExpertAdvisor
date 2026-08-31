#include "EconomicEventRepository.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <string_view>

namespace EA::EconomicCalendar
{
namespace
{

template <typename Value>
std::optional<Value> OptionalValue(
    const pqxx::row& row,
    const char* column)
{
    const pqxx::field field = row[column];

    if (field.is_null())
        return std::nullopt;

    return field.as<Value>();
}

void ValidateCurrency(const std::string& currency)
{
    if (currency.size() != 3)
        throw std::invalid_argument(
            "economic_event_currency_must_be_three_characters");

    if (!std::all_of(
            currency.begin(),
            currency.end(),
            [](unsigned char value)
            {
                return std::isupper(value) != 0;
            }))
    {
        throw std::invalid_argument(
            "economic_event_currency_must_be_uppercase");
    }
}

EconomicEvent MapEconomicEvent(const pqxx::row& row)
{
    EconomicEvent event;

    event.economicEventId =
        row["economic_event_id"].as<long long>();

    event.currency =
        row["currency"].as<std::string>();

    event.eventFamily =
        row["event_family"].as<std::string>();

    event.eventTimestampUnixMicros =
        row["event_timestamp_unix_micros"]
            .as<std::int64_t>();

    event.eventTimestampUtc =
        row["event_timestamp_utc_text"]
            .as<std::string>();

    event.sourceAgency =
        row["source_agency"].as<std::string>();

    event.sourceEventId =
        OptionalValue<std::string>(
            row,
            "source_event_id");

    event.sourceUrl =
        row["source_url"].as<std::string>();

    event.referencePeriod =
        OptionalValue<std::string>(
            row,
            "reference_period");

    event.eventImportance =
        row["event_importance"].as<int>();

    event.historicalTimeConfidence =
        row["historical_time_confidence"]
            .as<std::string>();

    event.sourceReleaseDate =
        OptionalValue<std::string>(
            row,
            "source_release_date");

    event.sourceReleaseTime =
        OptionalValue<std::string>(
            row,
            "source_release_time");

    event.sourceTimezone =
        OptionalValue<std::string>(
            row,
            "source_timezone");

    if (!row["consensus_value_low"].is_null())
    {
        EconomicEventSelectedConsensus selected;
        selected.forecast.valueKind =
            row["consensus_value_kind"].as<std::string>();
        selected.forecast.canonicalValueLow =
            row["consensus_value_low"].as<double>();
        selected.forecast.canonicalValueHigh =
            OptionalValue<double>(row, "consensus_value_high");
        selected.forecast.unit =
            row["consensus_unit"].as<std::string>();
        selected.forecast.scale =
            row["consensus_scale"].as<double>();
        selected.forecast.qualifier =
            OptionalValue<std::string>(row, "consensus_qualifier");
        selected.provider =
            row["consensus_source"].as<std::string>();

        event.selectedConsensus = std::move(selected);
    }

    if (!row["release_actual_value_low"].is_null())
    {
        EconomicEventReleaseActual actual;
        actual.actual.valueKind =
            row["release_actual_value_kind"].as<std::string>();
        actual.actual.canonicalValueLow =
            row["release_actual_value_low"].as<double>();
        actual.actual.canonicalValueHigh = OptionalValue<double>(
            row, "release_actual_value_high");
        actual.actual.unit =
            row["release_actual_unit"].as<std::string>();
        actual.actual.scale =
            row["release_actual_scale"].as<double>();
        actual.actual.qualifier = OptionalValue<std::string>(
            row, "release_actual_qualifier");
        actual.availableAtUnixMicros =
            row["release_actual_available_at_unix_micros"]
                .as<std::int64_t>();
        actual.sourceAgency =
            row["release_actual_source_agency"].as<std::string>();
        actual.sourceObservationId =
            row["release_actual_source_observation_id"].as<std::string>();
        actual.sourceArtifactPath =
            row["release_actual_source_artifact_path"].as<std::string>();
        actual.sourceArtifactSha256 =
            row["release_actual_source_artifact_sha256"].as<std::string>();
        actual.semanticContract =
            row["release_actual_semantic_contract"].as<std::string>();
        event.releaseActual = std::move(actual);
    }

    return event;
}

std::string EconomicEventProjection(
    std::string_view eventRelation,
    std::string_view consensusRelation,
    std::string_view releaseActualRelation)
{
    const std::string prefix = eventRelation.empty()
        ? std::string{}
        : std::string{eventRelation} + ".";
    const std::string consensusPrefix = consensusRelation.empty()
        ? std::string{}
        : std::string{consensusRelation} + ".";
    const std::string releaseActualPrefix = releaseActualRelation.empty()
        ? std::string{}
        : std::string{releaseActualRelation} + ".";

    return
        prefix + "economic_event_id, " +
        prefix + "currency, " +
        prefix + "event_family, " +
        "ROUND(EXTRACT(EPOCH FROM " + prefix +
            "event_timestamp_utc) * 1000000)::bigint "
            "AS event_timestamp_unix_micros, " +
        "to_char(" + prefix +
            "event_timestamp_utc AT TIME ZONE 'UTC', "
            "'YYYY-MM-DD\"T\"HH24:MI:SS.US\"Z\"') "
            "AS event_timestamp_utc_text, " +
        prefix + "source_agency, " +
        prefix + "source_event_id, " +
        prefix + "source_url, " +
        prefix + "reference_period, " +
        prefix + "event_importance, " +
        prefix + "historical_time_confidence, " +
        prefix + "source_release_date::text AS source_release_date, " +
        prefix + "source_release_time::text AS source_release_time, " +
        prefix + "source_timezone, " +
        consensusPrefix + "consensus_value_low, " +
        consensusPrefix + "consensus_value_high, " +
        consensusPrefix + "consensus_value_kind, " +
        consensusPrefix + "consensus_unit, " +
        consensusPrefix + "consensus_scale, " +
        consensusPrefix + "consensus_qualifier, " +
        consensusPrefix + "consensus_source, " +
        "ROUND(EXTRACT(EPOCH FROM " + releaseActualPrefix +
            "available_at) * 1000000)::bigint "
            "AS release_actual_available_at_unix_micros, " +
        releaseActualPrefix + "actual_value_kind "
            "AS release_actual_value_kind, " +
        releaseActualPrefix + "actual_canonical_value_low "
            "AS release_actual_value_low, " +
        releaseActualPrefix + "actual_canonical_value_high "
            "AS release_actual_value_high, " +
        releaseActualPrefix + "actual_unit AS release_actual_unit, " +
        releaseActualPrefix + "actual_scale AS release_actual_scale, " +
        releaseActualPrefix +
            "actual_qualifier AS release_actual_qualifier, " +
        releaseActualPrefix +
            "source_agency AS release_actual_source_agency, " +
        releaseActualPrefix +
            "source_observation_id AS release_actual_source_observation_id, " +
        releaseActualPrefix +
            "source_artifact_path AS release_actual_source_artifact_path, " +
        releaseActualPrefix +
            "source_artifact_sha256 AS release_actual_source_artifact_sha256, " +
        releaseActualPrefix +
            "semantic_contract AS release_actual_semantic_contract ";
}

std::vector<EconomicEvent> MapEconomicEvents(const pqxx::result& rows)
{
    std::vector<EconomicEvent> events;
    events.reserve(rows.size());

    for (const pqxx::row& row : rows)
        events.push_back(MapEconomicEvent(row));

    return events;
}

} // namespace


bool EconomicEventSchemaExists(
    pqxx::transaction_base& transaction)
{
    const pqxx::result rows = transaction.exec(
        "SELECT to_regclass("
        "'public.economic_event'"
        ") IS NOT NULL AS exists;");

    return
        rows.size() == 1 &&
        rows.one_row()["exists"].as<bool>();
}


std::vector<EconomicEvent> LoadEconomicEvents(
    pqxx::transaction_base& transaction,
    const std::string& currency,
    const std::string& startUtc,
    const std::string& endUtc)
{
    ValidateCurrency(currency);

    //
    // Half-open range:
    //
    //     start <= event < end
    //
    // This matches the interval convention used by the importer and avoids
    // overlap when adjacent windows are queried.
    //
    // Both the numeric epoch representation and the UTC text representation
    // are derived explicitly from timestamptz. Therefore neither result
    // depends on the active PostgreSQL session timezone.
    //
    const pqxx::result rows = transaction.exec(
        "SELECT " + EconomicEventProjection("e", "c", "a") +
        "FROM economic_event e "
        "LEFT JOIN economic_event_selected_consensus c "
        "USING (economic_event_id) "
        "LEFT JOIN economic_event_feature_release_actual a ON "
        "a.economic_event_id = e.economic_event_id "
        "AND a.available_at < $3::timestamptz "
        "WHERE e.currency = $1 "
        "AND e.event_timestamp_utc >= $2::timestamptz "
        "AND e.event_timestamp_utc < $3::timestamptz "
        "ORDER BY "
        "e.event_timestamp_utc ASC, "
        "e.economic_event_id ASC;",
        pqxx::params{
            currency,
            startUtc,
            endUtc});

    return MapEconomicEvents(rows);
}


std::vector<EconomicEvent> LoadEconomicEventsForFeatureRange(
    pqxx::transaction_base& transaction,
    const std::string& currency,
    const std::string& startUtc,
    const std::string& endUtc)
{
    ValidateCurrency(currency);

    // DISTINCT ON seeds one latest row for every authoritative canonical
    // stream. Several canonical streams may map to one model family; retaining
    // each stream's latest prior row lets the shared C++ mapper select the
    // truly latest model-family timestamp without duplicating that mapping in
    // SQL. The disjoint UNION ALL then includes every event required while
    // target bars are processed.  The release-actual join is also bounded by
    // the range's information upper bound.  This prevents an authoritative
    // observation that was not yet knowable anywhere in the requested range
    // from crossing the repository boundary.  The chronological feature
    // engine retains the same strict check for each individual completed bar.
    const pqxx::result rows = transaction.exec(
        "WITH prior_canonical_stream AS ("
        "SELECT DISTINCT ON (source_agency, event_family) "
        "economic_event_id "
        "FROM economic_event "
        "WHERE currency = $1 "
        "AND event_timestamp_utc < $2::timestamptz "
        "ORDER BY source_agency, event_family, "
        "event_timestamp_utc DESC, economic_event_id DESC"
        "), selected_event AS ("
        "SELECT e.* FROM economic_event e "
        "JOIN prior_canonical_stream p USING (economic_event_id) "
        "UNION ALL "
        "SELECT e.* FROM economic_event e "
        "WHERE e.currency = $1 "
        "AND e.event_timestamp_utc >= $2::timestamptz "
        "AND e.event_timestamp_utc <= $3::timestamptz"
        ") SELECT " + EconomicEventProjection("e", "c", "a") +
        "FROM selected_event e "
        "LEFT JOIN economic_event_selected_consensus c "
        "USING (economic_event_id) "
        "LEFT JOIN economic_event_feature_release_actual a ON "
        "a.economic_event_id = e.economic_event_id "
        "AND a.available_at < $3::timestamptz "
        "ORDER BY e.event_timestamp_utc ASC, e.economic_event_id ASC;",
        pqxx::params{
            currency,
            startUtc,
            endUtc});

    return MapEconomicEvents(rows);
}

} // namespace EA::EconomicCalendar
