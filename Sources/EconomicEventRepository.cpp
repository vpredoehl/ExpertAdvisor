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

    return event;
}

std::string EconomicEventProjection(
    std::string_view eventRelation,
    std::string_view consensusRelation)
{
    const std::string prefix = eventRelation.empty()
        ? std::string{}
        : std::string{eventRelation} + ".";
    const std::string consensusPrefix = consensusRelation.empty()
        ? std::string{}
        : std::string{consensusRelation} + ".";

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
        consensusPrefix + "consensus_source ";
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
        "SELECT " + EconomicEventProjection("e", "c") +
        "FROM economic_event e "
        "LEFT JOIN economic_event_selected_consensus c "
        "USING (economic_event_id) "
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
    // target bars are processed.
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
        ") SELECT " + EconomicEventProjection("e", "c") +
        "FROM selected_event e "
        "LEFT JOIN economic_event_selected_consensus c "
        "USING (economic_event_id) "
        "ORDER BY e.event_timestamp_utc ASC, e.economic_event_id ASC;",
        pqxx::params{
            currency,
            startUtc,
            endUtc});

    return MapEconomicEvents(rows);
}

} // namespace EA::EconomicCalendar
