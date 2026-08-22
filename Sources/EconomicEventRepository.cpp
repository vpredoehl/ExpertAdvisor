#include "EconomicEventRepository.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

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

    return event;
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
        "SELECT "
        "economic_event_id, "
        "currency, "
        "event_family, "
        "ROUND("
        "EXTRACT(EPOCH FROM event_timestamp_utc) "
        "* 1000000"
        ")::bigint AS event_timestamp_unix_micros, "
        "to_char("
        "event_timestamp_utc AT TIME ZONE 'UTC', "
        "'YYYY-MM-DD\"T\"HH24:MI:SS.US\"Z\"'"
        ") AS event_timestamp_utc_text, "
        "source_agency, "
        "source_event_id, "
        "source_url, "
        "reference_period, "
        "event_importance, "
        "historical_time_confidence, "
        "source_release_date::text "
        "AS source_release_date, "
        "source_release_time::text "
        "AS source_release_time, "
        "source_timezone "
        "FROM economic_event "
        "WHERE currency = $1 "
        "AND event_timestamp_utc >= $2::timestamptz "
        "AND event_timestamp_utc < $3::timestamptz "
        "ORDER BY "
        "event_timestamp_utc ASC, "
        "economic_event_id ASC;",
        pqxx::params{
            currency,
            startUtc,
            endUtc});

    std::vector<EconomicEvent> events;
    events.reserve(rows.size());

    for (const pqxx::row& row : rows)
        events.push_back(MapEconomicEvent(row));

    return events;
}

} // namespace EA::EconomicCalendar
