#include "EconomicEventImportRepository.hpp"

#include "EconomicEventImportValidation.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace EA::EconomicCalendar
{
namespace
{

struct StoredEconomicEvent
{
    std::string currency;
    std::string eventFamily;
    std::int64_t eventTimestampUnixMicros = 0;
    std::string sourceAgency;
    std::optional<std::string> sourceEventId;
    std::string sourceUrl;
    std::optional<std::string> referencePeriod;
    int eventImportance = 0;
    std::string historicalTimeConfidence;
    std::optional<std::string> sourceReleaseDate;
    std::optional<std::string> sourceReleaseTime;
    std::optional<std::string> sourceTimezone;
};

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

StoredEconomicEvent MapStored(const pqxx::row& row)
{
    StoredEconomicEvent stored;
    stored.currency = row["currency"].as<std::string>();
    stored.eventFamily = row["event_family"].as<std::string>();
    stored.eventTimestampUnixMicros =
        row["event_timestamp_unix_micros"].as<std::int64_t>();
    stored.sourceAgency = row["source_agency"].as<std::string>();
    stored.sourceEventId = OptionalValue<std::string>(row, "source_event_id");
    stored.sourceUrl = row["source_url"].as<std::string>();
    stored.referencePeriod = OptionalValue<std::string>(row, "reference_period");
    stored.eventImportance = row["event_importance"].as<int>();
    stored.historicalTimeConfidence =
        row["historical_time_confidence"].as<std::string>();
    stored.sourceReleaseDate =
        OptionalValue<std::string>(row, "source_release_date");
    stored.sourceReleaseTime =
        OptionalValue<std::string>(row, "source_release_time");
    stored.sourceTimezone = OptionalValue<std::string>(row, "source_timezone");
    return stored;
}

bool Equal(
    const StoredEconomicEvent& stored,
    const AuthoritativeEconomicEventCandidate& candidate)
{
    return
        stored.currency == candidate.currency &&
        stored.eventFamily == candidate.eventFamily &&
        stored.eventTimestampUnixMicros == candidate.eventTimestampUnixMicros &&
        stored.sourceAgency == candidate.sourceAgency &&
        stored.sourceEventId ==
            std::optional<std::string>{candidate.sourceEventId} &&
        stored.sourceUrl == candidate.sourceUrl &&
        stored.referencePeriod == candidate.referencePeriod &&
        stored.eventImportance == candidate.eventImportance &&
        stored.historicalTimeConfidence == candidate.historicalTimeConfidence &&
        stored.sourceReleaseDate == candidate.sourceReleaseDate &&
        stored.sourceReleaseTime == candidate.sourceReleaseTime &&
        stored.sourceTimezone == candidate.sourceTimezone;
}

AuthoritativeEconomicEventCandidate CandidateFromStored(
    const StoredEconomicEvent& stored)
{
    AuthoritativeEconomicEventCandidate candidate;
    candidate.currency = stored.currency;
    candidate.eventFamily = stored.eventFamily;
    candidate.eventTimestampUnixMicros = stored.eventTimestampUnixMicros;
    candidate.sourceAgency = stored.sourceAgency;
    candidate.sourceEventId = stored.sourceEventId.value_or("");
    candidate.sourceUrl = stored.sourceUrl;
    candidate.referencePeriod = stored.referencePeriod;
    candidate.eventImportance = stored.eventImportance;
    candidate.historicalTimeConfidence = stored.historicalTimeConfidence;
    candidate.sourceReleaseDate = stored.sourceReleaseDate;
    candidate.sourceReleaseTime = stored.sourceReleaseTime;
    candidate.sourceTimezone = stored.sourceTimezone;
    return candidate;
}

std::vector<StoredEconomicEvent> LoadCollisions(
    pqxx::transaction_base& transaction,
    const AuthoritativeEconomicEventCandidate& candidate)
{
    const pqxx::result rows = transaction.exec(
        "SELECT currency, event_family, "
        "ROUND(EXTRACT(EPOCH FROM event_timestamp_utc) * 1000000)::bigint "
        "AS event_timestamp_unix_micros, source_agency, source_event_id, "
        "source_url, reference_period, event_importance, "
        "historical_time_confidence, source_release_date::text "
        "AS source_release_date, source_release_time::text "
        "AS source_release_time, source_timezone "
        "FROM economic_event "
        "WHERE (source_agency = $1 AND source_event_id = $2) "
        "OR (source_agency = $1 AND event_family = $3 "
        "AND event_timestamp_utc = "
        "TIMESTAMPTZ 'epoch' + $4::bigint * INTERVAL '1 microsecond') "
        "ORDER BY economic_event_id;",
        pqxx::params{
            candidate.sourceAgency,
            candidate.sourceEventId,
            candidate.eventFamily,
            candidate.eventTimestampUnixMicros});

    std::vector<StoredEconomicEvent> collisions;
    collisions.reserve(rows.size());
    for (const pqxx::row& row : rows)
        collisions.push_back(MapStored(row));
    return collisions;
}

EconomicEventImportReport Compare(
    pqxx::transaction_base& transaction,
    const std::vector<AuthoritativeEconomicEventCandidate>& candidates)
{
    EconomicEventImportReport report;
    report.items.reserve(candidates.size());

    for (const auto& candidate : candidates)
    {
        const auto collisions = LoadCollisions(transaction, candidate);
        EconomicEventImportItemResult item;
        item.sourceEventId = candidate.sourceEventId;

        const auto identity = std::find_if(
            collisions.begin(),
            collisions.end(),
            [&](const auto& stored)
            {
                return stored.sourceEventId ==
                    std::optional<std::string>{candidate.sourceEventId};
            });
        const bool disallowedTimestampCollision = std::any_of(
            collisions.begin(),
            collisions.end(),
            [&](const auto& stored)
            {
                return stored.eventFamily == candidate.eventFamily &&
                    stored.eventTimestampUnixMicros ==
                        candidate.eventTimestampUnixMicros &&
                    stored.sourceEventId !=
                        std::optional<std::string>{candidate.sourceEventId} &&
                    !IsPermittedEconomicEventTimestampCoexistence(
                        CandidateFromStored(stored), candidate);
            });

        if (identity == collisions.end() && !disallowedTimestampCollision)
        {
            item.disposition = EconomicEventImportDisposition::inserted;
            item.diagnostic = "new_authoritative_identity";
            ++report.inserted;
        }
        else if (identity != collisions.end() && Equal(*identity, candidate) &&
                 !disallowedTimestampCollision)
        {
            item.disposition = EconomicEventImportDisposition::unchanged;
            item.diagnostic = "exact_match";
            ++report.unchanged;
        }
        else
        {
            item.disposition = EconomicEventImportDisposition::rejected;
            const bool exactTimestampConflict =
                !collisions.empty() &&
                identity != collisions.end() &&
                identity->sourceEventId ==
                    std::optional<std::string>{candidate.sourceEventId} &&
                identity->historicalTimeConfidence == "exact" &&
                identity->eventTimestampUnixMicros !=
                    candidate.eventTimestampUnixMicros;
            item.diagnostic = exactTimestampConflict
                ? "existing_exact_timestamp_conflict"
                : "authoritative_identity_or_timestamp_collision";
            ++report.rejected;
        }

        report.items.push_back(std::move(item));
    }

    return report;
}

void InsertCandidate(
    pqxx::transaction_base& transaction,
    const AuthoritativeEconomicEventCandidate& candidate)
{
    transaction.exec(
        "INSERT INTO economic_event ("
        "currency, event_family, event_timestamp_utc, source_agency, "
        "source_event_id, source_url, reference_period, event_importance, "
        "historical_time_confidence, source_release_date, "
        "source_release_time, source_timezone) VALUES ("
        "$1, $2, TIMESTAMPTZ 'epoch' + $3::bigint * INTERVAL '1 microsecond', "
        "$4, $5, $6, $7, $8, $9, $10::date, $11::time, $12);",
        pqxx::params{
            candidate.currency,
            candidate.eventFamily,
            candidate.eventTimestampUnixMicros,
            candidate.sourceAgency,
            candidate.sourceEventId,
            candidate.sourceUrl,
            candidate.referencePeriod,
            candidate.eventImportance,
            candidate.historicalTimeConfidence,
            candidate.sourceReleaseDate,
            candidate.sourceReleaseTime,
            candidate.sourceTimezone});
}

} // namespace


const char* EconomicEventImportDispositionName(
    EconomicEventImportDisposition disposition)
{
    switch (disposition)
    {
        case EconomicEventImportDisposition::inserted:
            return "inserted";
        case EconomicEventImportDisposition::unchanged:
            return "unchanged";
        case EconomicEventImportDisposition::updated:
            return "updated";
        case EconomicEventImportDisposition::rejected:
            return "rejected";
    }
    return "unknown";
}


EconomicEventImportReport CompareEconomicEventImportBatch(
    pqxx::transaction_base& transaction,
    const std::vector<AuthoritativeEconomicEventCandidate>& candidates)
{
    return Compare(transaction, candidates);
}


EconomicEventImportReport ApplyEconomicEventImportBatch(
    pqxx::connection& connection,
    const std::vector<AuthoritativeEconomicEventCandidate>& candidates)
{
    pqxx::work transaction{connection};
    EconomicEventImportReport report = Compare(transaction, candidates);

    if (report.rejected != 0)
    {
        // No writes have occurred.  Reclassify would-be inserts because this
        // atomic batch is deliberately not partially applied.
        for (auto& item : report.items)
        {
            if (item.disposition == EconomicEventImportDisposition::inserted)
            {
                item.disposition = EconomicEventImportDisposition::rejected;
                item.diagnostic = "batch_rolled_back_due_to_conflict";
                ++report.rejected;
            }
        }
        report.inserted = 0;
        return report;
    }

    for (std::size_t index = 0; index < candidates.size(); ++index)
    {
        if (report.items[index].disposition ==
            EconomicEventImportDisposition::inserted)
        {
            InsertCandidate(transaction, candidates[index]);
        }
    }

    transaction.commit();
    return report;
}

} // namespace EA::EconomicCalendar
