#include "EconomicEventConsensusRepository.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>

namespace EA::EconomicCalendar
{
namespace
{

struct Column
{
    std::string_view name;
    std::string_view type;
};

constexpr std::array<Column, 41> kPayloadColumns{{
    {"consensus_source", "text"},
    {"source_report_id", "bigint"},
    {"source_event_id", "bigint"},
    {"source_event_name", "text"},
    {"source_period", "text"},
    {"source_priority", "smallint"},
    {"source_timestamp_epoch", "bigint"},
    {"source_date", "timestamp without time zone"},
    {"source_artifact_path", "text"},
    {"match_rule", "text"},
    {"semantic_contract", "text"},
    {"forecast_raw", "text"},
    {"forecast_parse_status", "text"},
    {"forecast_value_kind", "text"},
    {"forecast_value_low", "numeric"},
    {"forecast_value_high", "numeric"},
    {"forecast_canonical_value_low", "numeric"},
    {"forecast_canonical_value_high", "numeric"},
    {"forecast_unit", "text"},
    {"forecast_scale", "numeric"},
    {"forecast_qualifier", "text"},
    {"previous_raw", "text"},
    {"previous_parse_status", "text"},
    {"previous_value_kind", "text"},
    {"previous_value_low", "numeric"},
    {"previous_value_high", "numeric"},
    {"previous_canonical_value_low", "numeric"},
    {"previous_canonical_value_high", "numeric"},
    {"previous_unit", "text"},
    {"previous_scale", "numeric"},
    {"previous_qualifier", "text"},
    {"actual_raw", "text"},
    {"actual_parse_status", "text"},
    {"actual_value_kind", "text"},
    {"actual_value_low", "numeric"},
    {"actual_value_high", "numeric"},
    {"actual_canonical_value_low", "numeric"},
    {"actual_canonical_value_high", "numeric"},
    {"actual_unit", "text"},
    {"actual_scale", "numeric"},
    {"actual_qualifier", "text"}
}};

void AppendValue(pqxx::params& parameters, const ConsensusParsedValue& value)
{
    parameters.append(value.raw);
    parameters.append(value.parseStatus);
    parameters.append(value.valueKind);
    parameters.append(value.valueLow);
    parameters.append(value.valueHigh);
    parameters.append(value.canonicalValueLow);
    parameters.append(value.canonicalValueHigh);
    parameters.append(value.unit);
    parameters.append(value.scale);
    parameters.append(value.qualifier);
}

pqxx::params CandidateParameters(
    const EconomicEventConsensusCandidate& candidate)
{
    pqxx::params parameters;
    parameters.append(candidate.economicEventId);
    parameters.append(candidate.consensusSource);
    parameters.append(candidate.sourceReportId);
    parameters.append(candidate.secondarySourceEventId);
    parameters.append(candidate.secondarySourceEventName);
    parameters.append(candidate.secondarySourcePeriod);
    parameters.append(candidate.secondarySourcePriority);
    parameters.append(candidate.secondarySourceTimestampEpoch);
    parameters.append(candidate.secondarySourceDate);
    parameters.append(candidate.secondarySourceArtifactPath);
    parameters.append(candidate.matchRule);
    parameters.append(candidate.semanticContract);
    AppendValue(parameters, candidate.forecast);
    AppendValue(parameters, candidate.previous);
    AppendValue(parameters, candidate.actual);
    return parameters;
}

const std::string& ExactPayloadSql()
{
    static const std::string sql = []
    {
        std::string value =
            "SELECT EXISTS (SELECT 1 FROM economic_event_consensus WHERE "
            "economic_event_id = $1";
        for (std::size_t index = 0; index < kPayloadColumns.size(); ++index)
        {
            value += " AND ";
            value += kPayloadColumns[index].name;
            value += " IS NOT DISTINCT FROM $" + std::to_string(index + 2);
            value += "::";
            value += kPayloadColumns[index].type;
        }
        value += ") AS exact;";
        return value;
    }();
    return sql;
}

const std::string& InsertSql()
{
    static const std::string sql = []
    {
        std::string value =
            "INSERT INTO economic_event_consensus (economic_event_id";
        for (const auto& column : kPayloadColumns)
        {
            value += ", ";
            value += column.name;
        }
        value += ") VALUES ($1::bigint";
        for (std::size_t index = 0; index < kPayloadColumns.size(); ++index)
        {
            value += ", $" + std::to_string(index + 2) + "::";
            value += kPayloadColumns[index].type;
        }
        value += ") ON CONFLICT DO NOTHING RETURNING economic_event_id;";
        return value;
    }();
    return sql;
}

std::string OfficialIdentityDiagnostic(
    pqxx::transaction_base& transaction,
    const EconomicEventConsensusCandidate& candidate)
{
    const pqxx::row row = transaction.exec(
        "SELECT "
        "EXISTS (SELECT 1 FROM economic_event WHERE economic_event_id = $1) "
        "AS present, "
        "EXISTS (SELECT 1 FROM economic_event "
        "WHERE economic_event_id = $1 AND event_family = $2 "
        "AND event_timestamp_utc = $3::timestamptz "
        "AND source_agency = $4 "
        "AND source_event_id IS NOT DISTINCT FROM $5::text "
        "AND reference_period IS NOT DISTINCT FROM $6::text "
        "AND source_release_date IS NOT DISTINCT FROM $7::date) AS exact;",
        pqxx::params{
            candidate.economicEventId,
            candidate.eventFamily,
            candidate.eventTimestampUtc,
            candidate.sourceAgency,
            candidate.sourceEventId,
            candidate.referencePeriod,
            candidate.sourceReleaseDate}).one_row();
    if (!row["present"].as<bool>())
        return "economic_event_id_not_found";
    return row["exact"].as<bool>() ? std::string{}
                                    : "authoritative_identity_conflict";
}

bool StoredIdentityExists(
    pqxx::transaction_base& transaction,
    std::int64_t economicEventId)
{
    return transaction.query_value<bool>(
        "SELECT EXISTS (SELECT 1 FROM economic_event_consensus "
        "WHERE economic_event_id = $1);",
        pqxx::params{economicEventId});
}

bool StoredSourceEventMapsElsewhere(
    pqxx::transaction_base& transaction,
    const EconomicEventConsensusCandidate& candidate)
{
    return transaction.query_value<bool>(
        "SELECT EXISTS (SELECT 1 FROM economic_event_consensus "
        "WHERE consensus_source = $1 AND source_event_id = $2 "
        "AND economic_event_id <> $3);",
        pqxx::params{
            candidate.consensusSource,
            candidate.secondarySourceEventId,
            candidate.economicEventId});
}

EconomicEventConsensusImportReport Compare(
    pqxx::transaction_base& transaction,
    const std::vector<EconomicEventConsensusCandidate>& candidates)
{
    EconomicEventConsensusImportReport report;
    report.items.reserve(candidates.size());
    for (const auto& candidate : candidates)
    {
        EconomicEventConsensusImportItemResult item;
        item.economicEventId = candidate.economicEventId;
        item.diagnostic = OfficialIdentityDiagnostic(transaction, candidate);
        if (!item.diagnostic.empty())
        {
            item.disposition = EconomicEventConsensusImportDisposition::rejected;
            ++report.rejected;
        }
        else if (StoredIdentityExists(transaction, candidate.economicEventId))
        {
            const bool exact = transaction.query_value<bool>(
                ExactPayloadSql(), CandidateParameters(candidate));
            if (exact)
            {
                item.disposition =
                    EconomicEventConsensusImportDisposition::unchanged;
                item.diagnostic = "exact_match";
                ++report.unchanged;
            }
            else
            {
                item.disposition =
                    EconomicEventConsensusImportDisposition::rejected;
                item.diagnostic = "immutable_persisted_payload_conflict";
                ++report.rejected;
            }
        }
        else if (StoredSourceEventMapsElsewhere(transaction, candidate))
        {
            item.disposition = EconomicEventConsensusImportDisposition::rejected;
            item.diagnostic = "secondary_source_event_identity_conflict";
            ++report.rejected;
        }
        else
        {
            item.disposition = EconomicEventConsensusImportDisposition::inserted;
            item.diagnostic = "new_consensus_enrichment";
            ++report.inserted;
        }
        report.items.push_back(std::move(item));
    }
    return report;
}

} // namespace


EconomicEventConsensusImportReport CompareEconomicEventConsensusBatch(
    pqxx::transaction_base& transaction,
    const std::vector<EconomicEventConsensusCandidate>& candidates)
{
    return Compare(transaction, candidates);
}


EconomicEventConsensusImportReport ApplyEconomicEventConsensusBatch(
    pqxx::connection& connection,
    const std::vector<EconomicEventConsensusCandidate>& candidates)
{
    pqxx::work transaction{connection};
    EconomicEventConsensusImportReport report = Compare(transaction, candidates);
    if (report.rejected != 0)
    {
        for (auto& item : report.items)
        {
            if (item.disposition ==
                EconomicEventConsensusImportDisposition::inserted)
            {
                item.disposition =
                    EconomicEventConsensusImportDisposition::rejected;
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
            EconomicEventConsensusImportDisposition::inserted)
        {
            const pqxx::result inserted = transaction.exec(
                InsertSql(), CandidateParameters(candidates[index]));
            if (inserted.empty())
            {
                const bool exact = transaction.query_value<bool>(
                    ExactPayloadSql(), CandidateParameters(candidates[index]));
                if (!exact)
                    throw std::runtime_error(
                        "concurrent_consensus_identity_conflict:economic_event_id=" +
                        std::to_string(candidates[index].economicEventId));
                report.items[index].disposition =
                    EconomicEventConsensusImportDisposition::unchanged;
                report.items[index].diagnostic = "exact_concurrent_retry";
                --report.inserted;
                ++report.unchanged;
            }
        }
    }
    transaction.commit();
    return report;
}

} // namespace EA::EconomicCalendar
