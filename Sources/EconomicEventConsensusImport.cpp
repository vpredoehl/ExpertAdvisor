#include "EconomicEventConsensusImport.hpp"

#include "EconomicEventConsensusRepository.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace EA::EconomicCalendar
{
namespace
{

constexpr std::array<std::string_view, 52> kRequiredColumns{
    "economic_event_id", "event_family", "event_timestamp_utc",
    "source_agency", "source_event_id", "reference_period",
    "event_importance", "historical_time_confidence",
    "source_release_date", "source_release_time", "source_timezone",
    "official_source_url", "consensus_source", "oanda_report_id",
    "oanda_event_id", "oanda_event", "oanda_period", "oanda_priority",
    "oanda_timestamp", "oanda_date", "oanda_source_file", "match_rule",
    "forecast_raw", "previous_raw", "actual_raw",
    "forecast_parse_status", "forecast_value_kind", "forecast_value_low",
    "forecast_value_high", "forecast_canonical_value_low",
    "forecast_canonical_value_high", "forecast_unit", "forecast_scale",
    "forecast_qualifier", "previous_parse_status", "previous_value_kind",
    "previous_value_low", "previous_value_high",
    "previous_canonical_value_low", "previous_canonical_value_high",
    "previous_unit", "previous_scale", "previous_qualifier",
    "actual_parse_status", "actual_value_kind", "actual_value_low",
    "actual_value_high", "actual_canonical_value_low",
    "actual_canonical_value_high", "actual_unit", "actual_scale",
    "actual_qualifier"};

struct Decimal
{
    __int128 coefficient = 0;
    std::size_t fractionalDigits = 0;
};

std::string EnvironmentOr(const char* name, const char* fallback)
{
    const char* value = std::getenv(name);
    return value && *value ? value : fallback;
}

std::string ConnectionString()
{
    return "host=" + EnvironmentOr("LSTM_DB_HOST", "127.0.0.1") +
        " user=" + EnvironmentOr("LSTM_DB_USER", "pqxx") +
        " dbname=" + EnvironmentOr("LSTM_DB_NAME", "LSTM");
}

std::optional<std::string> OptionalText(const std::string& value)
{
    return value.empty() ? std::nullopt
                         : std::optional<std::string>{value};
}

template <typename Integer>
Integer ParseInteger(
    const std::string& value,
    std::string_view column,
    std::size_t rowNumber)
{
    Integer result{};
    const char* begin = value.data();
    const char* end = begin + value.size();
    const auto parsed = std::from_chars(begin, end, result);
    if (value.empty() || parsed.ec != std::errc{} || parsed.ptr != end)
    {
        throw std::invalid_argument(
            "invalid_integer:row=" + std::to_string(rowNumber) +
            ":column=" + std::string{column});
    }
    return result;
}

Decimal ParseDecimal(
    const std::string& value,
    std::string_view field,
    std::int64_t economicEventId)
{
    if (value.empty())
        throw std::invalid_argument(
            "missing_decimal:economic_event_id=" +
            std::to_string(economicEventId) + ":field=" +
            std::string{field});

    std::size_t index = 0;
    bool negative = false;
    if (value[index] == '-' || value[index] == '+')
    {
        negative = value[index] == '-';
        ++index;
    }
    if (index == value.size())
        throw std::invalid_argument("invalid_decimal:" + std::string{field});

    std::string digits;
    bool decimalPointSeen = false;
    std::size_t fractionalDigits = 0;
    for (; index < value.size(); ++index)
    {
        const char character = value[index];
        if (character == '.' && !decimalPointSeen)
        {
            decimalPointSeen = true;
            continue;
        }
        if (character < '0' || character > '9')
            throw std::invalid_argument(
                "invalid_decimal:economic_event_id=" +
                std::to_string(economicEventId) + ":field=" +
                std::string{field});
        digits.push_back(character);
        if (decimalPointSeen)
            ++fractionalDigits;
    }
    if (digits.empty() || (!value.empty() && value.back() == '.'))
        throw std::invalid_argument("invalid_decimal:" + std::string{field});

    __int128 coefficient = 0;
    for (const char digit : digits)
    {
        constexpr __int128 kMaximum =
            (static_cast<__int128>(1) << 126) - 1;
        if (coefficient > (kMaximum - (digit - '0')) / 10)
            throw std::invalid_argument(
                "decimal_out_of_range:economic_event_id=" +
                std::to_string(economicEventId) + ":field=" +
                std::string{field});
        coefficient = coefficient * 10 + (digit - '0');
    }
    if (negative)
        coefficient = -coefficient;
    while (fractionalDigits > 0 && coefficient % 10 == 0)
    {
        coefficient /= 10;
        --fractionalDigits;
    }
    return {coefficient, fractionalDigits};
}

__int128 PowerOfTen(std::size_t exponent)
{
    __int128 result = 1;
    for (std::size_t index = 0; index < exponent; ++index)
        result *= 10;
    return result;
}

bool EqualDecimal(const Decimal& left, const Decimal& right)
{
    if (left.fractionalDigits == right.fractionalDigits)
        return left.coefficient == right.coefficient;
    if (left.fractionalDigits < right.fractionalDigits)
    {
        return left.coefficient *
                PowerOfTen(right.fractionalDigits - left.fractionalDigits) ==
            right.coefficient;
    }
    return left.coefficient ==
        right.coefficient *
            PowerOfTen(left.fractionalDigits - right.fractionalDigits);
}

Decimal Multiply(const Decimal& left, const Decimal& right)
{
    Decimal product{
        left.coefficient * right.coefficient,
        left.fractionalDigits + right.fractionalDigits};
    while (product.fractionalDigits > 0 && product.coefficient % 10 == 0)
    {
        product.coefficient /= 10;
        --product.fractionalDigits;
    }
    return product;
}

bool ValidDate(const std::string& value)
{
    if (value.size() != 10 || value[4] != '-' || value[7] != '-')
        return false;
    int year = 0;
    unsigned month = 0;
    unsigned day = 0;
    const auto parsePart = [&](std::size_t offset, std::size_t length, auto& out)
    {
        const char* begin = value.data() + offset;
        const char* end = begin + length;
        const auto parsed = std::from_chars(begin, end, out);
        return parsed.ec == std::errc{} && parsed.ptr == end;
    };
    if (!parsePart(0, 4, year) || !parsePart(5, 2, month) ||
        !parsePart(8, 2, day))
        return false;
    return std::chrono::year_month_day{
        std::chrono::year{year},
        std::chrono::month{month},
        std::chrono::day{day}}.ok();
}

std::vector<std::vector<std::string>> ParseCsv(
    const std::filesystem::path& path)
{
    std::ifstream stream{path, std::ios::binary};
    if (!stream)
        throw std::invalid_argument("cannot_open_consensus_csv:" + path.string());
    const std::string input{
        std::istreambuf_iterator<char>{stream},
        std::istreambuf_iterator<char>{}};
    if (!stream.good() && !stream.eof())
        throw std::runtime_error("cannot_read_consensus_csv:" + path.string());

    std::vector<std::vector<std::string>> rows;
    std::vector<std::string> row;
    std::string field;
    bool quoted = false;
    bool closedQuote = false;
    bool fieldStarted = false;

    const auto finishField = [&]
    {
        row.push_back(std::move(field));
        field.clear();
        closedQuote = false;
        fieldStarted = false;
    };
    const auto finishRow = [&]
    {
        finishField();
        rows.push_back(std::move(row));
        row.clear();
    };

    for (std::size_t index = 0; index < input.size(); ++index)
    {
        const char character = input[index];
        if (quoted)
        {
            if (character == '"')
            {
                if (index + 1 < input.size() && input[index + 1] == '"')
                {
                    field.push_back('"');
                    ++index;
                }
                else
                {
                    quoted = false;
                    closedQuote = true;
                }
            }
            else
            {
                field.push_back(character);
            }
            continue;
        }

        if (closedQuote && character != ',' && character != '\r' &&
            character != '\n')
            throw std::invalid_argument("characters_after_closing_csv_quote");
        if (character == '"')
        {
            if (fieldStarted || !field.empty())
                throw std::invalid_argument("quote_inside_unquoted_csv_field");
            quoted = true;
            fieldStarted = true;
        }
        else if (character == ',')
        {
            finishField();
        }
        else if (character == '\n')
        {
            finishRow();
        }
        else if (character == '\r')
        {
            if (index + 1 < input.size() && input[index + 1] == '\n')
                ++index;
            finishRow();
        }
        else
        {
            field.push_back(character);
            fieldStarted = true;
        }
    }
    if (quoted)
        throw std::invalid_argument("unterminated_csv_quote");
    if (!row.empty() || !field.empty() || fieldStarted || closedQuote)
        finishRow();
    if (rows.empty())
        throw std::invalid_argument("empty_consensus_csv");
    return rows;
}

ConsensusParsedValue ParsedValue(
    const std::vector<std::string>& row,
    const std::unordered_map<std::string, std::size_t>& columns,
    std::string_view prefix)
{
    const auto value = [&](std::string_view suffix) -> const std::string&
    {
        return row.at(columns.at(std::string{prefix} + std::string{suffix}));
    };
    ConsensusParsedValue parsed;
    parsed.raw = OptionalText(value("_raw"));
    parsed.parseStatus = value("_parse_status");
    parsed.valueKind = OptionalText(value("_value_kind"));
    parsed.valueLow = OptionalText(value("_value_low"));
    parsed.valueHigh = OptionalText(value("_value_high"));
    parsed.canonicalValueLow = OptionalText(value("_canonical_value_low"));
    parsed.canonicalValueHigh = OptionalText(value("_canonical_value_high"));
    parsed.unit = OptionalText(value("_unit"));
    parsed.scale = OptionalText(value("_scale"));
    parsed.qualifier = OptionalText(value("_qualifier"));
    return parsed;
}

void ValidateParsedValue(
    const EconomicEventConsensusCandidate& candidate,
    const ConsensusParsedValue& value,
    std::string_view field)
{
    const auto fail = [&](std::string_view reason)
    {
        throw std::invalid_argument(
            "invalid_consensus_value:economic_event_id=" +
            std::to_string(candidate.economicEventId) + ":field=" +
            std::string{field} + ":reason=" + std::string{reason});
    };
    if (value.parseStatus == "missing")
    {
        if (value.raw || value.valueKind || value.valueLow || value.valueHigh ||
            value.canonicalValueLow || value.canonicalValueHigh || value.unit ||
            value.scale || value.qualifier)
            fail("missing_has_payload");
        return;
    }
    if (value.parseStatus != "parsed")
        fail("unsupported_parse_status");
    if (!value.raw || value.raw->empty() || !value.valueKind ||
        !value.valueLow || !value.canonicalValueLow || !value.unit ||
        !value.scale)
        fail("parsed_payload_incomplete");
    if (*value.valueKind != "scalar" && *value.valueKind != "range")
        fail("unsupported_value_kind");
    if ((*value.valueKind == "scalar" &&
         (value.valueHigh || value.canonicalValueHigh)) ||
        (*value.valueKind == "range" &&
         (!value.valueHigh || !value.canonicalValueHigh)))
        fail("endpoint_shape_mismatch");

    const Decimal low = ParseDecimal(*value.valueLow, field,
                                     candidate.economicEventId);
    const Decimal canonicalLow = ParseDecimal(
        *value.canonicalValueLow, field, candidate.economicEventId);
    const Decimal scale = ParseDecimal(*value.scale, field,
                                       candidate.economicEventId);
    if (scale.coefficient <= 0)
        fail("nonpositive_scale");
    if (!EqualDecimal(Multiply(low, scale), canonicalLow))
        fail("canonical_low_scale_mismatch");
    if (*value.valueKind == "range")
    {
        const Decimal high = ParseDecimal(*value.valueHigh, field,
                                          candidate.economicEventId);
        const Decimal canonicalHigh = ParseDecimal(
            *value.canonicalValueHigh, field, candidate.economicEventId);
        if (!EqualDecimal(Multiply(high, scale), canonicalHigh))
            fail("canonical_high_scale_mismatch");
    }

    const bool countFamily = candidate.eventFamily == "EMPLOYMENT" ||
        candidate.eventFamily == "JOLTS";
    if (countFamily)
    {
        if (*value.unit != "count" || value.qualifier)
            fail("count_unit_or_qualifier_mismatch");
    }
    else
    {
        if (*value.unit != "percent" || *value.scale != "1")
            fail("percent_unit_or_scale_mismatch");
        const bool monthlyFamily = candidate.eventFamily == "CPI" ||
            candidate.eventFamily == "DURABLE_GOODS" ||
            candidate.eventFamily == "PCE" || candidate.eventFamily == "PPI" ||
            candidate.eventFamily == "RETAIL_SALES";
        const std::optional<std::string> expectedQualifier =
            candidate.consensusSource == "OANDA" &&
                    candidate.eventFamily == "CPI" &&
                    candidate.sourceReportId == 698
                ? std::optional<std::string>{"y/y"}
                : monthlyFamily ? std::optional<std::string>{"m/m"}
                                : std::nullopt;
        if (value.qualifier != expectedQualifier)
            fail("percent_qualifier_mismatch");
    }
    if (candidate.eventFamily != "FOMC" && *value.valueKind != "scalar")
        fail("range_only_permitted_for_fomc");

    std::string suffix;
    if (*value.unit == "percent")
    {
        suffix = "%";
        if (value.qualifier)
            suffix += " " + *value.qualifier;
    }
    else if (EqualDecimal(scale, ParseDecimal(
                 "1000", field, candidate.economicEventId)))
    {
        suffix = " k";
    }
    else if (EqualDecimal(scale, ParseDecimal(
                      "1000000", field, candidate.economicEventId)))
    {
        suffix = " mn";
    }
    else if (!EqualDecimal(scale, ParseDecimal(
                           "1", field, candidate.economicEventId)))
    {
        fail("unsupported_count_scale");
    }

    std::string rawNumber;
    if (candidate.consensusSource == "MYFXBOOK")
    {
        rawNumber = *value.raw;
    }
    else if (*value.unit == "percent" && !value.qualifier)
    {
        rawNumber = *value.raw;
        while (!rawNumber.empty() && rawNumber.back() == ' ')
            rawNumber.pop_back();
        std::size_t percentCount = 0;
        while (!rawNumber.empty() && rawNumber.back() == '%')
        {
            rawNumber.pop_back();
            ++percentCount;
        }
        while (!rawNumber.empty() && rawNumber.back() == ' ')
            rawNumber.pop_back();
        if (percentCount == 0)
            fail("raw_suffix_mismatch");
    }
    else
    {
        if (value.raw->size() <= suffix.size() ||
            value.raw->substr(value.raw->size() - suffix.size()) != suffix)
            fail("raw_suffix_mismatch");
        rawNumber =
            value.raw->substr(0, value.raw->size() - suffix.size());
    }
    std::replace(rawNumber.begin(), rawNumber.end(), ',', '.');
    if (*value.valueKind == "scalar")
    {
        if (!EqualDecimal(ParseDecimal(rawNumber, field,
                                       candidate.economicEventId), low))
            fail("raw_scalar_mismatch");
    }
    else
    {
        const std::size_t separator = rawNumber.find('-', 1);
        if (separator == std::string::npos ||
            rawNumber.find('-', separator + 1) != std::string::npos)
            fail("raw_range_shape_mismatch");
        const Decimal rawLow = ParseDecimal(
            rawNumber.substr(0, separator), field, candidate.economicEventId);
        const Decimal rawHigh = ParseDecimal(
            rawNumber.substr(separator + 1), field,
            candidate.economicEventId);
        const Decimal high = ParseDecimal(*value.valueHigh, field,
                                          candidate.economicEventId);
        if (!EqualDecimal(rawLow, low) || !EqualDecimal(rawHigh, high))
            fail("raw_range_mismatch");
    }
}

void ValidateCandidate(const EconomicEventConsensusCandidate& candidate)
{
    static const std::map<std::string, std::int64_t> reports{
        {"CPI", 699}, {"DURABLE_GOODS", 59}, {"EMPLOYMENT", 707},
        {"FOMC", 82}, {"GDP", 690}, {"JOLTS", 1371}, {"PCE", 694},
        {"PPI", 703}, {"RETAIL_SALES", 696}};
    static const std::map<std::string, std::set<std::string>> names{
        {"CPI", {"Consumer Price Index"}},
        {"DURABLE_GOODS", {"Durable Goods Orders - pre.."}},
        {"EMPLOYMENT", {"Non-Farm Employment Change"}},
        {"FOMC", {"FOMC Interest Rate Decision"}},
        {"GDP", {"GDP (Annualized) - pre..", "GDP (Annualized) - rev..",
                 "GDP (Annualized) - fin.."}},
        {"JOLTS", {"JOLTS Job Openings"}},
        {"PCE", {"Personal Consumption Expenditures"}},
        {"PPI", {"Producer Price Index"}},
        {"RETAIL_SALES", {"Retail Sales"}}};

    if (candidate.economicEventId <= 0 ||
        (candidate.consensusSource != "OANDA" &&
         candidate.consensusSource != "MYFXBOOK") ||
        candidate.secondarySourceEventId == 0 ||
        candidate.sourceAgency.empty() || candidate.sourceEventId.empty() ||
        candidate.secondarySourceObservationId.empty() ||
        candidate.secondarySourceEventName.empty() ||
        candidate.secondarySourceArtifactPath.empty() ||
        candidate.candidateClassification.empty() ||
        candidate.matchRule.empty() ||
        candidate.semanticContract.empty() || candidate.providerProvenance.empty())
        throw std::invalid_argument(
            "invalid_consensus_identity:economic_event_id=" +
            std::to_string(candidate.economicEventId));

    if (candidate.consensusSource == "MYFXBOOK")
    {
        const bool gap = candidate.candidateClassification ==
                "myfxbook_jolts_gap_fill" &&
            candidate.eventFamily == "JOLTS";
        const bool blankFill = candidate.candidateClassification ==
                "myfxbook_oanda_blank_fill" &&
            (candidate.eventFamily == "CPI" ||
             candidate.eventFamily == "PPI" ||
             candidate.eventFamily == "RETAIL_SALES");
        if (candidate.sourceReportId || candidate.secondarySourcePeriod ||
            candidate.secondarySourcePriority ||
            candidate.secondarySourceTimestampEpoch ||
            candidate.secondarySourceDate ||
            !candidate.secondarySourceArtifactSha256 ||
            candidate.secondarySourceArtifactSha256->size() != 64 ||
            candidate.semanticContract !=
                "myfxbook_consensus_observation_v1" ||
            (!gap && !blankFill) ||
            candidate.forecast.parseStatus != "parsed" ||
            candidate.previous.parseStatus != "missing" ||
            candidate.actual.parseStatus != "missing")
            throw std::invalid_argument(
                "invalid_myfxbook_consensus_identity:economic_event_id=" +
                std::to_string(candidate.economicEventId));
        if (!ValidDate(candidate.sourceReleaseDate) ||
            candidate.eventTimestampUtc.size() < 20 ||
            candidate.eventTimestampUtc.substr(0, 10) !=
                candidate.sourceReleaseDate)
            throw std::invalid_argument(
                "invalid_consensus_date:economic_event_id=" +
                std::to_string(candidate.economicEventId));
        ValidateParsedValue(candidate, candidate.forecast, "forecast");
        ValidateParsedValue(candidate, candidate.previous, "previous");
        ValidateParsedValue(candidate, candidate.actual, "actual");
        return;
    }

    if (!candidate.sourceReportId || *candidate.sourceReportId <= 0 ||
        candidate.secondarySourceEventId <= 0 ||
        !candidate.secondarySourcePeriod ||
        candidate.secondarySourcePeriod->empty() ||
        !candidate.secondarySourcePriority ||
        *candidate.secondarySourcePriority < 1 ||
        *candidate.secondarySourcePriority > 3 ||
        !candidate.secondarySourceTimestampEpoch ||
        *candidate.secondarySourceTimestampEpoch <= 0 ||
        !candidate.secondarySourceDate ||
        candidate.candidateClassification !=
            (candidate.forecast.parseStatus == "parsed"
                ? "oanda_populated_initial" : "oanda_matched_blank") ||
        candidate.semanticContract != "oanda_economic_consensus_candidate_v1")
        throw std::invalid_argument(
            "invalid_oanda_consensus_identity:economic_event_id=" +
            std::to_string(candidate.economicEventId));
    const auto report = reports.find(candidate.eventFamily);
    if (report == reports.end())
        throw std::invalid_argument("unsupported_consensus_family:" +
                                    candidate.eventFamily);
    const bool cpi698 = candidate.eventFamily == "CPI" &&
        candidate.economicEventId == 779 &&
        candidate.sourceReleaseDate == "2025-12-18" &&
        candidate.sourceReportId == 698;
    if (candidate.sourceReportId != report->second && !cpi698)
        throw std::invalid_argument(
            "unapproved_report_mapping:economic_event_id=" +
            std::to_string(candidate.economicEventId));
    if (!names.at(candidate.eventFamily).contains(
            candidate.secondarySourceEventName))
        throw std::invalid_argument(
            "unapproved_event_name:economic_event_id=" +
            std::to_string(candidate.economicEventId));
    if (cpi698 && candidate.matchRule !=
            "pinned_report_698_single_release_exception")
        throw std::invalid_argument("cpi_698_match_rule_mismatch");
    if (candidate.sourceReportId == 698 && !cpi698)
        throw std::invalid_argument("unapproved_cpi_698_exception");

    if (!ValidDate(candidate.sourceReleaseDate) ||
        candidate.eventTimestampUtc.size() < 20 ||
        candidate.eventTimestampUtc.substr(0, 10) !=
            candidate.sourceReleaseDate ||
        candidate.eventTimestampUtc[10] != 'T' ||
        candidate.secondarySourceDate->size() != 16 ||
        !ValidDate(candidate.secondarySourceDate->substr(0, 10)) ||
        (*candidate.secondarySourceDate)[10] != ' ' ||
        (*candidate.secondarySourceDate)[13] != ':')
        throw std::invalid_argument(
            "invalid_consensus_date:economic_event_id=" +
            std::to_string(candidate.economicEventId));

    if (candidate.economicEventId == 1979)
    {
        if (candidate.eventFamily != "PCE" ||
            candidate.sourceReleaseDate != "2026-01-22" ||
            candidate.referencePeriod !=
                std::optional<std::string>{"October and November 2025"} ||
            candidate.secondarySourcePeriod !=
                std::optional<std::string>{"November"} ||
            candidate.sourceReportId != 694 ||
            candidate.matchRule != "authoritative_reference_period")
            throw std::invalid_argument("invalid_known_multimonth_pce_mapping");
    }

    ValidateParsedValue(candidate, candidate.forecast, "forecast");
    ValidateParsedValue(candidate, candidate.previous, "previous");
    ValidateParsedValue(candidate, candidate.actual, "actual");
}

std::vector<EconomicEventConsensusCandidate> ValidateAndOrder(
    std::vector<EconomicEventConsensusCandidate> candidates)
{
    std::set<std::int64_t> economicEventIds;
    std::map<std::pair<std::string, std::string>, std::int64_t> sourceEvents;
    for (const auto& candidate : candidates)
    {
        ValidateCandidate(candidate);
        if (!economicEventIds.insert(candidate.economicEventId).second)
            throw std::invalid_argument(
                "duplicate_economic_event_id:" +
                std::to_string(candidate.economicEventId));
        const auto identity = std::make_pair(
            candidate.consensusSource,
            candidate.secondarySourceObservationId);
        const auto [iterator, inserted] = sourceEvents.emplace(
            identity, candidate.economicEventId);
        if (!inserted && iterator->second != candidate.economicEventId)
            throw std::invalid_argument(
                "secondary_source_observation_maps_to_multiple_official_events:" +
                candidate.secondarySourceObservationId);
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& left, const auto& right)
              {
                  return left.economicEventId < right.economicEventId;
              });
    return candidates;
}

struct CsvTable
{
    std::unordered_map<std::string, std::size_t> columns;
    std::vector<std::vector<std::string>> rows;
};

CsvTable LoadCsvTable(
    const std::filesystem::path& path,
    const std::set<std::string>& required)
{
    auto parsed = ParseCsv(path);
    CsvTable table;
    for (std::size_t index = 0; index < parsed.front().size(); ++index)
    {
        if (!table.columns.emplace(parsed.front()[index], index).second)
            throw std::invalid_argument(
                "duplicate_consensus_csv_column:" + parsed.front()[index]);
    }
    for (const auto& column : required)
    {
        if (!table.columns.contains(column))
            throw std::invalid_argument(
                "missing_consensus_csv_column:" + column);
    }
    table.rows.assign(
        std::make_move_iterator(parsed.begin() + 1),
        std::make_move_iterator(parsed.end()));
    for (std::size_t index = 0; index < table.rows.size(); ++index)
    {
        if (table.rows[index].size() != table.columns.size())
            throw std::invalid_argument(
                "consensus_csv_column_count_mismatch:row=" +
                std::to_string(index + 2));
    }
    return table;
}

const std::string& CsvValue(
    const CsvTable& table,
    const std::vector<std::string>& row,
    std::string_view column)
{
    return row.at(table.columns.at(std::string{column}));
}

std::string JsonEscape(std::string_view value)
{
    std::string escaped;
    for (const char character : value)
    {
        switch (character)
        {
            case '\\': escaped += "\\\\"; break;
            case '"': escaped += "\\\""; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default: escaped.push_back(character); break;
        }
    }
    return escaped;
}

struct MyfxbookEvidence
{
    std::string eventFamily;
    std::string sourceFamily;
    std::int64_t eventId = 0;
    std::string releaseDate;
    std::string actual;
    std::string consensus;
    std::string classification;
    std::string eligible;
    std::string capturePath;
    std::string captureSha256;
    int seriesOrdinal = 0;
    int observationOrdinal = 0;
};

using EvidenceKey = std::pair<std::string, std::string>;

std::map<EvidenceKey, std::vector<MyfxbookEvidence>> LoadMyfxbookEvidence(
    const std::filesystem::path& path,
    std::size_t& sourceExclusions)
{
    const CsvTable table = LoadCsvTable(path, {
        "event_family", "myfxbook_source_family", "myfxbook_event_id",
        "release_date", "myfxbook_actual", "myfxbook_consensus",
        "classification", "automatic_candidate_eligible",
        "source_capture_filename", "source_capture_sha256",
        "source_series_ordinal", "source_observation_ordinal"});
    std::map<EvidenceKey, std::vector<MyfxbookEvidence>> evidence;
    for (std::size_t index = 0; index < table.rows.size(); ++index)
    {
        const auto& row = table.rows[index];
        MyfxbookEvidence item;
        item.eventFamily = CsvValue(table, row, "event_family");
        item.sourceFamily = CsvValue(table, row, "myfxbook_source_family");
        item.eventId = ParseInteger<std::int64_t>(
            CsvValue(table, row, "myfxbook_event_id"),
            "myfxbook_event_id", index + 2);
        item.releaseDate = CsvValue(table, row, "release_date");
        item.actual = CsvValue(table, row, "myfxbook_actual");
        item.consensus = CsvValue(table, row, "myfxbook_consensus");
        item.classification = CsvValue(table, row, "classification");
        item.eligible = CsvValue(table, row, "automatic_candidate_eligible");
        item.capturePath = CsvValue(table, row, "source_capture_filename");
        item.captureSha256 = CsvValue(table, row, "source_capture_sha256");
        item.seriesOrdinal = ParseInteger<int>(
            CsvValue(table, row, "source_series_ordinal"),
            "source_series_ordinal", index + 2);
        item.observationOrdinal = ParseInteger<int>(
            CsvValue(table, row, "source_observation_ordinal"),
            "source_observation_ordinal", index + 2);
        if (!ValidDate(item.releaseDate) || item.capturePath.empty() ||
            item.captureSha256.size() != 64)
            throw std::invalid_argument(
                "invalid_myfxbook_normalized_evidence:row=" +
                std::to_string(index + 2));
        const bool eligible = item.classification == "unique_populated" &&
            item.eligible == "1" && !item.consensus.empty();
        const bool excluded = item.classification == "ambiguous_duplicate" ||
            item.classification == "unique_null" ||
            item.classification == "manual_review";
        if (!eligible && !excluded)
            throw std::invalid_argument(
                "invalid_myfxbook_classification:row=" +
                std::to_string(index + 2));
        if (excluded)
            ++sourceExclusions;
        evidence[{item.eventFamily, item.releaseDate}].push_back(
            std::move(item));
    }
    return evidence;
}

const MyfxbookEvidence& UniqueEligibleEvidence(
    const std::map<EvidenceKey, std::vector<MyfxbookEvidence>>& evidence,
    const std::string& family,
    const std::string& releaseDate)
{
    const auto iterator = evidence.find({family, releaseDate});
    if (iterator == evidence.end() || iterator->second.size() != 1 ||
        iterator->second.front().classification != "unique_populated" ||
        iterator->second.front().eligible != "1")
        throw std::invalid_argument(
            "myfxbook_candidate_not_unique_populated:" + family + ":" +
            releaseDate);
    return iterator->second.front();
}

struct CanonicalMatch
{
    std::int64_t economicEventId = 0;
    std::string eventFamily;
    std::string eventTimestampUtc;
    std::string sourceAgency;
    std::string sourceEventId;
    std::optional<std::string> referencePeriod;
    std::string sourceReleaseDate;
};

std::vector<CanonicalMatch> LoadCanonicalMatches(
    pqxx::transaction_base& transaction,
    const std::string& family,
    const std::string& releaseDate,
    const std::optional<std::string>& sourceAgency = std::nullopt,
    const std::optional<std::string>& timestamp = std::nullopt,
    const std::optional<std::string>& referencePeriod = std::nullopt)
{
    const pqxx::result rows = transaction.exec(
        "SELECT economic_event_id, event_family, "
        "to_char(event_timestamp_utc AT TIME ZONE 'UTC', "
        "'YYYY-MM-DD\"T\"HH24:MI:SS.US\"Z\"') AS timestamp_utc, "
        "source_agency, source_event_id, reference_period, "
        "source_release_date::text AS release_date "
        "FROM economic_event WHERE event_family = $1 "
        "AND source_release_date = $2::date "
        "AND ($3::text IS NULL OR source_agency = $3) "
        "AND ($4::timestamptz IS NULL OR event_timestamp_utc = $4) "
        "AND ($5::text IS NULL OR reference_period IS NOT DISTINCT FROM $5) "
        "ORDER BY economic_event_id;",
        pqxx::params{
            family, releaseDate, sourceAgency, timestamp, referencePeriod});
    std::vector<CanonicalMatch> matches;
    for (const pqxx::row& row : rows)
    {
        CanonicalMatch match;
        match.economicEventId = row["economic_event_id"].as<std::int64_t>();
        match.eventFamily = row["event_family"].as<std::string>();
        match.eventTimestampUtc = row["timestamp_utc"].as<std::string>();
        match.sourceAgency = row["source_agency"].as<std::string>();
        match.sourceEventId = row["source_event_id"].as<std::string>();
        if (!row["reference_period"].is_null())
            match.referencePeriod =
                row["reference_period"].as<std::string>();
        match.sourceReleaseDate = row["release_date"].as<std::string>();
        matches.push_back(std::move(match));
    }
    return matches;
}

EconomicEventConsensusCandidate MakeMyfxbookCandidate(
    const MyfxbookEvidence& evidence,
    const CanonicalMatch& canonical,
    const std::string& classification,
    const std::string& matchRule,
    const std::filesystem::path& derivedArtifact,
    const std::string& extraProvenance)
{
    EconomicEventConsensusCandidate candidate;
    candidate.economicEventId = canonical.economicEventId;
    candidate.eventFamily = canonical.eventFamily;
    candidate.eventTimestampUtc = canonical.eventTimestampUtc;
    candidate.sourceAgency = canonical.sourceAgency;
    candidate.sourceEventId = canonical.sourceEventId;
    candidate.referencePeriod = canonical.referencePeriod;
    candidate.sourceReleaseDate = canonical.sourceReleaseDate;
    candidate.consensusSource = "MYFXBOOK";
    candidate.secondarySourceEventId = evidence.eventId;
    candidate.secondarySourceObservationId =
        "myfxbook:event:" + std::to_string(evidence.eventId) + ":date:" +
        evidence.releaseDate + ":series:" +
        std::to_string(evidence.seriesOrdinal) + ":observation:" +
        std::to_string(evidence.observationOrdinal);
    candidate.secondarySourceEventName = evidence.sourceFamily;
    candidate.secondarySourceArtifactPath = evidence.capturePath;
    candidate.secondarySourceArtifactSha256 = evidence.captureSha256;
    candidate.candidateClassification = classification;
    candidate.matchRule = matchRule;
    candidate.semanticContract = "myfxbook_consensus_observation_v1";
    candidate.providerProvenance =
        "{\"provider\":\"MYFXBOOK\",\"myfxbook_event_id\":" +
        std::to_string(evidence.eventId) +
        ",\"source_family\":\"" + JsonEscape(evidence.sourceFamily) +
        "\",\"source_series_ordinal\":" +
        std::to_string(evidence.seriesOrdinal) +
        ",\"source_observation_ordinal\":" +
        std::to_string(evidence.observationOrdinal) +
        ",\"myfxbook_actual\":\"" + JsonEscape(evidence.actual) +
        "\",\"normalized_artifact\":\"" +
        JsonEscape(derivedArtifact.string()) + "\"" + extraProvenance + "}";
    candidate.forecast.raw = evidence.consensus;
    candidate.forecast.parseStatus = "parsed";
    candidate.forecast.valueKind = "scalar";
    candidate.forecast.valueLow = evidence.consensus;
    candidate.forecast.canonicalValueLow = evidence.consensus;
    candidate.forecast.unit =
        evidence.eventFamily == "JOLTS" ? "count" : "percent";
    candidate.forecast.scale = "1";
    if (evidence.eventFamily == "CPI" || evidence.eventFamily == "PPI" ||
        evidence.eventFamily == "RETAIL_SALES")
        candidate.forecast.qualifier = "m/m";
    candidate.previous.parseStatus = "missing";
    candidate.actual.parseStatus = "missing";
    return candidate;
}

struct CliArguments
{
    std::filesystem::path input;
    std::filesystem::path evidenceRoot;
    EconomicEventConsensusImportMode mode =
        EconomicEventConsensusImportMode::dryRun;
    bool modeSpecified = false;
};

CliArguments ParseCli(int argc, const char* const argv[])
{
    CliArguments parsed;
    bool commandSeen = false;
    for (int index = 1; index < argc; ++index)
    {
        const std::string argument{argv[index]};
        if (argument == "--import-economic-consensus")
        {
            if (commandSeen)
                throw std::invalid_argument(
                    "--import-economic-consensus specified more than once");
            commandSeen = true;
        }
        else if (argument == "--input")
        {
            if (!parsed.input.empty() || ++index >= argc)
                throw std::invalid_argument("--input requires one path");
            parsed.input = argv[index];
        }
        else if (argument.rfind("--input=", 0) == 0)
        {
            if (!parsed.input.empty())
                throw std::invalid_argument("--input specified more than once");
            parsed.input = argument.substr(8);
        }
        else if (argument == "--evidence-root")
        {
            if (!parsed.evidenceRoot.empty() || ++index >= argc)
                throw std::invalid_argument(
                    "--evidence-root requires one path");
            parsed.evidenceRoot = argv[index];
        }
        else if (argument.rfind("--evidence-root=", 0) == 0)
        {
            if (!parsed.evidenceRoot.empty())
                throw std::invalid_argument(
                    "--evidence-root specified more than once");
            parsed.evidenceRoot = argument.substr(16);
        }
        else if (argument == "--dry-run" || argument == "--apply")
        {
            if (parsed.modeSpecified)
                throw std::invalid_argument(
                    "exactly one of --dry-run or --apply is required");
            parsed.mode = argument == "--apply"
                ? EconomicEventConsensusImportMode::apply
                : EconomicEventConsensusImportMode::dryRun;
            parsed.modeSpecified = true;
        }
        else
        {
            throw std::invalid_argument(
                "unsupported economic-consensus import option: " + argument);
        }
    }
    if (!commandSeen)
        throw std::invalid_argument("--import-economic-consensus is required");
    if (parsed.input.empty() == parsed.evidenceRoot.empty())
        throw std::invalid_argument(
            "exactly one of --input or --evidence-root is required");
    if (!parsed.modeSpecified)
        throw std::invalid_argument(
            "exactly one of --dry-run or --apply is required");
    return parsed;
}

} // namespace


std::vector<EconomicEventConsensusCandidate>
LoadAndValidateOandaEconomicConsensusCsv(const std::filesystem::path& path)
{
    const auto rows = ParseCsv(path);
    std::unordered_map<std::string, std::size_t> columns;
    for (std::size_t index = 0; index < rows.front().size(); ++index)
    {
        if (!columns.emplace(rows.front()[index], index).second)
            throw std::invalid_argument("duplicate_consensus_csv_column:" +
                                        rows.front()[index]);
    }
    for (const std::string_view required : kRequiredColumns)
    {
        if (!columns.contains(std::string{required}))
            throw std::invalid_argument("missing_consensus_csv_column:" +
                                        std::string{required});
    }

    std::vector<EconomicEventConsensusCandidate> candidates;
    candidates.reserve(rows.size() - 1);
    const auto value = [&](const std::vector<std::string>& row,
                           std::string_view column) -> const std::string&
    {
        return row.at(columns.at(std::string{column}));
    };
    for (std::size_t index = 1; index < rows.size(); ++index)
    {
        const auto& row = rows[index];
        if (row.size() != rows.front().size())
            throw std::invalid_argument(
                "consensus_csv_column_count_mismatch:row=" +
                std::to_string(index + 1));
        EconomicEventConsensusCandidate candidate;
        candidate.economicEventId = ParseInteger<std::int64_t>(
            value(row, "economic_event_id"), "economic_event_id", index + 1);
        candidate.eventFamily = value(row, "event_family");
        candidate.eventTimestampUtc = value(row, "event_timestamp_utc");
        candidate.sourceAgency = value(row, "source_agency");
        candidate.sourceEventId = value(row, "source_event_id");
        candidate.referencePeriod = OptionalText(value(row, "reference_period"));
        candidate.sourceReleaseDate = value(row, "source_release_date");
        candidate.consensusSource = value(row, "consensus_source");
        candidate.sourceReportId = ParseInteger<std::int64_t>(
            value(row, "oanda_report_id"), "oanda_report_id", index + 1);
        candidate.secondarySourceEventId = ParseInteger<std::int64_t>(
            value(row, "oanda_event_id"), "oanda_event_id", index + 1);
        candidate.secondarySourceObservationId = "oanda:event:" +
            std::to_string(candidate.secondarySourceEventId);
        candidate.secondarySourceEventName = value(row, "oanda_event");
        candidate.secondarySourcePeriod = value(row, "oanda_period");
        candidate.secondarySourcePriority = ParseInteger<int>(
            value(row, "oanda_priority"), "oanda_priority", index + 1);
        candidate.secondarySourceTimestampEpoch = ParseInteger<std::int64_t>(
            value(row, "oanda_timestamp"), "oanda_timestamp", index + 1);
        candidate.secondarySourceDate = value(row, "oanda_date");
        candidate.secondarySourceArtifactPath = value(row, "oanda_source_file");
        candidate.matchRule = value(row, "match_rule");
        candidate.forecast = ParsedValue(row, columns, "forecast");
        candidate.previous = ParsedValue(row, columns, "previous");
        candidate.actual = ParsedValue(row, columns, "actual");
        candidate.candidateClassification =
            candidate.forecast.parseStatus == "parsed"
                ? "oanda_populated_initial" : "oanda_matched_blank";
        candidate.providerProvenance =
            "{\"provider\":\"OANDA\",\"oanda_report_id\":" +
            std::to_string(*candidate.sourceReportId) +
            ",\"oanda_event_id\":" +
            std::to_string(candidate.secondarySourceEventId) +
            ",\"source_artifact\":\"" +
            JsonEscape(candidate.secondarySourceArtifactPath) +
            "\",\"phase\":\"phase_1_initial_population\"}";
        candidates.push_back(std::move(candidate));
    }
    if (candidates.empty())
        throw std::invalid_argument("consensus_csv_has_no_candidates");
    return ValidateAndOrder(std::move(candidates));
}


EconomicEventConsensusWorkflowReport
RunAuthoritativeEconomicEventConsensusWorkflow(
    pqxx::connection& connection,
    const EconomicEventConsensusEvidencePaths& paths,
    EconomicEventConsensusImportMode mode)
{
    EconomicEventConsensusWorkflowReport report;
    auto oanda = LoadAndValidateOandaEconomicConsensusCsv(
        paths.oandaCandidates);
    std::vector<EconomicEventConsensusCandidate> candidates;
    candidates.reserve(oanda.size() + 116);
    for (auto& candidate : oanda)
    {
        if (candidate.forecast.parseStatus == "parsed")
        {
            candidates.push_back(std::move(candidate));
            ++report.oandaCandidates;
        }
        else
        {
            ++report.oandaMatchedBlankEvidence;
            ++report.sourceExclusions;
        }
    }

    const auto evidence = LoadMyfxbookEvidence(
        paths.myfxbookNormalized, report.sourceExclusions);

    pqxx::read_transaction transaction{connection};
    report.canonicalEventsExamined =
        transaction.query_value<std::size_t>(
            "SELECT count(*) FROM economic_event;");

    const CsvTable gap = LoadCsvTable(paths.myfxbookGapCandidates, {
        "event_family", "release_date", "myfxbook_source_family",
        "myfxbook_event_id", "myfxbook_actual", "myfxbook_consensus",
        "consensus_source", "candidate_classification",
        "target_coverage_source", "myfxbook_capture_filename",
        "myfxbook_capture_sha256", "source_series_ordinal",
        "source_observation_ordinal"});
    for (std::size_t index = 0; index < gap.rows.size(); ++index)
    {
        const auto& row = gap.rows[index];
        const std::string& family = CsvValue(gap, row, "event_family");
        const std::string& releaseDate = CsvValue(gap, row, "release_date");
        const auto& source = UniqueEligibleEvidence(
            evidence, family, releaseDate);
        if (family != "JOLTS" ||
            CsvValue(gap, row, "candidate_classification") !=
                "myfxbook_unique_populated_gap_fill" ||
            CsvValue(gap, row, "consensus_source") != "myfxbook" ||
            CsvValue(gap, row, "target_coverage_source") != "oanda" ||
            CsvValue(gap, row, "myfxbook_source_family") !=
                source.sourceFamily ||
            CsvValue(gap, row, "myfxbook_event_id") !=
                std::to_string(source.eventId) ||
            CsvValue(gap, row, "myfxbook_actual") != source.actual ||
            CsvValue(gap, row, "myfxbook_consensus") != source.consensus ||
            CsvValue(gap, row, "myfxbook_capture_filename") !=
                source.capturePath ||
            CsvValue(gap, row, "myfxbook_capture_sha256") !=
                source.captureSha256 ||
            ParseInteger<int>(CsvValue(gap, row, "source_series_ordinal"),
                "source_series_ordinal", index + 2) != source.seriesOrdinal ||
            ParseInteger<int>(
                CsvValue(gap, row, "source_observation_ordinal"),
                "source_observation_ordinal", index + 2) !=
                source.observationOrdinal)
            throw std::invalid_argument(
                "invalid_myfxbook_gap_candidate:row=" +
                std::to_string(index + 2));
        ++report.myfxbookJoltsGapCandidates;
        const auto matches = LoadCanonicalMatches(
            transaction, family, releaseDate);
        if (matches.empty())
        {
            ++report.missingCanonicalMatches;
            continue;
        }
        if (matches.size() != 1)
        {
            ++report.ambiguousCanonicalMatches;
            continue;
        }
        candidates.push_back(MakeMyfxbookCandidate(
            source, matches.front(), "myfxbook_jolts_gap_fill",
            "canonical_family_release_date_unique",
            paths.myfxbookNormalized,
            ",\"candidate_artifact\":\"" +
                JsonEscape(paths.myfxbookGapCandidates.string()) +
                "\",\"merge_rule\":\"oanda_coverage_gap\""));
    }

    const std::map<EvidenceKey, std::string> expectedBlankFills{
        {{"PPI", "2013-12-13"}, "-0.1"},
        {{"RETAIL_SALES", "2022-09-15"}, "0.0"},
        {{"CPI", "2023-12-12"}, "0.0"}};
    const CsvTable blank = LoadCsvTable(
        paths.myfxbookBlankReconciliation, {
            "event_family", "official_event_timestamp_utc",
            "official_release_date", "official_source_agency",
            "official_reference_period", "oanda_report_id",
            "oanda_event_id", "original_oanda_forecast_present",
            "oanda_source_file", "oanda_matches_filename",
            "oanda_matches_sha256", "myfxbook_capture_filename",
            "myfxbook_capture_sha256", "candidate_myfxbook_event_id",
            "candidate_myfxbook_actual", "candidate_myfxbook_consensus",
            "candidate_consensus_source", "merge_classification",
            "merge_exclusion_reason"});
    std::set<EvidenceKey> observedBlankFills;
    for (std::size_t index = 0; index < blank.rows.size(); ++index)
    {
        const auto& row = blank.rows[index];
        if (CsvValue(blank, row, "merge_classification") !=
            "myfxbook_populated_candidate")
            continue;
        const std::string& family = CsvValue(blank, row, "event_family");
        const std::string& releaseDate =
            CsvValue(blank, row, "official_release_date");
        const EvidenceKey key{family, releaseDate};
        const auto expected = expectedBlankFills.find(key);
        if (expected == expectedBlankFills.end() ||
            !observedBlankFills.insert(key).second ||
            CsvValue(blank, row, "candidate_myfxbook_consensus") !=
                expected->second ||
            CsvValue(blank, row, "original_oanda_forecast_present") != "0" ||
            CsvValue(blank, row, "candidate_consensus_source") != "myfxbook" ||
            !CsvValue(blank, row, "merge_exclusion_reason").empty())
            throw std::invalid_argument(
                "unapproved_myfxbook_oanda_blank_fill:row=" +
                std::to_string(index + 2));
        const auto& source = UniqueEligibleEvidence(
            evidence, family, releaseDate);
        if (CsvValue(blank, row, "candidate_myfxbook_event_id") !=
                std::to_string(source.eventId) ||
            CsvValue(blank, row, "candidate_myfxbook_actual") !=
                source.actual ||
            CsvValue(blank, row, "candidate_myfxbook_consensus") !=
                source.consensus ||
            CsvValue(blank, row, "myfxbook_capture_filename") !=
                source.capturePath ||
            CsvValue(blank, row, "myfxbook_capture_sha256") !=
                source.captureSha256 ||
            CsvValue(blank, row, "oanda_matches_sha256").size() != 64)
            throw std::invalid_argument(
                "myfxbook_blank_fill_normalized_evidence_mismatch:row=" +
                std::to_string(index + 2));
        ++report.myfxbookOandaBlankCandidates;
        const auto matches = LoadCanonicalMatches(
            transaction, family, releaseDate,
            CsvValue(blank, row, "official_source_agency"),
            CsvValue(blank, row, "official_event_timestamp_utc"),
            CsvValue(blank, row, "official_reference_period"));
        if (matches.empty())
        {
            ++report.missingCanonicalMatches;
            continue;
        }
        if (matches.size() != 1)
        {
            ++report.ambiguousCanonicalMatches;
            continue;
        }
        candidates.push_back(MakeMyfxbookCandidate(
            source, matches.front(), "myfxbook_oanda_blank_fill",
            "oanda_blank_official_identity_unique",
            paths.myfxbookNormalized,
            ",\"reconciliation_artifact\":\"" +
                JsonEscape(paths.myfxbookBlankReconciliation.string()) +
                "\",\"oanda_matches_artifact\":\"" +
                JsonEscape(CsvValue(blank, row, "oanda_matches_filename")) +
                "\",\"oanda_matches_sha256\":\"" +
                JsonEscape(CsvValue(blank, row, "oanda_matches_sha256")) +
                "\",\"oanda_report_id\":" +
                CsvValue(blank, row, "oanda_report_id") +
                ",\"oanda_event_id\":" +
                CsvValue(blank, row, "oanda_event_id") +
                ",\"oanda_source_artifact\":\"" +
                JsonEscape(CsvValue(blank, row, "oanda_source_file")) +
                "\",\"merge_rule\":\"oanda_matched_blank_fill\""));
    }
    if (observedBlankFills.size() != expectedBlankFills.size())
        throw std::invalid_argument(
            "missing_verified_myfxbook_oanda_blank_fill");
    transaction.commit();

    report.matchedCanonicalEvents = candidates.size();
    const bool matchingFailure = report.missingCanonicalMatches != 0 ||
        report.ambiguousCanonicalMatches != 0;
    report.persistence = RunEconomicEventConsensusImport(
        connection, candidates,
        matchingFailure ? EconomicEventConsensusImportMode::dryRun : mode);
    for (const auto& item : report.persistence.items)
    {
        if (item.diagnostic == "economic_event_id_not_found" ||
            item.diagnostic == "authoritative_identity_conflict")
        {
            ++report.missingCanonicalMatches;
            --report.matchedCanonicalEvents;
        }
    }
    if (matchingFailure && mode == EconomicEventConsensusImportMode::apply)
    {
        for (auto& item : report.persistence.items)
        {
            if (item.disposition ==
                EconomicEventConsensusImportDisposition::inserted)
            {
                item.disposition =
                    EconomicEventConsensusImportDisposition::rejected;
                item.diagnostic = "batch_not_applied_due_to_source_matching_error";
                --report.persistence.inserted;
                ++report.persistence.rejected;
            }
        }
    }
    return report;
}


EconomicEventConsensusImportReport RunEconomicEventConsensusImport(
    pqxx::connection& connection,
    const std::vector<EconomicEventConsensusCandidate>& candidates,
    EconomicEventConsensusImportMode mode)
{
    const auto ordered = ValidateAndOrder(candidates);
    if (mode == EconomicEventConsensusImportMode::dryRun)
    {
        pqxx::read_transaction transaction{connection};
        return CompareEconomicEventConsensusBatch(transaction, ordered);
    }
    return ApplyEconomicEventConsensusBatch(connection, ordered);
}


const char* EconomicEventConsensusImportDispositionName(
    EconomicEventConsensusImportDisposition disposition)
{
    switch (disposition)
    {
        case EconomicEventConsensusImportDisposition::inserted:
            return "inserted";
        case EconomicEventConsensusImportDisposition::unchanged:
            return "unchanged";
        case EconomicEventConsensusImportDisposition::rejected:
            return "rejected";
    }
    return "unknown";
}


bool IsEconomicEventConsensusImportCommand(
    int argc,
    const char* const argv[])
{
    for (int index = 1; index < argc; ++index)
    {
        if (std::string_view{argv[index]} == "--import-economic-consensus")
            return true;
    }
    return false;
}


int RunEconomicEventConsensusImportCli(
    int argc,
    const char* const argv[])
{
    try
    {
        const CliArguments arguments = ParseCli(argc, argv);
        pqxx::connection connection{ConnectionString()};
        if (!arguments.evidenceRoot.empty())
        {
            const EconomicEventConsensusEvidencePaths paths{
                arguments.evidenceRoot /
                    "oanda/oanda_consensus_enrichment_candidates.csv",
                arguments.evidenceRoot /
                    "myfxbook/derived/myfxbook_consensus_normalized.csv",
                arguments.evidenceRoot /
                    "myfxbook/derived/myfxbook_oanda_gap_fill_candidates.csv",
                arguments.evidenceRoot /
                    "myfxbook/derived/myfxbook_oanda_blank_reconciliation.csv"};
            const auto workflow =
                RunAuthoritativeEconomicEventConsensusWorkflow(
                    connection, paths, arguments.mode);
            std::cout << "ECONOMIC_EVENT_CONSENSUS_IMPORT_SUMMARY"
                      << ",source=OANDA+MYFXBOOK"
                      << ",mode="
                      << (arguments.mode ==
                                  EconomicEventConsensusImportMode::dryRun
                              ? "dry-run" : "apply")
                      << ",canonical_events_examined="
                      << workflow.canonicalEventsExamined
                      << ",oanda_candidates=" << workflow.oandaCandidates
                      << ",oanda_matched_blanks="
                      << workflow.oandaMatchedBlankEvidence
                      << ",myfxbook_jolts_gap_candidates="
                      << workflow.myfxbookJoltsGapCandidates
                      << ",myfxbook_oanda_blank_candidates="
                      << workflow.myfxbookOandaBlankCandidates
                      << ",matched_canonical_events="
                      << workflow.matchedCanonicalEvents
                      << ",missing_canonical_matches="
                      << workflow.missingCanonicalMatches
                      << ",ambiguous_canonical_matches="
                      << workflow.ambiguousCanonicalMatches
                      << ",source_exclusions=" << workflow.sourceExclusions
                      << ",inserted=" << workflow.persistence.inserted
                      << ",unchanged=" << workflow.persistence.unchanged
                      << ",rejected=" << workflow.persistence.rejected
                      << '\n';
            for (const auto& item : workflow.persistence.items)
            {
                if (item.disposition ==
                    EconomicEventConsensusImportDisposition::rejected)
                    std::cout << "ECONOMIC_EVENT_CONSENSUS_IMPORT_ITEM"
                              << ",economic_event_id="
                              << item.economicEventId
                              << ",disposition=rejected,diagnostic="
                              << item.diagnostic << '\n';
            }
            return workflow.persistence.rejected == 0 &&
                    workflow.missingCanonicalMatches == 0 &&
                    workflow.ambiguousCanonicalMatches == 0
                ? 0 : 2;
        }

        const auto candidates =
            LoadAndValidateOandaEconomicConsensusCsv(arguments.input);
        const auto report = RunEconomicEventConsensusImport(
            connection, candidates, arguments.mode);
        std::cout << "ECONOMIC_EVENT_CONSENSUS_IMPORT_SUMMARY"
                  << ",source=OANDA"
                  << ",mode="
                  << (arguments.mode == EconomicEventConsensusImportMode::dryRun
                          ? "dry-run" : "apply")
                  << ",input=" << candidates.size()
                  << ",inserted=" << report.inserted
                  << ",unchanged=" << report.unchanged
                  << ",rejected=" << report.rejected << '\n';
        for (const auto& item : report.items)
        {
            if (item.disposition ==
                    EconomicEventConsensusImportDisposition::rejected)
            {
                std::cout << "ECONOMIC_EVENT_CONSENSUS_IMPORT_ITEM"
                          << ",economic_event_id=" << item.economicEventId
                          << ",disposition=rejected"
                          << ",diagnostic=" << item.diagnostic << '\n';
            }
        }
        return report.rejected == 0 ? 0 : 2;
    }
    catch (const std::exception& error)
    {
        std::cerr << "ECONOMIC_EVENT_CONSENSUS_IMPORT_FAILED,error="
                  << error.what() << '\n';
        return 1;
    }
}

} // namespace EA::EconomicCalendar
