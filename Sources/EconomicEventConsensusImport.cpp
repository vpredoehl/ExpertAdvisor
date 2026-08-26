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
            candidate.eventFamily == "CPI" && candidate.sourceReportId == 698
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
    if (*value.unit == "percent" && !value.qualifier)
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

    if (candidate.economicEventId <= 0 || candidate.consensusSource != "OANDA" ||
        candidate.sourceReportId <= 0 || candidate.secondarySourceEventId <= 0 ||
        candidate.secondarySourcePriority < 1 ||
        candidate.secondarySourcePriority > 3 ||
        candidate.secondarySourceTimestampEpoch <= 0 ||
        candidate.sourceAgency.empty() || candidate.sourceEventId.empty() ||
        candidate.secondarySourcePeriod.empty() ||
        candidate.secondarySourceArtifactPath.empty() ||
        candidate.matchRule.empty() ||
        candidate.semanticContract != "oanda_economic_consensus_candidate_v1")
        throw std::invalid_argument(
            "invalid_consensus_identity:economic_event_id=" +
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
        candidate.secondarySourceDate.size() != 16 ||
        !ValidDate(candidate.secondarySourceDate.substr(0, 10)) ||
        candidate.secondarySourceDate[10] != ' ' ||
        candidate.secondarySourceDate[13] != ':')
        throw std::invalid_argument(
            "invalid_consensus_date:economic_event_id=" +
            std::to_string(candidate.economicEventId));

    if (candidate.economicEventId == 1979)
    {
        if (candidate.eventFamily != "PCE" ||
            candidate.sourceReleaseDate != "2026-01-22" ||
            candidate.referencePeriod !=
                std::optional<std::string>{"October and November 2025"} ||
            candidate.secondarySourcePeriod != "November" ||
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
    std::map<std::pair<std::string, std::int64_t>, std::int64_t> sourceEvents;
    for (const auto& candidate : candidates)
    {
        ValidateCandidate(candidate);
        if (!economicEventIds.insert(candidate.economicEventId).second)
            throw std::invalid_argument(
                "duplicate_economic_event_id:" +
                std::to_string(candidate.economicEventId));
        const auto identity = std::make_pair(
            candidate.consensusSource, candidate.secondarySourceEventId);
        const auto [iterator, inserted] = sourceEvents.emplace(
            identity, candidate.economicEventId);
        if (!inserted && iterator->second != candidate.economicEventId)
            throw std::invalid_argument(
                "secondary_source_event_maps_to_multiple_official_events:" +
                std::to_string(candidate.secondarySourceEventId));
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& left, const auto& right)
              {
                  return left.economicEventId < right.economicEventId;
              });
    return candidates;
}

struct CliArguments
{
    std::filesystem::path input;
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
    if (parsed.input.empty())
        throw std::invalid_argument("--input is required");
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
        candidates.push_back(std::move(candidate));
    }
    if (candidates.empty())
        throw std::invalid_argument("consensus_csv_has_no_candidates");
    return ValidateAndOrder(std::move(candidates));
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
        const auto candidates =
            LoadAndValidateOandaEconomicConsensusCsv(arguments.input);
        pqxx::connection connection{ConnectionString()};
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
