#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <pqxx/pqxx>

#include "../Headers/HistoricalFxTimestamp.hpp"
#include "../Sources/EconomicEventFeatures.hpp"
#include "../Sources/EconomicEventRepository.hpp"

using namespace EA::EconomicCalendar;

namespace
{

std::string EnvironmentOr(
    const char* name,
    const char* fallback)
{
    const char* value = std::getenv(name);
    return value && *value ? value : fallback;
}


std::string Trim(std::string value)
{
    const std::size_t first =
        value.find_first_not_of(" \t\r\n");

    if (first == std::string::npos)
        return {};

    const std::size_t last =
        value.find_last_not_of(" \t\r\n");

    return value.substr(first, last - first + 1);
}


std::optional<std::string> ExtractTimestampColumn(
    const std::string& line)
{
    const std::size_t pipe = line.find('|');

    if (pipe == std::string::npos)
        return std::nullopt;

    const std::string value = Trim(line.substr(0, pipe));

    if (
        value.size() != 19 ||
        value[4] != '-' ||
        value[7] != '-' ||
        value[10] != ' ' ||
        value[13] != ':' ||
        value[16] != ':')
    {
        return std::nullopt;
    }

    return value;
}


std::vector<PriceTP> LoadRealInput15mBars(
    const std::string& path,
    const std::string& fromCivil,
    const std::string& throughCivil)
{
    std::ifstream input{path};

    if (!input)
        throw std::runtime_error("cannot_open_input15m:" + path);

    std::vector<PriceTP> bars;
    std::string line;

    while (std::getline(input, line))
    {
        const auto timestamp = ExtractTimestampColumn(line);

        if (!timestamp)
            continue;

        if (*timestamp < fromCivil)
            continue;

        if (*timestamp > throughCivil)
            break;

        PriceTP time{};

        if (
            !EA::HistoricalFxTimestamp::ParseNewYorkCivilTimestamp(
                *timestamp,
                time))
        {
            throw std::runtime_error(
                "cannot_parse_input15m_timestamp:" +
                *timestamp);
        }

        bars.push_back(time);
    }

    if (bars.empty())
        throw std::runtime_error("no_input15m_bars_in_requested_range");

    return bars;
}


std::size_t FindBarIndex(
    const std::vector<PriceTP>& bars,
    const std::string& civilTimestamp)
{
    PriceTP target{};

    if (
        !EA::HistoricalFxTimestamp::ParseNewYorkCivilTimestamp(
            civilTimestamp,
            target))
    {
        throw std::runtime_error("cannot_parse_expected_bar_timestamp");
    }

    for (std::size_t index = 0; index < bars.size(); ++index)
    {
        if (bars[index] == target)
            return index;
    }

    throw std::runtime_error(
        "expected_bar_not_found:" +
        civilTimestamp);
}


void AssertRanges(
    const EconomicEventFeatureValues& values)
{
    const auto ordered = values.Ordered();

    for (std::size_t index = 0; index < ordered.size(); ++index)
        assert(std::isfinite(ordered[index]));

    for (std::size_t index = 0; index < kEconomicEventModelFamilyCount; ++index)
        assert(ordered[index] == 0.0F || ordered[index] == 1.0F);

    for (std::size_t index = kEconomicEventModelFamilyCount;
         index < kPreConsensusEconomicEventFeatureWidth;
         ++index)
        assert(ordered[index] >= 0.0F && ordered[index] <= 1.0F);

    assert(values.relevantEventHasConsensus == 0.0F ||
           values.relevantEventHasConsensus == 1.0F);
    assert(values.relevantEventConsensusIsRange == 0.0F ||
           values.relevantEventConsensusIsRange == 1.0F);
    assert(values.releasedEventHasSurprise == 0.0F ||
           values.releasedEventHasSurprise == 1.0F);
    assert(values.releasedEventSurpriseAbs >= 0.0F);
    assert(values.releasedEventSurpriseDirection == -1.0F ||
           values.releasedEventSurpriseDirection == 0.0F ||
           values.releasedEventSurpriseDirection == 1.0F);
}

} // namespace


int main()
{
    const std::vector<PriceTP> bars =
        LoadRealInput15mBars(
            EnvironmentOr("INPUT15M_PATH", "input15m.txt"),
            "2010-01-08 07:45:00",
            "2010-01-15 09:00:00");

    const std::size_t employmentBefore =
        FindBarIndex(bars, "2010-01-08 08:15:00");
    const std::size_t employmentBar =
        FindBarIndex(bars, "2010-01-08 08:30:00");
    const std::size_t employmentAfter =
        FindBarIndex(bars, "2010-01-08 08:45:00");
    const std::size_t cpiBar =
        FindBarIndex(bars, "2010-01-15 08:30:00");

    pqxx::connection connection{
        "host=" + EnvironmentOr("LSTM_DB_HOST", "localhost") +
        " user=" + EnvironmentOr("LSTM_DB_USER", "pqxx") +
        " dbname=" + EnvironmentOr("LSTM_DB_NAME", "LSTM")};

    // libpqxx read_transaction and the test shell's PGOPTIONS both enforce
    // read-only production access.
    pqxx::read_transaction transaction{connection};
    assert(EconomicEventSchemaExists(transaction));

    const std::vector<EconomicEvent> events =
        LoadEconomicEvents(
            transaction,
            "USD",
            "2000-01-01 00:00:00+00",
            "2100-01-01 00:00:00+00");

    assert(!events.empty());

    bool foundEmployment = false;
    bool foundCpi = false;
    std::array<bool, kEconomicEventModelFamilyCount> foundModelFamilies{};

    for (std::size_t index = 0; index < events.size(); ++index)
    {
        if (index > 0)
        {
            assert(
                events[index - 1].eventTimestampUnixMicros <=
                events[index].eventTimestampUnixMicros);
        }

        const EconomicEventModelFamily modelFamily =
            MapEconomicEventModelFamily(
                events[index].sourceAgency,
                events[index].eventFamily);

        foundModelFamilies[
            static_cast<std::size_t>(modelFamily)] =
            true;

        foundEmployment =
            foundEmployment ||
            events[index].eventFamily == "EMPLOYMENT";

        foundCpi =
            foundCpi ||
            events[index].eventFamily == "CPI";
    }

    assert(foundEmployment);
    assert(foundCpi);

    for (bool found : foundModelFamilies)
        assert(found);

    EconomicEventFeatureEngine firstRun{events};
    EconomicEventFeatureEngine secondRun{events};

    std::vector<EconomicEventFeatureValues> firstValues;
    std::vector<EconomicEventFeatureValues> secondValues;
    firstValues.reserve(bars.size());
    secondValues.reserve(bars.size());

    for (const PriceTP bar : bars)
    {
        firstValues.push_back(
            firstRun.AdvanceCompletedBar(bar));
        secondValues.push_back(
            secondRun.AdvanceCompletedBar(bar));

        AssertRanges(firstValues.back());
        AssertRanges(secondValues.back());
        assert(
            firstValues.back().Ordered() ==
            secondValues.back().Ordered());
    }

    // 08:30 is an exact input15m bar boundary. Strict bar-close causality
    // excludes it from 08:15 and activates it on 08:30.
    assert(
        firstValues[employmentBefore].employmentEvent ==
        0.0F);
    assert(
        firstValues[employmentBefore].employmentRecencyDecay ==
        0.0F);

    assert(
        firstValues[employmentBar].employmentEvent ==
        1.0F);
    assert(
        firstValues[employmentBar].employmentRecencyDecay >
        0.0F);

    assert(
        firstValues[employmentAfter].employmentEvent ==
        0.0F);
    assert(
        firstValues[employmentAfter].employmentRecencyDecay >
        0.0F);
    assert(
        firstValues[employmentAfter].employmentRecencyDecay <
        firstValues[employmentBar].employmentRecencyDecay);

    assert(firstValues[cpiBar].inflationEvent == 1.0F);
    assert(firstValues[cpiBar].inflationRecencyDecay > 0.0F);

    assert(firstRun.ConsumedEventCount() > 0);
    assert(
        firstRun.ConsumedEventCount() ==
        secondRun.ConsumedEventCount());

    std::cout
        << "ECONOMIC_EVENT_FEATURES_REAL_INPUT_INTEGRATION_PASS"
        << ",bars="
        << bars.size()
        << ",events="
        << events.size()
        << ",consumed="
        << firstRun.ConsumedEventCount()
        << '\n';

    return 0;
}
