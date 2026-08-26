#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <pqxx/pqxx>

namespace EA::EconomicCalendar
{

struct ConsensusParsedValue
{
    std::optional<std::string> raw;
    std::string parseStatus;
    std::optional<std::string> valueKind;
    std::optional<std::string> valueLow;
    std::optional<std::string> valueHigh;
    std::optional<std::string> canonicalValueLow;
    std::optional<std::string> canonicalValueHigh;
    std::optional<std::string> unit;
    std::optional<std::string> scale;
    std::optional<std::string> qualifier;
};

struct EconomicEventConsensusCandidate
{
    std::int64_t economicEventId = 0;
    std::string eventFamily;
    std::string eventTimestampUtc;
    std::string sourceAgency;
    std::string sourceEventId;
    std::optional<std::string> referencePeriod;
    std::string sourceReleaseDate;

    std::string consensusSource;
    std::int64_t sourceReportId = 0;
    std::int64_t secondarySourceEventId = 0;
    std::string secondarySourceEventName;
    std::string secondarySourcePeriod;
    int secondarySourcePriority = 0;
    std::int64_t secondarySourceTimestampEpoch = 0;
    std::string secondarySourceDate;
    std::string secondarySourceArtifactPath;
    std::string matchRule;
    std::string semanticContract = "oanda_economic_consensus_candidate_v1";

    ConsensusParsedValue forecast;
    ConsensusParsedValue previous;
    ConsensusParsedValue actual;
};

enum class EconomicEventConsensusImportMode
{
    dryRun,
    apply
};

enum class EconomicEventConsensusImportDisposition
{
    inserted,
    unchanged,
    rejected
};

struct EconomicEventConsensusImportItemResult
{
    std::int64_t economicEventId = 0;
    EconomicEventConsensusImportDisposition disposition =
        EconomicEventConsensusImportDisposition::rejected;
    std::string diagnostic;
};

struct EconomicEventConsensusImportReport
{
    std::size_t inserted = 0;
    std::size_t unchanged = 0;
    std::size_t rejected = 0;
    std::vector<EconomicEventConsensusImportItemResult> items;
};

std::vector<EconomicEventConsensusCandidate>
LoadAndValidateOandaEconomicConsensusCsv(const std::filesystem::path& path);

EconomicEventConsensusImportReport RunEconomicEventConsensusImport(
    pqxx::connection& connection,
    const std::vector<EconomicEventConsensusCandidate>& candidates,
    EconomicEventConsensusImportMode mode);

const char* EconomicEventConsensusImportDispositionName(
    EconomicEventConsensusImportDisposition disposition);

bool IsEconomicEventConsensusImportCommand(
    int argc,
    const char* const argv[]);

// Isolated persistence-only CLI.  It is dispatched before all model,
// scheduler, training, and inference argument handling.
int RunEconomicEventConsensusImportCli(
    int argc,
    const char* const argv[]);

} // namespace EA::EconomicCalendar
