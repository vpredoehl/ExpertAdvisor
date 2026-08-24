#include "EconomicEventImportService.hpp"

#include "BeaEconomicReleaseAdapter.hpp"
#include "BlsScheduleReleaseAdapter.hpp"
#include "CensusEconomicReleaseAdapter.hpp"
#include "DolEtaWeeklyClaimsAdapter.hpp"
#include "EconomicEventImportValidation.hpp"
#include "FederalReserveEconomicReleaseAdapter.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace EA::EconomicCalendar
{
namespace
{

std::string EnvironmentOr(
    const char* name,
    const char* fallback)
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

struct CliArguments
{
    std::string agency;
    std::filesystem::path manifest;
    EconomicEventImportMode mode = EconomicEventImportMode::dryRun;
    bool modeSpecified = false;
};

CliArguments ParseCli(
    int argc,
    const char* const argv[])
{
    CliArguments parsed;
    bool commandSeen = false;

    for (int index = 1; index < argc; ++index)
    {
        const std::string argument{argv[index]};
        if (argument == "--import-economic-events")
        {
            if (commandSeen)
                throw std::invalid_argument("--import-economic-events specified more than once");
            if (++index >= argc)
                throw std::invalid_argument("--import-economic-events requires an agency");
            parsed.agency = argv[index];
            commandSeen = true;
        }
        else if (argument.rfind("--import-economic-events=", 0) == 0)
        {
            if (commandSeen)
                throw std::invalid_argument("--import-economic-events specified more than once");
            parsed.agency = argument.substr(std::string{"--import-economic-events="}.size());
            commandSeen = true;
        }
        else if (argument == "--manifest")
        {
            if (!parsed.manifest.empty())
                throw std::invalid_argument("--manifest specified more than once");
            if (++index >= argc)
                throw std::invalid_argument("--manifest requires a path");
            parsed.manifest = argv[index];
        }
        else if (argument.rfind("--manifest=", 0) == 0)
        {
            if (!parsed.manifest.empty())
                throw std::invalid_argument("--manifest specified more than once");
            parsed.manifest = argument.substr(std::string{"--manifest="}.size());
        }
        else if (argument == "--dry-run" || argument == "--apply")
        {
            if (parsed.modeSpecified)
                throw std::invalid_argument("exactly one of --dry-run or --apply is required");
            parsed.mode = argument == "--apply"
                ? EconomicEventImportMode::apply
                : EconomicEventImportMode::dryRun;
            parsed.modeSpecified = true;
        }
        else
        {
            throw std::invalid_argument("unsupported economic-event import option: " + argument);
        }
    }

    if (!commandSeen || parsed.agency.empty())
        throw std::invalid_argument("--import-economic-events requires an agency");
    if (parsed.agency != "dol-eta" && parsed.agency != "bea" &&
        parsed.agency != "bls" &&
        parsed.agency != "census" && parsed.agency != "federal-reserve")
        throw std::invalid_argument("unsupported economic-event agency: " + parsed.agency);
    if (parsed.manifest.empty())
        throw std::invalid_argument("--manifest is required");
    if (!parsed.modeSpecified)
        throw std::invalid_argument("exactly one of --dry-run or --apply is required");
    return parsed;
}

} // namespace


EconomicEventImportReport RunEconomicEventImport(
    pqxx::connection& connection,
    std::vector<AuthoritativeEconomicEventCandidate> candidates,
    EconomicEventImportMode mode)
{
    const auto ordered = ValidateAndOrderEconomicEventCandidates(
        std::move(candidates));

    if (mode == EconomicEventImportMode::dryRun)
    {
        pqxx::read_transaction transaction{connection};
        return CompareEconomicEventImportBatch(transaction, ordered);
    }

    return ApplyEconomicEventImportBatch(connection, ordered);
}


bool IsEconomicEventImportCommand(
    int argc,
    const char* const argv[])
{
    for (int index = 1; index < argc; ++index)
    {
        const std::string argument{argv[index]};
        if (argument == "--import-economic-events" ||
            argument.rfind("--import-economic-events=", 0) == 0)
        {
            return true;
        }
    }
    return false;
}


int RunEconomicEventImportCli(
    int argc,
    const char* const argv[])
{
    try
    {
        const CliArguments arguments = ParseCli(argc, argv);

        // Acquisition, digest verification, parsing, and pure validation all
        // finish before any database transaction is opened.
        std::vector<AuthoritativeEconomicEventCandidate> candidates;
        if (arguments.agency == "bea")
            candidates = LoadBeaEconomicReleaseManifest(arguments.manifest);
        else if (arguments.agency == "bls")
            candidates = LoadBlsScheduleReleaseManifest(arguments.manifest);
        else if (arguments.agency == "census")
            candidates = LoadCensusEconomicReleaseManifest(arguments.manifest);
        else if (arguments.agency == "federal-reserve")
            candidates = LoadFederalReserveEconomicReleaseManifest(
                arguments.manifest);
        else
            candidates = LoadDolEtaWeeklyClaimsManifest(arguments.manifest);
        candidates = ValidateAndOrderEconomicEventCandidates(std::move(candidates));

        pqxx::connection connection{ConnectionString()};
        const EconomicEventImportReport report = RunEconomicEventImport(
            connection,
            std::move(candidates),
            arguments.mode);

        std::cout << "ECONOMIC_EVENT_IMPORT_SUMMARY"
                  << ",agency=" << arguments.agency
                  << ",mode="
                  << (arguments.mode == EconomicEventImportMode::dryRun
                          ? "dry-run"
                          : "apply")
                  << ",inserted=" << report.inserted
                  << ",unchanged=" << report.unchanged
                  << ",updated=" << report.updated
                  << ",rejected=" << report.rejected
                  << std::endl;

        for (const auto& item : report.items)
        {
            std::cout << "ECONOMIC_EVENT_IMPORT_ITEM"
                      << ",source_event_id=" << item.sourceEventId
                      << ",disposition="
                      << EconomicEventImportDispositionName(item.disposition)
                      << ",diagnostic=" << item.diagnostic
                      << std::endl;
        }
        return report.rejected == 0 ? 0 : 2;
    }
    catch (const std::exception& error)
    {
        std::cerr << "ECONOMIC_EVENT_IMPORT_FAILED,error="
                  << error.what() << std::endl;
        return 1;
    }
}

} // namespace EA::EconomicCalendar
