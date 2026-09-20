#include "SchedulerCore/SchedulerDaemonCli.hpp"
#include "GlobalExperimentControl.hpp"

#if defined(__has_include)
#if __has_include("GeneratedBuildProvenance.hpp")
#include "GeneratedBuildProvenance.hpp"
#endif
#endif

#include "SchedulerExecutablePath.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>

namespace
{

std::string ShellQuoted(const std::string& value)
{
    std::string result{"'"};
    for (const char character : value)
        result += character == '\'' ? "'\\''" : std::string(1, character);
    return result + "'";
}

std::optional<std::string> ExecutableSha256(const std::string& path)
{
    const std::string command =
        "/usr/bin/shasum -a 256 -- " + ShellQuoted(path) + " 2>/dev/null";
    FILE* pipe = ::popen(command.c_str(), "r");
    if (pipe == nullptr) return std::nullopt;

    std::array<char, 256> buffer{};
    std::ostringstream output;
    while (::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe))
        output << buffer.data();
    if (::pclose(pipe) != 0) return std::nullopt;

    const std::string text = output.str();
    const auto separator = text.find_first_of(" \t\r\n");
    const std::string digest = text.substr(0, separator);
    if (digest.size() != 64U ||
        !std::all_of(digest.begin(), digest.end(), [](unsigned char character)
        {
            return (character >= '0' && character <= '9') ||
                (character >= 'a' && character <= 'f');
        }))
    {
        return std::nullopt;
    }
    return "sha256:" + digest;
}

std::string SourceCommit()
{
#if defined(EXPERTADVISOR_SOURCE_COMMIT)
    const std::string sourceCommit = EXPERTADVISOR_SOURCE_COMMIT;
    if (sourceCommit.size() == 40U &&
        std::all_of(sourceCommit.begin(), sourceCommit.end(),
            [](unsigned char character)
            {
                return (character >= '0' && character <= '9') ||
                    (character >= 'a' && character <= 'f');
            }))
    {
        return sourceCommit;
    }
#endif
    return "unavailable";
}

std::string LstmDbConnectionString()
{
    const char* host = std::getenv("LSTM_DB_HOST");
    const char* database = std::getenv("LSTM_DB_NAME");

    return "hostaddr=" +
           std::string{host && *host ? host : "127.0.0.1"} +
           " gssencmode=disable user=pqxx dbname=" +
           std::string{database && *database ? database : "LSTM"};
}

int PrintStandaloneSchedulerBuildIdentity()
{
    try
    {
        const std::string executablePath =
            EA::ExperimentScheduler::ResolveCanonicalExecutablePath();
        const auto executableSha256 = ExecutableSha256(executablePath);
        std::cout << "SCHEDULER_BUILD_IDENTITY"
                  << ",identity_contract_version=1"
                  << ",artifact_role=lstm-scheduler"
                  << ",source_commit=" << SourceCommit()
                  << ",canonical_executable=" << executablePath
                  << ",executable_sha256="
                  << executableSha256.value_or("unavailable")
                  << std::endl;
        return executableSha256.has_value() ? 0 : 1;
    }
    catch (const std::exception& error)
    {
        std::cerr << "SCHEDULER_BUILD_IDENTITY_UNAVAILABLE"
                  << ",diagnostic=" << error.what() << std::endl;
        return 1;
    }
}

} // namespace

int main(int argc, const char* argv[])
{
    if (argc == 2 && std::string{argv[1]} == "--build-identity")
        return PrintStandaloneSchedulerBuildIdentity();

    if (EA::GlobalExperimentControl::IsWorkerAttemptReconciliationCli(
            argc, argv))
    {
        return EA::GlobalExperimentControl::RunWorkerAttemptReconciliationCli(
            argc,
            argv,
            LstmDbConnectionString(),
            std::cout,
            std::cerr);
    }

    return EA::SchedulerCore::RunStandaloneSchedulerDaemonCli(argc, argv);
}
