#include "ManagedInferenceWorkerCli.hpp"

#if defined(__has_include)
#if __has_include("GeneratedBuildProvenance.hpp")
#include "GeneratedBuildProvenance.hpp"
#endif
#endif

#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>
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
    const std::string digest = output.str().substr(0, 64);
    if (digest.size() != 64U ||
        !std::all_of(digest.begin(), digest.end(), [](unsigned char character)
        {
            return (character >= '0' && character <= '9') ||
                (character >= 'a' && character <= 'f');
        }))
        return std::nullopt;
    return "sha256:" + digest;
}

std::string SourceCommit()
{
#if defined(EXPERTADVISOR_SOURCE_COMMIT)
    return EXPERTADVISOR_SOURCE_COMMIT;
#else
    return "unavailable";
#endif
}

int PrintBuildIdentity(const char* executable)
{
    try
    {
        const std::string canonical =
            std::filesystem::canonical(executable).string();
        const auto digest = ExecutableSha256(canonical);
        std::cout << "INFER_WORKER_BUILD_IDENTITY"
                  << ",identity_contract_version=1"
                  << ",artifact_role=lstm-infer-worker"
                  << ",source_commit=" << SourceCommit()
                  << ",canonical_executable=" << canonical
                  << ",executable_sha256="
                  << digest.value_or("unavailable") << std::endl;
        return digest ? 0 : 1;
    }
    catch (const std::exception& error)
    {
        std::cerr << "INFER_WORKER_BUILD_IDENTITY_UNAVAILABLE"
                  << ",diagnostic=" << error.what() << std::endl;
        return 1;
    }
}
} // namespace

int main(int argc, const char* argv[])
{
    if (argc == 2 && std::string{argv[1]} == "--build-identity")
        return PrintBuildIdentity(argv[0]);
    return EA::Inference::RunStandaloneManagedInferenceWorkerCli(argc, argv);
}
