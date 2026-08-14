#include "CampaignOperationsProductionAdmissionService.hpp"

#if defined(__has_include)
#if __has_include("GeneratedBuildProvenance.hpp")
#include "GeneratedBuildProvenance.hpp"
#endif
#endif

#include <cstdlib>
#include <filesystem>
#include <string>
#include <unistd.h>

using namespace EA::CampaignOperations;

namespace
{

int Fail(const char* diagnostic)
{
    (void)diagnostic;
    return EXIT_FAILURE;
}

bool ExpectedValid()
{
#if defined(EXPECT_VALID_BUILD_PROVENANCE)
    return EXPECT_VALID_BUILD_PROVENANCE != 0;
#else
    return false;
#endif
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 2) return Fail("executable path argument required");
    const std::string executablePath =
        std::filesystem::absolute(argv[1]).string();
    if (::chdir("/") != 0) return Fail("unable to change to root directory");

    const auto actual = CaptureActualManagerBuildContract(executablePath);
    if (!ExpectedValid())
        return actual ? Fail("invalid provenance was accepted") : EXIT_SUCCESS;
#if defined(EXPERTADVISOR_SOURCE_COMMIT)
    const std::string embeddedCommit = EXPERTADVISOR_SOURCE_COMMIT;
#else
    return Fail("valid provenance header was not available to the test");
#endif
    if (!actual || actual->sourceCommit != embeddedCommit ||
        embeddedCommit.size() != 40U)
        return Fail("valid embedded provenance was not captured");

    const std::string expectedCanonical =
        "campaign_operations_manager_build_v1"
        ";manager_service_contract=" +
        std::to_string(std::string(kManagerServiceContract).size()) + ":" +
        kManagerServiceContract +
        ";source_commit=" + embeddedCommit +
        ";source_tree_state=clean;build_configuration=Release"
        ";compiler_contract=" +
        std::to_string(actual->compilerContract.size()) + ":" +
        actual->compilerContract + ";executable_sha256=" +
        actual->executableSha256 + ";build_contract_version=1";
    return actual->identity.canonicalText() == expectedCanonical ?
        EXIT_SUCCESS : Fail("manager build contract v1 bytes changed");
}
