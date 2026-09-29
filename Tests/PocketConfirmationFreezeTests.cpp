#include "PocketConfirmationFreeze.hpp"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace
{
using namespace EA::Pocket::Prospective;
namespace Confirmation = EA::Pocket::Prospective::Confirmation;

void TestCanonicalOfflineFreeze()
{
    const auto configuration=Confirmation::LoadAndValidateConfiguration("Scripts/pocket_prospective_confirmation_2025_v1.conf");
    assert(configuration.study == "pocket-prospective-confirmation-2025-v1");
    assert(configuration.study != kStudyId);
    assert(configuration.scoringPartition.name == "confirmation");
    assert(configuration.scoringPartition.start == 1735689600);
    assert(configuration.scoringPartition.end == 1767225600);
    assert(configuration.resolutionEnd == 1767283200);
    assert(configuration.lookbacks == kLookbacks && configuration.horizons == kHorizons);
    assert(configuration.symbols.size() == 6 && configuration.symbols[5].symbol == "USDJPY");
    assert(Confirmation::CanonicalConfigurationText() == ReadTextFile("Scripts/pocket_prospective_confirmation_2025_v1.conf"));
    assert(configuration.configurationSha256 == Sha256(Confirmation::CanonicalPayload()));
    assert(Confirmation::kPrimaryArtifactSchema != "phase-pocket-4-derived-preconfirmation-report-v2");
    assert(Confirmation::kDerivedReportSchema != "phase-pocket-4-derived-preconfirmation-report-v2");
    assert(Confirmation::ValidationSummary(configuration).find("offline=true") != std::string::npos);
}

void TestNoOutputAndNoNoncanonicalConfiguration()
{
    const auto existing=std::filesystem::temp_directory_path()/"pocket_confirmation_existing_target";
    std::filesystem::create_directories(existing); bool rejected=false;
    try { Confirmation::ValidateAbsentOutputTarget(existing); } catch(const std::invalid_argument&) { rejected=true; }
    assert(rejected); std::filesystem::remove_all(existing);
    const auto malformed=std::filesystem::temp_directory_path()/"pocket_confirmation_malformed.conf";
    { std::ofstream output(malformed); output << Confirmation::CanonicalConfigurationText() << "unknown=x\n"; }
    rejected=false; try { (void)Confirmation::LoadAndValidateConfiguration(malformed); } catch(const std::invalid_argument&) { rejected=true; }
    assert(rejected); std::filesystem::remove(malformed);
}
}
int main(){TestCanonicalOfflineFreeze();TestNoOutputAndNoNoncanonicalConfiguration();std::cout<<"PocketConfirmationFreezeTests passed\n";}
