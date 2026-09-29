#include "PocketConfirmationDerived.hpp"

#include <cassert>
#include <filesystem>
#include <iostream>

namespace
{
using namespace EA::Pocket;
using namespace EA::Pocket::Prospective;
namespace Confirmation=EA::Pocket::Prospective::Confirmation;

EvaluatedObservation Fixture()
{
    const RunConfiguration config=Confirmation::EvaluatorConfiguration();
    PocketObservation observation{PocketDirection::Bullish,{10,11},20,Confirmation::kScoringStart-900,21,Confirmation::kScoringStart,21,Confirmation::kScoringStart,"15m_completed"};
    EvaluatedObservation output{ObservationIdentity(config,"AUDCAD",15,observation),"AUDCAD","confirmation",15,observation,{}};
    for(std::size_t index=0;index<kHorizons.size();++index){auto& label=output.outcomes[index];label.horizon=kHorizons[index];label.complete=true;label.touchAt=1;label.closeAt=2;label.mfe=.001;label.mae=.0005;label.directionalCloseReturn=-.0002;if(label.horizon==64)label.race="neither";}
    return output;
}
void TestSyntheticConfirmationContracts()
{
    const auto source=std::filesystem::temp_directory_path()/"pocket_confirmation_primary_synthetic";
    const auto derived=std::filesystem::temp_directory_path()/"pocket_confirmation_derived_synthetic";
    std::filesystem::remove_all(source);std::filesystem::remove_all(derived);
    const RunConfiguration config=Confirmation::EvaluatorConfiguration();
    ImmutableArtifactWriter(source).Publish(config,"synthetic=true",{Fixture()},Confirmation::CanonicalConfigurationText(),Confirmation::kPrimaryArtifactSchema);
    Confirmation::VerifyPrimaryArtifact(source);
    const std::string original=ReadTextFile(source/"observations.csv");
    Confirmation::DerivedReport::Analyze(source,derived,"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
    Confirmation::DerivedReport::Verify(derived);
    assert(ReadTextFile(source/"observations.csv")==original);
    assert(ReadTextFile(derived/"manifest.txt").find("analyzer_schema=phase-pocket-4-derived-confirmation-report-v1")!=std::string::npos);
    std::filesystem::remove_all(source);std::filesystem::remove_all(derived);
}
}
int main(){TestSyntheticConfirmationContracts();std::cout<<"PocketConfirmationExecutionTests passed\n";}
