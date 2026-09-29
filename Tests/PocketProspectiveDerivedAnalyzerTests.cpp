#include "PocketProspectiveDerivedAnalyzer.hpp"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace
{
using namespace EA::Pocket;
using namespace EA::Pocket::Prospective;
namespace Derived = EA::Pocket::Prospective::Derived;

EvaluatedObservation Fixture(std::string symbol, PocketDirection direction, std::size_t confirmation,
    std::int64_t timestamp, std::string race)
{
    PocketObservation observation{direction, direction == PocketDirection::Bullish ? PocketPriceRange{10,11} : PocketPriceRange{9,10},
        confirmation-1,timestamp-900,confirmation,timestamp,confirmation,timestamp,"15m_completed"};
    EvaluatedObservation value{ObservationIdentity(FrozenConfiguration(),symbol,15,observation),symbol,"exploratory",15,observation,{}};
    for(std::size_t i=0;i<kHorizons.size();++i) { auto& label=value.outcomes[i]; label.horizon=kHorizons[i]; label.complete=true; label.touchAt=1; label.closeAt=2; label.mfe=.002; label.mae=.001; label.directionalCloseReturn=direction==PocketDirection::Bullish ? -.0005 : .0005; if(label.horizon==64) label.race=race; }
    return value;
}
std::filesystem::path Root(std::string_view suffix) { return std::filesystem::temp_directory_path()/ ("pocket_derived_fixture_"+std::string(suffix)+"_"+std::to_string(::getpid())); }
void Append(const std::filesystem::path& file) { std::ofstream out(file,std::ios::app); out<<"tamper\n"; }

void TestVerifierPrecedesParsingAndTampering()
{
    const auto source=Root("source"), output=Root("output"); std::filesystem::remove_all(source);std::filesystem::remove_all(output);
    const std::vector<EvaluatedObservation> records{Fixture("AUDCAD",PocketDirection::Bullish,21,1262322900,"continuation_first")};
    ImmutableArtifactWriter(source).Publish(FrozenConfiguration(),"synthetic=true",records);
    const std::string original=ReadTextFile(source/"observations.csv");
    Derived::Analyze(source,output,"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
    Derived::Writer::Verify(output); assert(ReadTextFile(source/"observations.csv")==original);
    assert(ReadTextFile(output/"outcomes.csv").find("continuation_first")!=std::string::npos);
    bool exists=false; try { Derived::Analyze(source,output,"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"); } catch(const std::invalid_argument&) {exists=true;} assert(exists);
    std::filesystem::remove_all(output); Append(source/"observations.csv"); bool rejected=false; try { Derived::Analyze(source,output,"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"); }catch(const std::invalid_argument&){rejected=true;} assert(rejected);
    std::filesystem::remove_all(source); std::filesystem::remove_all(output);
}
void TestMetricsAndRaceCategories()
{
    const auto source=Root("metrics"), output=Root("metrics_output");std::filesystem::remove_all(source);std::filesystem::remove_all(output);
    std::vector<EvaluatedObservation> records;
    const std::array<std::string,4> races{{"continuation_first","revisit_first","same_bar_intrabar_order_indeterminate","neither"}};
    for(std::size_t i=0;i<races.size();++i) records.push_back(Fixture("AUDCAD",i%2?PocketDirection::Bearish:PocketDirection::Bullish,21+i*64,1262322900+static_cast<std::int64_t>(i)*57600,races[i]));
    records[0].outcomes[2].mfe=.002; records[0].outcomes[2].mae=.001; records[0].outcomes[2].directionalCloseReturn=-.0005;
    records[3].outcomes[2].complete=false; records[3].outcomes[2].censor=CensorReason::Tail; records[3].outcomes[2].touchAt.reset(); records[3].outcomes[2].closeAt.reset();
    ImmutableArtifactWriter(source).Publish(FrozenConfiguration(),"synthetic=true",records);
    Derived::Analyze(source,output,"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb");
    const std::string report=ReadTextFile(output/"outcomes.csv"); const std::string structural=ReadTextFile(output/"structural.csv");
    assert(report.find("directional_close_return_median_pips")!=std::string::npos && report.find("race_censored")!=std::string::npos);
    assert(report.find("same_bar_intrabar_order_indeterminate")!=std::string::npos && structural.find("greedy_temporal_thinned_64")!=std::string::npos);
    std::filesystem::remove_all(source);std::filesystem::remove_all(output);
}
void TestManifestConfigAggregateTamperAndAtomicFailure()
{
    for(const char* file:{"manifest.txt","configuration.conf","aggregates.csv"}) {
        const auto source=Root(file),output=Root("bad");std::filesystem::remove_all(source);std::filesystem::remove_all(output);
        ImmutableArtifactWriter(source).Publish(FrozenConfiguration(),"synthetic=true",{Fixture("AUDCAD",PocketDirection::Bullish,21,1262322900,"neither")}); Append(source/file);
        bool rejected=false;try{Derived::Analyze(source,output,"cccccccccccccccccccccccccccccccccccccccc","cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc");}catch(const std::invalid_argument&){rejected=true;}assert(rejected);assert(!std::filesystem::exists(output));std::filesystem::remove_all(source);
    }
}
}
int main(){TestVerifierPrecedesParsingAndTampering();TestMetricsAndRaceCategories();TestManifestConfigAggregateTamperAndAtomicFailure();std::cout<<"PocketProspectiveDerivedAnalyzerTests passed\n";}
