#include "../Sources/CausalFibonacciIncrementalInformationAnalysis.hpp"

#include <cassert>
#include <fstream>

namespace A = EA::CausalFibonacciIncrementalInformation::Analysis;

int main()
{
    const std::string confirmation =
        "ignored,audcadrmp,1735689600,1,confirmation_2025,99,0,0,nan,0,,99,0,0,nan,0,";
    assert(!A::ParsePre2025Row(confirmation).has_value());

    const std::string development =
        "id,audcadrmp,1262304900,1,development,2,1262305800,1262308500,0.01,1,,1,1262305800,1262310300,0,1,";
    bool shortRejected = false;
    try { (void)A::ParsePre2025Row(development); }
    catch (const std::invalid_argument&) { shortRejected = true; }
    assert(shortRejected);

    const auto root = std::filesystem::temp_directory_path() / "ea_fibonacci_pre2025_analysis_fixture";
    const auto artifact = root / "artifact", output = root / "output";
    std::filesystem::remove_all(root);
    std::vector<EA::CausalFibonacciIncrementalInformation::Row> rows;
    const std::array<std::string, 6> symbols{{"audcadrmp","audusdrmp","eurusdrmp","gbpusdrmp","usdcadrmp","usdjpyrmp"}};
    for (const auto& symbol : symbols) {
        std::uint64_t ordinal = 0;
        for (const auto [start, count] : std::array<std::pair<std::int64_t, int>, 4>{{
            {EA::CausalFibonacciIncrementalInformation::kDevelopmentStart + 900, 6},
            {EA::CausalFibonacciIncrementalInformation::kValidationStart + 900, 3},
            {EA::CausalFibonacciIncrementalInformation::kPre2025Start + 900, 3},
            {EA::CausalFibonacciIncrementalInformation::kConfirmationStart + 900, 1}}}) {
            for (int n = 0; n < count; ++n, ++ordinal) {
                EA::CausalFibonacciIncrementalInformation::Row row;
                row.symbol = symbol; row.decisionTimestamp = start + n * 900; row.sourceRowOrdinal = ordinal;
                for (std::size_t i = 0; i < row.baseline.size(); ++i) row.baseline[i] = static_cast<float>(ordinal + i * .01);
                for (std::size_t i = 0; i < row.fibonacci.size(); ++i) row.fibonacci[i] = static_cast<float>((ordinal + i) % 5);
                row.fibonacci[0] = ordinal % 2;
                const bool confirmation = start == EA::CausalFibonacciIncrementalInformation::kConfirmationStart + 900;
                row.h4 = {confirmation ? 99 : static_cast<int>(ordinal % 3), row.decisionTimestamp + 900, row.decisionTimestamp + 900, .001f, !confirmation, {}};
                row.h6 = {confirmation ? 99 : static_cast<int>((ordinal + 1) % 3), row.decisionTimestamp + 900, row.decisionTimestamp + 900, -.001f, !confirmation, {}};
                rows.push_back(row);
            }
        }
    }
    EA::CausalFibonacciIncrementalInformation::WriteArtifact(artifact,
        EA::CausalFibonacciIncrementalInformation::FixtureArtifactProvenance(), rows);
    A::Run({artifact, output, "fixture-analysis"});
    std::ifstream conditional(output / "conditional_incremental.csv");
    const std::string contents((std::istreambuf_iterator<char>(conditional)), {});
    assert(contents.find("confirmation_2025") == std::string::npos);
    assert(std::filesystem::exists(output / "cross_symbol_equal_summary.csv"));
    std::filesystem::remove_all(root);
}
