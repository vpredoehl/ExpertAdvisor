#include "../Sources/CausalFibonacciIncrementalInformationConfirmation.hpp"

#include <cassert>
#include <fstream>
#include <iostream>

namespace C = EA::CausalFibonacciIncrementalInformation::Confirmation;
namespace F = EA::CausalFibonacciIncrementalInformation;
namespace A = EA::CausalFibonacciIncrementalInformation::Analysis;

namespace {
const std::array<std::string, 6> kSymbols{{"audcadrmp","audusdrmp","eurusdrmp","gbpusdrmp","usdcadrmp","usdjpyrmp"}};

F::Row Row(const std::string& symbol, std::int64_t timestamp, std::uint64_t ordinal, int label, float signal, bool drastic = false)
{
    F::Row row; row.symbol = symbol; row.decisionTimestamp = timestamp; row.sourceRowOrdinal = ordinal;
    for (std::size_t i=0; i<row.baseline.size(); ++i) row.baseline[i] = static_cast<float>((i % 5) + ordinal % 3);
    for (std::size_t i=0; i<row.fibonacci.size(); ++i) row.fibonacci[i] = static_cast<float>((i % 4) + ordinal % 5);
    row.fibonacci[0] = ordinal % 2; row.fibonacci[1] = signal; row.fibonacci[12] = signal > 0 ? 1.0f : 0.0f;
    if (drastic) { for (auto& value : row.baseline) value = -10000.0f; for (auto& value : row.fibonacci) value = 10000.0f; row.fibonacci[0] = 1.0f; }
    const auto end = F::PartitionEnd(*F::PartitionFor(timestamp));
    row.h4 = {label, timestamp + 900, std::min(timestamp + 1800, end - 1), static_cast<float>(label - 1) * .001f, true, ""};
    row.h6 = {(label + 1) % 3, timestamp + 900, std::min(timestamp + 1800, end - 1), static_cast<float>(label - 1) * .002f, true, ""};
    return row;
}

std::vector<F::Row> FixtureRows(bool drasticConfirmation = false, bool perturbNonDevelopment = false, bool extraSymbol = false)
{
    std::vector<F::Row> rows;
    for (std::size_t symbol=0; symbol<kSymbols.size(); ++symbol) {
        for (int i=0; i<9; ++i) rows.push_back(Row(kSymbols[symbol], F::kDevelopmentStart + 900 + i * 900, i + 1, i % 3, static_cast<float>(i - 4)));
        for (int i=0; i<2; ++i) { auto row=Row(kSymbols[symbol], F::kValidationStart + 900 + i * 900, 20 + i, i % 3, perturbNonDevelopment ? 9999.0f : 0.5f, perturbNonDevelopment); if (perturbNonDevelopment) { row.h4.assignedClass=(i+2)%3; row.h6.assignedClass=(i+1)%3; row.h4.terminalLogReturn=99.0f; row.h6.terminalLogReturn=-99.0f; } rows.push_back(std::move(row)); }
        for (int i=0; i<2; ++i) { auto row=Row(kSymbols[symbol], F::kPre2025Start + 900 + i * 900, 30 + i, i % 3, perturbNonDevelopment ? -9999.0f : -0.5f, perturbNonDevelopment); if (perturbNonDevelopment) { row.h4.assignedClass=(i+1)%3; row.h6.assignedClass=(i+2)%3; row.h4.terminalLogReturn=-99.0f; row.h6.terminalLogReturn=99.0f; } rows.push_back(std::move(row)); }
        for (int i=0; i<6; ++i) rows.push_back(Row(kSymbols[symbol], F::kConfirmationStart + 900 + i * 900, 40 + i, drasticConfirmation ? (2 - i % 3) : i % 3, static_cast<float>(i - 2), drasticConfirmation));
        if (extraSymbol && symbol == 1) for (int i=0; i<9; ++i) rows.push_back(Row("cadchfrmp", F::kDevelopmentStart + 900 + i * 900, i + 1, i % 3, static_cast<float>(i)));
    }
    return rows;
}

void WritePre2025Fixture(const std::filesystem::path& directory, const std::string& extractionManifest, const std::string& rows, const std::string& protocol = std::string(F::kProtocolId), const std::string& sha = std::string(F::kProtocolSha256), const std::string& sealed = "sealed_and_discarded_before_parsing", double iterations = 4000)
{
    std::filesystem::create_directories(directory);
    const std::vector<std::string> names{"associations.csv","conditional_incremental.csv","coverage_degeneracy.csv","cross_symbol_equal_summary.csv","fibonacci_ledgers.csv","monthly_deltas.csv","reconstruction.csv","structural_ledgers.csv"};
    for (const auto& name : names) { std::ofstream output(directory / name); output << "fixture\n"; }
    std::ofstream manifest(directory / "manifest.json");
    manifest << "{\n\"runner_identity\":\"causal-fibonacci-layout9-pre2025-artifact-analysis-v2\",\n\"protocol_id\":\"" << protocol << "\",\n\"protocol_sha256\":\"" << sha << "\",\n\"input_manifest_sha256\":\"" << extractionManifest << "\",\n\"input_rows_sha256\":\"" << rows << "\",\n\"code_commit\":\"222598c9655bd3398aa70bd7909bca1e871be35d\",\n\"confirmation_2025\":\"" << sealed << "\",\n\"lambda\":1.0,\n\"lbfgs_max_iterations\":" << iterations << ",\n\"lbfgs_gradient_infinity_tolerance\":1e-8,\n\"lbfgs_relative_objective_tolerance\":1e-12\n}\n";
    manifest.close();
    std::ofstream sums(directory / "sha256sums.txt");
    for (const auto& name : names) sums << F::FileSha256(directory / name) << "  " << name << '\n';
    sums << F::FileSha256(directory / "manifest.json") << "  manifest.json\n";
    sums.close();
}

C::VerificationContract Contract(const std::filesystem::path& artifact, const std::filesystem::path& pre)
{
    const auto contents = C::ReadFile(artifact / "manifest.json", "fixture");
    C::VerificationContract contract;
    contract.extractionProtocolId = F::JsonStringField(contents, "protocol_id");
    contract.extractionProtocolSha256 = F::JsonStringField(contents, "protocol_sha256");
    contract.extractionManifestSha256 = F::FileSha256(artifact / "manifest.json");
    contract.extractionRowsSha256 = F::FileSha256(artifact / "rows.csv");
    contract.pre2025ManifestSha256 = F::FileSha256(pre / "manifest.json");
    contract.pre2025ChecksumsSha256 = F::FileSha256(pre / "sha256sums.txt");
    return contract;
}

void BuildFixture(const std::filesystem::path& root, bool drastic = false, bool perturb = false, bool extra = false)
{
    const auto artifact = root / "artifact";
    F::WriteFixtureArtifact(artifact, F::FixtureArtifactProvenance(), FixtureRows(drastic, perturb, extra));
    WritePre2025Fixture(root / "pre", F::FileSha256(artifact / "manifest.json"), F::FileSha256(artifact / "rows.csv"));
}

C::Options OptionsFor(const std::filesystem::path& root)
{
    C::Options options; options.artifactDirectory=root/"artifact"; options.frozenPre2025ResultDirectory=root/"pre"; options.outputDirectory=root/"out"; options.codeCommit="fixture-commit"; options.explicitlyAuthorized=true; options.noRefit=true; options.verification=Contract(options.artifactDirectory, options.frozenPre2025ResultDirectory); return options;
}

bool Rejects(const std::function<void()>& action)
{
    try { action(); } catch (const std::exception&) { return true; }
    return false;
}

std::vector<double> FittedState(const std::filesystem::path& artifact)
{
    const auto loaded = C::LoadIsolatedRows(artifact);
    const auto& development = loaded.development.at("audcadrmp");
    const auto& confirmation = loaded.confirmation.at("audcadrmp");
    const auto joined = C::JoinDevelopmentAndConfirmation(development, confirmation);
    const auto transform = A::FitDevelopmentInputTransform(joined);
    const auto selected = A::EligibleRows(joined, F::Partition::Development, [](const A::ParsedRow& row) -> const A::ParsedTarget& { return row.h4; });
    const auto model = A::FitMultiRows(joined, selected, transform.augmented, [](const A::ParsedRow& row) -> const A::ParsedTarget& { return row.h4; });
    assert(model.available);
    std::vector<double> state;
    for (const auto& feature : transform.augmented) { state.push_back(feature.mean); state.push_back(feature.standardDeviation); }
    for (std::size_t feature = 0; feature < F::kFibonacciWidth; ++feature)
        for (const bool h6 : {false, true}) for (const double edge : A::FrozenDevelopmentBinEdges(joined, feature, h6)) state.push_back(edge);
    state.insert(state.end(), model.parameters.begin(), model.parameters.end()); return state;
}
}

int main()
{
    const auto root = std::filesystem::temp_directory_path() / "ea_fibonacci_confirmation_fixture";
    std::filesystem::remove_all(root); BuildFixture(root);
    auto options = OptionsFor(root);
    const auto isolated = C::LoadIsolatedRows(root / "artifact");
    assert(isolated.development.at("audcadrmp").size() == 9);
    assert(isolated.confirmation.at("audcadrmp").size() == 6);

    // Authorization and no-refit are independently fail-closed before input parsing.
    options.explicitlyAuthorized = false; assert(Rejects([&] { C::Run(options); })); assert(!std::filesystem::exists(root / "out"));
    options.explicitlyAuthorized = true; options.frozenPre2025ResultDirectory.clear(); assert(Rejects([&] { C::Run(options); }));
    options = OptionsFor(root); options.noRefit = false; assert(Rejects([&] { C::Run(options); }));

    // Every frozen provenance binding is checked against a synthetic immutable contract.
    options = OptionsFor(root); auto bad = options.verification; bad.extractionProtocolId = "wrong"; options.verification=bad; assert(Rejects([&] { C::Run(options); }));
    options = OptionsFor(root); bad=options.verification; bad.extractionProtocolSha256 = "wrong"; options.verification=bad; assert(Rejects([&] { C::Run(options); }));
    options = OptionsFor(root); bad=options.verification; bad.extractionManifestSha256 = "wrong"; options.verification=bad; assert(Rejects([&] { C::Run(options); }));
    options = OptionsFor(root); bad=options.verification; bad.extractionRowsSha256 = "wrong"; options.verification=bad; assert(Rejects([&] { C::Run(options); }));
    options = OptionsFor(root); bad=options.verification; bad.pre2025ManifestSha256 = "wrong"; options.verification=bad; assert(Rejects([&] { C::Run(options); }));
    options = OptionsFor(root); bad=options.verification; bad.pre2025ChecksumsSha256 = "wrong"; options.verification=bad; assert(Rejects([&] { C::Run(options); }));
    options = OptionsFor(root); { std::ofstream corrupt(root/"pre"/"conditional_incremental.csv", std::ios::app); corrupt << "tamper\n"; } assert(Rejects([&] { C::Run(options); })); BuildFixture(root); // restore the disposable fixture after tamper qualification.

    // Direct malformed inputs fail before any fitting/evaluation is possible.
    assert(Rejects([&] { C::ParseConfirmationEvaluationRow("x,audcadrmp,1735689600,1,wrong_partition"); }));
    assert(Rejects([&] { C::ParseConfirmationEvaluationRow("x,audcadrmp,1262304900,1,confirmation_2025"); }));
    assert(Rejects([&] { C::ValidateConfirmationRowsHeader("row_identity,symbol,decision_timestamp,source_row_ordinal,partition,h5_class"); }));

    // A changed solver or unsealed pre-2025 manifest is rejected even when its own checksums are rebuilt.
    const auto solverRoot = root / "solver"; BuildFixture(solverRoot); WritePre2025Fixture(solverRoot/"pre", F::FileSha256(solverRoot/"artifact"/"manifest.json"), F::FileSha256(solverRoot/"artifact"/"rows.csv"), std::string(F::kProtocolId), std::string(F::kProtocolSha256), "sealed_and_discarded_before_parsing", 3999); assert(Rejects([&] { C::Run(OptionsFor(solverRoot)); }));
    const auto sealRoot = root / "seal"; BuildFixture(sealRoot); WritePre2025Fixture(sealRoot/"pre", F::FileSha256(sealRoot/"artifact"/"manifest.json"), F::FileSha256(sealRoot/"artifact"/"rows.csv"), std::string(F::kProtocolId), std::string(F::kProtocolSha256), "not_sealed"); assert(Rejects([&] { C::Run(OptionsFor(sealRoot)); }));
    const auto protocolRoot = root / "protocol"; BuildFixture(protocolRoot); WritePre2025Fixture(protocolRoot/"pre", F::FileSha256(protocolRoot/"artifact"/"manifest.json"), F::FileSha256(protocolRoot/"artifact"/"rows.csv"), "wrong", std::string(F::kProtocolSha256)); assert(Rejects([&] { C::Run(OptionsFor(protocolRoot)); }));
    const auto shaRoot = root / "sha"; BuildFixture(shaRoot); WritePre2025Fixture(shaRoot/"pre", F::FileSha256(shaRoot/"artifact"/"manifest.json"), F::FileSha256(shaRoot/"artifact"/"rows.csv"), std::string(F::kProtocolId), "wrong"); assert(Rejects([&] { C::Run(OptionsFor(shaRoot)); }));

    // The isolated boundary produces complete, checksummed output and refuses overwrite.
    options = OptionsFor(root); C::Run(options); C::VerifyConfirmationOutputDirectory(root/"out"); assert(Rejects([&] { C::Run(options); }));
    assert(C::ReadFile(root/"out"/"conditional_incremental.csv", "fixture").find(",all,6,") != std::string::npos);
    { std::ofstream corrupt(root/"out"/"conditional_incremental.csv", std::ios::app); corrupt << "tamper\n"; }
    assert(Rejects([&] { C::VerifyConfirmationOutputDirectory(root/"out"); }));

    // Confirmation and validation/prelock changes cannot change development transforms or fit parameters.
    const auto confirmationA = root / "a"; const auto confirmationB = root / "b"; const auto nonDevelopment = root / "n";
    BuildFixture(confirmationA); BuildFixture(confirmationB, true); BuildFixture(nonDevelopment, false, true);
    assert(FittedState(confirmationA/"artifact") == FittedState(confirmationB/"artifact"));
    assert(FittedState(confirmationA/"artifact") == FittedState(nonDevelopment/"artifact"));
    auto a = OptionsFor(confirmationA); auto b = OptionsFor(confirmationB); C::Run(a); C::Run(b);
    assert(F::FileSha256(confirmationA/"out"/"conditional_incremental.csv") != F::FileSha256(confirmationB/"out"/"conditional_incremental.csv"));

    // A repeat fixture run is byte-identical and the result surface is restricted to frozen diagnostics.
    const auto repeat = root / "repeat"; BuildFixture(repeat); C::Run(OptionsFor(repeat));
    for (const auto& name : {"associations.csv","conditional_incremental.csv","coverage_degeneracy.csv","cross_symbol_equal_summary.csv","fibonacci_ledgers.csv","manifest.json","monthly_deltas.csv","reconstruction.csv","structural_ledgers.csv","sha256sums.txt","completion.txt"})
        assert(F::FileSha256(confirmationA/"out"/name) == F::FileSha256(repeat/"out"/name));
    assert(!std::filesystem::exists(repeat/"out"/"profitability.csv"));

    // Unsupported universe members are not admitted.
    const auto extra = root / "extra"; BuildFixture(extra, false, false, true); assert(Rejects([&] { C::Run(OptionsFor(extra)); }));
    std::filesystem::remove_all(root);
    std::cout << "CausalFibonacciIncrementalInformationConfirmationTests passed\n";
}
