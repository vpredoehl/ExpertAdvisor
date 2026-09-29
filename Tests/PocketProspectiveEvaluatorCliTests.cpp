#include "../Sources/PocketResearchCli.hpp"

#include <cassert>
#include <iostream>

namespace
{
constexpr const char* kCommit = "0123456789abcdef0123456789abcdef01234567";
constexpr const char* kSha = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

bool ParseRejects(std::initializer_list<const char*> options)
{
    std::vector<const char*> args{"LSTM_Release", "--pocket-prospective-evaluator"};
    args.insert(args.end(), options.begin(), options.end());
    try { (void)EA::Pocket::Prospective::Cli::Parse(static_cast<int>(args.size()), args.data()); }
    catch (const std::invalid_argument&) { return true; }
    return false;
}

struct RecordingSqlQuoter final
{
    std::string quote(const std::string& value) const { return "'" + value + "'"; }
    std::string quote(int value) const { return std::to_string(value); }
};

void TestCopyStreamQueryIsAnUnterminatedQueryExpression()
{
    RecordingSqlQuoter transaction;
    const EA::Pocket::Prospective::SymbolSource source{"AUDCAD", "audcadrmp", 0.0001};
    const std::string query = EA::Pocket::Prospective::Cli::ReadOnlyBarsStreamQuery(transaction, source);
    // This is the exact templated builder used by ReadOnlyBars with its real
    // pqxx transaction. pqxx::stream forms COPY (<query>) TO STDOUT, so a
    // semicolon would make the generated COPY statement invalid.
    const std::string copy = "COPY (" + query + ") TO STDOUT";
    assert(!query.empty() && query.back() != ';');
    assert(query.find(';') == std::string::npos);
    assert(copy.find("ORDER BY dt;) TO STDOUT") == std::string::npos);
    assert(copy.find("ORDER BY dt) TO STDOUT") != std::string::npos);
    assert(query.find("candlestick('audcadrmp'") != std::string::npos);
    const std::string confirmation = EA::Pocket::Prospective::Cli::ReadOnlyBarsStreamQuery(
        transaction, source, EA::Pocket::Prospective::Confirmation::kResolutionEnd);
    assert(confirmation.find("2026-01-01 16:00:00") != std::string::npos);
}

void TestDedicatedEntrypointSharesVerificationWiring()
{
    using namespace EA::Pocket::Prospective;
    const auto artifact = std::filesystem::temp_directory_path() /
        ("pocket_research_entrypoint_" + std::to_string(::getpid()));
    std::filesystem::remove_all(artifact);
    ImmutableArtifactWriter(artifact).Publish(FrozenConfiguration(), "synthetic=true", {});
    const std::string artifactText = artifact.string();
    const char* args[] = {"PocketResearch_Release", "--verify-pocket-prospective-artifact", artifactText.c_str()};
    const auto monolith = Cli::TryRun(static_cast<int>(std::size(args)), args);
    const int dedicated = EA::Pocket::Research::RunCli(static_cast<int>(std::size(args)), args);
    assert(monolith.has_value() && *monolith == 0 && dedicated == *monolith);
    std::filesystem::remove_all(artifact);
}
}

int main()
{
    using EA::Pocket::Prospective::Cli::Parse;
    const char* valid[] = {"LSTM_Release", "--pocket-prospective-evaluator", "--validate-only", "--config",
        "Scripts/pocket_prospective_preconfirmation_v1.conf", "--output-dir", "/tmp/pocket-output", "--git-commit", kCommit,
        "--executable-identity", kSha};
    const auto options = Parse(static_cast<int>(std::size(valid)), valid);
    assert(options.validateOnly && !options.execute);
    assert(ParseRejects({"--validate-only", "--config", "x", "--git-commit", kCommit, "--executable-identity", kSha}));
    assert(ParseRejects({"--execute", "--validate-only", "--config", "x", "--output-dir", "/tmp/x", "--git-commit", kCommit, "--executable-identity", kSha}));
    assert(ParseRejects({"--validate-only", "--config", "x", "--output-dir", "/tmp/x", "--git-commit", "short", "--executable-identity", kSha}));
    assert(ParseRejects({"--validate-only", "--config", "x", "--output-dir", "/tmp/x", "--git-commit", kCommit, "--executable-identity", kSha, "--symbol", "EURUSD"}));
    TestCopyStreamQueryIsAnUnterminatedQueryExpression();
    TestDedicatedEntrypointSharesVerificationWiring();
    const char* confirmationValidateOnly[] = {"PocketResearch_Release", "--pocket-confirmation-evaluator", "--validate-only", "--config",
        "Scripts/pocket_prospective_confirmation_2025_v1.conf", "--output-dir", "/tmp/pocket-confirmation", "--git-commit", kCommit,
        "--executable-identity", kSha};
    // Rejected before ExecuteFrozenStudy opens a source connection.
    assert(EA::Pocket::Research::RunCli(static_cast<int>(std::size(confirmationValidateOnly)), confirmationValidateOnly) == 1);
    std::cout << "PocketProspectiveEvaluatorCliTests passed\n";
}
