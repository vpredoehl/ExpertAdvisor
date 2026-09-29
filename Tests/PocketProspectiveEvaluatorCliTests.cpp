#include "../Sources/PocketProspectiveEvaluatorCli.hpp"

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
    std::cout << "PocketProspectiveEvaluatorCliTests passed\n";
}
