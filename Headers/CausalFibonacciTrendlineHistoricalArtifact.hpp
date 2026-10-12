#pragma once
#include "CausalFibonacciTrendlineHistoricalEvaluation.hpp"
#include "TG4HistoricalEmpiricalEvaluation.hpp"
#include <filesystem>
#include <fstream>

namespace EA::FibonacciResearch::TrendlineHistorical
{
std::string FileHash(const std::filesystem::path& path);
struct Provenance
{
    std::string gitCommit, sourceSHA256, compiler, libraryVersion;
    std::string executableSHA256, configurationFileSHA256;
    std::filesystem::path sourceManifest, sourceArchive, executable, configurationFile;
    std::size_t preflightRows = 0;
    std::string sourceSymbol = "audcadrmp";
};
// Streams bounded-memory artifacts. COMPLETE is processing completion, never
// a statistical result; missing calendar coverage stays explicitly UNVERIFIED.
class ArtifactWriter
{
public:
    explicit ArtifactWriter(std::filesystem::path directory);
    Sinks GetSinks();
    void Complete(const EventEngine& engine, const TG4::EvaluationConfiguration& detector, const Provenance& provenance);
private:
    std::filesystem::path directory_;
    std::ofstream decisions_, outcomes_, geometry_, episodes_, gaps_, exclusions_;
    std::map<std::string, std::string> hashes_;
    void Add(const Decision& d);
    void Add(const OutcomeRecord& o);
    void Add(const GeometryMeasurement& m);
    void Add(const Episode& e);
    void Add(const Gap& g);
    void Write(std::string_view filename, std::string_view text);
    void Copy(std::string_view filename, const std::filesystem::path& source);
    void HashArtifact(const std::string& filename);
};
} // namespace EA::FibonacciResearch::TrendlineHistorical
