#pragma once
#include "CausalFibonacciTrendlineHistoricalEvaluation.hpp"
#include "TG4HistoricalEmpiricalEvaluation.hpp"
#include <filesystem>
#include <fstream>
#include <memory>

namespace EA::FibonacciResearch::TrendlineHistorical
{
std::string FileHash(const std::filesystem::path& path);
inline constexpr std::string_view kArtifactFormat = "fib-trendline-artifacts-v3";
enum class EpisodePublication { Summary, Full };
inline const char* Name(EpisodePublication tier) { return tier == EpisodePublication::Summary ? "summary" : "full"; }
enum class CompressionOperation { Compress, Write, Finalize };
// Optional fault seam for synthetic publication tests; never exposed by the CLI.
using CompressionHook = std::function<void(CompressionOperation)>;
struct DecodedContent { std::string sha256; std::size_t bytes = 0, rows = 0; };
class CompressedCsv
{
public:
    static constexpr std::size_t kBufferBytes = 65536;
    explicit CompressedCsv(const std::filesystem::path& path, CompressionHook hook = {});
    ~CompressedCsv();
    void Append(std::string_view bytes);
    DecodedContent Finish();
private:
    struct State;
    std::unique_ptr<State> state_;
};
// Strict single-member gzip validation, including CRC/trailer and trailing bytes.
// The optional consumer receives bounded decoded chunks; no whole-file buffering.
DecodedContent InspectGzip(const std::filesystem::path& path,
    const std::function<void(std::string_view)>& consumer = {});
// Publication-integrity validation accepts legacy semantic-v2 CSV publications
// and the revised compressed format. It does not establish scientific validity.
void ValidateArtifactPublication(const std::filesystem::path& directory);
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
    explicit ArtifactWriter(std::filesystem::path directory,
        EpisodePublication tier = EpisodePublication::Summary, CompressionHook hook = {});
    ~ArtifactWriter();
    Sinks GetSinks();
    void Complete(const EventEngine& engine, const TG4::EvaluationConfiguration& detector, const Provenance& provenance);
private:
    std::filesystem::path directory_;
    std::ofstream decisions_, outcomes_, gaps_, exclusions_;
    EpisodePublication tier_;
    std::unique_ptr<CompressedCsv> geometry_, episodes_;
    std::map<std::string, DecodedContent> decoded_;
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
