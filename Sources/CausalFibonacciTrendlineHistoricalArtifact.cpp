#include "CausalFibonacciTrendlineHistoricalArtifact.hpp"
#include <zlib.h>

namespace EA::FibonacciResearch::TrendlineHistorical
{
namespace
{
void Configure(std::ofstream& stream)
{ stream.exceptions(std::ios::badbit | std::ios::failbit); stream.imbue(std::locale::classic()); stream << std::setprecision(17); }
std::string Json(std::string_view value)
{
    std::string out = "\"";
    for (unsigned char c : value)
    {
        if (c == '"' || c == '\\') { out += '\\'; out += static_cast<char>(c); }
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c < 32) throw std::invalid_argument("unsupported metadata control character");
        else out += static_cast<char>(c);
    }
    return out + '"';
}
void Cell(std::ostream& out, const std::optional<double>& value) { if (value) out << Number(*value); }
void Measurement(std::ostream& out, const GeometryMeasurement& m)
{
    out << m.timestamp << ',' << m.ab << ',' << m.line << ',' << Number(m.ratio) << ',' << Number(m.fibPrice)
        << ',' << Number(m.linePrice) << ',' << Number(m.dF) << ',' << Number(m.dL) << ',' << Number(m.dS) << ',' << Name(m.cohort);
}
std::string DetectorConfiguration(const TG4::EvaluationConfiguration& c)
{
    // Serialize detector parameters only. Do not import the TG4 study's
    // warmup, scoring contract, single-ratio provenance or outcome definitions.
    const auto& g = c.geometry; const auto& b = c.behavior; const auto& f = c.fibonacci;
    std::ostringstream out; out.imbue(std::locale::classic()); out << std::setprecision(17);
    out << "{\n\"schema\":\"research-tg3-detector-configuration-v2\",\n"
        << "\"geometry\":{\"intervening_tolerance\":" << g.interveningPriceTolerance
        << ",\"touch_tolerance\":" << g.touchPriceTolerance << ",\"anchor_lookback\":" << g.maxFractalAnchorLookbackBars
        << ",\"max_fractals\":" << g.maxConfirmedFractalsPerKind << ",\"candidate_age\":" << g.maxCandidateAgeBars
        << ",\"max_candidates\":" << g.maxCandidates << ",\"atr_period\":" << g.atrPeriod << "},\n"
        << "\"reference_bar_scale\":" << c.referenceBarScale << ",\n"
        << "\"behavior\":{\"break_policy\":" << static_cast<int>(b.breakPolicy)
        << ",\"rearm_policy\":" << static_cast<int>(b.rearmPolicy) << ",\"retest_policy\":" << static_cast<int>(b.retestContactPolicy)
        << ",\"outer_pairing_policy\":" << static_cast<int>(b.outerPairingPolicy)
        << ",\"break_tolerance\":" << b.breakPriceTolerance << ",\"retest_tolerance\":" << b.retestPriceTolerance
        << ",\"target_tolerance\":" << b.outerTargetPriceTolerance << ",\"retest_horizon\":" << b.retestHorizonBars
        << ",\"target_horizon\":" << b.outerTargetHorizonBars << ",\"active_limit\":" << b.maxActiveBreakObservations
        << ",\"retained_limit\":" << b.maxRetainedBreakObservations << "},\n"
        << "\"fibonacci\":{\"ratios\":[0.382,0.500,0.618],\"absolute_tolerance\":" << f.absolutePriceTolerance
        << ",\"anchor_policy\":" << static_cast<int>(f.anchorSelectionPolicy) << ",\"direction_policy\":" << static_cast<int>(f.directionalStudyPolicy)
        << ",\"confluence_policy\":" << static_cast<int>(f.confluencePolicy) << ",\"max_fractals\":" << f.maxConfirmedFractalsPerKind
        << ",\"ab_age\":" << f.maxABAgeBars << ",\"active_ab_limit\":" << f.maxActiveABStructures
        << ",\"active_observation_limit\":" << f.maxActiveConfluenceObservations
        << ",\"retained_observation_limit\":" << f.maxRetainedConfluenceObservations << "}\n}\n";
    return out.str();
}
}
std::string FileHash(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot hash provenance/artifact file: " + path.string());
    Sha256 hash; std::array<char, 16384> buffer{};
    while (file)
    {
        file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        hash.Add(std::string_view(buffer.data(), static_cast<std::size_t>(file.gcount())));
    }
    if (!file.eof()) throw std::runtime_error("artifact hash read failure");
    return hash.Digest();
}
struct CompressedCsv::State
{
    std::ofstream file;
    z_stream codec{};
    gz_header header{};
    std::array<unsigned char, kBufferBytes> buffer{};
    Sha256 hash;
    std::size_t bytes = 0, newlines = 0;
    bool initialized = false, finished = false, failed = false;
    CompressionHook hook;
    State(const std::filesystem::path& path, CompressionHook callback) : hook(std::move(callback))
    {
        file.open(path, std::ios::binary); Configure(file);
        if (deflateInit2(&codec, 6, Z_DEFLATED, 31, 8, Z_DEFAULT_STRATEGY) != Z_OK)
            throw std::runtime_error("gzip initialization failed");
        initialized = true;
        header.time = 0; header.os = 255; // No filename, comment or variable timestamp.
        if (deflateSetHeader(&codec, &header) != Z_OK)
        { deflateEnd(&codec); initialized = false; throw std::runtime_error("gzip fixed header failed"); }
    }
    ~State() { if (initialized) deflateEnd(&codec); }
    int Pump(int flush)
    {
        codec.next_out = buffer.data(); codec.avail_out = static_cast<uInt>(buffer.size());
        if (hook) hook(flush == Z_FINISH ? CompressionOperation::Finalize : CompressionOperation::Compress);
        const int result = deflate(&codec, flush);
        // With NO_FLUSH, draining an exactly full output buffer can leave no
        // pending bytes. zlib reports that harmless no-progress condition as
        // Z_BUF_ERROR; all codec errors and finalization errors remain fatal.
        if (result != Z_OK && result != Z_STREAM_END &&
            !(result == Z_BUF_ERROR && flush == Z_NO_FLUSH && !codec.avail_in && codec.avail_out == buffer.size()))
            throw std::runtime_error("gzip compression/finalization failed");
        const auto count = buffer.size() - codec.avail_out;
        if (count)
        {
            if (hook) hook(CompressionOperation::Write);
            file.write(reinterpret_cast<const char*>(buffer.data()), static_cast<std::streamsize>(count));
        }
        return result;
    }
};
CompressedCsv::CompressedCsv(const std::filesystem::path& path, CompressionHook hook)
    : state_(std::make_unique<State>(path, std::move(hook))) {}
CompressedCsv::~CompressedCsv() = default;
void CompressedCsv::Append(std::string_view bytes)
{
    auto& s = *state_;
    if (s.finished || s.failed) throw std::logic_error("closed or failed gzip stream");
    try
    {
        s.hash.Add(bytes); s.bytes += bytes.size(); s.newlines += std::count(bytes.begin(), bytes.end(), '\n');
        while (!bytes.empty())
        {
            const auto count = std::min(bytes.size(), kBufferBytes);
            s.codec.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(bytes.data()));
            s.codec.avail_in = static_cast<uInt>(count);
            do { s.Pump(Z_NO_FLUSH); } while (s.codec.avail_in || !s.codec.avail_out);
            bytes.remove_prefix(count);
        }
    }
    catch (...) { s.failed = true; throw; }
}
DecodedContent CompressedCsv::Finish()
{
    auto& s = *state_;
    if (s.finished || s.failed) throw std::logic_error("closed or failed gzip stream");
    try
    {
        s.codec.next_in = nullptr; s.codec.avail_in = 0;
        while (s.Pump(Z_FINISH) != Z_STREAM_END) {}
        if (deflateEnd(&s.codec) != Z_OK) throw std::runtime_error("gzip finalization cleanup failed");
        s.initialized = false; s.file.close(); s.finished = true;
        return {s.hash.Digest(), s.bytes, s.newlines ? s.newlines - 1 : 0};
    }
    catch (...) { s.failed = true; throw; }
}
DecodedContent InspectGzip(const std::filesystem::path& path, const std::function<void(std::string_view)>& consumer)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot read compressed artifact");
    z_stream codec{};
    if (inflateInit2(&codec, 31) != Z_OK) throw std::runtime_error("gzip validator initialization failed");
    struct Guard { z_stream& codec; ~Guard() { inflateEnd(&codec); } } guard{codec};
    std::array<unsigned char, CompressedCsv::kBufferBytes> input{}, output{};
    Sha256 hash; std::size_t bytes = 0, newlines = 0; bool ended = false;
    while (!ended)
    {
        if (!codec.avail_in)
        {
            file.read(reinterpret_cast<char*>(input.data()), static_cast<std::streamsize>(input.size()));
            if (file.bad() || (!file.eof() && file.fail())) throw std::runtime_error("gzip validator read failed");
            if (!file.gcount()) throw std::runtime_error("truncated gzip artifact");
            codec.next_in = input.data(); codec.avail_in = static_cast<uInt>(file.gcount());
        }
        codec.next_out = output.data(); codec.avail_out = static_cast<uInt>(output.size());
        const int result = inflate(&codec, Z_NO_FLUSH);
        if (result != Z_OK && result != Z_STREAM_END) throw std::runtime_error("corrupt gzip artifact");
        const std::string_view chunk(reinterpret_cast<const char*>(output.data()), output.size() - codec.avail_out);
        hash.Add(chunk); bytes += chunk.size(); newlines += std::count(chunk.begin(), chunk.end(), '\n');
        if (consumer) consumer(chunk);
        ended = result == Z_STREAM_END;
    }
    if (codec.avail_in || file.peek() != std::char_traits<char>::eof() || file.bad())
        throw std::runtime_error("trailing data or concatenated gzip member");
    return {hash.Digest(), bytes, newlines ? newlines - 1 : 0};
}
namespace
{
std::string ReadText(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("missing publication metadata");
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
}
void ValidateArtifactPublication(const std::filesystem::path& directory)
{
    const auto checksumHash = FileHash(directory / "checksums.sha256");
    const auto marker = ReadText(directory / "COMPLETE");
    if (marker.find("processing=COMPLETE\n") == std::string::npos ||
        marker.find("checksums_sha256=" + checksumHash + "\n") == std::string::npos)
        throw std::runtime_error("completion/checksum chain mismatch");
    std::map<std::string, std::string> files;
    std::istringstream checksums(ReadText(directory / "checksums.sha256")); std::string line;
    while (std::getline(checksums, line))
    {
        if (line.size() < 67 || line.substr(64, 2) != "  ") throw std::runtime_error("invalid checksum row");
        const auto hash = line.substr(0, 64), name = line.substr(66);
        if (!IsHash(hash) || name.empty() || name.find_first_of("/\\") != std::string::npos || name == "." || name == ".." ||
            !files.emplace(name, hash).second || std::filesystem::is_symlink(directory / name) || FileHash(directory / name) != hash)
            throw std::runtime_error("artifact checksum mismatch or unsafe filename");
    }
    for (const char* required : {"decisions.csv", "outcomes.csv", "protocol.json", "provenance_manifest.json", "coverage.csv", "exclusions.csv"})
        if (!files.contains(required)) throw std::runtime_error("missing required artifact checksum");
    const auto protocolText = ReadText(directory / "protocol.json");
    if (protocolText.find("\"schema\":\"" + std::string(kSchema) + "\"") == std::string::npos)
        throw std::runtime_error("unsupported semantic protocol");
    if (!files.contains("compression_manifest.csv"))
    {
        if (protocolText.find("\"artifact_format\"") != std::string::npos ||
            !files.contains("geometry_measurements.csv") || !files.contains("interaction_episodes.csv"))
            throw std::runtime_error("missing legacy v2 diagnostics");
        return; // Legacy publications remain immutable and validatable.
    }
    const bool summary = protocolText.find("\"episode_publication\":\"summary\"") != std::string::npos;
    const bool full = protocolText.find("\"episode_publication\":\"full\"") != std::string::npos;
    if (protocolText.find("\"artifact_format\":\"" + std::string(kArtifactFormat) + "\"") == std::string::npos ||
        summary == full || files.contains("interaction_episodes.csv.gz") != full ||
        files.contains("interaction_episodes.csv") || files.contains("geometry_measurements.csv") ||
        !files.contains("geometry_measurements.csv.gz") || !files.contains("episode_summary.csv"))
        throw std::runtime_error("missing revised diagnostics");
    std::istringstream metadata(ReadText(directory / "compression_manifest.csv"));
    std::getline(metadata, line);
    if (line != "filename,codec,level,window_bits,mem_level,strategy,mtime,os,zlib_version,compressed_sha256,decoded_sha256,decoded_bytes,decoded_rows")
        throw std::runtime_error("invalid compression metadata schema");
    std::set<std::string> compressed;
    while (std::getline(metadata, line))
    {
        std::vector<std::string> cells; std::istringstream row(line); std::string cell;
        while (std::getline(row, cell, ',')) cells.push_back(cell);
        if (cells.size() != 13 || (cells[0] != "geometry_measurements.csv.gz" && cells[0] != "interaction_episodes.csv.gz") ||
            cells[1] != "gzip" || cells[2] != "6" || cells[3] != "31" || cells[4] != "8" || cells[5] != "default" ||
            cells[6] != "0" || cells[7] != "255" || cells[8].empty() || !IsHash(cells[10]) ||
            !files.contains(cells[0]) || !compressed.insert(cells[0]).second ||
            files.at(cells[0]) != cells[9]) throw std::runtime_error("invalid compression hash metadata");
        const auto content = InspectGzip(directory / cells[0]);
        if (content.sha256 != cells[10] || std::to_string(content.bytes) != cells[11] || std::to_string(content.rows) != cells[12])
            throw std::runtime_error("decoded artifact content mismatch");
    }
    if (!compressed.contains("geometry_measurements.csv.gz") ||
        files.contains("interaction_episodes.csv.gz") != compressed.contains("interaction_episodes.csv.gz"))
        throw std::runtime_error("incomplete compressed metadata");
}
ArtifactWriter::ArtifactWriter(std::filesystem::path directory, EpisodePublication tier, CompressionHook hook)
    : directory_(std::move(directory)), tier_(tier)
{
    if (tier_ != EpisodePublication::Summary && tier_ != EpisodePublication::Full)
        throw std::invalid_argument("unsupported episode publication tier");
    if (std::filesystem::exists(directory_) || !std::filesystem::create_directories(directory_))
        throw std::invalid_argument("output directory must be new");
    decisions_.open(directory_ / "decisions.csv"); outcomes_.open(directory_ / "outcomes.csv");
    geometry_ = std::make_unique<CompressedCsv>(directory_ / "geometry_measurements.csv.gz", hook);
    if (tier_ == EpisodePublication::Full)
        episodes_ = std::make_unique<CompressedCsv>(directory_ / "interaction_episodes.csv.gz", hook);
    gaps_.open(directory_ / "gaps.csv"); exclusions_.open(directory_ / "exclusions.csv");
    for (auto* file : {&decisions_, &outcomes_, &gaps_, &exclusions_}) Configure(*file);
    decisions_ << "decision_id,bar_start_utc_epoch,completion_utc_epoch,status,reason,cohort,close,atr,ab_count,line_count,pair_count,measurement_count,universe_complete,geometry_evaluated,fib_near,line_near,intersection,min_dF,min_dL,min_dS,near_fib_levels,near_lines,intersection_count,support_0382,support_0500,support_0618,utc_hour,utc_weekday,log_atr_close,return4_atr,return16_atr,return64_atr,log_atr_tr64,mean_ab_age,mean_line_age,mean_ab_range_atr,mean_line_slope_atr,new_ab_fraction,new_line_fraction,bars_since_gap\n";
    outcomes_ << "decision_id,status,reason,resolution_bar_start_utc_epoch,primary_atr_change,signed_log_return,mfe_atr,mae_atr,directional_hit,flat\n";
    const std::string header = "bar_start_utc_epoch,ab_natural_key,line_natural_key,ratio,fib_price,line_price,dF,dL,dS,cohort";
    geometry_->Append(header + "\n");
    if (episodes_) episodes_->Append(header + ",episode_sequence,role\n");
    gaps_ << "previous_utc_epoch,next_utc_epoch,gap_seconds\n";
    exclusions_ << "decision_id,bar_start_utc_epoch,stage,reason\n";
}
ArtifactWriter::~ArtifactWriter() = default;
Sinks ArtifactWriter::GetSinks()
{
    return {[this](const Decision& d) { Add(d); }, [this](const OutcomeRecord& o) { Add(o); },
        [this](const GeometryMeasurement& m) { Add(m); }, [this](const Episode& e) { Add(e); }, [this](const Gap& g) { Add(g); }};
}
void ArtifactWriter::Add(const Decision& d)
{
    const auto& a = d.aggregate; const auto& x = d.covariates;
    decisions_ << d.identity << ',' << d.timestamp << ',' << d.timestamp + 900 << ',' << d.status << ',' << d.reason << ',';
    if (d.cohort) decisions_ << Name(*d.cohort);
    decisions_ << ','; Cell(decisions_, d.close); decisions_ << ','; Cell(decisions_, d.atr);
    decisions_ << ',' << a.abs << ',' << a.lines << ',' << a.pairs << ',' << a.measurements << ',' << a.complete
        << ',' << a.evaluated << ',' << a.fibNear << ',' << a.lineNear << ',' << a.intersection << ',';
    Cell(decisions_, a.minDF); decisions_ << ','; Cell(decisions_, a.minDL); decisions_ << ','; Cell(decisions_, a.minDS);
    decisions_ << ',' << a.nearFibLevels << ',' << a.nearLines << ',' << a.intersections;
    for (bool flag : a.supportingRatios) decisions_ << ',' << flag;
    decisions_ << ',' << x.utcHour << ',' << x.utcWeekday << ',';
    Cell(decisions_, x.logAtrClose); decisions_ << ','; Cell(decisions_, x.return4); decisions_ << ',';
    Cell(decisions_, x.return16); decisions_ << ','; Cell(decisions_, x.return64); decisions_ << ','; Cell(decisions_, x.logAtrTR64);
    decisions_ << ',' << Number(x.meanABAge) << ',' << Number(x.meanLineAge) << ',' << Number(x.meanABRangeAtr)
        << ',' << Number(x.meanLineSlopeAtr) << ',' << Number(x.newABFraction) << ',' << Number(x.newLineFraction) << ',' << x.barsSinceGap << '\n';
    if (d.status != "classified") exclusions_ << d.identity << ',' << d.timestamp << ",decision," << d.reason << '\n';
}
void ArtifactWriter::Add(const OutcomeRecord& o)
{
    outcomes_ << o.decision << ',' << o.status << ',' << o.reason << ',';
    if (o.resolutionTimestamp) outcomes_ << *o.resolutionTimestamp;
    outcomes_ << ',';
    if (o.outcome)
    {
        const auto& v = *o.outcome;
        outcomes_ << Number(v.signedAtrChange) << ',' << Number(v.signedLogReturn) << ',' << Number(v.favorableAtr)
            << ',' << Number(v.adverseAtr) << ',' << v.directionalHit << ',' << v.flat;
    }
    else outcomes_ << ",,,,,";
    outcomes_ << '\n';
    if (o.status != "resolved") exclusions_ << o.decision << ",,outcome," << o.reason << '\n';
}
void ArtifactWriter::Add(const GeometryMeasurement& m)
{
    std::ostringstream row; row.imbue(std::locale::classic()); Measurement(row, m); row << '\n'; geometry_->Append(row.str());
}
void ArtifactWriter::Add(const Episode& e)
{
    if (!episodes_) return;
    std::ostringstream row; row.imbue(std::locale::classic()); Measurement(row, e.measurement);
    row << ',' << e.sequence << ",DIAGNOSTIC_ONLY\n"; episodes_->Append(row.str());
}
void ArtifactWriter::Add(const Gap& g) { gaps_ << g.previous << ',' << g.next << ',' << g.seconds << '\n'; }
void ArtifactWriter::HashArtifact(const std::string& filename) { hashes_[filename] = FileHash(directory_ / filename); }
void ArtifactWriter::Write(std::string_view filename, std::string_view text)
{
    std::ofstream out(directory_ / filename, std::ios::binary); Configure(out); out << text; out.close(); HashArtifact(std::string(filename));
}
void ArtifactWriter::Copy(std::string_view filename, const std::filesystem::path& source)
{
    std::filesystem::copy_file(source, directory_ / filename); HashArtifact(std::string(filename));
    if (FileHash(source) != hashes_.at(std::string(filename))) throw std::runtime_error("provenance copy mismatch");
}
void ArtifactWriter::Complete(const EventEngine& engine, const TG4::EvaluationConfiguration& detector, const Provenance& p)
{
    if (!engine.IsFinalized()) throw std::logic_error("finalize replay before publication");
    const auto& a = engine.GetAudit(); const auto& cfg = engine.GetProtocol(); const auto range = engine.Range();
    if (a.candles != p.preflightRows || p.gitCommit.size() != 40 || p.gitCommit.find_first_not_of("0123456789abcdef") != std::string::npos ||
        !IsHash(p.sourceSHA256) || !IsHash(p.executableSHA256) || !IsHash(p.configurationFileSHA256) ||
        p.compiler.empty() || p.libraryVersion.empty() || p.sourceSymbol != "audcadrmp" ||
        detector.geometry.atrPeriod != 14 || detector.fibonacci.retracementRatios != std::vector<double>(kRatios.begin(), kRatios.end()) ||
        detector.fibonacci.directionalStudyPolicy != TG3::DirectionalStudyPolicy::SourceUTLUpABOnly ||
        FileHash(p.sourceManifest) != p.sourceSHA256 || FileHash(p.executable) != p.executableSHA256 ||
        FileHash(p.configurationFile) != p.configurationFileSHA256)
        throw std::runtime_error("incomplete or inconsistent provenance/source row-count contract");
    for (auto* file : {&decisions_, &outcomes_, &gaps_, &exclusions_}) file->close();
    decoded_["geometry_measurements.csv.gz"] = geometry_->Finish();
    if (episodes_) decoded_["interaction_episodes.csv.gz"] = episodes_->Finish();
    const auto observedEpisodes = a.episodeRecords[0][0] + a.episodeRecords[0][1] + a.episodeRecords[0][2];
    if (decoded_.at("geometry_measurements.csv.gz").rows != a.scheduledMeasurements ||
        (episodes_ && decoded_.at("interaction_episodes.csv.gz").rows != observedEpisodes))
        throw std::runtime_error("diagnostic publication row-count reconciliation failed");
    for (const char* filename : {"decisions.csv", "outcomes.csv", "gaps.csv", "exclusions.csv"}) HashArtifact(filename);
    std::ostringstream compression;
    compression << "filename,codec,level,window_bits,mem_level,strategy,mtime,os,zlib_version,compressed_sha256,decoded_sha256,decoded_bytes,decoded_rows\n";
    for (const auto& [filename, content] : decoded_)
    {
        HashArtifact(filename);
        const auto verified = InspectGzip(directory_ / filename);
        if (verified.sha256 != content.sha256 || verified.bytes != content.bytes || verified.rows != content.rows)
            throw std::runtime_error("compressed artifact verification failed");
        compression << filename << ",gzip,6,31,8,default,0,255," << zlibVersion() << ',' << hashes_.at(filename)
            << ',' << content.sha256 << ',' << content.bytes << ',' << content.rows << '\n';
    }
    Write("compression_manifest.csv", compression.str());
    const std::array phases{"full_replay", "prefix", "scoring"};
    const std::array events{"initialization", "union_entry", "complement_entry"};
    std::ostringstream episodes;
    episodes << "phase,metric,value,status\n";
    for (std::size_t phase = 0; phase < phases.size(); ++phase)
    {
        for (std::size_t event = 0; event < events.size(); ++event)
            episodes << phases[phase] << ',' << events[event] << ',' << a.episodeRecords[phase][event] << ",OBSERVED\n";
        const std::size_t peak = phase == 0 ? a.episodeLedger.full : (phase == 1 ? a.episodeLedger.prefix : a.episodeLedger.scoring);
        episodes << phases[phase] << ",tracked_state_peak," << peak << ",POST_UPDATE_RETAINED\n"
            << phases[phase] << ",skipped_observation_opportunities," << a.skippedEpisodeOpportunities[phase] << ",OBSERVED\n";
    }
    episodes << "full_replay,tracking_complete," << (a.diagnosticsComplete && a.geometryComplete) << ",OBSERVED\n"
        << "full_replay,episode_saturated," << a.firstEpisodeSaturationBar.has_value() << ",OBSERVED\n"
        << "full_replay,publication_detail_omitted," << (tier_ == EpisodePublication::Summary) << ",INTENTIONAL_PUBLICATION_POLICY\n"
        << "full_replay,first_saturation_bar_zero_based,";
    if (a.firstEpisodeSaturationBar) episodes << *a.firstEpisodeSaturationBar;
    episodes << ',' << (a.firstEpisodeSaturationBar ? "OBSERVED" : "NOT_APPLICABLE") << "\nfull_replay,first_saturation_utc_epoch,";
    if (a.firstEpisodeSaturationTimestamp) episodes << *a.firstEpisodeSaturationTimestamp;
    episodes << ',' << (a.firstEpisodeSaturationTimestamp ? "OBSERVED" : "NOT_APPLICABLE") << '\n';
    Write("episode_summary.csv", episodes.str());
    Copy("source_manifest.sha256", p.sourceManifest); Copy("source_snapshot.tar.gz", p.sourceArchive);
    Copy("detector_source_configuration.conf", p.configurationFile);
    const std::string detectorJson = DetectorConfiguration(detector);
    Write("detector_configuration.json", detectorJson);
    const std::string effective = engine.CanonicalConfiguration() + "\nexecution_line_batch_size="
        + std::to_string(cfg.lineBatchSize) + "\n" + detectorJson;
    const std::string effectiveHash = Hash(effective);
    Write("effective_configuration.txt", effective);
    std::ostringstream coverage;
    coverage << "metric,count,status\nsource_rows," << a.candles << ",OBSERVED\nprefix_rows," << a.prefixCandles
        << ",OBSERVED\nscored_range_rows," << a.scoredRangeCandles << ",OBSERVED\nscheduled_slots," << a.scheduled
        << ",OBSERVED\nclassified_decisions," << a.classified << ",OBSERVED\nresolved_outcomes," << a.resolved
        << ",OBSERVED\ncensored_outcomes," << a.censored << ",OBSERVED\npeak_detector_identities," << a.peakIdentities
        << ",OBSERVED\npeak_episode_states," << a.peakEpisodes << ",OBSERVED\n";
    const std::array<std::pair<const char*, const HighWatermark*>, 6> peaks{{
        {"retained_ab_structures", &a.retainedAB}, {"retained_trendline_candidates", &a.retainedLines},
        {"eligible_geometry_pairs", &a.eligiblePairs}, {"evaluated_geometry_pairs", &a.evaluatedPairs},
        {"identity_ledger", &a.identityLedger}, {"episode_ledger", &a.episodeLedger}}};
    for (const auto& [name, peak] : peaks)
    {
        const char* meaning = std::string_view(name) == "eligible_geometry_pairs" ? "RETAINED_ELIGIBLE_PRE_PAIR_CAP_DEMAND" : "POST_UPDATE_RETAINED_OR_EVALUATED";
        coverage << "peak_full_replay_" << name << ',' << peak->full << ',' << meaning << '\n'
            << "peak_prefix_" << name << ',' << peak->prefix << ',' << meaning << '\n'
            << "peak_scoring_" << name << ',' << peak->scoring << ',' << meaning << '\n';
    }
    const std::array statusNames{"closed_market", "missing_open_session", "missing_slot_unverified", "warmup", "embargo",
        "no_geometry", "invalid_atr", "invalid_covariates", "partition_purged", "closed_horizon", "capacity_invalid", "calendar_conflict", "classified"};
    for (const char* name : statusNames)
    {
        const auto it = a.decisions.find(name);
        coverage << "decision_" << name << ',' << (it == a.decisions.end() ? 0 : it->second) << ','
            << (engine.Calendar().identity == "UNVERIFIED" && (std::string_view(name) == "closed_market" || std::string_view(name) == "missing_open_session") ? "UNVERIFIED" : "OBSERVED") << '\n';
    }
    for (const auto& [reason, count] : a.censoring) coverage << "censor_" << reason << ',' << count << ",OBSERVED\n";
    for (const auto& [reason, count] : a.diagnostics) coverage << "diagnostic_" << reason << ',' << count << ",OBSERVED\n";
    for (int year = 2010; year < (cfg.partition == Partition::Development ? 2020 : 2025); ++year)
    {
        const auto it = a.sourceYears.find(std::to_string(year));
        coverage << "source_year_" << year << ',' << (it == a.sourceYears.end() ? 0 : it->second) << ",OBSERVED\n";
    }
    Write("coverage.csv", coverage.str());
    std::ostringstream protocol;
    protocol << "{\n\"artifact_format\":" << Json(kArtifactFormat) << ",\n\"episode_publication\":" << Json(Name(tier_)) << ",\n\"episode_detail_omitted\":" << (tier_ == EpisodePublication::Summary ? "true" : "false") << ",\n\"schema\":" << Json(kSchema) << ",\n\"protocol_fingerprint\":" << Json(engine.Fingerprint())
        << ",\n\"effective_configuration_sha256\":" << Json(effectiveHash)
        << ",\n\"symbol\":\"AUDCAD\",\n\"source_symbol\":\"audcadrmp\",\n\"timeframe\":\"15m\",\n"
        << "\"replay_origin_utc_epoch\":" << kOrigin << ",\n\"partition\":" << Json(Name(cfg.partition))
        << ",\n\"score_start_utc_epoch\":" << range.start << ",\n\"score_end_exclusive_utc_epoch\":" << range.end
        << ",\n\"warmup_completed_candles\":" << cfg.warmupBars << ",\n\"calibration_embargo_completed_candles\":4,\n"
        << "\"timestamp_convention\":\"canonical New York civil bar starts converted to UTC; partitions use UTC bar-start years; decisions at bar completion\",\n"
        << "\"decision_clock\":\"UTC bar-start minute 45; missing slots recorded, never replaced\",\n"
        << "\"aggregation\":\"complete commutative UpAB/UTL natural geometry universe; one primary row per clock slot\",\n"
        << "\"ratios\":[0.382,0.500,0.618],\n\"market_atr_threshold\":0.25,\n\"structural_atr_threshold\":0.25,\n"
        << "\"cohorts\":{\"A\":\"any_fib_near AND NOT any_line_near\",\"B\":\"any_line_near AND NOT any_fib_near\","
        << "\"C\":\"any same AB/line/ratio with dF,dL,dS<=0.25\",\"D\":\"neither near, descriptive, not exchangeable control\","
        << "\"F_stratum\":\"both near without qualifying same-pair intersection\"},\n"
        << "\"missing_geometry\":\"EXCLUDED, never D\",\n\"horizon_completed_candles\":4,\n"
        << "\"primary_outcome\":\"(Close[t+4]-Close[t])/ATR[t], sign +1\",\n"
        << "\"secondary_outcomes\":[\"signed_log_return\",\"MFE_ATR\",\"MAE_ATR\",\"directional_hit\",\"exact_flat_flag\"],\n"
        << "\"gap_policy\":\"known calendar closure in forward horizon excluded prospectively; otherwise any adjacent delta !=900 seconds censors; no fabrication, no clock reset, no detector reset\",\n"
        << "\"purge_policy\":\"t+4 bar start must be strictly before score end; pending outcomes never cross partitions\",\n"
        << "\"calendar_status\":" << Json(engine.Calendar().identity) << ",\n\"calendar_sha256\":" << Json(engine.Calendar().sha256)
        << ",\n\"coverage_status\":\"UNVERIFIED: counts are observed coverage, not proof of complete historical/session coverage\",\n"
        << "\"geometry_complete\":" << (a.geometryComplete ? "true" : "false")
        << ",\n\"diagnostic_episodes_complete\":" << (a.diagnosticsComplete ? "true" : "false")
        << ",\n\"episode_role\":\"diagnostic first union/complement entry at each completed candle; one outside candle rearms; no primary overlap locks\",\n"
        << "\"capacity\":{\"max_pairs_per_candle\":" << cfg.maxPairsPerCandle << ",\"max_detector_identities\":" << cfg.maxDetectorIdentities
        << ",\"max_episode_states\":" << cfg.maxEpisodeStates << ",\"line_batch_size\":" << cfg.lineBatchSize
        << ",\"policy\":\"primary incompleteness and source AB/line capacity pruning prevent COMPLETE; diagnostic ledger saturation reported separately; no research active identity eviction\"},\n"
        << "\"statistical_estimation\":\"NOT_EXECUTED\",\n\"inferential_validity\":\"NOT_ESTABLISHED\",\n"
        << "\"primary_contrast\":\"C versus B adjusted predictive association, not a causal intervention effect\",\n"
        << "\"covariates_exported\":[\"UTC_hour_weekday\",\"min_trendline_distance\",\"log_ATR_Close\",\"return4_16_64_ATR\",\"log_ATR_TR64\","
        << "\"AB_line_multiplicity\",\"mean_AB_line_age\",\"mean_AB_range_ATR\",\"mean_line_slope_ATR\",\"new_structure_fractions\",\"bars_since_gap\"],\n"
        << "\"covariates_not_computed\":[\"previous_trendline_near_state\",\"time_since_trendline_near\",\"richer_age_distributions\"],\n"
        << "\"support_model\":{\"status\":\"NOT_EXECUTED\",\"fit_partition\":\"development only\",\"interval\":[0.10,0.90],"
        << "\"freeze\":\"feature basis, fitting procedure and coefficients before calibration\"},\n"
        << "\"uncertainty\":{\"status\":\"NOT_EXECUTED\",\"method\":\"year-stratified 28-day moving-block bootstrap\",\"replicates\":10000,"
        << "\"confidence\":0.95,\"sensitivity_days\":56,\"independence\":\"not implied by disjoint outcome windows or geometry IDs\"},\n"
        << "\"adequacy\":{\"status\":\"NOT_EXECUTED\",\"minimum_B\":500,\"minimum_C\":500,\"minimum_28_day_coverage_blocks\":30,"
        << "\"minimum_B_C_per_block\":5,\"independent_blocks\":\"UNVERIFIED\",\"planning_effect_ATR\":0.10,\"power_target\":0.80},\n"
        << "\"multiplicity\":\"one primary contrast/outcome; Holm adjustment for predeclared secondary family; alternative ratios/phases exploratory\",\n"
        << "\"negative_controls\":{\"status\":\"NOT_EXECUTED\",\"specification\":\"outcome-blind matched-cardinality sham ratios, dependence-preserving block outcome reassignment, other fixed clock phases; freeze exact definitions before calibration\"},\n"
        << "\"final_2025_disabled\":true,\n\"partitions_provisional\":true,\n"
        << "\"row_order\":\"decisions UTC clock order; outcomes terminal chronological order; geometries natural AB/line/ratio order; episodes completed-candle then natural geometry order\"\n}\n";
    Write("protocol.json", protocol.str());
    std::ostringstream manifest;
    manifest << "{\n\"artifact_format\":" << Json(kArtifactFormat)
        << ",\n\"episode_publication\":" << Json(Name(tier_))
        << ",\n\"diagnostic_tracking_complete\":" << (a.diagnosticsComplete && a.geometryComplete ? "true" : "false")
        << ",\n\"episode_detail_omitted\":" << (tier_ == EpisodePublication::Summary ? "true" : "false")
        << ",\n\"compression\":{\"codec\":\"gzip\",\"level\":6,\"window_bits\":31,\"mem_level\":8,\"strategy\":\"default\",\"mtime\":0,\"os\":255,\"zlib_version\":" << Json(zlibVersion()) << ",\"header_version\":" << Json(ZLIB_VERSION)
        << ",\"content_manifest\":\"compression_manifest.csv\",\"buffer_bytes\":65536},\n"
        << "\"capacity_high_watermarks\":{\"artifact\":\"coverage.csv\",\"phases\":[\"full_replay\",\"prefix\",\"scoring\"],\"retained_counts\":\"post-update raw detector vectors, all directions and aliases; not upstream pre-capacity demand\",\"eligible_pairs\":\"natural deduplicated retained eligible universe before research pair cap; unavailable when input capacity prevents canonicalization\"},\n"
        << "\"compressed_artifacts\":{";
    bool firstContent = true;
    for (const auto& [filename, content] : decoded_)
    {
        if (!firstContent) manifest << ',';
        firstContent = false;
        manifest << Json(filename) << ":{\"compressed_sha256\":" << Json(hashes_.at(filename))
            << ",\"decoded_sha256\":" << Json(content.sha256) << ",\"decoded_bytes\":" << content.bytes
            << ",\"decoded_rows\":" << content.rows << '}';
    }
    manifest << "},\n\"capacity_high_watermark_values\":{";
    bool firstPeak = true;
    for (const auto& [name, peak] : peaks)
    {
        if (!firstPeak) manifest << ',';
        firstPeak = false;
        manifest << Json(name) << ":{\"full_replay\":" << peak->full << ",\"prefix\":" << peak->prefix
            << ",\"scoring\":" << peak->scoring << '}';
    }
    manifest << "},\n\"episode_summary_artifact\":\"episode_summary.csv\",\n"
        << "\"schema\":\"fib-trendline-provenance-v2\",\n\"git_commit\":" << Json(p.gitCommit)
        << ",\n\"source_manifest_sha256\":" << Json(p.sourceSHA256) << ",\n\"source_archive_sha256\":" << Json(hashes_.at("source_snapshot.tar.gz"))
        << ",\n\"executable_sha256\":" << Json(p.executableSHA256) << ",\n\"compiler\":" << Json(p.compiler)
        << ",\n\"libpqxx_version\":" << Json(p.libraryVersion) << ",\n\"configuration_source_sha256\":" << Json(p.configurationFileSHA256)
        << ",\n\"effective_configuration_sha256\":" << Json(effectiveHash) << ",\n\"canonical_candle_content_sha256\":" << Json(engine.CandleDigest())
        << ",\n\"candle_hash_serialization\":\"UTC epoch,open,high,low,close; classic-locale binary64 max-digits decimal, signed zero normalized; newline per actual replay candle including prefix; no target or synthetic bars\",\n"
        << "\"source_rows\":" << a.candles << ",\n\"first_source_utc_epoch\":";
    if (a.firstTimestamp) manifest << *a.firstTimestamp; else manifest << "null";
    manifest << ",\n\"last_source_utc_epoch\":";
    if (a.lastTimestamp) manifest << *a.lastTimestamp; else manifest << "null";
    if (!p.candleSnapshotSHA256.empty())
    {
        if (!IsHash(p.candleSnapshotSHA256) || p.candleSnapshotEncoding.empty() || !p.candleSnapshotBytes)
            throw std::runtime_error("missing candle snapshot provenance");
        manifest << ",\n\"research_candle_snapshot\":{\"sha256\":" << Json(p.candleSnapshotSHA256)
            << ",\"encoding\":" << Json(p.candleSnapshotEncoding) << ",\"bytes\":" << p.candleSnapshotBytes
            << ",\"rows\":" << p.preflightRows
            << ",\"loading\":\"unchanged canonical loader; immutable bounded UTC/OHLC snapshot; transaction closed before evaluator\"}";
    }
    manifest << ",\n\"source_symbol\":\"audcadrmp\",\n\"source\":\"canonical candlestick loader, not 2025 reference view\",\n"
        << "\"database_access\":\"read_only_repeatable_read\",\n\"coverage\":\"UNVERIFIED\",\n\"statistical_estimation\":\"NOT_EXECUTED\"\n}\n";
    Write("provenance_manifest.json", manifest.str());
    std::string checksums;
    for (const auto& [file, hash] : hashes_) checksums += hash + "  " + file + "\n";
    Write("checksums.sha256", checksums);
    for (const auto& [file, hash] : hashes_)
        if (FileHash(directory_ / file) != hash) throw std::runtime_error("checksum verification failed before publication");
    if (!a.geometryComplete || !a.calendarConsistent)
    { Write("INVALID", "processing finished; primary geometry/calendar invalid; no inferential completion\n"); throw std::runtime_error("invalid primary study; COMPLETE prohibited"); }
    const std::string marker = "schema=" + std::string(kSchema) + "\nartifact_format=" + std::string(kArtifactFormat) + "\nprocessing=COMPLETE\nstatistical_estimation=NOT_EXECUTED\ninferential_validity=NOT_ESTABLISHED\ncoverage=UNVERIFIED\nchecksums_sha256=" + hashes_.at("checksums.sha256") + "\n";
    Write("COMPLETE.pending", marker);
    std::filesystem::rename(directory_ / "COMPLETE.pending", directory_ / "COMPLETE");
}
} // namespace EA::FibonacciResearch::TrendlineHistorical
