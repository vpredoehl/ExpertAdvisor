#include "CausalFibonacciTrendlineHistoricalArtifact.hpp"

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
ArtifactWriter::ArtifactWriter(std::filesystem::path directory) : directory_(std::move(directory))
{
    if (std::filesystem::exists(directory_) || !std::filesystem::create_directories(directory_))
        throw std::invalid_argument("output directory must be new");
    decisions_.open(directory_ / "decisions.csv"); outcomes_.open(directory_ / "outcomes.csv");
    geometry_.open(directory_ / "geometry_measurements.csv"); episodes_.open(directory_ / "interaction_episodes.csv");
    gaps_.open(directory_ / "gaps.csv"); exclusions_.open(directory_ / "exclusions.csv");
    for (auto* file : {&decisions_, &outcomes_, &geometry_, &episodes_, &gaps_, &exclusions_}) Configure(*file);
    decisions_ << "decision_id,bar_start_utc_epoch,completion_utc_epoch,status,reason,cohort,close,atr,ab_count,line_count,pair_count,measurement_count,universe_complete,geometry_evaluated,fib_near,line_near,intersection,min_dF,min_dL,min_dS,near_fib_levels,near_lines,intersection_count,support_0382,support_0500,support_0618,utc_hour,utc_weekday,log_atr_close,return4_atr,return16_atr,return64_atr,log_atr_tr64,mean_ab_age,mean_line_age,mean_ab_range_atr,mean_line_slope_atr,new_ab_fraction,new_line_fraction,bars_since_gap\n";
    outcomes_ << "decision_id,status,reason,resolution_bar_start_utc_epoch,primary_atr_change,signed_log_return,mfe_atr,mae_atr,directional_hit,flat\n";
    const std::string header = "bar_start_utc_epoch,ab_natural_key,line_natural_key,ratio,fib_price,line_price,dF,dL,dS,cohort";
    geometry_ << header << '\n'; episodes_ << header << ",episode_sequence,role\n";
    gaps_ << "previous_utc_epoch,next_utc_epoch,gap_seconds\n";
    exclusions_ << "decision_id,bar_start_utc_epoch,stage,reason\n";
}
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
void ArtifactWriter::Add(const GeometryMeasurement& m) { Measurement(geometry_, m); geometry_ << '\n'; }
void ArtifactWriter::Add(const Episode& e) { Measurement(episodes_, e.measurement); episodes_ << ',' << e.sequence << ",DIAGNOSTIC_ONLY\n"; }
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
    for (auto* file : {&decisions_, &outcomes_, &geometry_, &episodes_, &gaps_, &exclusions_}) file->close();
    for (const char* filename : {"decisions.csv", "outcomes.csv", "geometry_measurements.csv", "interaction_episodes.csv", "gaps.csv", "exclusions.csv"}) HashArtifact(filename);
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
    protocol << "{\n\"schema\":" << Json(kSchema) << ",\n\"protocol_fingerprint\":" << Json(engine.Fingerprint())
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
    manifest << "{\n\"schema\":\"fib-trendline-provenance-v2\",\n\"git_commit\":" << Json(p.gitCommit)
        << ",\n\"source_manifest_sha256\":" << Json(p.sourceSHA256) << ",\n\"source_archive_sha256\":" << Json(hashes_.at("source_snapshot.tar.gz"))
        << ",\n\"executable_sha256\":" << Json(p.executableSHA256) << ",\n\"compiler\":" << Json(p.compiler)
        << ",\n\"libpqxx_version\":" << Json(p.libraryVersion) << ",\n\"configuration_source_sha256\":" << Json(p.configurationFileSHA256)
        << ",\n\"effective_configuration_sha256\":" << Json(effectiveHash) << ",\n\"canonical_candle_content_sha256\":" << Json(engine.CandleDigest())
        << ",\n\"candle_hash_serialization\":\"UTC epoch,open,high,low,close; classic-locale binary64 max-digits decimal, signed zero normalized; newline per actual replay candle including prefix; no target or synthetic bars\",\n"
        << "\"source_rows\":" << a.candles << ",\n\"first_source_utc_epoch\":";
    if (a.firstTimestamp) manifest << *a.firstTimestamp; else manifest << "null";
    manifest << ",\n\"last_source_utc_epoch\":";
    if (a.lastTimestamp) manifest << *a.lastTimestamp; else manifest << "null";
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
    const std::string marker = "schema=" + std::string(kSchema) + "\nprocessing=COMPLETE\nstatistical_estimation=NOT_EXECUTED\ninferential_validity=NOT_ESTABLISHED\ncoverage=UNVERIFIED\nchecksums_sha256=" + hashes_.at("checksums.sha256") + "\n";
    Write("COMPLETE.pending", marker);
    std::filesystem::rename(directory_ / "COMPLETE.pending", directory_ / "COMPLETE");
}
} // namespace EA::FibonacciResearch::TrendlineHistorical
