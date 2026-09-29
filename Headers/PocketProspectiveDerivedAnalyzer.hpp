#pragma once

// File-only corrective reporting for the sealed Phase Pocket 3
// preconfirmation artifact.  This deliberately has no market-data, pqxx, or
// detector dependency: it verifies the immutable source before it parses a
// single observation row and never recreates observations.

#include "PocketProspectiveEvaluator.hpp"

#include <charconv>
#include <chrono>
#include <unordered_map>

namespace EA::Pocket::Prospective::Derived
{
inline constexpr std::string_view kAnalyzerSchema =
    "phase-pocket-4-derived-preconfirmation-report-v2";
inline constexpr std::string_view kAnalyzerSchemaV1 =
    "phase-pocket-4-derived-preconfirmation-report-v1";
inline constexpr std::size_t kBootstrapReplicates = 2000;

struct Row final
{
    std::string identity, symbol, partition, direction, censor, race;
    std::size_t lookback = 0, confirmationBar = 0, horizon = 0, validFutureBars = 0;
    std::int64_t confirmationTimestamp = 0;
    double lower = 0, upper = 0, mfe = 0, mae = 0, directionalReturn = 0;
    bool complete = false;
    std::optional<std::size_t> touchAt, closeAt;
};

inline std::vector<std::string> SplitCsvExact(std::string_view line)
{
    std::vector<std::string> out; std::size_t start = 0;
    for (;;) { const auto comma = line.find(',', start); out.emplace_back(line.substr(start, comma - start));
        if (comma == std::string_view::npos) return out; start = comma + 1; }
}
inline std::size_t Unsigned(std::string_view value, std::string_view error)
{
    std::size_t result{}; const auto [at, code] = std::from_chars(value.data(), value.data() + value.size(), result);
    if (code != std::errc{} || at != value.data() + value.size()) throw std::invalid_argument(std::string(error));
    return result;
}
inline std::int64_t Signed(std::string_view value, std::string_view error)
{
    std::int64_t result{}; const auto [at, code] = std::from_chars(value.data(), value.data() + value.size(), result);
    if (code != std::errc{} || at != value.data() + value.size()) throw std::invalid_argument(std::string(error));
    return result;
}
inline double Finite(std::string_view value, std::string_view error)
{
    std::string copy(value); char* at = nullptr; const double result = std::strtod(copy.c_str(), &at);
    if (!at || *at || !std::isfinite(result)) throw std::invalid_argument(std::string(error)); return result;
}
inline std::optional<std::size_t> OptionalUnsigned(std::string_view value)
{ return value.empty() ? std::nullopt : std::optional<std::size_t>(Unsigned(value, "POCKET_DERIVED_INVALID_OPTIONAL_INTEGER")); }

inline std::size_t ConfirmationBarFromIdentity(const std::string& identity, const RunConfiguration& config,
    const Row& row)
{
    const auto parts = SplitCsvExact(std::string_view(identity)); // identities cannot contain comma; use colon below.
    (void)parts;
    std::vector<std::string> fields; std::size_t start = 0;
    for (;;) { const auto colon = identity.find(':', start); fields.push_back(identity.substr(start, colon - start));
        if (colon == std::string::npos) break; start = colon + 1; }
    if (fields.size() != 8 || fields[0] != config.study || fields[1] != config.configurationSha256 ||
        fields[2] != row.symbol || fields[3] != "15m" || Unsigned(fields[4], "POCKET_DERIVED_IDENTITY") != row.lookback ||
        Unsigned(fields[7], "POCKET_DERIVED_IDENTITY") != static_cast<std::size_t>(row.confirmationTimestamp))
        throw std::invalid_argument("POCKET_DERIVED_IDENTITY_MISMATCH");
    (void)Unsigned(fields[5], "POCKET_DERIVED_IDENTITY");
    return Unsigned(fields[6], "POCKET_DERIVED_IDENTITY");
}

inline std::vector<Row> ReadVerifiedRowsContract(const std::filesystem::path& source,
    const RunConfiguration& config, std::string_view configurationText,
    std::string_view primarySchema)
{
    // This call intentionally precedes ReadTextFile(observations.csv).
    ImmutableArtifactWriter::VerifyDirectoryContract(source, config, configurationText, primarySchema);
    // The original verifier establishes the content hashes.  Reject surplus or
    // malformed manifest material too: there is no self-hash for a manifest,
    // so accepting appended text would make its provenance ambiguous.
    const std::string sourceManifest = ReadTextFile(source / "manifest.txt");
    const std::vector<std::string_view> manifestKeys = primarySchema.empty()
        ? std::vector<std::string_view>{{"study", "protocol", "protocol_document_sha256", "detector", "detector_baseline", "configuration_sha256", "configuration_file_sha256", "observations_sha256", "aggregates_sha256", "provenance"}}
        : std::vector<std::string_view>{{"study", "protocol", "protocol_document_sha256", "detector", "detector_baseline", "configuration_sha256", "configuration_file_sha256", "observations_sha256", "aggregates_sha256", "artifact_schema", "provenance"}};
    std::istringstream manifestInput(sourceManifest); std::string manifestLine; std::size_t manifestIndex=0;
    while (std::getline(manifestInput, manifestLine)) {
        if (manifestIndex >= manifestKeys.size() || manifestLine.rfind(std::string(manifestKeys[manifestIndex])+"=",0)!=0)
            throw std::invalid_argument("POCKET_DERIVED_SOURCE_MANIFEST_NONCANONICAL");
        ++manifestIndex;
    }
    if (manifestIndex != manifestKeys.size()) throw std::invalid_argument("POCKET_DERIVED_SOURCE_MANIFEST_NONCANONICAL");
    if (config.protocol != kProtocolId || config.detector != kDetectorId ||
        config.partitions.front().end > config.resolutionEnd)
        throw std::invalid_argument("POCKET_DERIVED_SOURCE_NOT_FROZEN_PRECONFIRMATION");
    const std::string content = ReadTextFile(source / "observations.csv");
    std::istringstream input(content); std::string line;
    const std::string header = "identity,symbol,partition,lookback,direction,event_timestamp,confirmation_timestamp,lower,upper,horizon,complete,censor,valid_future_bars,touch_at,close_at,mfe,mae,directional_close_return,race";
    if (!std::getline(input, line) || line != header) throw std::invalid_argument("POCKET_DERIVED_OBSERVATION_SCHEMA");
    std::vector<Row> rows;
    while (std::getline(input, line)) {
        if (line.empty()) continue; const auto f = SplitCsvExact(line);
        if (f.size() != 19) throw std::invalid_argument("POCKET_DERIVED_OBSERVATION_COLUMNS");
        Row row; row.identity=f[0]; row.symbol=f[1]; row.partition=f[2]; row.lookback=Unsigned(f[3], "POCKET_DERIVED_LOOKBACK");
        row.direction=f[4]; row.confirmationTimestamp=Signed(f[6], "POCKET_DERIVED_TIMESTAMP"); row.lower=Finite(f[7], "POCKET_DERIVED_NUMBER");
        row.upper=Finite(f[8], "POCKET_DERIVED_NUMBER"); row.horizon=Unsigned(f[9], "POCKET_DERIVED_HORIZON");
        if (f[10] == "true") row.complete=true; else if (f[10] != "false") throw std::invalid_argument("POCKET_DERIVED_COMPLETE");
        row.censor=f[11]; row.validFutureBars=Unsigned(f[12], "POCKET_DERIVED_FUTURE_BARS"); row.touchAt=OptionalUnsigned(f[13]); row.closeAt=OptionalUnsigned(f[14]);
        row.mfe=Finite(f[15], "POCKET_DERIVED_NUMBER"); row.mae=Finite(f[16], "POCKET_DERIVED_NUMBER"); row.directionalReturn=Finite(f[17], "POCKET_DERIVED_NUMBER"); row.race=f[18];
        if (std::find(kLookbacks.begin(), kLookbacks.end(), row.lookback) == kLookbacks.end() ||
            std::find(kHorizons.begin(), kHorizons.end(), row.horizon) == kHorizons.end() ||
            (row.direction != "bullish" && row.direction != "bearish") || !(row.upper > row.lower) ||
            row.confirmationTimestamp >= config.resolutionEnd || !PartitionFor(config, row.confirmationTimestamp) ||
            PartitionFor(config, row.confirmationTimestamp)->name != row.partition || row.mfe < 0 || row.mae < 0)
            throw std::invalid_argument("POCKET_DERIVED_ROW_OUTSIDE_FROZEN_CONTRACT");
        row.confirmationBar=ConfirmationBarFromIdentity(row.identity, config, row);
        rows.push_back(std::move(row));
    }
    if (rows.empty()) throw std::invalid_argument("POCKET_DERIVED_EMPTY_OBSERVATIONS");
    return rows;
}
inline std::vector<Row> ReadVerifiedRows(const std::filesystem::path& source)
{
    return ReadVerifiedRowsContract(source, FrozenConfiguration(), CanonicalConfigurationText(), {});
}

inline std::optional<double> Quantile(std::vector<double> values, std::size_t numerator)
{
    if (values.empty()) return std::nullopt; std::sort(values.begin(), values.end());
    return values[(values.size() - 1) * numerator / 100];
}
inline std::string Cell(std::optional<double> value) { return value ? CsvNumber(*value) : "undefined"; }

struct Statistics final
{
    std::size_t eligible=0, complete=0, censored=0, touch=0, close=0, raceContinuation=0, raceRevisit=0, raceSame=0, raceNeither=0, raceCensored=0;
    std::map<std::string, std::size_t> censorReasons;
    std::vector<double> touchTimes, closeTimes, mfe, mae, returns, mfePips, maePips, returnPips, mfeWidths, maeWidths, returnWidths;
};
inline Statistics Calculate(const std::vector<const Row*>& rows, double pip)
{
    Statistics s; s.eligible=rows.size();
    for (const Row* row : rows) {
        if (!row->complete) { ++s.censored; ++s.censorReasons[row->censor]; if (row->horizon == 64) ++s.raceCensored; continue; }
        ++s.complete; if (row->touchAt) { ++s.touch; s.touchTimes.push_back(*row->touchAt); } if (row->closeAt) { ++s.close; s.closeTimes.push_back(*row->closeAt); }
        const double width=row->upper-row->lower; s.mfe.push_back(row->mfe); s.mae.push_back(row->mae); s.returns.push_back(row->directionalReturn);
        s.mfePips.push_back(row->mfe/pip); s.maePips.push_back(row->mae/pip); s.returnPips.push_back(row->directionalReturn/pip);
        s.mfeWidths.push_back(row->mfe/width); s.maeWidths.push_back(row->mae/width); s.returnWidths.push_back(row->directionalReturn/width);
        if (row->horizon == 64) { const std::string& race=row->race;
            if (race=="continuation_first") ++s.raceContinuation; else if (race=="revisit_first") ++s.raceRevisit;
            else if (race=="same_bar_intrabar_order_indeterminate") ++s.raceSame; else if (race=="neither") ++s.raceNeither;
            else throw std::invalid_argument("POCKET_DERIVED_RACE_INVALID"); }
    }
    return s;
}
inline void WriteStatistics(std::ostringstream& out, std::string_view aggregation, std::string_view symbol,
    std::size_t lookback, std::string_view partition, std::string_view direction, std::size_t horizon, const Statistics& s)
{
    const auto rate = [&s](std::size_t count) -> std::optional<double> { return s.complete ? std::optional<double>(static_cast<double>(count)/s.complete) : std::nullopt; };
    const auto median64 = [horizon](const std::vector<double>& values, std::size_t complete, std::size_t resolved) -> std::optional<double> {
        if (horizon == 64 && resolved * 2 < complete) return std::nullopt; return Quantile(values, 50); };
    const auto censorCount = [&s](std::string_view name) { const auto found=s.censorReasons.find(std::string(name)); return found==s.censorReasons.end()?std::size_t{0}:found->second; };
    out << aggregation << ',' << symbol << ',' << lookback << ',' << partition << ',' << direction << ',' << horizon << ','
        << s.eligible << ',' << s.complete << ',' << s.censored << ',' << s.touch << ',' << s.close << ',' << Cell(rate(s.touch)) << ',' << Cell(rate(s.close)) << ','
        << Cell(median64(s.touchTimes,s.complete,s.touch)) << ',' << Cell(Quantile(s.touchTimes,25)) << ',' << Cell(Quantile(s.touchTimes,75)) << ','
        << Cell(median64(s.closeTimes,s.complete,s.close)) << ',' << Cell(Quantile(s.closeTimes,25)) << ',' << Cell(Quantile(s.closeTimes,75)) << ','
        << Cell(Quantile(s.mfe,25)) << ',' << Cell(Quantile(s.mfe,50)) << ',' << Cell(Quantile(s.mfe,75)) << ',' << Cell(Quantile(s.mae,25)) << ',' << Cell(Quantile(s.mae,50)) << ',' << Cell(Quantile(s.mae,75)) << ',' << Cell(Quantile(s.returns,25)) << ',' << Cell(Quantile(s.returns,50)) << ',' << Cell(Quantile(s.returns,75)) << ','
        << Cell(Quantile(s.mfePips,25)) << ',' << Cell(Quantile(s.mfePips,50)) << ',' << Cell(Quantile(s.mfePips,75)) << ',' << Cell(Quantile(s.maePips,25)) << ',' << Cell(Quantile(s.maePips,50)) << ',' << Cell(Quantile(s.maePips,75)) << ',' << Cell(Quantile(s.returnPips,25)) << ',' << Cell(Quantile(s.returnPips,50)) << ',' << Cell(Quantile(s.returnPips,75)) << ','
        << Cell(Quantile(s.mfeWidths,25)) << ',' << Cell(Quantile(s.mfeWidths,50)) << ',' << Cell(Quantile(s.mfeWidths,75)) << ',' << Cell(Quantile(s.maeWidths,25)) << ',' << Cell(Quantile(s.maeWidths,50)) << ',' << Cell(Quantile(s.maeWidths,75)) << ',' << Cell(Quantile(s.returnWidths,25)) << ',' << Cell(Quantile(s.returnWidths,50)) << ',' << Cell(Quantile(s.returnWidths,75)) << ','
        << s.raceContinuation << ',' << s.raceRevisit << ',' << s.raceSame << ',' << s.raceNeither << ',' << s.raceCensored << ','
        << censorCount("gap") << ',' << censorCount("boundary") << ',' << censorCount("tail") << ',' << censorCount("invalid_input") << '\n';
}

inline std::string OutcomesCsv(const std::vector<Row>& rows, const RunConfiguration& config)
{
    using Key=std::tuple<std::string,std::size_t,std::string,std::string,std::size_t>;
    std::map<Key,std::vector<const Row*>> cohorts;
    for (const Row& row:rows) cohorts[{row.symbol,row.lookback,row.partition,row.direction,row.horizon}].push_back(&row);
    std::ostringstream out; out << "aggregation,symbol,lookback,partition,direction,horizon,eligible,complete,censored,touches,closes,touch_rate,close_rate,touch_median_bars,touch_p25_bars,touch_p75_bars,fill_median_bars,fill_p25_bars,fill_p75_bars,mfe_p25_price,mfe_median_price,mfe_p75_price,mae_p25_price,mae_median_price,mae_p75_price,directional_close_return_p25_price,directional_close_return_median_price,directional_close_return_p75_price,mfe_p25_pips,mfe_median_pips,mfe_p75_pips,mae_p25_pips,mae_median_pips,mae_p75_pips,directional_close_return_p25_pips,directional_close_return_median_pips,directional_close_return_p75_pips,mfe_p25_widths,mfe_median_widths,mfe_p75_widths,mae_p25_widths,mae_median_widths,mae_p75_widths,directional_close_return_p25_widths,directional_close_return_median_widths,directional_close_return_p75_widths,race_continuation_first,race_revisit_first,race_same_bar_intrabar_order_indeterminate,race_neither,race_censored,censor_gap,censor_boundary,censor_tail,censor_invalid_input\n";
    for (const auto& [key, values]:cohorts) { const auto& [symbol,l,p,d,h]=key; WriteStatistics(out,"event_weighted",symbol,l,p,d,h,Calculate(values,SourceFor(config,symbol).pipSize)); }
    // Equal-symbol output is deliberately separate.  Counts are undefined; each
    // numeric statistic is the unweighted mean of defined symbol statistics.
    // The source event-weighted rows remain available for full symbol detail.
    return out.str();
}

inline std::optional<double> MeanDefined(const std::vector<std::optional<double>>& values)
{
    double sum=0; std::size_t count=0; for(const auto& value:values) if(value){sum+=*value;++count;}
    return count ? std::optional<double>(sum/count) : std::nullopt;
}
inline std::string EqualSymbolCsv(const std::vector<Row>& rows, const RunConfiguration& config)
{
    using SymbolKey=std::tuple<std::string,std::size_t,std::string,std::string,std::size_t>;
    using Key=std::tuple<std::size_t,std::string,std::string,std::size_t>;
    std::map<SymbolKey,std::vector<const Row*>> symbols;
    for(const Row& row:rows) symbols[{row.symbol,row.lookback,row.partition,row.direction,row.horizon}].push_back(&row);
    std::map<Key,std::vector<Statistics>> cohorts;
    for(const auto& [key, values]:symbols){const auto& [symbol,l,p,d,h]=key;cohorts[{l,p,d,h}].push_back(Calculate(values,SourceFor(config,symbol).pipSize));}
    const auto median=[](const std::vector<double>& values){return Quantile(values,50);};
    std::ostringstream out; out<<"aggregation,symbol,lookback,partition,direction,horizon,contributors,touch_rate,close_rate,touch_median_bars,fill_median_bars,mfe_p25_price,mfe_median_price,mfe_p75_price,mae_p25_price,mae_median_price,mae_p75_price,directional_close_return_p25_price,directional_close_return_median_price,directional_close_return_p75_price,mfe_p25_pips,mfe_median_pips,mfe_p75_pips,mae_p25_pips,mae_median_pips,mae_p75_pips,directional_close_return_p25_pips,directional_close_return_median_pips,directional_close_return_p75_pips,mfe_p25_widths,mfe_median_widths,mfe_p75_widths,mae_p25_widths,mae_median_widths,mae_p75_widths,directional_close_return_p25_widths,directional_close_return_median_widths,directional_close_return_p75_widths\n";
    for(const auto& [key, stats]:cohorts){const auto& [l,p,d,h]=key;std::vector<std::optional<double>> touch,close,touchTime,fillTime,mfe25,mfe, mfe75,mae25,mae,mae75,ret25,ret,ret75,mfeP25,mfeP,mfeP75,maeP25,maeP,maeP75,retP25,retP,retP75,mfeW25,mfeW,mfeW75,maeW25,maeW,maeW75,retW25,retW,retW75; for(const auto& s:stats){touch.push_back(s.complete?std::optional<double>(static_cast<double>(s.touch)/s.complete):std::nullopt);close.push_back(s.complete?std::optional<double>(static_cast<double>(s.close)/s.complete):std::nullopt);touchTime.push_back(h==64&&s.touch*2<s.complete?std::nullopt:median(s.touchTimes));fillTime.push_back(h==64&&s.close*2<s.complete?std::nullopt:median(s.closeTimes)); for(const auto n:{25u,50u,75u}){const auto mf=Quantile(s.mfe,n), ma=Quantile(s.mae,n), re=Quantile(s.returns,n), mp=Quantile(s.mfePips,n),ap=Quantile(s.maePips,n),rp=Quantile(s.returnPips,n),mw=Quantile(s.mfeWidths,n),aw=Quantile(s.maeWidths,n),rw=Quantile(s.returnWidths,n); if(n==25){mfe25.push_back(mf);mae25.push_back(ma);ret25.push_back(re);mfeP25.push_back(mp);maeP25.push_back(ap);retP25.push_back(rp);mfeW25.push_back(mw);maeW25.push_back(aw);retW25.push_back(rw);}else if(n==50){mfe.push_back(mf);mae.push_back(ma);ret.push_back(re);mfeP.push_back(mp);maeP.push_back(ap);retP.push_back(rp);mfeW.push_back(mw);maeW.push_back(aw);retW.push_back(rw);}else{mfe75.push_back(mf);mae75.push_back(ma);ret75.push_back(re);mfeP75.push_back(mp);maeP75.push_back(ap);retP75.push_back(rp);mfeW75.push_back(mw);maeW75.push_back(aw);retW75.push_back(rw);}}}
        out<<"equal_symbol,ALL,"<<l<<','<<p<<','<<d<<','<<h<<','<<stats.size()<<','<<Cell(MeanDefined(touch))<<','<<Cell(MeanDefined(close))<<','<<Cell(MeanDefined(touchTime))<<','<<Cell(MeanDefined(fillTime)); for(const auto* values:{&mfe25,&mfe,&mfe75,&mae25,&mae,&mae75,&ret25,&ret,&ret75,&mfeP25,&mfeP,&mfeP75,&maeP25,&maeP,&maeP75,&retP25,&retP,&retP75,&mfeW25,&mfeW,&mfeW75,&maeW25,&maeW,&maeW75,&retW25,&retW,&retW75})out<<','<<Cell(MeanDefined(*values)); out<<'\n'; }
    return out.str();
}

// The uncertainty file is long-form so every predeclared primary descriptive
// summary has the same explicit bootstrap contract without conflating an
// interval with a point-estimate column in either aggregation file.
inline const std::array<std::string_view, 35>& BootstrapStatistics()
{
    static const std::array<std::string_view,35> names{{"touch_rate","close_rate","touch_p25_bars","touch_median_bars","touch_p75_bars","fill_p25_bars","fill_median_bars","fill_p75_bars","mfe_p25_price","mfe_median_price","mfe_p75_price","mae_p25_price","mae_median_price","mae_p75_price","directional_close_return_p25_price","directional_close_return_median_price","directional_close_return_p75_price","mfe_p25_pips","mfe_median_pips","mfe_p75_pips","mae_p25_pips","mae_median_pips","mae_p75_pips","directional_close_return_p25_pips","directional_close_return_median_pips","directional_close_return_p75_pips","mfe_p25_widths","mfe_median_widths","mfe_p75_widths","mae_p25_widths","mae_median_widths","mae_p75_widths","directional_close_return_p25_widths","directional_close_return_median_widths","directional_close_return_p75_widths"}};
    return names;
}
inline std::optional<double> Statistic(const std::vector<const Row*>& rows, const RunConfiguration& config, std::string_view name)
{
    std::size_t complete=0,touch=0,close=0,continuation=0,revisit=0; std::vector<double> values;
    const auto collect=[&](double value){values.push_back(value);};
    for(const Row* row:rows) { if(!row->complete) continue; ++complete; touch+=row->touchAt.has_value(); close+=row->closeAt.has_value(); if(row->horizon==64){continuation+=row->race=="continuation_first";revisit+=row->race=="revisit_first";} if(name.rfind("touch_",0)==0&&row->touchAt) collect(*row->touchAt); else if(name.rfind("fill_",0)==0&&row->closeAt) collect(*row->closeAt); else if(name.rfind("mfe_",0)==0){const double pip=SourceFor(config,row->symbol).pipSize; if(name.find("_pips")!=std::string_view::npos) collect(row->mfe/pip); else if(name.find("_widths")!=std::string_view::npos) collect(row->mfe/(row->upper-row->lower)); else collect(row->mfe);} else if(name.rfind("mae_",0)==0){const double pip=SourceFor(config,row->symbol).pipSize; if(name.find("_pips")!=std::string_view::npos) collect(row->mae/pip); else if(name.find("_widths")!=std::string_view::npos) collect(row->mae/(row->upper-row->lower)); else collect(row->mae);} else if(name.rfind("directional_",0)==0){const double pip=SourceFor(config,row->symbol).pipSize; if(name.find("_pips")!=std::string_view::npos) collect(row->directionalReturn/pip); else if(name.find("_widths")!=std::string_view::npos) collect(row->directionalReturn/(row->upper-row->lower)); else collect(row->directionalReturn);} }
    if(name=="touch_rate") return complete?std::optional<double>(static_cast<double>(touch)/complete):std::nullopt;
    if(name=="close_rate") return complete?std::optional<double>(static_cast<double>(close)/complete):std::nullopt;
    if(name=="race_continuation_first_proportion") return (!rows.empty() && rows.front()->horizon==64 && complete)?std::optional<double>(static_cast<double>(continuation)/complete):std::nullopt;
    if(name=="race_revisit_first_proportion") return (!rows.empty() && rows.front()->horizon==64 && complete)?std::optional<double>(static_cast<double>(revisit)/complete):std::nullopt;
    const std::size_t q=name.find("p25")!=std::string_view::npos?25:name.find("p75")!=std::string_view::npos?75:50;
    return Quantile(std::move(values),q);
}
inline std::vector<std::vector<const Row*>> WeekBlocks(const std::vector<const Row*>& rows)
{
    std::map<std::int64_t,std::vector<const Row*>> weeks; for(const Row* row:rows) weeks[row->confirmationTimestamp/(7*24*60*60)].push_back(row);
    std::vector<std::vector<const Row*>> result;for(auto& [week,block]:weeks){(void)week;result.push_back(std::move(block));}return result;
}
inline std::pair<std::optional<double>,std::optional<double>> Interval(std::vector<double> values)
{
    if(values.empty()) return {};std::sort(values.begin(),values.end());return {values[(values.size()-1)*25/1000],values[(values.size()-1)*975/1000]};
}
inline std::pair<std::optional<double>,std::optional<double>> EventInterval(const std::map<std::string,std::vector<const Row*>>& symbols,const RunConfiguration& config,std::string_view statistic,std::uint64_t seed)
{
    std::vector<std::vector<std::vector<const Row*>>> blocks;for(const auto& [symbol,rows]:symbols){(void)symbol;auto weeks=WeekBlocks(rows);if(!weeks.empty())blocks.push_back(std::move(weeks));}
    if(blocks.empty())return {};std::mt19937_64 random(seed);std::vector<double> draws;
    for(std::size_t replicate=0;replicate<kBootstrapReplicates;++replicate){std::vector<const Row*> sample;for(const auto& symbolWeeks:blocks){std::uniform_int_distribution<std::size_t> choose(0,symbolWeeks.size()-1);for(std::size_t draw=0;draw<symbolWeeks.size();++draw){const auto selected=choose(random);sample.insert(sample.end(),symbolWeeks[selected].begin(),symbolWeeks[selected].end());}}if(const auto value=Statistic(sample,config,statistic))draws.push_back(*value);}return Interval(std::move(draws));
}
inline std::pair<std::optional<double>,std::optional<double>> EqualSymbolInterval(const std::map<std::string,std::vector<const Row*>>& symbols,const RunConfiguration& config,std::string_view statistic,std::uint64_t seed)
{
    std::array<std::vector<std::vector<const Row*>>,6> blocks;for(std::size_t i=0;i<config.symbols.size();++i){const auto found=symbols.find(config.symbols[i].symbol);if(found!=symbols.end())blocks[i]=WeekBlocks(found->second);}std::mt19937_64 random(seed);std::uniform_int_distribution<std::size_t> pickSymbol(0,5);std::vector<double> draws;
    for(std::size_t replicate=0;replicate<kBootstrapReplicates;++replicate){double sum=0;std::size_t contributors=0;for(std::size_t selected=0;selected<6;++selected){const auto& weeks=blocks[pickSymbol(random)];if(weeks.empty())continue;std::uniform_int_distribution<std::size_t> pickWeek(0,weeks.size()-1);std::vector<const Row*> sample;for(std::size_t draw=0;draw<weeks.size();++draw){const auto chosen=pickWeek(random);sample.insert(sample.end(),weeks[chosen].begin(),weeks[chosen].end());}if(const auto value=Statistic(sample,config,statistic)){sum+=*value;++contributors;}}if(contributors)draws.push_back(sum/contributors);}return Interval(std::move(draws));
}
inline std::string UncertaintyCsv(const std::vector<Row>& rows,const RunConfiguration& config)
{
    using Key=std::tuple<std::size_t,std::string,std::string,std::size_t>;std::map<Key,std::map<std::string,std::vector<const Row*>>> cohorts;for(const Row& row:rows)cohorts[{row.lookback,row.partition,row.direction,row.horizon}][row.symbol].push_back(&row);std::ostringstream out;out<<"aggregation,symbol,lookback,partition,direction,horizon,statistic,estimate,lower95,upper95,bootstrap_replicates\n";const auto base=DerivedBootstrapSeed(config.configurationSha256);
    for(const auto& [key,symbols]:cohorts){const auto& [l,p,d,h]=key;std::vector<const Row*> pooled;for(const auto& [symbol,values]:symbols){(void)symbol;pooled.insert(pooled.end(),values.begin(),values.end());}std::size_t statisticIndex=0;for(const auto name:BootstrapStatistics()){const auto estimate=Statistic(pooled,config,name);const auto interval=EventInterval(symbols,config,name,base+statisticIndex);out<<"event_weighted,ALL,"<<l<<','<<p<<','<<d<<','<<h<<','<<name<<','<<Cell(estimate)<<','<<Cell(interval.first)<<','<<Cell(interval.second)<<','<<kBootstrapReplicates<<'\n';const auto equal=EqualSymbolInterval(symbols,config,name,base+0x9e3779b97f4a7c15ULL+statisticIndex);std::vector<std::optional<double>> individual;for(const auto& source:config.symbols){const auto found=symbols.find(source.symbol);individual.push_back(found==symbols.end()?std::nullopt:Statistic(found->second,config,name));}out<<"equal_symbol,ALL,"<<l<<','<<p<<','<<d<<','<<h<<','<<name<<','<<Cell(MeanDefined(individual))<<','<<Cell(equal.first)<<','<<Cell(equal.second)<<','<<kBootstrapReplicates<<'\n';++statisticIndex;}}
    return out.str();
}

inline std::string StructuralCsv(const std::vector<Row>& rows, const RunConfiguration& config)
{
    struct Obs { const Row* row; }; std::map<std::tuple<std::string,std::size_t,std::string>,std::map<std::string,const Row*>> unique;
    for (const Row& row:rows) unique[{row.symbol,row.lookback,row.partition}][row.identity]=&row;
    std::ostringstream out; out << "symbol,lookback,partition,emitted_eligible,bullish,bearish,bullish_share,bearish_share,width_median_price,width_median_pips,confirmation_spacing_median_bars,confirmation_spacing_median_seconds,simultaneous_observations,temporal_overlap_pairs_64,price_range_overlap_pairs_64,repeated_same_direction_within_64,utc_week_cluster_max,utc_week_cluster_p25,utc_week_cluster_median,utc_week_cluster_p75,greedy_temporal_thinned_64\n";
    for (const auto& [key, identities] : unique) { const auto& [symbol,l,p]=key; std::vector<const Row*> values; for(const auto& [id,row]:identities){(void)id;values.push_back(row);} std::sort(values.begin(),values.end(),[](auto a,auto b){return std::tie(a->confirmationBar,a->identity)<std::tie(b->confirmationBar,b->identity);});
        std::size_t bull=0,sim=0,temporal=0,range=0,repeated=0,thinned=0; std::vector<double> widths, spacing, seconds, weeks; std::map<std::int64_t,std::size_t> timestamp, week;
        std::optional<std::size_t> retained; for(std::size_t i=0;i<values.size();++i){const Row* r=values[i]; bull += r->direction=="bullish"; widths.push_back(r->upper-r->lower); ++timestamp[r->confirmationTimestamp]; ++week[r->confirmationTimestamp/(7*24*60*60)]; if(!retained || r->confirmationBar>=*retained+64){++thinned;retained=r->confirmationBar;} if(i){spacing.push_back(static_cast<double>(r->confirmationBar-values[i-1]->confirmationBar)); seconds.push_back(static_cast<double>(r->confirmationTimestamp-values[i-1]->confirmationTimestamp));} for(std::size_t j=0;j<i;++j){const Row* prior=values[j]; if(r->confirmationBar<prior->confirmationBar+64){++temporal; if(std::max(r->lower,prior->lower)<=std::min(r->upper,prior->upper)) ++range; if(r->direction==prior->direction) ++repeated;}}}
        for(const auto& [t,n]:timestamp){(void)t;if(n>1)sim+=n-1;} for(const auto& [w,n]:week){(void)w;weeks.push_back(n);} const double pip=SourceFor(config,symbol).pipSize; const auto q=[&](std::vector<double> x,std::size_t n){return Cell(Quantile(std::move(x),n));}; double maxWeek=weeks.empty()?0:*std::max_element(weeks.begin(),weeks.end());
        out<<symbol<<','<<l<<','<<p<<','<<values.size()<<','<<bull<<','<<values.size()-bull<<','<<Cell(values.empty()?std::nullopt:std::optional<double>(static_cast<double>(bull)/values.size()))<<','<<Cell(values.empty()?std::nullopt:std::optional<double>(static_cast<double>(values.size()-bull)/values.size()))<<','<<q(widths,50)<<','; for(double& x:widths)x/=pip; out<<q(widths,50)<<','<<q(spacing,50)<<','<<q(seconds,50)<<','<<sim<<','<<temporal<<','<<range<<','<<repeated<<','<<CsvNumber(maxWeek)<<','<<q(weeks,25)<<','<<q(weeks,50)<<','<<q(weeks,75)<<','<<thinned<<'\n';
    } return out.str();
}

class Writer final
{
public:
    explicit Writer(std::filesystem::path target) : target_(std::move(target)), staging_(target_.string()+".tmp."+std::to_string(::getpid()))
    { if(target_.empty()||std::filesystem::exists(target_)||std::filesystem::exists(staging_)) throw std::invalid_argument("POCKET_DERIVED_OUTPUT_TARGET_EXISTS_OR_EMPTY"); std::filesystem::create_directories(staging_); }
    ~Writer(){if(!published_){std::error_code error;std::filesystem::remove_all(staging_,error);}}
    void Publish(const std::filesystem::path& source,std::string_view git,std::string_view executable,const std::vector<Row>& rows,
        const RunConfiguration& config, std::string_view analyzerSchema) {
        Write("outcomes.csv",OutcomesCsv(rows,config)); Write("equal_symbol.csv",EqualSymbolCsv(rows,config)); Write("uncertainty.csv",UncertaintyCsv(rows,config)); Write("structural.csv",StructuralCsv(rows,config));
        const std::string sourceManifest=ReadTextFile(source/"manifest.txt"); auto field=[&](std::string_view key){const std::string prefix=std::string(key)+"=";const auto a=sourceManifest.find(prefix);if(a==std::string::npos)throw std::invalid_argument("POCKET_DERIVED_SOURCE_MANIFEST");const auto b=sourceManifest.find('\n',a);return sourceManifest.substr(a+prefix.size(),b-a-prefix.size());};
        const std::string manifest="analyzer_schema="+std::string(analyzerSchema)+"\nsource_study="+field("study")+"\nsource_protocol="+field("protocol")+"\nsource_protocol_document_sha256="+field("protocol_document_sha256")+"\nsource_detector="+field("detector")+"\nsource_configuration_sha256="+field("configuration_sha256")+"\nsource_observations_sha256="+field("observations_sha256")+"\nsource_aggregates_sha256="+field("aggregates_sha256")+"\ngit_commit="+std::string(git)+"\nexecutable_sha256="+std::string(executable)+"\nbootstrap_seed="+std::to_string(DerivedBootstrapSeed(config.configurationSha256))+"\nbootstrap_replicates="+std::to_string(kBootstrapReplicates)+"\noutcomes_sha256="+FileSha256(staging_/"outcomes.csv")+"\nequal_symbol_sha256="+FileSha256(staging_/"equal_symbol.csv")+"\nuncertainty_sha256="+FileSha256(staging_/"uncertainty.csv")+"\nstructural_sha256="+FileSha256(staging_/"structural.csv")+"\n"; Write("manifest.txt",manifest); VerifyContract(staging_,analyzerSchema,config.configurationSha256); std::filesystem::rename(staging_,target_);published_=true;
    }
    static void Verify(const std::filesystem::path& directory) { const std::string manifest=ReadTextFile(directory/"manifest.txt"); if(manifest.rfind("analyzer_schema="+std::string(kAnalyzerSchemaV1),0)==0) { VerifyContract(directory,kAnalyzerSchemaV1,FrozenConfiguration().configurationSha256); return; } VerifyContract(directory,kAnalyzerSchema,FrozenConfiguration().configurationSha256); }
    static void VerifyContract(const std::filesystem::path& directory,std::string_view expectedSchema,std::string_view configurationHash) { const std::string m=ReadTextFile(directory/"manifest.txt"); auto f=[&](std::string_view key){const std::string x=std::string(key)+"=";const auto a=m.find(x);if(a==std::string::npos)throw std::invalid_argument("POCKET_DERIVED_MANIFEST");const auto b=m.find('\n',a);return m.substr(a+x.size(),b-a-x.size());};const std::string schema=f("analyzer_schema");if(schema==kAnalyzerSchemaV1&&expectedSchema==kAnalyzerSchemaV1){if(f("outcomes_sha256")!=FileSha256(directory/"outcomes.csv")||f("equal_symbol_sha256")!=FileSha256(directory/"equal_symbol.csv")||f("structural_sha256")!=FileSha256(directory/"structural.csv"))throw std::invalid_argument("POCKET_DERIVED_TAMPER_OR_INCOMPLETE");return;}if(schema!=expectedSchema||f("bootstrap_replicates")!="2000"||f("bootstrap_seed")!=std::to_string(DerivedBootstrapSeed(configurationHash))||f("outcomes_sha256")!=FileSha256(directory/"outcomes.csv")||f("equal_symbol_sha256")!=FileSha256(directory/"equal_symbol.csv")||f("uncertainty_sha256")!=FileSha256(directory/"uncertainty.csv")||f("structural_sha256")!=FileSha256(directory/"structural.csv"))throw std::invalid_argument("POCKET_DERIVED_TAMPER_OR_INCOMPLETE"); }
private:
    void Write(std::string_view n,std::string_view s){std::ofstream o(staging_/std::string(n),std::ios::binary);if(!o)throw std::runtime_error("POCKET_DERIVED_WRITE");o<<s;o.close();if(!o)throw std::runtime_error("POCKET_DERIVED_WRITE");} std::filesystem::path target_,staging_;bool published_=false;
};
inline void Analyze(const std::filesystem::path& source,const std::filesystem::path& target,std::string_view git,std::string_view executable)
{
    if(git.size()!=40 || !std::all_of(git.begin(),git.end(),[](char c){return std::isxdigit(static_cast<unsigned char>(c));}) || !IsHexSha256(executable)) throw std::invalid_argument("POCKET_DERIVED_PROVENANCE_REQUIRED");
    const std::vector<Row> rows=ReadVerifiedRows(source); Writer(target).Publish(source,git,executable,rows,FrozenConfiguration(),kAnalyzerSchema);
}
inline void AnalyzeContract(const std::filesystem::path& source,const std::filesystem::path& target,std::string_view git,std::string_view executable,
    const RunConfiguration& config,std::string_view configurationText,std::string_view primarySchema,std::string_view derivedSchema)
{
    if(git.size()!=40 || !std::all_of(git.begin(),git.end(),[](char c){return std::isxdigit(static_cast<unsigned char>(c));}) || !IsHexSha256(executable)) throw std::invalid_argument("POCKET_DERIVED_PROVENANCE_REQUIRED");
    const auto rows=ReadVerifiedRowsContract(source,config,configurationText,primarySchema); Writer(target).Publish(source,git,executable,rows,config,derivedSchema);
}
} // namespace EA::Pocket::Prospective::Derived
