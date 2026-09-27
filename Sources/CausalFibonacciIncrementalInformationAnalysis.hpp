#pragma once

// Deterministic, artifact-only pre-2025 analysis boundary.  It never opens a
// database and discards confirmation rows immediately after reading their
// partition field, before parsing targets or predictors.

#include <iostream>
#include "../Headers/CausalFibonacciIncrementalInformation.hpp"

#include <charconv>
#include <cstdlib>
#include <functional>
#include <unordered_map>

namespace EA::CausalFibonacciIncrementalInformation::Analysis {

inline constexpr std::string_view kRunnerIdentity =
    "causal-fibonacci-layout9-pre2025-artifact-analysis-v1";

struct ParsedTarget { int assignedClass = 1; float terminalReturn = 0.0f; bool eligible = false; };
struct ParsedRow {
    std::string symbol;
    std::int64_t timestamp = 0;
    std::uint64_t ordinal = 0;
    Partition partition = Partition::Development;
    std::array<float, kBaselineWidth> baseline{};
    std::array<float, kFibonacciWidth> fibonacci{};
    ParsedTarget h4, h6;
};

struct Options { std::filesystem::path artifactDirectory, outputDirectory; std::string codeCommit; };

inline std::string_view FieldAt(std::string_view line, std::size_t desired)
{
    std::size_t field = 0, start = 0;
    for (std::size_t i = 0; i <= line.size(); ++i) {
        if (i == line.size() || line[i] == ',') {
            if (field == desired) return line.substr(start, i - start);
            ++field; start = i + 1;
        }
    }
    throw std::invalid_argument("fibonacci_artifact_row_schema_too_short");
}

template <typename T>
inline T Decimal(std::string_view text, const char* reason)
{
    T value{};
    const char* begin = text.data(); const char* end = begin + text.size();
    const auto parsed = std::from_chars(begin, end, value);
    if (parsed.ec != std::errc{} || parsed.ptr != end)
        throw std::invalid_argument(std::string("fibonacci_artifact_invalid_") + reason);
    return value;
}
inline float DecimalFloat(std::string_view text, const char* reason)
{
    std::string copy{text}; char* end = nullptr; const float value = std::strtof(copy.c_str(), &end);
    if (end != copy.c_str() + copy.size() || !std::isfinite(value))
        throw std::invalid_argument(std::string("fibonacci_artifact_invalid_") + reason);
    return value;
}
inline bool DecimalBool(std::string_view text, const char* reason)
{
    if (text == "0") return false; if (text == "1") return true;
    throw std::invalid_argument(std::string("fibonacci_artifact_invalid_") + reason);
}
inline std::optional<Partition> ParsedPartition(std::string_view text)
{
    if (text == "development") return Partition::Development;
    if (text == "validation") return Partition::Validation;
    if (text == "pre2025_lock_test") return Partition::Pre2025LockTest;
    if (text == "confirmation_2025") return Partition::Confirmation2025;
    return std::nullopt;
}

// Returns nullopt for confirmation rows.  Callers must not inspect any field
// after the partition test on that path.
inline std::optional<ParsedRow> ParsePre2025Row(std::string_view line)
{
    constexpr std::size_t kPartitionField = 4;
    const auto partition = ParsedPartition(FieldAt(line, kPartitionField));
    if (!partition) throw std::invalid_argument("fibonacci_artifact_unknown_partition");
    if (*partition == Partition::Confirmation2025) return std::nullopt;
    ParsedRow row; row.partition = *partition;
    row.symbol = std::string(FieldAt(line, 1));
    row.timestamp = Decimal<std::int64_t>(FieldAt(line, 2), "timestamp");
    row.ordinal = Decimal<std::uint64_t>(FieldAt(line, 3), "ordinal");
    if (PartitionFor(row.timestamp) != row.partition)
        throw std::invalid_argument("fibonacci_artifact_partition_timestamp_mismatch");
    row.h4 = {Decimal<int>(FieldAt(line, 5), "h4_class"), DecimalFloat(FieldAt(line, 8), "h4_terminal_return"), DecimalBool(FieldAt(line, 9), "h4_eligible")};
    row.h6 = {Decimal<int>(FieldAt(line, 11), "h6_class"), DecimalFloat(FieldAt(line, 14), "h6_terminal_return"), DecimalBool(FieldAt(line, 15), "h6_eligible")};
    for (std::size_t i = 0; i < kBaselineWidth; ++i) row.baseline[i] = DecimalFloat(FieldAt(line, 17 + i), "baseline");
    for (std::size_t i = 0; i < kFibonacciWidth; ++i) row.fibonacci[i] = DecimalFloat(FieldAt(line, 17 + kBaselineWidth + i), "fibonacci");
    return row;
}

inline void ValidateRowsHeader(const std::string& header)
{
    if (FieldAt(header, 0) != "row_identity" || FieldAt(header, 1) != "symbol" ||
        FieldAt(header, 4) != "partition" || FieldAt(header, 17) != "baseline_0" ||
        FieldAt(header, 17 + kBaselineWidth) != "fibonacci_0" ||
        FieldAt(header, 17 + kBaselineWidth + kFibonacciWidth - 1) != "fibonacci_22")
        throw std::invalid_argument("fibonacci_artifact_rows_schema_mismatch");
}

inline std::map<std::string, std::vector<ParsedRow>> LoadPre2025Rows(const std::filesystem::path& artifact)
{
    std::ifstream input(artifact / "rows.csv"); if (!input) throw std::runtime_error("fibonacci_artifact_rows_unreadable");
    std::string line; if (!std::getline(input, line)) throw std::runtime_error("fibonacci_artifact_rows_header_missing"); ValidateRowsHeader(line);
    std::map<std::string, std::vector<ParsedRow>> grouped;
    std::string previousSymbol; std::int64_t previousTimestamp = std::numeric_limits<std::int64_t>::min();
    while (std::getline(input, line)) {
        const auto parsed = ParsePre2025Row(line);
        if (!parsed) continue; // sealed confirmation: no target/predictor parsing or accumulation.
        if (parsed->symbol != previousSymbol) { if (!previousSymbol.empty() && parsed->symbol < previousSymbol) throw std::invalid_argument("fibonacci_artifact_pre2025_sort_order_invalid"); previousSymbol = parsed->symbol; previousTimestamp = std::numeric_limits<std::int64_t>::min(); }
        if (parsed->timestamp <= previousTimestamp) throw std::invalid_argument("fibonacci_artifact_pre2025_duplicate_or_unordered_timestamp");
        previousTimestamp = parsed->timestamp; grouped[parsed->symbol].push_back(std::move(*parsed));
    }
    const std::array<std::string, 6> expected{{"audcadrmp","audusdrmp","eurusdrmp","gbpusdrmp","usdcadrmp","usdjpyrmp"}};
    if (grouped.size() != expected.size()) throw std::invalid_argument("fibonacci_artifact_pre2025_symbol_universe_mismatch");
    for (const auto& symbol : expected) if (!grouped.contains(symbol)) throw std::invalid_argument("fibonacci_artifact_pre2025_symbol_missing:" + symbol);
    return grouped;
}

template <typename Consumer>
inline void StreamPre2025Symbols(const std::filesystem::path& artifact, Consumer consumer)
{
    std::ifstream input(artifact / "rows.csv"); if (!input) throw std::runtime_error("fibonacci_artifact_rows_unreadable");
    std::string line; if (!std::getline(input, line)) throw std::runtime_error("fibonacci_artifact_rows_header_missing"); ValidateRowsHeader(line);
    const std::array<std::string, 6> expected{{"audcadrmp","audusdrmp","eurusdrmp","gbpusdrmp","usdcadrmp","usdjpyrmp"}};
    std::size_t expectedIndex = 0; std::string currentSymbol; std::vector<ParsedRow> current; std::int64_t previousTimestamp = std::numeric_limits<std::int64_t>::min();
    const auto flush = [&] { if (currentSymbol.empty()) return; if (expectedIndex >= expected.size() || currentSymbol != expected[expectedIndex]) throw std::invalid_argument("fibonacci_artifact_pre2025_symbol_universe_mismatch"); consumer(currentSymbol, current); ++expectedIndex; current.clear(); current.shrink_to_fit(); };
    while (std::getline(input, line)) {
        const auto parsed = ParsePre2025Row(line);
        if (!parsed) continue; // sealed confirmation: no target/predictor parsing or accumulation.
        if (currentSymbol.empty()) currentSymbol = parsed->symbol;
        if (parsed->symbol != currentSymbol) { flush(); currentSymbol = parsed->symbol; previousTimestamp = std::numeric_limits<std::int64_t>::min(); }
        if (parsed->timestamp <= previousTimestamp) throw std::invalid_argument("fibonacci_artifact_pre2025_duplicate_or_unordered_timestamp");
        previousTimestamp = parsed->timestamp; current.push_back(std::move(*parsed));
    }
    flush();
    if (expectedIndex != expected.size()) throw std::invalid_argument("fibonacci_artifact_pre2025_symbol_universe_mismatch");
}

struct FeatureRef { bool fibonacci = false; std::size_t column = 0; double mean = 0.0, standardDeviation = 1.0; bool categorical = false; };
struct InputTransform { std::vector<FeatureRef> baseline, augmented; };

inline InputTransform FitDevelopmentInputTransform(const std::vector<ParsedRow>& rows)
{
    InputTransform result; const FeatureSchema schema = FrozenFeatureSchema();
    for (std::size_t column = 0; column < kBaselineWidth; ++column) {
        double sum = 0.0; std::size_t count = 0; for (const auto& row : rows) if (row.partition == Partition::Development) { sum += row.baseline[column]; ++count; }
        if (count == 0) throw std::invalid_argument("fibonacci_no_development_rows");
        const bool categorical = schema.baseline[column].categorical;
        const double mean = categorical ? 0.0 : sum / count; double variance = 0.0;
        if (!categorical) for (const auto& row : rows) if (row.partition == Partition::Development) { const double d = row.baseline[column] - mean; variance += d * d; }
        const double sd = categorical ? 1.0 : std::sqrt(variance / count);
        if (categorical || sd > 0.0) result.baseline.push_back({false, column, mean, sd, categorical});
    }
    result.augmented = result.baseline;
    for (std::size_t column = 0; column < kFibonacciWidth; ++column) {
        const bool categorical = column == 0; double sum = 0.0; std::size_t count = 0;
        for (const auto& row : rows) if (row.partition == Partition::Development) { sum += row.fibonacci[column]; ++count; }
        const double mean = categorical ? 0.0 : sum / count; double variance = 0.0;
        if (!categorical) for (const auto& row : rows) if (row.partition == Partition::Development) { const double d = row.fibonacci[column] - mean; variance += d*d; }
        const double sd = categorical ? 1.0 : std::sqrt(variance / count);
        if (categorical || sd > 0.0) result.augmented.push_back({true, column, mean, sd, categorical});
    }
    return result;
}
inline double InputValue(const ParsedRow& row, const FeatureRef& ref)
{
    const double value = ref.fibonacci ? row.fibonacci[ref.column] : row.baseline[ref.column];
    return ref.categorical ? value : (value - ref.mean) / ref.standardDeviation;
}

template <typename Label>
inline std::vector<std::size_t> EligibleRows(const std::vector<ParsedRow>& rows, Partition partition, Label label)
{
    std::vector<std::size_t> out; for (std::size_t i = 0; i < rows.size(); ++i) if (rows[i].partition == partition && label(rows[i]).eligible) out.push_back(i); return out;
}

struct RidgeModel { bool available = false; std::string unavailableReason; std::vector<double> beta; double targetMean = 0.0, targetStandardDeviation = 1.0; };
inline RidgeModel FitRidge(const std::vector<ParsedRow>& rows, const std::vector<std::size_t>& selected, const std::vector<FeatureRef>& features, std::size_t fibonacciColumn)
{
    RidgeModel result; if (selected.empty()) { result.unavailableReason = "no_development_target_rows"; return result; }
    for (auto i : selected) result.targetMean += rows[i].fibonacci[fibonacciColumn]; result.targetMean /= selected.size();
    for (auto i : selected) { const double d = rows[i].fibonacci[fibonacciColumn] - result.targetMean; result.targetStandardDeviation += d*d; }
    result.targetStandardDeviation = std::sqrt(result.targetStandardDeviation / selected.size());
    if (result.targetStandardDeviation == 0.0) { result.unavailableReason = "zero_variance_development_target"; return result; }
    const std::size_t p = features.size() + 1; std::vector<std::vector<double>> gram(p, std::vector<double>(p)); std::vector<double> rhs(p);
    for (auto index : selected) { const auto& row = rows[index]; std::vector<double> x{1.0}; for (const auto& feature : features) x.push_back(InputValue(row, feature)); const double y = (row.fibonacci[fibonacciColumn] - result.targetMean) / result.targetStandardDeviation; for (std::size_t a=0;a<p;++a) { rhs[a]+=x[a]*y; for(std::size_t b=0;b<p;++b) gram[a][b]+=x[a]*x[b]; } }
    for (std::size_t i=1;i<p;++i) gram[i][i]+=1.0;
    result.beta=SolveLinearSystem(std::move(gram),std::move(rhs)); result.available=true; return result;
}
inline ReconstructionMetrics EvaluateRidge(const RidgeModel& model, const std::vector<ParsedRow>& rows, const std::vector<std::size_t>& selected, const std::vector<FeatureRef>& features, std::size_t fibonacciColumn)
{
    ReconstructionMetrics result; if (!model.available) { result.unavailableReason=model.unavailableReason; return result; } if(selected.empty()){result.unavailableReason="empty_holdout";return result;}
    double mean=0.0;for(auto i:selected)mean+=rows[i].fibonacci[fibonacciColumn];mean/=selected.size();double squared=0.0,total=0.0;
    for(auto index:selected){const auto& row=rows[index];double standardized=model.beta[0];for(std::size_t j=0;j<features.size();++j)standardized+=model.beta[j+1]*InputValue(row,features[j]);const double prediction=model.targetMean+model.targetStandardDeviation*standardized;const double d=row.fibonacci[fibonacciColumn]-prediction;squared+=d*d;const double c=row.fibonacci[fibonacciColumn]-mean;total+=c*c;}
    result.available=true;result.rmse=std::sqrt(squared/selected.size());if(total>0.0)result.rSquared=1.0-squared/total;else result.unavailableReason="zero_variance_holdout_target";return result;
}

struct BinaryRowsModel { bool available=false;std::string unavailableReason;std::vector<double> beta; };
inline BinaryRowsModel FitBinaryRows(const std::vector<ParsedRow>& rows,const std::vector<std::size_t>& selected,const std::vector<FeatureRef>& features)
{
    BinaryRowsModel result;if(selected.empty()){result.unavailableReason="no_development_target_rows";return result;}bool zero=false,one=false;for(auto i:selected){zero|=rows[i].fibonacci[0]<.5f;one|=rows[i].fibonacci[0]>=.5f;}if(!zero||!one){result.unavailableReason="development_one_class_binary_target";return result;}const std::size_t p=features.size();
    const auto optimized=OptimizeLbfgs(std::vector<double>(p+1),[&](const std::vector<double>& b){double loss=0.0;std::vector<double> gradient(p+1);for(auto index:selected){const auto& row=rows[index];double z=b[0];for(std::size_t j=0;j<p;++j)z+=b[j+1]*InputValue(row,features[j]);const int y=row.fibonacci[0]>=.5f;const double probability=StableSigmoid(z);loss+=std::max(z,0.0)-z*y+std::log1p(std::exp(-std::abs(z)));const double e=probability-y;gradient[0]+=e;for(std::size_t j=0;j<p;++j)gradient[j+1]+=e*InputValue(row,features[j]);}for(std::size_t j=1;j<=p;++j){loss+=b[j]*b[j];gradient[j]+=2.0*b[j];}return std::pair{loss,gradient};});
    if(!optimized.converged){result.unavailableReason="binary_lbfgs_not_converged";return result;}result.available=true;result.beta=optimized.parameters;return result;
}
inline ReconstructionMetrics EvaluateBinaryRows(const BinaryRowsModel& model,const std::vector<ParsedRow>& rows,const std::vector<std::size_t>& selected,const std::vector<FeatureRef>& features)
{
    ReconstructionMetrics result;if(!model.available){result.unavailableReason=model.unavailableReason;return result;}if(selected.empty()){result.unavailableReason="empty_holdout";return result;}double loss=0,brier=0;for(auto index:selected){const auto& row=rows[index];double z=model.beta[0];for(std::size_t j=0;j<features.size();++j)z+=model.beta[j+1]*InputValue(row,features[j]);const double p=std::clamp(StableSigmoid(z),1e-15,1.0-1e-15);const int y=row.fibonacci[0]>=.5f;loss-=y?std::log(p):std::log(1-p);const double d=p-y;brier+=d*d;}result.available=true;result.logLoss=loss/selected.size();result.brier=brier/selected.size();return result;
}

struct MultiRowsModel { bool available=false;std::string unavailableReason;std::size_t featureCount=0;std::vector<double> parameters; };
inline std::array<double,3> MultiProbability(const MultiRowsModel& model,const ParsedRow& row,const std::vector<FeatureRef>& features)
{
    std::array<double,3> z{};double maximum=-std::numeric_limits<double>::infinity();for(std::size_t c=0;c<3;++c){z[c]=model.parameters[c*(features.size()+1)];for(std::size_t j=0;j<features.size();++j)z[c]+=model.parameters[c*(features.size()+1)+j+1]*InputValue(row,features[j]);maximum=std::max(maximum,z[c]);}double sum=0;for(auto&v:z){v=std::exp(v-maximum);sum+=v;}for(auto&v:z)v/=sum;return z;
}
template <typename Label>
inline MultiRowsModel FitMultiRows(const std::vector<ParsedRow>& rows,const std::vector<std::size_t>& selected,const std::vector<FeatureRef>& features,Label label)
{
    MultiRowsModel result;if(selected.empty()){result.unavailableReason="no_development_target_rows";return result;}std::array<bool,3> present{};for(auto i:selected){const int y=label(rows[i]).assignedClass;if(y<0||y>2){result.unavailableReason="invalid_directional_target";return result;}present[y]=true;}if(!present[0]||!present[1]||!present[2]){result.unavailableReason="development_missing_directional_class";return result;}const std::size_t p=features.size();
    const auto optimized=OptimizeLbfgs(std::vector<double>(3*(p+1)),[&](const std::vector<double>& b){double loss=0;std::vector<double> g(b.size());for(auto index:selected){const auto& row=rows[index];const int y=label(row).assignedClass;std::array<double,3> z{};double maximum=-std::numeric_limits<double>::infinity();for(std::size_t c=0;c<3;++c){z[c]=b[c*(p+1)];for(std::size_t j=0;j<p;++j)z[c]+=b[c*(p+1)+j+1]*InputValue(row,features[j]);maximum=std::max(maximum,z[c]);}double sum=0;for(auto&v:z){v=std::exp(v-maximum);sum+=v;}for(auto&v:z)v/=sum;loss-=std::log(std::max(z[y],1e-300));for(std::size_t c=0;c<3;++c){const double e=z[c]-(y==static_cast<int>(c));g[c*(p+1)]+=e;for(std::size_t j=0;j<p;++j)g[c*(p+1)+j+1]+=e*InputValue(row,features[j]);}}for(std::size_t c=0;c<3;++c)for(std::size_t j=1;j<=p;++j){const std::size_t n=c*(p+1)+j;loss+=b[n]*b[n];g[n]+=2*b[n];}return std::pair{loss,g};});
    if(!optimized.converged){result.unavailableReason="multinomial_lbfgs_not_converged";return result;}result.available=true;result.featureCount=p;result.parameters=optimized.parameters;return result;
}

struct MultiRowsMetrics { bool available=false;std::string unavailableReason;double logLoss=std::numeric_limits<double>::quiet_NaN(),brier=std::numeric_limits<double>::quiet_NaN(),accuracy=std::numeric_limits<double>::quiet_NaN();std::vector<double> rowLoss,rowBrier; };
template <typename Label>
inline MultiRowsMetrics EvaluateMultiRows(const MultiRowsModel& model,const std::vector<ParsedRow>& rows,const std::vector<std::size_t>& selected,const std::vector<FeatureRef>& features,Label label)
{
    MultiRowsMetrics result;if(!model.available){result.unavailableReason=model.unavailableReason;return result;}if(selected.empty()){result.unavailableReason="empty_holdout";return result;}double loss=0,brier=0,correct=0;for(auto index:selected){const auto& row=rows[index];const int y=label(row).assignedClass;if(y<0||y>2){result.unavailableReason="invalid_directional_target";return result;}const auto p=MultiProbability(model,row,features);const double rowLoss=-std::log(std::max(p[y],1e-300));double rowBrier=0;std::size_t best=0;for(std::size_t c=0;c<3;++c){const double d=p[c]-(c==static_cast<std::size_t>(y));rowBrier+=d*d;if(p[c]>p[best])best=c;}rowBrier/=3;loss+=rowLoss;brier+=rowBrier;correct+=best==static_cast<std::size_t>(y);result.rowLoss.push_back(rowLoss);result.rowBrier.push_back(rowBrier);}result.available=true;result.logLoss=loss/selected.size();result.brier=brier/selected.size();result.accuracy=correct/selected.size();return result;
}

inline std::string Csv(std::string_view value) { if(value.find_first_of(",\"\r\n")==std::string_view::npos)return std::string(value);std::string out="\"";for(char c:value)out+=c=='\"'?"\"\"":std::string(1,c);return out+"\""; }
inline std::string Number(double value) { if(!std::isfinite(value))return "";std::ostringstream out;out<<std::setprecision(17)<<value;return out.str(); }

struct StateAccumulator { std::size_t rows=0;std::array<std::size_t,3> classes{};std::vector<double> returns;void Add(const ParsedTarget& target){if(!target.eligible)return;++rows;++classes[target.assignedClass];returns.push_back(target.terminalReturn);} };
inline void WriteStructuralRows(std::ofstream& out,const std::string& symbol,const std::vector<ParsedRow>& rows)
{
    for(const Partition partition:{Partition::Development,Partition::Validation,Partition::Pre2025LockTest})for(const bool h6:{false,true}){std::array<StateAccumulator,3> state{};for(const auto&row:rows)if(row.partition==partition){const auto&t=h6?row.h6:row.h4; if(t.eligible)state[static_cast<std::size_t>(StateFor(Row{row.symbol,row.timestamp,row.ordinal,row.baseline,row.fibonacci,{},{}}))].Add(t);}for(std::size_t s=0;s<3;++s){const auto&v=state[s];double mean=0;for(double x:v.returns)mean+=x;if(!v.returns.empty())mean/=v.returns.size();out<<symbol<<','<<PartitionName(partition)<<','<<(h6?"H6":"H4")<<','<<s<<','<<v.rows<<','<<v.classes[0]<<','<<v.classes[1]<<','<<v.classes[2]<<','<<Number(mean)<<','<<Number(Median(v.returns))<<'\n';}}
}

inline bool EventState(const ParsedRow& row) noexcept { return row.fibonacci[0] > .5f && (row.fibonacci[1] > 0.0f || row.fibonacci[12] > 0.0f); }
inline void WriteFibonacciLedgers(std::ofstream& out,const std::string& symbol,const std::vector<ParsedRow>& rows)
{
    for(std::size_t f=0;f<kFibonacciWidth;++f){std::vector<double> edges;if(f==0)edges={0.0,1.0};else{std::vector<double> values;for(const auto&r:rows)if(r.partition==Partition::Development)values.push_back(r.fibonacci[f]);std::sort(values.begin(),values.end());for(std::size_t q=1;q<=5&&!values.empty();++q)edges.push_back(values[(q*values.size()+4)/5-1]);edges.erase(std::unique(edges.begin(),edges.end()),edges.end());if(edges.size()<2)edges.clear();}
        for(const Partition partition:{Partition::Development,Partition::Validation,Partition::Pre2025LockTest})for(const bool h6:{false,true}){if(edges.empty()){out<<symbol<<','<<PartitionName(partition)<<','<<(h6?"H6":"H4")<<','<<f<<",,0,0,0,0,,,insufficient_distinct_development_values_after_tied_edge_collapse\n";continue;}std::vector<StateAccumulator> bins(edges.size());for(const auto&r:rows)if(r.partition==partition){const auto&t=h6?r.h6:r.h4;if(!t.eligible)continue;const double value=r.fibonacci[f];const auto bin=static_cast<std::size_t>(std::lower_bound(edges.begin(),edges.end(),value)-edges.begin());bins[bin].Add(t);}for(std::size_t b=0;b<bins.size();++b){const auto&v=bins[b];double mean=0;for(double x:v.returns)mean+=x;if(!v.returns.empty())mean/=v.returns.size();out<<symbol<<','<<PartitionName(partition)<<','<<(h6?"H6":"H4")<<','<<f<<','<<b<<','<<Number(edges[b])<<','<<v.rows<<','<<v.classes[0]<<','<<v.classes[1]<<','<<v.classes[2]<<','<<Number(mean)<<','<<Number(Median(v.returns))<<",\n";}}
    }
}

struct ConditionalRecord { std::string symbol;Partition partition;bool h6=false;double logLoss=std::numeric_limits<double>::quiet_NaN(),brier=std::numeric_limits<double>::quiet_NaN();bool available=false; };
inline void WriteEqualSymbolSummary(std::ofstream& out,const std::vector<ConditionalRecord>& records)
{
    out<<"partition,horizon,metric,symbols,available,positive,zero_or_negative,unavailable,median,min,max\n";
    for(const Partition partition:{Partition::Validation,Partition::Pre2025LockTest})for(const bool h6:{false,true})for(const bool brier:{false,true}){std::vector<double> values;std::string symbols;std::size_t positive=0,nonpositive=0,unavailable=0;for(const auto&r:records)if(r.partition==partition&&r.h6==h6){if(!symbols.empty())symbols+=';';symbols+=r.symbol;if(!r.available){++unavailable;continue;}const double value=brier?r.brier:r.logLoss;values.push_back(value);if(value>0)++positive;else ++nonpositive;}out<<PartitionName(partition)<<','<<(h6?"H6":"H4")<<','<<(brier?"brier":"log_loss")<<','<<Csv(symbols)<<','<<values.size()<<','<<positive<<','<<nonpositive<<','<<unavailable<<','<<Number(Median(values))<<','<<Number(values.empty()?std::numeric_limits<double>::quiet_NaN():*std::min_element(values.begin(),values.end()))<<','<<Number(values.empty()?std::numeric_limits<double>::quiet_NaN():*std::max_element(values.begin(),values.end()))<<'\n';}
}

inline std::vector<MonthlyDelta> Monthly(const std::vector<ParsedRow>& rows,const std::vector<std::size_t>& selected,const std::vector<double>& deltas)
{std::vector<std::int64_t> timestamps;timestamps.reserve(selected.size());for(auto i:selected)timestamps.push_back(rows[i].timestamp);return MonthlyDependenceAwareDeltas(timestamps,deltas);}

inline void Run(const Options& options)
{
    if(options.artifactDirectory.empty()||options.outputDirectory.empty()||options.codeCommit.empty())throw std::invalid_argument("fibonacci_pre2025_analysis_required_argument_missing");
    VerifyFrozenProtocolDocument("docs/phases/target-generation/FibonacciExtensions/FIBONACCI_LAYOUT9_INCREMENTAL_INFORMATION_PROTOCOL.md"); VerifyArtifactDirectory(options.artifactDirectory);
    if(std::filesystem::exists(options.outputDirectory))throw std::invalid_argument("fibonacci_pre2025_analysis_refuses_existing_output_directory");std::filesystem::create_directories(options.outputDirectory);
    std::ofstream structural(options.outputDirectory/"structural_ledgers.csv"), fibonacciLedger(options.outputDirectory/"fibonacci_ledgers.csv"), reconstruction(options.outputDirectory/"reconstruction.csv"), conditional(options.outputDirectory/"conditional_incremental.csv"), monthly(options.outputDirectory/"monthly_deltas.csv"), association(options.outputDirectory/"associations.csv"), crossSymbol(options.outputDirectory/"cross_symbol_equal_summary.csv");
    if(!structural||!fibonacciLedger||!reconstruction||!conditional||!monthly||!association||!crossSymbol)throw std::runtime_error("fibonacci_pre2025_result_open_failed");
    structural<<"symbol,partition,horizon,structural_state,rows,down,neutral,up,mean_terminal_log_return,median_terminal_log_return\n";
    fibonacciLedger<<"symbol,partition,horizon,fibonacci_column,bin,upper_edge,rows,down,neutral,up,mean_terminal_log_return,median_terminal_log_return,unavailable_reason\n";
    reconstruction<<"symbol,partition,fibonacci_column,diagnostic,available,unavailable_reason,rmse,r_squared,log_loss,brier\n";
    conditional<<"symbol,partition,horizon,subset,rows,baseline_log_loss,augmented_log_loss,delta_log_loss,baseline_brier,augmented_brier,delta_brier,baseline_accuracy,augmented_accuracy,delta_accuracy,available,unavailable_reason\n";
    monthly<<"symbol,partition,horizon,metric,calendar_month,rows,mean_delta,median_delta,p10_delta,p90_delta\n";
    association<<"symbol,partition,fibonacci_column,baseline_column,association_type,value,nearest_baseline_proxy\n";
    std::vector<ConditionalRecord> conditionalRecords;
    StreamPre2025Symbols(options.artifactDirectory,[&](const std::string& symbol,const std::vector<ParsedRow>& rows){
        std::cerr << "FIBONACCI_PRE2025_SYMBOL_START symbol=" << symbol << ",rows=" << rows.size() << '\n';
        WriteStructuralRows(structural,symbol,rows);WriteFibonacciLedgers(fibonacciLedger,symbol,rows);const auto transform=FitDevelopmentInputTransform(rows);const auto dev=EligibleRows(rows,Partition::Development,[](const auto&){return ParsedTarget{0,0,true};});
        const auto binary=FitBinaryRows(rows,dev,transform.baseline);
        std::cerr << "FIBONACCI_PRE2025_RECONSTRUCTION_BINARY_COMPLETE symbol=" << symbol << '\n';
        std::array<RidgeModel,kFibonacciWidth-1> ridgeModels{};
        for(std::size_t f=1;f<kFibonacciWidth;++f)ridgeModels[f-1]=FitRidge(rows,dev,transform.baseline,f);
        std::cerr << "FIBONACCI_PRE2025_RECONSTRUCTION_RIDGE_COMPLETE symbol=" << symbol << '\n';
        for(const Partition partition:{Partition::Development,Partition::Validation,Partition::Pre2025LockTest}){
            std::vector<std::size_t> selected;for(std::size_t i=0;i<rows.size();++i)if(rows[i].partition==partition)selected.push_back(i);
            const auto binaryMetric=EvaluateBinaryRows(binary,rows,selected,transform.baseline);reconstruction<<symbol<<','<<PartitionName(partition)<<",0,scale_validity_logistic,"<<binaryMetric.available<<','<<Csv(binaryMetric.unavailableReason)<<",,,"<<Number(binaryMetric.logLoss)<<','<<Number(binaryMetric.brier)<<'\n';
            for(std::size_t f=1;f<kFibonacciWidth;++f){const auto metric=EvaluateRidge(ridgeModels[f-1],rows,selected,transform.baseline,f);reconstruction<<symbol<<','<<PartitionName(partition)<<','<<f<<",ridge,"<<metric.available<<','<<Csv(metric.unavailableReason)<<','<<Number(metric.rmse)<<','<<Number(metric.rSquared)<<",,"<<'\n';}
            const auto matrix=CompleteAssociationMatrix([&](){std::vector<Row> converted;converted.reserve(selected.size());for(auto i:selected){const auto&r=rows[i];converted.push_back({r.symbol,r.timestamp,r.ordinal,r.baseline,r.fibonacci,{},{}});}return converted;}(),symbol,partition);
            for(std::size_t b=0;b<kBaselineWidth;++b)association<<symbol<<','<<PartitionName(partition)<<",0,"<<b<<",point_biserial,"<<Number(matrix.scaleValidityPointBiserial[b])<<','<<matrix.nearestBaselineProxy[0]<<'\n';
            for(std::size_t f=1;f<kFibonacciWidth;++f)for(std::size_t b=0;b<kBaselineWidth;++b)association<<symbol<<','<<PartitionName(partition)<<','<<f<<','<<b<<",spearman,"<<Number(matrix.spearman[f-1][b])<<','<<matrix.nearestBaselineProxy[f]<<'\n';
            std::cerr << "FIBONACCI_PRE2025_ASSOCIATION_COMPLETE symbol=" << symbol << ",partition=" << PartitionName(partition) << '\n';
        }
        for(const bool h6:{false,true}){
            std::cerr << "FIBONACCI_PRE2025_MULTINOMIAL_START symbol=" << symbol << ",horizon=" << (h6?"H6":"H4") << '\n';
            const auto label=[h6](const ParsedRow&r)->const ParsedTarget&{return h6?r.h6:r.h4;};const auto development=EligibleRows(rows,Partition::Development,label);const auto base=FitMultiRows(rows,development,transform.baseline,label);const auto augmented=FitMultiRows(rows,development,transform.augmented,label);
            std::cerr << "FIBONACCI_PRE2025_MULTINOMIAL_FIT_COMPLETE symbol=" << symbol << ",horizon=" << (h6?"H6":"H4") << '\n';
            for(const Partition partition:{Partition::Development,Partition::Validation,Partition::Pre2025LockTest}){const auto selected=EligibleRows(rows,partition,label);const auto bm=EvaluateMultiRows(base,rows,selected,transform.baseline,label);const auto am=EvaluateMultiRows(augmented,rows,selected,transform.augmented,label);const bool available=bm.available&&am.available&&bm.rowLoss.size()==am.rowLoss.size();const std::string reason=available?"":(!bm.available?bm.unavailableReason:am.unavailableReason);double dl=std::numeric_limits<double>::quiet_NaN(),db=dl,da=dl;if(available){dl=bm.logLoss-am.logLoss;db=bm.brier-am.brier;da=bm.accuracy-am.accuracy;}conditional<<symbol<<','<<PartitionName(partition)<<','<<(h6?"H6":"H4")<<",all,"<<selected.size()<<','<<Number(bm.logLoss)<<','<<Number(am.logLoss)<<','<<Number(dl)<<','<<Number(bm.brier)<<','<<Number(am.brier)<<','<<Number(db)<<','<<Number(bm.accuracy)<<','<<Number(am.accuracy)<<','<<Number(da)<<','<<available<<','<<Csv(reason)<<'\n';conditionalRecords.push_back({symbol,partition,h6,dl,db,available});const auto eventSelected=[&](){std::vector<std::size_t> r;for(auto i:selected)if(EventState(rows[i]))r.push_back(i);return r;}();const auto eb=EvaluateMultiRows(base,rows,eventSelected,transform.baseline,label);const auto ea=EvaluateMultiRows(augmented,rows,eventSelected,transform.augmented,label);const bool eventAvailable=eb.available&&ea.available&&eb.rowLoss.size()==ea.rowLoss.size();conditional<<symbol<<','<<PartitionName(partition)<<','<<(h6?"H6":"H4")<<",event_state,"<<eventSelected.size()<<','<<Number(eb.logLoss)<<','<<Number(ea.logLoss)<<','<<Number(eventAvailable?eb.logLoss-ea.logLoss:std::numeric_limits<double>::quiet_NaN())<<','<<Number(eb.brier)<<','<<Number(ea.brier)<<','<<Number(eventAvailable?eb.brier-ea.brier:std::numeric_limits<double>::quiet_NaN())<<','<<Number(eb.accuracy)<<','<<Number(ea.accuracy)<<','<<Number(eventAvailable?eb.accuracy-ea.accuracy:std::numeric_limits<double>::quiet_NaN())<<','<<eventAvailable<<','<<Csv(eventAvailable?"":(!eb.available?eb.unavailableReason:ea.unavailableReason))<<'\n';if(available&&(partition==Partition::Validation||partition==Partition::Pre2025LockTest)){std::vector<double> loss,brier;loss.reserve(bm.rowLoss.size());brier.reserve(bm.rowBrier.size());for(std::size_t i=0;i<bm.rowLoss.size();++i){loss.push_back(bm.rowLoss[i]-am.rowLoss[i]);brier.push_back(bm.rowBrier[i]-am.rowBrier[i]);}for(const auto&[metric,values]:std::array<std::pair<const char*,std::vector<double>>,2>{{{"log_loss",loss},{"brier",brier}}})for(const auto&block:Monthly(rows,selected,values))monthly<<symbol<<','<<PartitionName(partition)<<','<<(h6?"H6":"H4")<<','<<metric<<','<<block.calendarMonth<<','<<block.rows<<','<<Number(block.mean)<<','<<Number(block.median)<<','<<Number(block.p10)<<','<<Number(block.p90)<<'\n';}}}
        std::cerr << "FIBONACCI_PRE2025_SYMBOL_COMPLETE symbol=" << symbol << '\n';
    });
    WriteEqualSymbolSummary(crossSymbol,conditionalRecords);structural.close();fibonacciLedger.close();reconstruction.close();conditional.close();monthly.close();association.close();crossSymbol.close();
    const auto manifestPath=options.outputDirectory/"manifest.json";std::ofstream manifest(manifestPath);manifest<<"{\n\"runner_identity\":\""<<kRunnerIdentity<<"\",\n\"protocol_id\":\""<<kProtocolId<<"\",\n\"protocol_sha256\":\""<<kProtocolSha256<<"\",\n\"input_manifest_sha256\":\""<<FileSha256(options.artifactDirectory/"manifest.json")<<"\",\n\"input_rows_sha256\":\""<<FileSha256(options.artifactDirectory/"rows.csv")<<"\",\n\"code_commit\":\""<<options.codeCommit<<"\",\n\"confirmation_2025\":\"sealed_and_discarded_before_parsing\",\n\"lambda\":1.0,\n\"lbfgs_max_iterations\":250,\n\"lbfgs_gradient_infinity_tolerance\":1e-8,\n\"lbfgs_relative_objective_tolerance\":1e-12\n}\n";manifest.close();
    std::ofstream sums(options.outputDirectory/"sha256sums.txt");for(const char* name:{"structural_ledgers.csv","fibonacci_ledgers.csv","reconstruction.csv","conditional_incremental.csv","monthly_deltas.csv","associations.csv","cross_symbol_equal_summary.csv","manifest.json"})sums<<FileSha256(options.outputDirectory/name)<<"  "<<name<<'\n';
}
} // namespace EA::CausalFibonacciIncrementalInformation::Analysis
