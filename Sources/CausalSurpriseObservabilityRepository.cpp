#include "CausalSurpriseObservabilityRepository.hpp"

#include "CanonicalSymbol.hpp"
#include "FeatureAblation.hpp"
#include "HistoricalFxTimestamp.hpp"
#include "ModelInputExpansion.hpp"

#include <optional>
#include <stdexcept>
#include <string>

namespace EA::CausalSurpriseObservability
{
namespace
{

template <typename Value>
std::optional<Value> OptionalValue(
    const pqxx::row& row,
    const char* column)
{
    if (row[column].is_null()) return std::nullopt;
    return row[column].as<Value>();
}

} // namespace

ExperimentContext LoadExperimentContext(
    pqxx::transaction_base& transaction,
    long long experimentId)
{
    if (experimentId <= 0)
        throw std::invalid_argument(
            "causal_surprise_observability_experiment_id_must_be_positive");

    const pqxx::result rows = transaction.exec(
        "SELECT experiment_id,symbol,prediction_horizon,"
        "train_start::text,train_end::text,infer_start::text,infer_end::text,"
        "model_input_width,model_input_semantic_layout_version,"
        "donchian20_mode,feature_warmup_scope,donchian_lookback,"
        "feature_ablation_mask FROM experiment WHERE experiment_id=$1;",
        pqxx::params{experimentId});
    if (rows.size() != 1)
        throw std::runtime_error(
            "causal_surprise_observability_experiment_not_found:" +
            std::to_string(experimentId));

    const pqxx::row row = rows.one_row();
    ExperimentContext context;
    context.experimentId = row["experiment_id"].as<long long>();
    context.symbol = CanonicalSymbol::Normalize(
        row["symbol"].as<std::string>());
    context.predictionHorizon = row["prediction_horizon"].as<int>();
    context.trainStart = row["train_start"].as<std::string>();
    context.trainEnd = row["train_end"].as<std::string>();
    context.inferStart = OptionalValue<std::string>(row, "infer_start");
    context.inferEnd = OptionalValue<std::string>(row, "infer_end");
    context.modelInputWidth = OptionalValue<int>(row, "model_input_width");
    context.modelInputSemanticLayoutVersion = OptionalValue<int>(
        row, "model_input_semantic_layout_version");
    context.donchian20Mode = ParseDonchian20Mode(
        row["donchian20_mode"].as<std::string>());
    context.featureWarmupScope = ParseFeatureWarmupScope(
        row["feature_warmup_scope"].as<std::string>());
    context.donchianLookback = ParseDonchianLookback(
        row["donchian_lookback"].as<std::string>());
    context.featureAblationMask = FeatureAblationMask::Parse(
        row["feature_ablation_mask"].as<std::string>()).CanonicalText();

    if (context.modelInputWidth.has_value() !=
        context.modelInputSemanticLayoutVersion.has_value())
    {
        throw std::runtime_error(
            "causal_surprise_observability_partial_model_input_identity");
    }
    if (context.modelInputWidth)
    {
        if (*context.modelInputWidth <= 0)
            throw std::runtime_error(
                "causal_surprise_observability_invalid_model_input_width");
        ValidateModelInputSemanticMetadataForExpansion(
            kModelInputSemanticMetaSchemaVersion,
            *context.modelInputSemanticLayoutVersion,
            static_cast<std::size_t>(*context.modelInputWidth));
    }
    (void)ResolveRanges(context, Scope::train);
    return context;
}

BarPopulation LoadBarPopulation(
    pqxx::transaction_base& transaction,
    const std::string& symbol,
    const std::string& sourceStart,
    const std::string& outputStart,
    const std::string& outputEnd)
{
    const std::string canonicalSymbol = CanonicalSymbol::Normalize(symbol);
    if (sourceStart.empty() || outputStart.empty() || outputEnd.empty() ||
        outputStart >= outputEnd)
    {
        throw std::invalid_argument(
            "causal_surprise_observability_invalid_bar_range");
    }

    const pqxx::result rows = transaction.exec(
        "SELECT dt::text AS dt_text,"
        "(dt < $3::timestamp) AS warmup "
        "FROM candlestick($1::text,15,'minute',$2::timestamp,"
        "$4::timestamp) ORDER BY dt;",
        pqxx::params{
            canonicalSymbol, sourceStart, outputStart, outputEnd});

    BarPopulation population;
    population.sourceBarStarts.reserve(rows.size());
    bool observedOutputRow = false;
    for (const pqxx::row& row : rows)
    {
        const bool warmup = row["warmup"].as<bool>();
        if (warmup && observedOutputRow)
            throw std::runtime_error(
                "causal_surprise_observability_nonprefix_warmup_row");
        observedOutputRow = observedOutputRow || !warmup;
        if (warmup) ++population.warmupRowCount;

        PriceTP timestamp;
        const std::string text = row["dt_text"].as<std::string>();
        if (!HistoricalFxTimestamp::ParseNewYorkCivilTimestamp(
                text, timestamp))
        {
            throw std::runtime_error(
                "causal_surprise_observability_invalid_fx_timestamp:" +
                text);
        }
        population.sourceBarStarts.push_back(timestamp);
    }
    return population;
}

} // namespace EA::CausalSurpriseObservability
