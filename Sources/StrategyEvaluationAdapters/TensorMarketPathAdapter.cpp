#include "TensorMarketPathAdapter.hpp"

#include <chrono>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace EA::StrategyEvaluationAdapters
{
namespace
{

std::int64_t UnixSeconds(PriceTP timestamp)
{
    return std::chrono::duration_cast<std::chrono::seconds>(
        timestamp.time_since_epoch()).count();
}

StrategyEvaluation::MarketPathPoint PointAt(const Tensor& tensor,
                                            std::size_t sourceRow)
{
    const auto iterator = tensor.begin() +
        static_cast<std::ptrdiff_t>(sourceRow);
    StrategyEvaluation::MarketPathPoint point;
    point.sourceRow = sourceRow;
    point.timestampUnixSeconds = UnixSeconds(
        tensor.RawTimeAtIterator(iterator));
    point.open = tensor.RawOpenAtIterator(iterator);
    point.high = tensor.RawHighAtIterator(iterator);
    point.low = tensor.RawLowAtIterator(iterator);
    point.close = tensor.RawCloseAtIterator(iterator);
    return point;
}

} // namespace

StrategyEvaluation::AuthoritativeMarketPath AdaptTensorMarketPath(
    const Tensor& tensor,
    const TensorMarketPathAdapterContext& context,
    const std::vector<TensorInferenceDecision>& decisions)
{
    if (context.schemaVersion !=
        kTensorMarketPathAdapterContextSchemaVersion)
    {
        throw std::invalid_argument(
            "unsupported_tensor_market_path_adapter_context_version");
    }
    if (context.inferenceWindowSize == 0 ||
        context.predictionHorizon == 0)
    {
        throw std::invalid_argument(
            "invalid_tensor_market_path_window_or_horizon");
    }
    if (context.inferenceWindowSize - 1 >
        std::numeric_limits<std::size_t>::max() -
            context.predictionHorizon)
    {
        throw std::invalid_argument("tensor_market_path_offset_overflow");
    }
    const std::size_t terminalOffset =
        context.inferenceWindowSize - 1 + context.predictionHorizon;

    std::vector<StrategyEvaluation::StrategyEvaluationObservation>
        observations;
    observations.reserve(decisions.size());
    for (std::size_t ordinal = 0; ordinal < decisions.size(); ++ordinal)
    {
        const TensorInferenceDecision& decision = decisions[ordinal];
        if (decision.windowStartRow >= tensor.RowCount() ||
            terminalOffset > tensor.RowCount() - 1 - decision.windowStartRow)
        {
            throw std::invalid_argument(
                "tensor_market_path_window_out_of_bounds");
        }

        const std::size_t decisionRow = decision.windowStartRow +
            context.inferenceWindowSize - 1;
        const std::size_t terminalRow = decisionRow +
            context.predictionHorizon;
        const auto decisionIterator = tensor.begin() +
            static_cast<std::ptrdiff_t>(decisionRow);
        const auto terminalIterator = tensor.begin() +
            static_cast<std::ptrdiff_t>(terminalRow);

        StrategyEvaluation::StrategyEvaluationObservation observation;
        observation.observationOrdinal = ordinal;
        observation.inferenceWindowStartRow = decision.windowStartRow;
        observation.decisionRow = decisionRow;
        observation.terminalRow = terminalRow;
        observation.probabilities = decision.probabilities;
        observation.predictedClass =
            StrategyEvaluation::PredictedClassForProbabilities(
                decision.probabilities);
        observation.decisionClose =
            tensor.RawCloseAtIterator(decisionIterator);
        observation.terminalClose =
            tensor.RawCloseAtIterator(terminalIterator);
        observation.decisionTimestampUnixSeconds = UnixSeconds(
            tensor.RawTimeAtIterator(decisionIterator));
        observation.terminalTimestampUnixSeconds = UnixSeconds(
            tensor.RawTimeAtIterator(terminalIterator));
        observation.marketPath.reserve(context.predictionHorizon);
        for (std::size_t offset = 1;
             offset <= context.predictionHorizon; ++offset)
        {
            observation.marketPath.push_back(
                PointAt(tensor, decisionRow + offset));
        }
        observations.push_back(std::move(observation));
    }

    StrategyEvaluation::MarketPathProvenance provenance;
    provenance.adapterFamily = kTensorMarketPathAdapterFamily;
    provenance.adapterVersion = kTensorMarketPathAdapterVersion;
    provenance.modelId = context.modelId;
    provenance.inferenceScientificIdentityCanonical =
        context.inferenceScientificIdentityCanonical;
    provenance.inferenceScientificIdentityHash =
        context.inferenceScientificIdentityHash;
    provenance.symbol = tensor.TableName();
    provenance.inferenceWindowSize = context.inferenceWindowSize;
    provenance.predictionHorizon = context.predictionHorizon;
    provenance.evaluationStart = context.evaluationStart;
    provenance.evaluationEnd = context.evaluationEnd;
    provenance.barIntervalSeconds = 15 * 60;
    provenance.timestampSemantics =
        "source_new_york_civil_bar_start_converted_to_utc_unix_seconds_v1";
    provenance.ohlcIntervalSemantics =
        "postgresql_candlestick_15_minute_ask_ohlc_bucket_v1";
    provenance.priceDomain = "ask";
    provenance.marketDataSource =
        "postgresql_candlestick_function_order_by_dt_v1";
    provenance.marketDataSourceRelation = tensor.TableName();
    provenance.pathOrdering =
        "strict_timestamp_ascending_subsequent_tensor_row_order_v1";
    provenance.metricDefinitionCanonical =
        context.metricDefinitionCanonical;
    provenance.metricDefinitionHash = context.metricDefinitionHash;
    return StrategyEvaluation::BuildAuthoritativeMarketPath(
        std::move(provenance), std::move(observations));
}

} // namespace EA::StrategyEvaluationAdapters
