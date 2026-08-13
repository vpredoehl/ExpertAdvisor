#include "ExperimentRecommendationCampaignOutcomeAssessmentRepository.hpp"

#include "ExperimentRecommendation.hpp"
#include "ExperimentRecommendationCampaignApprovalRepository.hpp"
#include "ExperimentRecommendationCampaignMaterializationRepository.hpp"
#include "ExperimentRecommendationCampaignStatusRepository.hpp"
#include "ExperimentRecommendationConversionRepository.hpp"
#include "ExperimentRecommendationRepository.hpp"
#include "CanonicalSymbol.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace EA::ExperimentRecommendation
{
namespace
{

using Consistency =
    RecommendationCampaignOutcomeAssessmentConsistencyState;
using Context = RecommendationCampaignOutcomeAssessmentComparisonContext;
using Metric = RecommendationCampaignOutcomeAssessmentMetric;
using MetricCollection =
    RecommendationCampaignOutcomeAssessmentMetricCollection;
using MetricSupport =
    RecommendationCampaignOutcomeAssessmentMetricComparisonSupport;
using ResultEvidence =
    RecommendationCampaignOutcomeAssessmentResultEvidence;
using ResultIdentity =
    RecommendationCampaignOutcomeAssessmentResultIdentity;
using ScientificMember =
    RecommendationCampaignOutcomeAssessmentScientificMemberEvidence;
using SourceEvidence =
    RecommendationCampaignOutcomeAssessmentSourceEvidence;

template <typename Value>
std::optional<Value> OptionalValue(
    const pqxx::row& row,
    const char* column)
{
    if (row[column].is_null()) return std::nullopt;
    return row[column].as<Value>();
}

std::optional<std::string> OptionalText(
    const pqxx::row& row,
    const char* column)
{
    return OptionalValue<std::string>(row, column);
}

template <typename Values>
std::pair<std::string, pqxx::params> IdPlaceholders(const Values& values)
{
    pqxx::params parameters;
    std::ostringstream placeholders;
    std::size_t index = 0;
    for (const auto value : values)
    {
        if (index != 0) placeholders << ',';
        placeholders << '$' << index + 1;
        parameters.append(value);
        ++index;
    }
    return {placeholders.str(), std::move(parameters)};
}

struct SourceRecord
{
    long long recommendationId = -1;
    long long sourceExperimentId = -1;
    std::optional<long long> sourceModelId;
    std::optional<long long> sourceAnalysisId;
    std::string sourceSymbol;
    int sourcePredictionHorizon = 0;
    std::optional<double> leaderScore;
    std::optional<double> inferenceAccuracy;
};

struct ExperimentRecord
{
    long long experimentId = -1;
    std::string symbol;
    int predictionHorizon = 0;
    double threshold = 0.0;
    std::optional<std::string> inferenceRangeStart;
    std::optional<std::string> inferenceRangeEnd;
    std::optional<long long> lastModelId;
};

struct InferenceRecord
{
    long long id = -1;
    long long modelId = -1;
    std::string symbol;
    int predictionHorizon = 0;
    double threshold = 0.0;
    int windowSize = 0;
    int labelRuleId = 0;
    int targetType = 0;
    std::string rangeStart;
    std::string rangeEnd;
    std::string status;
};

struct AnalysisRecord
{
    long long analysisId = -1;
    long long experimentId = -1;
    std::optional<long long> modelId;
    std::string status;
    std::string scope;
    std::optional<double> inferenceAccuracy;
    std::optional<double> leaderScore;
};

class CanonicalReader
{
public:
    explicit CanonicalReader(std::string_view text) : text_(text) {}

    void Expect(std::string_view expected)
    {
        if (!text_.substr(position_).starts_with(expected))
            throw std::invalid_argument(
                "campaign_outcome_assessment_source_invocation_invalid");
        position_ += expected.size();
    }

    std::string_view ReadUntil(std::string_view delimiter)
    {
        const std::size_t found = text_.find(delimiter, position_);
        if (found == std::string_view::npos)
            throw std::invalid_argument(
                "campaign_outcome_assessment_source_invocation_invalid");
        const auto value = text_.substr(position_, found - position_);
        position_ = found + delimiter.size();
        return value;
    }

    std::string_view Read(std::size_t length)
    {
        if (length > text_.size() - position_)
            throw std::invalid_argument(
                "campaign_outcome_assessment_source_invocation_invalid");
        const auto value = text_.substr(position_, length);
        position_ += length;
        return value;
    }

    std::string_view Remaining() const
    {
        return text_.substr(position_);
    }

private:
    std::string_view text_;
    std::size_t position_ = 0;
};

template <typename Integer>
Integer ParseCanonicalInteger(std::string_view text)
{
    Integer value{};
    const auto result = std::from_chars(
        text.data(), text.data() + text.size(), value);
    if (text.empty() || result.ec != std::errc{} ||
        result.ptr != text.data() + text.size() ||
        std::to_string(value) != text)
        throw std::invalid_argument(
            "campaign_outcome_assessment_source_invocation_invalid");
    return value;
}

double ParseCanonicalDouble(std::string_view text)
{
    double value = 0.0;
    const auto result = std::from_chars(text.data(),
        text.data() + text.size(), value, std::chars_format::general);
    if (text.empty() || result.ec != std::errc{} ||
        result.ptr != text.data() + text.size() ||
        CanonicalRecommendationDouble(value) != text)
        throw std::invalid_argument(
            "campaign_outcome_assessment_source_invocation_invalid");
    return value;
}

std::optional<double> ParseOptionalCanonicalDouble(std::string_view text)
{
    if (text == "NULL") return std::nullopt;
    return ParseCanonicalDouble(text);
}

std::optional<long long> ParseOptionalCanonicalLongLong(
    std::string_view text)
{
    if (text == "NULL") return std::nullopt;
    return ParseCanonicalInteger<long long>(text);
}

std::optional<std::string> ParseOptionalCanonicalDate(
    std::string_view text)
{
    if (text == "NULL") return std::nullopt;
    return CanonicalExperimentDateText(std::string{text});
}

ExperimentInvocationConfiguration ParseSourceInvocation(
    const std::string& canonical)
{
    CanonicalReader invocation{canonical};
    invocation.Expect(
        "experiment_recommendation_invocation_v2;"
        "semantic_configuration_length=");
    const std::size_t semanticLength = ParseCanonicalInteger<std::size_t>(
        invocation.ReadUntil(";semantic_configuration="));
    const std::string semantic{invocation.Read(semanticLength)};
    invocation.Expect(";checkpoint_interval=");
    const int checkpointInterval = ParseCanonicalInteger<int>(
        invocation.ReadUntil(";resume_model_id="));
    const auto resumeModelId = ParseOptionalCanonicalLongLong(
        invocation.Remaining());

    CanonicalReader configuration{semantic};
    const bool legacySemantic =
        configuration.Remaining().starts_with(
            "experiment_recommendation_semantic_configuration_v3;");
    if (legacySemantic)
        configuration.Expect(
            "experiment_recommendation_semantic_configuration_v3;symbol=");
    else
        configuration.Expect(
            "experiment_recommendation_semantic_configuration_v4;symbol=");
    ExperimentInvocationConfiguration parsed;
    parsed.configuration.symbol = std::string{
        configuration.ReadUntil(";prediction_horizon=")};
    parsed.configuration.predictionHorizon = ParseCanonicalInteger<int>(
        configuration.ReadUntil(";label_threshold="));
    parsed.configuration.labelThreshold = ParseCanonicalDouble(
        configuration.ReadUntil(";core_lr_mult="));
    parsed.configuration.coreLrMult = ParseOptionalCanonicalDouble(
        configuration.ReadUntil(";head_lr_mult="));
    parsed.configuration.headLrMult = ParseOptionalCanonicalDouble(
        configuration.ReadUntil(";target_epochs="));
    parsed.configuration.targetEpochs = ParseCanonicalInteger<int>(
        configuration.ReadUntil(";train_start_date="));
    parsed.configuration.trainStartDate = CanonicalExperimentDateText(
        std::string{configuration.ReadUntil(";train_end_date=")});
    parsed.configuration.trainEndDate = CanonicalExperimentDateText(
        std::string{configuration.ReadUntil(";infer_start_date=")});
    parsed.configuration.inferStartDate = ParseOptionalCanonicalDate(
        configuration.ReadUntil(";infer_end_date="));
    if (legacySemantic)
    {
        parsed.configuration.inferEndDate = ParseOptionalCanonicalDate(
            configuration.Remaining());
    }
    else
    {
        parsed.configuration.inferEndDate = ParseOptionalCanonicalDate(
            configuration.ReadUntil(";donchian20_mode="));
        parsed.configuration.donchian20Mode = ParseDonchian20Mode(
            std::string{configuration.Remaining()});
    }
    parsed.checkpointInterval = checkpointInterval;
    parsed.resumeModelId = resumeModelId;

    const auto validated = BuildRecommendationInvocationIdentity(parsed);
    if (!legacySemantic && validated.canonicalText != canonical)
        throw std::invalid_argument(
            "campaign_outcome_assessment_source_invocation_invalid");
    return validated.invocation;
}

std::map<long long, SourceRecord> LoadSources(
    pqxx::transaction_base& transaction,
    const std::vector<long long>& recommendationIds)
{
    if (recommendationIds.empty()) return {};
    const auto [placeholders, parameters] =
        IdPlaceholders(recommendationIds);
    const pqxx::result rows = transaction.exec(
        "SELECT recommendation_id,source_experiment_id,source_model_id,"
        "source_analysis_id,source_symbol,source_prediction_horizon,"
        "source_leader_score,source_infer_accuracy "
        "FROM experiment_recommendation WHERE recommendation_id IN (" +
            placeholders + ") ORDER BY recommendation_id;",
        parameters);
    std::map<long long, SourceRecord> result;
    for (const auto& row : rows)
    {
        SourceRecord value;
        value.recommendationId = row["recommendation_id"].as<long long>();
        value.sourceExperimentId =
            row["source_experiment_id"].as<long long>();
        value.sourceModelId = OptionalValue<long long>(row, "source_model_id");
        value.sourceAnalysisId =
            OptionalValue<long long>(row, "source_analysis_id");
        value.sourceSymbol = row["source_symbol"].as<std::string>();
        value.sourcePredictionHorizon =
            row["source_prediction_horizon"].as<int>();
        value.leaderScore =
            OptionalValue<double>(row, "source_leader_score");
        value.inferenceAccuracy =
            OptionalValue<double>(row, "source_infer_accuracy");
        if (!result.emplace(value.recommendationId, std::move(value)).second)
            throw std::runtime_error(
                "campaign_outcome_assessment_duplicate_source_record");
    }
    return result;
}

std::map<long long, ExperimentRecord> LoadExperiments(
    pqxx::transaction_base& transaction,
    const std::set<long long>& experimentIds)
{
    if (experimentIds.empty()) return {};
    auto [placeholders, parameters] = IdPlaceholders(experimentIds);
    const std::size_t timeZoneParameter = experimentIds.size() + 1;
    parameters.append(kRecommendationDateTimeZone);
    const std::string timeZone = "$" +
        std::to_string(timeZoneParameter);
    const pqxx::result rows = transaction.exec(
        "SELECT experiment_id,symbol,prediction_horizon,c_next_threshold,"
        "CASE WHEN infer_start IS NULL THEN NULL ELSE to_char(infer_start AT "
        "TIME ZONE " + timeZone + ",'YYYY-MM-DD') END AS infer_start,"
        "CASE WHEN infer_end IS NULL THEN NULL ELSE to_char(infer_end AT "
        "TIME ZONE " + timeZone + ",'YYYY-MM-DD') END AS infer_end,"
        "last_model_id FROM experiment WHERE experiment_id IN (" +
        placeholders + ") ORDER BY experiment_id;",
        parameters);
    std::map<long long, ExperimentRecord> result;
    for (const auto& row : rows)
    {
        ExperimentRecord value;
        value.experimentId = row["experiment_id"].as<long long>();
        value.symbol = row["symbol"].as<std::string>();
        value.predictionHorizon =
            row["prediction_horizon"].as<int>();
        value.threshold = row["c_next_threshold"].as<double>();
        value.inferenceRangeStart = OptionalText(row, "infer_start");
        value.inferenceRangeEnd = OptionalText(row, "infer_end");
        value.lastModelId = OptionalValue<long long>(row, "last_model_id");
        if (!result.emplace(value.experimentId, std::move(value)).second)
            throw std::runtime_error(
                "campaign_outcome_assessment_duplicate_experiment_record");
    }
    return result;
}

std::map<long long, std::optional<long long>> LoadModels(
    pqxx::transaction_base& transaction,
    const std::set<long long>& modelIds)
{
    if (modelIds.empty()) return {};
    const auto [placeholders, parameters] = IdPlaceholders(modelIds);
    const pqxx::result rows = transaction.exec(
        "SELECT model_id,experiment_id FROM model WHERE model_id IN (" +
            placeholders + ") ORDER BY model_id;",
        parameters);
    std::map<long long, std::optional<long long>> result;
    for (const auto& row : rows)
    {
        const long long id = row["model_id"].as<long long>();
        if (!result.emplace(
                id, OptionalValue<long long>(row, "experiment_id")).second)
            throw std::runtime_error(
                "campaign_outcome_assessment_duplicate_model_record");
    }
    return result;
}

std::vector<InferenceRecord> LoadInferenceRecords(
    pqxx::transaction_base& transaction,
    const std::set<long long>& modelIds)
{
    if (modelIds.empty()) return {};
    const auto [placeholders, parameters] = IdPlaceholders(modelIds);
    const pqxx::result rows = transaction.exec(
        "SELECT id,model_id,symbol,prediction_horizon,threshold_logret,"
        "window_size,label_rule_id,target_type,from_date,to_date,status "
        "FROM inference_eval_result WHERE model_id IN (" + placeholders +
        ") AND inference_scope='final' AND checkpoint_eval_id IS NULL "
        "ORDER BY model_id,id;",
        parameters);
    std::vector<InferenceRecord> result;
    result.reserve(rows.size());
    for (const auto& row : rows)
    {
        InferenceRecord value;
        value.id = row["id"].as<long long>();
        value.modelId = row["model_id"].as<long long>();
        value.symbol = row["symbol"].as<std::string>();
        value.predictionHorizon =
            row["prediction_horizon"].as<int>();
        value.threshold = row["threshold_logret"].as<double>();
        value.windowSize = row["window_size"].as<int>();
        value.labelRuleId = row["label_rule_id"].as<int>();
        value.targetType = row["target_type"].as<int>();
        value.rangeStart = row["from_date"].as<std::string>();
        value.rangeEnd = row["to_date"].as<std::string>();
        value.status = row["status"].as<std::string>();
        result.push_back(std::move(value));
    }
    return result;
}

std::vector<AnalysisRecord> LoadAnalysisRecords(
    pqxx::transaction_base& transaction,
    const std::set<long long>& experimentIds)
{
    if (experimentIds.empty()) return {};
    const auto [placeholders, parameters] = IdPlaceholders(experimentIds);
    const pqxx::result rows = transaction.exec(
        "SELECT analysis_id,experiment_id,model_id,analysis_status,"
        "analysis_scope,infer_accuracy,leader_score "
        "FROM experiment_analysis_result WHERE experiment_id IN (" +
        placeholders + ") AND analysis_scope='final' ORDER BY "
        "experiment_id,analysis_id;",
        parameters);
    std::vector<AnalysisRecord> result;
    result.reserve(rows.size());
    for (const auto& row : rows)
    {
        AnalysisRecord value;
        value.analysisId = row["analysis_id"].as<long long>();
        value.experimentId = row["experiment_id"].as<long long>();
        value.modelId = OptionalValue<long long>(row, "model_id");
        value.status = row["analysis_status"].as<std::string>();
        value.scope = row["analysis_scope"].as<std::string>();
        value.inferenceAccuracy =
            OptionalValue<double>(row, "infer_accuracy");
        value.leaderScore = OptionalValue<double>(row, "leader_score");
        result.push_back(std::move(value));
    }
    return result;
}

struct InferenceSelection
{
    std::string symbol;
    int predictionHorizon = 0;
    double threshold = 0.0;
    std::optional<std::string> rangeStart;
    std::optional<std::string> rangeEnd;
};

InferenceSelection SelectionFrom(const ExperimentRecord& experiment)
{
    return {experiment.symbol, experiment.predictionHorizon,
        experiment.threshold, experiment.inferenceRangeStart,
        experiment.inferenceRangeEnd};
}

InferenceSelection SelectionFrom(
    const ExperimentInvocationConfiguration& invocation)
{
    return {invocation.configuration.symbol,
        invocation.configuration.predictionHorizon,
        invocation.configuration.labelThreshold,
        invocation.configuration.inferStartDate,
        invocation.configuration.inferEndDate};
}

std::vector<const InferenceRecord*> ExactInferenceRecords(
    const std::vector<InferenceRecord>& records,
    const InferenceSelection& selection,
    long long modelId,
    bool completedOnly)
{
    std::vector<const InferenceRecord*> exact;
    if (!selection.rangeStart || !selection.rangeEnd)
        return exact;
    for (const auto& record : records)
    {
        if (record.modelId == modelId &&
            record.symbol == selection.symbol &&
            record.predictionHorizon == selection.predictionHorizon &&
            record.threshold == selection.threshold &&
            record.rangeStart == *selection.rangeStart &&
            record.rangeEnd == *selection.rangeEnd &&
            (!completedOnly || record.status == "completed"))
            exact.push_back(&record);
    }
    return exact;
}

std::vector<const AnalysisRecord*> ExactAnalysisRecords(
    const std::vector<AnalysisRecord>& records,
    long long experimentId,
    long long modelId)
{
    std::vector<const AnalysisRecord*> exact;
    for (const auto& record : records)
        if (record.experimentId == experimentId &&
            record.modelId == std::optional<long long>{modelId} &&
            record.scope == "final")
            exact.push_back(&record);
    return exact;
}

std::optional<Context> ContextFrom(
    const InferenceRecord& record,
    Consistency& consistency)
{
    const std::string labelDefinition =
        "inference_eval_result_label_v1;label_rule_id=" +
        std::to_string(record.labelRuleId) +
        ";target_type=" + std::to_string(record.targetType);
    if (record.symbol.empty() || record.predictionHorizon <= 0 ||
        record.windowSize <= 0 || !std::isfinite(record.threshold) ||
        record.rangeStart.empty() || record.rangeEnd.empty())
    {
        consistency = Consistency::Inconsistent;
        return std::nullopt;
    }
    try
    {
        return Context(EA::CanonicalSymbol::Normalize(record.symbol),
            record.predictionHorizon, record.threshold, record.windowSize,
            labelDefinition, CanonicalExperimentDateText(record.rangeStart),
            CanonicalExperimentDateText(record.rangeEnd));
    }
    catch (const std::exception&)
    {
        consistency = Consistency::Inconsistent;
        return std::nullopt;
    }
}

std::optional<double> FiniteMetric(
    std::optional<double> value,
    Consistency& consistency)
{
    if (value && !std::isfinite(*value))
    {
        consistency = Consistency::Inconsistent;
        return std::nullopt;
    }
    return value;
}

std::vector<Metric> Metrics(
    std::optional<double> inferenceAccuracy,
    std::optional<double> leaderScore,
    bool identitiesPresent,
    Consistency& consistency)
{
    if (!identitiesPresent) return {};
    return {
        Metric("inference_accuracy",
            FiniteMetric(inferenceAccuracy, consistency),
            MetricSupport::NumericDelta),
        Metric("leader_score", FiniteMetric(leaderScore, consistency),
            MetricSupport::NumericDelta)};
}

std::optional<SourceEvidence> BuildSourceEvidence(
    const RecommendationCampaignStatusMember& member,
    const std::map<long long, SourceRecord>& sources,
    const std::map<long long,
        const PersistedRecommendationConversionProposal*>& proposals,
    const std::map<long long, std::optional<long long>>& models,
    const std::vector<InferenceRecord>& inferenceRecords,
    const std::vector<AnalysisRecord>& analysisRecords,
    Consistency& consistency)
{
    const auto source = sources.find(member.recommendationId);
    const auto proposal = proposals.find(member.proposalId);
    if (source == sources.end() || proposal == proposals.end() ||
        source->second.sourceExperimentId != member.sourceExperimentId)
    {
        consistency = Consistency::Inconsistent;
        return std::nullopt;
    }

    ExperimentInvocationConfiguration sourceInvocation;
    try
    {
        sourceInvocation = ParseSourceInvocation(
            proposal->second->proposal.sourceInvocationCanonical);
    }
    catch (const std::exception&)
    {
        consistency = Consistency::Inconsistent;
        return std::nullopt;
    }
    try
    {
        if (EA::CanonicalSymbol::Normalize(source->second.sourceSymbol) !=
                sourceInvocation.configuration.symbol ||
            source->second.sourcePredictionHorizon !=
                sourceInvocation.configuration.predictionHorizon)
            consistency = Consistency::Inconsistent;
    }
    catch (const std::exception&)
    {
        consistency = Consistency::Inconsistent;
    }
    if (proposal->second->proposal.recommendationId !=
            member.recommendationId ||
        proposal->second->proposal.sourceExperimentId !=
            member.sourceExperimentId)
    {
        consistency = Consistency::Inconsistent;
        return std::nullopt;
    }

    if (!source->second.sourceModelId)
    {
        if (source->second.sourceAnalysisId)
            consistency = Consistency::Inconsistent;
        return std::nullopt;
    }
    if (*source->second.sourceModelId <= 0)
    {
        consistency = Consistency::Inconsistent;
        return std::nullopt;
    }

    const long long modelId = *source->second.sourceModelId;
    const auto model = models.find(modelId);
    if (model == models.end() || !model->second ||
        model->second != std::optional<long long>{member.sourceExperimentId})
        consistency = Consistency::Inconsistent;

    const auto exactInference = ExactInferenceRecords(
        inferenceRecords, SelectionFrom(sourceInvocation), modelId, true);
    if (exactInference.size() != 1)
    {
        if (exactInference.size() > 1)
            consistency = Consistency::Inconsistent;
        return std::nullopt;
    }
    if (exactInference.front()->id <= 0)
    {
        consistency = Consistency::Inconsistent;
        return std::nullopt;
    }
    const auto context = ContextFrom(*exactInference.front(), consistency);
    if (!context) return std::nullopt;

    bool analysisIdentityPresent = false;
    std::optional<long long> sourceAnalysisId =
        source->second.sourceAnalysisId;
    if (sourceAnalysisId && *sourceAnalysisId <= 0)
    {
        consistency = Consistency::Inconsistent;
        sourceAnalysisId.reset();
    }
    if (sourceAnalysisId)
    {
        const auto found = std::find_if(analysisRecords.begin(),
            analysisRecords.end(), [&](const AnalysisRecord& record)
        {
            return record.analysisId == *sourceAnalysisId;
        });
        if (found != analysisRecords.end())
        {
            analysisIdentityPresent = found->experimentId ==
                    member.sourceExperimentId &&
                found->modelId == source->second.sourceModelId &&
                found->scope == "final" && found->status == "completed";
            if (!analysisIdentityPresent)
                consistency = Consistency::Inconsistent;
        }
        else
            consistency = Consistency::Inconsistent;
    }

    const bool metricIdentityPresent =
        analysisIdentityPresent || sourceAnalysisId.has_value();
    return SourceEvidence(member.sourceExperimentId,
        source->second.sourceModelId, sourceAnalysisId, *context,
        MetricCollection(Metrics(source->second.inferenceAccuracy,
            source->second.leaderScore, metricIdentityPresent,
            consistency)));
}

std::optional<ResultEvidence> BuildResultEvidence(
    const RecommendationCampaignStatusMember& member,
    const std::map<long long, ExperimentRecord>& experiments,
    const std::map<long long, std::optional<long long>>& models,
    const std::vector<InferenceRecord>& inferenceRecords,
    const std::vector<AnalysisRecord>& analysisRecords,
    Consistency& consistency)
{
    std::vector<ResultIdentity> identities;
    std::optional<Context> context;
    std::vector<Metric> metrics;

    if (member.experimentId && member.modelId)
    {
        const auto experiment = experiments.find(*member.experimentId);
        const auto model = models.find(*member.modelId);
        if (model == models.end() || !model->second ||
            model->second != member.experimentId)
            consistency = Consistency::Inconsistent;
        if (experiment != experiments.end())
        {
            const auto exactInference = ExactInferenceRecords(
                inferenceRecords, SelectionFrom(experiment->second),
                *member.modelId, false);
            for (const auto* record : exactInference)
                if (record->id > 0)
                    identities.emplace_back(
                        "inference_eval_result", record->id);
                else
                    consistency = Consistency::Inconsistent;
            if (exactInference.size() == 1)
            {
                const auto mapped = ContextFrom(
                    *exactInference.front(), consistency);
                if (mapped) context.emplace(*mapped);
            }
            else if (exactInference.size() > 1)
                consistency = Consistency::Inconsistent;

            const auto exactAnalysis = ExactAnalysisRecords(analysisRecords,
                *member.experimentId, *member.modelId);
            for (const auto* record : exactAnalysis)
                if (record->analysisId > 0)
                    identities.emplace_back(
                        "experiment_analysis_result", record->analysisId);
                else
                    consistency = Consistency::Inconsistent;
            if (exactAnalysis.size() == 1)
                metrics = Metrics(exactAnalysis.front()->inferenceAccuracy,
                    exactAnalysis.front()->leaderScore, true, consistency);
            else if (exactAnalysis.size() > 1)
                consistency = Consistency::Inconsistent;
        }
    }

    const bool succeeded = member.terminalResult ==
        RecommendationCampaignStatusTerminalResult::succeeded;
    if (!succeeded && identities.empty()) return std::nullopt;
    return ResultEvidence(member.experimentId, member.modelId,
        std::move(identities), std::move(context),
        MetricCollection(std::move(metrics)));
}

std::vector<ScientificMember> LoadScientificMembers(
    pqxx::transaction_base& transaction,
    const RecommendationCampaignStatusSnapshot& status)
{
    std::vector<long long> recommendationIds;
    std::vector<long long> proposalIds;
    std::set<long long> experimentIds;
    std::set<long long> modelIds;
    recommendationIds.reserve(status.members.size());
    proposalIds.reserve(status.members.size());
    for (const auto& member : status.members)
    {
        recommendationIds.push_back(member.recommendationId);
        proposalIds.push_back(member.proposalId);
        experimentIds.insert(member.sourceExperimentId);
        if (member.experimentId) experimentIds.insert(*member.experimentId);
    }
    const auto sources = LoadSources(transaction, recommendationIds);
    const auto proposalValues =
        ListRecommendationConversionProposalsByIds(transaction, proposalIds);
    std::map<long long, const PersistedRecommendationConversionProposal*>
        proposals;
    for (const auto& proposal : proposalValues)
        if (!proposals.emplace(proposal.proposalId, &proposal).second)
            throw std::runtime_error(
                "campaign_outcome_assessment_duplicate_proposal_record");
    for (const auto& [unused, source] : sources)
    {
        static_cast<void>(unused);
        if (source.sourceModelId) modelIds.insert(*source.sourceModelId);
    }
    for (const auto& member : status.members)
        if (member.modelId) modelIds.insert(*member.modelId);

    const auto experiments = LoadExperiments(transaction, experimentIds);
    const auto models = LoadModels(transaction, modelIds);
    const auto inferenceRecords =
        LoadInferenceRecords(transaction, modelIds);
    const auto analysisRecords =
        LoadAnalysisRecords(transaction, experimentIds);

    std::vector<ScientificMember> result;
    result.reserve(status.members.size());
    for (const auto& member : status.members)
    {
        Consistency consistency = Consistency::Consistent;
        auto sourceEvidence = BuildSourceEvidence(member, sources, proposals,
            models, inferenceRecords, analysisRecords, consistency);
        auto resultEvidence = BuildResultEvidence(member, experiments,
            models, inferenceRecords, analysisRecords,
            consistency);
        result.push_back({member.memberOrdinal, consistency,
            std::move(sourceEvidence), std::move(resultEvidence)});
    }
    return result;
}

} // namespace

RecommendationCampaignOutcomeAssessmentRequest
NormalizeRecommendationCampaignOutcomeAssessmentRequest(
    const RecommendationCampaignOutcomeAssessmentRequest& request)
{
    if (request.materializationId <= 0)
        throw std::invalid_argument(
            "campaign_outcome_assessment_materialization_id_invalid");
    return request;
}

bool RecommendationCampaignOutcomeAssessmentSchemasExist(
    pqxx::transaction_base& transaction)
{
    if (!RecommendationCampaignStatusSchemasExist(transaction) ||
        !RecommendationCampaignApprovalSchemaExists(transaction))
        return false;
    return transaction.exec(R"SQL(
SELECT NOT EXISTS (
  SELECT 1 FROM (VALUES
    ('experiment_recommendation','recommendation_id'),
    ('experiment_recommendation','source_experiment_id'),
    ('experiment_recommendation','source_model_id'),
    ('experiment_recommendation','source_analysis_id'),
    ('experiment_recommendation','source_symbol'),
    ('experiment_recommendation','source_prediction_horizon'),
    ('experiment_recommendation','source_leader_score'),
    ('experiment_recommendation','source_infer_accuracy'),
    ('experiment','c_next_threshold'),('experiment','infer_start'),
    ('experiment','infer_end'),('experiment','last_model_id'),
    ('model','model_id'),('model','experiment_id'),
    ('inference_eval_result','id'),('inference_eval_result','model_id'),
    ('inference_eval_result','symbol'),
    ('inference_eval_result','prediction_horizon'),
    ('inference_eval_result','threshold_logret'),
    ('inference_eval_result','window_size'),
    ('inference_eval_result','label_rule_id'),
    ('inference_eval_result','target_type'),
    ('inference_eval_result','from_date'),
    ('inference_eval_result','to_date'),
    ('inference_eval_result','status'),
    ('inference_eval_result','inference_scope'),
    ('inference_eval_result','checkpoint_eval_id'),
    ('experiment_analysis_result','analysis_id'),
    ('experiment_analysis_result','experiment_id'),
    ('experiment_analysis_result','model_id'),
    ('experiment_analysis_result','analysis_status'),
    ('experiment_analysis_result','analysis_scope'),
    ('experiment_analysis_result','infer_accuracy'),
    ('experiment_analysis_result','leader_score')
  ) required(table_name,column_name)
  WHERE NOT EXISTS (
    SELECT 1 FROM information_schema.columns c
    WHERE c.table_schema=current_schema()
      AND c.table_name=required.table_name
      AND c.column_name=required.column_name));
)SQL").one_row()[0].as<bool>();
}

RecommendationCampaignOutcomeAssessmentEvidenceSnapshot
LoadRecommendationCampaignOutcomeAssessmentEvidence(
    pqxx::transaction_base& transaction,
    const RecommendationCampaignOutcomeAssessmentRequest& request)
{
    const auto normalized =
        NormalizeRecommendationCampaignOutcomeAssessmentRequest(request);
    if (!RecommendationCampaignOutcomeAssessmentSchemasExist(transaction))
        throw std::runtime_error(
            "campaign_outcome_assessment_schemas_required");

    const auto materialization = FindRecommendationCampaignMaterialization(
        transaction, normalized.materializationId);
    if (!materialization)
        throw std::runtime_error(
            "campaign_outcome_assessment_materialization_not_found");
    const auto approval = FindRecommendationCampaignApproval(
        transaction, materialization->campaignApprovalId);
    if (!approval)
        throw std::runtime_error(
            "campaign_outcome_assessment_campaign_approval_not_found");
    if (approval->evidence.approvalIdentityHash !=
            materialization->approvalIdentityHash)
        throw std::runtime_error(
            "campaign_outcome_assessment_approval_materialization_mismatch");

    auto status = LoadRecommendationCampaignStatusSnapshot(
        transaction, {normalized.materializationId});
    auto scientific = LoadScientificMembers(transaction, status);
    return {
        RecommendationCampaignOutcomeAssessmentCampaignIdentity(
            approval->campaignApprovalId,
            approval->evidence.approvalIdentityCanonical,
            approval->evidence.approvalIdentityHash),
        RecommendationCampaignOutcomeAssessmentMaterializationIdentity(
            materialization->materializationId,
            materialization->campaignApprovalId,
            materialization->approvalIdentityHash,
            materialization->contractVersion,
            materialization->selectedMemberCount,
            materialization->identityCanonical,
            materialization->identityHash),
        std::move(status), std::move(scientific)};
}

RecommendationCampaignOutcomeAssessmentEvidenceSnapshot
ReadRecommendationCampaignOutcomeAssessmentEvidence(
    pqxx::connection& connection,
    const RecommendationCampaignOutcomeAssessmentRequest& request)
{
    pqxx::read_transaction transaction{connection};
    transaction.exec(
        "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY;");
    return LoadRecommendationCampaignOutcomeAssessmentEvidence(
        transaction, request);
}

} // namespace EA::ExperimentRecommendation
