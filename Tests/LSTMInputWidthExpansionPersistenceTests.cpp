#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <pqxx/pqxx>

#include "PgModelIO.hpp"
#include "PricePoint.hpp"
#include "Tensor.hpp"

extern "C" bool LstmRuntimeDiagnosticLoggingEnabled() { return false; }

namespace
{

std::string EnvironmentOr(const char* name, const char* fallback)
{
    const char* value = std::getenv(name);
    return value && *value ? value : fallback;
}

template <typename Matrix>
void Fill(Matrix& matrix, float base)
{
    auto low = MetaNN::LowerAccess(matrix);
    float* values = low.MutableRawMemory();
    const std::size_t count = matrix.Shape()[0] * matrix.Shape()[1];
    for (std::size_t i = 0; i < count; ++i)
        values[i] = base + static_cast<float>(i + 1) / 4096.0f;
}

template <typename Matrix>
void AssertExact(const Matrix& left, const Matrix& right)
{
    assert(left.Shape() == right.Shape());
    const std::vector<double> a = DBIO::flattenRowMajor(left);
    const std::vector<double> b = DBIO::flattenRowMajor(right);
    assert(a.size() == b.size());
    assert(std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
}

template <typename Callable>
void ExpectFailureContaining(Callable&& callable, const std::string& marker)
{
    try
    {
        callable();
        assert(false && "expected failure");
    }
    catch (const std::exception& error)
    {
        assert(std::string{error.what()}.find(marker) != std::string::npos);
    }
}

} // namespace

int main()
{
    hidden_size = 1;
    n_out = 1;
    window_size = 4;
    prediction_horizon = 1;

    Tensor tensor{"eurusdrmp"};
    for (std::size_t i = 0; i < 96; ++i)
    {
        const float close = 1.0f + 0.0005f * static_cast<float>(i) +
                            0.0002f * static_cast<float>(i % 5);
        Feature bar{close - 0.0001f, close, close + 0.0003f,
                    close - 0.0003f,
                    PriceTP{std::chrono::seconds{
                        static_cast<long long>(i + 1) * 900}}};
        bar.tickVolume = 100.0f + static_cast<float>((i * 17) % 53);
        tensor.Add(bar);
    }

    EA::LSTM source{
        tensor, 1.0f, 0.0f, EA::LSTM::TargetType::UpNeutralDownReturn,
        EA::kSessionPhaseModelInputWidth};
    Fill(source.param, -0.02f);
    Fill(source.bias, 0.01f);
    Fill(source.returnHeadWeight, 0.375f);
    Fill(source.returnHeadBias, -0.5f);
    Fill(source.returnHeadDirWeight, 0.625f);
    Fill(source.returnHeadDirBias, -0.75f);
    source.optimizerUpdateCount = 17;
    source.completedEpochs = 40;

    // One deterministic production batch proves coefficient-zero legacy
    // equivalence and gradient routing: the auxiliary objective leaves the
    // classification-head update exact, updates its own scalar head, and adds
    // only its projection to the shared-core gradient.
    EA::LSTM legacyGradientPath{
        tensor, 1.0f, 0.0f, EA::LSTM::TargetType::UpNeutralDownReturn,
        EA::kSessionPhaseModelInputWidth};
    EA::LSTM auxiliaryGradientPath{
        tensor, 1.0f, 0.0f, EA::LSTM::TargetType::UpNeutralDownReturn,
        EA::kSessionPhaseModelInputWidth};
    auxiliaryGradientPath.SetTrainingObjective(
        EA::TrainingObjective::ProfitabilityAuxiliary());
    for (EA::LSTM* model : {&legacyGradientPath, &auxiliaryGradientPath})
    {
        Fill(model->param, -0.015625f);
        Fill(model->bias, 0.0078125f);
        Fill(model->returnHeadWeight, 0.125f);
        Fill(model->returnHeadBias, -0.0625f);
        Fill(model->returnHeadDirWeight, 0.25f);
        Fill(model->returnHeadDirBias, -0.125f);
    }
    const auto legacyScalarWeightBefore =
        DBIO::flattenRowMajor(legacyGradientPath.returnHeadWeight);
    const auto legacyScalarBiasBefore =
        DBIO::flattenRowMajor(legacyGradientPath.returnHeadBias);
    std::ostringstream suppressedTrainingOutput;
    std::streambuf* originalCout = std::cout.rdbuf(
        suppressedTrainingOutput.rdbuf());
    (void)legacyGradientPath.CalculateBatch(tensor.GetBatchClamped(0), 0);
    (void)auxiliaryGradientPath.CalculateBatch(tensor.GetBatchClamped(0), 0);
    std::cout.rdbuf(originalCout);
    AssertExact(legacyGradientPath.returnHeadDirWeight,
                auxiliaryGradientPath.returnHeadDirWeight);
    AssertExact(legacyGradientPath.returnHeadDirBias,
                auxiliaryGradientPath.returnHeadDirBias);
    assert(DBIO::flattenRowMajor(legacyGradientPath.returnHeadWeight) ==
           legacyScalarWeightBefore);
    assert(DBIO::flattenRowMajor(legacyGradientPath.returnHeadBias) ==
           legacyScalarBiasBefore);
    assert(DBIO::flattenRowMajor(auxiliaryGradientPath.returnHeadWeight) !=
           legacyScalarWeightBefore);
    assert(DBIO::flattenRowMajor(auxiliaryGradientPath.param) !=
           DBIO::flattenRowMajor(legacyGradientPath.param));

    const std::string connectionString =
        "hostaddr=" + EnvironmentOr("LSTM_DB_HOST", "127.0.0.1") +
        " user=pqxx dbname=" + EnvironmentOr("LSTM_DB_NAME", "LSTM");
    pqxx::connection connection{connectionString};

    long long sourceModelId = -1;
    {
        pqxx::work transaction{connection};
        sourceModelId = DBIO::PgModelIO::createModel(
            transaction, "input-width-expansion-source", "isolated fixture");
        DBIO::PgModelIO::saveAll(
            transaction, sourceModelId, source, "eurusdrmp",
            "2020-01-01", "2021-01-01");
        transaction.commit();
    }

    // Marker-less/legacy classification models may lack the inactive scalar
    // head and must continue to load. The 3-class head remains authoritative.
    long long legacyMissingScalarModelId = -1;
    {
        pqxx::work transaction{connection};
        legacyMissingScalarModelId = DBIO::PgModelIO::createModel(
            transaction, "legacy-missing-scalar-head", "isolated fixture");
        DBIO::PgModelIO::saveAll(
            transaction, legacyMissingScalarModelId, source, "eurusdrmp",
            "2020-01-01", "2021-01-01");
        transaction.exec(
            "DELETE FROM matrix WHERE model_id=$1 AND "
            "param_name IN ('returnHeadWeight','returnHeadBias');",
            pqxx::params{legacyMissingScalarModelId});
        transaction.commit();
    }
    EA::LSTM loadedLegacyWithoutScalar{
        tensor, 1.0f, 0.0f, EA::LSTM::TargetType::UpNeutralDownReturn,
        EA::kSessionPhaseModelInputWidth};
    {
        pqxx::work transaction{connection};
        transaction.exec("SET TRANSACTION READ ONLY;");
        DBIO::PgModelIO::loadAll(
            transaction, legacyMissingScalarModelId,
            loadedLegacyWithoutScalar);
        DBIO::PgModelIO::validateTrainingResumeState(
            transaction, legacyMissingScalarModelId);
        transaction.commit();
    }
    AssertExact(source.returnHeadDirWeight,
                loadedLegacyWithoutScalar.returnHeadDirWeight);
    AssertExact(source.returnHeadDirBias,
                loadedLegacyWithoutScalar.returnHeadDirBias);

    // Auxiliary head tensors and the exact objective canonical/hash pair
    // round-trip together. Corrupting either scalar tensor fails closed.
    EA::LSTM auxiliarySource{
        tensor, 1.0f, 0.0f, EA::LSTM::TargetType::UpNeutralDownReturn,
        EA::kSessionPhaseModelInputWidth};
    auxiliarySource.SetTrainingObjective(
        EA::TrainingObjective::ProfitabilityAuxiliary());
    Fill(auxiliarySource.param, -0.03125f);
    Fill(auxiliarySource.bias, 0.015625f);
    Fill(auxiliarySource.returnHeadWeight, 0.8125f);
    Fill(auxiliarySource.returnHeadBias, -0.9375f);
    Fill(auxiliarySource.returnHeadDirWeight, 0.5625f);
    Fill(auxiliarySource.returnHeadDirBias, -0.6875f);

    long long auxiliaryModelId = -1;
    long long brokenAuxiliaryModelId = -1;
    {
        pqxx::work transaction{connection};
        auxiliaryModelId = DBIO::PgModelIO::createModel(
            transaction, "auxiliary-objective-roundtrip", "isolated fixture");
        DBIO::PgModelIO::saveAll(
            transaction, auxiliaryModelId, auxiliarySource, "eurusdrmp",
            "2020-01-01", "2021-01-01", kDefaultDonchian20Mode,
            EA::kDefaultFeatureWarmupScope, kDefaultDonchianLookback,
            std::nullopt, EA::TrainingObjective::ProfitabilityAuxiliary());
        brokenAuxiliaryModelId = DBIO::PgModelIO::createModel(
            transaction, "auxiliary-objective-broken", "isolated fixture");
        DBIO::PgModelIO::saveAll(
            transaction, brokenAuxiliaryModelId, auxiliarySource,
            "eurusdrmp", "2020-01-01", "2021-01-01",
            kDefaultDonchian20Mode, EA::kDefaultFeatureWarmupScope,
            kDefaultDonchianLookback, std::nullopt,
            EA::TrainingObjective::ProfitabilityAuxiliary());
        transaction.exec(
            "DELETE FROM matrix WHERE model_id=$1 AND "
            "param_name='returnHeadBias';",
            pqxx::params{brokenAuxiliaryModelId});
        transaction.commit();
    }
    EA::LSTM auxiliaryReloaded{
        tensor, 1.0f, 0.0f, EA::LSTM::TargetType::UpNeutralDownReturn,
        EA::kSessionPhaseModelInputWidth};
    {
        pqxx::work transaction{connection};
        transaction.exec("SET TRANSACTION READ ONLY;");
        DBIO::PgModelIO::loadAll(
            transaction, auxiliaryModelId, auxiliaryReloaded);
        DBIO::PgModelIO::validateTrainingResumeState(
            transaction, auxiliaryModelId);
        const auto objective = DBIO::PgModelIO::loadTrainingObjectiveMeta(
            transaction, auxiliaryModelId);
        assert(objective ==
               EA::TrainingObjective::ProfitabilityAuxiliary());
        assert(EA::TrainingObjective::Identity(objective) ==
               "fnv1a64:f7a9a20f7f72eee5");
        transaction.commit();
    }
    AssertExact(auxiliarySource.returnHeadWeight,
                auxiliaryReloaded.returnHeadWeight);
    AssertExact(auxiliarySource.returnHeadBias,
                auxiliaryReloaded.returnHeadBias);
    ExpectFailureContaining(
        [&] {
            EA::LSTM brokenReload{
                tensor, 1.0f, 0.0f,
                EA::LSTM::TargetType::UpNeutralDownReturn,
                EA::kSessionPhaseModelInputWidth};
            pqxx::work transaction{connection};
            transaction.exec("SET TRANSACTION READ ONLY;");
            DBIO::PgModelIO::loadAll(
                transaction, brokenAuxiliaryModelId, brokenReload);
        },
        "auxiliary objective requires complete scalar auxiliary head parameters");
    {
        pqxx::work transaction{connection};
        transaction.exec("SET TRANSACTION READ ONLY;");
        const auto objective =
            DBIO::PgModelIO::loadTrainingObjectiveMeta(
                transaction, sourceModelId);
        assert(objective == EA::TrainingObjective::Legacy());
        assert(EA::TrainingObjective::Identity(objective) ==
               EA::TrainingObjective::Identity(
                   EA::TrainingObjective::Legacy()));
        const auto dims = DBIO::PgModelIO::loadParameterDims(
            transaction, sourceModelId, "model_input_semantics_meta");
        const auto values = DBIO::PgModelIO::loadParameterValues(
            transaction, sourceModelId, "model_input_semantics_meta");
        assert(dims.n_rows == 1);
        assert(dims.n_cols == 2);
        assert(values == std::vector<double>({1.0, 2.0}));
        transaction.commit();
    }

    EA::LSTM ordinaryLegacy{
        tensor, 1.0f, 0.0f, EA::LSTM::TargetType::UpNeutralDownReturn,
        EA::kSessionPhaseModelInputWidth};
    EA::LSTM expanded{
        tensor, 1.0f, 0.0f, EA::LSTM::TargetType::UpNeutralDownReturn,
        EA::kCurrentModelInputWidth};
    {
        pqxx::work transaction{connection};
        transaction.exec("SET TRANSACTION READ ONLY;");
        DBIO::PgModelIO::loadAll(transaction, sourceModelId, ordinaryLegacy);
        DBIO::PgModelIO::loadAll(transaction, sourceModelId, expanded, true);
        DBIO::PgModelIO::loadOptimizerMeta(
            transaction, sourceModelId, expanded);
        transaction.commit();
    }
    assert(ordinaryLegacy.InputFeatureCount() ==
           static_cast<int>(EA::kSessionPhaseModelInputWidth));
    AssertExact(source.param, ordinaryLegacy.param);

    const EA::InputWidthExpansionPlan plan =
        EA::BuildInputWidthExpansionPlan(
            EA::kSessionPhaseModelInputWidth);
    const std::vector<double> expected =
        EA::ExpandFusedLstmParameterRowMajor(
            DBIO::flattenRowMajor(source.param), hidden_size, plan);
    const std::vector<double> actual = DBIO::flattenRowMajor(expanded.param);
    assert(expected == actual);
    AssertExact(source.bias, expanded.bias);
    AssertExact(source.returnHeadWeight, expanded.returnHeadWeight);
    AssertExact(source.returnHeadBias, expanded.returnHeadBias);
    AssertExact(source.returnHeadDirWeight, expanded.returnHeadDirWeight);
    AssertExact(source.returnHeadDirBias, expanded.returnHeadDirBias);

    // The production forward path consumes current-width rows after expansion.
    // Current-only Tensor columns are nonzero in this mature synthetic window,
    // but zero input weights preserve the source prediction before training.
    const auto predictionWindow = tensor.GetWindow(tensor.begin() + 48);
    bool hasNonzeroCurrentOnlyFeature = false;
    for (const auto& row : predictionWindow)
    {
        const auto low = MetaNN::LowerAccess(row);
        const float* values = low.RawMemory();
        for (std::size_t column = plan.sourceTensorFeatureCount;
             column < plan.expandedTensorFeatureCount;
             ++column)
        {
            if (values[column] != 0.0f)
                hasNonzeroCurrentOnlyFeature = true;
        }
    }
    assert(hasNonzeroCurrentOnlyFeature);
    const auto auxiliarySourceProbabilities =
        auxiliarySource.PredictNextDirectionProbs(predictionWindow);
    const auto auxiliaryReloadedProbabilities =
        auxiliaryReloaded.PredictNextDirectionProbs(predictionWindow);
    assert(auxiliarySourceProbabilities == auxiliaryReloadedProbabilities);
    const auto legacyProbabilities =
        ordinaryLegacy.PredictNextDirectionProbs(predictionWindow);
    const auto expandedProbabilities =
        expanded.PredictNextDirectionProbs(predictionWindow);
    for (std::size_t i = 0; i < legacyProbabilities.size(); ++i)
        assert(std::fabs(legacyProbabilities[i] - expandedProbabilities[i]) <=
               1.0e-6f);

    // Exercise the real production forward/BPTT/SGD path.  At least one
    // zero-initialized newly appended input row must receive a finite update.
    const std::vector<double> beforeTraining =
        DBIO::flattenRowMajor(expanded.param);
    const std::size_t updatesBefore = expanded.optimizerUpdateCount;
    (void)expanded.CalculateBatch(tensor.GetBatchClamped(0), 0);
    const std::vector<double> afterTraining =
        DBIO::flattenRowMajor(expanded.param);
    assert(expanded.optimizerUpdateCount > updatesBefore);
    bool learnedNewInputWeight = false;
    const std::size_t gateColumns = 4 * hidden_size;
    for (std::size_t row = plan.sourceTensorFeatureCount;
         row < plan.expandedTensorFeatureCount;
         ++row)
    {
        for (std::size_t column = 0; column < gateColumns; ++column)
        {
            const std::size_t index = row * gateColumns + column;
            assert(std::isfinite(afterTraining[index]));
            if (afterTraining[index] != beforeTraining[index])
                learnedNewInputWeight = true;
        }
    }
    assert(learnedNewInputWeight);

    const EA::InputWidthExpansionProvenance provenance =
        EA::MakeInputWidthExpansionProvenance(sourceModelId, plan);
    expanded.completedEpochs = source.completedEpochs;

    long long descendantModelId = -1;
    {
        pqxx::work transaction{connection};
        descendantModelId = DBIO::PgModelIO::createModel(
            transaction, "input-width-expansion-descendant",
            "isolated expanded checkpoint", std::nullopt, sourceModelId);
        DBIO::PgModelIO::saveAll(
            transaction, descendantModelId, expanded, "eurusdrmp",
            "2020-01-01", "2021-01-01", kDefaultDonchian20Mode,
            EA::kDefaultFeatureWarmupScope, kDefaultDonchianLookback,
            provenance);
        transaction.commit();
    }

    EA::LSTM reloaded{
        tensor, 1.0f, 0.0f, EA::LSTM::TargetType::UpNeutralDownReturn,
        EA::kCurrentModelInputWidth};
    {
        pqxx::work transaction{connection};
        transaction.exec("SET TRANSACTION READ ONLY;");
        DBIO::PgModelIO::loadAll(transaction, descendantModelId, reloaded);
        DBIO::PgModelIO::loadOptimizerMeta(
            transaction, descendantModelId, reloaded);
        const auto meta = DBIO::PgModelIO::loadRequiredModelMeta(
            transaction, descendantModelId);
        assert(meta.inputWidth == EA::kCurrentModelInputWidth);
        assert(meta.hiddenSize == hidden_size);
        const auto reloadedProvenance =
            DBIO::PgModelIO::loadRequiredInputWidthExpansionMeta(
                transaction, descendantModelId);
        assert(reloadedProvenance.CanonicalText() ==
               provenance.CanonicalText());
        const auto reloadedObjective =
            DBIO::PgModelIO::loadTrainingObjectiveMeta(
                transaction, descendantModelId);
        assert(reloadedObjective == EA::TrainingObjective::Legacy());
        EA::TrainingObjective::RequireResumeCompatible(
            reloadedObjective, EA::TrainingObjective::Legacy());
        const auto parent = transaction.exec(
            "SELECT parent_model_id FROM model WHERE model_id=$1;",
            pqxx::params{descendantModelId}).one_row();
        assert(parent[0].as<long long>() == sourceModelId);
        const auto validatedProvenance =
            DBIO::PgModelIO::validateModelInputSemanticsForExpansion(
                transaction, descendantModelId);
        assert(validatedProvenance.has_value());
        assert(validatedProvenance->CanonicalText() ==
               provenance.CanonicalText());
        transaction.commit();
    }
    assert(reloaded.optimizerUpdateCount == expanded.optimizerUpdateCount);
    AssertExact(expanded.param, reloaded.param);
    AssertExact(expanded.bias, reloaded.bias);
    AssertExact(expanded.returnHeadWeight, reloaded.returnHeadWeight);
    AssertExact(expanded.returnHeadBias, reloaded.returnHeadBias);
    AssertExact(expanded.returnHeadDirWeight, reloaded.returnHeadDirWeight);
    AssertExact(expanded.returnHeadDirBias, reloaded.returnHeadDirBias);

    // Marker-bearing expanded models fail closed when their durable parent
    // chain no longer reaches the recorded expansion source.
    ExpectFailureContaining(
        [&] {
            pqxx::work transaction{connection};
            transaction.exec(
                "UPDATE model SET parent_model_id=NULL WHERE model_id=$1;",
                pqxx::params{descendantModelId});
            (void)DBIO::PgModelIO::validateModelInputSemanticsForExpansion(
                transaction, descendantModelId);
        },
        "PROVENANCE_SOURCE_LINEAGE_MISMATCH");

    // Saving the descendant must not alter the source checkpoint.
    EA::LSTM sourceAfter{
        tensor, 1.0f, 0.0f, EA::LSTM::TargetType::UpNeutralDownReturn,
        EA::kSessionPhaseModelInputWidth};
    {
        pqxx::work transaction{connection};
        transaction.exec("SET TRANSACTION READ ONLY;");
        DBIO::PgModelIO::loadAll(transaction, sourceModelId, sourceAfter);
        transaction.commit();
    }
    AssertExact(source.param, sourceAfter.param);
    return 0;
}
