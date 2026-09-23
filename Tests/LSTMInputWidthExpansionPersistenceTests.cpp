#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
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

EA::EconomicCalendar::EconomicEvent ProvenEconomicEvent(
    std::int64_t eventSeconds)
{
    using namespace EA::EconomicCalendar;
    EconomicEvent event;
    event.economicEventId = 20;
    event.currency = "USD";
    event.sourceAgency = "BEA";
    event.eventFamily = "PCE";
    event.eventTimestampUnixMicros = eventSeconds * 1'000'000LL;
    EconomicEventSelectedConsensus selected;
    selected.provider = "OANDA";
    selected.forecast = EconomicEventConsensusValue{
        "scalar", 0.3, std::nullopt, "percent", 1.0, std::nullopt};
    event.selectedConsensus = std::move(selected);
    EconomicEventReleaseActual actual;
    actual.actual = EconomicEventConsensusValue{
        "scalar", 0.5, std::nullopt, "percent", 1.0, std::nullopt};
    actual.availableAtUnixMicros = eventSeconds * 1'000'000LL;
    actual.sourceAgency = "BEA";
    actual.sourceObservationId = "phase20:initial";
    actual.sourceArtifactPath = "phase20/release.html";
    actual.sourceArtifactSha256 = std::string(64, 'a');
    actual.semanticContract = "phase20_initial_actual_v1";
    event.releaseActual = std::move(actual);
    return event;
}

} // namespace

int main()
{
    hidden_size = 1;
    n_out = 1;
    window_size = 4;
    prediction_horizon = 1;

    Tensor tensor{"eurusdrmp", kDefaultDonchian20Mode,
                  kDefaultDonchianLookback,
                  {ProvenEconomicEvent(10 * 900)}};
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

    // ProvenEconomicEvent(10 * 900) falls in the bar whose start timestamp
    // is 10 * 900.  Tensor bars are generated at (i + 1) * 900, so that is
    // Tensor row 9.  The occurrence indicator is intentionally one only on
    // the containing bar, not on the following bar.
    const auto postReleaseRow = MetaNN::LowerAccess(*(tensor.begin() + 9));
    const float* postRelease = postReleaseRow.RawMemory();
    assert(postRelease[inflationEventCol] == 1.0F);
    assert(postRelease[authoritativeInitialHasSurpriseCol] == 1.0F);
    assert(postRelease[authoritativeInitialSurpriseCol] != 0.0F);

    // A newly initialized width-80 model consumes the same nonzero event row
    // through both production inference and training tensor-copy paths.
    EA::LSTM freshWidth77{
        tensor, hidden_size, 1.0f, 0.0f, EA::LSTM::TargetType::UpNeutralDownReturn,
        EA::kCurrentModelInputWidth};
    assert(freshWidth77.InputFeatureCount() == 80);
    assert(freshWidth77.param.Shape()[0] ==
           EA::kCurrentModelInputWidth + hidden_size);
    const auto freshProbabilities = freshWidth77.PredictNextDirectionProbs(
        tensor.GetWindow(tensor.begin() + 48));
    float probabilitySum = 0.0F;
    for (const float probability : freshProbabilities)
    {
        assert(std::isfinite(probability));
        probabilitySum += probability;
    }
    assert(std::fabs(probabilitySum - 1.0F) < 1.0e-5F);
    const std::size_t freshUpdatesBefore = freshWidth77.optimizerUpdateCount;
    (void)freshWidth77.CalculateBatch(tensor.GetBatchClamped(0), 0);
    assert(freshWidth77.optimizerUpdateCount > freshUpdatesBefore);

    EA::LSTM source{
        tensor, hidden_size, 1.0f, 0.0f, EA::LSTM::TargetType::UpNeutralDownReturn,
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
        tensor, hidden_size, 1.0f, 0.0f, EA::LSTM::TargetType::UpNeutralDownReturn,
        EA::kSessionPhaseModelInputWidth};
    EA::LSTM auxiliaryGradientPath{
        tensor, hidden_size, 1.0f, 0.0f, EA::LSTM::TargetType::UpNeutralDownReturn,
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

    constexpr const char* kCalendarSnapshotHash = "fnv1a64:2222222222222222";
    long long calendarSnapshotId = -1;
    {
        pqxx::work transaction{connection};
        calendarSnapshotId = transaction.exec_params(
            "INSERT INTO economic_calendar_snapshot("
            "content_hash,created_by,snapshot_state,finalized_at,"
            "canonical_event_count,selected_consensus_count,"
            "release_actual_count,proven_first_release_actual_count,"
            "provenance_unavailable_count,ambiguous_first_release_count,"
            "source_family_counts) VALUES($1,'phase22t-test','finalized',"
            "clock_timestamp(),0,0,0,0,0,0,'{}'::jsonb) "
            "RETURNING economic_calendar_snapshot_id;",
            kCalendarSnapshotHash).one_row()[0].as<long long>();
        transaction.commit();
    }

    long long freshWidth77ModelId = -1;
    {
        pqxx::work transaction{connection};
        const long long experimentId = transaction.exec(
            "INSERT INTO experiment(symbol,prediction_horizon,"
            "c_next_threshold,target_epochs,checkpoint_interval,train_start,"
            "train_end,status,phase,duplicate_nonce,model_input_width,"
            "model_input_semantic_layout_version,"
            "economic_calendar_snapshot_id,economic_calendar_snapshot_hash) VALUES "
            "('eurusdrmp',1,0.0,1,0,'2020-01-01','2021-01-01',"
            "'pending','train',2090001,80,8,$1,$2) RETURNING experiment_id;",
            pqxx::params{calendarSnapshotId, kCalendarSnapshotHash})
            .one_row()[0].as<long long>();
        freshWidth77ModelId = DBIO::PgModelIO::createModel(
            transaction, "phase2-fresh-width77-event-smoke",
            "nonzero authoritative economic-event fixture", experimentId);
        DBIO::PgModelIO::saveAll(
            transaction, freshWidth77ModelId, freshWidth77, "eurusdrmp",
            "2020-01-01", "2021-01-01");
        transaction.commit();
    }

    ExpectFailureContaining(
        [&] {
            pqxx::work transaction{connection};
            const long long experimentId = transaction.exec(
                "INSERT INTO experiment(symbol,prediction_horizon,"
                "c_next_threshold,target_epochs,checkpoint_interval,"
                "train_start,train_end,status,phase,duplicate_nonce,"
                "model_input_width,model_input_semantic_layout_version) "
                "VALUES ('eurusdrmp',1,0.0,1,0,'2020-01-01',"
                "'2021-01-01','pending','train',2090002,75,5) "
                "RETURNING experiment_id;")
                .one_row()[0].as<long long>();
            const long long modelId = DBIO::PgModelIO::createModel(
                transaction, "phase20-identity-width-mismatch",
                "must fail before parameter persistence", experimentId);
            DBIO::PgModelIO::saveAll(
                transaction, modelId, source, "eurusdrmp",
                "2020-01-01", "2021-01-01");
        },
        "EXPERIMENT_MODEL_INPUT_IDENTITY_MISMATCH");

    EA::LSTM freshWidth77Reloaded{
        tensor, hidden_size, 1.0f, 0.0f, EA::LSTM::TargetType::UpNeutralDownReturn,
        EA::kCurrentModelInputWidth};
    {
        pqxx::work transaction{connection};
        transaction.exec("SET TRANSACTION READ ONLY;");
        DBIO::PgModelIO::loadAll(
            transaction, freshWidth77ModelId, freshWidth77Reloaded);
        const auto meta = DBIO::PgModelIO::loadRequiredModelMeta(
            transaction, freshWidth77ModelId);
        assert(meta.inputWidth == EA::kCurrentModelInputWidth);
        transaction.commit();
    }
    AssertExact(freshWidth77.param, freshWidth77Reloaded.param);
    AssertExact(freshWidth77.bias, freshWidth77Reloaded.bias);
    AssertExact(
        freshWidth77.returnHeadDirWeight,
        freshWidth77Reloaded.returnHeadDirWeight);
    AssertExact(
        freshWidth77.returnHeadDirBias,
        freshWidth77Reloaded.returnHeadDirBias);
    assert(freshWidth77.PredictNextDirectionProbs(
               tensor.GetWindow(tensor.begin() + 48)) ==
           freshWidth77Reloaded.PredictNextDirectionProbs(
               tensor.GetWindow(tensor.begin() + 48)));

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

    // The detached reader returns only owned C++ values.  Commit and leave the
    // transaction scope before Tensor-backed application to prove no pqxx view
    // or transaction lifetime reaches the applier.
    DBIO::PgModelIO::PersistedModelMaterialization sourceMaterialization;
    {
        pqxx::work transaction{connection};
        transaction.exec("SET TRANSACTION READ ONLY;");
        sourceMaterialization =
            DBIO::PgModelIO::ReadPersistedModelMaterialization(
                transaction, sourceModelId);
        transaction.commit();
    }
    assert(sourceMaterialization.identity.modelId == sourceModelId);
    assert(!sourceMaterialization.identity.experimentId.has_value());
    assert(sourceMaterialization.identity.featureAblationMask.empty());
    assert(!sourceMaterialization.identity.economicCalendarSnapshotId.has_value());
    assert(!sourceMaterialization.identity.economicCalendarSnapshotHash.has_value());
    assert(sourceMaterialization.modelMeta.inputWidth ==
           EA::kSessionPhaseModelInputWidth);
    assert(sourceMaterialization.semanticMetadata.has_value());
    assert(!sourceMaterialization.inputWidthExpansionProvenance.has_value());
    assert(sourceMaterialization.targetMeta.has_value());
    assert(sourceMaterialization.targetMeta->targetType ==
           EA::LSTM::TargetType::UpNeutralDownReturn);
    assert(sourceMaterialization.trainingObjective ==
           EA::TrainingObjective::Legacy());
    assert(sourceMaterialization.trainingObjectiveCanonical.has_value());
    assert(sourceMaterialization.trainingObjectiveHash.has_value());
    assert(EA::TrainingObjective::ResolvePersisted(
               sourceMaterialization.trainingObjectiveCanonical,
               sourceMaterialization.trainingObjectiveHash) ==
           EA::TrainingObjective::Legacy());
    assert(sourceMaterialization.optimizerMeta.has_value());
    assert(sourceMaterialization.optimizerMeta->updateCount == 17);
    assert(sourceMaterialization.completedEpoch.has_value());
    assert(*sourceMaterialization.completedEpoch == 40);
    assert(sourceMaterialization.trainSymbol == "eurusdrmp");
    assert(sourceMaterialization.trainRange.has_value());
    assert(sourceMaterialization.trainRange->first == "2020-01-01");
    assert(sourceMaterialization.trainRange->second == "2021-01-01");
    assert(sourceMaterialization.param.values ==
           DBIO::flattenRowMajor(source.param));

    EA::LSTM materializedNormal{
        tensor, hidden_size, 1.0f, 0.0f,
        EA::LSTM::TargetType::UpNeutralDownReturn,
        EA::kSessionPhaseModelInputWidth};
    const void* boundTensorBefore = materializedNormal.BoundTensorAddress();
    DBIO::PgModelIO::ApplyPersistedModelMaterialization(
        sourceMaterialization, materializedNormal);
    assert(materializedNormal.BoundTensorAddress() == boundTensorBefore);
    AssertExact(source.param, materializedNormal.param);
    AssertExact(source.bias, materializedNormal.bias);
    AssertExact(source.returnHeadWeight, materializedNormal.returnHeadWeight);
    AssertExact(source.returnHeadBias, materializedNormal.returnHeadBias);
    AssertExact(source.returnHeadDirWeight,
                materializedNormal.returnHeadDirWeight);
    AssertExact(source.returnHeadDirBias, materializedNormal.returnHeadDirBias);
    assert(materializedNormal.optimizerUpdateCount == source.optimizerUpdateCount);
    assert(materializedNormal.completedEpochs == source.completedEpochs);

    EA::LSTM materializedExpanded{
        tensor, hidden_size, 1.0f, 0.0f,
        EA::LSTM::TargetType::UpNeutralDownReturn,
        EA::kCurrentModelInputWidth};
    DBIO::PgModelIO::ApplyPersistedModelMaterialization(
        sourceMaterialization, materializedExpanded, true);
    const auto materializedExpansionPlan = EA::BuildInputWidthExpansionPlan(
        EA::kSessionPhaseModelInputWidth);
    assert(DBIO::flattenRowMajor(materializedExpanded.param) ==
           EA::ExpandFusedLstmParameterRowMajor(
               DBIO::flattenRowMajor(source.param), hidden_size,
               materializedExpansionPlan));

    // The experiment-bound fixture verifies that ownership/ablation fields are
    // detached too; this fixture intentionally has no calendar identity.
    DBIO::PgModelIO::PersistedModelMaterialization experimentMaterialization;
    {
        pqxx::work transaction{connection};
        transaction.exec("SET TRANSACTION READ ONLY;");
        experimentMaterialization =
            DBIO::PgModelIO::ReadPersistedModelMaterialization(
                transaction, freshWidth77ModelId);
        transaction.commit();
    }
    assert(experimentMaterialization.identity.experimentId.has_value());
    assert(experimentMaterialization.identity.featureAblationMask.empty());
    assert(experimentMaterialization.identity.economicCalendarSnapshotId ==
           calendarSnapshotId);
    assert(experimentMaterialization.identity.economicCalendarSnapshotHash ==
           kCalendarSnapshotHash);

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
        tensor, hidden_size, 1.0f, 0.0f, EA::LSTM::TargetType::UpNeutralDownReturn,
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

    DBIO::PgModelIO::PersistedModelMaterialization legacyHeadMaterialization;
    {
        pqxx::work transaction{connection};
        transaction.exec("SET TRANSACTION READ ONLY;");
        legacyHeadMaterialization =
            DBIO::PgModelIO::ReadPersistedModelMaterialization(
                transaction, legacyMissingScalarModelId);
        transaction.commit();
    }
    assert(!legacyHeadMaterialization.returnHeadWeight.has_value());
    assert(!legacyHeadMaterialization.returnHeadBias.has_value());
    assert(legacyHeadMaterialization.returnHeadDirWeight.has_value());
    assert(legacyHeadMaterialization.returnHeadDirBias.has_value());
    EA::LSTM detachedLegacyWithoutScalar{
        tensor, hidden_size, 1.0f, 0.0f,
        EA::LSTM::TargetType::UpNeutralDownReturn,
        EA::kSessionPhaseModelInputWidth};
    const auto scalarWeightBefore =
        DBIO::flattenRowMajor(detachedLegacyWithoutScalar.returnHeadWeight);
    const auto scalarBiasBefore =
        DBIO::flattenRowMajor(detachedLegacyWithoutScalar.returnHeadBias);
    DBIO::PgModelIO::ApplyPersistedModelMaterialization(
        legacyHeadMaterialization, detachedLegacyWithoutScalar);
    assert(DBIO::flattenRowMajor(detachedLegacyWithoutScalar.returnHeadWeight) ==
           scalarWeightBefore);
    assert(DBIO::flattenRowMajor(detachedLegacyWithoutScalar.returnHeadBias) ==
           scalarBiasBefore);
    AssertExact(source.returnHeadDirWeight,
                detachedLegacyWithoutScalar.returnHeadDirWeight);

    // Auxiliary head tensors and the exact objective canonical/hash pair
    // round-trip together. Corrupting either scalar tensor fails closed.
    EA::LSTM auxiliarySource{
        tensor, hidden_size, 1.0f, 0.0f, EA::LSTM::TargetType::UpNeutralDownReturn,
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
        tensor, hidden_size, 1.0f, 0.0f, EA::LSTM::TargetType::UpNeutralDownReturn,
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
    DBIO::PgModelIO::PersistedModelMaterialization auxiliaryMaterialization;
    {
        pqxx::work transaction{connection};
        transaction.exec("SET TRANSACTION READ ONLY;");
        auxiliaryMaterialization =
            DBIO::PgModelIO::ReadPersistedModelMaterialization(
                transaction, auxiliaryModelId);
        transaction.commit();
    }
    assert(auxiliaryMaterialization.trainingObjective ==
           EA::TrainingObjective::ProfitabilityAuxiliary());
    assert(auxiliaryMaterialization.trainingObjectiveCanonical.has_value());
    assert(auxiliaryMaterialization.trainingObjectiveHash.has_value());
    assert(EA::TrainingObjective::Identity(
               auxiliaryMaterialization.trainingObjective) ==
           "fnv1a64:f7a9a20f7f72eee5");
    assert(auxiliaryMaterialization.returnHeadWeight.has_value());
    assert(auxiliaryMaterialization.returnHeadBias.has_value());
    ExpectFailureContaining(
        [&] {
            EA::LSTM brokenReload{
                tensor, hidden_size, 1.0f, 0.0f,
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
        assert(values == std::vector<double>(
            {1.0, static_cast<double>(
                      EA::kModelInputSemanticLayoutVersion)}));
        transaction.commit();
    }

    // Marker-bearing ordinary loads enforce semantic identity just as explicit
    // expansion does. A width-compatible parameter matrix cannot hide an
    // incompatible economic-event layout marker.
    long long incompatibleSemanticModelId = -1;
    {
        pqxx::work transaction{connection};
        incompatibleSemanticModelId = DBIO::PgModelIO::createModel(
            transaction, "ordinary-load-incompatible-semantics",
            "isolated fixture");
        DBIO::PgModelIO::saveAll(
            transaction, incompatibleSemanticModelId, source, "eurusdrmp",
            "2020-01-01", "2021-01-01");
        transaction.exec(
            "UPDATE matrix SET value=999 WHERE model_id=$1 AND "
            "param_name='model_input_semantics_meta' AND row_idx=0 AND "
            "col_idx=1;",
            pqxx::params{incompatibleSemanticModelId});
        transaction.commit();
    }
    ExpectFailureContaining(
        [&] {
            EA::LSTM incompatible{
                tensor, hidden_size, 1.0f, 0.0f,
                EA::LSTM::TargetType::UpNeutralDownReturn,
                EA::kSessionPhaseModelInputWidth};
            pqxx::work transaction{connection};
            transaction.exec("SET TRANSACTION READ ONLY;");
            DBIO::PgModelIO::loadAll(
                transaction, incompatibleSemanticModelId, incompatible);
        },
        "SEMANTIC_METADATA_INCOMPATIBLE");

    long long markerlessHistoricalModelId = -1;
    {
        pqxx::work transaction{connection};
        markerlessHistoricalModelId = DBIO::PgModelIO::createModel(
            transaction, "ordinary-load-markerless-historical",
            "isolated pre-semantic-marker fixture");
        DBIO::PgModelIO::saveAll(
            transaction, markerlessHistoricalModelId, source, "eurusdrmp",
            "2020-01-01", "2021-01-01");
        transaction.exec(
            "DELETE FROM matrix WHERE model_id=$1 AND "
            "param_name='model_input_semantics_meta';",
            pqxx::params{markerlessHistoricalModelId});
        transaction.commit();
    }
    EA::LSTM markerlessHistorical{
        tensor, hidden_size, 1.0f, 0.0f, EA::LSTM::TargetType::UpNeutralDownReturn,
        EA::kSessionPhaseModelInputWidth};
    {
        pqxx::work transaction{connection};
        transaction.exec("SET TRANSACTION READ ONLY;");
        DBIO::PgModelIO::loadAll(
            transaction, markerlessHistoricalModelId,
            markerlessHistorical);
        transaction.commit();
    }
    AssertExact(source.param, markerlessHistorical.param);
    {
        pqxx::work transaction{connection};
        transaction.exec("SET TRANSACTION READ ONLY;");
        const auto markerlessMaterialization =
            DBIO::PgModelIO::ReadPersistedModelMaterialization(
                transaction, markerlessHistoricalModelId);
        assert(!markerlessMaterialization.semanticMetadata.has_value());
        transaction.commit();
    }

    EA::LSTM ordinaryLegacy{
        tensor, hidden_size, 1.0f, 0.0f, EA::LSTM::TargetType::UpNeutralDownReturn,
        EA::kSessionPhaseModelInputWidth};
    EA::LSTM expanded{
        tensor, hidden_size, 1.0f, 0.0f, EA::LSTM::TargetType::UpNeutralDownReturn,
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
        tensor, hidden_size, 1.0f, 0.0f, EA::LSTM::TargetType::UpNeutralDownReturn,
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
    {
        pqxx::work transaction{connection};
        transaction.exec("SET TRANSACTION READ ONLY;");
        const auto descendantMaterialization =
            DBIO::PgModelIO::ReadPersistedModelMaterialization(
                transaction, descendantModelId);
        assert(descendantMaterialization.inputWidthExpansionProvenance.has_value());
        assert(descendantMaterialization.inputWidthExpansionProvenance->CanonicalText() ==
               provenance.CanonicalText());
        assert(descendantMaterialization.identity.parentModelId == sourceModelId);
        transaction.commit();
    }

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
        tensor, hidden_size, 1.0f, 0.0f, EA::LSTM::TargetType::UpNeutralDownReturn,
        EA::kSessionPhaseModelInputWidth};
    {
        pqxx::work transaction{connection};
        transaction.exec("SET TRANSACTION READ ONLY;");
        DBIO::PgModelIO::loadAll(transaction, sourceModelId, sourceAfter);
        transaction.commit();
    }
    AssertExact(source.param, sourceAfter.param);

    // Deterministic two-connection proof for the production detached reader.
    // The callback pauses exactly after its first model query, so R's
    // REPEATABLE READ snapshot is established before W commits the coherent
    // replacement checkpoint; no timing sleep is involved.
    EA::LSTM snapshotOld{
        tensor, hidden_size, 1.0f, 0.0f,
        EA::LSTM::TargetType::UpNeutralDownReturn,
        EA::kSessionPhaseModelInputWidth};
    Fill(snapshotOld.param, -0.125f);
    Fill(snapshotOld.bias, -0.25f);
    Fill(snapshotOld.returnHeadWeight, -0.375f);
    Fill(snapshotOld.returnHeadBias, -0.5f);
    Fill(snapshotOld.returnHeadDirWeight, -0.625f);
    Fill(snapshotOld.returnHeadDirBias, -0.75f);
    snapshotOld.optimizerUpdateCount = 31;
    snapshotOld.completedEpochs = 41;
    long long snapshotModelId = -1;
    {
        pqxx::work transaction{connection};
        snapshotModelId = DBIO::PgModelIO::createModel(
            transaction, "materialization-repeatable-read", "isolated fixture");
        DBIO::PgModelIO::saveAll(
            transaction, snapshotModelId, snapshotOld, "eurusdrmp",
            "2020-01-01", "2021-01-01");
        transaction.commit();
    }
    EA::LSTM snapshotNew{
        tensor, hidden_size, 1.0f, 0.0f,
        EA::LSTM::TargetType::UpNeutralDownReturn,
        EA::kSessionPhaseModelInputWidth};
    Fill(snapshotNew.param, 0.125f);
    Fill(snapshotNew.bias, 0.25f);
    Fill(snapshotNew.returnHeadWeight, 0.375f);
    Fill(snapshotNew.returnHeadBias, 0.5f);
    Fill(snapshotNew.returnHeadDirWeight, 0.625f);
    Fill(snapshotNew.returnHeadDirBias, 0.75f);
    snapshotNew.optimizerUpdateCount = 83;
    snapshotNew.completedEpochs = 97;

    std::mutex snapshotMutex;
    std::condition_variable snapshotCondition;
    bool readerFirstQueryComplete = false;
    bool writerCommitted = false;
    std::exception_ptr readerFailure;
    DBIO::PgModelIO::PersistedModelMaterialization snapshotOldState;
    std::thread reader([&]
    {
        try
        {
            pqxx::connection readerConnection{connectionString};
            pqxx::work transaction{readerConnection};
            transaction.exec(
                "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY;");
            DBIO::PgModelIO::PersistedModelMaterializationReadOptions options;
            options.afterFirstRead = [&]
            {
                std::unique_lock lock{snapshotMutex};
                readerFirstQueryComplete = true;
                snapshotCondition.notify_all();
                snapshotCondition.wait(lock, [&] { return writerCommitted; });
            };
            snapshotOldState =
                DBIO::PgModelIO::ReadPersistedModelMaterialization(
                    transaction, snapshotModelId, options);
            transaction.commit();
        }
        catch (...)
        {
            std::lock_guard lock{snapshotMutex};
            readerFailure = std::current_exception();
            writerCommitted = true;
            snapshotCondition.notify_all();
        }
    });
    {
        std::unique_lock lock{snapshotMutex};
        snapshotCondition.wait(lock, [&]
        {
            return readerFirstQueryComplete || readerFailure != nullptr;
        });
    }
    if (readerFailure)
    {
        reader.join();
        std::rethrow_exception(readerFailure);
    }
    {
        pqxx::connection writerConnection{connectionString};
        pqxx::work transaction{writerConnection};
        DBIO::PgModelIO::saveAll(
            transaction, snapshotModelId, snapshotNew, "eurusdrmp",
            "2020-01-01", "2021-01-01");
        transaction.commit();
    }
    {
        std::lock_guard lock{snapshotMutex};
        writerCommitted = true;
    }
    snapshotCondition.notify_all();
    reader.join();
    if (readerFailure) std::rethrow_exception(readerFailure);
    assert(snapshotOldState.param.values == DBIO::flattenRowMajor(snapshotOld.param));
    assert(snapshotOldState.bias.values == DBIO::flattenRowMajor(snapshotOld.bias));
    assert(snapshotOldState.returnHeadWeight->values ==
           DBIO::flattenRowMajor(snapshotOld.returnHeadWeight));
    assert(snapshotOldState.returnHeadDirWeight->values ==
           DBIO::flattenRowMajor(snapshotOld.returnHeadDirWeight));
    assert(snapshotOldState.optimizerMeta->updateCount ==
           snapshotOld.optimizerUpdateCount);
    assert(snapshotOldState.completedEpoch == snapshotOld.completedEpochs);
    EA::LSTM snapshotApplied{
        tensor, hidden_size, 1.0f, 0.0f,
        EA::LSTM::TargetType::UpNeutralDownReturn,
        EA::kSessionPhaseModelInputWidth};
    DBIO::PgModelIO::ApplyPersistedModelMaterialization(
        snapshotOldState, snapshotApplied);
    AssertExact(snapshotOld.param, snapshotApplied.param);
    AssertExact(snapshotOld.returnHeadDirBias, snapshotApplied.returnHeadDirBias);
    DBIO::PgModelIO::PersistedModelMaterialization snapshotNewState;
    {
        pqxx::work transaction{connection};
        transaction.exec("SET TRANSACTION READ ONLY;");
        snapshotNewState = DBIO::PgModelIO::ReadPersistedModelMaterialization(
            transaction, snapshotModelId);
        transaction.commit();
    }
    assert(snapshotNewState.param.values == DBIO::flattenRowMajor(snapshotNew.param));
    assert(snapshotNewState.bias.values == DBIO::flattenRowMajor(snapshotNew.bias));
    assert(snapshotNewState.returnHeadWeight->values ==
           DBIO::flattenRowMajor(snapshotNew.returnHeadWeight));
    assert(snapshotNewState.returnHeadDirWeight->values ==
           DBIO::flattenRowMajor(snapshotNew.returnHeadDirWeight));
    assert(snapshotNewState.optimizerMeta->updateCount ==
           snapshotNew.optimizerUpdateCount);
    assert(snapshotNewState.completedEpoch == snapshotNew.completedEpochs);
    return 0;
}
