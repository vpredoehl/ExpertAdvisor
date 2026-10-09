// Read-only comparison of benchmark-owned native checkpoints and every resumed
// update against an uninterrupted replay through the unchanged training code.
#define LSTM_QUALIFICATION_HELPERS_ONLY
#include "LSTMProductionTrainingQualification.mm"
#undef LSTM_QUALIFICATION_HELPERS_ONLY
#include <cstdlib>

namespace {
void Equal(const EA::LSTM& a, const EA::LSTM& b, const char* stage) {
    const EA::LSTM::EAMatrix* left[] = {&a.param, &a.bias, &a.returnHeadWeight,
        &a.returnHeadBias, &a.returnHeadDirWeight, &a.returnHeadDirBias,
        &a.prevHiddenState, &a.prevCellState};
    const EA::LSTM::EAMatrix* right[] = {&b.param, &b.bias, &b.returnHeadWeight,
        &b.returnHeadBias, &b.returnHeadDirWeight, &b.returnHeadDirBias,
        &b.prevHiddenState, &b.prevCellState};
    const char* names[] = {"param", "bias", "returnHeadWeight", "returnHeadBias",
        "returnHeadDirWeight", "returnHeadDirBias", "prevHiddenState", "prevCellState"};
    for (size_t m = 0; m < 8; ++m) {
        Require(left[m]->Shape() == right[m]->Shape(), "checkpoint_shape_mismatch");
        const auto x = MetaNN::LowerAccess(*left[m]);
        const auto y = MetaNN::LowerAccess(*right[m]);
        for (size_t i = 0; i < left[m]->Shape()[0] * left[m]->Shape()[1]; ++i) {
            if (std::memcmp(x.RawMemory()+i, y.RawMemory()+i, sizeof(float))) {
                throw std::runtime_error(std::string(stage)+",matrix="+names[m]+
                    ",row="+std::to_string(i/left[m]->Shape()[1])+",col="+
                    std::to_string(i%left[m]->Shape()[1])+",left="+
                    std::to_string(x.RawMemory()[i])+",right="+
                    std::to_string(y.RawMemory()[i]));
            }
        }
    }
    Require(std::memcmp(&a.learning_rate, &b.learning_rate, sizeof(float)) == 0,
            "checkpoint_learning_rate_mismatch");
    Require(a.optimizerUpdateCount == b.optimizerUpdateCount &&
            a.completedEpochs == b.completedEpochs, "checkpoint_counter_mismatch");
    Require(a.InputFeatureCount() == b.InputFeatureCount() && a.HiddenSize() == b.HiddenSize() &&
            a.targetType == b.targetType && a.targetUseZScore == b.targetUseZScore,
            "checkpoint_runtime_contract_mismatch");
    const float* scalarLeft[] = {&a.targetScale, &a.targetBias, &a.targetMean, &a.targetStd};
    const float* scalarRight[] = {&b.targetScale, &b.targetBias, &b.targetMean, &b.targetStd};
    for (size_t i = 0; i < 4; ++i)
        Require(std::memcmp(scalarLeft[i], scalarRight[i], sizeof(float)) == 0,
                "checkpoint_target_normalization_mismatch");
    EA::TrainingObjective::RequireResumeCompatible(a.trainingObjective, b.trainingObjective);
}
}

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            Require(argc == 6, "usage: checkpoint-fixture prefix source-epoch1 source-epoch2 restored-final experiment-id");
            Require(std::getenv("PGPORT") && std::string(std::getenv("PGPORT")) == "55483" &&
                    std::getenv("LSTM_DB_NAME") && std::string(std::getenv("LSTM_DB_NAME")) == "ea_phase25b3_lstm",
                    "benchmark_owned_database_required");
            const std::string connectionText = "hostaddr=127.0.0.1 port=55483 user=pqxx dbname=ea_phase25b3_lstm options='-c default_transaction_read_only=on'";
            DBIO::PgModelIO::PersistedModelMaterialization first, final, nativeRestored;
            unsigned int seed;
            {
                pqxx::connection connection{connectionText}; pqxx::work read{connection};
                read.exec("SET TRANSACTION READ ONLY;");
                Require(read.exec("SELECT inet_server_port(),current_database()").one_row()[0].as<int>() == 55483,
                        "isolated_port_required");
                first = DBIO::PgModelIO::ReadPersistedModelMaterialization(read, std::stoll(argv[2]));
                final = DBIO::PgModelIO::ReadPersistedModelMaterialization(read, std::stoll(argv[3]));
                nativeRestored = DBIO::PgModelIO::ReadPersistedModelMaterialization(read, std::stoll(argv[4]));
                seed = read.exec(pqxx::zview{"SELECT fresh_initialization_seed FROM experiment WHERE experiment_id=$1"},
                                 pqxx::params{std::stoll(argv[5])}).one_row()[0].as<unsigned int>();
                read.commit();
            }
            Require(seed == 1002 && first.completedEpoch == 1 && final.completedEpoch == 2 &&
                    nativeRestored.completedEpoch == 2 && first.trainConfigMeta && first.trainRange &&
                    first.identity.economicCalendarSnapshotId == 1 && first.identity.economicCalendarSnapshotHash &&
                    *first.identity.economicCalendarSnapshotHash == "fnv1a64:67610f94f5c8e7cc",
                    "checkpoint_contract_incomplete");
            const auto& config = first.trainConfigMeta->values;
            Require(config.size() == 14 && config[1] == 4 && config[3] == 64 &&
                    first.modelMeta.inputWidth == 171 && first.modelMeta.hiddenSize == 64,
                    "checkpoint_input_contract_mismatch");
            prediction_horizon = size_t(config[1]); c_next_threshold = float(config[2]);
            window_size = size_t(config[3]); hidden_size = first.modelMeta.hiddenSize;
            n_out = hidden_size; num_layers = size_t(config[8]); normalization_version = int(config[9]);
            epoch_count = 2; core_lr_mult = float(config[11]); head_weight_lr_mult = float(config[12]); head_bias_lr_mult = float(config[13]);
            auto prepared = EA::ModelInputPreparation::Prepare(
                {"cadchfrmp", first.trainRange->first, first.trainRange->second,
                 first.featureWarmupScope, first.donchian20Mode, first.donchianLookback,
                 EA::EconomicCalendar::EconomicCalendarSnapshotIdentity{1, *first.identity.economicCalendarSnapshotHash}},
                {"hostaddr=127.0.0.1 port=55483 user=pqxx dbname=ea_phase25b3_forex options='-c default_transaction_read_only=on'", connectionText});
            std::ofstream uninterrupted(std::string(argv[1])+".uninterrupted", std::ios::binary);
            std::ofstream restored(std::string(argv[1])+".restored", std::ios::binary);
            size_t secondEpochUpdates = 0;
            {
                Silence silence;
                auto create = [&]() {
                    EA::LSTM model{prepared.tensor, hidden_size, 1, 0, EA::LSTM::TargetType::UpNeutralDownReturn,
                        size_t{171}, first.identity.featureAblationMask, seed};
                    model.SetTrainingObjective(first.trainingObjective); return model;
                };
                auto original = create();
                prepared.tensor.ForEachBatchFrom(prepared.logicalOutputStartIndex, [&](auto batch) { original.CalculateBatch(batch, 0); });
                original.completedEpochs = 1;
                auto loaded = create();
                DBIO::PgModelIO::ApplyPersistedModelMaterialization(first, loaded);
                Equal(original, loaded, "persisted_epoch1");
                State(uninterrupted, original); State(restored, loaded);
                prepared.tensor.ForEachBatchFrom(prepared.logicalOutputStartIndex, [&](auto batch) {
                    const auto a = original.CalculateBatch(batch, 1);
                    const auto b = loaded.CalculateBatch(batch, 1);
                    Require(std::memcmp(&std::get<0>(a), &std::get<0>(b), sizeof(float)) == 0 &&
                            std::get<1>(a) == std::get<1>(b) && std::get<2>(a) == std::get<2>(b),
                            "restored_loss_or_window_count_mismatch");
                    Equal(original, loaded, "restored_update");
                    Write(uninterrupted, std::get<0>(a)); State(uninterrupted, original);
                    Write(restored, std::get<0>(b)); State(restored, loaded);
                    ++secondEpochUpdates;
                });
                original.completedEpochs = 2; loaded.completedEpochs = 2;
                auto reference = create(); DBIO::PgModelIO::ApplyPersistedModelMaterialization(final, reference);
                Equal(original, reference, "native_uninterrupted_final");
                DBIO::PgModelIO::ApplyPersistedModelMaterialization(nativeRestored, reference);
                Equal(loaded, reference, "native_restored_final");
            }
            std::cout << "CHECKPOINT_QUALIFIED,path=" << EA::MetalForwardAffine::SelectedPathName()
                      << ",rows=" << prepared.tensor.RowCount() << ",resumed_updates=" << secondEpochUpdates
                      << ",bitwise_equal=1\n";
        } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
    }
}
