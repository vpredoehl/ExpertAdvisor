#include <cassert>
#include <charconv>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "../Sources/ExperimentRecommendation.hpp"

using namespace EA::ExperimentRecommendation;

namespace
{

std::string ErrorFrom(const std::function<void()>& action)
{
    try { action(); }
    catch (const std::invalid_argument& error) { return error.what(); }
    assert(false && "expected std::invalid_argument");
    return {};
}

EffectiveExperimentConfiguration ExampleConfiguration()
{
    EffectiveExperimentConfiguration configuration;
    configuration.symbol = " EURUSD ";
    configuration.predictionHorizon = 12;
    configuration.labelThreshold = 0.001;
    configuration.coreLrMult = 1.0;
    configuration.headLrMult = 5.0;
    configuration.targetEpochs = 120;
    configuration.trainStartDate = "2010-01-01";
    configuration.trainEndDate = "2025-01-01";
    configuration.inferStartDate = "2025-01-01";
    configuration.inferEndDate = "2026-01-01";
    return configuration;
}

void AssertIdentityChange(
    const EffectiveExperimentConfiguration& original,
    const std::function<void(EffectiveExperimentConfiguration&)>& modify)
{
    EffectiveExperimentConfiguration changed = original;
    modify(changed);
    assert(RecommendationCandidateHash(changed) !=
           RecommendationCandidateHash(original));
}

} // namespace

int main()
{
    const RecommendationPolicy defaults;
    assert(!ValidateRecommendationPolicy(defaults));
    assert(RecommendationPolicyHash(defaults) ==
           "fnv1a64:5d38b796e2380a45");
    assert(RecommendationPolicyHash(defaults) == RecommendationPolicyHash(defaults));
    assert(RecommendationPolicyHash(defaults).starts_with("fnv1a64:"));

    const RecommendationPolicy keyOrderA = ParseRecommendationPolicy(
        "min_leader_score=0.2,min_infer_accuracy=0.3,"
        "allowed_parameters=head_lr_mult:core_lr_mult");
    const RecommendationPolicy keyOrderB = ParseRecommendationPolicy(
        "allowed_parameters=core_lr_mult:head_lr_mult,"
        "min_infer_accuracy=0.3,min_leader_score=0.2");
    assert(RecommendationPolicyCanonicalText(keyOrderA) ==
           RecommendationPolicyCanonicalText(keyOrderB));
    assert(RecommendationPolicyHash(keyOrderA) ==
           RecommendationPolicyHash(keyOrderB));
    const RecommendationPolicy whitespace = ParseRecommendationPolicy(
        " enabled = true , min_leader_score = 0.2 , min_infer_accuracy = 0.3 , "
        "allowed_parameters = core_lr_mult : head_lr_mult ");
    assert(RecommendationPolicyCanonicalText(whitespace) ==
           RecommendationPolicyCanonicalText(keyOrderA));

    const RecommendationPolicy listOrderA = ParseRecommendationPolicy(
        "core_lr_offsets=-0.25:0.25,head_lr_offsets=-0.5:0.5");
    const RecommendationPolicy listOrderB = ParseRecommendationPolicy(
        "head_lr_offsets=0.5:-0.5:-0.5,core_lr_offsets=0.25:-0.25:0.25");
    assert(RecommendationPolicyCanonicalText(listOrderA) ==
           RecommendationPolicyCanonicalText(listOrderB));
    assert(RecommendationPolicyHash(listOrderA) ==
           RecommendationPolicyHash(listOrderB));

    const RecommendationPolicy horizonOrderA = ParseRecommendationPolicy(
        "allowed_parameters=prediction_horizon,allow_horizon_changes=true,"
        "permitted_horizons=4:8:12:16");
    const RecommendationPolicy horizonOrderB = ParseRecommendationPolicy(
        "permitted_horizons=16:8:4:12:8,allow_horizon_changes=1,"
        "allowed_parameters=prediction_horizon:prediction_horizon");
    assert(RecommendationPolicyCanonicalText(horizonOrderA) ==
           RecommendationPolicyCanonicalText(horizonOrderB));
    assert(horizonOrderA.coreLrOffsets.empty());
    assert(horizonOrderA.headLrOffsets.empty());
    assert(horizonOrderA.labelThresholdOffsets.empty());

    RecommendationPolicy programmaticOrderA = defaults;
    RecommendationPolicy programmaticOrderB = defaults;
    programmaticOrderA.coreLrOffsets = {0.25, -0.25, 0.25};
    programmaticOrderB.coreLrOffsets = {-0.25, 0.25};
    assert(RecommendationPolicyHash(programmaticOrderA) ==
           RecommendationPolicyHash(programmaticOrderB));

    RecommendationPolicy changedPolicy = defaults;
    changedPolicy.minimumLeaderScore = 0.01;
    assert(RecommendationPolicyHash(changedPolicy) !=
           RecommendationPolicyHash(defaults));
    changedPolicy = defaults;
    changedPolicy.expirationDays = 30;
    assert(RecommendationPolicyHash(changedPolicy) !=
           RecommendationPolicyHash(defaults));

    assert(ErrorFrom([] { ParseRecommendationPolicy("unknown=1"); }) ==
           "unsupported_recommendation_policy_key:key=unknown");
    assert(ErrorFrom([] { ParseRecommendationPolicy("enabled=true,enabled=false"); }) ==
           "duplicate_recommendation_policy_key:key=enabled");
    assert(ErrorFrom([] { ParseRecommendationPolicy("enabled"); }) ==
           "malformed_recommendation_policy_assignment");
    assert(ErrorFrom([] { ParseRecommendationPolicy("=1"); }) ==
           "malformed_recommendation_policy_assignment");
    assert(ErrorFrom([] { ParseRecommendationPolicy(
        "enabled=true,,min_leader_score=0.2"); }) ==
        "malformed_recommendation_policy_assignment");
    assert(ErrorFrom([] { ParseRecommendationPolicy(
        "allowed_parameters=core_lr_mult::head_lr_mult"); }) ==
        "invalid_recommendation_policy_list:key=allowed_parameters");
    assert(ErrorFrom([] { ParseRecommendationPolicy(
        "enabled=true, enabled=false"); }) ==
        "duplicate_recommendation_policy_key:key=enabled");
    assert(ErrorFrom([] { ParseRecommendationPolicy("enabled=yes"); }) ==
           "invalid_recommendation_policy_boolean:key=enabled");
    assert(ErrorFrom([] { ParseRecommendationPolicy("enabled="); }) ==
           "invalid_recommendation_policy_boolean:key=enabled");
    assert(ErrorFrom([] { ParseRecommendationPolicy("max_per_scan=x"); }) ==
           "invalid_recommendation_policy_integer:key=max_per_scan");
    assert(ErrorFrom([] { ParseRecommendationPolicy("max_per_scan=0"); }) ==
           "invalid_recommendation_policy_integer:key=max_per_scan");
    assert(ErrorFrom([] { ParseRecommendationPolicy("min_leader_score=x"); }) ==
           "invalid_recommendation_policy_number:key=min_leader_score");
    assert(ErrorFrom([] { ParseRecommendationPolicy("min_leader_score=nan"); }) ==
           "invalid_recommendation_policy_number:key=min_leader_score");
    assert(ErrorFrom([] { ParseRecommendationPolicy("min_leader_score=inf"); }) ==
           "invalid_recommendation_policy_number:key=min_leader_score");
    assert(ErrorFrom([] { ParseRecommendationPolicy("core_lr_offsets=1::2"); }) ==
           "invalid_recommendation_policy_list:key=core_lr_offsets");
    assert(ErrorFrom([] { ParseRecommendationPolicy("core_lr_offsets=1:"); }) ==
           "invalid_recommendation_policy_list:key=core_lr_offsets");
    assert(ErrorFrom([] { ParseRecommendationPolicy(
        "allowed_parameters=neutral_class_weight"); }) ==
        "unsupported_recommendation_parameter:name=neutral_class_weight");
    assert(ErrorFrom([] { ParseRecommendationPolicy(
        "allowed_parameters=head_bias_lr_mult"); }) ==
        "unsupported_recommendation_parameter:name=head_bias_lr_mult");
    assert(ErrorFrom([] { ParseRecommendationPolicy("source_scope=nearby"); }) ==
           "invalid_recommendation_source_scope");
    assert(ErrorFrom([] { ParseRecommendationPolicy("max_predicted_neutral=-0.1"); }) ==
           "maximum_predicted_neutral_out_of_range");
    assert(ErrorFrom([] { ParseRecommendationPolicy("max_predicted_neutral=1.1"); }) ==
           "maximum_predicted_neutral_out_of_range");
    assert(ErrorFrom([] { ParseRecommendationPolicy(
        "allowed_parameters=prediction_horizon"); }) ==
        "prediction_horizon_requires_explicit_enablement");
    assert(ErrorFrom([] { ParseRecommendationPolicy(
        "allowed_parameters=prediction_horizon,allow_horizon_changes=true"); }) ==
        "prediction_horizon_requires_permitted_horizons");
    assert(ErrorFrom([] { ParseRecommendationPolicy(
        "allow_horizon_changes=true,permitted_horizons=4:8"); }) ==
        "horizon_changes_require_prediction_horizon_parameter");
    assert(ErrorFrom([] { ParseRecommendationPolicy("core_lr_offsets=0"); }) ==
           "invalid_core_lr_offset");
    assert(ErrorFrom([] { ParseRecommendationPolicy(
        "allowed_parameters=head_lr_mult,core_lr_offsets=0.25"); }) ==
        "core_lr_offsets_requires_allowed_parameter");
    assert(ErrorFrom([] { ParseRecommendationPolicy(
        "allowed_parameters=core_lr_mult,core_lr_offsets="); }) ==
        "core_lr_offsets_required");
    assert(ErrorFrom([] { ParseRecommendationPolicy("expiration_days=0"); }) ==
           "invalid_recommendation_policy_integer:key=expiration_days");

    RecommendationPolicy nonfinite = defaults;
    nonfinite.minimumLeaderScore = std::numeric_limits<double>::infinity();
    assert(ValidateRecommendationPolicy(nonfinite) ==
           "recommendation_policy_threshold_not_finite");
    nonfinite = defaults;
    nonfinite.coreLrOffsets = {std::numeric_limits<double>::quiet_NaN()};
    assert(ValidateRecommendationPolicy(nonfinite) == "invalid_core_lr_offset");
    RecommendationPolicy invalidHorizon = horizonOrderA;
    invalidHorizon.permittedHorizons = {0, 4};
    assert(ValidateRecommendationPolicy(invalidHorizon) ==
           "permitted_horizons_must_be_positive");
    assert(CanonicalRecommendationDouble(-0.0) ==
           CanonicalRecommendationDouble(0.0));
    assert(CanonicalRecommendationDouble(0.8) == "0.8");
    assert(CanonicalRecommendationDouble(0.1) == "0.1");
    assert(CanonicalRecommendationDouble(0.0001) == "0.0001");
    const auto assertRoundTrip = [](double value) {
        const std::string canonical = CanonicalRecommendationDouble(value);
        double parsed = 0.0;
        const auto result = std::from_chars(
            canonical.data(), canonical.data() + canonical.size(), parsed,
            std::chars_format::general);
        assert(result.ec == std::errc{});
        assert(result.ptr == canonical.data() + canonical.size());
        assert(parsed == value);
    };
    assertRoundTrip(1.7976931348623157e+308);
    assertRoundTrip(std::numeric_limits<double>::denorm_min());
    const double adjacentLow = 0.8;
    const double adjacentHigh = std::nextafter(
        adjacentLow, std::numeric_limits<double>::infinity());
    assertRoundTrip(adjacentLow);
    assertRoundTrip(adjacentHigh);
    assert(CanonicalRecommendationDouble(adjacentLow) !=
           CanonicalRecommendationDouble(adjacentHigh));
    assert(ErrorFrom([] { CanonicalRecommendationDouble(
        std::numeric_limits<double>::infinity()); }) ==
        "recommendation_identity_nonfinite_number");

    const EffectiveExperimentConfiguration configuration = ExampleConfiguration();
    const RecommendationCandidateIdentity identity =
        BuildRecommendationCandidateIdentity(configuration);
    assert(identity.configuration.symbol == "eurusd");
    assert(identity.canonicalText ==
        "experiment_recommendation_semantic_configuration_v17;symbol=eurusd;prediction_horizon=12;"
        "label_threshold=0.001;core_lr_mult=1;head_lr_mult=5;target_epochs=120;"
        "train_start_date=2010-01-01;train_end_date=2025-01-01;"
        "infer_start_date=2025-01-01;infer_end_date=2026-01-01;donchian20_mode=enabled;"
        "feature_warmup_scope=full_history_warmup;donchian_lookback=20");
    assert(identity.hash == RecommendationCandidateHash(configuration));
    const auto v16 = BuildRecommendationCandidateIdentity(
        configuration, RecommendationSemanticConfigurationVersion::v16);
    assert(v16.canonicalText ==
        "experiment_recommendation_semantic_configuration_v16;symbol=eurusd;prediction_horizon=12;"
        "label_threshold=0.001;core_lr_mult=1;head_lr_mult=5;target_epochs=120;"
        "train_start_date=2010-01-01;train_end_date=2025-01-01;"
        "infer_start_date=2025-01-01;infer_end_date=2026-01-01;donchian20_mode=enabled;"
        "feature_warmup_scope=full_history_warmup;donchian_lookback=20");
    const auto v15 = BuildRecommendationCandidateIdentity(
        configuration, RecommendationSemanticConfigurationVersion::v15);
    assert(v15.canonicalText ==
        "experiment_recommendation_semantic_configuration_v15;symbol=eurusd;prediction_horizon=12;"
        "label_threshold=0.001;core_lr_mult=1;head_lr_mult=5;target_epochs=120;"
        "train_start_date=2010-01-01;train_end_date=2025-01-01;"
        "infer_start_date=2025-01-01;infer_end_date=2026-01-01;donchian20_mode=enabled;"
        "feature_warmup_scope=full_history_warmup;donchian_lookback=20");
    const auto v14 = BuildRecommendationCandidateIdentity(
        configuration, RecommendationSemanticConfigurationVersion::v14);
    assert(v14.canonicalText ==
        "experiment_recommendation_semantic_configuration_v14;symbol=eurusd;prediction_horizon=12;"
        "label_threshold=0.001;core_lr_mult=1;head_lr_mult=5;target_epochs=120;"
        "train_start_date=2010-01-01;train_end_date=2025-01-01;"
        "infer_start_date=2025-01-01;infer_end_date=2026-01-01;donchian20_mode=enabled;"
        "feature_warmup_scope=full_history_warmup;donchian_lookback=20");
    const auto v13 = BuildRecommendationCandidateIdentity(
        configuration, RecommendationSemanticConfigurationVersion::v13);
    assert(v13.canonicalText ==
        "experiment_recommendation_semantic_configuration_v13;symbol=eurusd;prediction_horizon=12;"
        "label_threshold=0.001;core_lr_mult=1;head_lr_mult=5;target_epochs=120;"
        "train_start_date=2010-01-01;train_end_date=2025-01-01;"
        "infer_start_date=2025-01-01;infer_end_date=2026-01-01;donchian20_mode=enabled;"
        "feature_warmup_scope=full_history_warmup;donchian_lookback=20");
    const auto v12 = BuildRecommendationCandidateIdentity(
        configuration, RecommendationSemanticConfigurationVersion::v12);
    assert(v12.canonicalText ==
        "experiment_recommendation_semantic_configuration_v12;symbol=eurusd;prediction_horizon=12;"
        "label_threshold=0.001;core_lr_mult=1;head_lr_mult=5;target_epochs=120;"
        "train_start_date=2010-01-01;train_end_date=2025-01-01;"
        "infer_start_date=2025-01-01;infer_end_date=2026-01-01;donchian20_mode=enabled;"
        "feature_warmup_scope=full_history_warmup;donchian_lookback=20");
    assert(identity.canonicalText != v12.canonicalText);
    const auto v11 = BuildRecommendationCandidateIdentity(
        configuration, RecommendationSemanticConfigurationVersion::v11);
    assert(v11.canonicalText ==
        "experiment_recommendation_semantic_configuration_v11;symbol=eurusd;prediction_horizon=12;"
        "label_threshold=0.001;core_lr_mult=1;head_lr_mult=5;target_epochs=120;"
        "train_start_date=2010-01-01;train_end_date=2025-01-01;"
        "infer_start_date=2025-01-01;infer_end_date=2026-01-01;donchian20_mode=enabled;"
        "feature_warmup_scope=full_history_warmup;donchian_lookback=20");
    const auto v10 = BuildRecommendationCandidateIdentity(
        configuration, RecommendationSemanticConfigurationVersion::v10);
    assert(v10.canonicalText ==
        "experiment_recommendation_semantic_configuration_v10;symbol=eurusd;prediction_horizon=12;"
        "label_threshold=0.001;core_lr_mult=1;head_lr_mult=5;target_epochs=120;"
        "train_start_date=2010-01-01;train_end_date=2025-01-01;"
        "infer_start_date=2025-01-01;infer_end_date=2026-01-01;donchian20_mode=enabled;"
        "feature_warmup_scope=full_history_warmup;donchian_lookback=20");
    const auto v9 = BuildRecommendationCandidateIdentity(
        configuration, RecommendationSemanticConfigurationVersion::v9);
    assert(v9.canonicalText ==
        "experiment_recommendation_semantic_configuration_v9;symbol=eurusd;prediction_horizon=12;"
        "label_threshold=0.001;core_lr_mult=1;head_lr_mult=5;target_epochs=120;"
        "train_start_date=2010-01-01;train_end_date=2025-01-01;"
        "infer_start_date=2025-01-01;infer_end_date=2026-01-01;donchian20_mode=enabled;"
        "feature_warmup_scope=full_history_warmup;donchian_lookback=20");
    assert(identity.canonicalText != v9.canonicalText);
    const auto v8 = BuildRecommendationCandidateIdentity(
        configuration, RecommendationSemanticConfigurationVersion::v8);
    assert(v8.canonicalText ==
        "experiment_recommendation_semantic_configuration_v8;symbol=eurusd;prediction_horizon=12;"
        "label_threshold=0.001;core_lr_mult=1;head_lr_mult=5;target_epochs=120;"
        "train_start_date=2010-01-01;train_end_date=2025-01-01;"
        "infer_start_date=2025-01-01;infer_end_date=2026-01-01;donchian20_mode=enabled;"
        "feature_warmup_scope=full_history_warmup;donchian_lookback=20");
    assert(identity.canonicalText != v8.canonicalText);
    assert(identity.hash != v8.hash);
    const auto v7 = BuildRecommendationCandidateIdentity(
        configuration, RecommendationSemanticConfigurationVersion::v7);
    assert(v7.canonicalText ==
        "experiment_recommendation_semantic_configuration_v7;symbol=eurusd;prediction_horizon=12;"
        "label_threshold=0.001;core_lr_mult=1;head_lr_mult=5;target_epochs=120;"
        "train_start_date=2010-01-01;train_end_date=2025-01-01;"
        "infer_start_date=2025-01-01;infer_end_date=2026-01-01;donchian20_mode=enabled;"
        "feature_warmup_scope=full_history_warmup;donchian_lookback=20");
    assert(identity.canonicalText != v7.canonicalText);
    assert(identity.hash != v7.hash);
    const auto v6 = BuildRecommendationCandidateIdentity(
        configuration, RecommendationSemanticConfigurationVersion::v6);
    assert(v6.canonicalText ==
        "experiment_recommendation_semantic_configuration_v6;symbol=eurusd;prediction_horizon=12;"
        "label_threshold=0.001;core_lr_mult=1;head_lr_mult=5;target_epochs=120;"
        "train_start_date=2010-01-01;train_end_date=2025-01-01;"
        "infer_start_date=2025-01-01;infer_end_date=2026-01-01;donchian20_mode=enabled;"
        "feature_warmup_scope=full_history_warmup;donchian_lookback=20");
    assert(v6.hash == "fnv1a64:af809a41ed1afac2");
    assert(identity.canonicalText != v6.canonicalText);
    assert(identity.hash != v6.hash);
    const auto v3 = BuildRecommendationCandidateIdentity(
        configuration, RecommendationSemanticConfigurationVersion::v3);
    assert(v3.canonicalText ==
        "experiment_recommendation_semantic_configuration_v3;symbol=eurusd;prediction_horizon=12;"
        "label_threshold=0.001;core_lr_mult=1;head_lr_mult=5;target_epochs=120;"
        "train_start_date=2010-01-01;train_end_date=2025-01-01;"
        "infer_start_date=2025-01-01;infer_end_date=2026-01-01");
    assert(v3.hash == "fnv1a64:e55c515fcbe9a1ec");
    assert(RecommendationSemanticConfigurationVersionFromCanonicalText(
               v3.canonicalText) == RecommendationSemanticConfigurationVersion::v3);
    assert(RecommendationSemanticConfigurationVersionFromCanonicalText(
               identity.canonicalText) == RecommendationSemanticConfigurationVersion::v17);
    assert(RecommendationSemanticConfigurationVersionFromCanonicalText(
               v16.canonicalText) == RecommendationSemanticConfigurationVersion::v16);
    assert(RecommendationSemanticConfigurationVersionFromCanonicalText(
               v15.canonicalText) == RecommendationSemanticConfigurationVersion::v15);
    assert(RecommendationSemanticConfigurationVersionFromCanonicalText(
               v14.canonicalText) == RecommendationSemanticConfigurationVersion::v14);
    assert(RecommendationSemanticConfigurationVersionFromCanonicalText(
               v13.canonicalText) == RecommendationSemanticConfigurationVersion::v13);
    assert(RecommendationSemanticConfigurationVersionFromCanonicalText(
               v12.canonicalText) == RecommendationSemanticConfigurationVersion::v12);
    assert(RecommendationSemanticConfigurationVersionFromCanonicalText(
               v11.canonicalText) == RecommendationSemanticConfigurationVersion::v11);
    assert(RecommendationSemanticConfigurationVersionFromCanonicalText(
               v10.canonicalText) == RecommendationSemanticConfigurationVersion::v10);
    assert(RecommendationSemanticConfigurationVersionFromCanonicalText(
               v9.canonicalText) == RecommendationSemanticConfigurationVersion::v9);
    assert(RecommendationSemanticConfigurationVersionFromCanonicalText(
               v8.canonicalText) == RecommendationSemanticConfigurationVersion::v8);
    assert(RecommendationSemanticConfigurationVersionFromCanonicalText(
               v7.canonicalText) == RecommendationSemanticConfigurationVersion::v7);
    assert(RecommendationSemanticConfigurationVersionFromCanonicalText(
               v6.canonicalText) == RecommendationSemanticConfigurationVersion::v6);
    const auto v5 = BuildRecommendationCandidateIdentity(
        configuration, RecommendationSemanticConfigurationVersion::v5);
    assert(v5.canonicalText ==
        "experiment_recommendation_semantic_configuration_v5;symbol=eurusd;prediction_horizon=12;"
        "label_threshold=0.001;core_lr_mult=1;head_lr_mult=5;target_epochs=120;"
        "train_start_date=2010-01-01;train_end_date=2025-01-01;"
        "infer_start_date=2025-01-01;infer_end_date=2026-01-01;donchian20_mode=enabled;"
        "feature_warmup_scope=full_history_warmup");
    assert(RecommendationSemanticConfigurationVersionFromCanonicalText(
               v5.canonicalText) == RecommendationSemanticConfigurationVersion::v5);
    const auto v4 = BuildRecommendationCandidateIdentity(
        configuration, RecommendationSemanticConfigurationVersion::v4);
    assert(v4.canonicalText ==
        "experiment_recommendation_semantic_configuration_v4;symbol=eurusd;"
        "prediction_horizon=12;label_threshold=0.001;core_lr_mult=1;"
        "head_lr_mult=5;target_epochs=120;train_start_date=2010-01-01;"
        "train_end_date=2025-01-01;infer_start_date=2025-01-01;"
        "infer_end_date=2026-01-01;donchian20_mode=enabled");
    assert(RecommendationFeatureWarmupScopeFromCanonicalText(v4.canonicalText) ==
           EA::FeatureWarmupScope::LegacyColdBoundary);
    assert(RecommendationFeatureWarmupScopeFromCanonicalText(v5.canonicalText) ==
           EA::FeatureWarmupScope::FullHistoryWarmup);
    assert(RecommendationFeatureWarmupScopeFromCanonicalText(v6.canonicalText) ==
           EA::FeatureWarmupScope::FullHistoryWarmup);
    assert(RecommendationFeatureWarmupScopeFromCanonicalText(v7.canonicalText) ==
           EA::FeatureWarmupScope::FullHistoryWarmup);
    assert(RecommendationFeatureWarmupScopeFromCanonicalText(v8.canonicalText) ==
           EA::FeatureWarmupScope::FullHistoryWarmup);
    assert(RecommendationFeatureWarmupScopeFromCanonicalText(v9.canonicalText) ==
           EA::FeatureWarmupScope::FullHistoryWarmup);
    assert(RecommendationFeatureWarmupScopeFromCanonicalText(v15.canonicalText) ==
           EA::FeatureWarmupScope::FullHistoryWarmup);
    assert(RecommendationFeatureWarmupScopeFromCanonicalText(identity.canonicalText) ==
           EA::FeatureWarmupScope::FullHistoryWarmup);
    assert(RecommendationDonchianLookbackFromCanonicalText(v4.canonicalText) == 20);
    assert(RecommendationDonchianLookbackFromCanonicalText(v5.canonicalText) == 20);
    assert(RecommendationDonchianLookbackFromCanonicalText(v6.canonicalText) == 20);
    assert(RecommendationDonchianLookbackFromCanonicalText(v7.canonicalText) == 20);
    assert(RecommendationDonchianLookbackFromCanonicalText(v8.canonicalText) == 20);
    assert(RecommendationDonchianLookbackFromCanonicalText(v9.canonicalText) == 20);
    assert(RecommendationDonchianLookbackFromCanonicalText(v15.canonicalText) == 20);
    assert(RecommendationDonchianLookbackFromCanonicalText(identity.canonicalText) == 20);
    EffectiveExperimentConfiguration cold = configuration;
    cold.featureWarmupScope = EA::FeatureWarmupScope::LegacyColdBoundary;
    assert(BuildRecommendationCandidateIdentity(cold).hash != identity.hash);
    EffectiveExperimentConfiguration lookback10 = configuration;
    lookback10.donchianLookback = 10;
    assert(BuildRecommendationCandidateIdentity(lookback10).hash != identity.hash);
    EffectiveExperimentConfiguration zeroAblation = configuration;
    zeroAblation.donchian20Mode = Donchian20Mode::ZeroAblation;
    const auto zeroAblationIdentity =
        BuildRecommendationCandidateIdentity(zeroAblation);
    assert(zeroAblationIdentity.canonicalText.find(
               ";donchian20_mode=zero_ablation") != std::string::npos);
    assert(zeroAblationIdentity.hash ==
           RecommendationCanonicalHash(zeroAblationIdentity.canonicalText));
    assert(zeroAblationIdentity.hash != identity.hash);
    assert(zeroAblationIdentity.canonicalText != identity.canonicalText);
    assert(CanonicalExperimentDateText("2025-06-15") == "2025-06-15");
    assert(CanonicalExperimentDateText("2024-02-29") == "2024-02-29");
    for (const std::string invalidDate : {
             "2023-02-29",
             "2025-13-01",
             "2025-00-01",
             "2025-04-31",
             "2025-01-00",
             "2025-01-32",
             "2025-01-01T00:00:00Z",
             "2025-01-01+00:00",
             " 2025-01-01",
             "2025-01-01 ",
             "2025-01-01junk"})
    {
        assert(ErrorFrom([&] { CanonicalExperimentDateText(invalidDate); }) ==
               "invalid_recommendation_date");
    }

    EffectiveExperimentConfiguration equivalent = configuration;
    equivalent.symbol = "eurusd";
    assert(RecommendationCandidateHash(equivalent) == identity.hash);

    AssertIdentityChange(configuration,
        [](auto& value) { value.symbol = "GBPUSD"; });
    AssertIdentityChange(configuration,
        [](auto& value) { ++value.predictionHorizon; });
    AssertIdentityChange(configuration,
        [](auto& value) { value.labelThreshold += 0.0001; });
    AssertIdentityChange(configuration,
        [](auto& value) { value.coreLrMult = 2.0; });
    AssertIdentityChange(configuration,
        [](auto& value) { value.headLrMult = 6.0; });
    AssertIdentityChange(configuration,
        [](auto& value) { ++value.targetEpochs; });
    AssertIdentityChange(configuration,
        [](auto& value) { value.trainStartDate = "2010-01-02"; });
    AssertIdentityChange(configuration,
        [](auto& value) { value.trainEndDate = "2025-01-02"; });
    AssertIdentityChange(configuration,
        [](auto& value) { value.inferStartDate = "2025-01-02"; });
    AssertIdentityChange(configuration,
        [](auto& value) { value.inferEndDate = "2026-01-02"; });
    AssertIdentityChange(configuration,
        [](auto& value) { value.coreLrMult.reset(); });
    AssertIdentityChange(configuration,
        [](auto& value) { value.inferStartDate.reset(); });

    EffectiveExperimentConfiguration nullOptional = configuration;
    nullOptional.coreLrMult.reset();
    EffectiveExperimentConfiguration zeroOptional = configuration;
    zeroOptional.coreLrMult = 0.0;
    assert(RecommendationCandidateHash(nullOptional) !=
           RecommendationCandidateHash(zeroOptional));

    ExperimentInvocationConfiguration freshInvocation{
        configuration, 20, std::nullopt};
    ExperimentInvocationConfiguration checkpoint40{
        configuration, 40, std::nullopt};
    ExperimentInvocationConfiguration resume41{configuration, 20, 41};
    ExperimentInvocationConfiguration resume42{configuration, 20, 42};
    assert(RecommendationCandidateHash(freshInvocation.configuration) ==
           RecommendationCandidateHash(resume41.configuration));
    assert(RecommendationCandidateHash(resume41.configuration) ==
           RecommendationCandidateHash(resume42.configuration));
    assert(RecommendationCandidateHash(freshInvocation.configuration) ==
           RecommendationCandidateHash(checkpoint40.configuration));
    assert(ExperimentInvocationHash(freshInvocation) !=
           ExperimentInvocationHash(checkpoint40));
    assert(ExperimentInvocationHash(freshInvocation) !=
           ExperimentInvocationHash(resume41));
    assert(ExperimentInvocationHash(resume41) !=
           ExperimentInvocationHash(resume42));
    const RecommendationInvocationIdentity invocationIdentity =
        BuildRecommendationInvocationIdentity(resume41);
    assert(invocationIdentity.invocation.configuration.symbol == "eurusd");
    assert(invocationIdentity.hash == ExperimentInvocationHash(resume41));
    assert(invocationIdentity.hash == RecommendationCanonicalHash(
        invocationIdentity.canonicalText));
    assert(invocationIdentity.hash.starts_with("fnv1a64:"));
    assert(invocationIdentity.canonicalText.starts_with(
        "experiment_recommendation_invocation_v2;"));
    assert(invocationIdentity.canonicalText.find(
        ";checkpoint_interval=20;resume_model_id=41") != std::string::npos);
    RecommendationSource source;
    source.experimentId = 143;
    source.invocation = resume41;
    assert(source.invocation.resumeModelId == 41);
    assert(RecommendationCandidateHash(source.invocation.configuration) ==
           identity.hash);
    assert(ErrorFrom([&] {
        ExperimentInvocationConfiguration invalid{configuration, 20, 0};
        (void)ExperimentInvocationHash(invalid);
    }) == "recommendation_invocation_resume_model_id_must_be_positive");
    assert(ErrorFrom([&] {
        ExperimentInvocationConfiguration invalid{
            configuration, 0, std::nullopt};
        (void)ExperimentInvocationHash(invalid);
    }) == "recommendation_invocation_checkpoint_interval_must_be_positive");

    for (RecommendationStatus status : {
             RecommendationStatus::proposed,
             RecommendationStatus::rejected,
             RecommendationStatus::expired,
             RecommendationStatus::approved})
    {
        const std::string text = RecommendationStatusText(status);
        assert(ParseRecommendationStatus(text) == status);
    }
    assert(!ParseRecommendationStatus("invalid"));
    assert(RecommendationDuplicateTypeText(
               RecommendationDuplicateType::existingExperiment) ==
           "existing_experiment");

    std::cout << "DEFAULT_RECOMMENDATION_POLICY_CANONICAL="
              << RecommendationPolicyCanonicalText(defaults) << '\n'
              << "DEFAULT_RECOMMENDATION_POLICY_HASH="
              << RecommendationPolicyHash(defaults) << '\n'
              << "EXAMPLE_RECOMMENDATION_CANDIDATE_CANONICAL="
              << identity.canonicalText << '\n'
              << "EXAMPLE_RECOMMENDATION_CANDIDATE_HASH="
              << identity.hash << '\n'
              << "EXAMPLE_RECOMMENDATION_INVOCATION_CANONICAL="
              << invocationIdentity.canonicalText << '\n'
              << "EXAMPLE_RECOMMENDATION_INVOCATION_HASH="
              << invocationIdentity.hash << '\n'
              << "ExperimentRecommendationTests passed\n";
    return 0;
}
