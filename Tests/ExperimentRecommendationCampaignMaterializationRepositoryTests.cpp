#include "../Sources/ExperimentRecommendationCampaignMaterializationRepository.hpp"
#include "../Sources/ExperimentRecommendationCampaignMaterializationService.hpp"

#include "../Sources/ExperimentRecommendation.hpp"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <cassert>
#include <cctype>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>

using namespace EA::ExperimentRecommendation;

namespace
{
std::string ReadFile(const std::string& path)
{
    std::ifstream input{path};
    if (!input) throw std::runtime_error("unable_to_read_" + path);
    return {std::istreambuf_iterator<char>{input}, {}};
}

bool SafeDatabase(const std::string& value)
{
    std::string lower = value;
    std::transform(lower.begin(), lower.end(), lower.begin(),
        [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return !value.empty() && lower != "lstm" && std::all_of(
        value.begin(), value.end(), [](unsigned char value) {
            return std::isalnum(value) || value == '_' || value == '-';
        });
}

bool PermissionDenied(const pqxx::sql_error& error)
{
    if (error.sqlstate() == "42501") return true;
    return error.sqlstate().empty() &&
        std::string{error.what()}.find("permission denied") != std::string::npos;
}

RecommendationCampaignPlanningPolicy Policy()
{
    RecommendationCampaignPlanningPolicy value;
    value.enabled = true;
    value.maximumPerSourceExperiment.reset();
    return value;
}

RecommendationCampaignCandidateInput Candidate()
{
    RecommendationCampaignCandidateInput value;
    value.rankingSnapshotId = 7;
    value.rankingSnapshotIdentityCanonical = "materialization-snapshot";
    value.rankingSnapshotIdentityHash = RecommendationCanonicalHash(
        value.rankingSnapshotIdentityCanonical);
    value.rankingPolicyCanonical = "materialization-ranking-policy";
    value.rankingPolicyHash = RecommendationCanonicalHash(
        value.rankingPolicyCanonical);
    value.rankingVersion = 1;
    value.rankingMemberId = 101;
    value.rankingPosition = 1;
    value.rankingBucket = "advisory_ready";
    value.rankingScore = 0.9;
    value.recommendationId = 1;
    value.sourceExperimentId = 501;
    value.symbol = "eurusd";
    value.predictionHorizon = 12;
    value.family = "core_lr_mult";
    value.targetEpochs = 120;
    value.leaderScore = 0.9;
    value.inferenceAccuracy = 0.8;
    value.predictedNeutralProportion = 0.2;
    value.recommendationSemanticCanonical = "materialization-semantic";
    value.recommendationSemanticHash = RecommendationCanonicalHash(
        value.recommendationSemanticCanonical);
    value.recommendationInvocationCanonical = "materialization-invocation";
    value.recommendationInvocationHash = RecommendationCanonicalHash(
        value.recommendationInvocationCanonical);
    return value;
}

struct Fixture
{
    RecommendationCampaignPlan plan;
    RecommendationCampaignApprovalEvidence approval;
    ProposedExperimentSpecification proposal;
    std::string recommendationSemanticCanonical;
    RecommendationCampaignMaterializationEvidence materialization;
};

Fixture BuildFixture(
    RecommendationSemanticConfigurationVersion semanticVersion =
        RecommendationSemanticConfigurationVersion::v5,
    Donchian20Mode donchian20Mode = Donchian20Mode::Enabled,
    EA::FeatureWarmupScope featureWarmupScope =
        EA::FeatureWarmupScope::FullHistoryWarmup)
{
    Fixture fixture;
    RecommendationCampaignPlanInput input;
    input.policy = Policy();
    input.scope.rankingSnapshotId = 7;
    input.rankingSnapshotIdentityCanonical = "materialization-snapshot";
    input.rankingSnapshotIdentityHash = RecommendationCanonicalHash(
        input.rankingSnapshotIdentityCanonical);
    input.generatedAt = "display";
    input.candidates = {Candidate()};
    fixture.plan = PlanRecommendationCampaign(input);
    const auto review = ReviewRecommendationCampaignPlan(fixture.plan);
    RecommendationCampaignApprovalRequest approvalRequest;
    approvalRequest.expectedCampaignReviewIdentityHash = review.identityHash;
    approvalRequest.reviewerIdentity = "reviewer";
    approvalRequest.reasonText = "Approved exact campaign.";
    fixture.approval = BuildRecommendationCampaignApprovalEvidence(
        7, input.rankingSnapshotIdentityCanonical,
        input.rankingSnapshotIdentityHash, fixture.plan, review,
        approvalRequest);

    RecommendationConversionRequest conversion;
    conversion.recommendationExists = true;
    conversion.recommendationId = 1;
    conversion.recommendationStatus = RecommendationStatus::approved;
    conversion.sourceExperimentId = 501;
    conversion.recommendationSourceExperimentId = 501;
    conversion.reviewAuthorization = {true, 1,
        RecommendationReviewAction::approve, RecommendationStatus::approved,
        true, false, "authorization", RecommendationCanonicalHash("authorization")};
    conversion.evaluation.state = RecommendationConversionEvidenceState::completed;
    conversion.evaluation.valid = true;
    conversion.evaluation.recommendationId = 1;
    conversion.evaluation.sourceExperimentId = 501;
    conversion.evaluation.eligibility = RecommendationEligibility::eligible;
    conversion.evaluation.disposition =
        RecommendationEvaluationDisposition::advisoryReady;
    conversion.evaluation.evaluationIdentityCanonical = "evaluation";
    conversion.evaluation.evaluationIdentityHash = RecommendationCanonicalHash("evaluation");
    conversion.evaluation.evaluationPolicyCanonical = "evaluation-policy";
    conversion.evaluation.evaluationPolicyHash = RecommendationCanonicalHash("evaluation-policy");
    conversion.evaluation.scoringPolicyHash = RecommendationCanonicalHash("scoring-policy");
    conversion.score.state = RecommendationConversionEvidenceState::completed;
    conversion.score.valid = true;
    conversion.score.recommendationId = 1;
    conversion.score.finalScore = 0.9;
    conversion.score.scoringPolicyCanonical = "scoring-policy";
    conversion.score.scoringPolicyHash = RecommendationCanonicalHash("scoring-policy");
    conversion.sourceInvocation.configuration.symbol = "eurusd";
    conversion.sourceInvocation.configuration.predictionHorizon = 12;
    conversion.sourceInvocation.configuration.labelThreshold = 0.001;
    conversion.sourceInvocation.configuration.coreLrMult = 1.0;
    conversion.sourceInvocation.configuration.headLrMult = 1.0;
    conversion.sourceInvocation.configuration.targetEpochs = 120;
    conversion.sourceInvocation.configuration.trainStartDate = "2026-01-01";
    conversion.sourceInvocation.configuration.trainEndDate = "2026-02-01";
    conversion.sourceInvocation.configuration.donchian20Mode = donchian20Mode;
    conversion.sourceInvocation.configuration.featureWarmupScope = featureWarmupScope;
    conversion.sourceInvocation.checkpointInterval = 20;
    auto proposed = conversion.sourceInvocation;
    proposed.configuration.coreLrMult = 1.1;
    const auto invocation = BuildRecommendationInvocationIdentity(
        proposed, semanticVersion);
    const auto semantic = BuildRecommendationCandidateIdentity(
        invocation.invocation.configuration, semanticVersion);
    conversion.recommendationInvocationCanonical = invocation.canonicalText;
    conversion.recommendationInvocationHash = invocation.hash;
    conversion.recommendationSemanticCanonical = semantic.canonicalText;
    conversion.recommendationSemanticHash = semantic.hash;
    conversion.mutations.push_back({
        "core_lr_mult", CanonicalRecommendationDouble(1.0),
        CanonicalRecommendationDouble(1.1)});
    const auto result = BuildProposedExperimentSpecification(conversion);
    assert(result.eligibility.eligible && result.proposal);
    fixture.proposal = *result.proposal;
    fixture.recommendationSemanticCanonical =
        BuildRecommendationCandidateIdentity(
            fixture.proposal.proposedInvocation.configuration, semanticVersion).canonicalText;
    RecommendationCampaignMaterializationRequest request{
        42, "operator", "Materialize exact proposal set."};
    fixture.materialization = BuildRecommendationCampaignMaterializationEvidence(
        request, fixture.approval, fixture.plan, {fixture.proposal});
    return fixture;
}

void InsertApproval(pqxx::transaction_base& tx, const Fixture& fixture)
{
    const auto& e = fixture.approval;
    const auto& s = e.summary;
    tx.exec(
        "INSERT INTO experiment_recommendation_campaign_approval VALUES("
        "42,1,7,$1,$2,$3,$4,$5,$6,$7,1,$8,$9,0,$10,$11,$12,$13,$14,$15,$16,"
        "$17,$18,$19,$20,true,'approved',$21,$22,$23,$24,now());",
        pqxx::params{e.rankingSnapshotIdentityCanonical,
            e.rankingSnapshotIdentityHash,e.planningPolicyCanonical,
            e.planningPolicyHash,e.planningScopeCanonical,
            e.campaignPlanIdentityCanonical,e.campaignPlanIdentityHash,
            e.campaignReviewIdentityCanonical,e.campaignReviewIdentityHash,
            s.candidateCount,s.selectedCount,s.excludedCount,
            s.duplicateGroupCount,s.duplicateCandidateCount,
            s.consideredFamilyCount,s.selectedFamilyCount,
            s.consideredSymbolCount,s.selectedSymbolCount,
            s.consideredHorizonCount,s.selectedHorizonCount,
            e.reviewerIdentity,e.reasonText,e.approvalIdentityCanonical,
            e.approvalIdentityHash});
}

void InsertManifestOnly(
    pqxx::transaction_base& tx,
    const Fixture& fixture,
    bool forgeApprovalHash)
{
    const auto& evidence = fixture.materialization;
    const auto& approval = fixture.approval;
    tx.exec(
        "INSERT INTO experiment_recommendation_campaign_materialization ("
        "recommendation_campaign_approval_id,materialization_contract_version,"
        "approval_identity_canonical,approval_identity_hash,"
        "recommendation_ranking_snapshot_id,ranking_snapshot_identity_canonical,"
        "ranking_snapshot_identity_hash,planning_policy_canonical,"
        "planning_policy_hash,planning_scope_canonical,"
        "campaign_plan_identity_canonical,campaign_plan_identity_hash,"
        "campaign_review_identity_canonical,campaign_review_identity_hash,"
        "approval_decision,approval_reviewer_identity,approval_reason_text,"
        "materialized_by,materialization_reason_text,selected_member_count,"
        "initially_created_proposal_count,initially_reused_proposal_count,"
        "materialization_identity_canonical,materialization_identity_hash) "
        "VALUES($1,1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,'approved',"
        "$14,$15,$16,$17,1,1,0,$18,$19);",
        pqxx::params{evidence.campaignApprovalId,
            approval.approvalIdentityCanonical,
            forgeApprovalHash ? RecommendationCanonicalHash("forged")
                              : approval.approvalIdentityHash,
            approval.rankingSnapshotId,
            approval.rankingSnapshotIdentityCanonical,
            approval.rankingSnapshotIdentityHash,
            approval.planningPolicyCanonical,approval.planningPolicyHash,
            approval.planningScopeCanonical,
            approval.campaignPlanIdentityCanonical,
            approval.campaignPlanIdentityHash,
            approval.campaignReviewIdentityCanonical,
            approval.campaignReviewIdentityHash,approval.reviewerIdentity,
            approval.reasonText,evidence.operatorIdentity,evidence.reasonText,
            evidence.materializationIdentityCanonical,
            evidence.materializationIdentityHash});
}
} // namespace

int main()
{
    const char* configured = std::getenv("LSTM_TEST_DB_NAME");
    if (!configured || !SafeDatabase(configured)) return 2;
    const std::string database = configured;
    const std::string user = std::getenv("USER") ? std::getenv("USER") : "vjp";
    const std::string schema = "phase4d_materialization_" + std::to_string(getpid());
    const std::string ownerString = "host=127.0.0.1 dbname=" + database + " user=" + user;
    const std::string runtimeString = "host=127.0.0.1 dbname=" + database +
        " user=pqxx options='-c search_path=" + schema + "'";
    pqxx::connection owner{ownerString};
    const Fixture fixture = BuildFixture();
    const Fixture historicalV3 = BuildFixture(
        RecommendationSemanticConfigurationVersion::v3,
        Donchian20Mode::ZeroAblation);
    assert(fixture.recommendationSemanticCanonical.find(
               "experiment_recommendation_semantic_configuration_v5;") == 0);
    assert(fixture.recommendationSemanticCanonical.find(
               ";donchian20_mode=enabled") != std::string::npos);
    assert(fixture.recommendationSemanticCanonical.find(
               ";feature_warmup_scope=full_history_warmup") != std::string::npos);
    assert(RecommendationSemanticConfigurationVersionFromCanonicalText(
               fixture.recommendationSemanticCanonical) ==
           RecommendationSemanticConfigurationVersion::v5);
    assert(historicalV3.recommendationSemanticCanonical.find(
               "experiment_recommendation_semantic_configuration_v3;") == 0);
    assert(historicalV3.recommendationSemanticCanonical.find(
               "donchian20_mode") == std::string::npos);
    assert(RecommendationSemanticConfigurationVersionFromCanonicalText(
               historicalV3.recommendationSemanticCanonical) ==
           RecommendationSemanticConfigurationVersion::v3);
    assert(historicalV3.proposal.proposedInvocationCanonical.find(
               "donchian20_mode") == std::string::npos);
    try
    {
        pqxx::work setup{owner};
        setup.exec("CREATE SCHEMA " + setup.quote_name(schema));
        setup.exec("SET LOCAL search_path TO " + setup.quote_name(schema));
        setup.exec("CREATE TABLE model(model_id bigint PRIMARY KEY);"
                   "CREATE TABLE experiment("
                   "experiment_id bigint PRIMARY KEY,symbol text,"
                   "prediction_horizon integer,c_next_threshold double precision,"
                   "core_lr_mult double precision,head_lr_mult double precision,"
                   "target_epochs integer,checkpoint_interval integer,"
                   "train_start timestamptz,train_end timestamptz,"
                   "infer_start timestamptz,infer_end timestamptz,"
                   "resume_model_id bigint,donchian20_mode text,feature_warmup_scope text);"
                   "CREATE TABLE experiment_recommendation("
                   "recommendation_id bigint PRIMARY KEY,status text,"
                   "source_experiment_id bigint,"
                   "semantic_configuration_canonical text,semantic_hash text,"
                   "invocation_configuration_canonical text,invocation_hash text,"
                   "changed_parameter text,source_value_canonical text,"
                   "proposed_value_canonical text);"
                   "CREATE TABLE experiment_recommendation_ranking_snapshot("
                   "recommendation_ranking_snapshot_id bigint PRIMARY KEY);"
                   "CREATE TABLE experiment_recommendation_ranking_member("
                   "recommendation_ranking_member_id bigint PRIMARY KEY,"
                   "recommendation_ranking_snapshot_id bigint NOT NULL,"
                   "recommendation_evaluation_result_id bigint NOT NULL,"
                   "recommendation_id bigint NOT NULL,"
                   "global_ordinal integer NOT NULL,"
                   "recommendation_semantic_hash text NOT NULL,bucket text NOT NULL,"
                   "source_experiment_id bigint NOT NULL);"
                   "CREATE TABLE experiment_recommendation_evaluation_run("
                   "recommendation_evaluation_run_id bigint PRIMARY KEY,status text,"
                   "evaluation_policy_canonical text,evaluation_policy_hash text,"
                   "scoring_policy_hash text);"
                   "CREATE TABLE experiment_recommendation_evaluation_result("
                   "recommendation_evaluation_result_id bigint PRIMARY KEY,"
                   "recommendation_evaluation_run_id bigint,recommendation_id bigint,"
                   "recommendation_semantic_canonical text,"
                   "recommendation_semantic_hash text,"
                   "evaluation_identity_canonical text,evaluation_identity_hash text,"
                   "source_experiment_id bigint,eligibility text,disposition text);"
                   "CREATE TABLE experiment_recommendation_score_run("
                   "recommendation_score_run_id bigint PRIMARY KEY,status text);"
                   "CREATE TABLE experiment_recommendation_score("
                   "recommendation_score_id bigint PRIMARY KEY,"
                   "recommendation_score_run_id bigint,recommendation_id bigint,"
                   "recommendation_semantic_canonical text,source_experiment_id bigint,"
                   "final_score double precision,score_status text,"
                   "scoring_policy_canonical text,scoring_policy_hash text);"
                   "CREATE TABLE experiment_recommendation_review_event("
                   "recommendation_review_event_id bigint PRIMARY KEY,"
                   "recommendation_id bigint,recommendation_score_id bigint,"
                   "action text,resulting_status text,"
                   "recommendation_semantic_canonical text,"
                   "recommendation_semantic_hash text,source_experiment_id bigint);"
                   "CREATE TABLE experiment_recommendation_conversion_review_decision("
                   "recommendation_conversion_review_decision_id bigint PRIMARY KEY);"
                   "CREATE TABLE experiment_recommendation_conversion_execution("
                   "recommendation_conversion_execution_id bigint PRIMARY KEY);"
                   "CREATE TABLE experiment_recommendation_conversion_activation("
                   "recommendation_conversion_activation_id bigint PRIMARY KEY);"
                   "INSERT INTO experiment_recommendation_ranking_snapshot VALUES(7);"
                   "GRANT USAGE ON SCHEMA " + setup.quote_name(schema) + " TO pqxx;"
                   "GRANT SELECT ON ALL TABLES IN SCHEMA " + setup.quote_name(schema) + " TO pqxx;");
        setup.exec(
            "INSERT INTO experiment VALUES(501,'eurusd',12,0.001,1.0,1.0,"
            "120,20,'2026-01-01 00:00:00 America/Chicago',"
            "'2026-02-01 00:00:00 America/Chicago',NULL,NULL,NULL,'enabled',"
            "'full_history_warmup');");
        setup.exec(
            "INSERT INTO experiment_recommendation VALUES("
            "1,'approved',501,$1,$2,$3,$4,'core_lr_mult',$5,$6);",
            pqxx::params{fixture.recommendationSemanticCanonical,
                fixture.proposal.recommendationSemanticHash,
                fixture.proposal.proposedInvocationCanonical,
                RecommendationCanonicalHash(
                    fixture.proposal.proposedInvocationCanonical),
                fixture.proposal.sourceValueCanonical,
                fixture.proposal.proposedValueCanonical});
        setup.exec(
            "INSERT INTO experiment_recommendation_evaluation_run VALUES("
            "71,'completed','evaluation-policy',$1,$2);",
            pqxx::params{RecommendationCanonicalHash("evaluation-policy"),
                RecommendationCanonicalHash("scoring-policy")});
        setup.exec(
            "INSERT INTO experiment_recommendation_evaluation_result VALUES("
            "81,71,1,$1,$2,'evaluation',$3,501,'eligible','advisory_ready');",
            pqxx::params{fixture.recommendationSemanticCanonical,
                fixture.proposal.recommendationSemanticHash,
                RecommendationCanonicalHash("evaluation")});
        setup.exec(
            "INSERT INTO experiment_recommendation_score_run VALUES(91,'completed');");
        setup.exec(
            "INSERT INTO experiment_recommendation_score VALUES("
            "92,91,1,$1,501,0.9,'scored','scoring-policy',$2);",
            pqxx::params{fixture.recommendationSemanticCanonical,
                RecommendationCanonicalHash("scoring-policy")});
        setup.exec(
            "INSERT INTO experiment_recommendation_score_run VALUES(94,'completed');");
        setup.exec(
            "INSERT INTO experiment_recommendation_score VALUES("
            "95,94,1,$1,501,0.85,'scored','scoring-policy',$2);",
            pqxx::params{fixture.recommendationSemanticCanonical,
                RecommendationCanonicalHash("scoring-policy")});
        setup.exec(
            "INSERT INTO experiment_recommendation_review_event VALUES("
            "90,1,92,'reject','rejected',$1,$2,501),("
            "93,1,NULL,'approve','approved',$1,$2,501);",
            pqxx::params{fixture.recommendationSemanticCanonical,
                fixture.proposal.recommendationSemanticHash});
        setup.exec(
            "INSERT INTO experiment_recommendation_ranking_member VALUES("
            "101,7,81,1,1,$1,'advisory_ready',501);",
            pqxx::params{fixture.proposal.recommendationSemanticHash});
        setup.exec(ReadFile("Database/migrations/036_experiment_recommendation_conversion_proposal.sql"));
        setup.exec(ReadFile("Database/migrations/040_experiment_recommendation_campaign_approval.sql"));
        setup.exec(ReadFile("Database/migrations/041_experiment_recommendation_campaign_materialization.sql"));
        setup.exec(ReadFile("Database/migrations/041_experiment_recommendation_campaign_materialization.sql"));
        setup.exec(ReadFile("Tests/ExperimentRecommendationCampaignMaterializationMigrationTests.sql"));
        InsertApproval(setup, fixture);
        setup.commit();

        pqxx::connection runtime{runtimeString};
        assert(RecommendationCampaignMaterializationSchemaExists(runtime));
        {
            pqxx::work tx{runtime};
            const auto conversionRequest =
                LoadRecommendationCampaignConversionRequest(tx, 7, 101, 1);
            const auto conversion = BuildProposedExperimentSpecification(
                conversionRequest);
            assert(conversion.eligibility.eligible && conversion.proposal);
            assert(conversion.proposal->recommendationId == 1);
            assert(conversion.proposal->sourceExperimentId == 501);
            assert(conversionRequest.score.finalScore == 0.85);
            assert(conversionRequest.sourceInvocation.configuration.trainStartDate ==
                   "2026-01-01");
            assert(conversionRequest.sourceInvocation.configuration.donchian20Mode ==
                   Donchian20Mode::Enabled);
            assert(conversionRequest.sourceInvocation.configuration.featureWarmupScope ==
                   EA::FeatureWarmupScope::FullHistoryWarmup);
        }
        {
            pqxx::work mutate{owner};
            mutate.exec("SET LOCAL search_path TO " +
                mutate.quote_name(schema));
            mutate.exec("UPDATE experiment SET donchian20_mode='zero_ablation' "
                        "WHERE experiment_id=501;");
            mutate.commit();

            pqxx::work tx{runtime};
            const auto conversionRequest =
                LoadRecommendationCampaignConversionRequest(tx, 7, 101, 1);
            const auto conversion = BuildProposedExperimentSpecification(
                conversionRequest);
            assert(!conversion.eligibility.eligible);
            assert(conversion.eligibility.reason ==
                   RecommendationConversionReason::inconsistentProvenance);

            pqxx::work restore{owner};
            restore.exec("SET LOCAL search_path TO " +
                restore.quote_name(schema));
            restore.exec("UPDATE experiment SET donchian20_mode='enabled' "
                         "WHERE experiment_id=501;");
            restore.commit();
        }
        {
            // A historical v3 recommendation remains reproducible under its
            // own persisted contract, even though the source row now carries
            // the durable v4-only mode column.
            pqxx::work legacy{owner};
            legacy.exec("SET LOCAL search_path TO " + legacy.quote_name(schema));
            legacy.exec(
                "UPDATE experiment_recommendation SET "
                "semantic_configuration_canonical=$1,semantic_hash=$2,"
                "invocation_configuration_canonical=$3,invocation_hash=$4 "
                "WHERE recommendation_id=1;",
                pqxx::params{historicalV3.recommendationSemanticCanonical,
                    historicalV3.proposal.recommendationSemanticHash,
                    historicalV3.proposal.proposedInvocationCanonical,
                    RecommendationCanonicalHash(
                        historicalV3.proposal.proposedInvocationCanonical)});
            legacy.exec(
                "UPDATE experiment_recommendation_evaluation_result SET "
                "recommendation_semantic_canonical=$1,"
                "recommendation_semantic_hash=$2 WHERE recommendation_id=1;",
                pqxx::params{historicalV3.recommendationSemanticCanonical,
                    historicalV3.proposal.recommendationSemanticHash});
            legacy.exec(
                "UPDATE experiment_recommendation_score SET "
                "recommendation_semantic_canonical=$1 WHERE recommendation_id=1;",
                pqxx::params{historicalV3.recommendationSemanticCanonical});
            legacy.exec(
                "UPDATE experiment_recommendation_review_event SET "
                "recommendation_semantic_canonical=$1,"
                "recommendation_semantic_hash=$2 WHERE recommendation_id=1;",
                pqxx::params{historicalV3.recommendationSemanticCanonical,
                    historicalV3.proposal.recommendationSemanticHash});
            legacy.exec(
                "UPDATE experiment_recommendation_ranking_member SET "
                "recommendation_semantic_hash=$1 WHERE recommendation_id=1;",
                pqxx::params{historicalV3.proposal.recommendationSemanticHash});
            legacy.commit();

            pqxx::work tx{runtime};
            const auto request =
                LoadRecommendationCampaignConversionRequest(tx, 7, 101, 1);
            const auto conversion = BuildProposedExperimentSpecification(request);
            assert(conversion.eligibility.eligible && conversion.proposal);
            assert(conversion.proposal->proposedInvocationCanonical.find(
                "donchian20_mode") == std::string::npos);
            assert(conversion.proposal->proposedInvocation.configuration.
                       featureWarmupScope ==
                   EA::FeatureWarmupScope::LegacyColdBoundary);

            pqxx::work restore{owner};
            restore.exec("SET LOCAL search_path TO " +
                restore.quote_name(schema));
            restore.exec(
                "UPDATE experiment_recommendation SET "
                "semantic_configuration_canonical=$1,semantic_hash=$2,"
                "invocation_configuration_canonical=$3,invocation_hash=$4 "
                "WHERE recommendation_id=1;",
                pqxx::params{fixture.recommendationSemanticCanonical,
                    fixture.proposal.recommendationSemanticHash,
                    fixture.proposal.proposedInvocationCanonical,
                    RecommendationCanonicalHash(
                        fixture.proposal.proposedInvocationCanonical)});
            restore.exec(
                "UPDATE experiment_recommendation_evaluation_result SET "
                "recommendation_semantic_canonical=$1,"
                "recommendation_semantic_hash=$2 WHERE recommendation_id=1;",
                pqxx::params{fixture.recommendationSemanticCanonical,
                    fixture.proposal.recommendationSemanticHash});
            restore.exec(
                "UPDATE experiment_recommendation_score SET "
                "recommendation_semantic_canonical=$1 WHERE recommendation_id=1;",
                pqxx::params{fixture.recommendationSemanticCanonical});
            restore.exec(
                "UPDATE experiment_recommendation_review_event SET "
                "recommendation_semantic_canonical=$1,"
                "recommendation_semantic_hash=$2 WHERE recommendation_id=1;",
                pqxx::params{fixture.recommendationSemanticCanonical,
                    fixture.proposal.recommendationSemanticHash});
            restore.exec(
                "UPDATE experiment_recommendation_ranking_member SET "
                "recommendation_semantic_hash=$1 WHERE recommendation_id=1;",
                pqxx::params{fixture.proposal.recommendationSemanticHash});
            restore.commit();
        }
        {
            bool rejected = false;
            try
            {
                pqxx::work forged{runtime};
                InsertManifestOnly(forged, fixture, true);
                forged.commit();
            }
            catch (const pqxx::check_violation&)
            {
                rejected = true;
            }
            assert(rejected);
        }
        {
            bool rejected = false;
            try
            {
                pqxx::work incomplete{runtime};
                InsertManifestOnly(incomplete, fixture, false);
                incomplete.commit();
            }
            catch (const pqxx::check_violation&)
            {
                rejected = true;
            }
            assert(rejected);
        }
        long long recordedId = 0;
        {
            pqxx::work tx{runtime};
            const auto result = PersistRecommendationCampaignMaterialization(
                tx, fixture.materialization);
            assert(result.outcome ==
                RecommendationCampaignMaterializationPersistOutcome::recorded);
            assert(result.newlyCreatedProposalCount == 1);
            recordedId = result.materialization.materializationId;
            tx.commit();
        }
        {
            pqxx::work tx{runtime};
            const auto replay = PersistRecommendationCampaignMaterialization(
                tx, fixture.materialization);
            assert(replay.outcome ==
                RecommendationCampaignMaterializationPersistOutcome::existingIdentical);
            assert(replay.newlyCreatedProposalCount == 0);
            assert(replay.materialization.members.size() == 1);
            tx.commit();
        }
        const auto found = FindRecommendationCampaignMaterialization(
            runtime, recordedId);
        assert(found && found->campaignApprovalId == 42);
        assert(found->members[0].conversionProposalId > 0);
        assert(ListRecommendationCampaignMaterializations(runtime, 42, 10).size() == 1);
        {
            std::ostringstream output;
            std::ostringstream errors;
            const int status = RunMaterializeRecommendationCampaignCommand(
                runtimeString,
                RecommendationCampaignMaterializationRequest{
                    42, "operator", "Materialize exact proposal set."},
                output, errors);
            assert(status == 0);
            assert(errors.str().empty());
            assert(output.str().find(
                "RECOMMENDATION_CAMPAIGN_MATERIALIZATION_ALREADY_RECORDED") !=
                std::string::npos);
            assert(output.str().find(
                "campaign_materialization_existing_identical=true") !=
                std::string::npos);
            assert(output.str().find(
                "conversion_proposals_created_count=0") != std::string::npos);
        }
        {
            bool rejected = false;
            try
            {
                pqxx::work forged{runtime};
                forged.exec(
                    "INSERT INTO experiment_recommendation_campaign_"
                    "materialization_member ("
                    "recommendation_campaign_materialization_id,member_ordinal,"
                    "recommendation_ranking_member_id,recommendation_id,"
                    "source_experiment_id,ranking_position,"
                    "selected_member_identity_canonical,"
                    "selected_member_identity_hash,"
                    "recommendation_conversion_proposal_id,"
                    "proposal_identity_canonical,proposal_identity_hash) "
                    "VALUES($1,2,101,1,501,2,'forged',$2,$3,$4,$5);",
                    pqxx::params{recordedId,
                        RecommendationCanonicalHash("forged"),
                        found->members[0].conversionProposalId,
                        found->members[0].proposalIdentityCanonical,
                        found->members[0].proposalIdentityHash});
                forged.commit();
            }
            catch (const pqxx::check_violation&)
            {
                rejected = true;
            }
            assert(rejected);
        }
        {
            auto conflict = BuildRecommendationCampaignMaterializationEvidence(
                RecommendationCampaignMaterializationRequest{
                    42, "different-operator", "Materialize exact proposal set."},
                fixture.approval, fixture.plan, {fixture.proposal});
            bool rejected = false;
            try
            {
                pqxx::work tx{runtime};
                (void)PersistRecommendationCampaignMaterialization(tx, conflict);
            }
            catch (const std::runtime_error& error)
            {
                rejected = std::string{error.what()} ==
                    "recommendation_campaign_materialization_conflict";
            }
            assert(rejected);
        }
        {
            bool updateDenied = false;
            try
            {
                pqxx::work tx{runtime};
                tx.exec("UPDATE experiment_recommendation_campaign_materialization "
                        "SET materialized_by='forged';");
            }
            catch (const pqxx::sql_error& error)
            {
                updateDenied = PermissionDenied(error);
            }
            assert(updateDenied);
        }
        {
            pqxx::read_transaction verify{owner};
            verify.exec("SET LOCAL search_path TO " + verify.quote_name(schema));
            assert(verify.exec("SELECT count(*) FROM experiment;").one_row()[0].as<int>() == 1);
            assert(verify.exec("SELECT count(*) FROM experiment_recommendation_conversion_proposal;").one_row()[0].as<int>() == 1);
            assert(verify.exec("SELECT count(*) FROM experiment_recommendation_campaign_materialization;").one_row()[0].as<int>() == 1);
            assert(verify.exec("SELECT count(*) FROM experiment_recommendation_campaign_materialization_member;").one_row()[0].as<int>() == 1);
        }

        // Reset only the isolated fixture rows to verify rollback and concurrent
        // convergence without depending on sequence values.
        {
            pqxx::work reset{owner};
            reset.exec("SET LOCAL search_path TO " + reset.quote_name(schema));
            reset.exec("DELETE FROM experiment_recommendation_campaign_materialization_member;");
            reset.exec("DELETE FROM experiment_recommendation_campaign_materialization;");
            reset.exec("DELETE FROM experiment_recommendation_conversion_proposal;");
            reset.commit();
        }
        {
            pqxx::work rollback{runtime};
            const auto result = PersistRecommendationCampaignMaterialization(
                rollback, fixture.materialization);
            assert(result.outcome ==
                RecommendationCampaignMaterializationPersistOutcome::recorded);
        }
        {
            pqxx::read_transaction verify{owner};
            verify.exec("SET LOCAL search_path TO " + verify.quote_name(schema));
            assert(verify.exec("SELECT count(*) FROM experiment_recommendation_conversion_proposal;").one_row()[0].as<int>() == 0);
            assert(verify.exec("SELECT count(*) FROM experiment_recommendation_campaign_materialization;").one_row()[0].as<int>() == 0);
            assert(verify.exec("SELECT count(*) FROM experiment_recommendation_campaign_materialization_member;").one_row()[0].as<int>() == 0);
        }

        std::barrier start{2};
        std::atomic<int> recorded{0};
        std::atomic<int> replayed{0};
        std::exception_ptr threadError;
        std::mutex errorMutex;
        auto concurrentInsert = [&] {
            try
            {
                pqxx::connection threadConnection{runtimeString};
                pqxx::work tx{threadConnection};
                start.arrive_and_wait();
                const auto result = PersistRecommendationCampaignMaterialization(
                    tx, fixture.materialization);
                tx.commit();
                if (result.outcome ==
                    RecommendationCampaignMaterializationPersistOutcome::recorded)
                    ++recorded;
                else
                    ++replayed;
            }
            catch (...)
            {
                std::lock_guard lock{errorMutex};
                if (!threadError) threadError = std::current_exception();
            }
        };
        std::thread left{concurrentInsert};
        std::thread right{concurrentInsert};
        left.join();
        right.join();
        if (threadError) std::rethrow_exception(threadError);
        assert(recorded == 1);
        assert(replayed == 1);
        {
            pqxx::read_transaction verify{owner};
            verify.exec("SET LOCAL search_path TO " + verify.quote_name(schema));
            assert(verify.exec("SELECT count(*) FROM experiment;").one_row()[0].as<int>() == 1);
            assert(verify.exec("SELECT count(*) FROM experiment_recommendation_conversion_proposal;").one_row()[0].as<int>() == 1);
            assert(verify.exec("SELECT count(*) FROM experiment_recommendation_campaign_materialization;").one_row()[0].as<int>() == 1);
            assert(verify.exec("SELECT count(*) FROM experiment_recommendation_campaign_materialization_member;").one_row()[0].as<int>() == 1);
        }
    }
    catch (...)
    {
        pqxx::work cleanup{owner};
        cleanup.exec("DROP SCHEMA IF EXISTS " + cleanup.quote_name(schema) + " CASCADE");
        cleanup.commit();
        throw;
    }
    pqxx::work cleanup{owner};
    cleanup.exec("DROP SCHEMA " + cleanup.quote_name(schema) + " CASCADE");
    cleanup.commit();
    return 0;
}
