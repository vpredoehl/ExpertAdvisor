#include "ControlledExperimentFamilyRepository.hpp"

#include <map>
#include <set>
#include <stdexcept>

namespace EA::ControlledExperimentFamily {
namespace {

template <typename T>
std::optional<T> Optional(const pqxx::row &row, const char *name) {
  if (row[name].is_null())
    return std::nullopt;
  return row[name].as<T>();
}

Version MapVersion(const pqxx::row &row) {
  Version v;
  v.id = row["controlled_experiment_family_version_id"].as<long long>();
  v.familyId = row["controlled_experiment_family_id"].as<long long>();
  v.ordinal = row["version_ordinal"].as<int>();
  v.supersedesVersionId =
      Optional<long long>(row, "supersedes_family_version_id");
  v.specificationContractVersion =
      row["specification_contract_version"].as<int>();
  v.planningContractVersion = row["planning_contract_version"].as<int>();
  v.specificationIdentityCanonical =
      row["specification_identity_canonical"].as<std::string>();
  v.specificationIdentityHash =
      row["specification_identity_hash"].as<std::string>();
  v.specificationHashCollisionOrdinal =
      row["specification_hash_collision_ordinal"].as<int>();
  v.planIdentityCanonical = row["plan_identity_canonical"].as<std::string>();
  v.planIdentityHash = row["plan_identity_hash"].as<std::string>();
  v.planHashCollisionOrdinal = row["plan_hash_collision_ordinal"].as<int>();
  v.expectedCellCount = row["expected_cell_count"].as<int>();
  v.expectedMemberCount = row["expected_member_count"].as<int>();
  v.expectedArmCount = row["expected_arm_count"].as<int>();
  v.trainStart = row["train_start"].as<std::string>();
  v.trainEnd = row["train_end"].as<std::string>();
  v.inferStart = row["infer_start"].as<std::string>();
  v.inferEnd = row["infer_end"].as<std::string>();
  v.predictionThreshold = row["prediction_threshold"].as<std::string>();
  v.targetEpochs = row["target_epochs"].as<int>();
  v.modelInputWidth = row["model_input_width"].as<int>();
  v.modelInputSemanticLayoutVersion =
      row["model_input_semantic_layout_version"].as<int>();
  v.featureWarmupScope = row["feature_warmup_scope"].as<std::string>();
  v.donchian20Mode = row["donchian20_mode"].as<std::string>();
  v.donchianLookback = row["donchian_lookback"].as<int>();
  v.trainingContractVersion = row["training_contract_version"].as<int>();
  v.trainingContractCanonical =
      row["training_contract_canonical"].as<std::string>();
  v.trainingContractHash = row["training_contract_hash"].as<std::string>();
  v.targetGenerationContractCanonical =
      row["target_generation_contract_canonical"].as<std::string>();
  v.targetGenerationContractHash =
      row["target_generation_contract_hash"].as<std::string>();
  v.trainingObjectiveId = row["training_objective_id"].as<std::string>();
  v.trainingObjectiveCanonical =
      row["training_objective_canonical"].as<std::string>();
  v.trainingObjectiveHash = row["training_objective_hash"].as<std::string>();
  v.checkpointContractCanonical =
      row["checkpoint_contract_canonical"].as<std::string>();
  v.checkpointContractHash = row["checkpoint_contract_hash"].as<std::string>();
  v.continuationMode = row["continuation_mode"].as<std::string>();
  v.continuationPolicyCanonical =
      Optional<std::string>(row, "continuation_policy_canonical");
  v.continuationPolicyHash =
      Optional<std::string>(row, "continuation_policy_hash");
  v.economicCalendarSnapshotId =
      row["economic_calendar_snapshot_id"].as<long long>();
  v.economicCalendarSnapshotHash =
      row["economic_calendar_snapshot_hash"].as<std::string>();
  v.finalEvidenceContractCanonical =
      row["final_evidence_contract_canonical"].as<std::string>();
  v.finalEvidenceContractHash =
      row["final_evidence_contract_hash"].as<std::string>();
  v.initialSchedulerPriority =
      row["initial_scheduler_priority"].as<std::string>();
  v.createdBy = row["created_by"].as<std::string>();
  v.creationReason = row["creation_reason"].as<std::string>();
  v.renderedSpecificationSnapshot =
      row["rendered_specification_snapshot"].as<std::string>();
  return v;
}

void RequireGraphShape(const FamilyGraph &graph) {
  if (graph.familyKey.empty() || graph.familyCreatedBy.empty() ||
      graph.version.ordinal <= 0 || graph.arms.empty() || graph.cells.empty() ||
      graph.members.empty())
    throw std::invalid_argument("controlled_experiment_family_graph_invalid");
  if (graph.version.expectedArmCount != static_cast<int>(graph.arms.size()) ||
      graph.version.expectedCellCount != static_cast<int>(graph.cells.size()) ||
      graph.version.expectedMemberCount !=
          static_cast<int>(graph.members.size()))
    throw std::invalid_argument(
        "controlled_experiment_family_graph_count_mismatch");

  std::set<std::pair<int, int>> expectedPairs;
  for (const Cell &cell : graph.cells)
    for (const Arm &arm : graph.arms)
      expectedPairs.emplace(cell.ordinal, arm.ordinal);

  std::set<std::pair<int, int>> memberPairs;
  for (const Member &member : graph.members)
    memberPairs.emplace(member.cellOrdinal, member.armOrdinal);
  if (memberPairs != expectedPairs)
    throw std::invalid_argument(
        "controlled_experiment_family_member_pair_set_invalid");
}

} // namespace

bool SchemaExists(pqxx::transaction_base &transaction) {
  return transaction
      .exec("SELECT to_regclass('controlled_experiment_family_version') IS NOT "
            "NULL;")
      .one_row()[0]
      .as<bool>();
}

FamilyGraph CreateFamilyGraph(pqxx::transaction_base &t, const FamilyGraph &d) {
  RequireGraphShape(d);
  FamilyGraph result = d;
  result.familyId =
      t.exec("INSERT INTO controlled_experiment_family(family_key,created_by) "
             "VALUES($1,$2) RETURNING controlled_experiment_family_id;",
             pqxx::params{d.familyKey, d.familyCreatedBy})
          .one_row()[0]
          .as<long long>();
  const Version &v = d.version;
  result.version.id =
      t.exec(
           "INSERT INTO "
           "controlled_experiment_family_version(controlled_experiment_family_"
           "id,version_ordinal,supersedes_family_version_id,specification_"
           "contract_version,planning_contract_version,specification_identity_"
           "canonical,specification_identity_hash,specification_hash_collision_"
           "ordinal,plan_identity_canonical,plan_identity_hash,plan_hash_"
           "collision_ordinal,expected_cell_count,expected_member_count,"
           "expected_arm_count,train_start,train_end,infer_start,infer_end,"
           "prediction_threshold,target_epochs,resume_model_id,resume_expand_"
           "input_width,model_input_width,model_input_semantic_layout_version,"
           "feature_warmup_scope,donchian20_mode,donchian_lookback,training_"
           "contract_version,training_contract_canonical,training_contract_"
           "hash,target_generation_contract_canonical,target_generation_"
           "contract_hash,training_objective_id,training_objective_canonical,"
           "training_objective_hash,checkpoint_contract_canonical,checkpoint_"
           "contract_hash,continuation_mode,continuation_policy_canonical,"
           "continuation_policy_hash,economic_calendar_snapshot_id,economic_"
           "calendar_snapshot_hash,final_evidence_contract_canonical,final_"
           "evidence_contract_hash,initial_scheduler_priority,created_by,"
           "creation_reason,rendered_specification_snapshot) "
           "VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$"
           "18,$19::numeric,$20,NULL,false,$21,$22,$23,$24,$25,$26,$27,$28,$29,"
           "$30,$31,$32,$33,$34,$35,$36,$37,$38,$39,$40,$41,$42,$43,$44,$45,$"
           "46::jsonb) RETURNING controlled_experiment_family_version_id;",
           pqxx::params{result.familyId,
                        v.ordinal,
                        v.supersedesVersionId,
                        v.specificationContractVersion,
                        v.planningContractVersion,
                        v.specificationIdentityCanonical,
                        v.specificationIdentityHash,
                        v.specificationHashCollisionOrdinal,
                        v.planIdentityCanonical,
                        v.planIdentityHash,
                        v.planHashCollisionOrdinal,
                        v.expectedCellCount,
                        v.expectedMemberCount,
                        v.expectedArmCount,
                        v.trainStart,
                        v.trainEnd,
                        v.inferStart,
                        v.inferEnd,
                        v.predictionThreshold,
                        v.targetEpochs,
                        v.modelInputWidth,
                        v.modelInputSemanticLayoutVersion,
                        v.featureWarmupScope,
                        v.donchian20Mode,
                        v.donchianLookback,
                        v.trainingContractVersion,
                        v.trainingContractCanonical,
                        v.trainingContractHash,
                        v.targetGenerationContractCanonical,
                        v.targetGenerationContractHash,
                        v.trainingObjectiveId,
                        v.trainingObjectiveCanonical,
                        v.trainingObjectiveHash,
                        v.checkpointContractCanonical,
                        v.checkpointContractHash,
                        v.continuationMode,
                        v.continuationPolicyCanonical,
                        v.continuationPolicyHash,
                        v.economicCalendarSnapshotId,
                        v.economicCalendarSnapshotHash,
                        v.finalEvidenceContractCanonical,
                        v.finalEvidenceContractHash,
                        v.initialSchedulerPriority,
                        v.createdBy,
                        v.creationReason,
                        v.renderedSpecificationSnapshot})
          .one_row()[0]
          .as<long long>();
  result.version.familyId = result.familyId;
  std::map<int, long long> arms, cells;
  for (Arm &a : result.arms) {
    a.id = t.exec("INSERT INTO "
                  "controlled_experiment_family_arm(controlled_experiment_"
                  "family_version_id,arm_ordinal,arm_key,display_role,baseline_"
                  "arm_key,comparison_direction,declared_difference_canonical,"
                  "declared_difference_hash,feature_ablation_mask,feature_"
                  "ablation_mask_hash,arm_override_canonical,arm_override_hash)"
                  " VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12) RETURNING "
                  "controlled_experiment_family_arm_id;",
                  pqxx::params{
                      result.version.id, a.ordinal, a.key, a.displayRole,
                      a.baselineKey, a.comparisonDirection,
                      a.declaredDifferenceCanonical, a.declaredDifferenceHash,
                      a.featureAblationMask, a.featureAblationMaskHash,
                      a.overrideCanonical, a.overrideHash})
               .one_row()[0]
               .as<long long>();
    arms.emplace(a.ordinal, a.id);
  }
  for (Cell &c : result.cells) {
    c.id = t.exec("INSERT INTO "
                  "controlled_experiment_family_cell(controlled_experiment_"
                  "family_version_id,cell_ordinal,symbol,prediction_horizon,"
                  "fresh_initialization_seed,cell_identity_canonical,cell_"
                  "identity_hash) VALUES($1,$2,$3,$4,$5,$6,$7) RETURNING "
                  "controlled_experiment_family_cell_id;",
                  pqxx::params{result.version.id, c.ordinal, c.symbol,
                               c.predictionHorizon, c.freshInitializationSeed,
                               c.identityCanonical, c.identityHash})
               .one_row()[0]
               .as<long long>();
    cells.emplace(c.ordinal, c.id);
  }
  for (Member &m : result.members) {
    if (!cells.contains(m.cellOrdinal) || !arms.contains(m.armOrdinal))
      throw std::invalid_argument(
          "controlled_experiment_family_member_reference_invalid");
    m.id =
        t.exec("INSERT INTO "
               "controlled_experiment_family_member(controlled_experiment_"
               "family_version_id,controlled_experiment_family_cell_id,"
               "controlled_experiment_family_arm_id,member_ordinal,planned_"
               "experiment_identity_canonical,planned_experiment_identity_hash,"
               "planned_experiment_hash_collision_ordinal,experiment_id) "
               "VALUES($1,$2,$3,$4,$5,$6,$7,$8) RETURNING "
               "controlled_experiment_family_member_id;",
               pqxx::params{result.version.id, cells.at(m.cellOrdinal),
                            arms.at(m.armOrdinal), m.ordinal,
                            m.plannedIdentityCanonical, m.plannedIdentityHash,
                            m.hashCollisionOrdinal, m.experimentId})
            .one_row()[0]
            .as<long long>();
  }
  for (ExecutionRequirement &r : result.executionRequirements)
    r.id =
        t.exec(
             "INSERT INTO "
             "controlled_experiment_family_execution_requirement(controlled_"
             "experiment_family_version_id,lifecycle_phase,semantic_layout_"
             "version,model_input_width,semantic_worker_role,required_"
             "capabilities_canonical,required_capabilities_hash,source_commit,"
             "executable_sha256,runtime_identity,canonical_manifest_path,"
             "manifest_sha256) VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12) "
             "RETURNING controlled_experiment_family_execution_requirement_id;",
             pqxx::params{result.version.id, r.phase, r.semanticLayoutVersion,
                          r.modelInputWidth, r.semanticWorkerRole,
                          r.requiredCapabilitiesCanonical,
                          r.requiredCapabilitiesHash, r.sourceCommit,
                          r.executableSha256, r.runtimeIdentity,
                          r.canonicalManifestPath, r.manifestSha256})
            .one_row()[0]
            .as<long long>();
  return result;
}

std::optional<FamilyGraph> LoadFamilyGraph(pqxx::transaction_base &t,
                                           long long id) {
  const pqxx::result versions = t.exec(
      "SELECT v.*,f.family_key,f.created_by AS family_created_by FROM "
      "controlled_experiment_family_version v JOIN "
      "controlled_experiment_family f USING(controlled_experiment_family_id) "
      "WHERE v.controlled_experiment_family_version_id=$1;",
      pqxx::params{id});
  if (versions.empty())
    return std::nullopt;
  FamilyGraph g;
  g.familyId = versions[0]["controlled_experiment_family_id"].as<long long>();
  g.familyKey = versions[0]["family_key"].as<std::string>();
  g.familyCreatedBy = versions[0]["family_created_by"].as<std::string>();
  g.version = MapVersion(versions.one_row());
  for (const auto &r : t.exec(
           "SELECT * FROM controlled_experiment_family_arm WHERE "
           "controlled_experiment_family_version_id=$1 ORDER BY arm_ordinal;",
           pqxx::params{id}))
    g.arms.push_back({r["controlled_experiment_family_arm_id"].as<long long>(),
                      r["arm_ordinal"].as<int>(),
                      r["arm_key"].as<std::string>(),
                      r["display_role"].as<std::string>(),
                      Optional<std::string>(r, "baseline_arm_key"),
                      Optional<std::string>(r, "comparison_direction"),
                      r["declared_difference_canonical"].as<std::string>(),
                      r["declared_difference_hash"].as<std::string>(),
                      r["feature_ablation_mask"].as<std::string>(),
                      r["feature_ablation_mask_hash"].as<std::string>(),
                      r["arm_override_canonical"].as<std::string>(),
                      r["arm_override_hash"].as<std::string>()});
  for (const auto &r : t.exec(
           "SELECT * FROM controlled_experiment_family_cell WHERE "
           "controlled_experiment_family_version_id=$1 ORDER BY cell_ordinal;",
           pqxx::params{id}))
    g.cells.push_back(
        {r["controlled_experiment_family_cell_id"].as<long long>(),
         r["cell_ordinal"].as<int>(), r["symbol"].as<std::string>(),
         r["prediction_horizon"].as<int>(),
         r["fresh_initialization_seed"].as<unsigned int>(),
         r["cell_identity_canonical"].as<std::string>(),
         r["cell_identity_hash"].as<std::string>()});
  for (const auto &r :
       t.exec("SELECT m.*,c.cell_ordinal,a.arm_ordinal FROM "
              "controlled_experiment_family_member m JOIN "
              "controlled_experiment_family_cell c "
              "USING(controlled_experiment_family_cell_id) JOIN "
              "controlled_experiment_family_arm a "
              "USING(controlled_experiment_family_arm_id) WHERE "
              "m.controlled_experiment_family_version_id=$1 ORDER BY "
              "m.member_ordinal;",
              pqxx::params{id}))
    g.members.push_back(
        {r["controlled_experiment_family_member_id"].as<long long>(),
         r["member_ordinal"].as<int>(), r["cell_ordinal"].as<int>(),
         r["arm_ordinal"].as<int>(),
         r["planned_experiment_identity_canonical"].as<std::string>(),
         r["planned_experiment_identity_hash"].as<std::string>(),
         r["planned_experiment_hash_collision_ordinal"].as<int>(),
         Optional<long long>(r, "experiment_id")});
  for (const auto &r : t.exec(
           "SELECT * FROM controlled_experiment_family_execution_requirement "
           "WHERE controlled_experiment_family_version_id=$1 ORDER BY "
           "lifecycle_phase;",
           pqxx::params{id}))
    g.executionRequirements.push_back(
        {r["controlled_experiment_family_execution_requirement_id"]
             .as<long long>(),
         r["lifecycle_phase"].as<std::string>(),
         r["semantic_layout_version"].as<int>(),
         r["model_input_width"].as<int>(),
         r["semantic_worker_role"].as<std::string>(),
         r["required_capabilities_canonical"].as<std::string>(),
         r["required_capabilities_hash"].as<std::string>(),
         r["source_commit"].as<std::string>(),
         r["executable_sha256"].as<std::string>(),
         r["runtime_identity"].as<std::string>(),
         Optional<std::string>(r, "canonical_manifest_path"),
         Optional<std::string>(r, "manifest_sha256")});
  return g;
}
std::optional<FamilyGraph>
LoadFamilyGraphByPlanIdentity(pqxx::transaction_base &t,
                              const std::string &canonical) {
  const auto rows = t.exec(
      "SELECT controlled_experiment_family_version_id FROM "
      "controlled_experiment_family_version WHERE plan_identity_canonical=$1;",
      pqxx::params{canonical});
  if (rows.empty())
    return std::nullopt;
  return LoadFamilyGraph(t, rows.one_row()[0].as<long long>());
}
} // namespace EA::ControlledExperimentFamily
