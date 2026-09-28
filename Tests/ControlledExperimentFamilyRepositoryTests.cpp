#include "ControlledExperimentFamilyRepository.hpp"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace CEF = EA::ControlledExperimentFamily;

namespace {
std::string Env(const char *name) {
  const char *value = std::getenv(name);
  if (!value || !*value)
    throw std::runtime_error(name);
  return value;
}
constexpr const char *Hash = "fnv1a64:0123456789abcdef";
constexpr const char *Sha =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
constexpr const char *FibonacciMask =
    "fib_recent_price_scale_valid,fib_up_recent_union_count_log,"
    "fib_up_recent_h1_count_log,fib_up_recent_h2_count_log,"
    "fib_up_recent_h1_h2_both_count_log,fib_up_recent_h1_youngest_age_20,"
    "fib_up_recent_h2_youngest_age_20,fib_up_recent_median_1272_signed_atr,"
    "fib_up_recent_median_1618_signed_atr,"
    "fib_up_recent_median_pullback_0382_signed_atr,"
    "fib_up_recent_median_pullback_0500_signed_atr,"
    "fib_up_recent_median_pullback_0618_signed_atr,"
    "fib_down_recent_union_count_log,fib_down_recent_h1_count_log,"
    "fib_down_recent_h2_count_log,fib_down_recent_h1_h2_both_count_log,"
    "fib_down_recent_h1_youngest_age_20,fib_down_recent_h2_youngest_age_20,"
    "fib_down_recent_median_1272_signed_atr,"
    "fib_down_recent_median_1618_signed_atr,"
    "fib_down_recent_median_pullback_0382_signed_atr,"
    "fib_down_recent_median_pullback_0500_signed_atr,"
    "fib_down_recent_median_pullback_0618_signed_atr";
std::string HashFor(int value) {
  char text[32];
  std::snprintf(text, sizeof(text), "fnv1a64:%016x", value);
  return text;
}

CEF::FamilyGraph Fixture() {
  CEF::FamilyGraph g;
  g.familyKey = "controlled_fixture";
  g.familyCreatedBy = "fixture";
  auto &v = g.version;
  v.ordinal = 1;
  v.specificationContractVersion = 1;
  v.planningContractVersion = 1;
  v.specificationIdentityCanonical = "spec";
  v.specificationIdentityHash = Hash;
  v.planIdentityCanonical = "plan";
  v.planIdentityHash = "fnv1a64:0123456789abcdee";
  v.expectedCellCount = 48;
  v.expectedMemberCount = 96;
  v.expectedArmCount = 2;
  v.trainStart = "2010-01-01T00:00:00Z";
  v.trainEnd = "2022-01-01T00:00:00Z";
  v.inferStart = "2022-01-01T00:00:00Z";
  v.inferEnd = "2025-01-01T00:00:00Z";
  v.predictionThreshold = "0.0008";
  v.targetEpochs = 20;
  v.modelInputWidth = 103;
  v.modelInputSemanticLayoutVersion = 9;
  v.featureWarmupScope = "full_history_warmup";
  v.donchian20Mode = "enabled";
  v.donchianLookback = 20;
  v.trainingContractVersion = 1;
  v.trainingContractCanonical = "training";
  v.trainingContractHash = Hash;
  v.targetGenerationContractCanonical = "target";
  v.targetGenerationContractHash = Hash;
  v.trainingObjectiveId = "legacy_first_hit_weighted_ce_v1";
  v.trainingObjectiveCanonical = "objective";
  v.trainingObjectiveHash = Hash;
  v.checkpointContractCanonical = "checkpoint_disabled";
  v.checkpointContractHash = Hash;
  v.continuationMode = "prohibited";
  v.economicCalendarSnapshotId = 1;
  v.economicCalendarSnapshotHash = "fnv1a64:67610f94f5c8e7cc";
  v.finalEvidenceContractCanonical = "final";
  v.finalEvidenceContractHash = Hash;
  v.initialSchedulerPriority = "normal";
  v.createdBy = "fixture";
  v.creationReason = "disposable";
  v.renderedSpecificationSnapshot = "{}";
  g.arms = {{0,
             1,
             "control",
             "control",
             {},
             "control_minus_ablation",
             "none",
             Hash,
             "",
             Hash,
             "",
             Hash},
            {0, 2, "ablation", "ablation", std::string("control"),
             "control_minus_ablation", "feature_ablation_mask", Hash,
             FibonacciMask, "fnv1a64:a3f595680caadb2e", "", Hash}};
  const char *symbols[] = {"audcadrmp", "audusdrmp", "eurusdrmp",
                           "gbpusdrmp", "usdcadrmp", "usdjpyrmp"};
  int cell = 1, member = 1;
  for (const char *symbol : symbols)
    for (int horizon : {4, 6})
      for (unsigned seed : {43U, 47U, 53U, 59U}) {
        g.cells.push_back({0, cell, std::string(symbol), horizon, seed,
                           "cell" + std::to_string(cell), HashFor(cell)});
        for (int arm : {1, 2})
          g.members.push_back({0,
                               member++,
                               cell,
                               arm,
                               "member" + std::to_string(member),
                               HashFor(member),
                               0,
                               {}});
        ++cell;
      }
  g.executionRequirements = {
      {0, "train", 9, 103, "train", "train,train_feature_ablation_v1", Hash,
       "e964fa9e335e9ae63918187a7ffee7aa77f32b4b",
       "f56342895009e19cc69751259590cd945d64e5bfb5ffcd330f6aefc8d3fd26e9",
       "6c8d208a0aae281f3fbb1a2a38fe2c7d6deb92811b4a41d3615fac67defee34e",
       std::string("manifest"), Sha},
      {0,
       "infer",
       9,
       103,
       "infer",
       "infer",
       Hash,
       "e964fa9e335e9ae63918187a7ffee7aa77f32b4b",
       Sha,
       "6c8d208a0aae281f3fbb1a2a38fe2c7d6deb92811b4a41d3615fac67defee34e",
       {},
       {}}};
  return g;
}
} // namespace
int main() {
  pqxx::connection c{
      "host=" + Env("EA_CONTROLLED_FAMILY_DB_HOST") +
      " port=" + Env("EA_CONTROLLED_FAMILY_DB_PORT") +
      " user=pqxx dbname=" + Env("EA_CONTROLLED_FAMILY_DB_NAME")};
  {
    pqxx::work t{c};
    assert(CEF::SchemaExists(t));
    auto graph = CEF::CreateFamilyGraph(t, Fixture());
    assert(graph.cells.size() == 48 && graph.members.size() == 96);
    t.commit();
  }
  {
    pqxx::read_transaction t{c};
    auto graph = CEF::LoadFamilyGraphByPlanIdentity(t, "plan");
    assert(graph && graph->arms.size() == 2 && graph->cells.size() == 48 &&
           graph->members.size() == 96 &&
           graph->executionRequirements.size() == 2);
    assert(graph->cells.front().freshInitializationSeed == 43);
    assert(graph->cells[1].freshInitializationSeed == 47);
    assert(graph->cells[2].freshInitializationSeed == 53);
    assert(graph->cells[3].freshInitializationSeed == 59);
    assert(graph->arms[1].featureAblationMask == FibonacciMask);
    assert(graph->arms[1].featureAblationMaskHash ==
           "fnv1a64:a3f595680caadb2e");
    assert(graph->version.economicCalendarSnapshotHash ==
           "fnv1a64:67610f94f5c8e7cc");
    assert(graph->version.continuationMode == "prohibited");
    const auto train =
        std::find_if(graph->executionRequirements.begin(),
                     graph->executionRequirements.end(),
                     [](const CEF::ExecutionRequirement &requirement) {
                       return requirement.phase == "train";
                     });
    assert(train != graph->executionRequirements.end());
    assert(train->requiredCapabilitiesCanonical ==
           "train,train_feature_ablation_v1");
    assert(train->sourceCommit == "e964fa9e335e9ae63918187a7ffee7aa77f32b4b");
    assert(train->executableSha256 ==
           "f56342895009e19cc69751259590cd945d64e5bfb5ffcd330f6aefc8d3fd26e9");
    assert(train->runtimeIdentity ==
           "6c8d208a0aae281f3fbb1a2a38fe2c7d6deb92811b4a41d3615fac67defee34e");
  }
  {
    pqxx::work t{c};
    auto graph = Fixture();
    graph.familyKey = "incomplete";
    graph.version.expectedMemberCount = 95;
    graph.members.pop_back();
    bool rejected = false;
    try {
      (void)CEF::CreateFamilyGraph(t, graph);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    assert(rejected);
    t.abort();
  }
  {
    pqxx::work t{c};
    bool rejected = false;
    try {
      CEF::CreateFamilyGraph(t, Fixture());
    } catch (const pqxx::sql_error &) {
      rejected = true;
    }
    assert(rejected);
    t.abort();
  }
  {
    pqxx::work t{c};
    auto graph = Fixture();
    graph.familyKey = "rollback";
    graph.version.specificationIdentityCanonical = "rollback-spec";
    graph.version.specificationIdentityHash = "fnv1a64:0123456789abcddd";
    graph.version.planIdentityCanonical = "rollback-plan";
    graph.version.planIdentityHash = "fnv1a64:0123456789abcccc";
    CEF::CreateFamilyGraph(t, graph);
    t.abort();
  }
  {
    pqxx::read_transaction t{c};
    assert(!CEF::LoadFamilyGraphByPlanIdentity(t, "rollback-plan"));
  }
  {
    pqxx::work t{c};
    bool rejected = false;
    try {
      t.exec("UPDATE controlled_experiment_family_version SET "
             "creation_reason='mutation'");
    } catch (const pqxx::sql_error &) {
      rejected = true;
    }
    assert(rejected);
    t.abort();
  }
  {
    pqxx::work t{c};
    bool rejected = false;
    try {
      t.exec("INSERT INTO controlled_experiment_family_member "
             "(controlled_experiment_family_version_id, "
             "controlled_experiment_family_cell_id, "
             "controlled_experiment_family_arm_id, member_ordinal, "
             "planned_experiment_identity_canonical, "
             "planned_experiment_identity_hash) SELECT "
             "controlled_experiment_family_version_id, "
             "controlled_experiment_family_cell_id, "
             "controlled_experiment_family_arm_id, 1000, 'duplicate-member', "
             "'fnv1a64:0123456789abc999' FROM "
             "controlled_experiment_family_member LIMIT 1");
    } catch (const pqxx::sql_error &) {
      rejected = true;
    }
    assert(rejected);
    t.abort();
  }
}
