Reading prompt from stdin...
OpenAI Codex v0.145.0
--------
workdir: /Volumes/Developer SSD/ExpertAdvisor
model: gpt-5.6-sol
provider: openai
approval: never
sandbox: danger-full-access
reasoning effort: xhigh
reasoning summaries: none
session id: 019f89ea-3163-7a20-ae87-de125c1247e7
--------
user
Using xhigh reasoning, perform an independent Critical Engineering Evaluation of:

  CampaignOperations_Architecture_Output.txt

GOAL

Determine whether the Campaign Operations architectural recommendation should be accepted as the authoritative direction for the next architectural increment of the ExpertAdvisor/LSTM repository.

This is an independent engineering review, not an implementation exercise.

Do not implement code.
Do not modify files.
Do not create migrations.
Do not design CLI commands.
Do not begin the recommended implementation.
Do not redesign previously accepted Phases 4 through 6.
Treat completed and accepted authority chains as fixed unless the recommendation contains a concrete contradiction with them.

REVIEW BASIS

Inspect the current repository and its authoritative architecture documents, ADRs, domain contracts, repositories, services, migrations, tests, scheduler implementation, and experiment lifecycle implementation as necessary.

Evaluate the recommendation against the repository as it exists now rather than accepting its claims at face value.

In particular, independently determine whether:

- Phase 6 governance is complete;
- no meaningful Phase 6E implementation remains;
- Campaign Operations is the correct next bounded architectural concern;
- Campaign Operations should be separate from recommendation governance;
- Campaign Operations should not transfer recommendation or research-policy authority into the scheduler;
- the proposed subsystem responsibilities are coherent and complete;
- the proposed lifecycle and state model are appropriate;
- the proposed budgeting and reservation concepts have clear authority;
- the scheduler interaction contract preserves existing scheduler ownership;
- reconciliation and restart recovery have a single durable source of truth;
- transaction and concurrency boundaries are safe;
- deterministic replay and idempotence are adequately defined;
- cancellation and completion semantics are unambiguous;
- the privilege model follows least privilege;
- the proposed implementation sequence is safe and appropriately incremental.

REQUIRED REVIEW DIMENSIONS

Evaluate:

- architectural correctness;
- consistency with accepted ADRs and architecture volumes;
- compatibility with completed Phases 4 through 6;
- authority and ownership boundaries;
- domain separation;
- data ownership;
- lifecycle correctness;
- transaction boundaries;
- concurrency and lock ordering;
- idempotence and deterministic replay;
- recovery and reconciliation;
- scheduler integration;
- operator authority;
- auditability;
- security and least privilege;
- maintainability;
- extensibility;
- migration risk;
- testability;
- documentation consistency;
- hidden assumptions;
- missing decisions;
- implementation readiness.

FAIL-CLOSED REVIEW STANDARD

Do not approve the recommendation merely because its overall direction appears reasonable.

Identify any material ambiguity that could cause:

- duplicated authority;
- autonomous experiment creation;
- scheduler policy expansion;
- ambiguous campaign ownership;
- budget overcommitment;
- reservation leakage;
- duplicate operational requests;
- unsafe replay;
- irreconcilable partial state;
- cancellation races;
- inconsistent completion;
- unbounded recovery behavior;
- lock-order inversion;
- privilege escalation;
- misleading audit evidence.

Distinguish clearly between:

- blocking architectural defects;
- required revisions before implementation;
- implementation details that can safely be deferred;
- optional improvements.

OUTPUT

Produce:

1. Executive summary.
2. Repository-grounded architectural findings.
3. Strengths.
4. Blocking defects, if any.
5. Required revisions.
6. Non-blocking risks and deferred decisions.
7. Assessment of authority boundaries.
8. Assessment of scheduler interaction.
9. Assessment of lifecycle, transactions, concurrency, replay, and recovery.
10. Assessment of budgeting, reservations, and operational-request ownership.
11. Assessment of privileges, auditability, and testability.
12. Documentation and terminology findings.
13. Recommended disposition:
    - accept unchanged;
    - accept with required revisions;
    - revise and re-review;
    - reject.
14. Exact conditions that must be satisfied before implementation begins.
15. Overall readiness assessment.

Do not make repository changes.

codex
I’ll evaluate the recommendation against the repository’s actual contracts and accepted authority chains, using a fail-closed standard. I’ll keep the review strictly read-only and will distinguish architectural blockers from revisions, deferrable implementation details, and optional improvements.
  → Locate governing instructions, the recommendation, and authoritative architecture sources
  • Trace implemented Phase 4–6 authority chains, scheduler, lifecycle, persistence, and tests
  • Test each recommendation claim against repository evidence and identify contradictions or gaps
  • Produce the disposition, required preconditions, readiness assessment, and requested repository status
exec
/bin/bash -lc "pwd && rg --files -g 'AGENTS.md' -g 'CampaignOperations_Architecture_Output.txt' -g '*.md' -g '*.txt' -g '*.sql' -g '*.hpp' -g '*.h' -g '*.cpp' -g '*.mm' | sed -n '1,240p'" in /Volumes/Developer SSD/ExpertAdvisor
 succeeded in 0ms:
/Volumes/Developer SSD/ExpertAdvisor
Step5a_IndependentReview_Input.txt
Phase6A_FollowUpProposal_Implementation_Output.txt
Phase4C_Step2_Durable_Conversion_Proposal_Persistence_Prompt.txt
H6_accuracy_acceptance_summary.txt
Phase6D_Implementation_Output.txt
LSTM/ExperimentMetaAnalyzer.cpp
LSTM/main.cpp
LSTM/Tensor.cpp
LSTM/LSTM.cpp
Phase5Step5a_IndependentArchitecturalReview_Output.txt
phase4d_step3_campaign_approval_prompt.txt
H6_training_loss_tail_summary.txt
core_lr120_neutral1p0_threshold0008_horizon4_epoch60.txt
Phase5Step5c_ReviewCorrections_Input.txt
Phase6A_FollowUpProposal_Implementation_Input.txt
meta_analysis_report.md
infer_eval_H6_EURUSD_core120_epoch220.txt
Phase6A_IndependentReview_Output.txt
phase4c_step6_prompt_and_reasoning.txt
H8_finished_4_key_results.txt
Phase5Step5a_ContractCorrection_Input.txt
after_restart_0m.txt
H8_GBPUSD_core120_epoch240_rerun.txt
phase4d_step6_independent_review_prompt.txt
Phase4C_Step2_Focused_Persistence_Design_Review_Prompt.txt
infer_all_H6_USDCAD_epoch020_to_latest.txt
H8_AUDUSD_core120_epoch240_resume.txt
Phase6C_ReviewOfReview_Corrections_Output.txt
training_output/README.txt
Phase5Step5c_IndependentReReview_Input.txt
infer_eval_H6_GBPUSD_core120_epoch220.txt
Phase5Step5c_IndependentReview_Output.txt
Phase6_Next_Architecture_Recommendation_Prompt.txt
Phase6ReadinessReview_Output.txt
Phase4C_Step1_Manual_Conversion_Domain_Prompt.txt
infer_all_H6_EURUSD_epoch020_to_latest_H6_corrected.txt
CampaignOperations_Architecture_Recommendation_CEE_Prompt.txt
Phase6C_StagedCommitReview_Prompt_and_Reasoning.txt
H6_USDJPY_core120_epoch120_ckpt20.txt
Phase6E_Readiness_Authority_and_Implementation_Output.txt
H8_GBPUSD_core120_epoch240_resume.txt
Phase6C_OperatorReview_Implementation_Output.txt
before.txt
H8_GBPUSD_core120_epoch240.txt
H8_USDJPY_core120_epoch240_resume.txt
Phase5Step5b_Implementation_Input.txt
infer_eval_H6_AUDUSD_core120_epoch220.txt
meta_analysis_dashboard.md
Phase5Step4PreStageRegressionOutput.txt
Phase4C_Step4_Prompt_and_Reasoning.txt
Phase6D_Ratification_Correction_Prompt_and_Reasoning.txt
core_lr120_neutral1p0_threshold0008_horizon6_epoch60.txt
Phase6B_PersistencePreview_Implementation_Input.txt
Phase4D_Completion_Audit_Prompt_and_Reasoning.txt
Phase4B_Step2_Canonicalization_Comparison_Review_Prompt.txt
CampaignOperations_Architecture_Prompt.txt
after_restart_10m.txt
H6_finished_4_key_results.txt
Phase6C_IndependentReview_and_Corrections_Prompt_and_Reasoning.txt
Phase6C_FinalPostAmendVerification_Output.txt
H6_core120_head50_bias5.txt
Phase6C_FinalPostAmendVerification_Prompt_and_Reasoning.txt
Phase6D_Implementation_Prompt_and_Reasoning.txt
phase4d_step1_prompt_and_reasoning.txt
phase4d_step5_handoff_status_prompt_and_reasoning.txt
phase4c_step3_code_review_prompt_and_reasoning.txt
Phase5Step5b_IndependentReview_Output.txt
Database/README.md
Database/migrations/029_experiment_recommendation_source_rank_enforcement.sql
Database/migrations/010_schema_migrations_runtime_read.sql
Database/migrations/007_experiment_meta_analysis.sql
Database/migrations/009_experiment_run_metadata.sql
Database/migrations/039_experiment_recommendation_conversion_activation.sql
Database/migrations/005_experiment_scheduler.sql
Database/migrations/018_checkpoint_inference_result_scope.sql
Database/migrations/040_experiment_recommendation_campaign_approval.sql
Database/migrations/043_experiment_recommendation_campaign_follow_up_proposal_review.sql
Database/migrations/024_continuation_sequence_and_worker_attempt_hardening.sql
Database/migrations/035_experiment_recommendation_ranking.sql
Database/migrations/033_experiment_recommendation_review_permissions.sql
Database/migrations/014_checkpoint_stop_and_eval.sql
Database/migrations/042_experiment_recommendation_campaign_follow_up_proposal.sql
Database/migrations/023_continuation_policy_target_sequence.sql
Database/migrations/028_experiment_recommendation_scoring.sql
Database/migrations/004_inference_eval_result.sql
Database/migrations/038_experiment_recommendation_conversion_execution.sql
Database/migrations/030_experiment_recommendation_source_rank_transition_enforcement.sql
Database/migrations/027_experiment_recommendation_hardening.sql
Database/migrations/037_experiment_recommendation_conversion_review.sql
Database/migrations/008_experiment_paused_status.sql
Database/migrations/015_checkpoint_infer_phase1.sql
Database/migrations/019_checkpoint_policy_hardening.sql
Database/migrations/020_experiment_continuation_policy.sql
Database/migrations/022_continuation_policy_identity_bound.sql
Database/migrations/013_model_parent_model_id.sql
Database/migrations/021_continuation_policy_inheritance.sql
Database/migrations/031_experiment_recommendation_scan_provenance_enforcement.sql
Database/migrations/006_experiment_analysis.sql
Database/migrations/044_experiment_recommendation_campaign_follow_up_proposal_ratification.sql
Database/migrations/036_experiment_recommendation_conversion_proposal.sql
Database/migrations/026_experiment_recommendation_persistence.sql
Database/migrations/011_experiment_live_progress.sql
Database/migrations/034_experiment_recommendation_evaluation.sql
Database/migrations/032_experiment_recommendation_review.sql
Database/migrations/012_model_experiment_id.sql
Database/migrations/017_checkpoint_policy.sql
Database/migrations/016_checkpoint_analysis_scope.sql
Database/migrations/041_experiment_recommendation_campaign_materialization.sql
Phase6C_ReviewOfReview_Corrections_Prompt_and_Reasoning.txt
Phase5Step5a_ContractCorrection_Output.txt
Phase4B_Step2_Trigger_Review_Prompt.txt
phase4c_step5_prompt_and_reasoning.txt
CampaignOperationsArchitectureRecommendation_CEE_Prompt.txt
infer_all_H6_AUDCAD_epoch020_to_latest.txt
after_restart.txt
Phase6A_IndependentReview_Input.txt
Phase5Step5aOutput.txt
Phase5Step5c_Implementation_Input.txt
H6_detailed_acceptance_summary.txt
Phase5Step5c_IndependentReReview_Output.txt
Phase5Step5ArchitecturePlanningOutput.txt
H8_AUDCAD_core120_epoch240_resume.txt
Phase6C_PostCommitVerification_Output.txt
Step5a_IndependentReview_Output.txt
Phase6D_Focused_Independent_Implementation_Review_Output.txt
Phase4B_Step2_Independent_Review_Prompt.txt
h6_lr120_head25_bias2p5_epoch120.txt
Phase4B_Step1_Advisory_Evaluation_Foundation_Prompt.txt
H8_AUDUSD_core120_epoch120_ckpt20.txt
Phase6_Next_Architecture_Recommendation_Output.txt
docs/architecture/VOLUME_TEMPLATE.md
docs/architecture/Volume_III_Label_Generation.md
docs/architecture/Volume_X_Research_Automation.md
docs/architecture/Volume_XI_Scheduler.md
docs/architecture/Volume_VI_Inference_Evaluation.md
docs/architecture/Volume_XII_Database.md
docs/architecture/Volume_VII_Experiment_Lifecycle.md
docs/architecture/README.md
docs/architecture/Volume_II_Data_Pipeline.md
docs/architecture/Volume_I_Foundation.md
docs/architecture/Volume_IX_Trading_Profitability.md
docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md
docs/architecture/adr/ADR-0005-manual-recommendation-conversion.md
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md
docs/architecture/adr/README.md
docs/architecture/adr/ADR-0001-postgresql-source-of-truth.md
docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md
docs/architecture/adr/ADR_TEMPLATE.md
docs/architecture/adr/ADR-0002-deterministic-experiment-identity.md
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md
docs/architecture/Volume_V_Training_Engine.md
docs/architecture/Volume_VIII_Recommendation_Engine.md
docs/architecture/Volume_IV_Model_Architecture.md
phase4d_step4_prompt_and_reasoning.txt
Phase5Step5c_ReviewCorrections_Output.txt
H6_infer_summary_compare.txt
Phase4D_Step2_Prompt_and_Reasoning.txt
H6_core160_head25_bias2p5.txt
H6_GBPUSD_core120_epoch240_resume_from398.txt
Tests/ContinuationPolicyInheritanceTests.cpp
Headers/TargetLabel.hpp
Headers/LSTM.hpp
Tests/ExperimentRecommendationCampaignProposalReviewTests.cpp
Headers/SupportedSymbols.hpp
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationMigrationTests.sql
Headers/FxPriceSanity.hpp
Tests/ExperimentRecommendationCampaignApprovalTests.cpp
Headers/RunMetadata.hpp
Tests/ExperimentRecommendationRepositoryTests.cpp
Headers/db_cursor_iterator.hpp
Tests/ExperimentRecommendationCampaignMaterializationRepositoryTests.cpp
Headers/MatrixUtils.hpp
Tests/ExperimentRecommendationRankingTests.cpp
Headers/ExperimentMetaAnalyzer.hpp
Tests/ExperimentRecommendationCampaignOutcomeAssessmentTests.cpp
Headers/Params.hpp
Tests/ExperimentRecommendationCandidateGeneratorTests.cpp
Headers/PricePoint.hpp
Tests/ExperimentRecommendationCampaignExecutionTests.cpp
Headers/CanonicalSymbol.hpp
Tests/ExperimentRecommendationConversionProposalReviewTests.cpp
Headers/ExperimentScheduler.hpp
Tests/ExperimentRecommendationCampaignOutcomeAssessmentIntegrationTests.cpp
Tests/ExperimentRecommendationConversionWorkflowRepositoryTests.cpp
Tests/ExperimentRecommendationCampaignPlanningTests.cpp
Headers/Tensor.hpp
Tests/ExperimentRecommendationEvaluationTests.cpp
Headers/PgModelIO.hpp
Tests/ExperimentRecommendationConversionMigrationTests.sql
Headers/BuildConfig.hpp
Tests/ExperimentRecommendationCampaignActivationTests.cpp
phase4d_step3_review_prompt.txt
Tests/ExperimentRecommendationRankingRepositoryTests.cpp
investigate_child_exit_127_prompt_and_reasoning.txt
Tests/ExperimentRecommendationConversionExecutionRepositoryTests.cpp
phase4d_step5_review_prompt_and_reasoning.txt
Tests/ExperimentRecommendationReviewMigrationTests.sql
Phase6C_OperatorReview_Implementation_Input.txt
Tests/ExperimentRecommendationCampaignMaterializationMigrationTests.sql
Phase6D_Focused_Independent_Implementation_Review_Prompt_and_Reasoning.txt
Tests/ExperimentRecommendationConversionWorkflowTests.cpp
Phase6C_IndependentReview_and_Corrections_Output.txt
Tests/ExperimentRecommendationConversionExecutionMigrationTests.sql
Phase5_Prompt_and_Reasoning.txt
Tests/ExperimentRecommendationScoringMigrationTests.sql
Tests/ExperimentRecommendationCampaignFollowUpProposalRepositoryTests.cpp
Tests/ExperimentRecommendationConversionProposalReviewMigrationTests.sql
Tests/ExperimentRecommendationCampaignFollowUpProposalReviewMigrationTests.sql
Tests/ExperimentRecommendationReviewTests.cpp
Tests/ExperimentRecommendationConversionTests.cpp
Tests/ExperimentRecommendationEvaluationRepositoryTests.cpp
Tests/ExperimentRecommendationReviewRepositoryTests.cpp
Tests/ExperimentRecommendationConversionProposalReviewRepositoryTests.cpp
Tests/ExperimentRecommendationConversionRepositoryTests.cpp
Tests/ExperimentRecommendationServiceTests.cpp
Tests/ExperimentRecommendationScoringRepositoryTests.cpp
Tests/ExperimentRecommendationCampaignStatusTests.cpp
Tests/ExperimentRecommendationScoringTests.cpp
Tests/ExperimentRecommendationConversionActivationTests.cpp
Tests/ExperimentRecommendationCampaignFollowUpProposalMigrationTests.sql
Tests/ExperimentRecommendationRankingMigrationTests.sql
Tests/ExperimentRecommendationCampaignPlanningRepositoryTests.cpp
Tests/ExperimentRecommendationConversionActivationRepositoryTests.cpp
Tests/ContinuationPolicyPersistenceTests.cpp
Tests/ExperimentRecommendationCampaignReviewTests.cpp
Tests/ExperimentRecommendationCampaignActivationRepositoryTests.cpp
Tests/ExperimentRecommendationCampaignMaterializationTests.cpp
Tests/ExperimentRecommendationTests.cpp
Tests/ExperimentRecommendationCampaignHandoffRepositoryTests.cpp
Tests/ExperimentRecommendationCampaignFollowUpProposalTests.cpp
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationTests.cpp
Tests/ExperimentRecommendationEvaluationMigrationTests.sql
Tests/ExperimentRecommendationCampaignApprovalRepositoryTests.cpp
Tests/ExperimentRecommendationCampaignHandoffTests.cpp
Tests/SchedulerChildStatusTests.cpp
Tests/ExperimentRecommendationConversionActivationMigrationTests.sql
Tests/ExperimentRecommendationCampaignOutcomePolicyTests.cpp
Tests/ExperimentRecommendationCampaignProposalReviewRepositoryTests.cpp

exec
/bin/bash -lc "rg --files | rg '("'^|/)CampaignOperations_Architecture_Output'"\\.txt"'$|CampaignOperations|Campaign_Operations|AGENTS'"\\.md"'$'"' && wc -l CampaignOperations_Architecture_Output.txt docs/architecture/Volume_VII_Experiment_Lifecycle.md docs/architecture/Volume_VIII_Recommendation_Engine.md docs/architecture/Volume_X_Research_Automation.md docs/architecture/Volume_XI_Scheduler.md docs/architecture/Volume_XII_Database.md docs/architecture/adr/ADR-000{1,2,3,4,5,6,7,8,9}-*.md" in /Volumes/Developer SSD/ExpertAdvisor
 succeeded in 0ms:
AGENTS.md
     658 CampaignOperations_Architecture_Output.txt
     201 docs/architecture/Volume_VII_Experiment_Lifecycle.md
     800 docs/architecture/Volume_VIII_Recommendation_Engine.md
     189 docs/architecture/Volume_X_Research_Automation.md
     203 docs/architecture/Volume_XI_Scheduler.md
     281 docs/architecture/Volume_XII_Database.md
     112 docs/architecture/adr/ADR-0001-postgresql-source-of-truth.md
     109 docs/architecture/adr/ADR-0002-deterministic-experiment-identity.md
     108 docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md
     107 docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md
     107 docs/architecture/adr/ADR-0005-manual-recommendation-conversion.md
     138 docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md
     137 docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md
     149 docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md
     155 docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md
    3454 total

exec
/bin/bash -lc "sed -n '1,240p' AGENTS.md && sed -n '1,220p' CampaignOperations_Architecture_Output.txt && sed -n '221,440p' CampaignOperations_Architecture_Output.txt && sed -n '441,700p' CampaignOperations_Architecture_Output.txt" in /Volumes/Developer SSD/ExpertAdvisor
 succeeded in 0ms:
# ExpertAdvisor Development Instructions

## Project

- C++20 project built with Xcode 26.5 on macOS.
- Primary executable: LSTM_Release.
- Avoid Swift.
- Prefer existing C++ and Objective-C++ test patterns.
- PostgreSQL access uses libpqxx.
- Treat the database schema and persisted workflows as authoritative contracts.

## Build

From the repository root:

xcodebuild   -project ExpertAdvisor.xcodeproj   -scheme "LSTM Release"   -configuration Release   -derivedDataPath DerivedData/ExpertAdvisor   build

Do not run Clean unless explicitly requested.

## Tests

Run the smallest relevant tests first, then broader regression tests.

Do not launch tests or executables that may interfere with active scheduler,
training, inference, or analysis workers without checking first.

## Scheduler Safety

Before running LSTM_Release commands that could start work:

1. Inspect running LSTM_Release processes.
2. Check scheduler status.
3. Avoid interrupting active experiments unless explicitly requested.
4. Never alter production experiment rows merely to make tests pass.

## Architecture

Preserve the established layering:

CLI -> Service / Workflow -> Repository -> PostgreSQL

Do not bypass authoritative workflows with direct repository or SQL writes.

Preserve:

- deterministic identities
- transactional boundaries
- concurrency safety
- immutable persisted decisions and materializations
- explicit phase boundaries
- idempotent retry behavior

## Development Style

- Make the smallest change that satisfies the requested increment.
- Do not redesign adjacent phases.
- Do not implement future roadmap items.
- Prefer reuse of existing authoritative workflows.
- Add tests for concrete behavior and regressions.
- Update documentation when the public or operational contract changes.
- Treat compiler warnings as defects unless clearly justified.
- Compiler and tests are authoritative.
- Resolve compiler warnings, build errors, and failing tests before considering an implementation complete.

## Review Output

At completion, report:

1. Files changed.
2. Exact behavioral change.
3. Tests and build commands run.
4. Results.
5. Remaining risks or unverified assumptions.
6. `git status --short`
7. `git diff --stat`

## Current Development Status

Current branch:
- phase6

Completed major phases:
- Phase 1: Checkpoint inference
- Phase 2: Evaluation persistence
- Phase 3: Continuation framework
- Phase 4: Recommendation campaigns
- Phase 5: Campaign execution
- Phase 6A: ...
- Phase 6B: ...
- Phase 6C: ...

Always preserve compatibility with completed phases.
Do not redesign previously accepted architecture.

## Never

- Never run `xcodebuild clean` unless explicitly requested.
- Never modify production experiment data to simplify testing.
- Never bypass authoritative workflows.
- Never redesign completed phases during an incremental implementation.

Reading prompt from stdin...
OpenAI Codex v0.145.0
--------
workdir: /Volumes/Developer SSD/ExpertAdvisor
model: gpt-5.6-sol
provider: openai
approval: never
sandbox: danger-full-access
reasoning effort: xhigh
reasoning summaries: none
session id: 019f89e9-56ac-7783-bb06-4a7fc6cfef79
--------
user
Using High reasoning, perform a complete architectural design for the new Campaign Operations subsystem.

GOAL

Design the architecture that follows the completed Phase 6 governance work.

Do not implement code.

Do not modify existing production behavior.

Do not redesign completed Phases 4–6.

Treat all completed authority chains as fixed architecture.

BACKGROUND

The repository currently provides:

Planning
→ Review
→ Approval
→ Materialization
→ Execution
→ Follow-up Review
→ Governance

The governance chain is complete.

The next architectural concern is long-lived campaign operations.

Campaign Operations must remain a separate subsystem and must not transfer policy authority into the scheduler.

OBJECTIVES

Design the complete Campaign Operations architecture, including:

• subsystem responsibilities
• ownership boundaries
• lifecycle state machine
• operational entities
• database model
• repository interfaces
• service layer responsibilities
• scheduler interaction contract
• operator interaction
• reconciliation
• restart recovery
• audit requirements
• budgeting concepts
• reservation concepts
• execution requests
• completion semantics
• cancellation semantics
• deterministic replay
• failure recovery
• concurrency model
• transactional boundaries
• privilege model

Clearly identify:

• what belongs inside Campaign Operations
• what remains inside the scheduler
• what remains inside recommendation governance
• what belongs nowhere

REQUIREMENTS

Do not implement code.

Do not create migrations.

Do not design CLI commands.

Do not invent APIs unless required to explain architecture.

Do not redesign existing repository components.

Do not introduce automatic experiment creation.

Do not introduce autonomous decision making.

Do not change completed authority boundaries.

Produce:

1. Executive summary.
2. Architectural rationale.
3. Complete subsystem decomposition.
4. State machine.
5. Component interaction diagrams (textual).
6. Data ownership.
7. Transaction boundaries.
8. Failure recovery model.
9. Concurrency analysis.
10. Security / privilege analysis.
11. ADR recommendations.
12. Recommended implementation phases.
13. Risks.
14. Open architectural questions.

The deliverable should be an architecture document suitable for implementation planning.

codex
I’ll ground the design in the repository’s existing Phase 4–6 authority chains, schemas, and workflow terminology before drafting it. I’ll keep this strictly read-only: no code, migrations, CLI design, or repository edits.
  → Map existing Phase 4–6 architecture, entities, workflows, and scheduler boundaries
  • Derive Campaign Operations ownership, lifecycle, persistence, and interaction contracts
  • Validate design against authority, concurrency, recovery, and audit constraints
  • Deliver implementation-planning architecture document with ADRs, phases, risks, and open questions
exec
/bin/bash -lc "pwd && rg --files -g 'AGENTS.md' -g '*.md' -g '*.hpp' -g '*.h' -g '*.cpp' -g '*.mm' -g '*.sql' -g '*.xcodeproj/**' | sed -n '1,240p'" in /Volumes/Developer SSD/ExpertAdvisor
 succeeded in 0ms:
/Volumes/Developer SSD/ExpertAdvisor
LSTM/ExperimentMetaAnalyzer.cpp
LSTM/main.cpp
LSTM/Tensor.cpp
LSTM/LSTM.cpp
meta_analysis_report.md
MetaNN/MetaNN/facilities/_.h
MetaNN/MetaNN/facilities/var_type_dict.h
MetaNN/MetaNN/facilities/cont_metafuns/_.h
MetaNN/MetaNN/facilities/cont_metafuns/set.h
MetaNN/MetaNN/facilities/cont_metafuns/helpers.h
MetaNN/MetaNN/facilities/cont_metafuns/value_sequential.h
MetaNN/MetaNN/facilities/cont_metafuns/sequential.h
MetaNN/MetaNN/facilities/cont_metafuns/multi_map.h
MetaNN/MetaNN/facilities/cont_metafuns/map.h
MetaNN/MetaNN/facilities/type_id.h
MetaNN/MetaNN/facilities/null_param.h
MetaNN/MetaNN/facilities/traits.h
MetaNN/MetaNN/data/_.h
MetaNN/MetaNN/data/bias_vector.h
MetaNN/readme.md
MetaNN/MetaNN/data/facilities/category_tags.h
MetaNN/readme_eng.md
MetaNN/MetaNN/data/facilities/continuous_memory.h
MetaNN/MetaNN/data/facilities/allocators.h
MetaNN/MetaNN/data/facilities/continuous_memory_metal.mm
MetaNN/MetaNN/data/facilities/device_tags.h
MetaNN/MetaNN/data/facilities/shape.h
MetaNN/MetaNN/data/facilities/lower_access.h
MetaNN/MetaNN/data/facilities/traits.h
MetaNN/MetaNN/data/tensor.h
MetaNN/MetaNN/data/batch/batch_sequence.h
MetaNN/MetaNN/data/dynamic.h
MetaNN/MetaNN/data/trivial_tensor.h
MetaNN/MetaNN/data/scalable_tensor.h
MetaNN/MetaNN/data/zero_tensor.h
MetaNN/MetaNN/model/_.h
MetaNN/MetaNN/model/facilities/weight_buffer.h
MetaNN/Tests/OtherTests/facilities/test_map.cpp
MetaNN/Tests/OtherTests/facilities/test_set.cpp
MetaNN/Tests/OtherTests/facilities/test_multi_map.cpp
MetaNN/Tests/OtherTests/facilities/test_sequential.cpp
MetaNN/Tests/OtherTests/main.cpp
MetaNN/Tests/OtherTests/model/_.h
MetaNN/MetaNN/model/param_initializer/_.h
MetaNN/MetaNN/model/param_initializer/uniform_filler.h
MetaNN/MetaNN/model/param_initializer/var_scale_filler.h
MetaNN/MetaNN/model/param_initializer/constant_filler.h
MetaNN/Tests/OtherTests/model/param_initializer/test_uniform_filler.cpp
MetaNN/Tests/OtherTests/model/param_initializer/test_param_initializer.cpp
MetaNN/Tests/OtherTests/model/param_initializer/test_var_scale_filler.cpp
MetaNN/Tests/OtherTests/model/param_initializer/test_gaussian_filler.cpp
MetaNN/Tests/OtherTests/model/param_initializer/test_constant_filler.cpp
MetaNN/Tests/OtherTests/model/param_initializer/param_initializer.cpp
MetaNN/MetaNN/model/param_initializer/facilities/policies.h
MetaNN/MetaNN/model/param_initializer/facilities/fill_with_spec_dist.h
MetaNN/MetaNN/model/param_initializer/param_initializer.h
MetaNN/MetaNN/model/param_initializer/gaussian_filler.h
MetaNN/MetaNN/data_copy/metal_copy.mm
MetaNN/MetaNN/data_copy/metal_copy.h
MetaNN/MetaNN/data_copy/data_copy.h
AGENTS.md
MetaNN/MetaNN/model/weight_cont/_.h
MetaNN/MetaNN/model/weight_cont/grad_collector.h
MetaNN/MetaNN/model/weight_cont/load_buffer.h
MetaNN/Tests/OtherTests/policies/_.h
MetaNN/Tests/OtherTests/policies/test_policy_operations.cpp
MetaNN/MetaNN/metal/metal_add.h
MetaNN/MetaNN/metal/metal_add.mm
MetaNN/MetaNN/metal/metal_matmul.h
MetaNN/MetaNN/metal/metal_matmul.mm
MetaNN/MetalSwiftTest/main.mm
MetaNN/MetaNN/evaluate/eval_handle.h
MetaNN/MetaNN/evaluate/eval_plan.h
MetaNN/MetaNN/evaluate/eval_buffer.h
MetaNN/MetaNN/evaluate/eval_dispatcher.h
MetaNN/MetaNN/evaluate/eval_item.h
MetaNN/MetaNN/evaluate/eval_group.h
MetaNN/MetaNN/meta_nn.h
MetaNN/Tests/DataOpTest/main.cpp
MetaNN/Tests/data_gen.h
MetaNN/Tests/calculate_tags.h
MetaNN/MetaNN/policies/_.h
MetaNN/MetaNN/policies/policy_container.h
MetaNN/MetaNN/policies/policy_macro_begin.h
MetaNN/MetaNN/policies/policy_macro_end.h
MetaNN/MetaNN/policies/policy_operations.h
meta_analysis_dashboard.md
MetaNN/MetaNN/layers/_.h
MetaNN/Tests/DataOpTest/data/_.h
MetaNN/Tests/DataOpTest/data/test_tensor.cpp
MetaNN/Tests/DataOpTest/data/facilities/_.h
MetaNN/Tests/DataOpTest/data/facilities/test_shape.cpp
MetaNN/Tests/DataOpTest/data/test_bias_vector.cpp
MetaNN/Tests/DataOpTest/data/test_zero_tensor.cpp
MetaNN/Tests/DataOpTest/data/test_scalar.cpp
MetaNN/Tests/DataOpTest/data/test_scalable_tensor.cpp
MetaNN/Tests/DataOpTest/data/test_trivial_tensor.cpp
MetaNN/Tests/DataOpTest/data/test_dynamic.cpp
MetaNN/Tests/LayerTest/main.cpp
MetaNN/MetaNN/layers/facilities/_.h
MetaNN/MetaNN/layers/facilities/layer_in_map.h
MetaNN/MetaNN/layers/facilities/interface_fun.h
MetaNN/MetaNN/layers/facilities/policies.h
MetaNN/MetaNN/layers/facilities/make_layer.h
MetaNN/MetaNN/layers/facilities/traits.h
Database/README.md
MetaNN/Tests/DataOpTest/operation/_.h
docs/architecture/VOLUME_TEMPLATE.md
docs/architecture/Volume_III_Label_Generation.md
docs/architecture/Volume_X_Research_Automation.md
docs/architecture/Volume_XI_Scheduler.md
docs/architecture/Volume_VI_Inference_Evaluation.md
docs/architecture/Volume_XII_Database.md
docs/architecture/Volume_VII_Experiment_Lifecycle.md
docs/architecture/README.md
docs/architecture/Volume_II_Data_Pipeline.md
docs/architecture/Volume_I_Foundation.md
docs/architecture/Volume_IX_Trading_Profitability.md
ExpertAdvisor.xcodeproj/xcshareddata/xcschemes/LSTM Release.xcscheme
ExpertAdvisor.xcodeproj/xcshareddata/xcschemes/LSTM Debug.xcscheme
MetaNN/Tests/LayerTest/composite/test_weight_layer.cpp
MetaNN/Tests/LayerTest/composite/test_linear_layer.cpp
MetaNN/Tests/LayerTest/composite/_.h
MetaNN/Tests/LayerTest/composite/test_compose_kenrel.cpp
MetaNN/Tests/LayerTest/composite/test_bias_layer.cpp
MetaNN/Tests/LayerTest/composite/test_single_layer_perceptron.cpp
MetaNN/Tests/DataOpTest/operation/tensor/_.h
MetaNN/Tests/DataOpTest/operation/tensor/test_dot.cpp
MetaNN/Tests/DataOpTest/operation/tensor/test_slice.cpp
MetaNN/Tests/DataOpTest/operation/tensor/test_reshape.cpp
MetaNN/Tests/DataOpTest/operation/tensor/test_tile.cpp
MetaNN/Tests/DataOpTest/operation/tensor/test_permute.cpp
MetaNN/MetaNN/layers/composite/_.h
MetaNN/MetaNN/layers/composite/bias_layer.h
MetaNN/MetaNN/layers/composite/compose_kernel.h
MetaNN/MetaNN/layers/composite/linear_layer.h
MetaNN/MetaNN/layers/composite/weight_layer.h
MetaNN/MetaNN/layers/composite/single_layer_perceptron.h
docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md
docs/architecture/adr/ADR-0005-manual-recommendation-conversion.md
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md
docs/architecture/adr/README.md
docs/architecture/adr/ADR-0001-postgresql-source-of-truth.md
docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md
docs/architecture/adr/ADR_TEMPLATE.md
docs/architecture/adr/ADR-0002-deterministic-experiment-identity.md
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md
docs/architecture/Volume_V_Training_Engine.md
docs/architecture/Volume_VIII_Recommendation_Engine.md
docs/architecture/Volume_IV_Model_Architecture.md
Database/migrations/029_experiment_recommendation_source_rank_enforcement.sql
Database/migrations/010_schema_migrations_runtime_read.sql
Database/migrations/007_experiment_meta_analysis.sql
Database/migrations/009_experiment_run_metadata.sql
Database/migrations/039_experiment_recommendation_conversion_activation.sql
Database/migrations/005_experiment_scheduler.sql
Database/migrations/018_checkpoint_inference_result_scope.sql
Database/migrations/040_experiment_recommendation_campaign_approval.sql
Database/migrations/043_experiment_recommendation_campaign_follow_up_proposal_review.sql
Database/migrations/024_continuation_sequence_and_worker_attempt_hardening.sql
Database/migrations/035_experiment_recommendation_ranking.sql
Database/migrations/033_experiment_recommendation_review_permissions.sql
Database/migrations/014_checkpoint_stop_and_eval.sql
Database/migrations/042_experiment_recommendation_campaign_follow_up_proposal.sql
Database/migrations/023_continuation_policy_target_sequence.sql
Database/migrations/028_experiment_recommendation_scoring.sql
Database/migrations/004_inference_eval_result.sql
Database/migrations/038_experiment_recommendation_conversion_execution.sql
Database/migrations/030_experiment_recommendation_source_rank_transition_enforcement.sql
Database/migrations/027_experiment_recommendation_hardening.sql
Database/migrations/037_experiment_recommendation_conversion_review.sql
Database/migrations/008_experiment_paused_status.sql
Database/migrations/015_checkpoint_infer_phase1.sql
Database/migrations/019_checkpoint_policy_hardening.sql
Database/migrations/020_experiment_continuation_policy.sql
Database/migrations/022_continuation_policy_identity_bound.sql
Database/migrations/013_model_parent_model_id.sql
Database/migrations/021_continuation_policy_inheritance.sql
Database/migrations/031_experiment_recommendation_scan_provenance_enforcement.sql
Database/migrations/006_experiment_analysis.sql
Database/migrations/044_experiment_recommendation_campaign_follow_up_proposal_ratification.sql
Database/migrations/036_experiment_recommendation_conversion_proposal.sql
Database/migrations/026_experiment_recommendation_persistence.sql
Database/migrations/011_experiment_live_progress.sql
Database/migrations/034_experiment_recommendation_evaluation.sql
Database/migrations/032_experiment_recommendation_review.sql
Database/migrations/012_model_experiment_id.sql
Database/migrations/017_checkpoint_policy.sql
Database/migrations/016_checkpoint_analysis_scope.sql
Database/migrations/041_experiment_recommendation_campaign_materialization.sql
Tests/ExperimentRecommendationCampaignStatusRepositoryTests.cpp
Tests/ExperimentRecommendationCampaignLaunchRepositoryTests.cpp
Tests/ExperimentRecommendationCampaignFollowUpProposalReviewTests.cpp
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp
Tests/ExperimentRecommendationCampaignApprovalMigrationTests.sql
Tests/ExperimentRecommendationCampaignLaunchTests.cpp
Tests/ExperimentRecommendationCampaignFollowUpProposalReviewRepositoryTests.cpp
Tests/ExperimentRecommendationCampaignExecutionRepositoryTests.cpp
Tests/ExperimentRecommendationPersistenceTests.cpp
Tests/ExperimentRecommendationCampaignOutcomePolicyTests.cpp
Tests/ExperimentRecommendationCampaignProposalReviewRepositoryTests.cpp
Tests/ExperimentRecommendationConversionActivationMigrationTests.sql
Tests/SchedulerChildStatusTests.cpp
Tests/ExperimentRecommendationCampaignHandoffTests.cpp
MetaNN/Tests/LayerTest/recurrent/_.h
MetaNN/Tests/LayerTest/recurrent/test_recurrent_layer.cpp
MetaNN/Tests/LayerTest/recurrent/test_gru.cpp
MetaNN/Tests/DataOpTest/operation/math/_.h
MetaNN/Tests/DataOpTest/operation/math/test_acos.cpp
MetaNN/Tests/DataOpTest/operation/math/test_negative.cpp
MetaNN/Tests/DataOpTest/operation/math/test_asin.cpp
MetaNN/Tests/DataOpTest/operation/math/test_add.cpp
MetaNN/Tests/DataOpTest/operation/math/test_divide.cpp
MetaNN/Tests/DataOpTest/operation/math/test_tanh.cpp
MetaNN/Tests/DataOpTest/operation/math/test_abs.cpp
MetaNN/Tests/DataOpTest/operation/math/test_acosh.cpp
MetaNN/Tests/DataOpTest/operation/math/test_substract.cpp
MetaNN/Tests/DataOpTest/operation/math/test_cos.cpp
MetaNN/Tests/DataOpTest/operation/math/test_sin.cpp
MetaNN/Tests/DataOpTest/operation/math/test_exp.cpp
MetaNN/Tests/DataOpTest/operation/math/test_tan.cpp
MetaNN/Tests/DataOpTest/operation/math/test_sigmoid.cpp
MetaNN/Tests/DataOpTest/operation/math/test_asinh.cpp
MetaNN/Tests/DataOpTest/operation/math/test_multiply.cpp
MetaNN/Tests/DataOpTest/operation/math/test_reduce_sum.cpp
MetaNN/Tests/DataOpTest/operation/math/test_sign.cpp
MetaNN/Tests/DataOpTest/operation/math/test_square.cpp
MetaNN/Tests/DataOpTest/operation/others/_.h
MetaNN/Tests/DataOpTest/operation/others/test_interpolate.cpp
MetaNN/MetaNN/layers/math/_.h
Sources/ContinuationPolicy.cpp
Sources/ExperimentRecommendationCampaignExecution.cpp
Sources/ExperimentRecommendationCampaignStatus.hpp
Sources/ExperimentRecommendationCampaignHandoffRepository.cpp
Sources/ExperimentRecommendationConversionExecutionRepository.hpp
Sources/ExperimentRecommendationCampaignFollowUpProposalReview.hpp
Sources/ExperimentRecommendationCampaignMaterializationService.cpp
Sources/ExperimentRecommendationConversionProposalReviewRepository.cpp

exec
/bin/bash -lc "rg --files Sources Tests Database docs | rg '(Campaign|campaign|Recommendation|recommendation|Scheduler|scheduler|FollowUp|follow_up|Govern|govern|Continuation|continuation)' | sort" in /Volumes/Developer SSD/ExpertAdvisor
 succeeded in 0ms:
Database/migrations/005_experiment_scheduler.sql
Database/migrations/020_experiment_continuation_policy.sql
Database/migrations/021_continuation_policy_inheritance.sql
Database/migrations/022_continuation_policy_identity_bound.sql
Database/migrations/023_continuation_policy_target_sequence.sql
Database/migrations/024_continuation_sequence_and_worker_attempt_hardening.sql
Database/migrations/026_experiment_recommendation_persistence.sql
Database/migrations/027_experiment_recommendation_hardening.sql
Database/migrations/028_experiment_recommendation_scoring.sql
Database/migrations/029_experiment_recommendation_source_rank_enforcement.sql
Database/migrations/030_experiment_recommendation_source_rank_transition_enforcement.sql
Database/migrations/031_experiment_recommendation_scan_provenance_enforcement.sql
Database/migrations/032_experiment_recommendation_review.sql
Database/migrations/033_experiment_recommendation_review_permissions.sql
Database/migrations/034_experiment_recommendation_evaluation.sql
Database/migrations/035_experiment_recommendation_ranking.sql
Database/migrations/036_experiment_recommendation_conversion_proposal.sql
Database/migrations/037_experiment_recommendation_conversion_review.sql
Database/migrations/038_experiment_recommendation_conversion_execution.sql
Database/migrations/039_experiment_recommendation_conversion_activation.sql
Database/migrations/040_experiment_recommendation_campaign_approval.sql
Database/migrations/041_experiment_recommendation_campaign_materialization.sql
Database/migrations/042_experiment_recommendation_campaign_follow_up_proposal.sql
Database/migrations/043_experiment_recommendation_campaign_follow_up_proposal_review.sql
Database/migrations/044_experiment_recommendation_campaign_follow_up_proposal_ratification.sql
Sources/ContinuationPolicy.cpp
Sources/ContinuationPolicy.hpp
Sources/ContinuationPolicyInheritance.cpp
Sources/ContinuationPolicyInheritance.hpp
Sources/ContinuationPolicyPersistence.cpp
Sources/ContinuationPolicyPersistence.hpp
Sources/ExperimentRecommendation.cpp
Sources/ExperimentRecommendation.hpp
Sources/ExperimentRecommendationCampaignActivation.cpp
Sources/ExperimentRecommendationCampaignActivation.hpp
Sources/ExperimentRecommendationCampaignActivationRepository.cpp
Sources/ExperimentRecommendationCampaignActivationRepository.hpp
Sources/ExperimentRecommendationCampaignActivationService.cpp
Sources/ExperimentRecommendationCampaignActivationService.hpp
Sources/ExperimentRecommendationCampaignApproval.cpp
Sources/ExperimentRecommendationCampaignApproval.hpp
Sources/ExperimentRecommendationCampaignApprovalRepository.cpp
Sources/ExperimentRecommendationCampaignApprovalRepository.hpp
Sources/ExperimentRecommendationCampaignApprovalService.cpp
Sources/ExperimentRecommendationCampaignApprovalService.hpp
Sources/ExperimentRecommendationCampaignExecution.cpp
Sources/ExperimentRecommendationCampaignExecution.hpp
Sources/ExperimentRecommendationCampaignExecutionRepository.cpp
Sources/ExperimentRecommendationCampaignExecutionRepository.hpp
Sources/ExperimentRecommendationCampaignExecutionService.cpp
Sources/ExperimentRecommendationCampaignExecutionService.hpp
Sources/ExperimentRecommendationCampaignFollowUpProposal.cpp
Sources/ExperimentRecommendationCampaignFollowUpProposal.hpp
Sources/ExperimentRecommendationCampaignFollowUpProposalPreview.cpp
Sources/ExperimentRecommendationCampaignFollowUpProposalPreview.hpp
Sources/ExperimentRecommendationCampaignFollowUpProposalRatification.cpp
Sources/ExperimentRecommendationCampaignFollowUpProposalRatification.hpp
Sources/ExperimentRecommendationCampaignFollowUpProposalRatificationRepository.cpp
Sources/ExperimentRecommendationCampaignFollowUpProposalRatificationRepository.hpp
Sources/ExperimentRecommendationCampaignFollowUpProposalRatificationService.cpp
Sources/ExperimentRecommendationCampaignFollowUpProposalRatificationService.hpp
Sources/ExperimentRecommendationCampaignFollowUpProposalRepository.cpp
Sources/ExperimentRecommendationCampaignFollowUpProposalRepository.hpp
Sources/ExperimentRecommendationCampaignFollowUpProposalReview.cpp
Sources/ExperimentRecommendationCampaignFollowUpProposalReview.hpp
Sources/ExperimentRecommendationCampaignFollowUpProposalReviewPresentation.cpp
Sources/ExperimentRecommendationCampaignFollowUpProposalReviewPresentation.hpp
Sources/ExperimentRecommendationCampaignFollowUpProposalReviewRepository.cpp
Sources/ExperimentRecommendationCampaignFollowUpProposalReviewRepository.hpp
Sources/ExperimentRecommendationCampaignHandoff.cpp
Sources/ExperimentRecommendationCampaignHandoff.hpp
Sources/ExperimentRecommendationCampaignHandoffRepository.cpp
Sources/ExperimentRecommendationCampaignHandoffRepository.hpp
Sources/ExperimentRecommendationCampaignHandoffService.cpp
Sources/ExperimentRecommendationCampaignHandoffService.hpp
Sources/ExperimentRecommendationCampaignLaunch.cpp
Sources/ExperimentRecommendationCampaignLaunch.hpp
Sources/ExperimentRecommendationCampaignLaunchRepository.cpp
Sources/ExperimentRecommendationCampaignLaunchRepository.hpp
Sources/ExperimentRecommendationCampaignLaunchService.cpp
Sources/ExperimentRecommendationCampaignLaunchService.hpp
Sources/ExperimentRecommendationCampaignMaterialization.cpp
Sources/ExperimentRecommendationCampaignMaterialization.hpp
Sources/ExperimentRecommendationCampaignMaterializationRepository.cpp
Sources/ExperimentRecommendationCampaignMaterializationRepository.hpp
Sources/ExperimentRecommendationCampaignMaterializationService.cpp
Sources/ExperimentRecommendationCampaignMaterializationService.hpp
Sources/ExperimentRecommendationCampaignOutcomeAssessment.cpp
Sources/ExperimentRecommendationCampaignOutcomeAssessment.hpp
Sources/ExperimentRecommendationCampaignOutcomeAssessmentRepository.cpp
Sources/ExperimentRecommendationCampaignOutcomeAssessmentRepository.hpp
Sources/ExperimentRecommendationCampaignOutcomeAssessmentService.cpp
Sources/ExperimentRecommendationCampaignOutcomeAssessmentService.hpp
Sources/ExperimentRecommendationCampaignOutcomePolicy.cpp
Sources/ExperimentRecommendationCampaignOutcomePolicy.hpp
Sources/ExperimentRecommendationCampaignPlanning.cpp
Sources/ExperimentRecommendationCampaignPlanning.hpp
Sources/ExperimentRecommendationCampaignPlanningRepository.cpp
Sources/ExperimentRecommendationCampaignPlanningRepository.hpp
Sources/ExperimentRecommendationCampaignPlanningService.cpp
Sources/ExperimentRecommendationCampaignPlanningService.hpp
Sources/ExperimentRecommendationCampaignProposalReview.cpp
Sources/ExperimentRecommendationCampaignProposalReview.hpp
Sources/ExperimentRecommendationCampaignProposalReviewRepository.cpp
Sources/ExperimentRecommendationCampaignProposalReviewRepository.hpp
Sources/ExperimentRecommendationCampaignProposalReviewService.cpp
Sources/ExperimentRecommendationCampaignProposalReviewService.hpp
Sources/ExperimentRecommendationCampaignReview.cpp
Sources/ExperimentRecommendationCampaignReview.hpp
Sources/ExperimentRecommendationCampaignReviewService.cpp
Sources/ExperimentRecommendationCampaignReviewService.hpp
Sources/ExperimentRecommendationCampaignStatus.cpp
Sources/ExperimentRecommendationCampaignStatus.hpp
Sources/ExperimentRecommendationCampaignStatusRepository.cpp
Sources/ExperimentRecommendationCampaignStatusRepository.hpp
Sources/ExperimentRecommendationCampaignStatusService.cpp
Sources/ExperimentRecommendationCampaignStatusService.hpp
Sources/ExperimentRecommendationCandidateGenerator.cpp
Sources/ExperimentRecommendationCandidateGenerator.hpp
Sources/ExperimentRecommendationConversion.cpp
Sources/ExperimentRecommendationConversion.hpp
Sources/ExperimentRecommendationConversionActivation.cpp
Sources/ExperimentRecommendationConversionActivation.hpp
Sources/ExperimentRecommendationConversionActivationRepository.cpp
Sources/ExperimentRecommendationConversionActivationRepository.hpp
Sources/ExperimentRecommendationConversionActivationService.cpp
Sources/ExperimentRecommendationConversionActivationService.hpp
Sources/ExperimentRecommendationConversionExecutionRepository.cpp
Sources/ExperimentRecommendationConversionExecutionRepository.hpp
Sources/ExperimentRecommendationConversionExecutionService.cpp
Sources/ExperimentRecommendationConversionExecutionService.hpp
Sources/ExperimentRecommendationConversionProposalReview.cpp
Sources/ExperimentRecommendationConversionProposalReview.hpp
Sources/ExperimentRecommendationConversionProposalReviewRepository.cpp
Sources/ExperimentRecommendationConversionProposalReviewRepository.hpp
Sources/ExperimentRecommendationConversionProposalReviewService.cpp
Sources/ExperimentRecommendationConversionProposalReviewService.hpp
Sources/ExperimentRecommendationConversionRepository.cpp
Sources/ExperimentRecommendationConversionRepository.hpp
Sources/ExperimentRecommendationConversionWorkflow.cpp
Sources/ExperimentRecommendationConversionWorkflow.hpp
Sources/ExperimentRecommendationConversionWorkflowRepository.cpp
Sources/ExperimentRecommendationConversionWorkflowRepository.hpp
Sources/ExperimentRecommendationConversionWorkflowService.cpp
Sources/ExperimentRecommendationConversionWorkflowService.hpp
Sources/ExperimentRecommendationEvaluation.cpp
Sources/ExperimentRecommendationEvaluation.hpp
Sources/ExperimentRecommendationEvaluationRepository.cpp
Sources/ExperimentRecommendationEvaluationRepository.hpp
Sources/ExperimentRecommendationEvaluationService.cpp
Sources/ExperimentRecommendationEvaluationService.hpp
Sources/ExperimentRecommendationRanking.cpp
Sources/ExperimentRecommendationRanking.hpp
Sources/ExperimentRecommendationRankingRepository.cpp
Sources/ExperimentRecommendationRankingRepository.hpp
Sources/ExperimentRecommendationRankingService.cpp
Sources/ExperimentRecommendationRankingService.hpp
Sources/ExperimentRecommendationRepository.cpp
Sources/ExperimentRecommendationRepository.hpp
Sources/ExperimentRecommendationReview.cpp
Sources/ExperimentRecommendationReview.hpp
Sources/ExperimentRecommendationScoring.cpp
Sources/ExperimentRecommendationScoring.hpp
Sources/ExperimentRecommendationScoringService.cpp
Sources/ExperimentRecommendationService.cpp
Sources/ExperimentRecommendationService.hpp
Sources/ExperimentScheduler.cpp
Sources/SchedulerChildStatus.hpp
Tests/ContinuationPolicyInheritanceTests.cpp
Tests/ContinuationPolicyPersistenceTests.cpp
Tests/ExperimentRecommendationCampaignActivationRepositoryTests.cpp
Tests/ExperimentRecommendationCampaignActivationTests.cpp
Tests/ExperimentRecommendationCampaignApprovalMigrationTests.sql
Tests/ExperimentRecommendationCampaignApprovalRepositoryTests.cpp
Tests/ExperimentRecommendationCampaignApprovalTests.cpp
Tests/ExperimentRecommendationCampaignExecutionRepositoryTests.cpp
Tests/ExperimentRecommendationCampaignExecutionTests.cpp
Tests/ExperimentRecommendationCampaignFollowUpProposalMigrationTests.sql
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationMigrationTests.sql
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationTests.cpp
Tests/ExperimentRecommendationCampaignFollowUpProposalRepositoryTests.cpp
Tests/ExperimentRecommendationCampaignFollowUpProposalReviewMigrationTests.sql
Tests/ExperimentRecommendationCampaignFollowUpProposalReviewRepositoryTests.cpp
Tests/ExperimentRecommendationCampaignFollowUpProposalReviewTests.cpp
Tests/ExperimentRecommendationCampaignFollowUpProposalTests.cpp
Tests/ExperimentRecommendationCampaignHandoffRepositoryTests.cpp
Tests/ExperimentRecommendationCampaignHandoffTests.cpp
Tests/ExperimentRecommendationCampaignLaunchRepositoryTests.cpp
Tests/ExperimentRecommendationCampaignLaunchTests.cpp
Tests/ExperimentRecommendationCampaignMaterializationMigrationTests.sql
Tests/ExperimentRecommendationCampaignMaterializationRepositoryTests.cpp
Tests/ExperimentRecommendationCampaignMaterializationTests.cpp
Tests/ExperimentRecommendationCampaignOutcomeAssessmentIntegrationTests.cpp
Tests/ExperimentRecommendationCampaignOutcomeAssessmentTests.cpp
Tests/ExperimentRecommendationCampaignOutcomePolicyTests.cpp
Tests/ExperimentRecommendationCampaignPlanningRepositoryTests.cpp
Tests/ExperimentRecommendationCampaignPlanningTests.cpp
Tests/ExperimentRecommendationCampaignProposalReviewRepositoryTests.cpp
Tests/ExperimentRecommendationCampaignProposalReviewTests.cpp
Tests/ExperimentRecommendationCampaignReviewTests.cpp
Tests/ExperimentRecommendationCampaignStatusRepositoryTests.cpp
Tests/ExperimentRecommendationCampaignStatusTests.cpp
Tests/ExperimentRecommendationCandidateGeneratorTests.cpp
Tests/ExperimentRecommendationConversionActivationMigrationTests.sql
Tests/ExperimentRecommendationConversionActivationRepositoryTests.cpp
Tests/ExperimentRecommendationConversionActivationTests.cpp
Tests/ExperimentRecommendationConversionExecutionMigrationTests.sql
Tests/ExperimentRecommendationConversionExecutionRepositoryTests.cpp
Tests/ExperimentRecommendationConversionMigrationTests.sql
Tests/ExperimentRecommendationConversionProposalReviewMigrationTests.sql
Tests/ExperimentRecommendationConversionProposalReviewRepositoryTests.cpp
Tests/ExperimentRecommendationConversionProposalReviewTests.cpp
Tests/ExperimentRecommendationConversionRepositoryTests.cpp
Tests/ExperimentRecommendationConversionTests.cpp
Tests/ExperimentRecommendationConversionWorkflowRepositoryTests.cpp
Tests/ExperimentRecommendationConversionWorkflowTests.cpp
Tests/ExperimentRecommendationEvaluationMigrationTests.sql
Tests/ExperimentRecommendationEvaluationRepositoryTests.cpp
Tests/ExperimentRecommendationEvaluationTests.cpp
Tests/ExperimentRecommendationPersistenceTests.cpp
Tests/ExperimentRecommendationRankingMigrationTests.sql
Tests/ExperimentRecommendationRankingRepositoryTests.cpp
Tests/ExperimentRecommendationRankingTests.cpp
Tests/ExperimentRecommendationRepositoryTests.cpp
Tests/ExperimentRecommendationReviewMigrationTests.sql
Tests/ExperimentRecommendationReviewRepositoryTests.cpp
Tests/ExperimentRecommendationReviewTests.cpp
Tests/ExperimentRecommendationScoringMigrationTests.sql
Tests/ExperimentRecommendationScoringRepositoryTests.cpp
Tests/ExperimentRecommendationScoringTests.cpp
Tests/ExperimentRecommendationServiceTests.cpp
Tests/ExperimentRecommendationTests.cpp
Tests/SchedulerChildStatusTests.cpp
docs/Phase4AExperimentRecommendationFoundation.rst
docs/Phase4AExperimentRecommendationPersistence.rst
docs/Phase4AExperimentRecommendationReview.rst
docs/Phase4AExperimentRecommendationScoring.rst
docs/Phase4BExperimentRecommendationEvaluation.rst
docs/Phase4BExperimentRecommendationRanking.rst
docs/Phase4CExperimentRecommendationConversion.rst
docs/Phase4CExperimentRecommendationConversionActivation.rst
docs/Phase4CExperimentRecommendationConversionExecution.rst
docs/Phase4CExperimentRecommendationConversionPersistence.rst
docs/Phase4CExperimentRecommendationConversionProposalReview.rst
docs/Phase4CExperimentRecommendationConversionWorkflow.rst
docs/Phase4DExperimentRecommendationCampaignApproval.rst
docs/Phase4DExperimentRecommendationCampaignHandoff.rst
docs/Phase4DExperimentRecommendationCampaignMaterialization.rst
docs/Phase4DExperimentRecommendationCampaignPlanning.rst
docs/Phase4DExperimentRecommendationCampaignProposalReview.rst
docs/Phase4DExperimentRecommendationCampaignReview.rst
docs/Phase5ExperimentRecommendationCampaignActivation.rst
docs/Phase5ExperimentRecommendationCampaignExecution.rst
docs/Phase5ExperimentRecommendationCampaignLaunch.rst
docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst
docs/Phase5ExperimentRecommendationCampaignStatus.rst
docs/Phase6ARecommendationCampaignFollowUpProposal.rst
docs/Phase6BRecommendationCampaignFollowUpProposalPersistence.rst
docs/Phase6CRecommendationCampaignFollowUpProposalReview.rst
docs/Phase6DRecommendationCampaignFollowUpProposalRatification.rst
docs/architecture/Volume_VIII_Recommendation_Engine.md
docs/architecture/Volume_XI_Scheduler.md
docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md
docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md
docs/architecture/adr/ADR-0005-manual-recommendation-conversion.md
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md

time: command terminated abnormally
       23.56 real         0.66 user         1.14 sys
           147062784  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
               61270  page reclaims
               12329  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                 398  messages sent
                5162  messages received
                  26  signals received
                2199  voluntary context switches
               40308  involuntary context switches
          7893238807  instructions retired
          4854616510  cycles elapsed
           119080760  peak memory footprint
time: signal: Invalid argument

exec
/bin/bash -lc "rg -n '"'^#|''^##|''^###|''^Executive|''^Architect|''^State|''^Data|''^Transaction|''^Failure|''^Concurrency|''^Security|''^ADR|''^Recommended|''^Risks|''^Open|''^final|''^codex$'"' CampaignOperations_Architecture_Output.txt | tail -n 160 && sed -n '500,700p' CampaignOperations_Architecture_Output.txt" in /Volumes/Developer SSD/ExpertAdvisor
 succeeded in 0ms:
2:OpenAI Codex v0.145.0
117:codex
232:Database/README.md
279:Database/migrations/029_experiment_recommendation_source_rank_enforcement.sql
280:Database/migrations/010_schema_migrations_runtime_read.sql
281:Database/migrations/007_experiment_meta_analysis.sql
282:Database/migrations/009_experiment_run_metadata.sql
283:Database/migrations/039_experiment_recommendation_conversion_activation.sql
284:Database/migrations/005_experiment_scheduler.sql
285:Database/migrations/018_checkpoint_inference_result_scope.sql
286:Database/migrations/040_experiment_recommendation_campaign_approval.sql
287:Database/migrations/043_experiment_recommendation_campaign_follow_up_proposal_review.sql
288:Database/migrations/024_continuation_sequence_and_worker_attempt_hardening.sql
289:Database/migrations/035_experiment_recommendation_ranking.sql
290:Database/migrations/033_experiment_recommendation_review_permissions.sql
291:Database/migrations/014_checkpoint_stop_and_eval.sql
292:Database/migrations/042_experiment_recommendation_campaign_follow_up_proposal.sql
293:Database/migrations/023_continuation_policy_target_sequence.sql
294:Database/migrations/028_experiment_recommendation_scoring.sql
295:Database/migrations/004_inference_eval_result.sql
296:Database/migrations/038_experiment_recommendation_conversion_execution.sql
297:Database/migrations/030_experiment_recommendation_source_rank_transition_enforcement.sql
298:Database/migrations/027_experiment_recommendation_hardening.sql
299:Database/migrations/037_experiment_recommendation_conversion_review.sql
300:Database/migrations/008_experiment_paused_status.sql
301:Database/migrations/015_checkpoint_infer_phase1.sql
302:Database/migrations/019_checkpoint_policy_hardening.sql
303:Database/migrations/020_experiment_continuation_policy.sql
304:Database/migrations/022_continuation_policy_identity_bound.sql
305:Database/migrations/013_model_parent_model_id.sql
306:Database/migrations/021_continuation_policy_inheritance.sql
307:Database/migrations/031_experiment_recommendation_scan_provenance_enforcement.sql
308:Database/migrations/006_experiment_analysis.sql
309:Database/migrations/044_experiment_recommendation_campaign_follow_up_proposal_ratification.sql
310:Database/migrations/036_experiment_recommendation_conversion_proposal.sql
311:Database/migrations/026_experiment_recommendation_persistence.sql
312:Database/migrations/011_experiment_live_progress.sql
313:Database/migrations/034_experiment_recommendation_evaluation.sql
314:Database/migrations/032_experiment_recommendation_review.sql
315:Database/migrations/012_model_experiment_id.sql
316:Database/migrations/017_checkpoint_policy.sql
317:Database/migrations/016_checkpoint_analysis_scope.sql
318:Database/migrations/041_experiment_recommendation_campaign_materialization.sql
371:Database/migrations/005_experiment_scheduler.sql
372:Database/migrations/020_experiment_continuation_policy.sql
373:Database/migrations/021_continuation_policy_inheritance.sql
374:Database/migrations/022_continuation_policy_identity_bound.sql
375:Database/migrations/023_continuation_policy_target_sequence.sql
376:Database/migrations/024_continuation_sequence_and_worker_attempt_hardening.sql
377:Database/migrations/026_experiment_recommendation_persistence.sql
378:Database/migrations/027_experiment_recommendation_hardening.sql
379:Database/migrations/028_experiment_recommendation_scoring.sql
380:Database/migrations/029_experiment_recommendation_source_rank_enforcement.sql
381:Database/migrations/030_experiment_recommendation_source_rank_transition_enforcement.sql
382:Database/migrations/031_experiment_recommendation_scan_provenance_enforcement.sql
383:Database/migrations/032_experiment_recommendation_review.sql
384:Database/migrations/033_experiment_recommendation_review_permissions.sql
385:Database/migrations/034_experiment_recommendation_evaluation.sql
386:Database/migrations/035_experiment_recommendation_ranking.sql
387:Database/migrations/036_experiment_recommendation_conversion_proposal.sql
388:Database/migrations/037_experiment_recommendation_conversion_review.sql
389:Database/migrations/038_experiment_recommendation_conversion_execution.sql
390:Database/migrations/039_experiment_recommendation_conversion_activation.sql
391:Database/migrations/040_experiment_recommendation_campaign_approval.sql
392:Database/migrations/041_experiment_recommendation_campaign_materialization.sql
393:Database/migrations/042_experiment_recommendation_campaign_follow_up_proposal.sql
394:Database/migrations/043_experiment_recommendation_campaign_follow_up_proposal_review.sql
395:Database/migrations/044_experiment_recommendation_campaign_follow_up_proposal_ratification.sql
Sources/ExperimentRecommendationConversionExecutionService.cpp
Sources/ExperimentRecommendationConversionExecutionService.hpp
Sources/ExperimentRecommendationConversionProposalReview.cpp
Sources/ExperimentRecommendationConversionProposalReview.hpp
Sources/ExperimentRecommendationConversionProposalReviewRepository.cpp
Sources/ExperimentRecommendationConversionProposalReviewRepository.hpp
Sources/ExperimentRecommendationConversionProposalReviewService.cpp
Sources/ExperimentRecommendationConversionProposalReviewService.hpp
Sources/ExperimentRecommendationConversionRepository.cpp
Sources/ExperimentRecommendationConversionRepository.hpp
Sources/ExperimentRecommendationConversionWorkflow.cpp
Sources/ExperimentRecommendationConversionWorkflow.hpp
Sources/ExperimentRecommendationConversionWorkflowRepository.cpp
Sources/ExperimentRecommendationConversionWorkflowRepository.hpp
Sources/ExperimentRecommendationConversionWorkflowService.cpp
Sources/ExperimentRecommendationConversionWorkflowService.hpp
Sources/ExperimentRecommendationEvaluation.cpp
Sources/ExperimentRecommendationEvaluation.hpp
Sources/ExperimentRecommendationEvaluationRepository.cpp
Sources/ExperimentRecommendationEvaluationRepository.hpp
Sources/ExperimentRecommendationEvaluationService.cpp
Sources/ExperimentRecommendationEvaluationService.hpp
Sources/ExperimentRecommendationRanking.cpp
Sources/ExperimentRecommendationRanking.hpp
Sources/ExperimentRecommendationRankingRepository.cpp
Sources/ExperimentRecommendationRankingRepository.hpp
Sources/ExperimentRecommendationRankingService.cpp
Sources/ExperimentRecommendationRankingService.hpp
Sources/ExperimentRecommendationRepository.cpp
Sources/ExperimentRecommendationRepository.hpp
Sources/ExperimentRecommendationReview.cpp
Sources/ExperimentRecommendationReview.hpp
Sources/ExperimentRecommendationScoring.cpp
Sources/ExperimentRecommendationScoring.hpp
Sources/ExperimentRecommendationScoringService.cpp
Sources/ExperimentRecommendationService.cpp
Sources/ExperimentRecommendationService.hpp
Sources/ExperimentScheduler.cpp
Sources/SchedulerChildStatus.hpp
Tests/ContinuationPolicyInheritanceTests.cpp
Tests/ContinuationPolicyPersistenceTests.cpp
Tests/ExperimentRecommendationCampaignActivationRepositoryTests.cpp
Tests/ExperimentRecommendationCampaignActivationTests.cpp
Tests/ExperimentRecommendationCampaignApprovalMigrationTests.sql
Tests/ExperimentRecommendationCampaignApprovalRepositoryTests.cpp
Tests/ExperimentRecommendationCampaignApprovalTests.cpp
Tests/ExperimentRecommendationCampaignExecutionRepositoryTests.cpp
Tests/ExperimentRecommendationCampaignExecutionTests.cpp
Tests/ExperimentRecommendationCampaignFollowUpProposalMigrationTests.sql
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationMigrationTests.sql
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationTests.cpp
Tests/ExperimentRecommendationCampaignFollowUpProposalRepositoryTests.cpp
Tests/ExperimentRecommendationCampaignFollowUpProposalReviewMigrationTests.sql
Tests/ExperimentRecommendationCampaignFollowUpProposalReviewRepositoryTests.cpp
Tests/ExperimentRecommendationCampaignFollowUpProposalReviewTests.cpp
Tests/ExperimentRecommendationCampaignFollowUpProposalTests.cpp
Tests/ExperimentRecommendationCampaignHandoffRepositoryTests.cpp
Tests/ExperimentRecommendationCampaignHandoffTests.cpp
Tests/ExperimentRecommendationCampaignLaunchRepositoryTests.cpp
Tests/ExperimentRecommendationCampaignLaunchTests.cpp
Tests/ExperimentRecommendationCampaignMaterializationMigrationTests.sql
Tests/ExperimentRecommendationCampaignMaterializationRepositoryTests.cpp
Tests/ExperimentRecommendationCampaignMaterializationTests.cpp
Tests/ExperimentRecommendationCampaignOutcomeAssessmentIntegrationTests.cpp
Tests/ExperimentRecommendationCampaignOutcomeAssessmentTests.cpp
Tests/ExperimentRecommendationCampaignOutcomePolicyTests.cpp
Tests/ExperimentRecommendationCampaignPlanningRepositoryTests.cpp
Tests/ExperimentRecommendationCampaignPlanningTests.cpp
Tests/ExperimentRecommendationCampaignProposalReviewRepositoryTests.cpp
Tests/ExperimentRecommendationCampaignProposalReviewTests.cpp
Tests/ExperimentRecommendationCampaignReviewTests.cpp
Tests/ExperimentRecommendationCampaignStatusRepositoryTests.cpp
Tests/ExperimentRecommendationCampaignStatusTests.cpp
Tests/ExperimentRecommendationCandidateGeneratorTests.cpp
Tests/ExperimentRecommendationConversionActivationMigrationTests.sql
Tests/ExperimentRecommendationConversionActivationRepositoryTests.cpp
Tests/ExperimentRecommendationConversionActivationTests.cpp
Tests/ExperimentRecommendationConversionExecutionMigrationTests.sql
Tests/ExperimentRecommendationConversionExecutionRepositoryTests.cpp
Tests/ExperimentRecommendationConversionMigrationTests.sql
Tests/ExperimentRecommendationConversionProposalReviewMigrationTests.sql
Tests/ExperimentRecommendationConversionProposalReviewRepositoryTests.cpp
Tests/ExperimentRecommendationConversionProposalReviewTests.cpp
Tests/ExperimentRecommendationConversionRepositoryTests.cpp
Tests/ExperimentRecommendationConversionTests.cpp
Tests/ExperimentRecommendationConversionWorkflowRepositoryTests.cpp
Tests/ExperimentRecommendationConversionWorkflowTests.cpp
Tests/ExperimentRecommendationEvaluationMigrationTests.sql
Tests/ExperimentRecommendationEvaluationRepositoryTests.cpp
Tests/ExperimentRecommendationEvaluationTests.cpp
Tests/ExperimentRecommendationPersistenceTests.cpp
Tests/ExperimentRecommendationRankingMigrationTests.sql
Tests/ExperimentRecommendationRankingRepositoryTests.cpp
Tests/ExperimentRecommendationRankingTests.cpp
Tests/ExperimentRecommendationRepositoryTests.cpp
Tests/ExperimentRecommendationReviewMigrationTests.sql
Tests/ExperimentRecommendationReviewRepositoryTests.cpp
Tests/ExperimentRecommendationReviewTests.cpp
Tests/ExperimentRecommendationScoringMigrationTests.sql
Tests/ExperimentRecommendationScoringRepositoryTests.cpp
Tests/ExperimentRecommendationScoringTests.cpp
Tests/ExperimentRecommendationServiceTests.cpp
Tests/ExperimentRecommendationTests.cpp
Tests/SchedulerChildStatusTests.cpp
docs/Phase4AExperimentRecommendationFoundation.rst
docs/Phase4AExperimentRecommendationPersistence.rst
docs/Phase4AExperimentRecommendationReview.rst
docs/Phase4AExperimentRecommendationScoring.rst
docs/Phase4BExperimentRecommendationEvaluation.rst
docs/Phase4BExperimentRecommendationRanking.rst
docs/Phase4CExperimentRecommendationConversion.rst
docs/Phase4CExperimentRecommendationConversionActivation.rst
docs/Phase4CExperimentRecommendationConversionExecution.rst
docs/Phase4CExperimentRecommendationConversionPersistence.rst
docs/Phase4CExperimentRecommendationConversionProposalReview.rst
docs/Phase4CExperimentRecommendationConversionWorkflow.rst
docs/Phase4DExperimentRecommendationCampaignApproval.rst
docs/Phase4DExperimentRecommendationCampaignHandoff.rst
docs/Phase4DExperimentRecommendationCampaignMaterialization.rst
docs/Phase4DExperimentRecommendationCampaignPlanning.rst
docs/Phase4DExperimentRecommendationCampaignProposalReview.rst
docs/Phase4DExperimentRecommendationCampaignReview.rst
docs/Phase5ExperimentRecommendationCampaignActivation.rst
docs/Phase5ExperimentRecommendationCampaignExecution.rst
docs/Phase5ExperimentRecommendationCampaignLaunch.rst
docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst
docs/Phase5ExperimentRecommendationCampaignStatus.rst
docs/Phase6ARecommendationCampaignFollowUpProposal.rst
docs/Phase6BRecommendationCampaignFollowUpProposalPersistence.rst
docs/Phase6CRecommendationCampaignFollowUpProposalReview.rst
docs/Phase6DRecommendationCampaignFollowUpProposalRatification.rst
docs/architecture/Volume_VIII_Recommendation_Engine.md
docs/architecture/Volume_XI_Scheduler.md
docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md
docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md
docs/architecture/adr/ADR-0005-manual-recommendation-conversion.md
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md

time: command terminated abnormally
       23.56 real         0.66 user         1.14 sys
           147062784  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
               61270  page reclaims
               12329  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                 398  messages sent
                5162  messages received
                  26  signals received
                2199  voluntary context switches
               40308  involuntary context switches
          7893238807  instructions retired
          4854616510  cycles elapsed
           119080760  peak memory footprint
time: signal: Invalid argument

exec
/bin/bash -lc "tail -n 80 CampaignOperations_Architecture_Output.txt && stat -f 'size=%z bytes modified=%Sm' -t '%Y-%m-%d %H:%M:%S %z' CampaignOperations_Architecture_Output.txt && git status --short -- CampaignOperations_Architecture_Output.txt" in /Volumes/Developer SSD/ExpertAdvisor
 succeeded in 0ms:
Tests/ExperimentRecommendationConversionExecutionRepositoryTests.cpp
Tests/ExperimentRecommendationConversionMigrationTests.sql
Tests/ExperimentRecommendationConversionProposalReviewMigrationTests.sql
Tests/ExperimentRecommendationConversionProposalReviewRepositoryTests.cpp
Tests/ExperimentRecommendationConversionProposalReviewTests.cpp
Tests/ExperimentRecommendationConversionRepositoryTests.cpp
Tests/ExperimentRecommendationConversionTests.cpp
Tests/ExperimentRecommendationConversionWorkflowRepositoryTests.cpp
Tests/ExperimentRecommendationConversionWorkflowTests.cpp
Tests/ExperimentRecommendationEvaluationMigrationTests.sql
Tests/ExperimentRecommendationEvaluationRepositoryTests.cpp
Tests/ExperimentRecommendationEvaluationTests.cpp
Tests/ExperimentRecommendationPersistenceTests.cpp
Tests/ExperimentRecommendationRankingMigrationTests.sql
Tests/ExperimentRecommendationRankingRepositoryTests.cpp
Tests/ExperimentRecommendationRankingTests.cpp
Tests/ExperimentRecommendationRepositoryTests.cpp
Tests/ExperimentRecommendationReviewMigrationTests.sql
Tests/ExperimentRecommendationReviewRepositoryTests.cpp
Tests/ExperimentRecommendationReviewTests.cpp
Tests/ExperimentRecommendationScoringMigrationTests.sql
Tests/ExperimentRecommendationScoringRepositoryTests.cpp
Tests/ExperimentRecommendationScoringTests.cpp
Tests/ExperimentRecommendationServiceTests.cpp
Tests/ExperimentRecommendationTests.cpp
Tests/SchedulerChildStatusTests.cpp
docs/Phase4AExperimentRecommendationFoundation.rst
docs/Phase4AExperimentRecommendationPersistence.rst
docs/Phase4AExperimentRecommendationReview.rst
docs/Phase4AExperimentRecommendationScoring.rst
docs/Phase4BExperimentRecommendationEvaluation.rst
docs/Phase4BExperimentRecommendationRanking.rst
docs/Phase4CExperimentRecommendationConversion.rst
docs/Phase4CExperimentRecommendationConversionActivation.rst
docs/Phase4CExperimentRecommendationConversionExecution.rst
docs/Phase4CExperimentRecommendationConversionPersistence.rst
docs/Phase4CExperimentRecommendationConversionProposalReview.rst
docs/Phase4CExperimentRecommendationConversionWorkflow.rst
docs/Phase4DExperimentRecommendationCampaignApproval.rst
docs/Phase4DExperimentRecommendationCampaignHandoff.rst
docs/Phase4DExperimentRecommendationCampaignMaterialization.rst
docs/Phase4DExperimentRecommendationCampaignPlanning.rst
docs/Phase4DExperimentRecommendationCampaignProposalReview.rst
docs/Phase4DExperimentRecommendationCampaignReview.rst
docs/Phase5ExperimentRecommendationCampaignActivation.rst
docs/Phase5ExperimentRecommendationCampaignExecution.rst
docs/Phase5ExperimentRecommendationCampaignLaunch.rst
docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst
docs/Phase5ExperimentRecommendationCampaignStatus.rst
docs/Phase6ARecommendationCampaignFollowUpProposal.rst
docs/Phase6BRecommendationCampaignFollowUpProposalPersistence.rst
docs/Phase6CRecommendationCampaignFollowUpProposalReview.rst
docs/Phase6DRecommendationCampaignFollowUpProposalRatification.rst
docs/architecture/Volume_VIII_Recommendation_Engine.md
docs/architecture/Volume_XI_Scheduler.md
docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md
docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md
docs/architecture/adr/ADR-0005-manual-recommendation-conversion.md
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md

time: command terminated abnormally
       23.56 real         0.66 user         1.14 sys
           147062784  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
               61270  page reclaims
               12329  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                 398  messages sent
                5162  messages received
                  26  signals received
                2199  voluntary context switches
               40308  involuntary context switches
          7893238807  instructions retired
          4854616510  cycles elapsed
           119080760  peak memory footprint
time: signal: Invalid argument
size=32956 bytes modified=2026-07-22 08:00:20 -0500

codex
The named artifact has a fundamental completeness problem: it contains the original design prompt and a failed repository-listing transcript, but no architectural recommendation or final design. I’m continuing the repository-grounded review so the disposition is based on both that artifact defect and the actual Phase 4–6 contracts—not merely the missing text.
exec
/bin/bash -lc "git branch --show-current && git status --short && git log -n 12 --oneline --decorate && rg -n '"'^(#|##|###)|Phase 6|6E|Campaign Operations|campaign operations|authorit|scheduler|materializ|ratif|follow-up|lifecycle|budget|reservation|cancel|reconcil|replay|idempoten|privileg'"' docs/architecture/README.md docs/architecture/Volume_VII_Experiment_Lifecycle.md docs/architecture/Volume_VIII_Recommendation_Engine.md docs/architecture/Volume_X_Research_Automation.md docs/architecture/Volume_XI_Scheduler.md docs/architecture/Volume_XII_Database.md docs/architecture/adr/*.md docs/Phase4D*.rst docs/Phase5*.rst docs/Phase6*.rst | sed -n '1,1200p'" in /Volumes/Developer SSD/ExpertAdvisor
 succeeded in 176ms:
docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md:1:# ADR-0004: Scheduler ownership boundaries
docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md:10:## 1. Context and problem statement
docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md:14:invoked explicitly but are not scheduler work. Without a firm boundary, adding
docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md:16:implicit lifecycle transitions.
docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md:21:## 2. Decision
docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md:23:The scheduler owns only explicitly assigned lifecycle work classes.
docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md:25:- Eligibility and lifecycle transitions are durable database contracts.
docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md:26:- The scheduler atomically claims bounded work before launching a worker.
docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md:32:- New scheduler-managed work requires an ADR defining eligibility, claim,
docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md:33:  capacity, idempotency, recovery, observability, and regression boundaries.
docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md:34:- Recommendation generation, scoring, and review remain outside scheduler
docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md:37:## 3. Rationale and decision drivers
docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md:39:- Prevent duplicate workers and lost lifecycle updates.
docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md:44:## 4. Consequences
docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md:46:### 4.1 Positive consequences
docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md:48:- Clear ownership between lifecycle services, scheduler, and workers.
docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md:51:- Orphan recovery can reconcile durable attempt state with processes.
docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md:53:### 4.2 Negative consequences and trade-offs
docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md:56:- Process supervision and database reconciliation remain operationally complex.
docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md:59:### 4.3 Risks and mitigations
docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md:61:- Launch after claim fails: persist a distinct launch failure and reconcile.
docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md:62:- Multiple schedulers race: use database-visible conditional claims/locks.
docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md:66:## 5. Compatibility and migration
docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md:68:This ADR records existing scheduler boundaries and requires no schema change.
docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md:72:## 6. Verification and operational evidence
docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md:77:- Regression checks proving advisory commands do not enter scheduler polling.
docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md:80:## 7. Alternatives considered
docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md:82:### 7.1 Workers self-select follow-up work
docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md:84:Rejected because it bypasses centralized capacity, durable lifecycle ownership,
docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md:87:### 7.2 Hold a database transaction for worker lifetime
docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md:91:### 7.3 Treat every explicit command as pollable work
docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md:93:Rejected because command availability does not define scheduler eligibility or
docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md:96:## 8. References
docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md:103:## 9. Revision history
docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md:107:| 2026-07-15 | Recorded explicit scheduler work ownership and process/transaction boundaries. |
docs/architecture/adr/ADR-0002-deterministic-experiment-identity.md:1:# ADR-0002: Deterministic experiment identity
docs/architecture/adr/ADR-0002-deterministic-experiment-identity.md:10:## 1. Context and problem statement
docs/architecture/adr/ADR-0002-deterministic-experiment-identity.md:13:Database row IDs locate lifecycle records but do not describe semantic
docs/architecture/adr/ADR-0002-deterministic-experiment-identity.md:18:serve as authoritative equality without an additional collision contract.
docs/architecture/adr/ADR-0002-deterministic-experiment-identity.md:20:## 2. Decision
docs/architecture/adr/ADR-0002-deterministic-experiment-identity.md:24:- Semantic configuration canonical text is authoritative for the named
docs/architecture/adr/ADR-0002-deterministic-experiment-identity.md:28:- Row IDs, lifecycle status, timestamps, PIDs, logs, and result metrics do not
docs/architecture/adr/ADR-0002-deterministic-experiment-identity.md:38:## 3. Rationale and decision drivers
docs/architecture/adr/ADR-0002-deterministic-experiment-identity.md:45:## 4. Consequences
docs/architecture/adr/ADR-0002-deterministic-experiment-identity.md:47:### 4.1 Positive consequences
docs/architecture/adr/ADR-0002-deterministic-experiment-identity.md:54:### 4.2 Negative consequences and trade-offs
docs/architecture/adr/ADR-0002-deterministic-experiment-identity.md:60:### 4.3 Risks and mitigations
docs/architecture/adr/ADR-0002-deterministic-experiment-identity.md:68:## 5. Compatibility and migration
docs/architecture/adr/ADR-0002-deterministic-experiment-identity.md:75:## 6. Verification and operational evidence
docs/architecture/adr/ADR-0002-deterministic-experiment-identity.md:83:## 7. Alternatives considered
docs/architecture/adr/ADR-0002-deterministic-experiment-identity.md:85:### 7.1 Database row ID as experiment identity
docs/architecture/adr/ADR-0002-deterministic-experiment-identity.md:89:### 7.2 Hash-only equality
docs/architecture/adr/ADR-0002-deterministic-experiment-identity.md:93:### 7.3 One identity for semantics and invocation
docs/architecture/adr/ADR-0002-deterministic-experiment-identity.md:98:## 8. References
docs/architecture/adr/ADR-0002-deterministic-experiment-identity.md:105:## 9. Revision history
docs/architecture/Volume_VII_Experiment_Lifecycle.md:1:# Volume VII — Experiment Lifecycle
docs/architecture/Volume_VII_Experiment_Lifecycle.md:7:## 1. Purpose
docs/architecture/Volume_VII_Experiment_Lifecycle.md:10:checkpoint/continuation policy interaction, and lifecycle auditability.
docs/architecture/Volume_VII_Experiment_Lifecycle.md:12:## 2. Scope
docs/architecture/Volume_VII_Experiment_Lifecycle.md:14:### 2.1 In scope
docs/architecture/Volume_VII_Experiment_Lifecycle.md:19:### 2.2 Out of scope
docs/architecture/Volume_VII_Experiment_Lifecycle.md:21:Worker computation belongs to Volumes V/VI; scheduler polling and capacity to
docs/architecture/Volume_VII_Experiment_Lifecycle.md:24:### 2.3 Current implementation status
docs/architecture/Volume_VII_Experiment_Lifecycle.md:26:The platform implements persisted experiment lifecycle, checkpoint evaluation,
docs/architecture/Volume_VII_Experiment_Lifecycle.md:30:## 3. Responsibilities
docs/architecture/Volume_VII_Experiment_Lifecycle.md:32:### 3.1 Owned responsibilities
docs/architecture/Volume_VII_Experiment_Lifecycle.md:34:Own experiment configuration provenance, lifecycle state machine, attempt and
docs/architecture/Volume_VII_Experiment_Lifecycle.md:37:### 3.2 Dependencies
docs/architecture/Volume_VII_Experiment_Lifecycle.md:43:### 3.3 Prohibited responsibilities
docs/architecture/Volume_VII_Experiment_Lifecycle.md:48:## 4. Architecture
docs/architecture/Volume_VII_Experiment_Lifecycle.md:50:### 4.1 Components
docs/architecture/Volume_VII_Experiment_Lifecycle.md:52:Identity/domain rules, experiment repository, lifecycle service, continuation
docs/architecture/Volume_VII_Experiment_Lifecycle.md:53:policy/planner, operator CLI, and scheduler-facing claim interface.
docs/architecture/Volume_VII_Experiment_Lifecycle.md:55:### 4.2 Control flow
docs/architecture/Volume_VII_Experiment_Lifecycle.md:57:Validate identity/configuration → persist explicit experiment → scheduler
docs/architecture/Volume_VII_Experiment_Lifecycle.md:58:claims phase work → worker reports bounded outcome → lifecycle service advances
docs/architecture/Volume_VII_Experiment_Lifecycle.md:61:### 4.3 Ownership boundaries
docs/architecture/Volume_VII_Experiment_Lifecycle.md:63:Services validate transitions; repositories update atomically; scheduler owns
docs/architecture/Volume_VII_Experiment_Lifecycle.md:67:## 5. Data model
docs/architecture/Volume_VII_Experiment_Lifecycle.md:69:### 5.1 Authoritative entities
docs/architecture/Volume_VII_Experiment_Lifecycle.md:74:### 5.2 Provenance and versions
docs/architecture/Volume_VII_Experiment_Lifecycle.md:79:### 5.3 Invariants and legacy data
docs/architecture/Volume_VII_Experiment_Lifecycle.md:83:excluded where authoritative mapping is impossible.
docs/architecture/Volume_VII_Experiment_Lifecycle.md:85:## 6. Transactions
docs/architecture/Volume_VII_Experiment_Lifecycle.md:87:### 6.1 Read paths
docs/architecture/Volume_VII_Experiment_Lifecycle.md:89:Status/lineage inspection is read-only; scheduler eligibility reads a
docs/architecture/Volume_VII_Experiment_Lifecycle.md:92:### 6.2 Write paths
docs/architecture/Volume_VII_Experiment_Lifecycle.md:97:### 6.3 Failure semantics
docs/architecture/Volume_VII_Experiment_Lifecycle.md:102:## 7. Concurrency
docs/architecture/Volume_VII_Experiment_Lifecycle.md:104:### 7.1 Conflict domain
docs/architecture/Volume_VII_Experiment_Lifecycle.md:109:### 7.2 Locking and serialization
docs/architecture/Volume_VII_Experiment_Lifecycle.md:114:### 7.3 Winner, loser, and retry outcomes
docs/architecture/Volume_VII_Experiment_Lifecycle.md:119:## 8. CLI
docs/architecture/Volume_VII_Experiment_Lifecycle.md:121:### 8.1 Commands and validation
docs/architecture/Volume_VII_Experiment_Lifecycle.md:123:Queue, status, pause/resume/cancel/retry, checkpoint, and continuation commands
docs/architecture/Volume_VII_Experiment_Lifecycle.md:126:### 8.2 Machine output
docs/architecture/Volume_VII_Experiment_Lifecycle.md:131:### 8.3 Human output
docs/architecture/Volume_VII_Experiment_Lifecycle.md:136:## 9. Testing
docs/architecture/Volume_VII_Experiment_Lifecycle.md:138:### 9.1 Pure tests
docs/architecture/Volume_VII_Experiment_Lifecycle.md:142:### 9.2 Persistence and migration tests
docs/architecture/Volume_VII_Experiment_Lifecycle.md:146:### 9.3 Concurrency and integration tests
docs/architecture/Volume_VII_Experiment_Lifecycle.md:151:### 9.4 Regression boundaries
docs/architecture/Volume_VII_Experiment_Lifecycle.md:154:and scheduler capacity unless explicitly scoped.
docs/architecture/Volume_VII_Experiment_Lifecycle.md:156:## 10. Operational safety
docs/architecture/Volume_VII_Experiment_Lifecycle.md:158:### 10.1 Runtime isolation
docs/architecture/Volume_VII_Experiment_Lifecycle.md:160:Tests never target real experiments; lifecycle mutations require exact IDs and
docs/architecture/Volume_VII_Experiment_Lifecycle.md:163:### 10.2 Permissions and destructive operations
docs/architecture/Volume_VII_Experiment_Lifecycle.md:168:### 10.3 Observability and recovery
docs/architecture/Volume_VII_Experiment_Lifecycle.md:171:support reconciliation without treating process presence as sole truth.
docs/architecture/Volume_VII_Experiment_Lifecycle.md:173:## 11. Future extensions
docs/architecture/Volume_VII_Experiment_Lifecycle.md:175:### 11.1 Approved extension points
docs/architecture/Volume_VII_Experiment_Lifecycle.md:177:Versioned configuration contracts, explicit lifecycle services, and scheduler
docs/architecture/Volume_VII_Experiment_Lifecycle.md:180:### 11.2 Deferred capabilities
docs/architecture/Volume_VII_Experiment_Lifecycle.md:182:Recommendation conversion, experiment campaigns, budgets, and cross-machine
docs/architecture/Volume_VII_Experiment_Lifecycle.md:185:### 11.3 Required decisions
docs/architecture/Volume_VII_Experiment_Lifecycle.md:187:Each needs identity, authorization, capacity, idempotency, audit, and rollback
docs/architecture/Volume_VII_Experiment_Lifecycle.md:190:## 12. References
docs/architecture/Volume_VII_Experiment_Lifecycle.md:197:## 13. Revision history
docs/architecture/Volume_VII_Experiment_Lifecycle.md:201:| 0.1.0 | 2026-07-15 | Established the experiment-lifecycle outline. | — |
docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md:1:# ADR-0003: Advisory recommendation evaluation
docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md:10:## 1. Context and problem statement
docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md:15:but they do not provide a complete implementation identity, execution budget,
docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md:19:scores or status changes implicit operational authority.
docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md:21:## 2. Decision
docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md:30:  predict profitability, or establish scheduler eligibility.
docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md:31:- Recommendation commands remain outside scheduler polling.
docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md:36:## 3. Rationale and decision drivers
docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md:41:- Keep scheduler and experiment lifecycle ownership explicit.
docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md:43:## 4. Consequences
docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md:45:### 4.1 Positive consequences
docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md:49:- No Phase 4 command consumes scheduler capacity or changes experiments.
docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md:52:### 4.2 Negative consequences and trade-offs
docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md:58:### 4.3 Risks and mitigations
docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md:64:- Score-driven automation by accident: prevented by scheduler isolation and
docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md:67:## 5. Compatibility and migration
docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md:73:## 6. Verification and operational evidence
docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md:78:- CLI mutual-exclusion and scheduler-isolation tests.
docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md:79:- Runtime privilege checks for immutable review history.
docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md:81:## 7. Alternatives considered
docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md:83:### 7.1 Automatically queue the highest score
docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md:86:authority or evidence of expected profitability.
docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md:88:### 7.2 Treat approval as conversion authorization
docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md:90:Rejected because Step 5 lacks a conversion identity, budget, lifecycle request,
docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md:91:and scheduler-capacity contract.
docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md:93:### 7.3 Scheduler-managed recommendation polling
docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md:98:## 8. References
docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md:104:## 9. Revision history
docs/architecture/README.md:1:# LSTM Research Platform Architecture
docs/architecture/README.md:7:## 1. Purpose and authority
docs/architecture/README.md:22:## 2. Document organization
docs/architecture/README.md:44:The [ADR index](adr/README.md) defines the ADR lifecycle and numbering scheme.
docs/architecture/README.md:46:## 3. Architecture and implementation
docs/architecture/README.md:71:## 4. Proposing an architectural change
docs/architecture/README.md:90:## 5. Review expectations
docs/architecture/README.md:94:- Is the authoritative owner of each state transition clear?
docs/architecture/README.md:98:- Are scheduler and worker responsibilities separated?
docs/architecture/README.md:104:## 6. Relationship to Codex implementation prompts
docs/architecture/README.md:127:permission to operate schedulers, mutate production data, create experiments,
docs/architecture/README.md:130:## 7. Document maintenance
docs/architecture/README.md:141:## 8. Existing implementation references
docs/architecture/README.md:150:## 9. Revision history
docs/architecture/adr/ADR-0001-postgresql-source-of-truth.md:1:# ADR-0001: PostgreSQL as the durable source of truth
docs/architecture/adr/ADR-0001-postgresql-source-of-truth.md:10:## 1. Context and problem statement
docs/architecture/adr/ADR-0001-postgresql-source-of-truth.md:16:referential integrity, or a consistent lifecycle record.
docs/architecture/adr/ADR-0001-postgresql-source-of-truth.md:21:## 2. Decision
docs/architecture/adr/ADR-0001-postgresql-source-of-truth.md:23:PostgreSQL is the authoritative durable source of truth for platform state
docs/architecture/adr/ADR-0001-postgresql-source-of-truth.md:30:  authority.
docs/architecture/adr/ADR-0001-postgresql-source-of-truth.md:33:- Runtime roles receive least privilege, and append-only audit history denies
docs/architecture/adr/ADR-0001-postgresql-source-of-truth.md:34:  ordinary runtime update/delete privileges.
docs/architecture/adr/ADR-0001-postgresql-source-of-truth.md:37:snapshots of every external market-data source. Exact data-replay identity
docs/architecture/adr/ADR-0001-postgresql-source-of-truth.md:40:## 3. Rationale and decision drivers
docs/architecture/adr/ADR-0001-postgresql-source-of-truth.md:42:- Transactions provide atomic lifecycle and provenance updates.
docs/architecture/adr/ADR-0001-postgresql-source-of-truth.md:48:## 4. Consequences
docs/architecture/adr/ADR-0001-postgresql-source-of-truth.md:50:### 4.1 Positive consequences
docs/architecture/adr/ADR-0001-postgresql-source-of-truth.md:57:### 4.2 Negative consequences and trade-offs
docs/architecture/adr/ADR-0001-postgresql-source-of-truth.md:63:### 4.3 Risks and mitigations
docs/architecture/adr/ADR-0001-postgresql-source-of-truth.md:66:- Excess privilege: mitigated by role-specific grants and privilege tests.
docs/architecture/adr/ADR-0001-postgresql-source-of-truth.md:70:## 5. Compatibility and migration
docs/architecture/adr/ADR-0001-postgresql-source-of-truth.md:76:## 6. Verification and operational evidence
docs/architecture/adr/ADR-0001-postgresql-source-of-truth.md:84:## 7. Alternatives considered
docs/architecture/adr/ADR-0001-postgresql-source-of-truth.md:86:### 7.1 Logs and process state as authority
docs/architecture/adr/ADR-0001-postgresql-source-of-truth.md:91:### 7.2 Filesystem-only state
docs/architecture/adr/ADR-0001-postgresql-source-of-truth.md:93:Rejected as the primary lifecycle store because multi-process claims,
docs/architecture/adr/ADR-0001-postgresql-source-of-truth.md:97:### 7.3 Per-subsystem independent stores
docs/architecture/adr/ADR-0001-postgresql-source-of-truth.md:102:## 8. References
docs/architecture/adr/ADR-0001-postgresql-source-of-truth.md:108:## 9. Revision history
docs/architecture/Volume_X_Research_Automation.md:1:# Volume X — Research Automation
docs/architecture/Volume_X_Research_Automation.md:7:## 1. Purpose
docs/architecture/Volume_X_Research_Automation.md:10:bounded research while preserving explicit policy, budgets, audit, and human
docs/architecture/Volume_X_Research_Automation.md:11:authority.
docs/architecture/Volume_X_Research_Automation.md:13:## 2. Scope
docs/architecture/Volume_X_Research_Automation.md:15:### 2.1 In scope
docs/architecture/Volume_X_Research_Automation.md:17:Future campaign goals, proposal policies, budgets, stopping conditions,
docs/architecture/Volume_X_Research_Automation.md:20:### 2.2 Out of scope
docs/architecture/Volume_X_Research_Automation.md:23:creation, live trading, or bypassing scheduler/lifecycle ownership.
docs/architecture/Volume_X_Research_Automation.md:25:### 2.3 Current implementation status
docs/architecture/Volume_X_Research_Automation.md:30:## 3. Responsibilities
docs/architecture/Volume_X_Research_Automation.md:32:### 3.1 Owned responsibilities
docs/architecture/Volume_X_Research_Automation.md:34:Future automation owns declared campaign policy and budget accounting only.
docs/architecture/Volume_X_Research_Automation.md:36:### 3.2 Dependencies
docs/architecture/Volume_X_Research_Automation.md:41:### 3.3 Prohibited responsibilities
docs/architecture/Volume_X_Research_Automation.md:43:MUST NOT bypass review, invent authorization, exceed budgets, mutate model math,
docs/architecture/Volume_X_Research_Automation.md:46:## 4. Architecture
docs/architecture/Volume_X_Research_Automation.md:48:### 4.1 Components
docs/architecture/Volume_X_Research_Automation.md:50:Potential components are campaign policy, planner, budget ledger, proposal
docs/architecture/Volume_X_Research_Automation.md:51:adapter, decision/audit repository, operator controls, and scheduler adapter.
docs/architecture/Volume_X_Research_Automation.md:53:### 4.2 Control flow
docs/architecture/Volume_X_Research_Automation.md:56:policy gate → lifecycle request → scheduler execution → evidence update → stop.
docs/architecture/Volume_X_Research_Automation.md:58:### 4.3 Ownership boundaries
docs/architecture/Volume_X_Research_Automation.md:61:lifecycle creates. Scheduler claims and runs. Operators retain pause/stop power.
docs/architecture/Volume_X_Research_Automation.md:63:## 5. Data model
docs/architecture/Volume_X_Research_Automation.md:65:### 5.1 Authoritative entities
docs/architecture/Volume_X_Research_Automation.md:67:Future campaign, policy version, budget, proposal, decision, execution request,
docs/architecture/Volume_X_Research_Automation.md:70:### 5.2 Provenance and versions
docs/architecture/Volume_X_Research_Automation.md:72:Every decision records exact inputs, policy canonical text/version, budgets,
docs/architecture/Volume_X_Research_Automation.md:75:### 5.3 Invariants and legacy data
docs/architecture/Volume_X_Research_Automation.md:77:Budget consumption is monotonic and auditable. Missing authority or provenance
docs/architecture/Volume_X_Research_Automation.md:80:## 6. Transactions
docs/architecture/Volume_X_Research_Automation.md:82:### 6.1 Read paths
docs/architecture/Volume_X_Research_Automation.md:86:### 6.2 Write paths
docs/architecture/Volume_X_Research_Automation.md:88:Decision, budget reservation, and execution request require an explicit atomic
docs/architecture/Volume_X_Research_Automation.md:91:### 6.3 Failure semantics
docs/architecture/Volume_X_Research_Automation.md:93:Uncertain budget or request state fails closed and requires reconciliation.
docs/architecture/Volume_X_Research_Automation.md:95:## 7. Concurrency
docs/architecture/Volume_X_Research_Automation.md:97:### 7.1 Conflict domain
docs/architecture/Volume_X_Research_Automation.md:99:Same campaign budget, proposal, or execution request.
docs/architecture/Volume_X_Research_Automation.md:101:### 7.2 Locking and serialization
docs/architecture/Volume_X_Research_Automation.md:103:Future ledgers use row locks/conditional updates and authoritative request IDs.
docs/architecture/Volume_X_Research_Automation.md:105:### 7.3 Winner, loser, and retry outcomes
docs/architecture/Volume_X_Research_Automation.md:107:At most one accepted request consumes a reservation; retries return the same
docs/architecture/Volume_X_Research_Automation.md:108:request only when authoritative idempotency identity matches.
docs/architecture/Volume_X_Research_Automation.md:110:## 8. CLI
docs/architecture/Volume_X_Research_Automation.md:112:### 8.1 Commands and validation
docs/architecture/Volume_X_Research_Automation.md:117:### 8.2 Machine output
docs/architecture/Volume_X_Research_Automation.md:119:Events expose policy, budget, evidence, decision, request, and stop reason.
docs/architecture/Volume_X_Research_Automation.md:121:### 8.3 Human output
docs/architecture/Volume_X_Research_Automation.md:123:Summaries state remaining budget, current authority, and whether any work was
docs/architecture/Volume_X_Research_Automation.md:126:## 9. Testing
docs/architecture/Volume_X_Research_Automation.md:128:### 9.1 Pure tests
docs/architecture/Volume_X_Research_Automation.md:130:Policy, budgets, stopping, selection, and deterministic planning.
docs/architecture/Volume_X_Research_Automation.md:132:### 9.2 Persistence and migration tests
docs/architecture/Volume_X_Research_Automation.md:134:Ledger monotonicity, request idempotency, audit immutability, and permissions.
docs/architecture/Volume_X_Research_Automation.md:136:### 9.3 Concurrency and integration tests
docs/architecture/Volume_X_Research_Automation.md:138:Budget races, duplicate requests, pause/stop races, and scheduler isolation.
docs/architecture/Volume_X_Research_Automation.md:140:### 9.4 Regression boundaries
docs/architecture/Volume_X_Research_Automation.md:142:Recommendation, lifecycle, continuation, and scheduler semantics remain owned
docs/architecture/Volume_X_Research_Automation.md:145:## 10. Operational safety
docs/architecture/Volume_X_Research_Automation.md:147:### 10.1 Runtime isolation
docs/architecture/Volume_X_Research_Automation.md:152:### 10.2 Permissions and destructive operations
docs/architecture/Volume_X_Research_Automation.md:154:Least-privilege roles, explicit budgets, operator stop controls, and audit are
docs/architecture/Volume_X_Research_Automation.md:157:### 10.3 Observability and recovery
docs/architecture/Volume_X_Research_Automation.md:159:Durable campaign status, reservations, requests, outcomes, and reconciliation
docs/architecture/Volume_X_Research_Automation.md:162:## 11. Future extensions
docs/architecture/Volume_X_Research_Automation.md:164:### 11.1 Approved extension points
docs/architecture/Volume_X_Research_Automation.md:168:### 11.2 Deferred capabilities
docs/architecture/Volume_X_Research_Automation.md:173:### 11.3 Required decisions
docs/architecture/Volume_X_Research_Automation.md:175:Multiple ADRs are required for authority, budgets, safety, policy identity,
docs/architecture/Volume_X_Research_Automation.md:176:conversion, scheduler capacity, and shutdown before implementation.
docs/architecture/Volume_X_Research_Automation.md:178:## 12. References
docs/architecture/Volume_X_Research_Automation.md:185:## 13. Revision history
docs/architecture/Volume_XII_Database.md:1:# Volume XII — Database
docs/architecture/Volume_XII_Database.md:3:Status: Foundation aligned through Phase 6C; proposed Phase 6D recorded
docs/architecture/Volume_XII_Database.md:7:## 1. Purpose
docs/architecture/Volume_XII_Database.md:10:constraints, privileges, transaction conventions, and audit-history policy.
docs/architecture/Volume_XII_Database.md:12:## 2. Scope
docs/architecture/Volume_XII_Database.md:14:### 2.1 In scope
docs/architecture/Volume_XII_Database.md:19:### 2.2 Out of scope
docs/architecture/Volume_XII_Database.md:21:Domain formulas, lifecycle policy, scheduler selection, and operator
docs/architecture/Volume_XII_Database.md:24:### 2.3 Current implementation status
docs/architecture/Volume_XII_Database.md:27:are checksum-recorded by the project runner. Phase 6B migration 042 adds an
docs/architecture/Volume_XII_Database.md:28:append-only exact follow-up-proposal manifest and ordered member table for
docs/architecture/Volume_XII_Database.md:29:read-only operator preview. Phase 6C migration 043 adds one append-only
docs/architecture/Volume_XII_Database.md:31:adds lifecycle, experiment, execution, queue, or scheduler state.
docs/architecture/Volume_XII_Database.md:32:Phase 6D migration 044 adds one append-only governance-ratification event per
docs/architecture/Volume_XII_Database.md:33:exact eligible approved Phase 6C review. The fixed ratifier role and mandatory
docs/architecture/Volume_XII_Database.md:34:reviewer/ratifier separation distinguish it from merits review. It adds no
docs/architecture/Volume_XII_Database.md:35:Phase 6E, lifecycle, experiment, execution, queue, or scheduler state.
docs/architecture/Volume_XII_Database.md:37:## 3. Responsibilities
docs/architecture/Volume_XII_Database.md:39:### 3.1 Owned responsibilities
docs/architecture/Volume_XII_Database.md:42:uniqueness, runtime privileges, migration history, and transactional isolation.
docs/architecture/Volume_XII_Database.md:44:### 3.2 Dependencies
docs/architecture/Volume_XII_Database.md:49:### 3.3 Prohibited responsibilities
docs/architecture/Volume_XII_Database.md:52:policy without a domain contract, or rely on logs/process memory as authority.
docs/architecture/Volume_XII_Database.md:54:## 4. Architecture
docs/architecture/Volume_XII_Database.md:56:### 4.1 Components
docs/architecture/Volume_XII_Database.md:61:### 4.2 Control flow
docs/architecture/Volume_XII_Database.md:67:### 4.3 Ownership boundaries
docs/architecture/Volume_XII_Database.md:70:truth; migrations own schema evolution; scheduler/workers use least privilege.
docs/architecture/Volume_XII_Database.md:72:## 5. Data model
docs/architecture/Volume_XII_Database.md:74:### 5.1 Authoritative entities
docs/architecture/Volume_XII_Database.md:76:The schema represents experiments, models, evidence, policies, lifecycle,
docs/architecture/Volume_XII_Database.md:79:Phase 6B follow-up proposal rows durably preserve the complete immutable Phase
docs/architecture/Volume_XII_Database.md:81:are repository metadata outside the authoritative proposal identity.
docs/architecture/Volume_XII_Database.md:82:Phase 6C review events preserve the exact proposal ID/version/canonical/hash,
docs/architecture/Volume_XII_Database.md:86:Phase 6D ratification events preserve the exact review-event ID and complete
docs/architecture/Volume_XII_Database.md:88:approved eligibility, fixed authority role, ``ratified`` decision, distinct
docs/architecture/Volume_XII_Database.md:89:ratifier, basis, and complete ratification canonical/hash. Ratification-event
docs/architecture/Volume_XII_Database.md:90:ID and creation timestamp remain metadata outside identity, and ratification
docs/architecture/Volume_XII_Database.md:91:grants no Phase 6E or operational authority.
docs/architecture/Volume_XII_Database.md:93:### 5.2 Provenance and versions
docs/architecture/Volume_XII_Database.md:99:### 5.3 Invariants and legacy data
docs/architecture/Volume_XII_Database.md:106:## 6. Transactions
docs/architecture/Volume_XII_Database.md:108:### 6.1 Read paths
docs/architecture/Volume_XII_Database.md:113:### 6.2 Write paths
docs/architecture/Volume_XII_Database.md:117:Phase 6B inserts one proposal manifest and its complete ordered members under a
docs/architecture/Volume_XII_Database.md:119:Phase 6C serializes one proposal-ID conflict domain with a transaction advisory
docs/architecture/Volume_XII_Database.md:120:lock, verifies the exact Phase 6B row, and inserts at most one review event. It
docs/architecture/Volume_XII_Database.md:121:never updates or supersedes an event or changes a Phase 6B row.
docs/architecture/Volume_XII_Database.md:122:Phase 6D serializes one review-event-ID conflict domain with a transaction
docs/architecture/Volume_XII_Database.md:123:advisory lock, validates the complete Phase 6C/6B chain and approved
docs/architecture/Volume_XII_Database.md:125:ratification event. It never changes any upstream row.
docs/architecture/Volume_XII_Database.md:127:### 6.3 Failure semantics
docs/architecture/Volume_XII_Database.md:132:Phase 6C returns ``recorded`` for the first valid event,
docs/architecture/Volume_XII_Database.md:133:``existing_identical`` for exact replay, and a deterministic conflict for any
docs/architecture/Volume_XII_Database.md:135:Phase 6D returns the same recorded/existing-identical outcomes and conflicts on
docs/architecture/Volume_XII_Database.md:136:any different ratification payload for the same review.
docs/architecture/Volume_XII_Database.md:138:## 7. Concurrency
docs/architecture/Volume_XII_Database.md:140:### 7.1 Conflict domain
docs/architecture/Volume_XII_Database.md:145:### 7.2 Locking and serialization
docs/architecture/Volume_XII_Database.md:150:Phase 6C uses no proposal row lock: one proposal-ID-scoped advisory lock plus a
docs/architecture/Volume_XII_Database.md:151:unique proposal foreign key serializes attempts without mutating Phase 6B.
docs/architecture/Volume_XII_Database.md:152:Phase 6D similarly uses no upstream row lock: one review-event-ID-scoped
docs/architecture/Volume_XII_Database.md:155:### 7.3 Winner, loser, and retry outcomes
docs/architecture/Volume_XII_Database.md:158:commit; applications do not reinterpret arbitrary SQL failure as idempotency.
docs/architecture/Volume_XII_Database.md:159:Concurrent identical Phase 6C events converge on one row; concurrent differing
docs/architecture/Volume_XII_Database.md:161:Concurrent identical Phase 6D ratifications converge on one row; concurrent
docs/architecture/Volume_XII_Database.md:162:differing ratifications produce one recorded winner and one deterministic
docs/architecture/Volume_XII_Database.md:165:## 8. CLI
docs/architecture/Volume_XII_Database.md:167:### 8.1 Commands and validation
docs/architecture/Volume_XII_Database.md:172:### 8.2 Machine output
docs/architecture/Volume_XII_Database.md:177:### 8.3 Human output
docs/architecture/Volume_XII_Database.md:181:Phase 6C show/list records expose event/proposal IDs, identities, decision,
docs/architecture/Volume_XII_Database.md:182:reviewer, reason, timestamp, and explicit negative action-authority fields.
docs/architecture/Volume_XII_Database.md:183:Phase 6D adds no main-program CLI. Its repository lookup/list operations are
docs/architecture/Volume_XII_Database.md:186:## 9. Testing
docs/architecture/Volume_XII_Database.md:188:### 9.1 Pure tests
docs/architecture/Volume_XII_Database.md:193:### 9.2 Persistence and migration tests
docs/architecture/Volume_XII_Database.md:198:### 9.3 Concurrency and integration tests
docs/architecture/Volume_XII_Database.md:201:uniqueness, rollback, privileges, and unrelated-row concurrency.
docs/architecture/Volume_XII_Database.md:202:Phase 6C additionally verifies concurrent identical/conflicting reviews and
docs/architecture/Volume_XII_Database.md:204:Phase 6D additionally verifies clean and upgrade paths, exact eligibility,
docs/architecture/Volume_XII_Database.md:206:ratifications, rollback, NULL-ACL fallback, and read-only lookup/list without
docs/architecture/Volume_XII_Database.md:209:### 9.4 Regression boundaries
docs/architecture/Volume_XII_Database.md:212:permissions, scheduler behavior, and repository compatibility.
docs/architecture/Volume_XII_Database.md:213:Migration 043 tests preserve Phase 6B manifests/members and experiment and
docs/architecture/Volume_XII_Database.md:214:scheduler sentinels byte-for-byte.
docs/architecture/Volume_XII_Database.md:215:Migration 044 tests preserve Phase 6B manifests/members, Phase 6C reviews, and
docs/architecture/Volume_XII_Database.md:216:experiment/scheduler sentinels byte-for-byte.
docs/architecture/Volume_XII_Database.md:218:## 10. Operational safety
docs/architecture/Volume_XII_Database.md:220:### 10.1 Runtime isolation
docs/architecture/Volume_XII_Database.md:225:### 10.2 Permissions and destructive operations
docs/architecture/Volume_XII_Database.md:227:Runtime roles receive minimum table/sequence privileges. Immutable audit tables
docs/architecture/Volume_XII_Database.md:230:Phase 6C grants only review-table ``SELECT``, payload-column ``INSERT``, and
docs/architecture/Volume_XII_Database.md:233:Phase 6D grants the same narrow privilege shape for its ratification table and
docs/architecture/Volume_XII_Database.md:236:PostgreSQL default ACLs in privilege tests.
docs/architecture/Volume_XII_Database.md:238:### 10.3 Observability and recovery
docs/architecture/Volume_XII_Database.md:244:## 11. Future extensions
docs/architecture/Volume_XII_Database.md:246:### 11.1 Approved extension points
docs/architecture/Volume_XII_Database.md:249:least-privilege roles.
docs/architecture/Volume_XII_Database.md:251:### 11.2 Deferred capabilities
docs/architecture/Volume_XII_Database.md:256:### 11.3 Required decisions
docs/architecture/Volume_XII_Database.md:261:## 12. References
docs/architecture/Volume_XII_Database.md:265:- [ADR-0007](adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md)
docs/architecture/Volume_XII_Database.md:266:- [ADR-0008](adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md)
docs/architecture/Volume_XII_Database.md:267:- [ADR-0009](adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md)
docs/architecture/Volume_XII_Database.md:268:- [Phase 6B persistence and preview](../Phase6BRecommendationCampaignFollowUpProposalPersistence.rst)
docs/architecture/Volume_XII_Database.md:269:- [Phase 6C administrative review](../Phase6CRecommendationCampaignFollowUpProposalReview.rst)
docs/architecture/Volume_XII_Database.md:270:- [Phase 6D governance ratification](../Phase6DRecommendationCampaignFollowUpProposalRatification.rst)
docs/architecture/Volume_XII_Database.md:274:## 13. Revision history
docs/architecture/Volume_XII_Database.md:279:| 0.2.0 | 2026-07-20 | Recorded the append-only exact Phase 6B follow-up-proposal schema, collision-safe transaction, and least-privilege read-only preview boundary. | ADR-0001, ADR-0007 |
docs/architecture/Volume_XII_Database.md:280:| 0.3.0 | 2026-07-21 | Recorded one exact append-only Phase 6C administrative review event per proposal, deterministic replay/conflict, validated reload, and read-only presentation without action authority. | ADR-0001, ADR-0004, ADR-0008 |
docs/architecture/Volume_XII_Database.md:281:| 0.4.0 | 2026-07-22 | Recorded proposed append-only Phase 6D governance ratification per eligible approved review, fixed role, mandatory separation of duties, deterministic replay/conflict, and least privilege without Phase 6E authority. | ADR-0001, ADR-0004, ADR-0009 |
docs/architecture/Volume_XI_Scheduler.md:1:# Volume XI — Scheduler
docs/architecture/Volume_XI_Scheduler.md:7:## 1. Purpose
docs/architecture/Volume_XI_Scheduler.md:10:launch, lifecycle reconciliation, recovery, and scheduler observability.
docs/architecture/Volume_XI_Scheduler.md:12:## 2. Scope
docs/architecture/Volume_XI_Scheduler.md:14:### 2.1 In scope
docs/architecture/Volume_XI_Scheduler.md:19:### 2.2 Out of scope
docs/architecture/Volume_XI_Scheduler.md:24:### 2.3 Current implementation status
docs/architecture/Volume_XI_Scheduler.md:26:The scheduler manages experiment training, inference, analysis, checkpoint, and
docs/architecture/Volume_XI_Scheduler.md:30:## 3. Responsibilities
docs/architecture/Volume_XI_Scheduler.md:32:### 3.1 Owned responsibilities
docs/architecture/Volume_XI_Scheduler.md:37:### 3.2 Dependencies
docs/architecture/Volume_XI_Scheduler.md:39:Consumes Volume VII lifecycle state and invokes bounded workers from Volumes
docs/architecture/Volume_XI_Scheduler.md:42:### 3.3 Prohibited responsibilities
docs/architecture/Volume_XI_Scheduler.md:44:The scheduler MUST NOT invent experiments, approve recommendations, alter
docs/architecture/Volume_XI_Scheduler.md:47:## 4. Architecture
docs/architecture/Volume_XI_Scheduler.md:49:### 4.1 Components
docs/architecture/Volume_XI_Scheduler.md:52:command builder, process supervisor, completion reconciler, recovery logic,
docs/architecture/Volume_XI_Scheduler.md:55:### 4.2 Control flow
docs/architecture/Volume_XI_Scheduler.md:58:worker outside the transaction → observe completion → persist authoritative
docs/architecture/Volume_XI_Scheduler.md:61:### 4.3 Ownership boundaries
docs/architecture/Volume_XI_Scheduler.md:63:Lifecycle services define legal transitions; scheduler selects and claims;
docs/architecture/Volume_XI_Scheduler.md:67:## 5. Data model
docs/architecture/Volume_XI_Scheduler.md:69:### 5.1 Authoritative entities
docs/architecture/Volume_XI_Scheduler.md:72:class, exit/failure state, scheduler version, and recovery metadata.
docs/architecture/Volume_XI_Scheduler.md:74:### 5.2 Provenance and versions
docs/architecture/Volume_XI_Scheduler.md:77:metadata, scheduler/build provenance, and attempt ownership.
docs/architecture/Volume_XI_Scheduler.md:79:### 5.3 Invariants and legacy data
docs/architecture/Volume_XI_Scheduler.md:83:are reconciled conservatively.
docs/architecture/Volume_XI_Scheduler.md:85:## 6. Transactions
docs/architecture/Volume_XI_Scheduler.md:87:### 6.1 Read paths
docs/architecture/Volume_XI_Scheduler.md:92:### 6.2 Write paths
docs/architecture/Volume_XI_Scheduler.md:98:### 6.3 Failure semantics
docs/architecture/Volume_XI_Scheduler.md:101:distinct. Recovery never marks work complete without authoritative evidence.
docs/architecture/Volume_XI_Scheduler.md:103:## 7. Concurrency
docs/architecture/Volume_XI_Scheduler.md:105:### 7.1 Conflict domain
docs/architecture/Volume_XI_Scheduler.md:109:### 7.2 Locking and serialization
docs/architecture/Volume_XI_Scheduler.md:115:### 7.3 Winner, loser, and retry outcomes
docs/architecture/Volume_XI_Scheduler.md:117:Exactly one scheduler/attempt claims an operation. Losers observe changed state
docs/architecture/Volume_XI_Scheduler.md:121:## 8. CLI
docs/architecture/Volume_XI_Scheduler.md:123:### 8.1 Commands and validation
docs/architecture/Volume_XI_Scheduler.md:125:Scheduler start/status/recovery and operator lifecycle controls are explicit.
docs/architecture/Volume_XI_Scheduler.md:128:### 8.2 Machine output
docs/architecture/Volume_XI_Scheduler.md:130:Events identify scheduler cycle, experiment, phase, attempt, operation,
docs/architecture/Volume_XI_Scheduler.md:133:### 8.3 Human output
docs/architecture/Volume_XI_Scheduler.md:138:## 9. Testing
docs/architecture/Volume_XI_Scheduler.md:140:### 9.1 Pure tests
docs/architecture/Volume_XI_Scheduler.md:145:### 9.2 Persistence and migration tests
docs/architecture/Volume_XI_Scheduler.md:149:### 9.3 Concurrency and integration tests
docs/architecture/Volume_XI_Scheduler.md:151:Multiple schedulers, capacity races, launch failure, completion races, orphan
docs/architecture/Volume_XI_Scheduler.md:154:### 9.4 Regression boundaries
docs/architecture/Volume_XI_Scheduler.md:159:## 10. Operational safety
docs/architecture/Volume_XI_Scheduler.md:161:### 10.1 Runtime isolation
docs/architecture/Volume_XI_Scheduler.md:163:Tests do not start or signal a real scheduler without explicit authorization.
docs/architecture/Volume_XI_Scheduler.md:166:### 10.2 Permissions and destructive operations
docs/architecture/Volume_XI_Scheduler.md:168:Scheduler roles receive only required lifecycle/attempt privileges. Force-stop
docs/architecture/Volume_XI_Scheduler.md:171:### 10.3 Observability and recovery
docs/architecture/Volume_XI_Scheduler.md:174:diagnostics support reconciliation after scheduler or worker failure.
docs/architecture/Volume_XI_Scheduler.md:176:## 11. Future extensions
docs/architecture/Volume_XI_Scheduler.md:178:### 11.1 Approved extension points
docs/architecture/Volume_XI_Scheduler.md:182:### 11.2 Deferred capabilities
docs/architecture/Volume_XI_Scheduler.md:184:Distributed schedulers, remote workers, priority budgets, and research-campaign
docs/architecture/Volume_XI_Scheduler.md:187:### 11.3 Required decisions
docs/architecture/Volume_XI_Scheduler.md:190:idempotency, recovery, operator control, and regression scope.
docs/architecture/Volume_XI_Scheduler.md:192:## 12. References
docs/architecture/Volume_XI_Scheduler.md:195:- [ADR-0004](adr/ADR-0004-scheduler-ownership-boundaries.md)
docs/architecture/Volume_XI_Scheduler.md:199:## 13. Revision history
docs/architecture/Volume_XI_Scheduler.md:203:| 0.1.0 | 2026-07-15 | Established scheduler ownership and safety outline. | ADR-0004 |
docs/architecture/Volume_VIII_Recommendation_Engine.md:1:# Volume VIII — Recommendation Engine
docs/architecture/Volume_VIII_Recommendation_Engine.md:3:Status: Foundation aligned through Phase 6C; proposed Phase 6D recorded
docs/architecture/Volume_VIII_Recommendation_Engine.md:7:## 1. Purpose
docs/architecture/Volume_VIII_Recommendation_Engine.md:13:approval, materialization, handoff observation, and exact materialized-proposal
docs/architecture/Volume_VIII_Recommendation_Engine.md:16:Phase 4C paused conversion execution to every exact materialized campaign
docs/architecture/Volume_VIII_Recommendation_Engine.md:17:member, without activation, queueing, scheduler control, or workers.
docs/architecture/Volume_VIII_Recommendation_Engine.md:20:``paused/train`` to ``pending/train`` without starting scheduler or workers.
docs/architecture/Volume_VIII_Recommendation_Engine.md:22:that composes those same execution and activation authorities for the exact
docs/architecture/Volume_VIII_Recommendation_Engine.md:23:membership without adding campaign state or scheduler behavior.
docs/architecture/Volume_VIII_Recommendation_Engine.md:26:analysis evidence without adding durable campaign state or scheduler behavior.
docs/architecture/Volume_VIII_Recommendation_Engine.md:27:Phase 5 Steps 5a–5b add a pure, non-authoritative scientific outcome assessment
docs/architecture/Volume_VIII_Recommendation_Engine.md:28:and its bounded, read-only authoritative-persisted-evidence integration. Step
docs/architecture/Volume_VIII_Recommendation_Engine.md:31:success, or authorize follow-up.
docs/architecture/Volume_VIII_Recommendation_Engine.md:32:Phase 6A adds one pure immutable follow-up proposal bound to an exactly aligned
docs/architecture/Volume_VIII_Recommendation_Engine.md:35:persistence, approval, lifecycle, or scheduler authority.
docs/architecture/Volume_VIII_Recommendation_Engine.md:36:Phase 6B durably records that exact immutable advisory value and exposes a
docs/architecture/Volume_VIII_Recommendation_Engine.md:38:decision, approval, activation, execution, experiment, queue, scheduler, or
docs/architecture/Volume_VIII_Recommendation_Engine.md:39:follow-up authorization authority.
docs/architecture/Volume_VIII_Recommendation_Engine.md:40:Phase 6C records exactly one immutable approved or rejected administrative
docs/architecture/Volume_VIII_Recommendation_Engine.md:41:review for one exact persisted Phase 6B proposal. Approval permits only possible
docs/architecture/Volume_VIII_Recommendation_Engine.md:43:follow-up, lifecycle, experiment, worker, or scheduler authority.
docs/architecture/Volume_VIII_Recommendation_Engine.md:44:Phase 6D separately records one immutable governance ratification for
docs/architecture/Volume_VIII_Recommendation_Engine.md:45:advancement of one exact persisted Phase 6C ``approved`` review into the next
docs/architecture/Volume_VIII_Recommendation_Engine.md:46:separately controlled phase. The fixed ratifier role and mandatory reviewer/
docs/architecture/Volume_VIII_Recommendation_Engine.md:47:ratifier separation make this authority distinct from merits review. It grants
docs/architecture/Volume_VIII_Recommendation_Engine.md:48:no Phase 6E capability or operational authority.
docs/architecture/Volume_VIII_Recommendation_Engine.md:50:## 2. Scope
docs/architecture/Volume_VIII_Recommendation_Engine.md:52:### 2.1 In scope
docs/architecture/Volume_VIII_Recommendation_Engine.md:60:review, explicit non-executing approval, and approved-campaign materialization
docs/architecture/Volume_VIII_Recommendation_Engine.md:63:The pure Phase 5 outcome contracts and the pure identity-bound Phase 6A
docs/architecture/Volume_VIII_Recommendation_Engine.md:64:follow-up proposal are in scope as advisory values. Phase 6B exact append-only
docs/architecture/Volume_VIII_Recommendation_Engine.md:65:proposal persistence/read-only preview and Phase 6C explicit immutable
docs/architecture/Volume_VIII_Recommendation_Engine.md:66:administrative review/read-only presentation are also in scope. Phase 6D exact
docs/architecture/Volume_VIII_Recommendation_Engine.md:67:append-only governance ratification and its transactional typed service are in
docs/architecture/Volume_VIII_Recommendation_Engine.md:68:scope without Phase 6E, activation, or execution capability.
docs/architecture/Volume_VIII_Recommendation_Engine.md:70:### 2.2 Out of scope
docs/architecture/Volume_VIII_Recommendation_Engine.md:74:recommendation-driven scheduler polling, worker execution, and automatic
docs/architecture/Volume_VIII_Recommendation_Engine.md:77:### 2.3 Current implementation status
docs/architecture/Volume_VIII_Recommendation_Engine.md:92:them. Step 5 observes every exact materialized proposal, and Step 6 explicitly
docs/architecture/Volume_VIII_Recommendation_Engine.md:98:all exact executed members atomically; it performs no automatic follow-up.
docs/architecture/Volume_VIII_Recommendation_Engine.md:101:experiments ``pending/train`` without starting scheduler work.
docs/architecture/Volume_VIII_Recommendation_Engine.md:103:read-only database snapshot and performs no lifecycle action.
docs/architecture/Volume_VIII_Recommendation_Engine.md:108:eligibility, and the invariant absence of follow-up authority separate.
docs/architecture/Volume_VIII_Recommendation_Engine.md:109:Phase 6A validates exact Step 5A/5C canonical and direct identity alignment and
docs/architecture/Volume_VIII_Recommendation_Engine.md:110:builds a deterministic follow-up proposal only from an eligible favorable,
docs/architecture/Volume_VIII_Recommendation_Engine.md:112:operator-review, persistence, activation, execution, or scheduler behavior.
docs/architecture/Volume_VIII_Recommendation_Engine.md:113:Phase 6B stores that already-built value without recomputing it, reloads the
docs/architecture/Volume_VIII_Recommendation_Engine.md:116:Phase 6C binds an explicit operator, reason, and approved/rejected decision to
docs/architecture/Volume_VIII_Recommendation_Engine.md:119:Phase 6D loads one exact Phase 6C event and permits only its ``approved``
docs/architecture/Volume_VIII_Recommendation_Engine.md:120:outcome to receive one separately identified append-only ``ratified``
docs/architecture/Volume_VIII_Recommendation_Engine.md:122:``follow_up_governance_ratifier`` role. It adds no Phase 6E, activation, or
docs/architecture/Volume_VIII_Recommendation_Engine.md:126:## 3. Responsibilities
docs/architecture/Volume_VIII_Recommendation_Engine.md:128:### 3.1 Owned responsibilities
docs/architecture/Volume_VIII_Recommendation_Engine.md:130:Own deterministic proposal identity/generation, authoritative persistence and
docs/architecture/Volume_VIII_Recommendation_Engine.md:134:### 3.2 Dependencies
docs/architecture/Volume_VIII_Recommendation_Engine.md:138:experiment lifecycle; Phase 5 Steps 1–3 are confirmed orchestration callers
docs/architecture/Volume_VIII_Recommendation_Engine.md:141:and policy consume persisted evidence but have no lifecycle dependency or
docs/architecture/Volume_VIII_Recommendation_Engine.md:142:authority.
docs/architecture/Volume_VIII_Recommendation_Engine.md:143:The Phase 6A proposal consumes only their immutable typed values and has no
docs/architecture/Volume_VIII_Recommendation_Engine.md:144:database, CLI, service, repository, lifecycle, or scheduler dependency. The
docs/architecture/Volume_VIII_Recommendation_Engine.md:145:separate Phase 6B repository depends on PostgreSQL only to preserve and preview
docs/architecture/Volume_VIII_Recommendation_Engine.md:146:that exact value; it has no lifecycle or scheduler dependency.
docs/architecture/Volume_VIII_Recommendation_Engine.md:147:The Phase 6C contract depends only on the Phase 6A proposal identity and its
docs/architecture/Volume_VIII_Recommendation_Engine.md:148:persisted Phase 6B ID. Its repository depends on PostgreSQL only to verify and
docs/architecture/Volume_VIII_Recommendation_Engine.md:150:The Phase 6D contract depends only on the exact persisted Phase 6C review and
docs/architecture/Volume_VIII_Recommendation_Engine.md:151:its embedded Phase 6B proposal identity. Its repository and service use
docs/architecture/Volume_VIII_Recommendation_Engine.md:152:PostgreSQL only to verify and append governance-ratification evidence.
docs/architecture/Volume_VIII_Recommendation_Engine.md:154:### 3.3 Prohibited responsibilities
docs/architecture/Volume_VIII_Recommendation_Engine.md:158:scheduler state, infer reviewer identity, use
docs/architecture/Volume_VIII_Recommendation_Engine.md:162:Phase 6A eligibility MUST NOT be treated as operator approval, follow-up
docs/architecture/Volume_VIII_Recommendation_Engine.md:164:scheduler work.
docs/architecture/Volume_VIII_Recommendation_Engine.md:165:Phase 6B persistence and preview MUST NOT be treated as operator approval,
docs/architecture/Volume_VIII_Recommendation_Engine.md:166:activation, execution, queueing, scheduling, or follow-up authorization.
docs/architecture/Volume_VIII_Recommendation_Engine.md:167:Phase 6C approval MUST NOT be treated as activation, execution authorization,
docs/architecture/Volume_VIII_Recommendation_Engine.md:168:follow-up authorization, queueing, scheduling, scheduler work, worker launch,
docs/architecture/Volume_VIII_Recommendation_Engine.md:171:Phase 6D governance ratification MUST NOT be treated as Phase 6E capability,
docs/architecture/Volume_VIII_Recommendation_Engine.md:172:activation, execution authorization, follow-up authorization, queueing,
docs/architecture/Volume_VIII_Recommendation_Engine.md:173:scheduling, scheduler work, worker launch, experiment/model creation or
docs/architecture/Volume_VIII_Recommendation_Engine.md:177:## 4. Architecture
docs/architecture/Volume_VIII_Recommendation_Engine.md:179:### 4.1 Components
docs/architecture/Volume_VIII_Recommendation_Engine.md:182:- Pure identity-bound Phase 6A follow-up-proposal domain component.
docs/architecture/Volume_VIII_Recommendation_Engine.md:183:- Phase 6B append-only proposal repository and read-only preview component.
docs/architecture/Volume_VIII_Recommendation_Engine.md:184:- Pure immutable Phase 6C review contract, append-only event repository, and
docs/architecture/Volume_VIII_Recommendation_Engine.md:186:- Pure immutable Phase 6D ratification contract, append-only event repository,
docs/architecture/Volume_VIII_Recommendation_Engine.md:187:  and transactional ratification service.
docs/architecture/Volume_VIII_Recommendation_Engine.md:194:- CLI parsing/validation/presentation outside scheduler polling.
docs/architecture/Volume_VIII_Recommendation_Engine.md:196:### 4.2 Control flow
docs/architecture/Volume_VIII_Recommendation_Engine.md:212:-> explicit approved-campaign materialization to Phase 4C proposals
docs/architecture/Volume_VIII_Recommendation_Engine.md:217:-> existing scheduler lifecycle for ordinary pending experiments
docs/architecture/Volume_VIII_Recommendation_Engine.md:221:-> optional pure Phase 6A follow-up proposal for later operator review
docs/architecture/Volume_VIII_Recommendation_Engine.md:222:-> optional exact Phase 6B persistence and read-only operator preview
docs/architecture/Volume_VIII_Recommendation_Engine.md:223:-> optional explicit Phase 6C administrative approval or rejection
docs/architecture/Volume_VIII_Recommendation_Engine.md:224:-> optional independent Phase 6D governance ratification of advancement
docs/architecture/Volume_VIII_Recommendation_Engine.md:233:authorities inside one transaction; it does not call either public command.
docs/architecture/Volume_VIII_Recommendation_Engine.md:234:Phase 5 Step 4 reads the immutable materialization and bounded downstream
docs/architecture/Volume_VIII_Recommendation_Engine.md:235:evidence only; it neither invokes Steps 1–3 nor contacts the scheduler process.
docs/architecture/Volume_VIII_Recommendation_Engine.md:236:Step 5b reuses that status lifecycle inside one read-only snapshot. Step 5c
docs/architecture/Volume_VIII_Recommendation_Engine.md:238:assessment. Phase 6A consumes only the exact immutable assessment and policy
docs/architecture/Volume_VIII_Recommendation_Engine.md:239:decision and terminates at a non-authorizing proposal value. Phase 6B may
docs/architecture/Volume_VIII_Recommendation_Engine.md:241:arrow. Phase 6C may record only one administrative decision and likewise adds
docs/architecture/Volume_VIII_Recommendation_Engine.md:242:no downstream action arrow. Phase 6D may record only one separate governance
docs/architecture/Volume_VIII_Recommendation_Engine.md:243:ratification of advancement after an eligible Phase 6C review and also adds no
docs/architecture/Volume_VIII_Recommendation_Engine.md:244:downstream action arrow or Phase 6E capability.
docs/architecture/Volume_VIII_Recommendation_Engine.md:246:### 4.3 Ownership boundaries
docs/architecture/Volume_VIII_Recommendation_Engine.md:249:pure scoring, planning, campaign-review, outcome-policy, and follow-up-proposal
docs/architecture/Volume_VIII_Recommendation_Engine.md:254:planning, review, approval, outcome assessment, outcome policy, Phase 6A
docs/architecture/Volume_VIII_Recommendation_Engine.md:255:follow-up proposal, Phase 6B persistence/preview, and Phase 6C administrative
docs/architecture/Volume_VIII_Recommendation_Engine.md:256:review/presentation, and Phase 6D governance ratification never do.
docs/architecture/Volume_VIII_Recommendation_Engine.md:258:## 5. Data model
docs/architecture/Volume_VIII_Recommendation_Engine.md:260:### 5.1 Authoritative entities
docs/architecture/Volume_VIII_Recommendation_Engine.md:267:an experiment-lifecycle entity. Campaign plans and reviews are reconstructed,
docs/architecture/Volume_VIII_Recommendation_Engine.md:271:evidence but create no authoritative campaign decision.
docs/architecture/Volume_VIII_Recommendation_Engine.md:272:Phase 6A follow-up proposals are advisory values whose canonical identities
docs/architecture/Volume_VIII_Recommendation_Engine.md:274:materialization, ordered members, eligibility, and fixed non-authority
docs/architecture/Volume_VIII_Recommendation_Engine.md:275:semantics while excluding observation time. Phase 6B persists that exact value
docs/architecture/Volume_VIII_Recommendation_Engine.md:278:The Phase 6C review is a separate immutable administrative entity binding the
docs/architecture/Volume_VIII_Recommendation_Engine.md:280:reviewer, and reason. Its canonical text is authoritative. Review-event ID and
docs/architecture/Volume_VIII_Recommendation_Engine.md:282:The Phase 6D ratification is a separate immutable governance entity. It binds
docs/architecture/Volume_VIII_Recommendation_Engine.md:283:the Phase 6C event ID and exact review version/canonical/hash, reviewer, exact
docs/architecture/Volume_VIII_Recommendation_Engine.md:284:reviewed proposal identity, approved eligibility, fixed ratifier role,
docs/architecture/Volume_VIII_Recommendation_Engine.md:285:``ratified`` decision, distinct ratifier, basis, separation policy, and fixed
docs/architecture/Volume_VIII_Recommendation_Engine.md:289:### 5.2 Provenance and versions
docs/architecture/Volume_VIII_Recommendation_Engine.md:291:Canonical text is authoritative; tagged hashes accelerate lookup. Persisted
docs/architecture/Volume_VIII_Recommendation_Engine.md:295:### 5.3 Invariants and legacy data
docs/architecture/Volume_VIII_Recommendation_Engine.md:306:## 6. Transactions
docs/architecture/Volume_VIII_Recommendation_Engine.md:308:### 6.1 Read paths
docs/architecture/Volume_VIII_Recommendation_Engine.md:314:### 6.2 Write paths
docs/architecture/Volume_VIII_Recommendation_Engine.md:325:canonical text alone remains authoritative.
docs/architecture/Volume_VIII_Recommendation_Engine.md:326:Phase 6B proposal insertion locks only the proposal-hash bucket, compares
docs/architecture/Volume_VIII_Recommendation_Engine.md:327:authoritative canonical text exactly, inserts the manifest and ordered members
docs/architecture/Volume_VIII_Recommendation_Engine.md:329:not write any review, lifecycle, experiment, or scheduler table.
docs/architecture/Volume_VIII_Recommendation_Engine.md:330:Phase 6C review insertion locks only the proposal-ID conflict domain, reloads
docs/architecture/Volume_VIII_Recommendation_Engine.md:331:and compares the exact Phase 6B proposal identity, and inserts one event. Exact
docs/architecture/Volume_VIII_Recommendation_Engine.md:332:replay returns the event; changed decision, reviewer, reason, or identity
docs/architecture/Volume_VIII_Recommendation_Engine.md:334:Phase 6D ratification insertion locks only the review-event-ID conflict domain,
docs/architecture/Volume_VIII_Recommendation_Engine.md:335:reloads the exact Phase 6C review and Phase 6B proposal identity, enforces an
docs/architecture/Volume_VIII_Recommendation_Engine.md:336:approved review, fixed role, and distinct reviewer/ratifier, and inserts one
docs/architecture/Volume_VIII_Recommendation_Engine.md:337:event. Exact replay returns the event; any changed payload conflicts. It never
docs/architecture/Volume_VIII_Recommendation_Engine.md:338:updates Phase 6A, 6B, or 6C evidence.
docs/architecture/Volume_VIII_Recommendation_Engine.md:340:### 6.3 Failure semantics
docs/architecture/Volume_VIII_Recommendation_Engine.md:348:Malformed Phase 6C input, stored canonical/hash disagreement, Phase 6B
docs/architecture/Volume_VIII_Recommendation_Engine.md:350:Malformed Phase 6D input, missing/rejected/stale reviews, any complete-chain
docs/architecture/Volume_VIII_Recommendation_Engine.md:354:## 7. Concurrency
docs/architecture/Volume_VIII_Recommendation_Engine.md:356:### 7.1 Conflict domain
docs/architecture/Volume_VIII_Recommendation_Engine.md:364:### 7.2 Locking and serialization
docs/architecture/Volume_VIII_Recommendation_Engine.md:374:### 7.3 Winner, loser, and retry outcomes
docs/architecture/Volume_VIII_Recommendation_Engine.md:383:Concurrent identical Phase 6B proposal persistence converges on one exact
docs/architecture/Volume_VIII_Recommendation_Engine.md:386:Concurrent identical Phase 6C reviews converge on one event with one
docs/architecture/Volume_VIII_Recommendation_Engine.md:389:Concurrent identical Phase 6D ratifications converge on one event with one
docs/architecture/Volume_VIII_Recommendation_Engine.md:391:ratifications for the same review yield one recorded winner and one
docs/architecture/Volume_VIII_Recommendation_Engine.md:394:## 8. CLI
docs/architecture/Volume_VIII_Recommendation_Engine.md:396:### 8.1 Commands and validation
docs/architecture/Volume_VIII_Recommendation_Engine.md:401:are mutually exclusive with generation, scoring, scheduler, continuation, and
docs/architecture/Volume_VIII_Recommendation_Engine.md:402:experiment-lifecycle commands.
docs/architecture/Volume_VIII_Recommendation_Engine.md:404:### 8.2 Machine output
docs/architecture/Volume_VIII_Recommendation_Engine.md:409:### 8.3 Human output
docs/architecture/Volume_VIII_Recommendation_Engine.md:414:Phase 6B preview includes the full authoritative proposal identity, provenance,
docs/architecture/Volume_VIII_Recommendation_Engine.md:415:ordered members, and explicit false approval/activation/execution/scheduler
docs/architecture/Volume_VIII_Recommendation_Engine.md:417:Phase 6C show/list includes exact proposal/review identities, decision,
docs/architecture/Volume_VIII_Recommendation_Engine.md:419:follow-up, queue, schedule, scheduler, worker, experiment, and success fields.
docs/architecture/Volume_VIII_Recommendation_Engine.md:420:Phase 6D intentionally adds no main-program CLI or presentation. Its typed
docs/architecture/Volume_VIII_Recommendation_Engine.md:424:## 9. Testing
docs/architecture/Volume_VIII_Recommendation_Engine.md:426:### 9.1 Pure tests
docs/architecture/Volume_VIII_Recommendation_Engine.md:436:Phase 6A pure tests additionally cover exact upstream alignment, stable refusal
docs/architecture/Volume_VIII_Recommendation_Engine.md:437:reasons, golden canonical identity, all non-authority flags, upstream-order,
docs/architecture/Volume_VIII_Recommendation_Engine.md:441:Phase 6B focused tests cover exact round trip, immutable typed reload,
docs/architecture/Volume_VIII_Recommendation_Engine.md:442:canonical/hash preservation, exact retry, duplicate persisted identity,
docs/architecture/Volume_VIII_Recommendation_Engine.md:444:Phase 6C pure tests cover immutable types, approved/rejected construction,
docs/architecture/Volume_VIII_Recommendation_Engine.md:446:sensitivity, malformed input, and fixed negative authority.
docs/architecture/Volume_VIII_Recommendation_Engine.md:447:Phase 6D pure tests cover immutable types, approved-only construction,
docs/architecture/Volume_VIII_Recommendation_Engine.md:448:``ratified`` decision, fixed role, separation of duties, golden canonical/hash,
docs/architecture/Volume_VIII_Recommendation_Engine.md:450:negative authority.
docs/architecture/Volume_VIII_Recommendation_Engine.md:452:### 9.2 Persistence and migration tests
docs/architecture/Volume_VIII_Recommendation_Engine.md:455:immutability, review constraints, runtime privileges, SQLSTATEs, and migration
docs/architecture/Volume_VIII_Recommendation_Engine.md:456:repeatability. Evaluation tests additionally cover canonical idempotency,
docs/architecture/Volume_VIII_Recommendation_Engine.md:457:append-only result/component privileges, nullable blocked scores, and evidence
docs/architecture/Volume_VIII_Recommendation_Engine.md:459:narrow privileges, scope/captured-membership trigger enforcement, bounded
docs/architecture/Volume_VIII_Recommendation_Engine.md:462:hash collisions, concurrent insertion, restrictive privileges, and unchanged
docs/architecture/Volume_VIII_Recommendation_Engine.md:464:Phase 6B migration/repository tests cover bytewise canonical storage,
docs/architecture/Volume_VIII_Recommendation_Engine.md:466:narrow append-only privileges, migration repeatability, malformed persistence
docs/architecture/Volume_VIII_Recommendation_Engine.md:468:Phase 6C migration/repository tests cover repeatability, bytewise canonical
docs/architecture/Volume_VIII_Recommendation_Engine.md:469:storage, restrictive provenance, narrow privileges, generated-column
docs/architecture/Volume_VIII_Recommendation_Engine.md:470:protection, exact round trip/replay, all conflict forms, concurrent identical
docs/architecture/Volume_VIII_Recommendation_Engine.md:472:Phase 6B/experiment/scheduler fixtures, and read-only presentation without
docs/architecture/Volume_VIII_Recommendation_Engine.md:474:Phase 6D migration/repository/service tests cover clean and upgrade paths,
docs/architecture/Volume_VIII_Recommendation_Engine.md:476:eligibility, missing/stale/rejected review refusal, exact round trip/replay,
docs/architecture/Volume_VIII_Recommendation_Engine.md:478:read-only lookup/list behavior, and unchanged Phase 6B/6C/experiment/scheduler
docs/architecture/Volume_VIII_Recommendation_Engine.md:481:### 9.3 Concurrency and integration tests
docs/architecture/Volume_VIII_Recommendation_Engine.md:487:### 9.4 Regression boundaries
docs/architecture/Volume_VIII_Recommendation_Engine.md:490:inference, continuation, experiment lifecycle, or scheduler behavior.
docs/architecture/Volume_VIII_Recommendation_Engine.md:492:## 10. Operational safety
docs/architecture/Volume_VIII_Recommendation_Engine.md:494:### 10.1 Runtime isolation
docs/architecture/Volume_VIII_Recommendation_Engine.md:496:No recommendation command enters scheduler polling. Integration uses disposable
docs/architecture/Volume_VIII_Recommendation_Engine.md:499:### 10.2 Permissions and destructive operations
docs/architecture/Volume_VIII_Recommendation_Engine.md:502:SELECT/INSERT only; evaluation-run updates are limited to lifecycle columns.
docs/architecture/Volume_VIII_Recommendation_Engine.md:504:lifecycle/count columns. Owner/test connections perform exact fixture cleanup.
docs/architecture/Volume_VIII_Recommendation_Engine.md:511:authoritative even when concurrent transactions commit in another order.
docs/architecture/Volume_VIII_Recommendation_Engine.md:516:Phase 6C review events likewise grant only ``SELECT``, payload-column
docs/architecture/Volume_VIII_Recommendation_Engine.md:519:only and is not consumed by any Phase 6C action path.
docs/architecture/Volume_VIII_Recommendation_Engine.md:520:Phase 6D ratification events use the same narrow table/sequence privilege shape,
docs/architecture/Volume_VIII_Recommendation_Engine.md:522:trigger verifies complete Phase 6C/6B provenance and separation of duties.
docs/architecture/Volume_VIII_Recommendation_Engine.md:525:### 10.3 Observability and recovery
docs/architecture/Volume_VIII_Recommendation_Engine.md:530:## 11. Future extensions
docs/architecture/Volume_VIII_Recommendation_Engine.md:532:### 11.1 Approved extension points
docs/architecture/Volume_VIII_Recommendation_Engine.md:537:### 11.2 Deferred capabilities
docs/architecture/Volume_VIII_Recommendation_Engine.md:540:automatic research campaigns, profitability evidence, and scheduler-managed
docs/architecture/Volume_VIII_Recommendation_Engine.md:546:the existing scheduler lifecycle.
docs/architecture/Volume_VIII_Recommendation_Engine.md:548:### 11.3 Required decisions
docs/architecture/Volume_VIII_Recommendation_Engine.md:551:idempotent experiment creation, transactions, and audit history. Approval alone
docs/architecture/Volume_VIII_Recommendation_Engine.md:552:does not invoke conversion. Budgets, queueing, execution, and scheduler capacity
docs/architecture/Volume_VIII_Recommendation_Engine.md:557:repository. Exact canonical text remains authoritative; same-hash distinct
docs/architecture/Volume_VIII_Recommendation_Engine.md:561:access is limited to ``SELECT``/``INSERT``; no scheduler path reads these rows.
docs/architecture/Volume_VIII_Recommendation_Engine.md:565:idempotent per proposal. New request tokens permit auditable reversals; the
docs/architecture/Volume_VIII_Recommendation_Engine.md:574:references make retries and concurrent requests converge. The scheduler does
docs/architecture/Volume_VIII_Recommendation_Engine.md:581:lifecycle update are atomic. Exact retries return the same activation without
docs/architecture/Volume_VIII_Recommendation_Engine.md:582:reapplying the transition. The scheduler is unchanged and never reads the
docs/architecture/Volume_VIII_Recommendation_Engine.md:587:current experiment lifecycle. Stable states and deterministic diagnostics make
docs/architecture/Volume_VIII_Recommendation_Engine.md:588:healthy and inconsistent chains inspectable without adding schema, privileges,
docs/architecture/Volume_VIII_Recommendation_Engine.md:589:workflow mutation, scheduler polling, or worker behavior.
docs/architecture/Volume_VIII_Recommendation_Engine.md:598:reasons while excluding display time. There is no campaign table or scheduler
docs/architecture/Volume_VIII_Recommendation_Engine.md:599:consumer. Because no authoritative profitability evidence exists, a requested
docs/architecture/Volume_VIII_Recommendation_Engine.md:609:privilege, persisted campaign, workflow mutation, scheduler consumer, or worker
docs/architecture/Volume_VIII_Recommendation_Engine.md:616:Canonical review text is authoritative; identical retries converge, changed
docs/architecture/Volume_VIII_Recommendation_Engine.md:618:only be rejected. Runtime privileges are append-only and column-limited. The
docs/architecture/Volume_VIII_Recommendation_Engine.md:620:the scheduler.
docs/architecture/Volume_VIII_Recommendation_Engine.md:622:Phase 4D Step 4 adds one explicit, atomic materialization transaction for one
docs/architecture/Volume_VIII_Recommendation_Engine.md:626:4C proposals. Canonical text is authoritative and exact replay converges;
docs/architecture/Volume_VIII_Recommendation_Engine.md:632:experiment, or contact the scheduler or workers.
docs/architecture/Volume_VIII_Recommendation_Engine.md:635:Step 4 manifest and ordered members remain membership authority; greatest-ID
docs/architecture/Volume_VIII_Recommendation_Engine.md:636:Phase 4C review, execution, activation, and experiment lifecycle evidence are
docs/architecture/Volume_VIII_Recommendation_Engine.md:640:scheduler polling, or worker behavior.
docs/architecture/Volume_VIII_Recommendation_Engine.md:648:fails closed. No Phase 4D review authority or schema is added, and execution,
docs/architecture/Volume_VIII_Recommendation_Engine.md:658:execution fails closed. There is no new schema or execution authority, and no
docs/architecture/Volume_VIII_Recommendation_Engine.md:659:activation, pending transition, scheduler action, worker launch, or automatic
docs/architecture/Volume_VIII_Recommendation_Engine.md:668:activation fails closed. No schema, campaign activation authority, scheduler
docs/architecture/Volume_VIII_Recommendation_Engine.md:669:action, worker launch, direct process, or automatic follow-up is added.
docs/architecture/Volume_VIII_Recommendation_Engine.md:680:schema, campaign lifecycle authority, scheduler action, worker launch, direct
docs/architecture/Volume_VIII_Recommendation_Engine.md:681:process, or automatic follow-up is added.
docs/architecture/Volume_VIII_Recommendation_Engine.md:684:membership, lifecycle consistency, source/result provenance, comparison
docs/architecture/Volume_VIII_Recommendation_Engine.md:685:context, and metric deltas. Step 5b loads authoritative persisted campaign and
docs/architecture/Volume_VIII_Recommendation_Engine.md:687:deterministic, non-authoritative, point-in-time assessment from it. Neither step
docs/architecture/Volume_VIII_Recommendation_Engine.md:688:persists an assessment, declares campaign success, or authorizes follow-up.
docs/architecture/Volume_VIII_Recommendation_Engine.md:696:campaigns are not. Its result is non-persistent and non-authoritative, always
docs/architecture/Volume_VIII_Recommendation_Engine.md:697:records follow-up authorization as false, and adds no repository, CLI,
docs/architecture/Volume_VIII_Recommendation_Engine.md:698:scheduler, worker, schema, or lifecycle behavior.
docs/architecture/Volume_VIII_Recommendation_Engine.md:700:Phase 6A adds the distinct
docs/architecture/Volume_VIII_Recommendation_Engine.md:703:It validates authoritative canonical text as well as hashes and exact campaign,
docs/architecture/Volume_VIII_Recommendation_Engine.md:704:materialization, count, and ordered-member alignment. Only an eligible,
docs/architecture/Volume_VIII_Recommendation_Engine.md:709:execution, scheduler work, success evidence, or authority for later phases.
docs/architecture/Volume_VIII_Recommendation_Engine.md:711:Phase 6B persists only that already-built immutable proposal. One append-only
docs/architecture/Volume_VIII_Recommendation_Engine.md:712:manifest and ordered member set preserve every Phase 6A field. Exact canonical
docs/architecture/Volume_VIII_Recommendation_Engine.md:713:text remains authoritative, collision ordinals and timestamps stay outside
docs/architecture/Volume_VIII_Recommendation_Engine.md:714:identity, and reload reruns the Phase 6A payload/canonical invariants. Preview
docs/architecture/Volume_VIII_Recommendation_Engine.md:717:experiment mutation state. Phase 6B adds no operator decision or action path.
docs/architecture/Volume_VIII_Recommendation_Engine.md:719:Phase 6C records only one immutable administrative ``approved`` or ``rejected``
docs/architecture/Volume_VIII_Recommendation_Engine.md:720:event for one exact persisted Phase 6B proposal. The review canonical binds the
docs/architecture/Volume_VIII_Recommendation_Engine.md:722:negative action semantics. Exact retry is idempotent; every changed payload for
docs/architecture/Volume_VIII_Recommendation_Engine.md:724:does not activate, execute, authorize follow-up, queue, schedule, contact the
docs/architecture/Volume_VIII_Recommendation_Engine.md:725:scheduler, launch workers, create/modify experiments, or declare success.
docs/architecture/Volume_VIII_Recommendation_Engine.md:727:Phase 6D separately records one immutable ``ratified`` governance event for
docs/architecture/Volume_VIII_Recommendation_Engine.md:728:advancement after one exact persisted Phase 6C approved review. Its canonical
docs/architecture/Volume_VIII_Recommendation_Engine.md:730:distinct ratifier, basis, and separation policy. Exact retry is idempotent;
docs/architecture/Volume_VIII_Recommendation_Engine.md:732:the authoritative upstream chain. Phase 6D adds no CLI, Phase 6E, activation,
docs/architecture/Volume_VIII_Recommendation_Engine.md:733:execution, follow-up authorization, scheduler, worker, experiment, model,
docs/architecture/Volume_VIII_Recommendation_Engine.md:736:## 12. References
docs/architecture/Volume_VIII_Recommendation_Engine.md:756:- [Approved campaign materialization](../Phase4DExperimentRecommendationCampaignMaterialization.rst)
docs/architecture/Volume_VIII_Recommendation_Engine.md:764:- [Phase 6A recommendation campaign follow-up proposal](../Phase6ARecommendationCampaignFollowUpProposal.rst)
docs/architecture/Volume_VIII_Recommendation_Engine.md:765:- [Phase 6B follow-up proposal persistence and preview](../Phase6BRecommendationCampaignFollowUpProposalPersistence.rst)
docs/architecture/Volume_VIII_Recommendation_Engine.md:766:- [Phase 6C follow-up proposal administrative review](../Phase6CRecommendationCampaignFollowUpProposalReview.rst)
docs/architecture/Volume_VIII_Recommendation_Engine.md:767:- [Phase 6D follow-up proposal governance ratification](../Phase6DRecommendationCampaignFollowUpProposalRatification.rst)
docs/architecture/Volume_VIII_Recommendation_Engine.md:768:- [ADR-0006](adr/ADR-0006-phase-6a-follow-up-proposal.md)
docs/architecture/Volume_VIII_Recommendation_Engine.md:769:- [ADR-0007](adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md)
docs/architecture/Volume_VIII_Recommendation_Engine.md:770:- [ADR-0008](adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md)
docs/architecture/Volume_VIII_Recommendation_Engine.md:771:- [ADR-0009](adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md)
docs/architecture/Volume_VIII_Recommendation_Engine.md:773:## 13. Revision history
docs/architecture/Volume_VIII_Recommendation_Engine.md:782:| 0.6.0 | 2026-07-18 | Recorded append-only, idempotent Phase 4C Step 3 manual proposal review; approval remains non-executing. | ADR-0001, ADR-0003, ADR-0004 |
docs/architecture/Volume_VIII_Recommendation_Engine.md:783:| 0.7.0 | 2026-07-19 | Recorded explicit, idempotent Phase 4C Step 4 conversion to one paused experiment. | ADR-0001, ADR-0004, ADR-0005 |
docs/architecture/Volume_VIII_Recommendation_Engine.md:784:| 0.8.0 | 2026-07-19 | Recorded explicit, atomic Phase 4C Step 5 activation of a converted experiment into the existing pending lifecycle. | ADR-0001, ADR-0004, ADR-0005 |
docs/architecture/Volume_VIII_Recommendation_Engine.md:789:| 1.3.0 | 2026-07-19 | Recorded atomic Phase 4D Step 4 materialization of one approved campaign into exact Phase 4C proposals without experiment or scheduler execution. | ADR-0001, ADR-0003, ADR-0004, ADR-0005 |
docs/architecture/Volume_VIII_Recommendation_Engine.md:790:| 1.4.0 | 2026-07-19 | Recorded read-only Phase 4D Step 5 projection of materialized campaign membership onto current Phase 4C lifecycle evidence. | ADR-0001, ADR-0003, ADR-0004, ADR-0005 |
docs/architecture/Volume_VIII_Recommendation_Engine.md:791:| 1.5.0 | 2026-07-19 | Recorded explicit atomic Phase 4D Step 6 review of exact materialized proposals through ordinary Phase 4C review rows. | ADR-0001, ADR-0003, ADR-0004, ADR-0005 |
docs/architecture/Volume_VIII_Recommendation_Engine.md:792:| 1.6.0 | 2026-07-19 | Recorded explicit atomic Phase 5 Step 1 execution of exact materialized proposals through the existing Phase 4C paused-conversion authority. | ADR-0001, ADR-0004, ADR-0005 |
docs/architecture/Volume_VIII_Recommendation_Engine.md:793:| 1.7.0 | 2026-07-19 | Recorded explicit atomic Phase 5 Step 2 activation of exact materialized executions through the existing Phase 4C pending-transition authority. | ADR-0001, ADR-0004, ADR-0005 |
docs/architecture/Volume_VIII_Recommendation_Engine.md:794:| 1.8.0 | 2026-07-19 | Recorded explicit one-transaction Phase 5 Step 3 launch through the existing Phase 4C execution and activation authorities. | ADR-0001, ADR-0004, ADR-0005 |
docs/architecture/Volume_VIII_Recommendation_Engine.md:795:| 1.9.0 | 2026-07-19 | Recorded read-only Phase 5 Step 4 operational status for exact materialized members and current persisted lifecycle evidence. | ADR-0001, ADR-0004, ADR-0005 |
docs/architecture/Volume_VIII_Recommendation_Engine.md:796:| 2.0.0 | 2026-07-20 | Recorded Phase 5 Steps 5a–5b read-only outcome assessment and Step 5c pure advisory outcome policy with no persistence or follow-up authority. | ADR-0001, ADR-0003, ADR-0004, ADR-0005 |
docs/architecture/Volume_VIII_Recommendation_Engine.md:797:| 2.1.0 | 2026-07-20 | Recorded the pure, exact-identity-bound, non-authorizing Phase 6A follow-up proposal; later persistence, review, activation, and execution remain deferred. | ADR-0003, ADR-0004, ADR-0006 |
docs/architecture/Volume_VIII_Recommendation_Engine.md:798:| 2.2.0 | 2026-07-20 | Recorded exact append-only Phase 6B proposal persistence, immutable validated reload, and read-only preview without operator-decision or action authority. | ADR-0001, ADR-0004, ADR-0006, ADR-0007 |
docs/architecture/Volume_VIII_Recommendation_Engine.md:799:| 2.3.0 | 2026-07-21 | Recorded exact append-only Phase 6C approved/rejected administrative review, deterministic replay/conflict, and read-only presentation without action authority. | ADR-0001, ADR-0004, ADR-0006, ADR-0007, ADR-0008 |
docs/architecture/Volume_VIII_Recommendation_Engine.md:800:| 2.4.0 | 2026-07-22 | Recorded proposed append-only Phase 6D governance ratification after one eligible Phase 6C review, mandatory separation of duties, and deterministic replay/conflict without Phase 6E authority. | ADR-0001, ADR-0004, ADR-0006, ADR-0007, ADR-0008, ADR-0009 |
docs/architecture/adr/ADR-0005-manual-recommendation-conversion.md:1:# ADR-0005: Manual recommendation conversion creates a paused experiment
docs/architecture/adr/ADR-0005-manual-recommendation-conversion.md:10:## 1. Context and problem statement
docs/architecture/adr/ADR-0005-manual-recommendation-conversion.md:13:review history. A later explicit operation must be able to materialize an
docs/architecture/adr/ADR-0005-manual-recommendation-conversion.md:15:experiments under retries, or making the scheduler an implicit consumer.
docs/architecture/adr/ADR-0005-manual-recommendation-conversion.md:17:## 2. Decision
docs/architecture/adr/ADR-0005-manual-recommendation-conversion.md:22:- Proposal integrity is revalidated from its authoritative canonical evidence.
docs/architecture/adr/ADR-0005-manual-recommendation-conversion.md:30:- The experiment is not pending, queued, running, resumed, or scheduler-owned.
docs/architecture/adr/ADR-0005-manual-recommendation-conversion.md:31:- A later explicit lifecycle command is required before any execution.
docs/architecture/adr/ADR-0005-manual-recommendation-conversion.md:35:## 3. Rationale and decision drivers
docs/architecture/adr/ADR-0005-manual-recommendation-conversion.md:37:- Keep approval and materialization as separate durable events.
docs/architecture/adr/ADR-0005-manual-recommendation-conversion.md:42:## 4. Consequences
docs/architecture/adr/ADR-0005-manual-recommendation-conversion.md:44:### 4.1 Positive consequences
docs/architecture/adr/ADR-0005-manual-recommendation-conversion.md:47:- Operators can inspect the created experiment before taking lifecycle action.
docs/architecture/adr/ADR-0005-manual-recommendation-conversion.md:50:### 4.2 Negative consequences and trade-offs
docs/architecture/adr/ADR-0005-manual-recommendation-conversion.md:57:### 4.3 Risks and mitigations
docs/architecture/adr/ADR-0005-manual-recommendation-conversion.md:66:## 5. Compatibility and migration
docs/architecture/adr/ADR-0005-manual-recommendation-conversion.md:69:composite review/proposal uniqueness index. It does not change scheduler,
docs/architecture/adr/ADR-0005-manual-recommendation-conversion.md:72:## 6. Verification and operational evidence
docs/architecture/adr/ADR-0005-manual-recommendation-conversion.md:74:- Migration constraints, restrictive foreign keys, and runtime privileges.
docs/architecture/adr/ADR-0005-manual-recommendation-conversion.md:76:- Concurrent identical conversion and exact replay tests.
docs/architecture/adr/ADR-0005-manual-recommendation-conversion.md:80:## 7. Alternatives considered
docs/architecture/adr/ADR-0005-manual-recommendation-conversion.md:82:### 7.1 Create a pending experiment
docs/architecture/adr/ADR-0005-manual-recommendation-conversion.md:84:Rejected because scheduler startup could then execute conversion output without
docs/architecture/adr/ADR-0005-manual-recommendation-conversion.md:87:### 7.2 Convert as part of approval
docs/architecture/adr/ADR-0005-manual-recommendation-conversion.md:92:### 7.3 Queue immediately after conversion
docs/architecture/adr/ADR-0005-manual-recommendation-conversion.md:97:## 8. References
docs/architecture/adr/ADR-0005-manual-recommendation-conversion.md:101:- [ADR-0004](ADR-0004-scheduler-ownership-boundaries.md)
docs/architecture/adr/ADR-0005-manual-recommendation-conversion.md:103:## 9. Revision history
docs/architecture/adr/ADR-0005-manual-recommendation-conversion.md:107:| 2026-07-19 | Accepted explicit, idempotent conversion to one paused experiment. |
docs/architecture/adr/README.md:1:# Architecture Decision Records
docs/architecture/adr/README.md:3:## 1. Purpose
docs/architecture/adr/README.md:10:## 2. Numbering and filenames
docs/architecture/adr/README.md:18:## 3. Status values
docs/architecture/adr/README.md:22:| `Proposed` | Under review; not implementation authority. |
docs/architecture/adr/README.md:32:## 4. Changes requiring an ADR
docs/architecture/adr/README.md:34:Volume I §16.1 is authoritative. Typical triggers include identity semantics,
docs/architecture/adr/README.md:35:durable ownership, schema strategy, transaction/concurrency rules, scheduler
docs/architecture/adr/README.md:36:scope, lifecycle transitions, recommendation conversion, security/privileges,
docs/architecture/adr/README.md:39:## 5. Required sections
docs/architecture/adr/README.md:56:## 6. Proposal workflow
docs/architecture/adr/README.md:68:## 7. Index
docs/architecture/adr/README.md:75:| [ADR-0004](ADR-0004-scheduler-ownership-boundaries.md) | Accepted | Scheduler work requires explicit ownership and durable lifecycle claims. |
docs/architecture/adr/README.md:77:| [ADR-0006](ADR-0006-phase-6a-follow-up-proposal.md) | Accepted | Phase 6A follow-up proposals are pure, identity-bound, and non-authorizing. |
docs/architecture/adr/README.md:78:| [ADR-0007](ADR-0007-phase-6b-follow-up-proposal-persistence.md) | Accepted | Phase 6B persists exact advisory proposals for read-only preview without action authority. |
docs/architecture/adr/README.md:79:| [ADR-0008](ADR-0008-phase-6c-follow-up-proposal-administrative-review.md) | Accepted | Phase 6C records one immutable administrative proposal review without action authority. |
docs/architecture/adr/README.md:80:| [ADR-0009](ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md) | Proposed | Phase 6D separately ratifies governance advancement after an approved review, with mandatory separation of duties and no Phase 6E authority. |
docs/architecture/adr/README.md:82:## 8. Revision history
docs/architecture/adr/README.md:88:| 1.2.0 | 2026-07-20 | Added ADR-0006 for the pure non-authorizing Phase 6A follow-up proposal. |
docs/architecture/adr/README.md:89:| 1.3.0 | 2026-07-20 | Added ADR-0007 for exact append-only Phase 6B persistence and read-only preview. |
docs/architecture/adr/README.md:90:| 1.4.0 | 2026-07-21 | Added ADR-0008 for exact append-only Phase 6C administrative review without action authority. |
docs/architecture/adr/README.md:91:| 1.5.0 | 2026-07-22 | Proposed ADR-0009 for exact append-only Phase 6D governance ratification with mandatory separation of duties and no Phase 6E authority. |
docs/Phase4DExperimentRecommendationCampaignMaterialization.rst:11:not contact the scheduler or a worker.
docs/Phase4DExperimentRecommendationCampaignMaterialization.rst:36:materialization operator and reason; ordered selected-member identities; and
docs/Phase4DExperimentRecommendationCampaignMaterialization.rst:38:and excluded. Canonical text is authoritative; hashes are accelerators only.
docs/Phase4DExperimentRecommendationCampaignMaterialization.rst:40:Migration ``041_experiment_recommendation_campaign_materialization.sql`` adds
docs/Phase4DExperimentRecommendationCampaignMaterialization.rst:50:Atomicity, replay, and concurrency
docs/Phase4DExperimentRecommendationCampaignMaterialization.rst:55:new proposals and materialization evidence together. Pre-existing exact Phase
docs/Phase4DExperimentRecommendationCampaignMaterialization.rst:59:manifest is permitted per approval. Exact replay returns the existing manifest;
docs/Phase4DExperimentRecommendationCampaignMaterialization.rst:65:An exact retry first validates the existing manifest's authoritative canonical
docs/Phase4DExperimentRecommendationCampaignMaterialization.rst:67:operator request. It does not reconstruct the current plan on replay because
docs/Phase4DExperimentRecommendationCampaignMaterialization.rst:68:the proposals created by the original materialization are themselves durable
docs/Phase4DExperimentRecommendationCampaignMaterialization.rst:70:First-time materialization always performs the full stale-evidence
docs/Phase4DExperimentRecommendationCampaignMaterialization.rst:78:   LSTM_Release --materialize-recommendation-campaign \
docs/Phase4DExperimentRecommendationCampaignMaterialization.rst:80:     --campaign-materialized-by='operator@example' \
docs/Phase4DExperimentRecommendationCampaignMaterialization.rst:81:     --campaign-materialization-reason='Create exact Phase 4C proposal set.'
docs/Phase4DExperimentRecommendationCampaignMaterialization.rst:85:   LSTM_Release --show-recommendation-campaign-materialization=1
docs/Phase4DExperimentRecommendationCampaignMaterialization.rst:86:   LSTM_Release --list-recommendation-campaign-materializations \
docs/Phase4DExperimentRecommendationCampaignMaterialization.rst:87:     --campaign-approval-id=123 --campaign-materialization-limit=100
docs/Phase4DExperimentRecommendationCampaignMaterialization.rst:89:Output distinguishes a newly recorded manifest from exact replay and reports
docs/Phase4DExperimentRecommendationCampaignMaterialization.rst:92:scheduler, and workers as unchanged.
docs/Phase4DExperimentRecommendationCampaignMaterialization.rst:107:activation, background scans, scheduler integration, and worker launch remain
docs/Phase4DExperimentRecommendationCampaignMaterialization.rst:111:manifest membership and current Phase 4C lifecycle evidence. It does not alter
docs/Phase4DExperimentRecommendationCampaignMaterialization.rst:112:the materialization or advance any linked proposal.
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:1:# ADR-0006: Phase 6A follow-up proposal is advisory and identity-bound
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:10:## 1. Context and problem statement
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:15:an approval, authorization, persisted review, experiment, or scheduler work
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:16:item. The first Phase 6 increment needs a deterministic candidate that preserves
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:18:campaign materialization or silently reopening the scientific decision.
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:20:## 2. Decision
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:22:Phase 6A introduces the distinct pure domain entity
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:28:  materialization, ordered-member, count, and embedded-assessment alignment.
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:30:  favorable, and explicitly non-authorizing. Phase 6A does not recalculate
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:32:- Canonical text is authoritative identity. The tagged hash is only an
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:38:  non-authoritative, unapproved, inactive, non-executing, non-authorizing,
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:39:  non-scheduler work, and does not declare campaign success.
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:40:- Phase 6A adds no repository, SQL, migration, service, CLI, formatter,
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:41:  persisted operator review, scheduler consumer, queue, worker, or lifecycle
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:44:  later Phase 6 steps and receive no authority from this ADR.
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:46:## 3. Rationale and decision drivers
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:51:- Make collision handling safe by retaining authoritative canonical text.
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:53:  materialization or any experiment lifecycle request.
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:55:  authorities, transactions, idempotency, and concurrency contracts.
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:57:## 4. Consequences
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:59:### 4.1 Positive consequences
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:61:- Identical validated Phase 5 inputs yield an identical Phase 6A identity
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:63:- Every relevant upstream identity and explicit non-authority semantic is
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:67:- No database, scheduler, or execution dependency enters the pure boundary.
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:69:### 4.2 Negative consequences and trade-offs
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:75:- Phase 6A alone cannot record or enact an operator decision.
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:77:### 4.3 Risks and mitigations
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:79:- Hash collision: compare authoritative canonical text after any hash match.
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:81:  campaign, materialization, and ordered-member fields.
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:82:- Accidental authority creep: compile-time false safety flags, canonical
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:83:  bindings, pure tests, and absence of persistence/scheduler integration.
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:87:## 5. Compatibility and migration
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:90:schema, data, privilege, repository, CLI, runtime, or scheduler migration and
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:93:## 6. Verification and operational evidence
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:103:## 7. Alternatives considered
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:105:### 7.1 Generic materialization entity
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:108:that materializes an approved campaign into Phase 4C proposals.
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:110:### 7.2 Build from Step 5B formatted output
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:115:### 7.3 Recompute Step 5C eligibility in Phase 6A
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:120:### 7.4 Persist or approve the proposal in Phase 6A
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:123:execution require distinct later authority and operational designs.
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:125:## 8. References
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:129:- [Phase 6A follow-up proposal](../../Phase6ARecommendationCampaignFollowUpProposal.rst)
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:131:- [ADR-0004](ADR-0004-scheduler-ownership-boundaries.md)
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:134:## 9. Revision history
docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md:138:| 2026-07-20 | Accepted the pure identity-bound, non-authorizing Phase 6A follow-up proposal. |
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:1:# ADR-0008: Phase 6C records one immutable non-authorizing administrative review
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:10:## 1. Context and problem statement
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:12:Phase 6B durably preserves an exact Phase 6A advisory follow-up proposal but
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:15:execution, follow-up authorization, scheduling, queueing, experiment mutation,
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:20:## 2. Decision
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:22:Phase 6C defines one pure immutable review value, migration 043 with one
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:26:- Contract version 1 binds persisted proposal ID, exact Phase 6A
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:28:- Canonical text is authoritative and byte-length-prefixed; the tagged hash is
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:30:- The table has one restrictive Phase 6B foreign key, exact-provenance insert
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:34:  First persistence is ``recorded``, exact replay is ``existing_identical``,
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:37:  identity, reloads Phase 6B, and compares the exact proposal binding.
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:39:  ID, and render explicit negative authority fields.
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:46:## 3. Rationale and decision drivers
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:48:- Make the operator decision explicit without weakening the Phase 6A/6B
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:50:- Bind review to authoritative canonical proposal content rather than trusting
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:54:- Keep later action authority absent from both schema and API.
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:56:## 4. Consequences
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:58:### 4.1 Positive consequences
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:64:  and Phase 6B reload provide defense in depth.
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:67:### 4.2 Negative consequences and trade-offs
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:71:- Phase 6C cannot correct an operator mistake in place; a future explicitly
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:74:- List hydration reloads each referenced Phase 6B proposal to validate current
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:77:### 4.3 Risks and mitigations
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:79:- Authority creep: no action/lifecycle columns or adapters, plus fixed negative
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:83:- Hash collision or corruption: canonical equality is authoritative and all
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:86:  comparison against immutable Phase 6B storage.
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:88:## 5. Compatibility and migration
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:90:Migration 043 is additive after migration 042. It alters no Phase 6A/6B row,
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:91:experiment table, scheduler table, or lifecycle schema. Existing Phase 6A and
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:93:operationally approved, consists only of separately removing the new Phase 6C
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:96:## 6. Verification and operational evidence
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:100:  authority invariants.
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:102:  FK, least privilege, generated columns, exact round trip/replay, all conflict
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:107:- Phase 6A/6B, Phase 5, relevant Phase 4D, project validation, and Release
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:110:## 7. Alternatives considered
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:112:### 7.1 Add approved state to the Phase 6B proposal row
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:117:### 7.2 Treat persistence or preview as approval
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:122:### 7.3 Let approval activate or queue follow-up work
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:124:Rejected because action authority requires a later explicit contract,
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:127:### 7.4 Allow multiple current/superseding review states
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:129:Rejected for Phase 6C. One immutable effective decision gives deterministic
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:130:replay and conflict semantics without inventing expiration or correction
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:133:## 8. References
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:137:- [Phase 6A proposal](../../Phase6ARecommendationCampaignFollowUpProposal.rst)
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:138:- [Phase 6B persistence and preview](../../Phase6BRecommendationCampaignFollowUpProposalPersistence.rst)
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:139:- [Phase 6C administrative review](../../Phase6CRecommendationCampaignFollowUpProposalReview.rst)
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:141:- [ADR-0004](ADR-0004-scheduler-ownership-boundaries.md)
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:142:- [ADR-0006](ADR-0006-phase-6a-follow-up-proposal.md)
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:143:- [ADR-0007](ADR-0007-phase-6b-follow-up-proposal-persistence.md)
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:145:## 9. Revision history
docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md:149:| 2026-07-21 | Accepted exact append-only Phase 6C administrative review without action authority. |
docs/architecture/adr/ADR_TEMPLATE.md:1:# ADR-NNNN: Decision title
docs/architecture/adr/ADR_TEMPLATE.md:10:## 1. Context and problem statement
docs/architecture/adr/ADR_TEMPLATE.md:15:## 2. Decision
docs/architecture/adr/ADR_TEMPLATE.md:17:[State the decision in normative, testable language. Identify authoritative
docs/architecture/adr/ADR_TEMPLATE.md:20:## 3. Rationale and decision drivers
docs/architecture/adr/ADR_TEMPLATE.md:26:## 4. Consequences
docs/architecture/adr/ADR_TEMPLATE.md:28:### 4.1 Positive consequences
docs/architecture/adr/ADR_TEMPLATE.md:32:### 4.2 Negative consequences and trade-offs
docs/architecture/adr/ADR_TEMPLATE.md:36:### 4.3 Risks and mitigations
docs/architecture/adr/ADR_TEMPLATE.md:40:## 5. Compatibility and migration
docs/architecture/adr/ADR_TEMPLATE.md:46:## 6. Verification and operational evidence
docs/architecture/adr/ADR_TEMPLATE.md:51:## 7. Alternatives considered
docs/architecture/adr/ADR_TEMPLATE.md:53:### 7.1 Alternative name
docs/architecture/adr/ADR_TEMPLATE.md:57:## 8. References
docs/architecture/adr/ADR_TEMPLATE.md:63:## 9. Revision history
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:1:# ADR-0009: Phase 6D ratifies governance advancement after approved review
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:10:## 1. Context and problem statement
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:12:Phase 6C already records one immutable approved or rejected administrative
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:13:review of an exact persisted Phase 6B proposal. Its positive decision answers
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:15:Repeating that approval in Phase 6D would create no distinct authority.
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:19:provenance and ratified advancement into the next separately controlled
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:22:## 2. Proposed decision
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:24:Phase 6D defines pure immutable governance ratification, additive migration
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:27:- Phase 6C remains the merits review with `approved` or `rejected` outcomes.
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:28:- Phase 6D records only `ratified`: advancement of the exact approved review
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:30:- Ratifier identity MUST differ byte-for-byte from the Phase 6C reviewer
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:31:  identity. Self-ratification fails closed.
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:32:- The authority role is fixed as `follow_up_governance_ratifier`; callers do
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:36:  fixed role, ratified decision, ratifier, basis, separation rule, and fixed
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:37:  negative authority semantics.
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:38:- Canonical text is authoritative, byte-length framed, locale-independent,
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:40:- Migration 044 stores one ratification per review/proposal with restrictive
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:41:  provenance, approved eligibility, fixed-role, ratified-decision, and
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:44:  persistence is `recorded`, exact replay is `existing_identical`, and any
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:47:  `USAGE` only. Existing ratifications cannot be updated or removed.
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:48:- Phase 6D adds no CLI or operational adapter.
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:50:Ratification is governance evidence only. It grants no Phase 6E capability,
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:51:follow-up authorization, execution authorization, or lifecycle authority.
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:53:## 3. Rationale and decision drivers
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:55:- Give Phase 6D a materially distinct governance question and consequence.
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:57:- Consume authoritative persisted evidence without recomputing Phase 5 or
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:58:  Phase 6A truth.
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:63:## 4. Consequences
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:65:### 4.1 Positive consequences
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:68:  ratification of advancement.
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:69:- One actor cannot both approve the Phase 6C review and ratify it in Phase 6D.
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:71:  forged, malformed, or self-ratified evidence.
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:75:### 4.2 Negative consequences and trade-offs
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:77:- A second governance actor is required before a proposal can be ratified.
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:80:- A mistaken ratification cannot be changed in place; correction would require
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:84:### 4.3 Risks and mitigations
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:86:- Semantic duplication: distinct question, `ratified` decision, fixed role,
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:88:- Self-ratification: pure, service, repository, check-constraint, and trigger
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:90:- Hash collision: compare complete authoritative canonical and typed payload.
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:92:  validated Phase 6C repository reload.
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:96:## 5. Compatibility and migration
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:100:not alter Phase 6A, 6B, or 6C rows, identities, APIs, or privileges. It touches
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:101:no experiment, model, continuation, lifecycle, queue, or scheduler table.
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:103:## 6. Verification and operational evidence
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:107:  of duties, identity sensitivity, and negative authority.
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:110:  service syntax, replay/conflict, concurrency, rollback, malformed storage,
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:112:- Phase 6A/6B/6C regression tests, exact Xcode project membership, diff audit,
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:113:  and an incremental Release build only when scheduler-worker load is safe.
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:115:## 7. Alternatives considered
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:117:### 7.1 Keep a second administrative approval
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:119:Rejected because it repeats Phase 6C without a distinct question, role,
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:122:### 7.2 Allow the Phase 6C reviewer to ratify
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:124:Rejected because no committed authority permits self-ratification and it
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:125:would remove the independent governance check that distinguishes Phase 6D.
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:127:### 7.3 Treat Phase 6C approval as automatic ratification
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:132:### 7.4 Add Phase 6E activation or execution authority
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:137:## 8. References
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:141:- [Phase 6A proposal](../../Phase6ARecommendationCampaignFollowUpProposal.rst)
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:142:- [Phase 6B persistence](../../Phase6BRecommendationCampaignFollowUpProposalPersistence.rst)
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:143:- [Phase 6C review](../../Phase6CRecommendationCampaignFollowUpProposalReview.rst)
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:144:- [Phase 6D ratification](../../Phase6DRecommendationCampaignFollowUpProposalRatification.rst)
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:146:- [ADR-0004](ADR-0004-scheduler-ownership-boundaries.md)
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:147:- [ADR-0006](ADR-0006-phase-6a-follow-up-proposal.md)
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:148:- [ADR-0007](ADR-0007-phase-6b-follow-up-proposal-persistence.md)
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:149:- [ADR-0008](ADR-0008-phase-6c-follow-up-proposal-administrative-review.md)
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:151:## 9. Revision history
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:155:| 2026-07-22 | Proposed distinct Phase 6D governance ratification with mandatory separation of duties and no Phase 6E capability. |
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:1:# ADR-0007: Phase 6B persists exact advisory proposals for read-only preview
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:10:## 1. Context and problem statement
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:12:Phase 6A produces a pure immutable, non-authorizing follow-up proposal with a
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:15:introduce activation, execution, scheduling, experiment creation, or follow-up
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:16:authority. Reload must also reject database corruption rather than returning a
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:19:## 2. Decision
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:21:Phase 6B adds one append-only PostgreSQL manifest plus ordered member table,
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:25:- The manifest and members preserve every public Phase 6A field exactly.
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:26:- Canonical text is authoritative; the tagged hash only selects a lock/index
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:31:- Upstream approval/materialization/member triggers validate copied immutable
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:33:- Reload reconstructs the non-assignable Phase 6A object through one private
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:37:  negative authority fields.
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:38:- Runtime privileges are append-only and column-scoped. Phase 6B provides no
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:39:  approval, activation, execution, queue, scheduler, experiment, or follow-up
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:42:## 3. Rationale and decision drivers
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:44:- Preserve the identity produced by Phase 6A without parsing presentation
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:52:## 4. Consequences
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:54:### 4.1 Positive consequences
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:58:- Exact retries are idempotent and hash collisions do not merge identities.
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:59:- Database constraints, triggers, privileges, and C++ hydration provide
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:61:- Preview cannot mutate lifecycle or scheduler state.
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:63:### 4.2 Negative consequences and trade-offs
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:65:- Full canonical provenance is duplicated and may approach the Phase 6A 1 MiB
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:67:- Inserts depend on existing immutable Phase 4D approval/materialization rows.
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:70:- Previewing full authoritative canonical text can produce large output.
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:72:### 4.3 Risks and mitigations
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:78:- Authority creep: no decision/lifecycle schema or API and explicit false
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:80:- Mutable reconstruction: return the original const-field Phase 6A type and a
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:83:## 5. Compatibility and migration
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:86:and materialization tables. It does not alter experiment or scheduler tables.
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:87:The Phase 6A public builder and canonical grammar remain version 1 and
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:90:## 6. Verification and operational evidence
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:92:- Repeatable migration, exact privilege, FK, trigger, index, and collation
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:94:- Exact round-trip, immutable reload, canonical/hash preservation, idempotent
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:95:  duplicate replay, duplicate persisted identity rejection, and malformed row
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:99:- Phase 6A, Phase 5A/B/C, relevant Phase 4D regression tests, Release build,
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:102:## 7. Alternatives considered
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:104:### 7.1 Store only the proposal hash
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:106:Rejected because hashes are not authoritative and cannot reconstruct or
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:109:### 7.2 Store only canonical text and parse it on every read
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:111:Rejected because Phase 6A intentionally defines canonical generation, not a
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:113:bounded typed reload without making presentation text authoritative input.
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:115:### 7.3 Return a mutable persistence DTO
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:118:Phase 6A proposal and silently lose immutability guarantees.
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:120:### 7.4 Add operator approval or execution state
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:122:Rejected as a distinct later authority explicitly outside Phase 6B.
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:124:## 8. References
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:127:- [Phase 6A proposal](../../Phase6ARecommendationCampaignFollowUpProposal.rst)
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:128:- [Phase 6B persistence and preview](../../Phase6BRecommendationCampaignFollowUpProposalPersistence.rst)
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:130:- [ADR-0004](ADR-0004-scheduler-ownership-boundaries.md)
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:131:- [ADR-0006](ADR-0006-phase-6a-follow-up-proposal.md)
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:133:## 9. Revision history
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:137:| 2026-07-20 | Accepted exact append-only Phase 6B proposal persistence and read-only preview. |
docs/Phase4DExperimentRecommendationCampaignApproval.rst:10:proposal, conversion, activation, experiment, scheduler action, or worker.
docs/Phase4DExperimentRecommendationCampaignApproval.rst:27:The approval row stores the authoritative snapshot, planning-policy, planning-
docs/Phase4DExperimentRecommendationCampaignApproval.rst:39:canonical text is authoritative.
docs/Phase4DExperimentRecommendationCampaignApproval.rst:52:Schema and privileges
docs/Phase4DExperimentRecommendationCampaignApproval.rst:62:large authoritative text is not stored in a size-limited btree key; PostgreSQL
docs/Phase4DExperimentRecommendationCampaignApproval.rst:66:and receives no ``UPDATE``, ``DELETE``, or ``TRUNCATE`` privilege. No trigger
docs/Phase4DExperimentRecommendationCampaignApproval.rst:93:Machine output distinguishes recorded, replayed, stale, conflict, invalid,
docs/Phase4DExperimentRecommendationCampaignApproval.rst:96:proposal/experiment mutation, activation, scheduler, and worker side effects
docs/Phase4DExperimentRecommendationCampaignApproval.rst:109:It verifies exact replay, conflict, concurrent convergence, rollback, runtime
docs/Phase4DExperimentRecommendationCampaignApproval.rst:118:aware authorization, background scans, and scheduler integration remain
docs/Phase4DExperimentRecommendationCampaignApproval.rst:123:Phase 4D Step 4 may consume an ``approved`` row only to materialize the exact
docs/Phase4DExperimentRecommendationCampaignHandoff.rst:7:Step 5 projects one immutable Phase 4D campaign materialization onto the
docs/Phase4DExperimentRecommendationCampaignHandoff.rst:8:current Phase 4C proposal lifecycle. The materialization manifest and its
docs/Phase4DExperimentRecommendationCampaignHandoff.rst:9:one-based ordered member links remain the sole authority for campaign
docs/Phase4DExperimentRecommendationCampaignHandoff.rst:11:lifecycle rows are observed; they are never changed or repaired.
docs/Phase4DExperimentRecommendationCampaignHandoff.rst:13:The projection is not persisted authority and is not authorization. In
docs/Phase4DExperimentRecommendationCampaignHandoff.rst:22:Every materialization member is reported in its stored ordinal order with its
docs/Phase4DExperimentRecommendationCampaignHandoff.rst:38:4C Step 3, the greatest review-decision primary key is authoritative.
docs/Phase4DExperimentRecommendationCampaignHandoff.rst:39:``awaiting_review`` and ``no_authoritative_review`` count members for which no
docs/Phase4DExperimentRecommendationCampaignHandoff.rst:47:Malformed immutable materialization evidence—unsupported contract version,
docs/Phase4DExperimentRecommendationCampaignHandoff.rst:55:mismatch; missing linked experiment; and incompatible experiment lifecycle.
docs/Phase4DExperimentRecommendationCampaignHandoff.rst:67:returned materialization. The default list limit is 100 and the enforced
docs/Phase4DExperimentRecommendationCampaignHandoff.rst:69:authoritative member order. No migration, view, lock, sequence access, or new
docs/Phase4DExperimentRecommendationCampaignHandoff.rst:70:privilege is added.
docs/Phase4DExperimentRecommendationCampaignHandoff.rst:83:bounded list emits deterministic materialization-ID order and a final count.
docs/Phase4DExperimentRecommendationCampaignHandoff.rst:85:materialization, proposal, review, execution, activation, experiment,
docs/Phase4DExperimentRecommendationCampaignHandoff.rst:86:scheduler, or worker was created or started. Summary, list-completion, and
docs/Phase4DExperimentRecommendationCampaignHandoff.rst:100:approve/reject decision to every exact materialized proposal through ordinary
docs/Phase4DExperimentRecommendationCampaignHandoff.rst:107:activation rows and lifecycle states but does not invoke it. Phase 5 Step 3
docs/Phase4DExperimentRecommendationCampaignHandoff.rst:114:progression, repair, background scans, and scheduler integration remain
docs/Phase4DExperimentRecommendationCampaignHandoff.rst:115:separate. The scheduler does not query the handoff projection or any Phase 4D
docs/Phase4DExperimentRecommendationCampaignPlanning.rst:10:review, execution, activation, experiment, scheduler action, or worker.
docs/Phase4DExperimentRecommendationCampaignPlanning.rst:14:it. The scheduler does not poll plans or Phase 4C audit tables.
docs/Phase4DExperimentRecommendationCampaignPlanning.rst:20:Its immutable member order and final scores are authoritative. Recommendation
docs/Phase4DExperimentRecommendationCampaignPlanning.rst:23:campaign planning does not redefine review or lifecycle semantics.
docs/Phase4DExperimentRecommendationCampaignPlanning.rst:25:also checked against the authoritative recommendation row. A mismatch is
docs/Phase4DExperimentRecommendationCampaignPlanning.rst:32:migration or privilege.
docs/Phase4DExperimentRecommendationCampaignPlanning.rst:46:* explicit rejected, failed, and cancelled-workflow reconsideration flags;
docs/Phase4DExperimentRecommendationCampaignPlanning.rst:58:The current schema has no authoritative durable profitability metric. The
docs/Phase4DExperimentRecommendationCampaignPlanning.rst:88:execution, paused execution, activation/scheduler ownership, completion,
docs/Phase4DExperimentRecommendationCampaignPlanning.rst:89:failure, cancellation, and inconsistent evidence. Reconsideration flags only
docs/Phase4DExperimentRecommendationCampaignPlanning.rst:90:relax their named rejected/failed/cancelled exclusions. Inconsistent workflows
docs/Phase4DExperimentRecommendationCampaignPlanning.rst:96:The authoritative campaign-plan canonical text length-frames variable text and
docs/Phase4DExperimentRecommendationCampaignPlanning.rst:128:``read_only=true`` plus false proposal/experiment/activation/scheduler/worker
docs/Phase4DExperimentRecommendationCampaignPlanning.rst:153:execution, automatic activation, background scans, budgets, durable
docs/Phase4DExperimentRecommendationCampaignPlanning.rst:154:profitability policy, and scheduler integration remain outside Step 1.
docs/Phase4DExperimentRecommendationCampaignPlanning.rst:156:plan and materializes only its selected members as Phase 4C proposals.
docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst:9:materialization.  It is read-only, non-persistent, non-authoritative, and does
docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst:10:not declare campaign success or authorize follow-up work.
docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst:16:``cancelled``, or ``unknown``).  Consistency is independent
docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst:19:inconsistent lifecycle or workflow observation without changing or discarding
docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst:20:the exact materialized-member identity.  Scientific comparisons are derived
docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst:23:failure, terminal cancellation, and inconsistency.
docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst:37:Each materialized member may identify its expected downstream experiment.
docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst:43:top-level materialization membership is trustworthy; invalid or ambiguous
docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst:44:member identity fails the assessment.  Campaign and materialization inputs
docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst:48:Canonical text and hashing bind lifecycle, consistency, diagnostics, canonical
docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst:50:and metric summaries.  Authoritative campaign and materialization canonical
docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst:59:CLI, scheduler, worker, filesystem behavior, campaign-success policy, or
docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst:60:follow-up authorization.
docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst:62:Step 5b authoritative-evidence integration
docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst:67:authoritative persisted evidence: the exact materialization through the Phase
docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst:69:lifecycle and workflow consistency from the existing Phase 5 campaign-status
docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst:70:projection.  Step 5b builds a non-authoritative, point-in-time assessment from
docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst:71:that evidence.  It does not reconstruct a second lifecycle classifier.
docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst:74:hash.  Its materialization input receives the materialization row's persisted
docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst:76:materialization binding.  Exact materialization member ordinal, member ID,
docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst:86:Source evidence follows the materialized recommendation's persisted source
docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst:124:materialized ordinal.  Records expose contract and identity hashes, the exact
docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst:125:Step 5a aggregate classification and counts, lifecycle, separate input and
docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst:128:percent-encoded variable text, the materialization's persisted campaign-
docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst:129:approval hash, and explicit no-write/no-scheduler/no-success/no-follow-up
docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst:132:No schema migration or privilege change is required.  The assessment remains
docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst:134:queues, retries, and launches nothing and authorizes no follow-up.
docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst:141:recompute a metric or delta, or change any Step 5a lifecycle, consistency,
docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst:172:not-ready, cancelled, context-changed, metric-gap, missing-source,
docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst:175:A terminal failure is adverse lifecycle evidence rather than a scientific
docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst:179:inconsistency, unfinished lifecycle, or non-comparable successful evidence
docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst:189:cancellation, incompleteness, inconsistency, context change, or a metric gap
docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst:196:approval and materialization identities, ordered member identities including
docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst:197:the expected experiment, assessment lifecycle/consistency/outcome/diagnostics,
docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst:208:Step 5c adds no repository, service, formatter, CLI, migration, privilege,
docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst:209:table, sequence, lock, write, scheduler call, worker behavior, experiment or
docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst:210:campaign mutation, persistence, operator-review event, follow-up generation,
docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst:212:existing Step 5b CLI continues to report only the non-authoritative assessment
docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst:213:built from authoritative persisted evidence and does not silently apply outcome
docs/Phase6BRecommendationCampaignFollowUpProposalPersistence.rst:1:Phase 6B Recommendation Campaign Follow-Up Proposal Persistence and Preview
docs/Phase6BRecommendationCampaignFollowUpProposalPersistence.rst:7:Phase 6B durably records an already-built Phase 6A
docs/Phase6BRecommendationCampaignFollowUpProposalPersistence.rst:10:record an operator decision, approve follow-up, activate or execute anything,
docs/Phase6BRecommendationCampaignFollowUpProposalPersistence.rst:11:create an experiment, or interact with the scheduler.
docs/Phase6BRecommendationCampaignFollowUpProposalPersistence.rst:19:and materialization identities, the copied favorable/eligible/non-authorizing
docs/Phase6BRecommendationCampaignFollowUpProposalPersistence.rst:21:Phase 6A member identity, including nullable expected-experiment ID.
docs/Phase6BRecommendationCampaignFollowUpProposalPersistence.rst:24:canonical value retains the Phase 6A 1 MiB limit.  Tagged hashes have format
docs/Phase6BRecommendationCampaignFollowUpProposalPersistence.rst:25:checks but remain accelerators.  Exact canonical equality is authoritative.
docs/Phase6BRecommendationCampaignFollowUpProposalPersistence.rst:31:immutable Phase 4D campaign approval, materialization, and ordered
docs/Phase6BRecommendationCampaignFollowUpProposalPersistence.rst:32:materialization-member provenance copied into the proposal.  A deferred
docs/Phase6BRecommendationCampaignFollowUpProposalPersistence.rst:42:Phase 6A type.  It locks the tagged-hash bucket, compares exact canonical text,
docs/Phase6BRecommendationCampaignFollowUpProposalPersistence.rst:49:is the original non-assignable Phase 6A value, not a mutable persistence DTO.
docs/Phase6BRecommendationCampaignFollowUpProposalPersistence.rst:50:A private friendship grants only the Phase 6B hydration builder access to the
docs/Phase6BRecommendationCampaignFollowUpProposalPersistence.rst:51:Phase 6A constructors.  Hydration reruns supported-version, canonical/hash,
docs/Phase6BRecommendationCampaignFollowUpProposalPersistence.rst:52:campaign, materialization, count, ordered-member, duplicate-member,
docs/Phase6BRecommendationCampaignFollowUpProposalPersistence.rst:62:authoritative proposal canonical text, all source identity hashes, summary,
docs/Phase6BRecommendationCampaignFollowUpProposalPersistence.rst:63:ordered members, persistence timestamp, and explicit negative authority
docs/Phase6BRecommendationCampaignFollowUpProposalPersistence.rst:71:Persistence means only that the advisory Phase 6A value is durably available
docs/Phase6BRecommendationCampaignFollowUpProposalPersistence.rst:73:``eligible_for_operator_review`` is an operator decision or follow-up
docs/Phase6BRecommendationCampaignFollowUpProposalPersistence.rst:74:authorization.  Phase 6B adds no approval/rejection table or API, activation,
docs/Phase6BRecommendationCampaignFollowUpProposalPersistence.rst:75:execution, scheduler integration, queue, experiment creation, or follow-up
docs/Phase6BRecommendationCampaignFollowUpProposalPersistence.rst:81:Focused tests cover migration repeatability and privileges, exact round-trip
docs/Phase6BRecommendationCampaignFollowUpProposalPersistence.rst:82:and immutable reload, canonical/hash preservation, exact duplicate replay,
docs/Phase6BRecommendationCampaignFollowUpProposalPersistence.rst:85:6A and Phase 4D/5 regressions protect the upstream authority boundaries.
docs/Phase4DExperimentRecommendationCampaignProposalReview.rst:4:Purpose and authority
docs/Phase4DExperimentRecommendationCampaignProposalReview.rst:8:review authority. It applies one ``approve`` or ``reject`` decision to every
docs/Phase4DExperimentRecommendationCampaignProposalReview.rst:9:exact proposal linked by one persisted Step 4 campaign materialization. It does
docs/Phase4DExperimentRecommendationCampaignProposalReview.rst:13:review-decision primary key remains the authoritative current Phase 4C review.
docs/Phase4DExperimentRecommendationCampaignProposalReview.rst:15:The sole membership authority is the immutable
docs/Phase4DExperimentRecommendationCampaignProposalReview.rst:16:``experiment_recommendation_campaign_materialization`` manifest and its
docs/Phase4DExperimentRecommendationCampaignProposalReview.rst:18:``experiment_recommendation_campaign_materialization_member`` rows. The command
docs/Phase4DExperimentRecommendationCampaignProposalReview.rst:26:The existing Step 4 materialization loader and Step 5 immutable-input validator
docs/Phase4DExperimentRecommendationCampaignProposalReview.rst:40:There is no nested or read-only write transaction, experiment lock, scheduler
docs/Phase4DExperimentRecommendationCampaignProposalReview.rst:46:Operation contract version 1 canonically binds the materialization identity,
docs/Phase4DExperimentRecommendationCampaignProposalReview.rst:48:4C request ID contains the materialization primary key, contract version, and
docs/Phase4DExperimentRecommendationCampaignProposalReview.rst:55:opposite authoritative decision conflicts. A same-decision row from another
docs/Phase4DExperimentRecommendationCampaignProposalReview.rst:67:     --review-recommendation-campaign-materialization=MATERIALIZATION_ID \
docs/Phase4DExperimentRecommendationCampaignProposalReview.rst:76:materialization ID, a strict decision, non-empty normalized operator and reason,
docs/Phase4DExperimentRecommendationCampaignProposalReview.rst:81:manifest, membership, proposal, workflow, idempotence, and conflict validation,
docs/Phase4DExperimentRecommendationCampaignProposalReview.rst:88:stored ordinal order. It reports the materialization and operation identities,
docs/Phase4DExperimentRecommendationCampaignProposalReview.rst:104:Step 6 adds no migration, table, sequence, mutable status, or privilege. The
docs/Phase4DExperimentRecommendationCampaignProposalReview.rst:112:atomic convenience over the existing Phase 4C paused-execution authority;
docs/Phase4DExperimentRecommendationCampaignProposalReview.rst:113:activation and scheduler lifecycle remain separate. Phase 5 Step 3 is another
docs/Phase4DExperimentRecommendationCampaignProposalReview.rst:115:activation in one transaction without changing Step 6 authority. The scheduler
docs/Phase5ExperimentRecommendationCampaignExecution.rst:4:Purpose and authority
docs/Phase5ExperimentRecommendationCampaignExecution.rst:8:Phase 4C conversion-execution authority. Given one persisted Phase 4D campaign
docs/Phase5ExperimentRecommendationCampaignExecution.rst:9:materialization, it atomically invokes the ordinary Phase 4C paused conversion
docs/Phase5ExperimentRecommendationCampaignExecution.rst:11:status, alternate proposal eligibility rule, activation authority, scheduler
docs/Phase5ExperimentRecommendationCampaignExecution.rst:18:``experiment_recommendation_campaign_materialization_member`` rows belonging to
docs/Phase5ExperimentRecommendationCampaignExecution.rst:19:the requested ``experiment_recommendation_campaign_materialization``. The
docs/Phase5ExperimentRecommendationCampaignExecution.rst:20:existing materialization loader verifies its version, canonical identity,
docs/Phase5ExperimentRecommendationCampaignExecution.rst:49:The version-1 operation identity binds the complete immutable materialization
docs/Phase5ExperimentRecommendationCampaignExecution.rst:51:unordered input. Individual durable execution identity and idempotence remain
docs/Phase5ExperimentRecommendationCampaignExecution.rst:60:     --execute-recommendation-campaign-materialization=MATERIALIZATION_ID \
docs/Phase5ExperimentRecommendationCampaignExecution.rst:63:The separated form is also accepted. A positive materialization ID is required,
docs/Phase5ExperimentRecommendationCampaignExecution.rst:78:report materialization and operation identities, exact proposal/review/
docs/Phase5ExperimentRecommendationCampaignExecution.rst:80:explicit scheduler, worker, activation, queueing, and automatic-progression
docs/Phase5ExperimentRecommendationCampaignExecution.rst:92:start the scheduler, launch workers, or automatically invoke another command.
docs/Phase5ExperimentRecommendationCampaignExecution.rst:95:authority for every exact materialized member; Step 1 never invokes it.
docs/Phase5ExperimentRecommendationCampaignExecution.rst:99:Phase 5 Step 4 can observe the resulting exact member lifecycle in one read-only
docs/Phase5ExperimentRecommendationCampaignExecution.rst:105:No migration or privilege change is required. Campaign scope comes from the
docs/Phase5ExperimentRecommendationCampaignExecution.rst:106:immutable Phase 4D materialization, while each durable outcome is an ordinary
docs/Phase5ExperimentRecommendationCampaignExecution.rst:113:* ``docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md``
docs/Phase4DExperimentRecommendationCampaignReview.rst:9:activation, experiment, scheduler action, or worker. It adds no migration and
docs/Phase4DExperimentRecommendationCampaignReview.rst:10:uses the same bounded read transaction and authoritative Phase 4B/4C evidence
docs/Phase4DExperimentRecommendationCampaignReview.rst:32:Duplicate grouping compares authoritative canonical text, not hashes. The
docs/Phase4DExperimentRecommendationCampaignReview.rst:42:complete authoritative plan identity, policy hash, scope, validated summary,
docs/Phase4DExperimentRecommendationCampaignReview.rst:68:SQL and no database privileges. Repository verification uses only an explicit
docs/Phase4DExperimentRecommendationCampaignReview.rst:76:activation, background scans, and scheduler integration remain deferred.
docs/Phase4DExperimentRecommendationCampaignReview.rst:78:exact canonical evidence and stops at Phase 4C proposal materialization.
docs/Phase5ExperimentRecommendationCampaignStatus.rst:4:Purpose and authority
docs/Phase5ExperimentRecommendationCampaignStatus.rst:8:4D campaign materialization. It reports every exact member's ordinary Phase 4C
docs/Phase5ExperimentRecommendationCampaignStatus.rst:9:review, execution, activation, experiment lifecycle, worker metadata, model,
docs/Phase5ExperimentRecommendationCampaignStatus.rst:11:lifecycle authority and persists nothing.
docs/Phase5ExperimentRecommendationCampaignStatus.rst:14:``experiment_recommendation_campaign_materialization_member`` are the only
docs/Phase5ExperimentRecommendationCampaignStatus.rst:24:sequence call, and uses bounded exact-ID queries for the materialized proposals
docs/Phase5ExperimentRecommendationCampaignStatus.rst:30:identity binds the validated materialization and every ordered member's durable
docs/Phase5ExperimentRecommendationCampaignStatus.rst:31:review, execution, activation, experiment lifecycle, worker, model, phase-
docs/Phase5ExperimentRecommendationCampaignStatus.rst:34:Persisted lifecycle timestamps are evidence and are bound; ``observed_at`` and
docs/Phase5ExperimentRecommendationCampaignStatus.rst:45:paused, pending scheduler claim, running train/infer/analyze, completed, failed,

exec
/bin/bash -lc 'for f in docs/architecture/adr/ADR-0001-postgresql-source-of-truth.md docs/architecture/adr/ADR-0002-deterministic-experiment-identity.md docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md docs/architecture/adr/ADR-0005-manual-recommendation-conversion.md; do nl -ba "$f"; done' in /Volumes/Developer SSD/ExpertAdvisor
 succeeded in 0ms:
     1	# ADR-0001: PostgreSQL as the durable source of truth
     2
     3	Status: Accepted
     4	Date: 2026-07-15
     5	Deciders: Project architecture
     6	Affected volumes: Volume I §§7–9; Volume XII
     7	Supersedes: None
     8	Superseded by: None
     9
    10	## 1. Context and problem statement
    11
    12	The platform coordinates long-running training, inference, analysis,
    13	continuation, and advisory recommendation workflows. Processes may terminate,
    14	restart, overlap, or be inspected independently. Logs, PIDs, in-memory objects,
    15	and filesystem artifacts cannot alone provide transactional ownership,
    16	referential integrity, or a consistent lifecycle record.
    17
    18	The project already persists experiments, models, evidence, policies,
    19	recommendations, scores, reviews, and migration history in PostgreSQL.
    20
    21	## 2. Decision
    22
    23	PostgreSQL is the authoritative durable source of truth for platform state
    24	represented by the database schema.
    25
    26	- Repositories own SQL and map rows into typed domain records.
    27	- Related state and audit/provenance writes commit atomically.
    28	- Database constraints enforce durable invariants where safe.
    29	- Process memory, logs, files, and PIDs are evidence or caches, not competing
    30	  authority.
    31	- Schema evolution uses ordered checksum-recorded migrations.
    32	- A recorded migration is never rewritten; corrections are additive.
    33	- Runtime roles receive least privilege, and append-only audit history denies
    34	  ordinary runtime update/delete privileges.
    35
    36	This decision does not claim that PostgreSQL currently contains immutable
    37	snapshots of every external market-data source. Exact data-replay identity
    38	remains a separate future contract under Volume II.
    39
    40	## 3. Rationale and decision drivers
    41
    42	- Transactions provide atomic lifecycle and provenance updates.
    43	- Constraints and locks provide cross-process correctness.
    44	- Durable queries make recovery and audit independent of one process.
    45	- Ordered migrations make schema history observable and repeatable.
    46	- Typed repositories isolate storage details from domain policy.
    47
    48	## 4. Consequences
    49
    50	### 4.1 Positive consequences
    51
    52	- Restart-safe state and cross-process coordination.
    53	- Enforceable references, uniqueness, status shapes, and ownership.
    54	- Consistent inspection and recovery paths.
    55	- Auditable schema and decision history.
    56
    57	### 4.2 Negative consequences and trade-offs
    58
    59	- Repository and migration discipline are mandatory.
    60	- Database availability and backup quality become operational dependencies.
    61	- Long transactions and broad locks can harm concurrency if poorly designed.
    62
    63	### 4.3 Risks and mitigations
    64
    65	- Schema drift: prevented by the migration ledger and checksum checks.
    66	- Excess privilege: mitigated by role-specific grants and privilege tests.
    67	- Legacy incompatibility: handled with additive migrations, explicit exclusion,
    68	  and `NOT VALID` constraints where appropriate.
    69
    70	## 5. Compatibility and migration
    71
    72	This ADR records existing architecture and requires no schema change. Future
    73	schema changes follow Volume XII. Existing legacy rows are preserved; missing
    74	provenance is not guessed merely to satisfy a new contract.
    75
    76	## 6. Verification and operational evidence
    77
    78	- Migration runner repeatability and checksum mismatch detection.
    79	- Isolated migration tests and constraint/index/grant inspection.
    80	- Transaction rollback and concurrency tests using independent connections.
    81	- Backup and restore procedures appropriate to deployed environments.
    82	- Repository tests confirming raw rows do not escape persistence boundaries.
    83
    84	## 7. Alternatives considered
    85
    86	### 7.1 Logs and process state as authority
    87
    88	Rejected because they lack atomicity, referential integrity, durable ownership,
    89	and reliable recovery semantics.
    90
    91	### 7.2 Filesystem-only state
    92
    93	Rejected as the primary lifecycle store because multi-process claims,
    94	relational integrity, querying, and migrations would require rebuilding a
    95	database poorly.
    96
    97	### 7.3 Per-subsystem independent stores
    98
    99	Not selected for current architecture because it adds distributed consistency
   100	and operational complexity without a demonstrated need.
   101
   102	## 8. References
   103
   104	- [Volume I §7](../Volume_I_Foundation.md)
   105	- [Volume XII](../Volume_XII_Database.md)
   106	- [`migrate_lstm_db.sh`](../../../migrate_lstm_db.sh)
   107
   108	## 9. Revision history
   109
   110	| Date | Change |
   111	|---|---|
   112	| 2026-07-15 | Recorded the existing PostgreSQL ownership and migration strategy. |
     1	# ADR-0002: Deterministic experiment identity
     2
     3	Status: Accepted
     4	Date: 2026-07-15
     5	Deciders: Project architecture
     6	Affected volumes: Volume I §§3–5; Volume VII; Volume VIII
     7	Supersedes: None
     8	Superseded by: None
     9
    10	## 1. Context and problem statement
    11
    12	The platform must distinguish a research question from a particular execution.
    13	Database row IDs locate lifecycle records but do not describe semantic
    14	equivalence. Resume lineage and checkpoint cadence can distinguish invocations
    15	without necessarily changing the semantic hyperparameter/data question.
    16
    17	Hashes are useful lookup accelerators, but finite hashes can collide and cannot
    18	serve as authoritative equality without an additional collision contract.
    19
    20	## 2. Decision
    21
    22	Experiment-related identity is deterministic, versioned, and content-based.
    23
    24	- Semantic configuration canonical text is authoritative for the named
    25	  research equivalence.
    26	- Invocation identity is separately defined when lineage or operational inputs
    27	  must distinguish executions.
    28	- Row IDs, lifecycle status, timestamps, PIDs, logs, and result metrics do not
    29	  define semantic identity.
    30	- Canonicalization defines field order, nulls, numeric formatting, dates,
    31	  symbols, collections, and a version prefix.
    32	- Hashes accelerate lookup and diagnostics only. Every possible hash match is
    33	  resolved by canonical-text comparison, and collisions remain distinct and
    34	  observable.
    35	- Any identity-semantic change requires a new canonical version, compatibility
    36	  analysis, migration strategy, volume update, and ADR.
    37
    38	## 3. Rationale and decision drivers
    39
    40	- Deterministic duplicate detection across processes and time.
    41	- Separation of scientific equivalence from operational execution.
    42	- Explicit collision safety.
    43	- Stable provenance for recommendations, scores, and future comparisons.
    44
    45	## 4. Consequences
    46
    47	### 4.1 Positive consequences
    48
    49	- Equivalent configurations compare consistently independent of row IDs.
    50	- Resume and operational differences can remain visible without corrupting
    51	  semantic duplicate rules.
    52	- Hash collisions cannot silently merge research configurations.
    53
    54	### 4.2 Negative consequences and trade-offs
    55
    56	- Canonical grammar and versions require careful maintenance.
    57	- Historical rows lacking complete configuration may be unmappable.
    58	- Adding a behavior-affecting setting may require schema and identity changes.
    59
    60	### 4.3 Risks and mitigations
    61
    62	- Hidden compiled settings: document as unreconstructible and fail closed where
    63	  required until a durable implementation contract exists.
    64	- Locale/timezone drift: use locale-independent numbers and explicit Gregorian
    65	  date mapping.
    66	- Hash-only shortcuts: prevent through repository tests and schema design.
    67
    68	## 5. Compatibility and migration
    69
    70	This ADR records the identity approach reflected in Phase 4. It does not
    71	retroactively assign canonical identities to legacy rows. Existing canonical
    72	versions remain stable. New versions must coexist or migrate through an
    73	explicit accepted plan.
    74
    75	## 6. Verification and operational evidence
    76
    77	- Pure canonicalization and strict parsing tests.
    78	- Stable known canonical text and hash fixtures.
    79	- Adjacent floating-value, null, date, collection, and symbol cases.
    80	- Collision tests proving equal hash/unequal text remains distinct.
    81	- Repository duplicate and concurrency tests using full canonical values.
    82
    83	## 7. Alternatives considered
    84
    85	### 7.1 Database row ID as experiment identity
    86
    87	Rejected because it identifies one record, not semantic equivalence.
    88
    89	### 7.2 Hash-only equality
    90
    91	Rejected because collisions would silently merge or block distinct research.
    92
    93	### 7.3 One identity for semantics and invocation
    94
    95	Rejected because lineage and checkpoint controls have different equivalence
    96	meaning from the underlying research configuration.
    97
    98	## 8. References
    99
   100	- [Volume I §5](../Volume_I_Foundation.md)
   101	- [Volume VII](../Volume_VII_Experiment_Lifecycle.md)
   102	- [Volume VIII](../Volume_VIII_Recommendation_Engine.md)
   103	- [Phase 4 recommendation foundation](../../Phase4AExperimentRecommendationFoundation.rst)
   104
   105	## 9. Revision history
   106
   107	| Date | Change |
   108	|---|---|
   109	| 2026-07-15 | Recorded canonical semantic and invocation identity as the durable experiment-equivalence foundation. |
     1	# ADR-0003: Advisory recommendation evaluation
     2
     3	Status: Accepted
     4	Date: 2026-07-15
     5	Deciders: Project architecture
     6	Affected volumes: Volume I §§2.5, 10.3; Volume VIII
     7	Supersedes: None
     8	Superseded by: None
     9
    10	## 1. Context and problem statement
    11
    12	The recommendation subsystem can generate deterministic research proposals,
    13	persist duplicate/provenance evidence, score and rank proposals, and record an
    14	explicit operator review. These facts are useful for research prioritization,
    15	but they do not provide a complete implementation identity, execution budget,
    16	profitability guarantee, or experiment-creation authorization.
    17
    18	Combining proposal, evaluation, approval, conversion, and execution would make
    19	scores or status changes implicit operational authority.
    20
    21	## 2. Decision
    22
    23	Recommendation generation, scoring, ranking, explanation, and review are
    24	advisory and non-executing.
    25
    26	- Scores and ranks do not change recommendation status.
    27	- Review actions are explicit operator inputs, never threshold-derived.
    28	- Approval means only reviewed for possible later use.
    29	- Approval does not create an experiment, queue work, authorize execution,
    30	  predict profitability, or establish scheduler eligibility.
    31	- Recommendation commands remain outside scheduler polling.
    32	- Review history is immutable to the normal runtime role.
    33	- Recommendation-to-experiment conversion is a separate deferred capability
    34	  requiring its own accepted ADR and service boundary.
    35
    36	## 3. Rationale and decision drivers
    37
    38	- Preserve provenance between proposal, evidence, decision, and execution.
    39	- Prevent accidental work creation from heuristic scores.
    40	- Allow operator review without overstating model or financial confidence.
    41	- Keep scheduler and experiment lifecycle ownership explicit.
    42
    43	## 4. Consequences
    44
    45	### 4.1 Positive consequences
    46
    47	- Operators can inspect and disposition proposals safely.
    48	- Scores remain reproducible evidence rather than hidden policy triggers.
    49	- No Phase 4 command consumes scheduler capacity or changes experiments.
    50	- Future conversion can be designed with complete identity and audit.
    51
    52	### 4.2 Negative consequences and trade-offs
    53
    54	- Approved recommendations require a separate future workflow before use.
    55	- There is no automatic throughput from recommendation to experiment.
    56	- Operators must interpret advisory evidence without a profitability guarantee.
    57
    58	### 4.3 Risks and mitigations
    59
    60	- Misreading approval as authorization: mitigated by CLI disclaimers,
    61	  documentation, and absence of conversion code.
    62	- Silent status changes: prevented by explicit commands, terminal transitions,
    63	  row locks, and immutable events.
    64	- Score-driven automation by accident: prevented by scheduler isolation and
    65	  service boundaries.
    66
    67	## 5. Compatibility and migration
    68
    69	This ADR records Phase 4A Steps 1–5. It requires no new behavioral migration.
    70	Existing recommendation identities, duplicate rules, score history, review
    71	events, and experiment rows remain unchanged.
    72
    73	## 6. Verification and operational evidence
    74
    75	- Pure candidate, scoring, ranking, and review-transition tests.
    76	- Persistence and concurrency tests for duplicates, scores, and review events.
    77	- Before/after experiment-state and score-history comparisons.
    78	- CLI mutual-exclusion and scheduler-isolation tests.
    79	- Runtime privilege checks for immutable review history.
    80
    81	## 7. Alternatives considered
    82
    83	### 7.1 Automatically queue the highest score
    84
    85	Rejected because a deterministic prioritization score is not execution
    86	authority or evidence of expected profitability.
    87
    88	### 7.2 Treat approval as conversion authorization
    89
    90	Rejected because Step 5 lacks a conversion identity, budget, lifecycle request,
    91	and scheduler-capacity contract.
    92
    93	### 7.3 Scheduler-managed recommendation polling
    94
    95	Rejected for Phase 4A because it would collapse advisory evaluation into
    96	automation without explicit ownership and safety decisions.
    97
    98	## 8. References
    99
   100	- [Volume I §10.3](../Volume_I_Foundation.md)
   101	- [Volume VIII](../Volume_VIII_Recommendation_Engine.md)
   102	- [Phase 4 recommendation review](../../Phase4AExperimentRecommendationReview.rst)
   103
   104	## 9. Revision history
   105
   106	| Date | Change |
   107	|---|---|
   108	| 2026-07-15 | Recorded the advisory, non-executing recommendation boundary. |
     1	# ADR-0004: Scheduler ownership boundaries
     2
     3	Status: Accepted
     4	Date: 2026-07-15
     5	Deciders: Project architecture
     6	Affected volumes: Volume I §10; Volume VII; Volume XI
     7	Supersedes: None
     8	Superseded by: None
     9
    10	## 1. Context and problem statement
    11
    12	The platform coordinates long-running training, inference, analysis,
    13	checkpoint, and configured continuation work. Many other capabilities can be
    14	invoked explicitly but are not scheduler work. Without a firm boundary, adding
    15	a command could accidentally add polling, consume worker capacity, or create
    16	implicit lifecycle transitions.
    17
    18	Process launch also cannot safely share one long database transaction, and PID
    19	presence alone cannot establish durable ownership after failures.
    20
    21	## 2. Decision
    22
    23	The scheduler owns only explicitly assigned lifecycle work classes.
    24
    25	- Eligibility and lifecycle transitions are durable database contracts.
    26	- The scheduler atomically claims bounded work before launching a worker.
    27	- Worker launch and execution occur outside database transactions.
    28	- Workers perform one bounded operation and report facts; they do not select or
    29	  queue further work independently.
    30	- Capacity, retries, recovery, shutdown, and operator controls are defined per
    31	  work class.
    32	- New scheduler-managed work requires an ADR defining eligibility, claim,
    33	  capacity, idempotency, recovery, observability, and regression boundaries.
    34	- Recommendation generation, scoring, and review remain outside scheduler
    35	  polling under ADR-0003.
    36
    37	## 3. Rationale and decision drivers
    38
    39	- Prevent duplicate workers and lost lifecycle updates.
    40	- Keep database transactions short.
    41	- Make capacity and process ownership observable and recoverable.
    42	- Prevent unrelated explicit commands from becoming background automation.
    43
    44	## 4. Consequences
    45
    46	### 4.1 Positive consequences
    47
    48	- Clear ownership between lifecycle services, scheduler, and workers.
    49	- Narrow database claims support multiple processes safely.
    50	- Scheduler capacity remains predictable.
    51	- Orphan recovery can reconcile durable attempt state with processes.
    52
    53	### 4.2 Negative consequences and trade-offs
    54
    55	- Every work class needs explicit claim/recovery design.
    56	- Process supervision and database reconciliation remain operationally complex.
    57	- New automation cannot be added merely by placing a command in the poll loop.
    58
    59	### 4.3 Risks and mitigations
    60
    61	- Launch after claim fails: persist a distinct launch failure and reconcile.
    62	- Multiple schedulers race: use database-visible conditional claims/locks.
    63	- Stale PIDs: combine attempt identity and durable timestamps/state with process
    64	  inspection; never trust PID alone.
    65
    66	## 5. Compatibility and migration
    67
    68	This ADR records existing scheduler boundaries and requires no schema change.
    69	Future work classes may require additive attempt/capacity schema and must
    70	preserve current experiment and continuation behavior.
    71
    72	## 6. Verification and operational evidence
    73
    74	- Pure eligibility, capacity, command, and status-mapping tests.
    75	- Claim/concurrency tests with multiple database connections.
    76	- Launch-failure, completion-race, orphan-recovery, and shutdown tests.
    77	- Regression checks proving advisory commands do not enter scheduler polling.
    78	- Controlled integration that does not disturb running experiments.
    79
    80	## 7. Alternatives considered
    81
    82	### 7.1 Workers self-select follow-up work
    83
    84	Rejected because it bypasses centralized capacity, durable lifecycle ownership,
    85	and operator controls.
    86
    87	### 7.2 Hold a database transaction for worker lifetime
    88
    89	Rejected because long transactions harm concurrency and failure recovery.
    90
    91	### 7.3 Treat every explicit command as pollable work
    92
    93	Rejected because command availability does not define scheduler eligibility or
    94	resource policy.
    95
    96	## 8. References
    97
    98	- [Volume I §10](../Volume_I_Foundation.md)
    99	- [Volume VII](../Volume_VII_Experiment_Lifecycle.md)
   100	- [Volume XI](../Volume_XI_Scheduler.md)
   101	- [ADR-0003](ADR-0003-advisory-recommendation-evaluation.md)
   102
   103	## 9. Revision history
   104
   105	| Date | Change |
   106	|---|---|
   107	| 2026-07-15 | Recorded explicit scheduler work ownership and process/transaction boundaries. |
     1	# ADR-0005: Manual recommendation conversion creates a paused experiment
     2
     3	Status: Accepted
     4	Date: 2026-07-19
     5	Deciders: Project architecture
     6	Affected volumes: Volume VII; Volume VIII §11; Volume XI
     7	Supersedes: None
     8	Superseded by: None
     9
    10	## 1. Context and problem statement
    11
    12	Phase 4C persists an exact conversion proposal and a separate append-only human
    13	review history. A later explicit operation must be able to materialize an
    14	approved proposal without treating approval as execution, producing duplicate
    15	experiments under retries, or making the scheduler an implicit consumer.
    16
    17	## 2. Decision
    18
    19	An explicit manual command may convert one proposal only when its latest
    20	serialized review decision is `approve`.
    21
    22	- Proposal integrity is revalidated from its authoritative canonical evidence.
    23	- A transaction-scoped proposal-specific advisory lock serializes review writes
    24	  and conversion without granting runtime UPDATE access to immutable proposals.
    25	- One transaction resolves the current review, inserts one `paused`/`train`
    26	  experiment, and inserts immutable conversion evidence.
    27	- A unique proposal reference makes retries and concurrent identical requests
    28	  converge on the same experiment.
    29	- The audit row links the exact proposal, approving decision, and experiment.
    30	- The experiment is not pending, queued, running, resumed, or scheduler-owned.
    31	- A later explicit lifecycle command is required before any execution.
    32	- A completed conversion remains historical fact if a later review decision
    33	  reverses the proposal disposition.
    34
    35	## 3. Rationale and decision drivers
    36
    37	- Keep approval and materialization as separate durable events.
    38	- Preserve complete recommendation-to-experiment provenance.
    39	- Prevent ambiguous retry outcomes and duplicate experiment rows.
    40	- Keep worker launch, capacity, and continuation behavior unchanged.
    41
    42	## 4. Consequences
    43
    44	### 4.1 Positive consequences
    45
    46	- Manual conversion is atomic, auditable, and concurrency-safe.
    47	- Operators can inspect the created experiment before taking lifecycle action.
    48	- Scheduler restart alone cannot execute a newly converted proposal.
    49
    50	### 4.2 Negative consequences and trade-offs
    51
    52	- Conversion creates a paused experiment that requires another explicit action.
    53	- A conflicting pre-existing experiment identity prevents conversion rather
    54	  than silently linking unrelated work.
    55	- Review and conversion for the same proposal serialize on one advisory key.
    56
    57	### 4.3 Risks and mitigations
    58
    59	- Concurrent rejection versus conversion: both take the same transaction lock;
    60	  lock order determines which explicit operation is observed first. A lock-hash
    61	  collision only adds serialization and cannot establish proposal equality.
    62	- Configuration drift: the repository reconstructs and compares canonical
    63	  proposal and experiment invocation identities.
    64	- Retry after an uncertain response: the immutable execution row is returned.
    65
    66	## 5. Compatibility and migration
    67
    68	Migration 038 is additive. It creates only conversion-owned audit state and a
    69	composite review/proposal uniqueness index. It does not change scheduler,
    70	continuation, recommendation scoring, ranking, or review semantics.
    71
    72	## 6. Verification and operational evidence
    73
    74	- Migration constraints, restrictive foreign keys, and runtime privileges.
    75	- Pending/rejected denial and approved creation tests.
    76	- Concurrent identical conversion and exact replay tests.
    77	- Atomic rollback and immutable provenance tests.
    78	- CLI mutual exclusion and proof that the experiment remains paused.
    79
    80	## 7. Alternatives considered
    81
    82	### 7.1 Create a pending experiment
    83
    84	Rejected because scheduler startup could then execute conversion output without
    85	a separate explicit operational decision.
    86
    87	### 7.2 Convert as part of approval
    88
    89	Rejected because it collapses human governance and experiment creation into one
    90	event and makes review retries operationally consequential.
    91
    92	### 7.3 Queue immediately after conversion
    93
    94	Rejected because queueing, capacity, worker launch, and automation are outside
    95	Phase 4C Step 4 and remain under ADR-0004.
    96
    97	## 8. References
    98
    99	- [Volume VIII](../Volume_VIII_Recommendation_Engine.md)
   100	- [ADR-0003](ADR-0003-advisory-recommendation-evaluation.md)
   101	- [ADR-0004](ADR-0004-scheduler-ownership-boundaries.md)
   102
   103	## 9. Revision history
   104
   105	| Date | Change |
   106	|---|---|
   107	| 2026-07-19 | Accepted explicit, idempotent conversion to one paused experiment. |

exec
/bin/bash -lc 'for f in docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md; do nl -ba "$f"; done' in /Volumes/Developer SSD/ExpertAdvisor
 succeeded in 0ms:
     1	# ADR-0006: Phase 6A follow-up proposal is advisory and identity-bound
     2
     3	Status: Accepted
     4	Date: 2026-07-20
     5	Deciders: Project architecture
     6	Affected volumes: Volume VIII §§2–5, 9, 11
     7	Supersedes: None
     8	Superseded by: None
     9
    10	## 1. Context and problem statement
    11
    12	Phase 5A produces one immutable campaign outcome assessment, and Phase 5C
    13	produces one immutable advisory policy decision that may mark the exact
    14	assessment eligible for later operator review. That eligibility is not itself
    15	an approval, authorization, persisted review, experiment, or scheduler work
    16	item. The first Phase 6 increment needs a deterministic candidate that preserves
    17	the exact upstream identity without colliding conceptually with Phase 4D
    18	campaign materialization or silently reopening the scientific decision.
    19
    20	## 2. Decision
    21
    22	Phase 6A introduces the distinct pure domain entity
    23	`RecommendationCampaignFollowUpProposal` with contract version 1.
    24
    25	- It binds exactly one Phase 5A assessment v2 and one Phase 5C policy decision
    26	  v1 containing one policy v1.
    27	- Construction requires exact canonical-text, hash, campaign,
    28	  materialization, ordered-member, count, and embedded-assessment alignment.
    29	- Construction requires the Step 5C result to be eligible for operator review,
    30	  favorable, and explicitly non-authorizing. Phase 6A does not recalculate
    31	  eligibility.
    32	- Canonical text is authoritative identity. The tagged hash is only an
    33	  accelerator, and canonical text is compared even when hashes match.
    34	- Embedded canonical and delimiter-capable strings are byte-length framed.
    35	- Canonical text is independent of observation time and bounded to 1,048,576
    36	  bytes before immutable construction.
    37	- The proposal is read-only, database-free, non-persistent, advisory,
    38	  non-authoritative, unapproved, inactive, non-executing, non-authorizing,
    39	  non-scheduler work, and does not declare campaign success.
    40	- Phase 6A adds no repository, SQL, migration, service, CLI, formatter,
    41	  persisted operator review, scheduler consumer, queue, worker, or lifecycle
    42	  mutation.
    43	- Persistence, review, activation, and execution remain separate possible
    44	  later Phase 6 steps and receive no authority from this ADR.
    45
    46	## 3. Rationale and decision drivers
    47
    48	- Preserve exact upstream scientific and policy provenance without copying or
    49	  reinterpreting their truth tables.
    50	- Keep operator-review eligibility visibly separate from approval and action.
    51	- Make collision handling safe by retaining authoritative canonical text.
    52	- Avoid the misleading implication that an advisory proposal is Phase 4D
    53	  materialization or any experiment lifecycle request.
    54	- Keep later persistence and operational designs free to establish their own
    55	  authorities, transactions, idempotency, and concurrency contracts.
    56
    57	## 4. Consequences
    58
    59	### 4.1 Positive consequences
    60
    61	- Identical validated Phase 5 inputs yield an identical Phase 6A identity
    62	  regardless of locale or observation time.
    63	- Every relevant upstream identity and explicit non-authority semantic is
    64	  reviewable in one immutable candidate.
    65	- Unsupported, mismatched, non-favorable, noneligible, authorizing, or
    66	  oversized inputs fail closed with stable reasons.
    67	- No database, scheduler, or execution dependency enters the pure boundary.
    68
    69	### 4.2 Negative consequences and trade-offs
    70
    71	- Binding complete upstream canonical text duplicates data and can produce a
    72	  sizable canonical value.
    73	- The 1 MiB ceiling may reject exceptionally large but otherwise valid Phase 5
    74	  decisions; changing that ceiling requires explicit versioned design.
    75	- Phase 6A alone cannot record or enact an operator decision.
    76
    77	### 4.3 Risks and mitigations
    78
    79	- Hash collision: compare authoritative canonical text after any hash match.
    80	- Upstream identity drift: validate exact version/canonical/hash and direct
    81	  campaign, materialization, and ordered-member fields.
    82	- Accidental authority creep: compile-time false safety flags, canonical
    83	  bindings, pure tests, and absence of persistence/scheduler integration.
    84	- Oversized future payload: reject above the fixed byte ceiling before
    85	  proposal construction.
    86
    87	## 5. Compatibility and migration
    88
    89	The change is additive C++ domain code and documentation. It requires no
    90	schema, data, privilege, repository, CLI, runtime, or scheduler migration and
    91	does not modify Phase 5A, 5B, or 5C contracts.
    92
    93	## 6. Verification and operational evidence
    94
    95	- Golden canonical text and independently fixed tagged hash.
    96	- Exact upstream alignment and stable refusal-reason tests.
    97	- Favorable/non-favorable, invariance, identity-sensitivity, IEEE edge, size,
    98	  malformed-version, and construction-access tests.
    99	- Focused normal, ASan, and UBSan runs plus Phase 4D/5 regression tests.
   100	- Both existing Xcode targets compile the source in an incremental Release
   101	  build; no Debug or clean build is required.
   102
   103	## 7. Alternatives considered
   104
   105	### 7.1 Generic materialization entity
   106
   107	Rejected because it could be confused with the persisted Phase 4D operation
   108	that materializes an approved campaign into Phase 4C proposals.
   109
   110	### 7.2 Build from Step 5B formatted output
   111
   112	Rejected because presentation text is not the immutable typed Phase 5A/5C
   113	contract and would weaken exact identity alignment.
   114
   115	### 7.3 Recompute Step 5C eligibility in Phase 6A
   116
   117	Rejected because it would duplicate policy truth, risk divergence, and reopen
   118	scientific interpretation outside the versioned Step 5C owner.
   119
   120	### 7.4 Persist or approve the proposal in Phase 6A
   121
   122	Rejected because persistence, operator review, approval, activation, and
   123	execution require distinct later authority and operational designs.
   124
   125	## 8. References
   126
   127	- [Volume VIII](../Volume_VIII_Recommendation_Engine.md)
   128	- [Phase 5 outcome assessment and policy](../../Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst)
   129	- [Phase 6A follow-up proposal](../../Phase6ARecommendationCampaignFollowUpProposal.rst)
   130	- [ADR-0003](ADR-0003-advisory-recommendation-evaluation.md)
   131	- [ADR-0004](ADR-0004-scheduler-ownership-boundaries.md)
   132	- [ADR-0005](ADR-0005-manual-recommendation-conversion.md)
   133
   134	## 9. Revision history
   135
   136	| Date | Change |
   137	|---|---|
   138	| 2026-07-20 | Accepted the pure identity-bound, non-authorizing Phase 6A follow-up proposal. |
     1	# ADR-0007: Phase 6B persists exact advisory proposals for read-only preview
     2
     3	Status: Accepted
     4	Date: 2026-07-20
     5	Deciders: Project architecture
     6	Affected volumes: Volume VIII §§2–9, 11; Volume XII
     7	Supersedes: None
     8	Superseded by: None
     9
    10	## 1. Context and problem statement
    11
    12	Phase 6A produces a pure immutable, non-authorizing follow-up proposal with a
    13	complete canonical identity. Operators need durable, repeatable inspection of
    14	that exact value. Persistence must not turn eligibility into approval or
    15	introduce activation, execution, scheduling, experiment creation, or follow-up
    16	authority. Reload must also reject database corruption rather than returning a
    17	mutable or partially validated approximation.
    18
    19	## 2. Decision
    20
    21	Phase 6B adds one append-only PostgreSQL manifest plus ordered member table,
    22	one collision-aware repository, a private validated hydration seam, and a
    23	read-only preview service.
    24
    25	- The manifest and members preserve every public Phase 6A field exactly.
    26	- Canonical text is authoritative; the tagged hash only selects a lock/index
    27	  bucket. Same-hash/different-canonical proposals receive different storage
    28	  collision ordinals.
    29	- Exact retries return the existing row. A hash-scoped transaction advisory
    30	  lock serializes concurrent exact persistence.
    31	- Upstream approval/materialization/member triggers validate copied immutable
    32	  provenance, and a deferred trigger requires complete ordered membership.
    33	- Reload reconstructs the non-assignable Phase 6A object through one private
    34	  friend builder and reruns direct alignment plus full canonical/payload
    35	  invariants.
    36	- Preview runs under repeatable-read, read-only isolation and renders explicit
    37	  negative authority fields.
    38	- Runtime privileges are append-only and column-scoped. Phase 6B provides no
    39	  approval, activation, execution, queue, scheduler, experiment, or follow-up
    40	  authorization API or schema.
    41
    42	## 3. Rationale and decision drivers
    43
    44	- Preserve the identity produced by Phase 6A without parsing presentation
    45	  output or recomputing Phase 5 science/policy.
    46	- Make persisted corruption and duplicate exact identity fail closed.
    47	- Keep collision safety aligned with existing canonical-first repositories.
    48	- Make operator inspection useful while keeping it observably read-only.
    49	- Prevent persistence metadata such as row ID, collision ordinal, and creation
    50	  timestamp from changing proposal identity.
    51
    52	## 4. Consequences
    53
    54	### 4.1 Positive consequences
    55
    56	- A persisted proposal reloads as the same immutable typed value.
    57	- Exact canonical and every upstream canonical/hash pair survive round trip.
    58	- Exact retries are idempotent and hash collisions do not merge identities.
    59	- Database constraints, triggers, privileges, and C++ hydration provide
    60	  defense in depth against incomplete or malformed records.
    61	- Preview cannot mutate lifecycle or scheduler state.
    62
    63	### 4.2 Negative consequences and trade-offs
    64
    65	- Full canonical provenance is duplicated and may approach the Phase 6A 1 MiB
    66	  ceiling.
    67	- Inserts depend on existing immutable Phase 4D approval/materialization rows.
    68	- Owner-level database corruption remains possible, but repository reload
    69	  detects it and fails closed.
    70	- Previewing full authoritative canonical text can produce large output.
    71
    72	### 4.3 Risks and mitigations
    73
    74	- Hash collision: lock/index by hash, then compare canonical text exactly and
    75	  retain a collision ordinal outside identity.
    76	- Partial manifest: insert in one transaction and enforce member completeness
    77	  with a deferred constraint trigger.
    78	- Authority creep: no decision/lifecycle schema or API and explicit false
    79	  safety fields in every preview record.
    80	- Mutable reconstruction: return the original const-field Phase 6A type and a
    81	  non-assignable persisted wrapper.
    82
    83	## 5. Compatibility and migration
    84
    85	Migration ``042`` is additive and depends on the existing Phase 4D approval
    86	and materialization tables. It does not alter experiment or scheduler tables.
    87	The Phase 6A public builder and canonical grammar remain version 1 and
    88	unchanged; constructor validation is strengthened for exact hydration.
    89
    90	## 6. Verification and operational evidence
    91
    92	- Repeatable migration, exact privilege, FK, trigger, index, and collation
    93	  assertions.
    94	- Exact round-trip, immutable reload, canonical/hash preservation, idempotent
    95	  duplicate replay, duplicate persisted identity rejection, and malformed row
    96	  rejection.
    97	- Preview content and repeatable-read/read-only behavior with an untouched
    98	  sequence sentinel.
    99	- Phase 6A, Phase 5A/B/C, relevant Phase 4D regression tests, Release build,
   100	  and whitespace audit.
   101
   102	## 7. Alternatives considered
   103
   104	### 7.1 Store only the proposal hash
   105
   106	Rejected because hashes are not authoritative and cannot reconstruct or
   107	inspect exact provenance.
   108
   109	### 7.2 Store only canonical text and parse it on every read
   110
   111	Rejected because Phase 6A intentionally defines canonical generation, not a
   112	public parser, and normalized member rows support provenance constraints and
   113	bounded typed reload without making presentation text authoritative input.
   114
   115	### 7.3 Return a mutable persistence DTO
   116
   117	Rejected because callers could mistake reconstructed fields for a validated
   118	Phase 6A proposal and silently lose immutability guarantees.
   119
   120	### 7.4 Add operator approval or execution state
   121
   122	Rejected as a distinct later authority explicitly outside Phase 6B.
   123
   124	## 8. References
   125
   126	- [Volume VIII](../Volume_VIII_Recommendation_Engine.md)
   127	- [Phase 6A proposal](../../Phase6ARecommendationCampaignFollowUpProposal.rst)
   128	- [Phase 6B persistence and preview](../../Phase6BRecommendationCampaignFollowUpProposalPersistence.rst)
   129	- [ADR-0001](ADR-0001-postgresql-source-of-truth.md)
   130	- [ADR-0004](ADR-0004-scheduler-ownership-boundaries.md)
   131	- [ADR-0006](ADR-0006-phase-6a-follow-up-proposal.md)
   132
   133	## 9. Revision history
   134
   135	| Date | Change |
   136	|---|---|
   137	| 2026-07-20 | Accepted exact append-only Phase 6B proposal persistence and read-only preview. |
     1	# ADR-0008: Phase 6C records one immutable non-authorizing administrative review
     2
     3	Status: Accepted
     4	Date: 2026-07-21
     5	Deciders: Project architecture
     6	Affected volumes: Volume VIII §§2–11; Volume XII §§2–11
     7	Supersedes: None
     8	Superseded by: None
     9
    10	## 1. Context and problem statement
    11
    12	Phase 6B durably preserves an exact Phase 6A advisory follow-up proposal but
    13	intentionally records no operator decision. An operator must be able to approve
    14	or reject one persisted proposal without turning approval into activation,
    15	execution, follow-up authorization, scheduling, queueing, experiment mutation,
    16	or campaign-success evidence. Concurrent attempts, exact retries, corrupted
    17	storage, and owner-level provenance drift must have deterministic fail-closed
    18	behavior.
    19
    20	## 2. Decision
    21
    22	Phase 6C defines one pure immutable review value, migration 043 with one
    23	append-only review-event table, an exact-binding repository, and read-only show
    24	and list presentation.
    25
    26	- Contract version 1 binds persisted proposal ID, exact Phase 6A
    27	  version/canonical/hash, approved or rejected decision, reviewer, and reason.
    28	- Canonical text is authoritative and byte-length-prefixed; the tagged hash is
    29	  an accelerator. Generated event ID and timestamp remain outside identity.
    30	- The table has one restrictive Phase 6B foreign key, exact-provenance insert
    31	  enforcement, bytewise canonical collation, and one-event-per-proposal
    32	  uniqueness.
    33	- A proposal-ID-scoped transaction advisory lock serializes review attempts.
    34	  First persistence is ``recorded``, exact replay is ``existing_identical``,
    35	  and every non-identical attempt for the same proposal conflicts.
    36	- Reload reconstructs the immutable value, validates its full canonical
    37	  identity, reloads Phase 6B, and compares the exact proposal binding.
    38	- Show/list run repeatable-read and read-only, order lists by descending event
    39	  ID, and render explicit negative authority fields.
    40	- Runtime access is SELECT, payload-column INSERT, and sequence USAGE only.
    41	  Existing events are never updated, deleted, expired, or superseded.
    42
    43	An approved event means only administrative approval for possible
    44	consideration by a later explicitly authorized phase.
    45
    46	## 3. Rationale and decision drivers
    47
    48	- Make the operator decision explicit without weakening the Phase 6A/6B
    49	  advisory boundary.
    50	- Bind review to authoritative canonical proposal content rather than trusting
    51	  a hash or row ID alone.
    52	- Make rejected decisions as durable and auditable as approved decisions.
    53	- Give exact retry and concurrent conflict behavior one deterministic outcome.
    54	- Keep later action authority absent from both schema and API.
    55
    56	## 4. Consequences
    57
    58	### 4.1 Positive consequences
    59
    60	- One persisted proposal has at most one immutable administrative decision.
    61	- Exact retries converge and differing reviewer/reason/decision payloads cannot
    62	  silently replace history.
    63	- Constraints, trigger provenance, typed hydration, canonical recomputation,
    64	  and Phase 6B reload provide defense in depth.
    65	- Operators can inspect review history without locks, writes, or sequence use.
    66
    67	### 4.2 Negative consequences and trade-offs
    68
    69	- Proposal canonical text is duplicated in the review event to preserve exact
    70	  self-contained binding and may approach one MiB.
    71	- Phase 6C cannot correct an operator mistake in place; a future explicitly
    72	  designed phase would need a separate append-only policy if correction is
    73	  ever authorized.
    74	- List hydration reloads each referenced Phase 6B proposal to validate current
    75	  durable agreement.
    76
    77	### 4.3 Risks and mitigations
    78
    79	- Authority creep: no action/lifecycle columns or adapters, plus fixed negative
    80	  canonical and presentation fields.
    81	- Concurrent disagreement: one proposal-scoped transaction advisory lock and
    82	  unique proposal ID.
    83	- Hash collision or corruption: canonical equality is authoritative and all
    84	  hashes are recomputed on reload.
    85	- Upstream drift: restrictive foreign key, insert trigger, and repository
    86	  comparison against immutable Phase 6B storage.
    87
    88	## 5. Compatibility and migration
    89
    90	Migration 043 is additive after migration 042. It alters no Phase 6A/6B row,
    91	experiment table, scheduler table, or lifecycle schema. Existing Phase 6A and
    92	6B contract semantics and identities remain unchanged. Rollback, if ever
    93	operationally approved, consists only of separately removing the new Phase 6C
    94	objects; no production rollback is performed by the implementation work.
    95
    96	## 6. Verification and operational evidence
    97
    98	- Pure contract tests for both decisions, immutable types, deterministic and
    99	  locale-independent identity, sensitivity, malformed inputs, and negative
   100	  authority invariants.
   101	- Migration and repository tests for repeatability, C collation, restrictive
   102	  FK, least privilege, generated columns, exact round trip/replay, all conflict
   103	  forms, concurrency, rollback, malformed storage, and provenance mismatch.
   104	- Read-only show/list tests for not-found, ordering, limit validation, exact
   105	  safety output, unchanged sequences/tables, and absence of advisory/tuple
   106	  locks.
   107	- Phase 6A/6B, Phase 5, relevant Phase 4D, project validation, and Release
   108	  build regressions.
   109
   110	## 7. Alternatives considered
   111
   112	### 7.1 Add approved state to the Phase 6B proposal row
   113
   114	Rejected because it mutates advisory proposal evidence and obscures the
   115	separate administrative decision.
   116
   117	### 7.2 Treat persistence or preview as approval
   118
   119	Rejected because observation and storage do not establish reviewer identity or
   120	intent.
   121
   122	### 7.3 Let approval activate or queue follow-up work
   123
   124	Rejected because action authority requires a later explicit contract,
   125	transaction, and operational safety decision.
   126
   127	### 7.4 Allow multiple current/superseding review states
   128
   129	Rejected for Phase 6C. One immutable effective decision gives deterministic
   130	replay and conflict semantics without inventing expiration or correction
   131	policy.
   132
   133	## 8. References
   134
   135	- [Volume VIII](../Volume_VIII_Recommendation_Engine.md)
   136	- [Volume XII](../Volume_XII_Database.md)
   137	- [Phase 6A proposal](../../Phase6ARecommendationCampaignFollowUpProposal.rst)
   138	- [Phase 6B persistence and preview](../../Phase6BRecommendationCampaignFollowUpProposalPersistence.rst)
   139	- [Phase 6C administrative review](../../Phase6CRecommendationCampaignFollowUpProposalReview.rst)
   140	- [ADR-0001](ADR-0001-postgresql-source-of-truth.md)
   141	- [ADR-0004](ADR-0004-scheduler-ownership-boundaries.md)
   142	- [ADR-0006](ADR-0006-phase-6a-follow-up-proposal.md)
   143	- [ADR-0007](ADR-0007-phase-6b-follow-up-proposal-persistence.md)
   144
   145	## 9. Revision history
   146
   147	| Date | Change |
   148	|---|---|
   149	| 2026-07-21 | Accepted exact append-only Phase 6C administrative review without action authority. |
     1	# ADR-0009: Phase 6D ratifies governance advancement after approved review
     2
     3	Status: Proposed
     4	Date: 2026-07-22
     5	Deciders: Project architecture
     6	Affected volumes: Volume VIII §§2–11; Volume XII §§2–11
     7	Supersedes: None
     8	Superseded by: None
     9
    10	## 1. Context and problem statement
    11
    12	Phase 6C already records one immutable approved or rejected administrative
    13	review of an exact persisted Phase 6B proposal. Its positive decision answers
    14	whether the proposal is acceptable on its merits for possible advancement.
    15	Repeating that approval in Phase 6D would create no distinct authority.
    16
    17	The next increment instead needs explicit governance evidence that an
    18	independent authorized actor has inspected the exact approved review and full
    19	provenance and ratified advancement into the next separately controlled
    20	phase. That evidence must not itself grant the next phase's capabilities.
    21
    22	## 2. Proposed decision
    23
    24	Phase 6D defines pure immutable governance ratification, additive migration
    25	044, an exact-binding append-only repository, and one transactional service.
    26
    27	- Phase 6C remains the merits review with `approved` or `rejected` outcomes.
    28	- Phase 6D records only `ratified`: advancement of the exact approved review
    29	  into the next separately controlled phase.
    30	- Ratifier identity MUST differ byte-for-byte from the Phase 6C reviewer
    31	  identity. Self-ratification fails closed.
    32	- The authority role is fixed as `follow_up_governance_ratifier`; callers do
    33	  not supply arbitrary role text.
    34	- Contract version 1 binds the exact review-event ID, review and proposal
    35	  version/canonical/hash identities, approved review decision, reviewer,
    36	  fixed role, ratified decision, ratifier, basis, separation rule, and fixed
    37	  negative authority semantics.
    38	- Canonical text is authoritative, byte-length framed, locale-independent,
    39	  and bounded. The tagged hash is only an accelerator.
    40	- Migration 044 stores one ratification per review/proposal with restrictive
    41	  provenance, approved eligibility, fixed-role, ratified-decision, and
    42	  separation-of-duties enforcement.
    43	- A review-event-scoped transaction advisory lock serializes writes. First
    44	  persistence is `recorded`, exact replay is `existing_identical`, and any
    45	  changed payload conflicts.
    46	- Runtime access is table `SELECT`, payload-column `INSERT`, and sequence
    47	  `USAGE` only. Existing ratifications cannot be updated or removed.
    48	- Phase 6D adds no CLI or operational adapter.
    49
    50	Ratification is governance evidence only. It grants no Phase 6E capability,
    51	follow-up authorization, execution authorization, or lifecycle authority.
    52
    53	## 3. Rationale and decision drivers
    54
    55	- Give Phase 6D a materially distinct governance question and consequence.
    56	- Require independent human accountability at the advancement boundary.
    57	- Consume authoritative persisted evidence without recomputing Phase 5 or
    58	  Phase 6A truth.
    59	- Preserve canonical-first collision safety and deterministic retry behavior.
    60	- Keep every operational capability absent until separately designed and
    61	  accepted.
    62
    63	## 4. Consequences
    64
    65	### 4.1 Positive consequences
    66
    67	- The audit trail distinguishes proposal merits review from governance
    68	  ratification of advancement.
    69	- One actor cannot both approve the Phase 6C review and ratify it in Phase 6D.
    70	- Full typed provenance and defense-in-depth database enforcement detect stale,
    71	  forged, malformed, or self-ratified evidence.
    72	- Equivalent retries and concurrent requests converge without replacing
    73	  history.
    74
    75	### 4.2 Negative consequences and trade-offs
    76
    77	- A second governance actor is required before a proposal can be ratified.
    78	- Complete review and proposal canonical text is duplicated and may exceed two
    79	  MiB per event.
    80	- A mistaken ratification cannot be changed in place; correction would require
    81	  a separately accepted append-only design.
    82	- No main-program CLI exists in this increment.
    83
    84	### 4.3 Risks and mitigations
    85
    86	- Semantic duplication: distinct question, `ratified` decision, fixed role,
    87	  separation policy, and advancement-only canonical semantics.
    88	- Self-ratification: pure, service, repository, check-constraint, and trigger
    89	  enforcement.
    90	- Hash collision: compare complete authoritative canonical and typed payload.
    91	- Upstream drift: restrictive foreign keys, joined provenance trigger, and
    92	  validated Phase 6C repository reload.
    93	- Privilege escalation: explicit PUBLIC/runtime revokes, column grants,
    94	  invoker rights, pinned search path, and NULL-ACL fallback tests.
    95
    96	## 5. Compatibility and migration
    97
    98	Migration 044 is uncommitted and undeployed, so it is replaced in place rather
    99	than followed by migration 045. It is additive after migration 043 and does
   100	not alter Phase 6A, 6B, or 6C rows, identities, APIs, or privileges. It touches
   101	no experiment, model, continuation, lifecycle, queue, or scheduler table.
   102
   103	## 6. Verification and operational evidence
   104
   105	- Pure warnings-as-errors tests for immutable semantics, fixed role and
   106	  decisions, golden canonical/hash, locale, UTF-8/DEL validation, separation
   107	  of duties, identity sensitivity, and negative authority.
   108	- Isolated PostgreSQL clean-install and 043-upgrade tests for repeatability,
   109	  exact ACL/NULL-ACL behavior, provenance, fixed role, distinct actors,
   110	  service syntax, replay/conflict, concurrency, rollback, malformed storage,
   111	  collision handling, read-only queries, and upstream sentinels.
   112	- Phase 6A/6B/6C regression tests, exact Xcode project membership, diff audit,
   113	  and an incremental Release build only when scheduler-worker load is safe.
   114
   115	## 7. Alternatives considered
   116
   117	### 7.1 Keep a second administrative approval
   118
   119	Rejected because it repeats Phase 6C without a distinct question, role,
   120	policy, or consequence.
   121
   122	### 7.2 Allow the Phase 6C reviewer to ratify
   123
   124	Rejected because no committed authority permits self-ratification and it
   125	would remove the independent governance check that distinguishes Phase 6D.
   126
   127	### 7.3 Treat Phase 6C approval as automatic ratification
   128
   129	Rejected because merits review does not establish the separate governance
   130	decision to advance.
   131
   132	### 7.4 Add Phase 6E activation or execution authority
   133
   134	Rejected because the next phase requires a separate accepted capability and
   135	operational safety design.
   136
   137	## 8. References
   138
   139	- [Volume VIII](../Volume_VIII_Recommendation_Engine.md)
   140	- [Volume XII](../Volume_XII_Database.md)
   141	- [Phase 6A proposal](../../Phase6ARecommendationCampaignFollowUpProposal.rst)
   142	- [Phase 6B persistence](../../Phase6BRecommendationCampaignFollowUpProposalPersistence.rst)
   143	- [Phase 6C review](../../Phase6CRecommendationCampaignFollowUpProposalReview.rst)
   144	- [Phase 6D ratification](../../Phase6DRecommendationCampaignFollowUpProposalRatification.rst)
   145	- [ADR-0001](ADR-0001-postgresql-source-of-truth.md)
   146	- [ADR-0004](ADR-0004-scheduler-ownership-boundaries.md)
   147	- [ADR-0006](ADR-0006-phase-6a-follow-up-proposal.md)
   148	- [ADR-0007](ADR-0007-phase-6b-follow-up-proposal-persistence.md)
   149	- [ADR-0008](ADR-0008-phase-6c-follow-up-proposal-administrative-review.md)
   150
   151	## 9. Revision history
   152
   153	| Date | Change |
   154	|---|---|
   155	| 2026-07-22 | Proposed distinct Phase 6D governance ratification with mandatory separation of duties and no Phase 6E capability. |

exec
/bin/bash -lc "nl -ba docs/Phase6DRecommendationCampaignFollowUpProposalRatification.rst && nl -ba docs/architecture/Volume_VIII_Recommendation_Engine.md | sed -n '1,240p' && nl -ba docs/architecture/Volume_VIII_Recommendation_Engine.md | sed -n '240,520p' && nl -ba docs/architecture/Volume_VIII_Recommendation_Engine.md | sed -n '520,840p'" in /Volumes/Developer SSD/ExpertAdvisor
 succeeded in 0ms:
     1	Phase 6D Recommendation Campaign Follow-Up Proposal Governance Ratification
     2	===========================================================================
     3
     4	Purpose and distinct authority
     5	------------------------------
     6
     7	Phase 6C and Phase 6D answer different governance questions.
     8
     9	Phase 6C asks whether one exact persisted Phase 6B follow-up proposal is
    10	acceptable on its merits for possible advancement.  An identified reviewer
    11	records one immutable ``approved`` or ``rejected`` administrative review.
    12
    13	Phase 6D asks whether, given that exact approved review and its complete
    14	authoritative provenance, an authorized governance actor ratifies advancement
    15	of the reviewed proposal into the next separately controlled phase.  The only
    16	positive Phase 6D decision is ``ratified``.  Ratification is immutable
    17	governance evidence; it does not grant any capability in Phase 6E.
    18
    19	Eligibility and separation of duties
    20	------------------------------------
    21
    22	Only one exact persisted, fully reloadable Phase 6C ``approved`` review is
    23	eligible.  A rejected, missing, malformed, stale, unsupported, or
    24	identity-inconsistent review fails closed.  The ratifier identity must differ
    25	byte-for-byte from the Phase 6C reviewer identity.  Self-ratification is
    26	rejected by pure construction, service and repository validation, a database
    27	check constraint, and the provenance trigger.
    28
    29	The ratifier acts under the fixed version-1 role
    30	``follow_up_governance_ratifier``.  Callers cannot select or inject a different
    31	role.  Phase 6D never rebuilds the Phase 6A proposal or recomputes Phase 5
    32	assessment or policy truth; validated Phase 6C reload transitively validates
    33	the exact persisted Phase 6B proposal and embedded Phase 6A identity.
    34
    35	Immutable domain contract
    36	-------------------------
    37
    38	``RecommendationCampaignFollowUpProposalRatification`` is a pure,
    39	database-free, non-assignable contract-version-1 value.  It binds:
    40
    41	* the positive generated Phase 6C review-event ID;
    42	* exact Phase 6C review version, canonical text, tagged hash, ``approved``
    43	  decision, and reviewer identity;
    44	* exact reviewed Phase 6B proposal ID and Phase 6A version, canonical text,
    45	  and tagged hash;
    46	* the fixed ``follow_up_governance_ratifier`` authority role;
    47	* the Phase 6D ``ratified`` decision;
    48	* a distinct ratifier identity; and
    49	* a non-empty ratification basis.
    50
    51	Reviewer and ratifier identities use the stable 1--128 byte governance-actor
    52	syntax: ASCII alphanumeric first, followed only by ASCII alphanumeric or
    53	``._@:/+-``.  Ratification basis is 1--4096 bytes of valid UTF-8, preserves
    54	its exact bytes, requires non-whitespace content, and rejects NUL, DEL, and
    55	disallowed control bytes.
    56
    57	Canonical identity
    58	------------------
    59
    60	The canonical grammar begins with
    61	``experiment_recommendation_campaign_follow_up_proposal_ratification_v1``.
    62	It encodes the exact review and proposal provenance, reviewer, approved-review
    63	eligibility, fixed authority role, ``ratified`` decision, ratifier, basis,
    64	``separation_of_duties_required=true``, and the advancement-only governance
    65	meaning.  It also fixes ``phase_6e_capability_granted=false`` and all negative
    66	operational authority fields.
    67
    68	Delimiter-capable values use byte-length framing.  Decimal integers and byte
    69	lengths use the classic locale.  Canonical text is authoritative and bounded
    70	to 2,113,536 bytes; the tagged FNV-1a hash is only an accelerator.  Generated
    71	ratification-event ID and timestamp remain persistence metadata outside
    72	identity.
    73
    74	Persistence, replay, and concurrency
    75	------------------------------------
    76
    77	Migration ``044`` adds the append-only
    78	``experiment_recommendation_campaign_follow_up_ratification_event``
    79	table.  Restrictive foreign keys bind the exact review and proposal.  Fixed
    80	checks enforce the approved upstream decision, ratified Phase 6D decision,
    81	authority role, and separation of duties.  An invoker-rights trigger with a
    82	pinned safe search path verifies the complete Phase 6C/6B provenance,
    83	reviewer identity, eligibility, and distinct ratifier.  At most one
    84	ratification exists per exact review and proposal.
    85
    86	The repository validates the pure value, takes a transaction-scoped advisory
    87	lock over the review-event conflict domain, reloads the exact Phase 6C/6B
    88	chain, and inserts atomically.  The first valid write returns ``recorded``;
    89	exact replay returns ``existing_identical``.  A changed ratifier, basis, role,
    90	upstream identity, or canonical payload conflicts.  Equivalent concurrent
    91	writes converge; conflicting writes produce one winner and one deterministic
    92	conflict.  Canonical equality remains authoritative over hash equality.
    93
    94	Lookup by ratification-event, review-event, or proposal ID and bounded list
    95	operations are read-only.  Hydration rebuilds the immutable value and
    96	revalidates the upstream chain.  Reads do not advance sequences or acquire
    97	advisory or tuple locks.
    98
    99	Service boundary
   100	----------------
   101
   102	``RatifyRecommendationCampaignFollowUpProposal`` accepts only the exact
   103	review-event ID, expected review identity hash, ratifier identity, and
   104	ratification basis.  It validates request syntax, starts one transaction,
   105	loads the exact review, verifies expected identity and approved eligibility,
   106	enforces separation of duties, constructs the pure value once, persists it,
   107	and returns the authoritative result after commit.  The authority role and
   108	decision are fixed by the contract, not supplied by the request.
   109
   110	No main-program CLI is added in Phase 6D.
   111
   112	Privileges and explicit non-authorities
   113	---------------------------------------
   114
   115	Runtime role ``pqxx`` receives only table ``SELECT``, payload-column
   116	``INSERT``, and sequence ``USAGE``.  It cannot insert generated fields or
   117	update, delete, or truncate ratifications.  ``PUBLIC`` and runtime receive no
   118	trigger-function execution.  Privilege verification treats NULL ACLs through
   119	PostgreSQL defaults, including lowercase sequence object type ``s``.
   120
   121	Ratification does not activate or execute anything; authorize follow-up or
   122	campaign execution; queue or schedule work; signal or start the scheduler;
   123	launch workers; create or modify experiments, models, recommendations, or
   124	continuations; declare campaign success or profitability; or claim
   125	statistical validation.  Phase 6E remains a separate authority because any
   126	later capability requires its own accepted contract, transaction, safety
   127	rules, and explicit consumer.
   128
   129	Verification and operational safety
   130	-----------------------------------
   131
   132	Focused tests cover the fixed golden identity hash, locale independence,
   133	valid UTF-8, DEL and hostile-control rejection, approved-only eligibility,
   134	fixed role, self-ratification refusal, distinct-actor success, exact replay,
   135	changed-payload conflict, hash collisions, concurrency, rollback, malformed
   136	storage, append-only least privilege, NULL-ACL fallback, safe trigger context,
   137	read-only queries, clean install, upgrade from 043, repeatability, and
   138	unchanged Phase 6A/6B/6C, experiment, and scheduler evidence.
     1	# Volume VIII — Recommendation Engine
     2
     3	Status: Foundation aligned through Phase 6C; proposed Phase 6D recorded
     4	Version: 2.4.0
     5	Last revised: 2026-07-22
     6
     7	## 1. Purpose
     8
     9	Define the research-recommendation subsystem: deterministic identity, candidate
    10	generation, persistence, duplicate handling, scoring/ranking, explicit human
    11	review, immutable advisory evidence, the explicitly separated manual conversion
    12	chain, read-only campaign planning/review, and explicit non-executing campaign
    13	approval, materialization, handoff observation, and exact materialized-proposal
    14	review.
    15	Phase 5 Step 1 adds an explicit atomic convenience that applies the existing
    16	Phase 4C paused conversion execution to every exact materialized campaign
    17	member, without activation, queueing, scheduler control, or workers.
    18	Phase 5 Step 2 adds the separately confirmed atomic convenience over existing
    19	Phase 4C activation, transitioning every exact eligible member from
    20	``paused/train`` to ``pending/train`` without starting scheduler or workers.
    21	Phase 5 Step 3 adds a distinct confirmed one-transaction launch convenience
    22	that composes those same execution and activation authorities for the exact
    23	membership without adding campaign state or scheduler behavior.
    24	Phase 5 Step 4 adds one deterministic read-only operational snapshot of every
    25	exact member's current Phase 4C, experiment, worker-metadata, inference, and
    26	analysis evidence without adding durable campaign state or scheduler behavior.
    27	Phase 5 Steps 5a–5b add a pure, non-authoritative scientific outcome assessment
    28	and its bounded, read-only authoritative-persisted-evidence integration. Step
    29	5c adds a separate pure, versioned advisory policy over that immutable
    30	assessment; it interprets but does not alter evidence, declare campaign
    31	success, or authorize follow-up.
    32	Phase 6A adds one pure immutable follow-up proposal bound to an exactly aligned
    33	eligible favorable Step 5C decision and its exact Step 5A assessment. The
    34	proposal is only an advisory candidate for later operator review and adds no
    35	persistence, approval, lifecycle, or scheduler authority.
    36	Phase 6B durably records that exact immutable advisory value and exposes a
    37	repeatable-read, read-only operator preview. Persistence adds no operator
    38	decision, approval, activation, execution, experiment, queue, scheduler, or
    39	follow-up authorization authority.
    40	Phase 6C records exactly one immutable approved or rejected administrative
    41	review for one exact persisted Phase 6B proposal. Approval permits only possible
    42	consideration by a later explicitly authorized phase; it grants no action,
    43	follow-up, lifecycle, experiment, worker, or scheduler authority.
    44	Phase 6D separately records one immutable governance ratification for
    45	advancement of one exact persisted Phase 6C ``approved`` review into the next
    46	separately controlled phase. The fixed ratifier role and mandatory reviewer/
    47	ratifier separation make this authority distinct from merits review. It grants
    48	no Phase 6E capability or operational authority.
    49
    50	## 2. Scope
    51
    52	### 2.1 In scope
    53
    54	Recommendation policy and identities, pure candidates, scans, source evidence,
    55	duplicates, immutable scores/components, deterministic ranks/explanations,
    56	terminal review transitions, immutable review events, and deterministic
    57	evaluation classification/history, ranking snapshots and comparisons, plus the
    58	pure manually invoked proposed-experiment specification contract.
    59	The explicit Phase 4C manual conversion chain and Phase 4D campaign planning,
    60	review, explicit non-executing approval, and approved-campaign materialization
    61	into the existing Phase 4C proposal boundary, read-only handoff, and explicit
    62	campaign-wide Phase 4C proposal review are also in scope.
    63	The pure Phase 5 outcome contracts and the pure identity-bound Phase 6A
    64	follow-up proposal are in scope as advisory values. Phase 6B exact append-only
    65	proposal persistence/read-only preview and Phase 6C explicit immutable
    66	administrative review/read-only presentation are also in scope. Phase 6D exact
    67	append-only governance ratification and its transactional typed service are in
    68	scope without Phase 6E, activation, or execution capability.
    69
    70	### 2.2 Out of scope
    71
    72	Automatic proposal creation, autonomous campaign execution or campaign
    73	management, profitability claims,
    74	recommendation-driven scheduler polling, worker execution, and automatic
    75	approval/rejection/expiration.
    76
    77	### 2.3 Current implementation status
    78
    79	Phase 4A Steps 1–5, Phase 4B Steps 1–2, Phase 4C Steps 1–6, Phase 4D Steps 1–6,
    80	Phase 5 Steps 1–5c and Phases 6A–6D implement the in-scope capabilities.
    81	Phase 4B Step 1 classifies current persisted provenance and reuses the Step 4
    82	score formula unchanged. Step 2 ranks only persisted Step 1 results, stores
    83	exact snapshot membership, and compares compatible persisted components.
    84	Phase 4C provides the explicit proposal, review, paused conversion, activation,
    85	and workflow-observation chain. Phase 4D Step 1 reads one explicit immutable
    86	ranking snapshot and Phase 4C workflow evidence to produce a deterministic,
    87	bounded campaign plan without persisting or executing it. Step 2 validates and
    88	reviews that plan with deterministic duplicates and coverage. Step 3 records
    89	one explicit immutable operator approval or rejection for an exactly
    90	reconstructed review without executing it. Step 4 atomically creates or reuses
    91	only that review's selected Phase 4C proposals; it does not review or execute
    92	them. Step 5 observes every exact materialized proposal, and Step 6 explicitly
    93	applies one ordinary Phase 4C review decision to all exact members atomically.
    94	Phase 5 Step 1 then permits a separately confirmed atomic execution of those
    95	exact members through the existing Phase 4C paused-conversion transaction
    96	primitive. It neither activates nor queues the created experiments.
    97	Phase 5 Step 2 separately applies the existing Phase 4C activation primitive to
    98	all exact executed members atomically; it performs no automatic follow-up.
    99	Phase 5 Step 3 may instead compose the transaction-bound execution and
   100	activation primitives under one outer transaction, leaving all exact
   101	experiments ``pending/train`` without starting scheduler work.
   102	Phase 5 Step 4 observes those exact members through one repeatable-read,
   103	read-only database snapshot and performs no lifecycle action.
   104	Phase 5 Steps 5a–5b compare each exact member's frozen recommendation evidence
   105	with its final result evidence without recomputation, persistence, or success
   106	policy. Step 5c then applies only a database-free versioned interpretation
   107	policy, keeping evidence sufficiency, campaign interpretation, operator-review
   108	eligibility, and the invariant absence of follow-up authority separate.
   109	Phase 6A validates exact Step 5A/5C canonical and direct identity alignment and
   110	builds a deterministic follow-up proposal only from an eligible favorable,
   111	explicitly non-authorizing decision. It does not recompute policy truth or add
   112	operator-review, persistence, activation, execution, or scheduler behavior.
   113	Phase 6B stores that already-built value without recomputing it, reloads the
   114	same immutable typed proposal through a private validated seam, and previews
   115	it read-only. It adds no approval or action state.
   116	Phase 6C binds an explicit operator, reason, and approved/rejected decision to
   117	that exact persisted proposal in one append-only event. It adds administrative
   118	review evidence only and no activation or execution adapter.
   119	Phase 6D loads one exact Phase 6C event and permits only its ``approved``
   120	outcome to receive one separately identified append-only ``ratified``
   121	governance decision from a different actor under the fixed
   122	``follow_up_governance_ratifier`` role. It adds no Phase 6E, activation, or
   123	execution adapter.
   124	Detailed contracts remain in the Phase documents referenced in §12.
   125
   126	## 3. Responsibilities
   127
   128	### 3.1 Owned responsibilities
   129
   130	Own deterministic proposal identity/generation, authoritative persistence and
   131	duplicate provenance, advisory score history, explicit review disposition, and
   132	their presentation.
   133
   134	### 3.2 Dependencies
   135
   136	Consumes completed experiment/final-analysis evidence from Volumes VI/VII.
   137	Only the explicit Phase 4C execution and activation primitives cross into
   138	experiment lifecycle; Phase 5 Steps 1–3 are confirmed orchestration callers
   139	of those existing primitives. Ranking, campaign planning, campaign review, and
   140	campaign approval have no downstream execution dependency. Outcome assessment
   141	and policy consume persisted evidence but have no lifecycle dependency or
   142	authority.
   143	The Phase 6A proposal consumes only their immutable typed values and has no
   144	database, CLI, service, repository, lifecycle, or scheduler dependency. The
   145	separate Phase 6B repository depends on PostgreSQL only to preserve and preview
   146	that exact value; it has no lifecycle or scheduler dependency.
   147	The Phase 6C contract depends only on the Phase 6A proposal identity and its
   148	persisted Phase 6B ID. Its repository depends on PostgreSQL only to verify and
   149	append the administrative event; presentation is read-only.
   150	The Phase 6D contract depends only on the exact persisted Phase 6C review and
   151	its embedded Phase 6B proposal identity. Its repository and service use
   152	PostgreSQL only to verify and append governance-ratification evidence.
   153
   154	### 3.3 Prohibited responsibilities
   155
   156	Advisory evaluation, ranking, campaign planning, campaign review, and campaign
   157	approval MUST NOT create experiments, queue work, mutate experiment or
   158	scheduler state, infer reviewer identity, use
   159	score thresholds for authorization, or claim expected profitability. Phase 4C
   160	may create and activate exactly one experiment only through its separately
   161	invoked, audited manual commands.
   162	Phase 6A eligibility MUST NOT be treated as operator approval, follow-up
   163	authorization, campaign success, profitability evidence, persistence, or
   164	scheduler work.
   165	Phase 6B persistence and preview MUST NOT be treated as operator approval,
   166	activation, execution, queueing, scheduling, or follow-up authorization.
   167	Phase 6C approval MUST NOT be treated as activation, execution authorization,
   168	follow-up authorization, queueing, scheduling, scheduler work, worker launch,
   169	experiment creation/modification, campaign success, profitability, or
   170	statistical validation.
   171	Phase 6D governance ratification MUST NOT be treated as Phase 6E capability,
   172	activation, execution authorization, follow-up authorization, queueing,
   173	scheduling, scheduler work, worker launch, experiment/model creation or
   174	modification, continuation advancement, campaign success, profitability, or
   175	statistical validation.
   176
   177	## 4. Architecture
   178
   179	### 4.1 Components
   180
   181	- Pure identity/policy and candidate-generation domain components.
   182	- Pure identity-bound Phase 6A follow-up-proposal domain component.
   183	- Phase 6B append-only proposal repository and read-only preview component.
   184	- Pure immutable Phase 6C review contract, append-only event repository, and
   185	  read-only show/list presentation.
   186	- Pure immutable Phase 6D ratification contract, append-only event repository,
   187	  and transactional ratification service.
   188	- Pure manual conversion eligibility, source-consistency, and proposed-
   189	  specification domain component.
   190	- Repository-owned PostgreSQL mapping, scans, duplicates, scores, and reviews.
   191	- Append-only conversion-proposal persistence with exact canonical retries.
   192	- Services for explicit generation, scoring, evaluation, ranking/comparison,
   193	  inspection, and review use cases.
   194	- CLI parsing/validation/presentation outside scheduler polling.
   195
   196	### 4.2 Control flow
   197
   198	Completed evidence → deterministic source selection → pure candidates →
   199	collision-aware persistence → optional explicit score/evaluation/ranking. The
   200	Phase 4C proposal → review → paused execution → activation chain consists of
   201	separately invoked and audited manual actions. Phase 4D reads ranking and Phase
   202	4C history to plan and review; campaign approval records only an explicit human
   203	decision and adds no implicit arrow to mutation or execution.
   204
   205	The governed campaign path is:
   206
   207	```text
   208	recommendation ranking
   209	-> read-only campaign plan
   210	-> read-only campaign review
   211	-> explicit persisted campaign approval
   212	-> explicit approved-campaign materialization to Phase 4C proposals
   213	-> read-only exact campaign handoff
   214	-> explicit atomic Phase 4C proposal review for all exact campaign members
   215	-> either optional Phase 5 Step 1 execution then optional Step 2 activation
   216	-> or optional Phase 5 Step 3 atomic execution-and-activation launch
   217	-> existing scheduler lifecycle for ordinary pending experiments
   218	-> optional read-only Phase 5 Step 4 operational status snapshot
   219	-> optional read-only Step 5b outcome assessment
   220	-> optional pure Step 5c advisory policy interpretation
   221	-> optional pure Phase 6A follow-up proposal for later operator review
   222	-> optional exact Phase 6B persistence and read-only operator preview
   223	-> optional explicit Phase 6C administrative approval or rejection
   224	-> optional independent Phase 6D governance ratification of advancement
   225	```
   226
   227	All displayed stages through exact proposal review are implemented through
   228	Phase 4D Step 6; Phase 4C execution and activation remain separately invoked
   229	actions. Phase 5 Step 1 invokes only the existing Phase 4C execution operation
   230	for the exact campaign membership. Phase 5 Step 2 separately invokes only the
   231	existing Phase 4C activation operation; neither step invokes the other.
   232	Phase 5 Step 3 is a separate command that reuses both transaction-bound
   233	authorities inside one transaction; it does not call either public command.
   234	Phase 5 Step 4 reads the immutable materialization and bounded downstream
   235	evidence only; it neither invokes Steps 1–3 nor contacts the scheduler process.
   236	Step 5b reuses that status lifecycle inside one read-only snapshot. Step 5c
   237	contains no repository or CLI and consumes only an already-built immutable
   238	assessment. Phase 6A consumes only the exact immutable assessment and policy
   239	decision and terminates at a non-authorizing proposal value. Phase 6B may
   240	persist and preview only that exact value; it introduces no downstream action
   240	persist and preview only that exact value; it introduces no downstream action
   241	arrow. Phase 6C may record only one administrative decision and likewise adds
   242	no downstream action arrow. Phase 6D may record only one separate governance
   243	ratification of advancement after an eligible Phase 6C review and also adds no
   244	downstream action arrow or Phase 6E capability.
   245
   246	### 4.3 Ownership boundaries
   247
   248	Canonical text decides identity; repository transactions decide persistence;
   249	pure scoring, planning, campaign-review, outcome-policy, and follow-up-proposal
   250	components decide advisory output; pure review rules decide legal transitions;
   251	and the operator separately
   252	supplies review, conversion, and activation actions. Only Phase 4C Steps 4–5
   253	may create or activate the one provenance-linked experiment. Campaign
   254	planning, review, approval, outcome assessment, outcome policy, Phase 6A
   255	follow-up proposal, Phase 6B persistence/preview, and Phase 6C administrative
   256	review/presentation, and Phase 6D governance ratification never do.
   257
   258	## 5. Data model
   259
   260	### 5.1 Authoritative entities
   261
   262	Recommendation policy, semantic/invocation configuration, recommendation scan,
   263	recommendation, score run/result/component, evaluation run/result/component,
   264	ranking snapshot/member, and review event.
   265	Conversion proposal, review decision, execution, activation, and campaign
   266	approval records are separate immutable audit entities; only their explicitly linked experiment is
   267	an experiment-lifecycle entity. Campaign plans and reviews are reconstructed,
   268	not persisted; Step 3 persists only their exact approved/rejected provenance.
   269	Campaign outcome assessments and policy decisions are point-in-time,
   270	non-persistent advisory values; their canonical identities bind exact upstream
   271	evidence but create no authoritative campaign decision.
   272	Phase 6A follow-up proposals are advisory values whose canonical identities
   273	bind the exact assessment v2, policy v1, decision v1, campaign,
   274	materialization, ordered members, eligibility, and fixed non-authority
   275	semantics while excluding observation time. Phase 6B persists that exact value
   276	in one immutable manifest and ordered member set; repository row ID, collision
   277	ordinal, and creation timestamp remain storage metadata outside identity.
   278	The Phase 6C review is a separate immutable administrative entity binding the
   279	persisted proposal ID, exact proposal version/canonical/hash, decision,
   280	reviewer, and reason. Its canonical text is authoritative. Review-event ID and
   281	creation timestamp remain persistence metadata outside identity.
   282	The Phase 6D ratification is a separate immutable governance entity. It binds
   283	the Phase 6C event ID and exact review version/canonical/hash, reviewer, exact
   284	reviewed proposal identity, approved eligibility, fixed ratifier role,
   285	``ratified`` decision, distinct ratifier, basis, separation policy, and fixed
   286	negative action semantics. Ratification-event ID and creation timestamp remain
   287	persistence metadata outside identity.
   288
   289	### 5.2 Provenance and versions
   290
   291	Canonical text is authoritative; tagged hashes accelerate lookup. Persisted
   292	records retain source experiment/analysis/model, scan, policy, identity,
   293	structural mutation, source rank, score policy/components, and review snapshot.
   294
   295	### 5.3 Invariants and legacy data
   296
   297	Scan-associated rows require positive source rank and cannot detach from their
   298	originating scan. Proposed/approved identities remain active; rejected/expired
   299	matches remain historical blockers. Legacy NULL/NULL scan/rank rows are not
   300	backfilled and are excluded from operations requiring provenance.
   301
   302	Review transitions are only proposed → approved, rejected, or expired. Review
   303	events are append-only to the runtime role. Approval is advisory and leaves
   304	`approved_experiment_id` null under Step 5.
   305
   306	## 6. Transactions
   307
   308	### 6.1 Read paths
   309
   310	Evidence, list, detail, score history, evaluation history, ranking input, and
   311	review history use typed read-only repository paths. Ranking does not reopen
   312	source experiment metrics.
   313
   314	### 6.2 Write paths
   315
   316	Candidate persistence uses short collision-aware transactions. Score result and
   317	ordered components persist atomically. Evaluation results and their unchanged
   318	Step 4 components persist atomically without locking experiment evidence.
   319	Review locks one recommendation and updates status plus one immutable event in
   320	the same transaction. Ranking persists its complete member set atomically and
   321	then completes a ranking-owned snapshot after an exact count check.
   322	Conversion-proposal insertion locks only its accelerator-hash bucket, compares
   323	canonical text exactly under the bytewise ``C`` collation, and inserts one
   324	complete row in one transaction. The collision ordinal is storage metadata;
   325	canonical text alone remains authoritative.
   326	Phase 6B proposal insertion locks only the proposal-hash bucket, compares
   327	authoritative canonical text exactly, inserts the manifest and ordered members
   328	in one caller-owned transaction, and enforces completeness at commit. It does
   329	not write any review, lifecycle, experiment, or scheduler table.
   330	Phase 6C review insertion locks only the proposal-ID conflict domain, reloads
   331	and compares the exact Phase 6B proposal identity, and inserts one event. Exact
   332	replay returns the event; changed decision, reviewer, reason, or identity
   333	conflicts. It never updates or supersedes review history.
   334	Phase 6D ratification insertion locks only the review-event-ID conflict domain,
   335	reloads the exact Phase 6C review and Phase 6B proposal identity, enforces an
   336	approved review, fixed role, and distinct reviewer/ratifier, and inserts one
   337	event. Exact replay returns the event; any changed payload conflicts. It never
   338	updates Phase 6A, 6B, or 6C evidence.
   339
   340	### 6.3 Failure semantics
   341
   342	Invalid commands create no scan/event. Candidate failures are counted per scan.
   343	Score retries compare complete immutable results. Evaluation retains completed
   344	per-recommendation results if a later item fails, marks the run failed, and
   345	forbids adding new results to that terminal run; exact existing results remain
   346	verifiable. Review event failure rolls back status; terminal retries create no
   347	event. A ranking-member failure leaves no partial membership.
   348	Malformed Phase 6C input, stored canonical/hash disagreement, Phase 6B
   349	provenance mismatch, and conflicts fail closed and leave no partial event.
   350	Malformed Phase 6D input, missing/rejected/stale reviews, any complete-chain
   351	identity mismatch, stored canonical/hash disagreement, and conflicts also fail
   352	closed and leave no partial event.
   353
   354	## 7. Concurrency
   355
   356	### 7.1 Conflict domain
   357
   358	Generation conflicts on semantic configuration plus policy; scoring retries on
   359	score run plus recommendation; evaluation conflicts on canonical evaluation-run
   360	identity and evaluation run plus recommendation; ranking conflicts on canonical
   361	snapshot identity and snapshot plus evaluation result; review conflicts on one
   362	recommendation row.
   363
   364	### 7.2 Locking and serialization
   365
   366	Generation uses a transaction advisory key plus canonical rechecks and active
   367	uniqueness. Scoring uses uniqueness and complete retry comparison. Evaluation
   368	uses a short shared lock on its own run row while inserting an atomic result;
   369	it does not lock experiment evidence. Review uses `SELECT ... FOR UPDATE` and
   370	one-event uniqueness. Ranking serializes identical member verification with a
   371	short `FOR UPDATE` lock on its ranking-owned snapshot. Unrelated snapshots and
   372	unrelated work remain concurrent.
   373
   374	### 7.3 Winner, loser, and retry outcomes
   375
   376	Concurrent identical generation returns one created/one existing result.
   377	Equivalent score retry reuses only an exact persisted result. Concurrent review
   378	produces one terminal winner and one `recommendation_review_status_conflict`;
   379	exactly one event exists. Concurrent identical ranking requests converge on one
   380	snapshot with an exactly verified member set.
   381	Concurrent identical conversion proposals converge on one exact canonical row;
   382	same-hash different canonical rows remain distinct.
   383	Concurrent identical Phase 6B proposal persistence converges on one exact
   384	canonical row; same-hash/different-canonical proposals receive distinct
   385	storage collision ordinals.
   386	Concurrent identical Phase 6C reviews converge on one event with one
   387	``recorded`` and one ``existing_identical`` outcome. Concurrent differing
   388	reviews yield one recorded winner and one deterministic conflict.
   389	Concurrent identical Phase 6D ratifications converge on one event with one
   390	``recorded`` and one ``existing_identical`` outcome. Concurrent differing
   391	ratifications for the same review yield one recorded winner and one
   392	deterministic conflict.
   393
   394	## 8. CLI
   395
   396	### 8.1 Commands and validation
   397
   398	Generation, scoring, evaluation, ranking/comparison, their inspection commands,
   399	review actions, and review inspection are explicit families. Evaluation,
   400	ranking, and review mutation
   401	are mutually exclusive with generation, scoring, scheduler, continuation, and
   402	experiment-lifecycle commands.
   403
   404	### 8.2 Machine output
   405
   406	`EXPERIMENT_RECOMMENDATION_*` records use stable fields, percent-escaped text,
   407	explicit `NULL`, deterministic ordering, and separate conflict/failure events.
   408
   409	### 8.3 Human output
   410
   411	Human score/evaluation/review summaries are concise, control-byte safe, and
   412	state that the result is advisory and created/queued no experiment. Ranking
   413	separates advisory-ready, blocked, and non-actionable presentation.
   414	Phase 6B preview includes the full authoritative proposal identity, provenance,
   415	ordered members, and explicit false approval/activation/execution/scheduler
   416	fields in a read-only snapshot.
   417	Phase 6C show/list includes exact proposal/review identities, decision,
   418	reviewer, reason, timestamp, and fixed negative activation, execution,
   419	follow-up, queue, schedule, scheduler, worker, experiment, and success fields.
   420	Phase 6D intentionally adds no main-program CLI or presentation. Its typed
   421	service establishes only the explicit governance write boundary, while
   422	repository lookup/list operations remain read-only.
   423
   424	## 9. Testing
   425
   426	### 9.1 Pure tests
   427
   428	Cover canonical identity/collisions, eligibility, candidates, scoring formulas,
   429	ranking, evaluation classification/identity, review parsing/reasons, and all
   430	transition outcomes. Phase 5 outcome tests additionally cover evidence
   431	classification, policy judgment, the advisory comparable-member evidence-
   432	coverage minimum, mixed member/metric states, locale independence,
   433	deterministic identity, and explicit non-authorization. The coverage minimum
   434	is not statistical sufficiency, confidence, causality, profitability,
   435	repeatability, or campaign success.
   436	Phase 6A pure tests additionally cover exact upstream alignment, stable refusal
   437	reasons, golden canonical identity, all non-authority flags, upstream-order,
   438	locale and observation-time invariance, identity sensitivity, inherited IEEE
   439	edge behavior, the canonical-size boundary, and inaccessible malformed
   440	construction paths.
   441	Phase 6B focused tests cover exact round trip, immutable typed reload,
   442	canonical/hash preservation, exact retry, duplicate persisted identity,
   443	malformed hash/member data, and read-only preview safety fields.
   444	Phase 6C pure tests cover immutable types, approved/rejected construction,
   445	canonical grammar, locale independence, proposal/reviewer/reason/decision
   446	sensitivity, malformed input, and fixed negative authority.
   447	Phase 6D pure tests cover immutable types, approved-only construction,
   448	``ratified`` decision, fixed role, separation of duties, golden canonical/hash,
   449	locale independence, complete identity sensitivity, hostile input, and fixed
   450	negative authority.
   451
   452	### 9.2 Persistence and migration tests
   453
   454	Cover scans, duplicates, source rank/scan provenance, score ownership and
   455	immutability, review constraints, runtime privileges, SQLSTATEs, and migration
   456	repeatability. Evaluation tests additionally cover canonical idempotency,
   457	append-only result/component privileges, nullable blocked scores, and evidence
   458	foreign keys. Ranking tests cover exact membership, bucket/rank uniqueness,
   459	narrow privileges, scope/captured-membership trigger enforcement, bounded
   460	evidence loading, atomic member persistence, and Step 1 immutability.
   461	Conversion-proposal tests cover complete invocation round trips, exact retries,
   462	hash collisions, concurrent insertion, restrictive privileges, and unchanged
   463	experiment fixtures.
   464	Phase 6B migration/repository tests cover bytewise canonical storage,
   465	collision metadata, upstream provenance, deferred membership completeness,
   466	narrow append-only privileges, migration repeatability, malformed persistence
   467	rejection, and a preview read sentinel that remains untouched.
   468	Phase 6C migration/repository tests cover repeatability, bytewise canonical
   469	storage, restrictive provenance, narrow privileges, generated-column
   470	protection, exact round trip/replay, all conflict forms, concurrent identical
   471	and differing attempts, corruption/provenance rejection, rollback, unchanged
   472	Phase 6B/experiment/scheduler fixtures, and read-only presentation without
   473	sequence advancement or advisory/tuple locks.
   474	Phase 6D migration/repository/service tests cover clean and upgrade paths,
   475	repeatability, NULL-ACL fallback, safe trigger identity/context, approved-only
   476	eligibility, missing/stale/rejected review refusal, exact round trip/replay,
   477	sequential and concurrent conflicts, rollback, collision/mismatch rejection,
   478	read-only lookup/list behavior, and unchanged Phase 6B/6C/experiment/scheduler
   479	fixtures.
   480
   481	### 9.3 Concurrency and integration tests
   482
   483	Cover same-candidate generation, score retries, evaluation run/result retries
   484	and partial failures, review action races, unrelated reviews, rollback, exact
   485	cleanup, and before/after experiment/identity/score digests.
   486
   487	### 9.4 Regression boundaries
   488
   489	Steps 1–5 remain individually stable. Changes MUST NOT alter training,
   490	inference, continuation, experiment lifecycle, or scheduler behavior.
   491
   492	## 10. Operational safety
   493
   494	### 10.1 Runtime isolation
   495
   496	No recommendation command enters scheduler polling. Integration uses disposable
   497	rows and exact cleanup without disturbing genuine experiments.
   498
   499	### 10.2 Permissions and destructive operations
   500
   501	Runtime access to review history and evaluation results/components is
   502	SELECT/INSERT only; evaluation-run updates are limited to lifecycle columns.
   503	Ranking members are append-only and ranking-snapshot updates are limited to
   504	lifecycle/count columns. Owner/test connections perform exact fixture cleanup.
   505	Conversion proposals grant runtime ``SELECT``/``INSERT`` access. Manual review
   506	decisions grant runtime ``SELECT`` plus column-limited ``INSERT`` for decision
   507	payload only; generated decision IDs and timestamps are not caller-writable.
   508	Their source-provenance foreign keys are restrictive. Review history is
   509	append-only, and the greatest generated decision ID yields the current
   510	pending/approved/rejected administrative disposition. This sequence-ID order is
   511	authoritative even when concurrent transactions commit in another order.
   512	Conversion activations likewise grant only ``SELECT`` and column-limited
   513	``INSERT`` plus sequence usage. Activation evidence is append-only; runtime
   514	``UPDATE``, ``DELETE``, and ``TRUNCATE`` are denied.
   515	Other immutable histories follow Volume I §7.3.
   516	Phase 6C review events likewise grant only ``SELECT``, payload-column
   517	``INSERT``, and sequence ``USAGE``. Generated IDs/timestamps and runtime
   518	update/delete/truncate are denied. An approved event is administrative evidence
   519	only and is not consumed by any Phase 6C action path.
   520	Phase 6D ratification events use the same narrow table/sequence privilege shape,
   520	Phase 6D ratification events use the same narrow table/sequence privilege shape,
   521	plus explicit PUBLIC and trigger-function revocation. The safe invoker-rights
   522	trigger verifies complete Phase 6C/6B provenance and separation of duties.
   523	Ratification remains evidence and is not consumed by an action path.
   524
   525	### 10.3 Observability and recovery
   526
   527	Scans/runs expose completion and counters; scores/events retain immutable
   528	provenance; explicit conflicts distinguish safe rejection from system failure.
   529
   530	## 11. Future extensions
   531
   532	### 11.1 Approved extension points
   533
   534	Versioned policies, pure candidate/scoring components, typed repository/service
   535	APIs, and separate inspection commands.
   536
   537	### 11.2 Deferred capabilities
   538
   539	Experiment queueing/execution from conversion, multi-source attribution,
   540	automatic research campaigns, profitability evidence, and scheduler-managed
   541	recommendation work. Phase 4C Step 1 provides the pure proposed-
   542	specification contract, Step 2 provides durable proposal audit evidence, and
   543	Step 3 provides explicit append-only manual proposal review. Step 4 permits one
   544	explicit approved proposal to create one paused experiment. Step 5 permits a
   545	separate explicit operator action to make that exact experiment pending under
   546	the existing scheduler lifecycle.
   547
   548	### 11.3 Required decisions
   549
   550	ADR-0005 defines explicit conversion authorization, implementation identity,
   551	idempotent experiment creation, transactions, and audit history. Approval alone
   552	does not invoke conversion. Budgets, queueing, execution, and scheduler capacity
   553	remain deferred under ADR-0004.
   554
   555	Phase 4C Step 2 adds the append-only
   556	``experiment_recommendation_conversion_proposal`` record and a narrow
   557	repository. Exact canonical text remains authoritative; same-hash distinct
   558	canonical identities receive separate collision ordinals, while concurrent
   559	identical inserts converge on one proposal. Optional ranking provenance is
   560	advisory, excluded from identity, and cannot authorize conversion. Runtime
   561	access is limited to ``SELECT``/``INSERT``; no scheduler path reads these rows.
   562
   563	Phase 4C Step 3 adds explicit ``approve``/``reject`` decisions against an exact
   564	proposal primary key. A caller-visible request token makes CLI retries
   565	idempotent per proposal. New request tokens permit auditable reversals; the
   566	greatest decision ID defines the current disposition. Approval remains
   567	administrative evidence only and never creates or queues an experiment.
   568
   569	Phase 4C Step 4 adds one explicit manual conversion transaction. It takes a
   570	proposal-specific transaction advisory lock, revalidates the immutable
   571	proposal, requires the latest serialized decision to be ``approve``, creates
   572	one ``paused`` experiment, and records immutable provenance linking the
   573	proposal, approving decision, and experiment. Unique proposal and experiment
   574	references make retries and concurrent requests converge. The scheduler does
   575	not poll conversion rows and a converted experiment is not queued or started.
   576
   577	Phase 4C Step 5 adds one explicit activation transaction. It revalidates the
   578	Step 4 execution and exact created invocation, requires the experiment to
   579	remain in the pristine ``paused/train`` state, records immutable activation
   580	evidence, and changes only that experiment to ``pending/train``. The audit and
   581	lifecycle update are atomic. Exact retries return the same activation without
   582	reapplying the transition. The scheduler is unchanged and never reads the
   583	activation table; no worker is started by activation.
   584
   585	Phase 4C Step 6 adds a read-only workflow aggregate over proposal, greatest-ID
   586	review disposition, exact execution authorization, activation evidence, and
   587	current experiment lifecycle. Stable states and deterministic diagnostics make
   588	healthy and inconsistent chains inspectable without adding schema, privileges,
   589	workflow mutation, scheduler polling, or worker behavior.
   590
   591	Phase 4D Step 1 adds a pure, versioned campaign policy and a read-only aggregate
   592	over one explicit completed ranking snapshot, persisted recommendation source
   593	metrics, and the existing Phase 4C workflow derivation. Ranking global ordinal,
   594	recommendation ID, ranking member ID, then canonical evidence define the full
   595	presentation/selection order. Every considered
   596	candidate is retained with one decision and stable reasons; plan identity binds
   597	the full policy, scope, snapshot, evidence, workflow provenance, decisions, and
   598	reasons while excluding display time. There is no campaign table or scheduler
   599	consumer. Because no authoritative profitability evidence exists, a requested
   600	profitability threshold yields ``profitability_metric_unavailable`` rather than
   601	an inferred proxy.
   602
   603	Phase 4D Step 2 adds a pure review contract over the Step 1 plan and reuses the
   604	same read-only loader and planner. It verifies plan identity, policy/scope,
   605	counts, result shape, and deterministic ordering; preserves selected and
   606	excluded rows with their ordered reasons; and derives exact canonical duplicate
   607	groups plus deterministic family, symbol, and horizon coverage. Review identity
   608	binds these results while excluding display time. Step 2 adds no migration,
   609	privilege, persisted campaign, workflow mutation, scheduler consumer, or worker
   610	behavior.
   611
   612	Phase 4D Step 3 adds one immutable approval/rejection audit row for one exact
   613	Step 2 review. One PostgreSQL transaction loads the explicit ranking snapshot
   614	and Phase 4C workflow evidence, reconstructs and validates the plan and review,
   615	requires an operator-supplied expected review hash, and inserts the decision.
   616	Canonical review text is authoritative; identical retries converge, changed
   617	decision/reviewer/reason payload conflicts, and a zero-selection campaign may
   618	only be rejected. Runtime privileges are append-only and column-limited. The
   619	approval row never creates or modifies an experiment and is not consumed by
   620	the scheduler.
   621
   622	Phase 4D Step 4 adds one explicit, atomic materialization transaction for one
   623	approved campaign. It reconstructs the exact plan, review, and approval, then
   624	passes every selected member through the existing Phase 4C conversion contract.
   625	One immutable manifest and ordered member links record created or reused Phase
   626	4C proposals. Canonical text is authoritative and exact replay converges;
   627	changed operator, reason, membership, proposal, or approval evidence conflicts.
   628	First creation reconstructs the approved evidence; retry validates the
   629	immutable manifest directly because its newly created proposals intentionally
   630	alter subsequent planning evidence.
   631	Materialization does not review or execute those proposals, create or modify an
   632	experiment, or contact the scheduler or workers.
   633
   634	Phase 4D Step 5 adds a read-only campaign handoff projection. The immutable
   635	Step 4 manifest and ordered members remain membership authority; greatest-ID
   636	Phase 4C review, execution, activation, and experiment lifecycle evidence are
   637	observed in one repeatable-read transaction and one consistent snapshot.
   638	Explicit aggregate states and deterministic diagnostics expose incomplete or
   639	contradictory chains without persistence, repair, automatic progression,
   640	scheduler polling, or worker behavior.
   641
   642	Phase 4D Step 6 adds one explicit atomic operator review over the exact stored
   643	Step 4 member order. It validates the existing Step 4 manifest and Step 5/Phase
   644	4C workflow evidence, serializes only the linked proposal review sequences, and
   645	appends one ordinary Phase 4C approve/reject row per member in one transaction.
   646	Deterministic request identity makes exact whole-campaign retries converge;
   647	opposite, unrelated same-decision, malformed, or partially satisfied evidence
   648	fails closed. No Phase 4D review authority or schema is added, and execution,
   649	activation, experiment creation, scheduling, workers, repair, and automatic
   650	progression remain separate.
   651
   652	Phase 5 Step 1 adds one separately confirmed, all-member execution transaction
   653	over the same immutable Step 4 membership. Sorted existing Phase 4C proposal
   654	locks serialize the operation with direct review and direct execution. Every
   655	member is validated before the first insert, then the existing transaction-
   656	bound Phase 4C primitive creates ordinary paused experiments and immutable
   657	execution rows. Exact all-executed retry is already satisfied; mixed prior
   658	execution fails closed. There is no new schema or execution authority, and no
   659	activation, pending transition, scheduler action, worker launch, or automatic
   660	progression.
   661
   662	Phase 5 Step 2 adds one separately confirmed, all-member activation transaction
   663	over that same membership. Existing activation advisory locks and experiment
   664	row locks are acquired in sorted execution-ID and experiment-ID order. All
   665	members are validated before the first ordinary activation insert and exact
   666	``paused/train`` to ``pending/train`` update. Exact fully activated retry is
   667	already satisfied only while every experiment retains that post-state; mixed
   668	activation fails closed. No schema, campaign activation authority, scheduler
   669	action, worker launch, direct process, or automatic follow-up is added.
   670
   671	Phase 5 Step 3 adds one separately confirmed atomic launch over the same exact
   672	membership. It takes all proposal locks in ascending proposal-ID order,
   673	validates the complete execution phase, creates all required ordinary paused
   674	experiments and executions, then takes activation locks in ascending
   675	execution-ID order and experiment rows in ascending experiment-ID order. It
   676	validates the complete activation phase before applying any activation, and
   677	commits once. All-executed/no-activation state is completed atomically; partial
   678	execution or activation fails closed; an exact fully activated retry is
   679	already satisfied only while every experiment remains ``pending/train``. No
   680	schema, campaign lifecycle authority, scheduler action, worker launch, direct
   681	process, or automatic follow-up is added.
   682
   683	Phase 5 Step 5a defines a pure immutable outcome assessment over exact campaign
   684	membership, lifecycle consistency, source/result provenance, comparison
   685	context, and metric deltas. Step 5b loads authoritative persisted campaign and
   686	scientific evidence in one repeatable-read, read-only snapshot and builds a
   687	deterministic, non-authoritative, point-in-time assessment from it. Neither step
   688	persists an assessment, declares campaign success, or authorizes follow-up.
   689
   690	Phase 5 Step 5c adds a pure versioned policy over the Step 5a assessment. It
   691	uses the assessment's existing classifications and deltas, applies explicit
   692	metric-direction and advisory evidence-coverage rules, and separates evidence
   693	sufficiency, conservative interpretation, and eligibility for a later explicit
   694	operator review. Only a favorable, sufficient, all-comparable campaign is
   695	eligible for such review; neutral, unfavorable, mixed, and inconclusive
   696	campaigns are not. Its result is non-persistent and non-authoritative, always
   697	records follow-up authorization as false, and adds no repository, CLI,
   698	scheduler, worker, schema, or lifecycle behavior.
   699
   700	Phase 6A adds the distinct
   701	``RecommendationCampaignFollowUpProposal`` contract over one exact Step 5a
   702	assessment v2 and one exact Step 5c policy decision v1 containing policy v1.
   703	It validates authoritative canonical text as well as hashes and exact campaign,
   704	materialization, count, and ordered-member alignment. Only an eligible,
   705	favorable, explicitly non-authorizing decision yields a proposal. Contract v1
   706	binds every safety semantic and exact upstream canonical identity, excludes
   707	``observedAt``, and rejects canonical text above 1,048,576 bytes before
   708	construction. The proposal is not approval, persistence, activation,
   709	execution, scheduler work, success evidence, or authority for later phases.
   710
   711	Phase 6B persists only that already-built immutable proposal. One append-only
   712	manifest and ordered member set preserve every Phase 6A field. Exact canonical
   713	text remains authoritative, collision ordinals and timestamps stay outside
   714	identity, and reload reruns the Phase 6A payload/canonical invariants. Preview
   715	uses one repeatable-read, read-only transaction and explicitly reports false
   716	approval, activation, execution, authorization, queue, scheduling, worker, and
   717	experiment mutation state. Phase 6B adds no operator decision or action path.
   718
   719	Phase 6C records only one immutable administrative ``approved`` or ``rejected``
   720	event for one exact persisted Phase 6B proposal. The review canonical binds the
   721	proposal ID/version/canonical/hash, decision, reviewer, reason, and fixed
   722	negative action semantics. Exact retry is idempotent; every changed payload for
   723	that proposal conflicts. Approval grants only possible later consideration and
   724	does not activate, execute, authorize follow-up, queue, schedule, contact the
   725	scheduler, launch workers, create/modify experiments, or declare success.
   726
   727	Phase 6D separately records one immutable ``ratified`` governance event for
   728	advancement after one exact persisted Phase 6C approved review. Its canonical
   729	binds the complete review and proposal identity, reviewer, fixed role,
   730	distinct ratifier, basis, and separation policy. Exact retry is idempotent;
   731	every changed payload conflicts. The service loads rather than reconstructs
   732	the authoritative upstream chain. Phase 6D adds no CLI, Phase 6E, activation,
   733	execution, follow-up authorization, scheduler, worker, experiment, model,
   734	continuation, or success capability.
   735
   736	## 12. References
   737
   738	- [Volume I §§5–10](Volume_I_Foundation.md)
   739	- [ADR-0003](adr/ADR-0003-advisory-recommendation-evaluation.md)
   740	- [ADR-0005](adr/ADR-0005-manual-recommendation-conversion.md)
   741	- [Recommendation foundation](../Phase4AExperimentRecommendationFoundation.rst)
   742	- [Recommendation persistence](../Phase4AExperimentRecommendationPersistence.rst)
   743	- [Recommendation scoring](../Phase4AExperimentRecommendationScoring.rst)
   744	- [Recommendation review](../Phase4AExperimentRecommendationReview.rst)
   745	- [Recommendation evaluation](../Phase4BExperimentRecommendationEvaluation.rst)
   746	- [Recommendation ranking](../Phase4BExperimentRecommendationRanking.rst)
   747	- [Manual conversion contract](../Phase4CExperimentRecommendationConversion.rst)
   748	- [Manual conversion proposal persistence](../Phase4CExperimentRecommendationConversionPersistence.rst)
   749	- [Manual conversion proposal review](../Phase4CExperimentRecommendationConversionProposalReview.rst)
   750	- [Manual conversion execution](../Phase4CExperimentRecommendationConversionExecution.rst)
   751	- [Manual conversion activation](../Phase4CExperimentRecommendationConversionActivation.rst)
   752	- [Manual conversion workflow observability](../Phase4CExperimentRecommendationConversionWorkflow.rst)
   753	- [Read-only campaign planning](../Phase4DExperimentRecommendationCampaignPlanning.rst)
   754	- [Read-only campaign review](../Phase4DExperimentRecommendationCampaignReview.rst)
   755	- [Explicit campaign approval](../Phase4DExperimentRecommendationCampaignApproval.rst)
   756	- [Approved campaign materialization](../Phase4DExperimentRecommendationCampaignMaterialization.rst)
   757	- [Read-only campaign handoff status](../Phase4DExperimentRecommendationCampaignHandoff.rst)
   758	- [Materialized campaign proposal review](../Phase4DExperimentRecommendationCampaignProposalReview.rst)
   759	- [Atomic campaign conversion execution](../Phase5ExperimentRecommendationCampaignExecution.rst)
   760	- [Atomic campaign conversion activation](../Phase5ExperimentRecommendationCampaignActivation.rst)
   761	- [Atomic campaign conversion launch](../Phase5ExperimentRecommendationCampaignLaunch.rst)
   762	- [Recommendation campaign operational status](../Phase5ExperimentRecommendationCampaignStatus.rst)
   763	- [Recommendation campaign outcome assessment and policy](../Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst)
   764	- [Phase 6A recommendation campaign follow-up proposal](../Phase6ARecommendationCampaignFollowUpProposal.rst)
   765	- [Phase 6B follow-up proposal persistence and preview](../Phase6BRecommendationCampaignFollowUpProposalPersistence.rst)
   766	- [Phase 6C follow-up proposal administrative review](../Phase6CRecommendationCampaignFollowUpProposalReview.rst)
   767	- [Phase 6D follow-up proposal governance ratification](../Phase6DRecommendationCampaignFollowUpProposalRatification.rst)
   768	- [ADR-0006](adr/ADR-0006-phase-6a-follow-up-proposal.md)
   769	- [ADR-0007](adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md)
   770	- [ADR-0008](adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md)
   771	- [ADR-0009](adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md)
   772
   773	## 13. Revision history
   774
   775	| Version | Date | Change | ADR |
   776	|---|---|---|---|
   777	| 0.1.0 | 2026-07-15 | Established the Step 1–5-aligned recommendation architecture outline. | ADR-0003 |
   778	| 0.2.0 | 2026-07-15 | Recorded Phase 4B Step 1 deterministic advisory evaluation evidence without changing execution ownership. | ADR-0003, ADR-0004 |
   779	| 0.3.0 | 2026-07-15 | Recorded Phase 4B Step 2 immutable advisory ranking snapshots and policy-aware comparison. | ADR-0001, ADR-0003, ADR-0004 |
   780	| 0.4.0 | 2026-07-18 | Recorded the pure Phase 4C Step 1 manually authorized proposed-experiment contract; durable conversion and execution remain deferred. | ADR-0003, ADR-0004 |
   781	| 0.5.0 | 2026-07-18 | Recorded immutable Phase 4C Step 2 conversion-proposal persistence; experiment creation and execution remain deferred. | ADR-0001, ADR-0003, ADR-0004 |
   782	| 0.6.0 | 2026-07-18 | Recorded append-only, idempotent Phase 4C Step 3 manual proposal review; approval remains non-executing. | ADR-0001, ADR-0003, ADR-0004 |
   783	| 0.7.0 | 2026-07-19 | Recorded explicit, idempotent Phase 4C Step 4 conversion to one paused experiment. | ADR-0001, ADR-0004, ADR-0005 |
   784	| 0.8.0 | 2026-07-19 | Recorded explicit, atomic Phase 4C Step 5 activation of a converted experiment into the existing pending lifecycle. | ADR-0001, ADR-0004, ADR-0005 |
   785	| 0.9.0 | 2026-07-19 | Recorded read-only Phase 4C Step 6 end-to-end workflow state and integrity observability. | ADR-0001, ADR-0004, ADR-0005 |
   786	| 1.0.0 | 2026-07-19 | Recorded deterministic read-only Phase 4D Step 1 campaign planning from explicit persisted ranking and workflow evidence. | ADR-0001, ADR-0003, ADR-0004, ADR-0005 |
   787	| 1.1.0 | 2026-07-19 | Recorded deterministic read-only Phase 4D Step 2 campaign review, duplicate findings, and coverage. | ADR-0001, ADR-0003, ADR-0004, ADR-0005 |
   788	| 1.2.0 | 2026-07-19 | Recorded explicit immutable Phase 4D Step 3 campaign approval/rejection for one exact reconstructed review without execution. | ADR-0001, ADR-0003, ADR-0004, ADR-0005 |
   789	| 1.3.0 | 2026-07-19 | Recorded atomic Phase 4D Step 4 materialization of one approved campaign into exact Phase 4C proposals without experiment or scheduler execution. | ADR-0001, ADR-0003, ADR-0004, ADR-0005 |
   790	| 1.4.0 | 2026-07-19 | Recorded read-only Phase 4D Step 5 projection of materialized campaign membership onto current Phase 4C lifecycle evidence. | ADR-0001, ADR-0003, ADR-0004, ADR-0005 |
   791	| 1.5.0 | 2026-07-19 | Recorded explicit atomic Phase 4D Step 6 review of exact materialized proposals through ordinary Phase 4C review rows. | ADR-0001, ADR-0003, ADR-0004, ADR-0005 |
   792	| 1.6.0 | 2026-07-19 | Recorded explicit atomic Phase 5 Step 1 execution of exact materialized proposals through the existing Phase 4C paused-conversion authority. | ADR-0001, ADR-0004, ADR-0005 |
   793	| 1.7.0 | 2026-07-19 | Recorded explicit atomic Phase 5 Step 2 activation of exact materialized executions through the existing Phase 4C pending-transition authority. | ADR-0001, ADR-0004, ADR-0005 |
   794	| 1.8.0 | 2026-07-19 | Recorded explicit one-transaction Phase 5 Step 3 launch through the existing Phase 4C execution and activation authorities. | ADR-0001, ADR-0004, ADR-0005 |
   795	| 1.9.0 | 2026-07-19 | Recorded read-only Phase 5 Step 4 operational status for exact materialized members and current persisted lifecycle evidence. | ADR-0001, ADR-0004, ADR-0005 |
   796	| 2.0.0 | 2026-07-20 | Recorded Phase 5 Steps 5a–5b read-only outcome assessment and Step 5c pure advisory outcome policy with no persistence or follow-up authority. | ADR-0001, ADR-0003, ADR-0004, ADR-0005 |
   797	| 2.1.0 | 2026-07-20 | Recorded the pure, exact-identity-bound, non-authorizing Phase 6A follow-up proposal; later persistence, review, activation, and execution remain deferred. | ADR-0003, ADR-0004, ADR-0006 |
   798	| 2.2.0 | 2026-07-20 | Recorded exact append-only Phase 6B proposal persistence, immutable validated reload, and read-only preview without operator-decision or action authority. | ADR-0001, ADR-0004, ADR-0006, ADR-0007 |
   799	| 2.3.0 | 2026-07-21 | Recorded exact append-only Phase 6C approved/rejected administrative review, deterministic replay/conflict, and read-only presentation without action authority. | ADR-0001, ADR-0004, ADR-0006, ADR-0007, ADR-0008 |
   800	| 2.4.0 | 2026-07-22 | Recorded proposed append-only Phase 6D governance ratification after one eligible Phase 6C review, mandatory separation of duties, and deterministic replay/conflict without Phase 6E authority. | ADR-0001, ADR-0004, ADR-0006, ADR-0007, ADR-0008, ADR-0009 |

exec
/bin/bash -lc "wc -l Phase6E_Readiness_Authority_and_Implementation_Output.txt Phase6_Next_Architecture_Recommendation_Output.txt CampaignOperations_Architecture_Prompt.txt && rg -n '"'^(#|##|###)|Disposition|Recommendation|Phase 6|6E|Campaign Operations|operations|blocking|accept|reject|ready|complete|remaining|authority|scheduler'"' Phase6E_Readiness_Authority_and_Implementation_Output.txt Phase6_Next_Architecture_Recommendation_Output.txt | sed -n '1,500p' && tail -n 180 Phase6E_Readiness_Authority_and_Implementation_Output.txt && tail -n 220 Phase6_Next_Architecture_Recommendation_Output.txt" in /Volumes/Developer SSD/ExpertAdvisor
 succeeded in 0ms:
     525 Phase6E_Readiness_Authority_and_Implementation_Output.txt
   13904 Phase6_Next_Architecture_Recommendation_Output.txt
     102 CampaignOperations_Architecture_Prompt.txt
   14531 total
Phase6E_Readiness_Authority_and_Implementation_Output.txt:14:PHASE 6E
Phase6E_Readiness_Authority_and_Implementation_Output.txt:20:Phase 6D is now committed as:
Phase6E_Readiness_Authority_and_Implementation_Output.txt:22:    d78771f Implement Phase 6D governance ratification
Phase6E_Readiness_Authority_and_Implementation_Output.txt:25:commit, and the complete Phase 6D change set was committed.
Phase6E_Readiness_Authority_and_Implementation_Output.txt:27:The established Phase 6 authority chain is now:
Phase6E_Readiness_Authority_and_Implementation_Output.txt:29:Phase 6A
Phase6E_Readiness_Authority_and_Implementation_Output.txt:32:Phase 6B
Phase6E_Readiness_Authority_and_Implementation_Output.txt:35:Phase 6C
Phase6E_Readiness_Authority_and_Implementation_Output.txt:36:    Immutable administrative merits review with `approved` or `rejected`.
Phase6E_Readiness_Authority_and_Implementation_Output.txt:38:Phase 6D
Phase6E_Readiness_Authority_and_Implementation_Output.txt:40:    Phase 6C review, with mandatory reviewer/ratifier separation.
Phase6E_Readiness_Authority_and_Implementation_Output.txt:42:Phase 6D explicitly grants no Phase 6E capability and no operational authority.
Phase6E_Readiness_Authority_and_Implementation_Output.txt:44:The committed Phase 6D record says that Phase 6E requires a separate accepted
Phase6E_Readiness_Authority_and_Implementation_Output.txt:46:establish whether the next narrow authority is:
Phase6E_Readiness_Authority_and_Implementation_Output.txt:52:- handoff into an existing execution authority;
Phase6E_Readiness_Authority_and_Implementation_Output.txt:53:- or another more precise boundary already defined elsewhere in the repository.
Phase6E_Readiness_Authority_and_Implementation_Output.txt:55:Therefore this run must not assume that “Phase 6E” means a broad operational
Phase6E_Readiness_Authority_and_Implementation_Output.txt:61:2. determine whether Phase 6E has an already-established exact purpose;
Phase6E_Readiness_Authority_and_Implementation_Output.txt:63:   scheduling, and scheduler ownership;
Phase6E_Readiness_Authority_and_Implementation_Output.txt:64:4. define the narrowest legitimate Phase 6E capability;
Phase6E_Readiness_Authority_and_Implementation_Output.txt:65:5. implement only if the authority is sufficiently established;
Phase6E_Readiness_Authority_and_Implementation_Output.txt:68:The scheduler and production experiment workers may be active. Preserve them.
Phase6E_Readiness_Authority_and_Implementation_Output.txt:75:Perform a repository-grounded Phase 6E readiness analysis and, only if the
Phase6E_Readiness_Authority_and_Implementation_Output.txt:77:authority, implement the narrowest complete Phase 6E increment in the
Phase6E_Readiness_Authority_and_Implementation_Output.txt:80:Do not infer Phase 6E solely from its sequence number.
Phase6E_Readiness_Authority_and_Implementation_Output.txt:82:Do not assume Phase 6E means activation, execution, queueing, scheduling, or
Phase6E_Readiness_Authority_and_Implementation_Output.txt:83:scheduler integration until the committed repository establishes that meaning.
Phase6E_Readiness_Authority_and_Implementation_Output.txt:87:Determine the exact next authority after Phase 6D governance ratification.
Phase6E_Readiness_Authority_and_Implementation_Output.txt:91:A. If the committed repository defines Phase 6E precisely enough:
Phase6E_Readiness_Authority_and_Implementation_Output.txt:93:   Implement that exact narrow authority with complete domain, persistence,
Phase6E_Readiness_Authority_and_Implementation_Output.txt:97:B. If the repository does not define Phase 6E precisely enough:
Phase6E_Readiness_Authority_and_Implementation_Output.txt:99:   Do not invent an authority and do not make speculative implementation
Phase6E_Readiness_Authority_and_Implementation_Output.txt:103:   - the unresolved authority decision;
Phase6E_Readiness_Authority_and_Implementation_Output.txt:107:   - a recommended Phase 6E definition;
Phase6E_Readiness_Authority_and_Implementation_Output.txt:108:   - a ready-to-run implementation prompt for that recommended definition.
Phase6E_Readiness_Authority_and_Implementation_Output.txt:119:6. complete untracked-file list
Phase6E_Readiness_Authority_and_Implementation_Output.txt:124:11. local and remote branch/ref relationships relevant to Phase 6
Phase6E_Readiness_Authority_and_Implementation_Output.txt:125:12. scheduler and worker processes using OS process tools only
Phase6E_Readiness_Authority_and_Implementation_Output.txt:129:- Phase 6D commit `d78771f` is present;
Phase6E_Readiness_Authority_and_Implementation_Output.txt:137:If the worktree is clean and Phase 6D is the current baseline:
Phase6E_Readiness_Authority_and_Implementation_Output.txt:140:- branch from the exact current Phase 6D commit;
Phase6E_Readiness_Authority_and_Implementation_Output.txt:154:- Do not stop, suspend, resume, renice, or signal scheduler or worker processes.
Phase6E_Readiness_Authority_and_Implementation_Output.txt:155:- Do not run scheduler control commands.
Phase6E_Readiness_Authority_and_Implementation_Output.txt:156:- Do not run `LSTM_Release --scheduler-status`.
Phase6E_Readiness_Authority_and_Implementation_Output.txt:169:- Phase 6E;
Phase6E_Readiness_Authority_and_Implementation_Output.txt:170:- Phase 6F and later phases, if they reveal sequencing;
Phase6E_Readiness_Authority_and_Implementation_Output.txt:180:- scheduler ownership;
Phase6E_Readiness_Authority_and_Implementation_Output.txt:186:- existing Phase 4C conversion-execution authority;
Phase6E_Readiness_Authority_and_Implementation_Output.txt:187:- existing Phase 4D materialization/handoff authority;
Phase6E_Readiness_Authority_and_Implementation_Output.txt:189:- manual versus automatic authority;
Phase6E_Readiness_Authority_and_Implementation_Output.txt:195:- Phase 6A, 6B, 6C, and 6D phase documents;
Phase6E_Readiness_Authority_and_Implementation_Output.txt:199:- scheduler ownership ADRs;
Phase6E_Readiness_Authority_and_Implementation_Output.txt:207:Do not treat uncommitted prompt/output files as architecture authority.
Phase6E_Readiness_Authority_and_Implementation_Output.txt:213:1. What fact does Phase 6D ratification establish?
Phase6E_Readiness_Authority_and_Implementation_Output.txt:214:2. What capability remains absent after Phase 6D?
Phase6E_Readiness_Authority_and_Implementation_Output.txt:216:4. Does a committed component already own the eventual execution operation?
Phase6E_Readiness_Authority_and_Implementation_Output.txt:217:5. Should Phase 6E create new operational authority or only authorize/handoff to
Phase6E_Readiness_Authority_and_Implementation_Output.txt:218:   an existing authority?
Phase6E_Readiness_Authority_and_Implementation_Output.txt:222:9. Is scheduler admission distinct from worker launch?
Phase6E_Readiness_Authority_and_Implementation_Output.txt:223:10. Can Phase 6E safely remain non-operational while creating an authorization
Phase6E_Readiness_Authority_and_Implementation_Output.txt:227:12. What exact immutable upstream identity must Phase 6E consume?
Phase6E_Readiness_Authority_and_Implementation_Output.txt:229:14. What lifecycle state, if any, may Phase 6E create?
Phase6E_Readiness_Authority_and_Implementation_Output.txt:230:15. What operations must remain forbidden until later phases?
Phase6E_Readiness_Authority_and_Implementation_Output.txt:232:17. Is the authority precise enough to implement without architectural
Phase6E_Readiness_Authority_and_Implementation_Output.txt:242:- consume exact persisted Phase 6D ratification;
Phase6E_Readiness_Authority_and_Implementation_Output.txt:243:- never recompute Phase 5, Phase 6A, Phase 6C, or Phase 6D truth;
Phase6E_Readiness_Authority_and_Implementation_Output.txt:250:- add no scheduler polling behavior;
Phase6E_Readiness_Authority_and_Implementation_Output.txt:252:- add no experiment/model mutation unless the committed Phase 6E authority
Phase6E_Readiness_Authority_and_Implementation_Output.txt:254:- reuse an existing execution authority rather than duplicating it;
Phase6E_Readiness_Authority_and_Implementation_Output.txt:261:- proposed exact Phase 6E name;
Phase6E_Readiness_Authority_and_Implementation_Output.txt:262:- one-sentence authority question;
Phase6E_Readiness_Authority_and_Implementation_Output.txt:272:- why this is not semantic duplication of Phase 6C or 6D;
Phase6E_Readiness_Authority_and_Implementation_Output.txt:282:Implement the smallest complete Phase 6E increment.
Phase6E_Readiness_Authority_and_Implementation_Output.txt:284:Expected quality requirements, adapted to the exact authority found:
Phase6E_Readiness_Authority_and_Implementation_Output.txt:290:- exact Phase 6D ratification binding;
Phase6E_Readiness_Authority_and_Implementation_Output.txt:291:- complete transitive upstream identity;
Phase6E_Readiness_Authority_and_Implementation_Output.txt:292:- fixed authority role and decision/state;
Phase6E_Readiness_Authority_and_Implementation_Output.txt:300:- fixed negative flags for every authority not granted;
Phase6E_Readiness_Authority_and_Implementation_Output.txt:311:- exact Phase 6D/6C/6B provenance;
Phase6E_Readiness_Authority_and_Implementation_Output.txt:320:- no update/delete/truncate authority;
Phase6E_Readiness_Authority_and_Implementation_Output.txt:323:- no production-table mutation outside the exact accepted authority.
Phase6E_Readiness_Authority_and_Implementation_Output.txt:335:- malformed-storage rejection;
Phase6E_Readiness_Authority_and_Implementation_Output.txt:338:- read-only lookup/list operations where appropriate;
Phase6E_Readiness_Authority_and_Implementation_Output.txt:339:- no side effects outside the exact Phase 6E authority.
Phase6E_Readiness_Authority_and_Implementation_Output.txt:346:entry point for Phase 6E and the CLI cannot be mistaken for execution or
Phase6E_Readiness_Authority_and_Implementation_Output.txt:347:scheduler activation.
Phase6E_Readiness_Authority_and_Implementation_Output.txt:354:- print authority and non-authority boundaries;
Phase6E_Readiness_Authority_and_Implementation_Output.txt:355:- avoid production side effects unless the exact Phase 6E operation is
Phase6E_Readiness_Authority_and_Implementation_Output.txt:367:- exact Phase 6D binding;
Phase6E_Readiness_Authority_and_Implementation_Output.txt:371:- caller-injected authority rejection;
Phase6E_Readiness_Authority_and_Implementation_Output.txt:383:- read-only operations;
Phase6E_Readiness_Authority_and_Implementation_Output.txt:386:- unchanged Phase 6A–6D evidence;
Phase6E_Readiness_Authority_and_Implementation_Output.txt:387:- unchanged experiment, model, continuation, scheduler, and worker fixtures
Phase6E_Readiness_Authority_and_Implementation_Output.txt:388:  unless exact Phase 6E authority intentionally changes one;
Phase6E_Readiness_Authority_and_Implementation_Output.txt:391:- no operational side effects beyond the accepted authority.
Phase6E_Readiness_Authority_and_Implementation_Output.txt:397:- Phase 6E phase document;
Phase6E_Readiness_Authority_and_Implementation_Output.txt:428:- relevant Phase 6A–6D regressions;
Phase6E_Readiness_Authority_and_Implementation_Output.txt:429:- Phase 4/5 regressions implicated by reused authority;
Phase6E_Readiness_Authority_and_Implementation_Output.txt:433:- stale-name and authority searches;
Phase6E_Readiness_Authority_and_Implementation_Output.txt:471:11. confirm scheduler/worker processes were not interrupted;
Phase6E_Readiness_Authority_and_Implementation_Output.txt:481:2. Exact Phase 6E authority definition.
Phase6E_Readiness_Authority_and_Implementation_Output.txt:503:PHASE 6E IMPLEMENTATION COMPLETE AND READY FOR INDEPENDENT REVIEW
Phase6E_Readiness_Authority_and_Implementation_Output.txt:509:3. Why Phase 6E is not sufficiently defined.
Phase6E_Readiness_Authority_and_Implementation_Output.txt:510:4. Candidate authority definitions.
Phase6E_Readiness_Authority_and_Implementation_Output.txt:511:5. Recommended exact Phase 6E definition.
Phase6E_Readiness_Authority_and_Implementation_Output.txt:514:8. A complete follow-on implementation prompt.
Phase6E_Readiness_Authority_and_Implementation_Output.txt:520:PHASE 6E REQUIRES AUTHORITY-BOUNDARY DECISION BEFORE IMPLEMENTATION
Phase6E_Readiness_Authority_and_Implementation_Output.txt:524:PHASE 6E READINESS REVIEW BLOCKED
Phase6_Next_Architecture_Recommendation_Output.txt:15:the next evolution of the Phase 6 workflow.
Phase6_Next_Architecture_Recommendation_Output.txt:19:Do NOT implement Phase 6E.
Phase6_Next_Architecture_Recommendation_Output.txt:22:the existing repository architecture and completed Phase 6A–6D work.
Phase6_Next_Architecture_Recommendation_Output.txt:26:Phase 6D has completed independent review.
Phase6_Next_Architecture_Recommendation_Output.txt:30:Previous readiness analysis concluded there is no accepted repository
Phase6_Next_Architecture_Recommendation_Output.txt:31:definition for Phase 6E and correctly declined implementation.
Phase6_Next_Architecture_Recommendation_Output.txt:41:3.  Why it belongs after Phase 6D.
Phase6_Next_Architecture_Recommendation_Output.txt:70:I’ll perform a read-only architecture review—starting with the repository’s guidance, Phase 6A–6D artifacts, ADRs, state machines, and scheduler boundaries—then compare plausible next increments and recommend one. No files or database state will be changed.
Phase6_Next_Architecture_Recommendation_Output.txt:71:  → Inventory ADRs, Phase 6A–6D documentation, and relevant workflow code/tests
Phase6_Next_Architecture_Recommendation_Output.txt:72:  • Reconstruct current authority, state, ownership, transaction, concurrency, and scheduler contracts
Phase6_Next_Architecture_Recommendation_Output.txt:74:  • Deliver a review-ready architectural recommendation with risks and sequencing
Phase6_Next_Architecture_Recommendation_Output.txt:115:docs/architecture/Volume_VIII_Recommendation_Engine.md
Phase6_Next_Architecture_Recommendation_Output.txt:119:Tests/ExperimentRecommendationTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:120:Tests/ExperimentRecommendationCampaignHandoffRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:121:Tests/ExperimentRecommendationCampaignFollowUpProposalTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:122:Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:123:Tests/ExperimentRecommendationEvaluationMigrationTests.sql
Phase6_Next_Architecture_Recommendation_Output.txt:124:Tests/ExperimentRecommendationCampaignApprovalRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:125:Tests/ExperimentRecommendationCampaignHandoffTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:127:Tests/ExperimentRecommendationConversionActivationMigrationTests.sql
Phase6_Next_Architecture_Recommendation_Output.txt:128:Tests/ExperimentRecommendationCampaignOutcomePolicyTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:129:Tests/ExperimentRecommendationCampaignProposalReviewRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:130:Tests/ExperimentRecommendationCampaignFollowUpProposalReviewRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:131:Tests/ExperimentRecommendationCampaignExecutionRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:132:Tests/ExperimentRecommendationPersistenceTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:133:Tests/ExperimentRecommendationCampaignStatusRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:134:Tests/ExperimentRecommendationCampaignLaunchRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:135:Tests/ExperimentRecommendationCampaignFollowUpProposalReviewTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:136:Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:137:Tests/ExperimentRecommendationCampaignApprovalMigrationTests.sql
Phase6_Next_Architecture_Recommendation_Output.txt:138:Tests/ExperimentRecommendationCampaignLaunchTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:139:Tests/ExperimentRecommendationCampaignStatusTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:140:Tests/ExperimentRecommendationScoringTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:141:Tests/ExperimentRecommendationConversionActivationTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:142:Tests/ExperimentRecommendationCampaignFollowUpProposalMigrationTests.sql
Phase6_Next_Architecture_Recommendation_Output.txt:143:Tests/ExperimentRecommendationRankingMigrationTests.sql
Phase6_Next_Architecture_Recommendation_Output.txt:144:Tests/ExperimentRecommendationCampaignPlanningRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:145:Tests/ExperimentRecommendationConversionActivationRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:147:Tests/ExperimentRecommendationCampaignReviewTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:148:Tests/ExperimentRecommendationCampaignActivationRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:149:Tests/ExperimentRecommendationCampaignMaterializationTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:150:Tests/ExperimentRecommendationCampaignFollowUpProposalReviewMigrationTests.sql
Phase6_Next_Architecture_Recommendation_Output.txt:151:Tests/ExperimentRecommendationReviewTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:152:Tests/ExperimentRecommendationConversionTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:153:Tests/ExperimentRecommendationEvaluationRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:154:Tests/ExperimentRecommendationReviewRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:155:Tests/ExperimentRecommendationConversionProposalReviewRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:156:Tests/ExperimentRecommendationConversionRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:157:Tests/ExperimentRecommendationServiceTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:158:Tests/ExperimentRecommendationScoringRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:159:Tests/ExperimentRecommendationConversionWorkflowTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:160:Tests/ExperimentRecommendationConversionExecutionMigrationTests.sql
Phase6_Next_Architecture_Recommendation_Output.txt:161:Tests/ExperimentRecommendationScoringMigrationTests.sql
Phase6_Next_Architecture_Recommendation_Output.txt:162:Tests/ExperimentRecommendationCampaignFollowUpProposalRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:163:Tests/ExperimentRecommendationConversionProposalReviewMigrationTests.sql
Phase6_Next_Architecture_Recommendation_Output.txt:166:Tests/ExperimentRecommendationConversionExecutionRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:167:Tests/ExperimentRecommendationReviewMigrationTests.sql
Phase6_Next_Architecture_Recommendation_Output.txt:168:Tests/ExperimentRecommendationCampaignMaterializationMigrationTests.sql
Phase6_Next_Architecture_Recommendation_Output.txt:171:Tests/ExperimentRecommendationCampaignActivationTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:172:Tests/ExperimentRecommendationRankingRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:175:Tests/ExperimentRecommendationEvaluationTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:176:Tests/ExperimentRecommendationConversionMigrationTests.sql
Phase6_Next_Architecture_Recommendation_Output.txt:179:Tests/ExperimentRecommendationCampaignPlanningTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:182:Tests/ExperimentRecommendationCampaignOutcomeAssessmentIntegrationTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:183:Tests/ExperimentRecommendationConversionWorkflowRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:184:docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md
Phase6_Next_Architecture_Recommendation_Output.txt:185:Tests/ExperimentRecommendationConversionProposalReviewTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:186:Tests/ExperimentRecommendationCampaignExecutionTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:188:Tests/ExperimentRecommendationCampaignProposalReviewTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:189:Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationMigrationTests.sql
Phase6_Next_Architecture_Recommendation_Output.txt:190:Tests/ExperimentRecommendationCampaignApprovalTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:191:Tests/ExperimentRecommendationRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:192:Tests/ExperimentRecommendationCampaignMaterializationRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:193:Tests/ExperimentRecommendationRankingTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:194:Tests/ExperimentRecommendationCampaignOutcomeAssessmentTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:195:Tests/ExperimentRecommendationCandidateGeneratorTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:201:Database/migrations/005_experiment_scheduler.sql
Phase6_Next_Architecture_Recommendation_Output.txt:240:Sources/ExperimentRecommendationCampaignFollowUpProposalRepository.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:242:Sources/ExperimentRecommendationCampaignExecution.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:243:Sources/ExperimentRecommendationCampaignStatus.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:244:Sources/ExperimentRecommendationCampaignHandoffRepository.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:245:Sources/ExperimentRecommendationConversionExecutionRepository.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:246:Sources/ExperimentRecommendationCampaignFollowUpProposalReview.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:247:Sources/ExperimentRecommendationCampaignMaterializationService.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:248:Sources/ExperimentRecommendationConversionProposalReviewRepository.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:249:Sources/ExperimentRecommendationConversionActivation.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:250:Sources/ExperimentRecommendationCampaignReview.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:251:Sources/ExperimentRecommendationCampaignOutcomeAssessment.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:252:Sources/ExperimentRecommendationCampaignActivationRepository.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:253:Sources/ExperimentRecommendationCampaignActivationService.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:254:Sources/ExperimentRecommendationCampaignOutcomeAssessmentRepository.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:255:Sources/ExperimentRecommendationConversionProposalReviewService.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:256:Sources/ExperimentRecommendationCampaignApprovalService.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:258:Sources/ExperimentRecommendationCampaignFollowUpProposalReviewPresentation.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:259:Sources/ExperimentRecommendationConversionProposalReview.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:260:Sources/ExperimentRecommendationCampaignExecutionRepository.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:261:Sources/ExperimentRecommendationScoringService.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:262:Sources/ExperimentRecommendationCampaignFollowUpProposalReviewRepository.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:263:Sources/ExperimentRecommendationCampaignHandoffService.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:264:Sources/ExperimentRecommendationEvaluationService.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:265:Sources/ExperimentRecommendationCampaignMaterialization.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:266:Sources/ExperimentRecommendationConversionWorkflowService.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:267:Sources/ExperimentRecommendationCampaignExecutionService.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:268:Sources/ExperimentRecommendationConversionActivationService.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:269:Sources/ExperimentRecommendationConversionExecutionService.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:270:Sources/ExperimentRecommendationConversionRepository.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:271:Sources/ExperimentRecommendationCampaignApprovalRepository.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:272:Sources/ExperimentRecommendationCampaignApproval.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:273:Sources/ExperimentRecommendationRanking.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:274:Sources/ExperimentRecommendationCampaignHandoff.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:275:Sources/ExperimentRecommendation.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:276:Sources/ExperimentRecommendationCampaignStatusService.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:277:Sources/ExperimentRecommendationCampaignFollowUpProposalRatification.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:278:Sources/ExperimentRecommendationRepository.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:281:Sources/ExperimentRecommendationCampaignReviewService.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:282:Sources/ExperimentRecommendationRankingService.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:283:Sources/ExperimentRecommendationEvaluation.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:284:Sources/ExperimentRecommendationCampaignStatusRepository.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:285:Sources/ExperimentRecommendationCampaignFollowUpProposalRatificationService.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:286:Sources/ExperimentRecommendationCampaignPlanningRepository.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:287:Sources/ExperimentRecommendationCampaignProposalReview.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:288:Sources/ExperimentRecommendationCampaignProposalReviewService.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:289:Sources/ExperimentRecommendationCampaignLaunchRepository.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:290:Sources/ExperimentRecommendationCampaignFollowUpProposalPreview.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:291:Sources/ExperimentRecommendationCandidateGenerator.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:292:Sources/ExperimentRecommendationCampaignLaunch.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:293:Sources/ExperimentRecommendationCampaignLaunchService.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:294:Sources/ExperimentRecommendationConversionWorkflowRepository.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:295:Sources/ExperimentRecommendationCampaignMaterializationRepository.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:296:Sources/ExperimentRecommendationCampaignPlanningService.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:297:Sources/ExperimentRecommendationCampaignProposalReviewRepository.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:298:Sources/ExperimentRecommendationCampaignActivation.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:299:Sources/ExperimentRecommendationConversionActivationRepository.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:300:Sources/ExperimentRecommendationRankingRepository.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:301:Sources/ExperimentRecommendationConversionWorkflow.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:302:Sources/ExperimentRecommendationScoring.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:304:Sources/ExperimentRecommendationService.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:306:Sources/ExperimentRecommendationCampaignOutcomePolicy.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:307:Sources/ExperimentRecommendationCampaignOutcomeAssessmentService.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:308:Sources/ExperimentRecommendationReview.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:309:Sources/ExperimentRecommendationCampaignPlanning.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:311:Sources/ExperimentRecommendationEvaluationRepository.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:312:Sources/ExperimentRecommendationConversion.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:313:Sources/ExperimentRecommendationCampaignFollowUpProposalRatificationRepository.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:314:Sources/ExperimentRecommendationEvaluation.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:315:Sources/ExperimentRecommendationCampaignFollowUpProposal.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:316:Sources/ExperimentRecommendationRankingService.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:318:Sources/ExperimentRecommendationCampaignOutcomePolicy.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:319:Sources/ExperimentRecommendationCampaignFollowUpProposalRatificationService.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:320:Sources/ExperimentRecommendationService.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:321:Sources/ExperimentRecommendationCampaignProposalReview.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:322:Sources/ExperimentRecommendationCampaignProposalReviewService.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:323:Sources/ExperimentRecommendationCampaignPlanningRepository.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:324:Sources/ExperimentRecommendationCampaignLaunch.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:325:Sources/ExperimentRecommendationCampaignFollowUpProposalPreview.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:326:Sources/ExperimentRecommendationCampaignLaunchRepository.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:327:Sources/ExperimentRecommendationCandidateGenerator.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:328:Sources/ExperimentRecommendationCampaignPlanningService.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:329:Sources/ExperimentRecommendationConversionWorkflowRepository.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:330:Sources/ExperimentRecommendationCampaignMaterializationRepository.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:331:Sources/ExperimentRecommendationCampaignLaunchService.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:332:Sources/ExperimentRecommendationCampaignHandoff.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:333:Sources/ExperimentRecommendation.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:334:Sources/ExperimentRecommendationCampaignStatusService.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:335:Sources/ExperimentRecommendationCampaignReviewService.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:336:Sources/ExperimentRecommendationRepository.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:337:Sources/ExperimentRecommendationCampaignFollowUpProposalRatification.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:338:Sources/ExperimentRecommendationCampaignPlanning.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:344:Database/migrations/005_experiment_scheduler.sql
Phase6_Next_Architecture_Recommendation_Output.txt:389:Sources/ExperimentRecommendation.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:390:Sources/ExperimentRecommendation.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:391:Sources/ExperimentRecommendationCampaignActivation.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:392:Sources/ExperimentRecommendationCampaignActivation.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:393:Sources/ExperimentRecommendationCampaignActivationRepository.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:394:Sources/ExperimentRecommendationCampaignActivationRepository.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:395:Sources/ExperimentRecommendationCampaignActivationService.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:396:Sources/ExperimentRecommendationCampaignActivationService.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:397:Sources/ExperimentRecommendationCampaignApproval.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:398:Sources/ExperimentRecommendationCampaignApproval.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:399:Sources/ExperimentRecommendationCampaignApprovalRepository.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:400:Sources/ExperimentRecommendationCampaignApprovalRepository.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:401:Sources/ExperimentRecommendationCampaignApprovalService.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:402:Sources/ExperimentRecommendationCampaignApprovalService.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:403:Sources/ExperimentRecommendationCampaignExecution.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:404:Sources/ExperimentRecommendationCampaignExecution.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:405:Sources/ExperimentRecommendationCampaignExecutionRepository.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:406:Sources/ExperimentRecommendationCampaignExecutionRepository.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:407:Sources/ExperimentRecommendationCampaignExecutionService.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:408:Sources/ExperimentRecommendationCampaignExecutionService.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:409:Sources/ExperimentRecommendationCampaignFollowUpProposal.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:410:Sources/ExperimentRecommendationCampaignFollowUpProposal.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:411:Sources/ExperimentRecommendationCampaignFollowUpProposalPreview.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:412:Sources/ExperimentRecommendationCampaignFollowUpProposalPreview.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:413:Sources/ExperimentRecommendationCampaignFollowUpProposalRatification.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:414:Sources/ExperimentRecommendationCampaignFollowUpProposalRatification.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:415:Sources/ExperimentRecommendationCampaignFollowUpProposalRatificationRepository.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:416:Sources/ExperimentRecommendationCampaignFollowUpProposalRatificationRepository.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:417:Sources/ExperimentRecommendationCampaignFollowUpProposalRatificationService.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:418:Sources/ExperimentRecommendationCampaignFollowUpProposalRatificationService.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:419:Sources/ExperimentRecommendationCampaignFollowUpProposalRepository.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:420:Sources/ExperimentRecommendationCampaignFollowUpProposalRepository.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:421:Sources/ExperimentRecommendationCampaignFollowUpProposalReview.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:422:Sources/ExperimentRecommendationCampaignFollowUpProposalReview.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:423:Sources/ExperimentRecommendationCampaignFollowUpProposalReviewPresentation.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:424:Sources/ExperimentRecommendationCampaignFollowUpProposalReviewPresentation.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:425:Sources/ExperimentRecommendationCampaignFollowUpProposalReviewRepository.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:426:Sources/ExperimentRecommendationCampaignFollowUpProposalReviewRepository.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:427:Sources/ExperimentRecommendationCampaignHandoff.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:428:Sources/ExperimentRecommendationCampaignHandoff.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:429:Sources/ExperimentRecommendationCampaignHandoffRepository.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:430:Sources/ExperimentRecommendationCampaignHandoffRepository.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:431:Sources/ExperimentRecommendationCampaignHandoffService.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:432:Sources/ExperimentRecommendationCampaignHandoffService.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:433:Sources/ExperimentRecommendationCampaignLaunch.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:434:Sources/ExperimentRecommendationCampaignLaunch.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:435:Sources/ExperimentRecommendationCampaignLaunchRepository.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:436:Sources/ExperimentRecommendationCampaignLaunchRepository.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:437:Sources/ExperimentRecommendationCampaignLaunchService.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:438:Sources/ExperimentRecommendationCampaignLaunchService.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:439:Sources/ExperimentRecommendationCampaignMaterialization.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:440:Sources/ExperimentRecommendationCampaignMaterialization.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:441:Sources/ExperimentRecommendationCampaignMaterializationRepository.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:442:Sources/ExperimentRecommendationCampaignMaterializationRepository.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:443:Sources/ExperimentRecommendationCampaignMaterializationService.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:444:Sources/ExperimentRecommendationCampaignMaterializationService.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:445:Sources/ExperimentRecommendationCampaignOutcomeAssessment.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:446:Sources/ExperimentRecommendationCampaignOutcomeAssessment.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:447:Sources/ExperimentRecommendationCampaignOutcomeAssessmentRepository.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:448:Sources/ExperimentRecommendationCampaignOutcomeAssessmentRepository.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:449:Sources/ExperimentRecommendationCampaignOutcomeAssessmentService.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:450:Sources/ExperimentRecommendationCampaignOutcomeAssessmentService.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:451:Sources/ExperimentRecommendationCampaignOutcomePolicy.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:452:Sources/ExperimentRecommendationCampaignOutcomePolicy.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:453:Sources/ExperimentRecommendationCampaignPlanning.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:454:Sources/ExperimentRecommendationCampaignPlanning.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:455:Sources/ExperimentRecommendationCampaignPlanningRepository.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:456:Sources/ExperimentRecommendationCampaignPlanningRepository.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:457:Sources/ExperimentRecommendationCampaignPlanningService.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:458:Sources/ExperimentRecommendationCampaignPlanningService.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:459:Sources/ExperimentRecommendationCampaignProposalReview.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:460:Sources/ExperimentRecommendationCampaignProposalReview.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:461:Sources/ExperimentRecommendationCampaignProposalReviewRepository.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:462:Sources/ExperimentRecommendationCampaignProposalReviewRepository.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:463:Sources/ExperimentRecommendationCampaignProposalReviewService.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:464:Sources/ExperimentRecommendationCampaignProposalReviewService.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:465:Sources/ExperimentRecommendationCampaignReview.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:466:Sources/ExperimentRecommendationCampaignReview.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:467:Sources/ExperimentRecommendationCampaignReviewService.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:468:Sources/ExperimentRecommendationCampaignReviewService.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:469:Sources/ExperimentRecommendationCampaignStatus.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:470:Sources/ExperimentRecommendationCampaignStatus.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:471:Sources/ExperimentRecommendationCampaignStatusRepository.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:472:Sources/ExperimentRecommendationCampaignStatusRepository.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:473:Sources/ExperimentRecommendationCampaignStatusService.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:474:Sources/ExperimentRecommendationCampaignStatusService.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:475:Sources/ExperimentRecommendationCandidateGenerator.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:476:Sources/ExperimentRecommendationCandidateGenerator.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:477:Sources/ExperimentRecommendationConversion.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:478:Sources/ExperimentRecommendationConversion.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:479:Sources/ExperimentRecommendationConversionActivation.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:480:Sources/ExperimentRecommendationConversionActivation.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:481:Sources/ExperimentRecommendationConversionActivationRepository.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:482:Sources/ExperimentRecommendationConversionActivationRepository.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:483:Sources/ExperimentRecommendationConversionActivationService.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:484:Sources/ExperimentRecommendationConversionActivationService.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:485:Sources/ExperimentRecommendationConversionExecutionRepository.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:486:Sources/ExperimentRecommendationConversionExecutionRepository.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:487:Sources/ExperimentRecommendationConversionExecutionService.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:488:Sources/ExperimentRecommendationConversionExecutionService.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:489:Sources/ExperimentRecommendationConversionProposalReview.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:490:Sources/ExperimentRecommendationConversionProposalReview.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:491:Sources/ExperimentRecommendationConversionProposalReviewRepository.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:492:Sources/ExperimentRecommendationConversionProposalReviewRepository.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:493:Sources/ExperimentRecommendationConversionProposalReviewService.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:494:Sources/ExperimentRecommendationConversionProposalReviewService.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:495:Sources/ExperimentRecommendationConversionRepository.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:496:Sources/ExperimentRecommendationConversionRepository.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:497:Sources/ExperimentRecommendationConversionWorkflow.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:498:Sources/ExperimentRecommendationConversionWorkflow.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:499:Sources/ExperimentRecommendationConversionWorkflowRepository.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:500:Sources/ExperimentRecommendationConversionWorkflowRepository.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:501:Sources/ExperimentRecommendationConversionWorkflowService.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:502:Sources/ExperimentRecommendationConversionWorkflowService.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:503:Sources/ExperimentRecommendationEvaluation.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:504:Sources/ExperimentRecommendationEvaluation.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:505:Sources/ExperimentRecommendationEvaluationRepository.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:506:Sources/ExperimentRecommendationEvaluationRepository.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:507:Sources/ExperimentRecommendationEvaluationService.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:508:Sources/ExperimentRecommendationEvaluationService.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:509:Sources/ExperimentRecommendationRanking.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:510:Sources/ExperimentRecommendationRanking.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:511:Sources/ExperimentRecommendationRankingRepository.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:512:Sources/ExperimentRecommendationRankingRepository.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:513:Sources/ExperimentRecommendationRankingService.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:514:Sources/ExperimentRecommendationRankingService.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:515:Sources/ExperimentRecommendationRepository.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:516:Sources/ExperimentRecommendationRepository.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:517:Sources/ExperimentRecommendationReview.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:518:Sources/ExperimentRecommendationReview.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:519:Sources/ExperimentRecommendationScoring.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:520:Sources/ExperimentRecommendationScoring.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:521:Sources/ExperimentRecommendationScoringService.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:522:Sources/ExperimentRecommendationService.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:523:Sources/ExperimentRecommendationService.hpp
Phase6_Next_Architecture_Recommendation_Output.txt:530:Tests/ExperimentRecommendationCampaignActivationRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:531:Tests/ExperimentRecommendationCampaignActivationTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:532:Tests/ExperimentRecommendationCampaignApprovalMigrationTests.sql
Phase6_Next_Architecture_Recommendation_Output.txt:533:Tests/ExperimentRecommendationCampaignApprovalRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:534:Tests/ExperimentRecommendationCampaignApprovalTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:535:Tests/ExperimentRecommendationCampaignExecutionRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:536:Tests/ExperimentRecommendationCampaignExecutionTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:537:Tests/ExperimentRecommendationCampaignFollowUpProposalMigrationTests.sql
Phase6_Next_Architecture_Recommendation_Output.txt:538:Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationMigrationTests.sql
Phase6_Next_Architecture_Recommendation_Output.txt:539:Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:540:Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:541:Tests/ExperimentRecommendationCampaignFollowUpProposalRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:542:Tests/ExperimentRecommendationCampaignFollowUpProposalReviewMigrationTests.sql
Phase6_Next_Architecture_Recommendation_Output.txt:543:Tests/ExperimentRecommendationCampaignFollowUpProposalReviewRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:544:Tests/ExperimentRecommendationCampaignFollowUpProposalReviewTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:545:Tests/ExperimentRecommendationCampaignFollowUpProposalTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:546:Tests/ExperimentRecommendationCampaignHandoffRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:547:Tests/ExperimentRecommendationCampaignHandoffTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:548:Tests/ExperimentRecommendationCampaignLaunchRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:549:Tests/ExperimentRecommendationCampaignLaunchTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:550:Tests/ExperimentRecommendationCampaignMaterializationMigrationTests.sql
Phase6_Next_Architecture_Recommendation_Output.txt:551:Tests/ExperimentRecommendationCampaignMaterializationRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:552:Tests/ExperimentRecommendationCampaignMaterializationTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:553:Tests/ExperimentRecommendationCampaignOutcomeAssessmentIntegrationTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:554:Tests/ExperimentRecommendationCampaignOutcomeAssessmentTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:555:Tests/ExperimentRecommendationCampaignOutcomePolicyTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:556:Tests/ExperimentRecommendationCampaignPlanningRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:557:Tests/ExperimentRecommendationCampaignPlanningTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:558:Tests/ExperimentRecommendationCampaignProposalReviewRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:559:Tests/ExperimentRecommendationCampaignProposalReviewTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:560:Tests/ExperimentRecommendationCampaignReviewTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:561:Tests/ExperimentRecommendationCampaignStatusRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:562:Tests/ExperimentRecommendationCampaignStatusTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:563:Tests/ExperimentRecommendationCandidateGeneratorTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:564:Tests/ExperimentRecommendationConversionActivationMigrationTests.sql
Phase6_Next_Architecture_Recommendation_Output.txt:565:Tests/ExperimentRecommendationConversionActivationRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:566:Tests/ExperimentRecommendationConversionActivationTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:567:Tests/ExperimentRecommendationConversionExecutionMigrationTests.sql
Phase6_Next_Architecture_Recommendation_Output.txt:568:Tests/ExperimentRecommendationConversionExecutionRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:569:Tests/ExperimentRecommendationConversionMigrationTests.sql
Phase6_Next_Architecture_Recommendation_Output.txt:570:Tests/ExperimentRecommendationConversionProposalReviewMigrationTests.sql
Phase6_Next_Architecture_Recommendation_Output.txt:571:Tests/ExperimentRecommendationConversionProposalReviewRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:572:Tests/ExperimentRecommendationConversionProposalReviewTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:573:Tests/ExperimentRecommendationConversionRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:574:Tests/ExperimentRecommendationConversionTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:575:Tests/ExperimentRecommendationConversionWorkflowRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:576:Tests/ExperimentRecommendationConversionWorkflowTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:577:Tests/ExperimentRecommendationEvaluationMigrationTests.sql
Phase6_Next_Architecture_Recommendation_Output.txt:578:Tests/ExperimentRecommendationEvaluationRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:579:Tests/ExperimentRecommendationEvaluationTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:580:Tests/ExperimentRecommendationPersistenceTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:581:Tests/ExperimentRecommendationRankingMigrationTests.sql
Phase6_Next_Architecture_Recommendation_Output.txt:582:Tests/ExperimentRecommendationRankingRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:583:Tests/ExperimentRecommendationRankingTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:584:Tests/ExperimentRecommendationRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:585:Tests/ExperimentRecommendationReviewMigrationTests.sql
Phase6_Next_Architecture_Recommendation_Output.txt:586:Tests/ExperimentRecommendationReviewRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:587:Tests/ExperimentRecommendationReviewTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:588:Tests/ExperimentRecommendationScoringMigrationTests.sql
Phase6_Next_Architecture_Recommendation_Output.txt:589:Tests/ExperimentRecommendationScoringRepositoryTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:590:Tests/ExperimentRecommendationScoringTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:591:Tests/ExperimentRecommendationServiceTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:592:Tests/ExperimentRecommendationTests.cpp
Phase6_Next_Architecture_Recommendation_Output.txt:601:docs/architecture/Volume_VIII_Recommendation_Engine.md
Phase6_Next_Architecture_Recommendation_Output.txt:611:docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md
Phase6_Next_Architecture_Recommendation_Output.txt:623:# ExpertAdvisor Development Instructions
Phase6_Next_Architecture_Recommendation_Output.txt:625:## Project
Phase6_Next_Architecture_Recommendation_Output.txt:634:## Build
Phase6_Next_Architecture_Recommendation_Output.txt:642:## Tests
Phase6_Next_Architecture_Recommendation_Output.txt:646:Do not launch tests or executables that may interfere with active scheduler,
Phase6_Next_Architecture_Recommendation_Output.txt:649:## Scheduler Safety
Phase6_Next_Architecture_Recommendation_Output.txt:654:2. Check scheduler status.
Phase6_Next_Architecture_Recommendation_Output.txt:658:## Architecture
Phase6_Next_Architecture_Recommendation_Output.txt:675:## Development Style
Phase6_Next_Architecture_Recommendation_Output.txt:685:- Resolve compiler warnings, build errors, and failing tests before considering an implementation complete.
Phase6_Next_Architecture_Recommendation_Output.txt:687:## Review Output
Phase6_Next_Architecture_Recommendation_Output.txt:699:## Current Development Status
Phase6_Next_Architecture_Recommendation_Output.txt:708:- Phase 4: Recommendation campaigns
Phase6_Next_Architecture_Recommendation_Output.txt:710:- Phase 6A: ...
Phase6_Next_Architecture_Recommendation_Output.txt:711:- Phase 6B: ...
Phase6_Next_Architecture_Recommendation_Output.txt:712:- Phase 6C: ...
Phase6_Next_Architecture_Recommendation_Output.txt:714:Always preserve compatibility with completed phases.
Phase6_Next_Architecture_Recommendation_Output.txt:715:Do not redesign previously accepted architecture.
Phase6_Next_Architecture_Recommendation_Output.txt:717:## Never
Phase6_Next_Architecture_Recommendation_Output.txt:722:- Never redesign completed phases during an incremental implementation.
Phase6_Next_Architecture_Recommendation_Output.txt:724:# Architecture Decision Records
Phase6_Next_Architecture_Recommendation_Output.txt:726:## 1. Purpose
Phase6_Next_Architecture_Recommendation_Output.txt:733:## 2. Numbering and filenames
entry point for Phase 6E and the CLI cannot be mistaken for execution or
scheduler activation.

Any CLI must:

- require exact persisted identity;
- be explicit and non-automatic;
- support dry-run/read-only preview if meaningful;
- print authority and non-authority boundaries;
- avoid production side effects unless the exact Phase 6E operation is
  explicitly invoked.

TESTS

Provide focused coverage for:

- pure construction;
- immutable semantics;
- canonical golden value and hash;
- locale independence;
- UTF-8 and hostile controls;
- exact Phase 6D binding;
- missing, malformed, stale, or ineligible ratification;
- separation of duties;
- fixed role and decision;
- caller-injected authority rejection;
- exact replay;
- changed-payload conflict;
- concurrent equivalent and conflicting attempts;
- rollback;
- malformed storage;
- hash collision;
- append-only behavior;
- runtime least privilege;
- PUBLIC and NULL-ACL fallback;
- clean installation;
- upgrade from 044;
- read-only operations;
- no sequence advancement from reads;
- no advisory or tuple locks from reads;
- unchanged Phase 6A–6D evidence;
- unchanged experiment, model, continuation, scheduler, and worker fixtures
  unless exact Phase 6E authority intentionally changes one;
- relevant Phase 4, 5, and 6 regressions;
- project-file membership;
- no operational side effects beyond the accepted authority.

DOCUMENTATION

Create or update:

- Phase 6E phase document;
- ADR-0010 if an architecture decision is made;
- ADR index;
- Volume VIII;
- Volume XII;
- Database README if migration 045 is added;
- relevant references and revision histories.

Documentation must precisely distinguish:

- review;
- ratification;
- authorization;
- activation;
- execution;
- handoff;
- queueing;
- scheduling;
- worker launch.

VERIFICATION

Run the strongest safe checks.

At minimum:

- focused pure tests;
- migration/repository/service tests if implemented;
- clean and upgrade paths;
- concurrency;
- privilege tests;
- relevant Phase 6A–6D regressions;
- Phase 4/5 regressions implicated by reused authority;
- Xcode project membership;
- `xcodebuild -list -project ExpertAdvisor.xcodeproj`;
- `plutil -lint ExpertAdvisor.xcodeproj/project.pbxproj`;
- stale-name and authority searches;
- `git diff --check`;
- final tracked and untracked audit.

RELEASE BUILD

Use OS process tools to reassess load.

If safe, run an incremental build without Clean:

    xcodebuild \
      -project ExpertAdvisor.xcodeproj \
      -scheme "LSTM Release" \
      -configuration Release \
      -derivedDataPath DerivedData/ExpertAdvisor \
      build

Then verify:

    file DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release

If active workers make the build unsafe, defer it without interrupting them and
report the exact command as a mechanical verification gap.

FINAL AUDIT

Before reporting:

1. inspect every tracked modification;
2. inspect every untracked file directly;
3. confirm no unrelated changes;
4. confirm no generated artifacts are included;
5. confirm index remains empty;
6. run `git diff --check`;
7. record exact `git status --short`;
8. record exact `git diff --stat`;
9. list all untracked files;
10. confirm branch and HEAD;
11. confirm scheduler/worker processes were not interrupted;
12. confirm production database was not accessed;
13. confirm disposable objects were removed;
14. confirm no staging, commit, tag, or push occurred.

DELIVERABLE

If implementation proceeds, provide:

1. Executive summary.
2. Exact Phase 6E authority definition.
3. Repository evidence supporting that definition.
4. Distinction from Phases 6C and 6D.
5. Relationship to activation and execution.
6. Architecture and data-flow summary.
7. Files changed.
8. Domain contract.
9. Migration and privilege design.
10. Repository/service/CLI behavior.
11. Separation-of-duties policy.
12. Replay and concurrency behavior.
13. Tests and exact results.
14. Release build result or deferral.
15. Scheduler and production-data safety.
16. Remaining risks.
17. Exact final status and diff stat.
18. Complete untracked-file list.
19. Readiness for independent review.
20. Recommended next action.

End with exactly:

PHASE 6E IMPLEMENTATION COMPLETE AND READY FOR INDEPENDENT REVIEW

If implementation does not proceed, provide:

1. Executive readiness verdict.
2. Repository evidence.
3. Why Phase 6E is not sufficiently defined.
4. Candidate authority definitions.
5. Recommended exact Phase 6E definition.
6. Required ADR or architecture decision.
7. Risks and prohibited scope.
8. A complete follow-on implementation prompt.
9. Exact repository status.
10. Recommended next action.

End with exactly:

PHASE 6E REQUIRES AUTHORITY-BOUNDARY DECISION BEFORE IMPLEMENTATION

If repository state or safety conditions block useful work, end with exactly:

PHASE 6E READINESS REVIEW BLOCKED


Planning service
  → existing Phase 4A pure source-selection/candidate-generation components
```

Specific ownership:

- **Phase 6D repository:** Owns validated ratification hydration and the complete upstream chain.
- **Phase 6E repository:** Owns bounded, set-based loading of exact result experiment, model, analysis, and duplicate evidence.
- **Phase 6E domain:** Owns the plan contract, canonicalization, ordering, caps, stable reasons, and fixed non-authority semantics.
- **Phase 4A domain:** Continues to own mutation eligibility and candidate construction.
- **PostgreSQL:** Remains authoritative for persisted ratification and source evidence.
- **Experiment lifecycle and scheduler:** Own nothing in Phase 6E.

Raw Phase 6D or experiment rows must not escape repositories, and Phase 6E must not issue direct SQL outside its repository.

## 7. Transaction boundaries

Phase 6E should use one bounded:

`REPEATABLE READ, READ ONLY`

transaction to load:

1. The exact Phase 6D ratification.
2. The transitively validated Phase 6C/6B chain.
3. Ordered Phase 6A members.
4. Exact result experiments and their supported source evidence.
5. Current duplicate recommendation/experiment evidence.

The repository should use fixed-count, set-based queries and explicit ordering. It should acquire:

- no advisory locks;
- no tuple locks;
- no sequence values;
- no write transaction.

After loading the complete typed snapshot, the transaction can close before potentially larger canonicalization work. The plan is then calculated from that immutable snapshot.

Any future approval or materialization must be a different phase and transaction. It should reconstruct the Phase 6E plan and compare its full expected identity inside the write transaction, following the existing Phase 4D stale-evidence pattern ([Phase 4D approval](</Volumes/Developer SSD/ExpertAdvisor/docs/Phase4DExperimentRecommendationCampaignApproval.rst:17>)).

## 8. Concurrency considerations

Because Phase 6E is read-only:

- Concurrent identical plans do not conflict.
- Plans built from the same ratification, policy, and PostgreSQL snapshot must have identical canonical identities.
- Ordinary recommendation or experiment activity should not be blocked.
- Repeatable-read prevents mixed evidence within one plan.
- A plan may become stale immediately after the transaction ends.

A later write phase must recheck:

- ratification identity;
- source evidence identity;
- candidate canonical identities;
- current duplicate state;
- expected plan canonical text.

Concurrent insertion of a matching recommendation after planning should cause a deterministic reclassification or stale-plan rejection during future materialization—not duplicate creation.

## 9. Replay and determinism requirements

The plan identity should bind:

- Phase 6E contract version and fixed intent.
- Ratification event ID and complete ratification canonical/hash.
- Embedded recommendation-policy canonical/hash.
- Explicit hard bounds.
- Exact ordered proposal-member-to-result-experiment mapping.
- Source experiment, model, analysis, semantic, and invocation evidence.
- Source eligibility and selection results.
- Every included and excluded candidate.
- Candidate semantic and invocation canonical identities.
- Duplicate classification and matched authoritative IDs.
- Stable ordered reason codes.
- Fixed negative authority semantics.

It should exclude:

- observation time;
- database connection/session information;
- local process ID;
- CLI formatting;
- row-return order;
- generated display timestamps.

Canonical text remains authoritative; hashes remain accelerators. Same-hash/different-canonical values must remain distinct and observable.

The ADR should establish an explicit canonical-size limit and hard caps before allocating candidate collections. A limit of at most 1,000 considered sources and 1,000 plan candidates would align with the repository’s existing campaign safety boundary, though the exact value should be accepted in the ADR.

## 10. Scheduler interaction

None.

Phase 6E should not:

- add a scheduler work class;
- insert a pending experiment;
- queue a recommendation scan;
- consume capacity;
- start a worker;
- signal a running scheduler;
- poll Phase 6 tables.

A future materialized recommendation would still need to traverse the established recommendation review, ranking, Phase 4D campaign, Phase 4C conversion, and Phase 5 activation path before an experiment becomes ordinary scheduler-eligible work.

Any direct scheduler integration would require a separate ADR satisfying the full trigger, claim, capacity, retry, recovery, shutdown, and observability contract ([Volume I scheduler rules](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/Volume_I_Foundation.md:345>)).

## 11. Separation-of-duties implications

The existing Phase 6C reviewer/Phase 6D ratifier separation remains unchanged.

No additional mandatory actor separation is needed for Phase 6E because planning is read-only and records no durable decision. The planner’s identity should not enter scientific plan identity merely because someone previewed it.

A later phase that approves or materializes an exact plan must separately decide:

- whether the plan author may approve it;
- whether the Phase 6D ratifier may also approve the exact Phase 6E plan;
- whether a third distinct actor is operationally required.

The safe default is that choosing a post-ratification mutation policy must not become self-authorization. That decision should be explicit in the later action-authority ADR rather than smuggled into Phase 6E.

## 12. Risks

- **Ambiguous follow-up meaning.** Mapping result experiments into Phase 4A generation is the most repository-aligned interpretation, but it is not presently accepted behavior. ADR-0010 must make that meaning explicit.
- **Recursive research amplification.** A favorable campaign could repeatedly produce new plans. Mitigate with explicit invocation, fixed caps, no scheduler polling, and no automatic plan materialization.
- **Stale duplicate evidence.** Plans are point-in-time. Future writes must reconstruct and revalidate.
- **Canonical growth.** Phase 6D canonical payloads can already exceed two MiB; binding sources and candidates could grow materially. Define caps and size checks before construction.
- **Incomplete implementation identity.** Phase 4A documents behavior-affecting settings that cannot yet be reconstructed. Phase 6E must preserve the same fail-closed boundary and must not claim exact replay beyond the supported identity contract ([Phase 4A limitation](</Volumes/Developer SSD/ExpertAdvisor/docs/Phase4AExperimentRecommendationFoundation.rst:492>)).
- **Partial recommendation-scan persistence.** The existing generation command starts a scan before source loading and catches per-candidate persistence failures while continuing ([current generation service](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationService.cpp:323>)). It also supports only one optional source-experiment filter. It is therefore not yet an appropriate atomic Phase 6D consumer. Phase 6E should reuse its pure algorithms, not invoke its write orchestration.
- **False authority language.** Output must state that no recommendations, experiments, queues, or workers were created.
- **Governance status drift.** ADR-0009 is still marked `Proposed`, despite the Phase 6D implementation and reported independent review ([ADR index](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/adr/README.md:70>)).

## 13. Alternatives considered

| Alternative | Assessment |
|---|---|
| Direct Phase 6E activation/execution | Reject. No exact new experiment or campaign specification exists, and this would bypass Phase 4C/5 authority. |
| Copy or relaunch the previous materialization | Reject. Reusing old proposals describes old work, not a defined follow-up, and would collide with existing identities. |
| Invoke the existing recommendation scan writer directly | Defer. It lacks exact multi-source ratification provenance and permits partial per-candidate persistence. |
| Generic “follow-up authorization” row | Reject. It would repeat Phase 6D governance without defining an exact action. |
| Begin autonomous research automation | Reject for this increment. Volume X requires budgets, reservations, stop controls, recovery, and multiple new ADRs. |
| End permanently at Phase 6D | Safe but incomplete. Operators could act manually, but the action would have no typed linkage to the ratification. |
| Name this Phase 7A instead | Reasonable, but Phase 6A documentation already treats later Phase 6 steps as separately gated concerns, and Phase 6D explicitly names Phase 6E. Continuing with Phase 6E is less disruptive. |

## 14. Recommended implementation sequence

1. **Close Phase 6D governance first.** If the independent review constituted formal acceptance, change ADR-0009 from `Proposed` to `Accepted` and align the Volume VIII/XII status text. Otherwise Phase 6D remains non-authoritative under the repository’s ADR rules.

2. **Draft ADR-0010:** “Ratified follow-up planning re-enters the advisory recommendation pipeline.”

3. **Define the pure Phase 6E contract:**
   - fixed `generate_advisory_recommendations_from_result_experiments` intent;
   - ratification prerequisite;
   - embedded policy;
   - bounds;
   - canonical grammar;
   - reasons and safety flags.

4. **Add pure tests first:** golden canonical/hash, locale independence, ordering, collision behavior, bounds, malformed provenance, missing result experiments, and zero-candidate plans.

5. **Add the read-only evidence snapshot repository:** batched exact-ID queries, repeatable-read snapshot, source and duplicate validation.

6. **Build the planner by composing existing Phase 4A functions.** Do not clone candidate mutation rules or invoke the current scan writer.

7. **Add a read-only service and preview.** A CLI may follow, but it should remain an explicit standalone inspection command with no `--yes` write path.

8. **Add repository/concurrency tests:** unchanged sequences and tables, no locks, snapshot consistency, deterministic stale evidence, and bounded query behavior.

9. **Run focused regressions and the incremental Release build** after confirming scheduler safety.

10. **Defer later authorities:**
    - Phase 6F: exact plan review/approval and separation-of-duties decision.
    - Phase 6G: idempotent materialization into advisory recommendation scan/recommendation provenance.
    - Existing Phase 4A–5 workflows remain the only path toward experiments and scheduler work.

## 15. Should an ADR precede implementation?

**Yes—unambiguously.**

ADR-0010 must be accepted before Phase 6E implementation because the increment defines:

- a new semantic mapping from result experiments to recommendation sources;
- a new canonical identity;
- a new downstream ownership boundary;
- stale-evidence and replay rules;
- safety caps;
- future recommendation-materialization provenance.

Volume I explicitly requires an ADR for identity, ownership, transaction, lifecycle, conversion, interface, and security changes, and says implementation should not begin while the required ADR remains merely proposed ([Volume I governance](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/Volume_I_Foundation.md:529>)).

## Review record

- Files changed: none.
- Behavioral change: none; recommendation only.
- Tests/builds run: none, because no implementation or repository modification was requested.
- Read-only inspection: ADRs, Phase 4A–6D documentation, domain/service/repository code, migrations, tests, git history, and worktree status.
- Remaining key assumption: reviewers must explicitly accept that “follow-up” version 1 means generating advisory recommendations from each prior member’s result experiment.
- `git status --short`: clean.
- `git diff --stat`: empty.
- Reviewed branch/commit: `phase6e` at `d78771f`.
      505.13 real         2.22 user         3.69 sys
           179486720  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
              330338  page reclaims
               44954  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                 852  messages sent
               11762  messages received
                  48  signals received
                6246  voluntary context switches
              104121  involuntary context switches
         27769027393  instructions retired
         17779344575  cycles elapsed
           130107456  peak memory footprint

exec
/bin/bash -lc "rg -l -i 'campaign operations|campaign_operation|budget reservation|operational request|execution request|reservation leakage|operation lifecycle|campaign run' --glob '*.txt' --glob '*.md' --glob '*.rst' | sort && rg -n -i 'budget|reservation|operational request|execution request|campaign operations|campaign_operation|reconciliation|restart recovery' CampaignOperations_Architecture_Output.txt CampaignOperations_Architecture_Prompt.txt Phase6_Next_Architecture_Recommendation_Output.txt docs/architecture/*.md docs/architecture/adr/*.md docs/Phase*.rst | sed -n '1,700p'" in /Volumes/Developer SSD/ExpertAdvisor
 succeeded in 4874ms:
CampaignOperationsArchitectureRecommendation_CEE_Prompt.txt
CampaignOperations_Architecture_Output.txt
CampaignOperations_Architecture_Prompt.txt
CampaignOperations_Architecture_Recommendation_CEE_Output.txt
CampaignOperations_Architecture_Recommendation_CEE_Prompt.txt
Phase4D_Completion_Audit_Prompt_and_Reasoning.txt
Phase5Step5ArchitecturePlanningOutput.txt
Phase5Step5a_IndependentArchitecturalReview_Output.txt
Phase5Step5c_Implementation_Output.txt
Phase6A_FollowUpProposal_Implementation_Input.txt
Phase6A_FollowUpProposal_Implementation_Output.txt
Phase6C_OperatorReview_Implementation_Output.txt
Phase6ReadinessReview_Output.txt
Phase6_Next_Architecture_Recommendation_Output.txt
docs/architecture/Volume_X_Research_Automation.md
phase4d_step6_independent_review_prompt.txt
CampaignOperations_Architecture_Prompt.txt:1:Using High reasoning, perform a complete architectural design for the new Campaign Operations subsystem.
CampaignOperations_Architecture_Prompt.txt:29:The next architectural concern is long-lived campaign operations.
CampaignOperations_Architecture_Prompt.txt:31:Campaign Operations must remain a separate subsystem and must not transfer policy authority into the scheduler.
CampaignOperations_Architecture_Prompt.txt:35:Design the complete Campaign Operations architecture, including:
CampaignOperations_Architecture_Prompt.txt:46:• reconciliation
CampaignOperations_Architecture_Prompt.txt:47:• restart recovery
CampaignOperations_Architecture_Prompt.txt:49:• budgeting concepts
CampaignOperations_Architecture_Prompt.txt:50:• reservation concepts
CampaignOperations_Architecture_Prompt.txt:51:• execution requests
CampaignOperations_Architecture_Prompt.txt:62:• what belongs inside Campaign Operations
docs/architecture/Volume_X_Research_Automation.md:10:bounded research while preserving explicit policy, budgets, audit, and human
docs/architecture/Volume_X_Research_Automation.md:17:Future campaign goals, proposal policies, budgets, stopping conditions,
docs/architecture/Volume_X_Research_Automation.md:34:Future automation owns declared campaign policy and budget accounting only.
docs/architecture/Volume_X_Research_Automation.md:43:MUST NOT bypass review, invent authorization, exceed budgets, mutate model math,
docs/architecture/Volume_X_Research_Automation.md:50:Potential components are campaign policy, planner, budget ledger, proposal
docs/architecture/Volume_X_Research_Automation.md:67:Future campaign, policy version, budget, proposal, decision, execution request,
docs/architecture/Volume_X_Research_Automation.md:72:Every decision records exact inputs, policy canonical text/version, budgets,
docs/architecture/Volume_X_Research_Automation.md:77:Budget consumption is monotonic and auditable. Missing authority or provenance
docs/architecture/Volume_X_Research_Automation.md:88:Decision, budget reservation, and execution request require an explicit atomic
docs/architecture/Volume_X_Research_Automation.md:93:Uncertain budget or request state fails closed and requires reconciliation.
docs/architecture/Volume_X_Research_Automation.md:99:Same campaign budget, proposal, or execution request.
docs/architecture/Volume_X_Research_Automation.md:107:At most one accepted request consumes a reservation; retries return the same
docs/architecture/Volume_X_Research_Automation.md:119:Events expose policy, budget, evidence, decision, request, and stop reason.
docs/architecture/Volume_X_Research_Automation.md:123:Summaries state remaining budget, current authority, and whether any work was
docs/architecture/Volume_X_Research_Automation.md:130:Policy, budgets, stopping, selection, and deterministic planning.
docs/architecture/Volume_X_Research_Automation.md:138:Budget races, duplicate requests, pause/stop races, and scheduler isolation.
docs/architecture/Volume_X_Research_Automation.md:154:Least-privilege roles, explicit budgets, operator stop controls, and audit are
docs/architecture/Volume_X_Research_Automation.md:159:Durable campaign status, reservations, requests, outcomes, and reconciliation
docs/architecture/Volume_X_Research_Automation.md:175:Multiple ADRs are required for authority, budgets, safety, policy identity,
docs/architecture/Volume_I_Foundation.md:587:selection, and self-modifying policies require explicit budgets, safety limits,
CampaignOperations_Architecture_Output.txt:14:Using High reasoning, perform a complete architectural design for the new Campaign Operations subsystem.
CampaignOperations_Architecture_Output.txt:42:The next architectural concern is long-lived campaign operations.
CampaignOperations_Architecture_Output.txt:44:Campaign Operations must remain a separate subsystem and must not transfer policy authority into the scheduler.
CampaignOperations_Architecture_Output.txt:48:Design the complete Campaign Operations architecture, including:
CampaignOperations_Architecture_Output.txt:59:• reconciliation
CampaignOperations_Architecture_Output.txt:60:• restart recovery
CampaignOperations_Architecture_Output.txt:62:• budgeting concepts
CampaignOperations_Architecture_Output.txt:63:• reservation concepts
CampaignOperations_Architecture_Output.txt:64:• execution requests
CampaignOperations_Architecture_Output.txt:75:• what belongs inside Campaign Operations
CampaignOperations_Architecture_Output.txt:120:  • Derive Campaign Operations ownership, lifecycle, persistence, and interaction contracts
docs/architecture/Volume_XI_Scheduler.md:10:launch, lifecycle reconciliation, recovery, and scheduler observability.
docs/architecture/Volume_XI_Scheduler.md:174:diagnostics support reconciliation after scheduler or worker failure.
docs/architecture/Volume_XI_Scheduler.md:184:Distributed schedulers, remote workers, priority budgets, and research-campaign
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:94:- Exact round-trip, immutable reload, canonical/hash preservation, idempotent
docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md:15:but they do not provide a complete implementation identity, execution budget,
docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md:90:Rejected because Step 5 lacks a conversion identity, budget, lifecycle request,
docs/architecture/Volume_VII_Experiment_Lifecycle.md:171:support reconciliation without treating process presence as sole truth.
docs/architecture/Volume_VII_Experiment_Lifecycle.md:182:Recommendation conversion, experiment campaigns, budgets, and cross-machine
docs/Phase4CExperimentRecommendationConversionPersistence.rst:121:Manual CLI proposal preparation, budgets, queueing, scheduler capacity, and any
docs/Phase4CExperimentRecommendationConversion.rst:97:preparation through CLI, budgets, queueing, scheduler capacity, and any
docs/Phase4CExperimentRecommendationConversionActivation.rst:90:capacity decisions, budgets, and autonomous recommendation execution remain
docs/Phase4CExperimentRecommendationConversionExecution.rst:65:worker launch, automatic activation, budgets, and autonomous selection remain
docs/Phase4CExperimentRecommendationConversionProposalReview.rst:100:experiment, budgets, scheduler capacity, operational execution authorization,
docs/Phase4DExperimentRecommendationCampaignPlanning.rst:153:execution, automatic activation, background scans, budgets, durable
docs/architecture/Volume_VIII_Recommendation_Engine.md:442:canonical/hash preservation, exact retry, duplicate persisted identity,
docs/architecture/Volume_VIII_Recommendation_Engine.md:552:does not invoke conversion. Budgets, queueing, execution, and scheduler capacity
docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md:56:- Process supervision and database reconciliation remain operationally complex.
docs/Phase6BRecommendationCampaignFollowUpProposalPersistence.rst:82:and immutable reload, canonical/hash preservation, exact duplicate replay,
Phase6_Next_Architecture_Recommendation_Output.txt:1053:but they do not provide a complete implementation identity, execution budget,
Phase6_Next_Architecture_Recommendation_Output.txt:1128:Rejected because Step 5 lacks a conversion identity, budget, lifecycle request,
Phase6_Next_Architecture_Recommendation_Output.txt:1203:- Process supervision and database reconciliation remain operationally complex.
Phase6_Next_Architecture_Recommendation_Output.txt:1596:- Exact round-trip, immutable reload, canonical/hash preservation, idempotent
Phase6_Next_Architecture_Recommendation_Output.txt:2212:and immutable reload, canonical/hash preservation, exact duplicate replay,
Phase6_Next_Architecture_Recommendation_Output.txt:2496:but they do not provide a complete implementation identity, execution budget,
Phase6_Next_Architecture_Recommendation_Output.txt:2571:Rejected because Step 5 lacks a conversion identity, budget, lifecycle request,
Phase6_Next_Architecture_Recommendation_Output.txt:2646:- Process supervision and database reconciliation remain operationally complex.
Phase6_Next_Architecture_Recommendation_Output.txt:3399:execution, automatic activation, background scans, budgets, durable
Phase6_Next_Architecture_Recommendation_Output.txt:6381:does not invoke conversion. Budgets, queueing, execution, and scheduler capacity
Phase6_Next_Architecture_Recommendation_Output.txt:7133:A direct “activation” after ratification would be architecturally premature: the Phase 6 proposal identifies the completed campaign and favorable evidence, but it contains no new experiment specification, mutation set, budget, or target campaign membership. I’m checking whether those missing fields can be supplied by an existing authoritative recommendation/campaign workflow rather than creating a parallel execution path.
Phase6_Next_Architecture_Recommendation_Output.txt:8243:selection, and self-modifying policies require explicit budgets, safety limits,
Phase6_Next_Architecture_Recommendation_Output.txt:8263:bounded research while preserving explicit policy, budgets, audit, and human
Phase6_Next_Architecture_Recommendation_Output.txt:8270:Future campaign goals, proposal policies, budgets, stopping conditions,
Phase6_Next_Architecture_Recommendation_Output.txt:8287:Future automation owns declared campaign policy and budget accounting only.
Phase6_Next_Architecture_Recommendation_Output.txt:8296:MUST NOT bypass review, invent authorization, exceed budgets, mutate model math,
Phase6_Next_Architecture_Recommendation_Output.txt:8303:Potential components are campaign policy, planner, budget ledger, proposal
Phase6_Next_Architecture_Recommendation_Output.txt:8320:Future campaign, policy version, budget, proposal, decision, execution request,
Phase6_Next_Architecture_Recommendation_Output.txt:8325:Every decision records exact inputs, policy canonical text/version, budgets,
Phase6_Next_Architecture_Recommendation_Output.txt:8330:Budget consumption is monotonic and auditable. Missing authority or provenance
Phase6_Next_Architecture_Recommendation_Output.txt:8341:Decision, budget reservation, and execution request require an explicit atomic
Phase6_Next_Architecture_Recommendation_Output.txt:8346:Uncertain budget or request state fails closed and requires reconciliation.
Phase6_Next_Architecture_Recommendation_Output.txt:8352:Same campaign budget, proposal, or execution request.
Phase6_Next_Architecture_Recommendation_Output.txt:8360:At most one accepted request consumes a reservation; retries return the same
Phase6_Next_Architecture_Recommendation_Output.txt:8372:Events expose policy, budget, evidence, decision, request, and stop reason.
Phase6_Next_Architecture_Recommendation_Output.txt:8376:Summaries state remaining budget, current authority, and whether any work was
Phase6_Next_Architecture_Recommendation_Output.txt:8383:Policy, budgets, stopping, selection, and deterministic planning.
Phase6_Next_Architecture_Recommendation_Output.txt:8391:Budget races, duplicate requests, pause/stop races, and scheduler isolation.
Phase6_Next_Architecture_Recommendation_Output.txt:8407:Least-privilege roles, explicit budgets, operator stop controls, and audit are
Phase6_Next_Architecture_Recommendation_Output.txt:8412:Durable campaign status, reservations, requests, outcomes, and reconciliation
Phase6_Next_Architecture_Recommendation_Output.txt:8428:Multiple ADRs are required for authority, budgets, safety, policy identity,
Phase6_Next_Architecture_Recommendation_Output.txt:8452:launch, lifecycle reconciliation, recovery, and scheduler observability.
Phase6_Next_Architecture_Recommendation_Output.txt:8616:diagnostics support reconciliation after scheduler or worker failure.
Phase6_Next_Architecture_Recommendation_Output.txt:8626:Distributed schedulers, remote workers, priority budgets, and research-campaign
Phase6_Next_Architecture_Recommendation_Output.txt:10008:support reconciliation without treating process presence as sole truth.
Phase6_Next_Architecture_Recommendation_Output.txt:10019:Recommendation conversion, experiment campaigns, budgets, and cross-machine
Phase6_Next_Architecture_Recommendation_Output.txt:11628:   587	selection, and self-modifying policies require explicit budgets, safety limits,
Phase6_Next_Architecture_Recommendation_Output.txt:12010:   552	does not invoke conversion. Budgets, queueing, execution, and scheduler capacity
Phase6_Next_Architecture_Recommendation_Output.txt:12929:   552	does not invoke conversion. Budgets, queueing, execution, and scheduler capacity
Phase6_Next_Architecture_Recommendation_Output.txt:13077:   153	execution, automatic activation, background scans, budgets, durable
Phase6_Next_Architecture_Recommendation_Output.txt:13254:- budget;
Phase6_Next_Architecture_Recommendation_Output.txt:13496:| Begin autonomous research automation | Reject for this increment. Volume X requires budgets, reservations, stop controls, recovery, and multiple new ADRs. |
Phase6_Next_Architecture_Recommendation_Output.txt:13585:- budget;
Phase6_Next_Architecture_Recommendation_Output.txt:13827:| Begin autonomous research automation | Reject for this increment. Volume X requires budgets, reservations, stop controls, recovery, and multiple new ADRs. |

exec
/bin/bash -lc "wc -l CampaignOperations_Architecture_Recommendation_CEE_Output.txt CampaignOperationsArchitectureRecommendation_CEE_Prompt.txt CampaignOperations_Architecture_Recommendation_CEE_Prompt.txt && sed -n '1,140p' CampaignOperations_Architecture_Recommendation_CEE_Output.txt && tail -n 260 CampaignOperations_Architecture_Recommendation_CEE_Output.txt" in /Volumes/Developer SSD/ExpertAdvisor
 succeeded in 0ms:
    5800 CampaignOperations_Architecture_Recommendation_CEE_Output.txt
      64 CampaignOperationsArchitectureRecommendation_CEE_Prompt.txt
     124 CampaignOperations_Architecture_Recommendation_CEE_Prompt.txt
    5988 total
Reading prompt from stdin...
OpenAI Codex v0.145.0
--------
workdir: /Volumes/Developer SSD/ExpertAdvisor
model: gpt-5.6-sol
provider: openai
approval: never
sandbox: danger-full-access
reasoning effort: xhigh
reasoning summaries: none
session id: 019f89ea-3163-7a20-ae87-de125c1247e7
--------
user
Using xhigh reasoning, perform an independent Critical Engineering Evaluation of:

  CampaignOperations_Architecture_Output.txt

GOAL

Determine whether the Campaign Operations architectural recommendation should be accepted as the authoritative direction for the next architectural increment of the ExpertAdvisor/LSTM repository.

This is an independent engineering review, not an implementation exercise.

Do not implement code.
Do not modify files.
Do not create migrations.
Do not design CLI commands.
Do not begin the recommended implementation.
Do not redesign previously accepted Phases 4 through 6.
Treat completed and accepted authority chains as fixed unless the recommendation contains a concrete contradiction with them.

REVIEW BASIS

Inspect the current repository and its authoritative architecture documents, ADRs, domain contracts, repositories, services, migrations, tests, scheduler implementation, and experiment lifecycle implementation as necessary.

Evaluate the recommendation against the repository as it exists now rather than accepting its claims at face value.

In particular, independently determine whether:

- Phase 6 governance is complete;
- no meaningful Phase 6E implementation remains;
- Campaign Operations is the correct next bounded architectural concern;
- Campaign Operations should be separate from recommendation governance;
- Campaign Operations should not transfer recommendation or research-policy authority into the scheduler;
- the proposed subsystem responsibilities are coherent and complete;
- the proposed lifecycle and state model are appropriate;
- the proposed budgeting and reservation concepts have clear authority;
- the scheduler interaction contract preserves existing scheduler ownership;
- reconciliation and restart recovery have a single durable source of truth;
- transaction and concurrency boundaries are safe;
- deterministic replay and idempotence are adequately defined;
- cancellation and completion semantics are unambiguous;
- the privilege model follows least privilege;
- the proposed implementation sequence is safe and appropriately incremental.

REQUIRED REVIEW DIMENSIONS

Evaluate:

- architectural correctness;
- consistency with accepted ADRs and architecture volumes;
- compatibility with completed Phases 4 through 6;
- authority and ownership boundaries;
- domain separation;
- data ownership;
- lifecycle correctness;
- transaction boundaries;
- concurrency and lock ordering;
- idempotence and deterministic replay;
- recovery and reconciliation;
- scheduler integration;
- operator authority;
- auditability;
- security and least privilege;
- maintainability;
- extensibility;
- migration risk;
- testability;
- documentation consistency;
- hidden assumptions;
- missing decisions;
- implementation readiness.

FAIL-CLOSED REVIEW STANDARD

Do not approve the recommendation merely because its overall direction appears reasonable.

Identify any material ambiguity that could cause:

- duplicated authority;
- autonomous experiment creation;
- scheduler policy expansion;
- ambiguous campaign ownership;
- budget overcommitment;
- reservation leakage;
- duplicate operational requests;
- unsafe replay;
- irreconcilable partial state;
- cancellation races;
- inconsistent completion;
- unbounded recovery behavior;
- lock-order inversion;
- privilege escalation;
- misleading audit evidence.

Distinguish clearly between:

- blocking architectural defects;
- required revisions before implementation;
- implementation details that can safely be deferred;
- optional improvements.

OUTPUT

Produce:

1. Executive summary.
2. Repository-grounded architectural findings.
3. Strengths.
4. Blocking defects, if any.
5. Required revisions.
6. Non-blocking risks and deferred decisions.
7. Assessment of authority boundaries.
8. Assessment of scheduler interaction.
9. Assessment of lifecycle, transactions, concurrency, replay, and recovery.
10. Assessment of budgeting, reservations, and operational-request ownership.
11. Assessment of privileges, auditability, and testability.
12. Documentation and terminology findings.
13. Recommended disposition:
    - accept unchanged;
    - accept with required revisions;
    - revise and re-review;
    - reject.
14. Exact conditions that must be satisfied before implementation begins.
15. Overall readiness assessment.

Do not make repository changes.

codex
I’ll evaluate the recommendation against the repository’s actual contracts and accepted authority chains, using a fail-closed standard. I’ll keep the review strictly read-only and will distinguish architectural blockers from revisions, deferrable implementation details, and optional improvements.
- row-return order;
- generated display timestamps.

Canonical text remains authoritative; hashes remain accelerators. Same-hash/different-canonical values must remain distinct and observable.

The ADR should establish an explicit canonical-size limit and hard caps before allocating candidate collections. A limit of at most 1,000 considered sources and 1,000 plan candidates would align with the repository’s existing campaign safety boundary, though the exact value should be accepted in the ADR.

## 10. Scheduler interaction

None.

Phase 6E should not:

- add a scheduler work class;
- insert a pending experiment;
- queue a recommendation scan;
- consume capacity;
- start a worker;
- signal a running scheduler;
- poll Phase 6 tables.

A future materialized recommendation would still need to traverse the established recommendation review, ranking, Phase 4D campaign, Phase 4C conversion, and Phase 5 activation path before an experiment becomes ordinary scheduler-eligible work.

Any direct scheduler integration would require a separate ADR satisfying the full trigger, claim, capacity, retry, recovery, shutdown, and observability contract ([Volume I scheduler rules](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/Volume_I_Foundation.md:345>)).

## 11. Separation-of-duties implications

The existing Phase 6C reviewer/Phase 6D ratifier separation remains unchanged.

No additional mandatory actor separation is needed for Phase 6E because planning is read-only and records no durable decision. The planner’s identity should not enter scientific plan identity merely because someone previewed it.

A later phase that approves or materializes an exact plan must separately decide:

- whether the plan author may approve it;
- whether the Phase 6D ratifier may also approve the exact Phase 6E plan;
- whether a third distinct actor is operationally required.

The safe default is that choosing a post-ratification mutation policy must not become self-authorization. That decision should be explicit in the later action-authority ADR rather than smuggled into Phase 6E.

## 12. Risks

- **Ambiguous follow-up meaning.** Mapping result experiments into Phase 4A generation is the most repository-aligned interpretation, but it is not presently accepted behavior. ADR-0010 must make that meaning explicit.
- **Recursive research amplification.** A favorable campaign could repeatedly produce new plans. Mitigate with explicit invocation, fixed caps, no scheduler polling, and no automatic plan materialization.
- **Stale duplicate evidence.** Plans are point-in-time. Future writes must reconstruct and revalidate.
- **Canonical growth.** Phase 6D canonical payloads can already exceed two MiB; binding sources and candidates could grow materially. Define caps and size checks before construction.
- **Incomplete implementation identity.** Phase 4A documents behavior-affecting settings that cannot yet be reconstructed. Phase 6E must preserve the same fail-closed boundary and must not claim exact replay beyond the supported identity contract ([Phase 4A limitation](</Volumes/Developer SSD/ExpertAdvisor/docs/Phase4AExperimentRecommendationFoundation.rst:492>)).
- **Partial recommendation-scan persistence.** The existing generation command starts a scan before source loading and catches per-candidate persistence failures while continuing ([current generation service](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationService.cpp:323>)). It also supports only one optional source-experiment filter. It is therefore not yet an appropriate atomic Phase 6D consumer. Phase 6E should reuse its pure algorithms, not invoke its write orchestration.
- **False authority language.** Output must state that no recommendations, experiments, queues, or workers were created.
- **Governance status drift.** ADR-0009 is still marked `Proposed`, despite the Phase 6D implementation and reported independent review ([ADR index](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/adr/README.md:70>)).

## 13. Alternatives considered

| Alternative | Assessment |
|---|---|
| Direct Phase 6E activation/execution | Reject. No exact new experiment or campaign specification exists, and this would bypass Phase 4C/5 authority. |
| Copy or relaunch the previous materialization | Reject. Reusing old proposals describes old work, not a defined follow-up, and would collide with existing identities. |
| Invoke the existing recommendation scan writer directly | Defer. It lacks exact multi-source ratification provenance and permits partial per-candidate persistence. |
| Generic “follow-up authorization” row | Reject. It would repeat Phase 6D governance without defining an exact action. |
| Begin autonomous research automation | Reject for this increment. Volume X requires budgets, reservations, stop controls, recovery, and multiple new ADRs. |
| End permanently at Phase 6D | Safe but incomplete. Operators could act manually, but the action would have no typed linkage to the ratification. |
| Name this Phase 7A instead | Reasonable, but Phase 6A documentation already treats later Phase 6 steps as separately gated concerns, and Phase 6D explicitly names Phase 6E. Continuing with Phase 6E is less disruptive. |

## 14. Recommended implementation sequence

1. **Close Phase 6D governance first.** If the independent review constituted formal acceptance, change ADR-0009 from `Proposed` to `Accepted` and align the Volume VIII/XII status text. Otherwise Phase 6D remains non-authoritative under the repository’s ADR rules.

2. **Draft ADR-0010:** “Ratified follow-up planning re-enters the advisory recommendation pipeline.”

3. **Define the pure Phase 6E contract:**
   - fixed `generate_advisory_recommendations_from_result_experiments` intent;
   - ratification prerequisite;
   - embedded policy;
   - bounds;
   - canonical grammar;
   - reasons and safety flags.

4. **Add pure tests first:** golden canonical/hash, locale independence, ordering, collision behavior, bounds, malformed provenance, missing result experiments, and zero-candidate plans.

5. **Add the read-only evidence snapshot repository:** batched exact-ID queries, repeatable-read snapshot, source and duplicate validation.

6. **Build the planner by composing existing Phase 4A functions.** Do not clone candidate mutation rules or invoke the current scan writer.

7. **Add a read-only service and preview.** A CLI may follow, but it should remain an explicit standalone inspection command with no `--yes` write path.

8. **Add repository/concurrency tests:** unchanged sequences and tables, no locks, snapshot consistency, deterministic stale evidence, and bounded query behavior.

9. **Run focused regressions and the incremental Release build** after confirming scheduler safety.

10. **Defer later authorities:**
    - Phase 6F: exact plan review/approval and separation-of-duties decision.
    - Phase 6G: idempotent materialization into advisory recommendation scan/recommendation provenance.
    - Existing Phase 4A–5 workflows remain the only path toward experiments and scheduler work.

## 15. Should an ADR precede implementation?

**Yes—unambiguously.**

ADR-0010 must be accepted before Phase 6E implementation because the increment defines:

- a new semantic mapping from result experiments to recommendation sources;
- a new canonical identity;
- a new downstream ownership boundary;
- stale-evidence and replay rules;
- safety caps;
- future recommendation-materialization provenance.

Volume I explicitly requires an ADR for identity, ownership, transaction, lifecycle, conversion, interface, and security changes, and says implementation should not begin while the required ADR remains merely proposed ([Volume I governance](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/Volume_I_Foundation.md:529>)).

## Review record

- Files changed: none.
- Behavioral change: none; recommendation only.
- Tests/builds run: none, because no implementation or repository modification was requested.
- Read-only inspection: ADRs, Phase 4A–6D documentation, domain/service/repository code, migrations, tests, git history, and worktree status.
- Remaining key assumption: reviewers must explicitly accept that “follow-up” version 1 means generating advisory recommendations from each prior member’s result experiment.
- `git status --short`: clean.
- `git diff --stat`: empty.
- Reviewed branch/commit: `phase6e` at `d78771f`.
      505.13 real         2.22 user         3.69 sys
           179486720  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
              330338  page reclaims
               44954  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                 852  messages sent
               11762  messages received
                  48  signals received
                6246  voluntary context switches
              104121  involuntary context switches
         27769027393  instructions retired
         17779344575  cycles elapsed
           130107456  peak memory footprint

exec
/bin/bash -lc "rg -l -i 'campaign operations|campaign_operation|budget reservation|operational request|execution request|reservation leakage|operation lifecycle|campaign run' --glob '*.txt' --glob '*.md' --glob '*.rst' | sort && rg -n -i 'budget|reservation|operational request|execution request|campaign operations|campaign_operation|reconciliation|restart recovery' CampaignOperations_Architecture_Output.txt CampaignOperations_Architecture_Prompt.txt Phase6_Next_Architecture_Recommendation_Output.txt docs/architecture/*.md docs/architecture/adr/*.md docs/Phase*.rst | sed -n '1,700p'" in /Volumes/Developer SSD/ExpertAdvisor
 succeeded in 4874ms:
CampaignOperationsArchitectureRecommendation_CEE_Prompt.txt
CampaignOperations_Architecture_Output.txt
CampaignOperations_Architecture_Prompt.txt
CampaignOperations_Architecture_Recommendation_CEE_Output.txt
CampaignOperations_Architecture_Recommendation_CEE_Prompt.txt
Phase4D_Completion_Audit_Prompt_and_Reasoning.txt
Phase5Step5ArchitecturePlanningOutput.txt
Phase5Step5a_IndependentArchitecturalReview_Output.txt
Phase5Step5c_Implementation_Output.txt
Phase6A_FollowUpProposal_Implementation_Input.txt
Phase6A_FollowUpProposal_Implementation_Output.txt
Phase6C_OperatorReview_Implementation_Output.txt
Phase6ReadinessReview_Output.txt
Phase6_Next_Architecture_Recommendation_Output.txt
docs/architecture/Volume_X_Research_Automation.md
phase4d_step6_independent_review_prompt.txt
CampaignOperations_Architecture_Prompt.txt:1:Using High reasoning, perform a complete architectural design for the new Campaign Operations subsystem.
CampaignOperations_Architecture_Prompt.txt:29:The next architectural concern is long-lived campaign operations.
CampaignOperations_Architecture_Prompt.txt:31:Campaign Operations must remain a separate subsystem and must not transfer policy authority into the scheduler.
CampaignOperations_Architecture_Prompt.txt:35:Design the complete Campaign Operations architecture, including:
CampaignOperations_Architecture_Prompt.txt:46:• reconciliation
CampaignOperations_Architecture_Prompt.txt:47:• restart recovery
CampaignOperations_Architecture_Prompt.txt:49:• budgeting concepts
CampaignOperations_Architecture_Prompt.txt:50:• reservation concepts
CampaignOperations_Architecture_Prompt.txt:51:• execution requests
CampaignOperations_Architecture_Prompt.txt:62:• what belongs inside Campaign Operations
docs/architecture/Volume_X_Research_Automation.md:10:bounded research while preserving explicit policy, budgets, audit, and human
docs/architecture/Volume_X_Research_Automation.md:17:Future campaign goals, proposal policies, budgets, stopping conditions,
docs/architecture/Volume_X_Research_Automation.md:34:Future automation owns declared campaign policy and budget accounting only.
docs/architecture/Volume_X_Research_Automation.md:43:MUST NOT bypass review, invent authorization, exceed budgets, mutate model math,
docs/architecture/Volume_X_Research_Automation.md:50:Potential components are campaign policy, planner, budget ledger, proposal
docs/architecture/Volume_X_Research_Automation.md:67:Future campaign, policy version, budget, proposal, decision, execution request,
docs/architecture/Volume_X_Research_Automation.md:72:Every decision records exact inputs, policy canonical text/version, budgets,
docs/architecture/Volume_X_Research_Automation.md:77:Budget consumption is monotonic and auditable. Missing authority or provenance
docs/architecture/Volume_X_Research_Automation.md:88:Decision, budget reservation, and execution request require an explicit atomic
docs/architecture/Volume_X_Research_Automation.md:93:Uncertain budget or request state fails closed and requires reconciliation.
docs/architecture/Volume_X_Research_Automation.md:99:Same campaign budget, proposal, or execution request.
docs/architecture/Volume_X_Research_Automation.md:107:At most one accepted request consumes a reservation; retries return the same
docs/architecture/Volume_X_Research_Automation.md:119:Events expose policy, budget, evidence, decision, request, and stop reason.
docs/architecture/Volume_X_Research_Automation.md:123:Summaries state remaining budget, current authority, and whether any work was
docs/architecture/Volume_X_Research_Automation.md:130:Policy, budgets, stopping, selection, and deterministic planning.
docs/architecture/Volume_X_Research_Automation.md:138:Budget races, duplicate requests, pause/stop races, and scheduler isolation.
docs/architecture/Volume_X_Research_Automation.md:154:Least-privilege roles, explicit budgets, operator stop controls, and audit are
docs/architecture/Volume_X_Research_Automation.md:159:Durable campaign status, reservations, requests, outcomes, and reconciliation
docs/architecture/Volume_X_Research_Automation.md:175:Multiple ADRs are required for authority, budgets, safety, policy identity,
docs/architecture/Volume_I_Foundation.md:587:selection, and self-modifying policies require explicit budgets, safety limits,
CampaignOperations_Architecture_Output.txt:14:Using High reasoning, perform a complete architectural design for the new Campaign Operations subsystem.
CampaignOperations_Architecture_Output.txt:42:The next architectural concern is long-lived campaign operations.
CampaignOperations_Architecture_Output.txt:44:Campaign Operations must remain a separate subsystem and must not transfer policy authority into the scheduler.
CampaignOperations_Architecture_Output.txt:48:Design the complete Campaign Operations architecture, including:
CampaignOperations_Architecture_Output.txt:59:• reconciliation
CampaignOperations_Architecture_Output.txt:60:• restart recovery
CampaignOperations_Architecture_Output.txt:62:• budgeting concepts
CampaignOperations_Architecture_Output.txt:63:• reservation concepts
CampaignOperations_Architecture_Output.txt:64:• execution requests
CampaignOperations_Architecture_Output.txt:75:• what belongs inside Campaign Operations
CampaignOperations_Architecture_Output.txt:120:  • Derive Campaign Operations ownership, lifecycle, persistence, and interaction contracts
docs/architecture/Volume_XI_Scheduler.md:10:launch, lifecycle reconciliation, recovery, and scheduler observability.
docs/architecture/Volume_XI_Scheduler.md:174:diagnostics support reconciliation after scheduler or worker failure.
docs/architecture/Volume_XI_Scheduler.md:184:Distributed schedulers, remote workers, priority budgets, and research-campaign
docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md:94:- Exact round-trip, immutable reload, canonical/hash preservation, idempotent
docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md:15:but they do not provide a complete implementation identity, execution budget,
docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md:90:Rejected because Step 5 lacks a conversion identity, budget, lifecycle request,
docs/architecture/Volume_VII_Experiment_Lifecycle.md:171:support reconciliation without treating process presence as sole truth.
docs/architecture/Volume_VII_Experiment_Lifecycle.md:182:Recommendation conversion, experiment campaigns, budgets, and cross-machine
docs/Phase4CExperimentRecommendationConversionPersistence.rst:121:Manual CLI proposal preparation, budgets, queueing, scheduler capacity, and any
docs/Phase4CExperimentRecommendationConversion.rst:97:preparation through CLI, budgets, queueing, scheduler capacity, and any
docs/Phase4CExperimentRecommendationConversionActivation.rst:90:capacity decisions, budgets, and autonomous recommendation execution remain
docs/Phase4CExperimentRecommendationConversionExecution.rst:65:worker launch, automatic activation, budgets, and autonomous selection remain
docs/Phase4CExperimentRecommendationConversionProposalReview.rst:100:experiment, budgets, scheduler capacity, operational execution authorization,
docs/Phase4DExperimentRecommendationCampaignPlanning.rst:153:execution, automatic activation, background scans, budgets, durable
docs/architecture/Volume_VIII_Recommendation_Engine.md:442:canonical/hash preservation, exact retry, duplicate persisted identity,
docs/architecture/Volume_VIII_Recommendation_Engine.md:552:does not invoke conversion. Budgets, queueing, execution, and scheduler capacity
docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md:56:- Process supervision and database reconciliation remain operationally complex.
docs/Phase6BRecommendationCampaignFollowUpProposalPersistence.rst:82:and immutable reload, canonical/hash preservation, exact duplicate replay,
Phase6_Next_Architecture_Recommendation_Output.txt:1053:but they do not provide a complete implementation identity, execution budget,
Phase6_Next_Architecture_Recommendation_Output.txt:1128:Rejected because Step 5 lacks a conversion identity, budget, lifecycle request,
Phase6_Next_Architecture_Recommendation_Output.txt:1203:- Process supervision and database reconciliation remain operationally complex.
Phase6_Next_Architecture_Recommendation_Output.txt:1596:- Exact round-trip, immutable reload, canonical/hash preservation, idempotent
Phase6_Next_Architecture_Recommendation_Output.txt:2212:and immutable reload, canonical/hash preservation, exact duplicate replay,
Phase6_Next_Architecture_Recommendation_Output.txt:2496:but they do not provide a complete implementation identity, execution budget,
Phase6_Next_Architecture_Recommendation_Output.txt:2571:Rejected because Step 5 lacks a conversion identity, budget, lifecycle request,
Phase6_Next_Architecture_Recommendation_Output.txt:2646:- Process supervision and database reconciliation remain operationally complex.
Phase6_Next_Architecture_Recommendation_Output.txt:3399:execution, automatic activation, background scans, budgets, durable
Phase6_Next_Architecture_Recommendation_Output.txt:6381:does not invoke conversion. Budgets, queueing, execution, and scheduler capacity
Phase6_Next_Architecture_Recommendation_Output.txt:7133:A direct “activation” after ratification would be architecturally premature: the Phase 6 proposal identifies the completed campaign and favorable evidence, but it contains no new experiment specification, mutation set, budget, or target campaign membership. I’m checking whether those missing fields can be supplied by an existing authoritative recommendation/campaign workflow rather than creating a parallel execution path.
Phase6_Next_Architecture_Recommendation_Output.txt:8243:selection, and self-modifying policies require explicit budgets, safety limits,
Phase6_Next_Architecture_Recommendation_Output.txt:8263:bounded research while preserving explicit policy, budgets, audit, and human
Phase6_Next_Architecture_Recommendation_Output.txt:8270:Future campaign goals, proposal policies, budgets, stopping conditions,
Phase6_Next_Architecture_Recommendation_Output.txt:8287:Future automation owns declared campaign policy and budget accounting only.
Phase6_Next_Architecture_Recommendation_Output.txt:8296:MUST NOT bypass review, invent authorization, exceed budgets, mutate model math,
Phase6_Next_Architecture_Recommendation_Output.txt:8303:Potential components are campaign policy, planner, budget ledger, proposal
Phase6_Next_Architecture_Recommendation_Output.txt:8320:Future campaign, policy version, budget, proposal, decision, execution request,
Phase6_Next_Architecture_Recommendation_Output.txt:8325:Every decision records exact inputs, policy canonical text/version, budgets,
Phase6_Next_Architecture_Recommendation_Output.txt:8330:Budget consumption is monotonic and auditable. Missing authority or provenance
Phase6_Next_Architecture_Recommendation_Output.txt:8341:Decision, budget reservation, and execution request require an explicit atomic
Phase6_Next_Architecture_Recommendation_Output.txt:8346:Uncertain budget or request state fails closed and requires reconciliation.
Phase6_Next_Architecture_Recommendation_Output.txt:8352:Same campaign budget, proposal, or execution request.
Phase6_Next_Architecture_Recommendation_Output.txt:8360:At most one accepted request consumes a reservation; retries return the same
Phase6_Next_Architecture_Recommendation_Output.txt:8372:Events expose policy, budget, evidence, decision, request, and stop reason.
Phase6_Next_Architecture_Recommendation_Output.txt:8376:Summaries state remaining budget, current authority, and whether any work was
Phase6_Next_Architecture_Recommendation_Output.txt:8383:Policy, budgets, stopping, selection, and deterministic planning.
Phase6_Next_Architecture_Recommendation_Output.txt:8391:Budget races, duplicate requests, pause/stop races, and scheduler isolation.
Phase6_Next_Architecture_Recommendation_Output.txt:8407:Least-privilege roles, explicit budgets, operator stop controls, and audit are
Phase6_Next_Architecture_Recommendation_Output.txt:8412:Durable campaign status, reservations, requests, outcomes, and reconciliation
Phase6_Next_Architecture_Recommendation_Output.txt:8428:Multiple ADRs are required for authority, budgets, safety, policy identity,
Phase6_Next_Architecture_Recommendation_Output.txt:8452:launch, lifecycle reconciliation, recovery, and scheduler observability.
Phase6_Next_Architecture_Recommendation_Output.txt:8616:diagnostics support reconciliation after scheduler or worker failure.
Phase6_Next_Architecture_Recommendation_Output.txt:8626:Distributed schedulers, remote workers, priority budgets, and research-campaign
Phase6_Next_Architecture_Recommendation_Output.txt:10008:support reconciliation without treating process presence as sole truth.
Phase6_Next_Architecture_Recommendation_Output.txt:10019:Recommendation conversion, experiment campaigns, budgets, and cross-machine
Phase6_Next_Architecture_Recommendation_Output.txt:11628:   587	selection, and self-modifying policies require explicit budgets, safety limits,
Phase6_Next_Architecture_Recommendation_Output.txt:12010:   552	does not invoke conversion. Budgets, queueing, execution, and scheduler capacity
Phase6_Next_Architecture_Recommendation_Output.txt:12929:   552	does not invoke conversion. Budgets, queueing, execution, and scheduler capacity
Phase6_Next_Architecture_Recommendation_Output.txt:13077:   153	execution, automatic activation, background scans, budgets, durable
Phase6_Next_Architecture_Recommendation_Output.txt:13254:- budget;
Phase6_Next_Architecture_Recommendation_Output.txt:13496:| Begin autonomous research automation | Reject for this increment. Volume X requires budgets, reservations, stop controls, recovery, and multiple new ADRs. |
Phase6_Next_Architecture_Recommendation_Output.txt:13585:- budget;
Phase6_Next_Architecture_Recommendation_Output.txt:13827:| Begin autonomous research automation | Reject for this increment. Volume X requires budgets, reservations, stop controls, recovery, and multiple new ADRs. |


exec
/bin/bash -lc 'nl -ba docs/architecture/Volume_X_Research_Automation.md && nl -ba docs/architecture/Volume_XI_Scheduler.md && nl -ba docs/architecture/Volume_VII_Experiment_Lifecycle.md' in /Volumes/Developer SSD/ExpertAdvisor
 succeeded in 0ms:
     1	# Volume X — Research Automation
     2
     3	Status: Reserved outline; not implemented by this document
     4	Version: 0.1.0
     5	Last revised: 2026-07-15
     6
     7	## 1. Purpose
     8
     9	Define a future controlled automation layer for proposing and coordinating
    10	bounded research while preserving explicit policy, budgets, audit, and human
    11	authority.
    12
    13	## 2. Scope
    14
    15	### 2.1 In scope
    16
    17	Future campaign goals, proposal policies, budgets, stopping conditions,
    18	selection/evaluation loops, audit, and operator control.
    19
    20	### 2.2 Out of scope
    21
    22	Unbounded autonomy, self-modifying production code, implicit experiment
    23	creation, live trading, or bypassing scheduler/lifecycle ownership.
    24
    25	### 2.3 Current implementation status
    26
    27	Reserved. Existing recommendation and continuation capabilities do not form an
    28	autonomous research agent and MUST NOT be composed as one informally.
    29
    30	## 3. Responsibilities
    31
    32	### 3.1 Owned responsibilities
    33
    34	Future automation owns declared campaign policy and budget accounting only.
    35
    36	### 3.2 Dependencies
    37
    38	Consumes advisory evidence from Volumes VI/VIII/IX and requests work only
    39	through accepted Volume VII/XI interfaces.
    40
    41	### 3.3 Prohibited responsibilities
    42
    43	MUST NOT bypass review, invent authorization, exceed budgets, mutate model math,
    44	or directly control workers.
    45
    46	## 4. Architecture
    47
    48	### 4.1 Components
    49
    50	Potential components are campaign policy, planner, budget ledger, proposal
    51	adapter, decision/audit repository, operator controls, and scheduler adapter.
    52
    53	### 4.2 Control flow
    54
    55	Explicit campaign → bounded evidence snapshot → proposal → required review or
    56	policy gate → lifecycle request → scheduler execution → evidence update → stop.
    57
    58	### 4.3 Ownership boundaries
    59
    60	Automation proposes and accounts. Domain services validate. Experiment
    61	lifecycle creates. Scheduler claims and runs. Operators retain pause/stop power.
    62
    63	## 5. Data model
    64
    65	### 5.1 Authoritative entities
    66
    67	Future campaign, policy version, budget, proposal, decision, execution request,
    68	evidence snapshot, and stop reason.
    69
    70	### 5.2 Provenance and versions
    71
    72	Every decision records exact inputs, policy canonical text/version, budgets,
    73	software version, and downstream IDs.
    74
    75	### 5.3 Invariants and legacy data
    76
    77	Budget consumption is monotonic and auditable. Missing authority or provenance
    78	blocks action. Advisory status never becomes implicit authorization.
    79
    80	## 6. Transactions
    81
    82	### 6.1 Read paths
    83
    84	Planning consumes named immutable evidence snapshots.
    85
    86	### 6.2 Write paths
    87
    88	Decision, budget reservation, and execution request require an explicit atomic
    89	or compensating protocol before implementation.
    90
    91	### 6.3 Failure semantics
    92
    93	Uncertain budget or request state fails closed and requires reconciliation.
    94
    95	## 7. Concurrency
    96
    97	### 7.1 Conflict domain
    98
    99	Same campaign budget, proposal, or execution request.
   100
   101	### 7.2 Locking and serialization
   102
   103	Future ledgers use row locks/conditional updates and authoritative request IDs.
   104
   105	### 7.3 Winner, loser, and retry outcomes
   106
   107	At most one accepted request consumes a reservation; retries return the same
   108	request only when authoritative idempotency identity matches.
   109
   110	## 8. CLI
   111
   112	### 8.1 Commands and validation
   113
   114	Future start/pause/stop/status commands require explicit campaign IDs, limits,
   115	and confirmation for mutations.
   116
   117	### 8.2 Machine output
   118
   119	Events expose policy, budget, evidence, decision, request, and stop reason.
   120
   121	### 8.3 Human output
   122
   123	Summaries state remaining budget, current authority, and whether any work was
   124	actually requested or queued.
   125
   126	## 9. Testing
   127
   128	### 9.1 Pure tests
   129
   130	Policy, budgets, stopping, selection, and deterministic planning.
   131
   132	### 9.2 Persistence and migration tests
   133
   134	Ledger monotonicity, request idempotency, audit immutability, and permissions.
   135
   136	### 9.3 Concurrency and integration tests
   137
   138	Budget races, duplicate requests, pause/stop races, and scheduler isolation.
   139
   140	### 9.4 Regression boundaries
   141
   142	Recommendation, lifecycle, continuation, and scheduler semantics remain owned
   143	by their volumes.
   144
   145	## 10. Operational safety
   146
   147	### 10.1 Runtime isolation
   148
   149	Automation defaults disabled and cannot run from documentation or mere schema
   150	presence.
   151
   152	### 10.2 Permissions and destructive operations
   153
   154	Least-privilege roles, explicit budgets, operator stop controls, and audit are
   155	mandatory before deployment.
   156
   157	### 10.3 Observability and recovery
   158
   159	Durable campaign status, reservations, requests, outcomes, and reconciliation
   160	must survive process failure.
   161
   162	## 11. Future extensions
   163
   164	### 11.1 Approved extension points
   165
   166	None are executable yet; interfaces named here are architectural placeholders.
   167
   168	### 11.2 Deferred capabilities
   169
   170	Campaign planning, automated conversion, adaptive search, and autonomous
   171	experimentation.
   172
   173	### 11.3 Required decisions
   174
   175	Multiple ADRs are required for authority, budgets, safety, policy identity,
   176	conversion, scheduler capacity, and shutdown before implementation.
   177
   178	## 12. References
   179
   180	- [Volume I §17.4](Volume_I_Foundation.md)
   181	- [Volume VII](Volume_VII_Experiment_Lifecycle.md)
   182	- [Volume VIII](Volume_VIII_Recommendation_Engine.md)
   183	- [Volume XI](Volume_XI_Scheduler.md)
   184
   185	## 13. Revision history
   186
   187	| Version | Date | Change | ADR |
   188	|---|---|---|---|
   189	| 0.1.0 | 2026-07-15 | Reserved the research-automation architecture and safety gates. | — |
     1	# Volume XI — Scheduler
     2
     3	Status: Foundation outline
     4	Version: 0.1.0
     5	Last revised: 2026-07-15
     6
     7	## 1. Purpose
     8
     9	Define durable ownership of experiment work selection, capacity, process
    10	launch, lifecycle reconciliation, recovery, and scheduler observability.
    11
    12	## 2. Scope
    13
    14	### 2.1 In scope
    15
    16	Polling, eligibility, phase claims, worker slots, launch commands, attempt/PID
    17	state, completion handling, orphan recovery, operator controls, and shutdown.
    18
    19	### 2.2 Out of scope
    20
    21	Training/inference mathematics, recommendation policy/review, profitability,
    22	and research policy decisions are owned by other volumes.
    23
    24	### 2.3 Current implementation status
    25
    26	The scheduler manages experiment training, inference, analysis, checkpoint, and
    27	configured continuation workflows. Recommendation generation, scoring, and
    28	review remain explicitly outside its polling loop.
    29
    30	## 3. Responsibilities
    31
    32	### 3.1 Owned responsibilities
    33
    34	Own eligible-work polling, atomic claims, per-operation capacity, worker
    35	process launch, durable attempt state, completion transitions, and recovery.
    36
    37	### 3.2 Dependencies
    38
    39	Consumes Volume VII lifecycle state and invokes bounded workers from Volumes
    40	V/VI. PostgreSQL contracts are governed with Volume XII.
    41
    42	### 3.3 Prohibited responsibilities
    43
    44	The scheduler MUST NOT invent experiments, approve recommendations, alter
    45	model math, infer policy from scores, or treat PID presence as sole truth.
    46
    47	## 4. Architecture
    48
    49	### 4.1 Components
    50
    51	Poller, eligibility/priority rules, capacity manager, database claim service,
    52	command builder, process supervisor, completion reconciler, recovery logic,
    53	and status/diagnostic CLI.
    54
    55	### 4.2 Control flow
    56
    57	Read eligible durable state → acquire bounded claim → commit claim → launch
    58	worker outside the transaction → observe completion → persist authoritative
    59	outcome → release capacity.
    60
    61	### 4.3 Ownership boundaries
    62
    63	Lifecycle services define legal transitions; scheduler selects and claims;
    64	workers calculate and report; repositories persist. Advisory subsystems cannot
    65	enter this flow without an accepted ADR.
    66
    67	## 5. Data model
    68
    69	### 5.1 Authoritative entities
    70
    71	Experiment phase/status, worker attempt, operation, PID/start time, capacity
    72	class, exit/failure state, scheduler version, and recovery metadata.
    73
    74	### 5.2 Provenance and versions
    75
    76	Launches retain exact experiment/model IDs, operation, command/invocation
    77	metadata, scheduler/build provenance, and attempt ownership.
    78
    79	### 5.3 Invariants and legacy data
    80
    81	Status, phase, operation, worker fields, and attempt state must form valid
    82	shapes. Orphan recovery uses durable evidence plus process checks. Legacy rows
    83	are reconciled conservatively.
    84
    85	## 6. Transactions
    86
    87	### 6.1 Read paths
    88
    89	Polling may read candidates optimistically, but eligibility is rechecked in the
    90	claim transaction.
    91
    92	### 6.2 Write paths
    93
    94	Claims, start metadata, completion, recovery, and operator transitions have
    95	short explicit transactions. No transaction remains open across process launch
    96	or worker execution.
    97
    98	### 6.3 Failure semantics
    99
   100	Launch failure, worker failure, stale claim, and persistence failure remain
   101	distinct. Recovery never marks work complete without authoritative evidence.
   102
   103	## 7. Concurrency
   104
   105	### 7.1 Conflict domain
   106
   107	The same experiment phase/attempt and shared capacity class may conflict.
   108
   109	### 7.2 Locking and serialization
   110
   111	Claims use database-visible conditional state/locks. Capacity is enforced
   112	without globally serializing unrelated status inspection or completed work.
   113	Multi-row locks require stable order.
   114
   115	### 7.3 Winner, loser, and retry outcomes
   116
   117	Exactly one scheduler/attempt claims an operation. Losers observe changed state
   118	and skip deterministically. Retry and orphan recovery preserve attempt history
   119	and never duplicate completion.
   120
   121	## 8. CLI
   122
   123	### 8.1 Commands and validation
   124
   125	Scheduler start/status/recovery and operator lifecycle controls are explicit.
   126	Dry-run and once modes must accurately describe write behavior.
   127
   128	### 8.2 Machine output
   129
   130	Events identify scheduler cycle, experiment, phase, attempt, operation,
   131	capacity, PID, transition, reason, and error.
   132
   133	### 8.3 Human output
   134
   135	Summaries distinguish queued, claimed, running, recovering, failed, and
   136	completed work without overstating process observation.
   137
   138	## 9. Testing
   139
   140	### 9.1 Pure tests
   141
   142	Eligibility, capacity, priority, command construction, status mapping, and
   143	child-status interpretation.
   144
   145	### 9.2 Persistence and migration tests
   146
   147	Claim compare-and-set, state constraints, attempt metadata, and recovery.
   148
   149	### 9.3 Concurrency and integration tests
   150
   151	Multiple schedulers, capacity races, launch failure, completion races, orphan
   152	recovery, shutdown, and unrelated work concurrency.
   153
   154	### 9.4 Regression boundaries
   155
   156	Scheduler changes preserve model math, evidence calculation, recommendation
   157	status, continuation policy, and explicit operator semantics.
   158
   159	## 10. Operational safety
   160
   161	### 10.1 Runtime isolation
   162
   163	Tests do not start or signal a real scheduler without explicit authorization.
   164	Documentation changes never affect running processes.
   165
   166	### 10.2 Permissions and destructive operations
   167
   168	Scheduler roles receive only required lifecycle/attempt privileges. Force-stop
   169	and destructive recovery require explicit operator confirmation.
   170
   171	### 10.3 Observability and recovery
   172
   173	Durable state, logs, attempts, PIDs, timestamps, operations, exit codes, and
   174	diagnostics support reconciliation after scheduler or worker failure.
   175
   176	## 11. Future extensions
   177
   178	### 11.1 Approved extension points
   179
   180	Typed work classes, capacity policies, claim services, and worker adapters.
   181
   182	### 11.2 Deferred capabilities
   183
   184	Distributed schedulers, remote workers, priority budgets, and research-campaign
   185	integration.
   186
   187	### 11.3 Required decisions
   188
   189	Any new work class requires an ADR defining eligibility, claim, capacity,
   190	idempotency, recovery, operator control, and regression scope.
   191
   192	## 12. References
   193
   194	- [Volume I §§8–10](Volume_I_Foundation.md)
   195	- [ADR-0004](adr/ADR-0004-scheduler-ownership-boundaries.md)
   196	- [Volume VII](Volume_VII_Experiment_Lifecycle.md)
   197	- [Volume XII](Volume_XII_Database.md)
   198
   199	## 13. Revision history
   200
   201	| Version | Date | Change | ADR |
   202	|---|---|---|---|
   203	| 0.1.0 | 2026-07-15 | Established scheduler ownership and safety outline. | ADR-0004 |
     1	# Volume VII — Experiment Lifecycle
     2
     3	Status: Foundation outline
     4	Version: 0.1.0
     5	Last revised: 2026-07-15
     6
     7	## 1. Purpose
     8
     9	Define experiment configuration, identity, state transitions, lineage,
    10	checkpoint/continuation policy interaction, and lifecycle auditability.
    11
    12	## 2. Scope
    13
    14	### 2.1 In scope
    15
    16	Experiment creation, semantic/invocation identity, statuses/phases, attempts,
    17	models, checkpoints, continuation decisions, lineage, and terminal outcomes.
    18
    19	### 2.2 Out of scope
    20
    21	Worker computation belongs to Volumes V/VI; scheduler polling and capacity to
    22	Volume XI; advisory recommendations to Volume VIII.
    23
    24	### 2.3 Current implementation status
    25
    26	The platform implements persisted experiment lifecycle, checkpoint evaluation,
    27	continuation policy/decisions, recovery metadata, and lineage. This outline
    28	does not change those contracts.
    29
    30	## 3. Responsibilities
    31
    32	### 3.1 Owned responsibilities
    33
    34	Own experiment configuration provenance, lifecycle state machine, attempt and
    35	model linkage, explicit operator transitions, and continuation lineage.
    36
    37	### 3.2 Dependencies
    38
    39	Receives configurations from explicit queue commands and, only in a future
    40	accepted design, a recommendation conversion service. Supplies bounded work to
    41	Volume XI.
    42
    43	### 3.3 Prohibited responsibilities
    44
    45	Lifecycle code MUST NOT treat a recommendation approval as an experiment,
    46	infer configuration from incomplete provenance, or let workers self-queue.
    47
    48	## 4. Architecture
    49
    50	### 4.1 Components
    51
    52	Identity/domain rules, experiment repository, lifecycle service, continuation
    53	policy/planner, operator CLI, and scheduler-facing claim interface.
    54
    55	### 4.2 Control flow
    56
    57	Validate identity/configuration → persist explicit experiment → scheduler
    58	claims phase work → worker reports bounded outcome → lifecycle service advances
    59	or terminates → optional explicit continuation evaluation/queueing.
    60
    61	### 4.3 Ownership boundaries
    62
    63	Services validate transitions; repositories update atomically; scheduler owns
    64	operational claims; workers return facts; continuation policy cannot bypass
    65	ordinary experiment creation and capacity rules.
    66
    67	## 5. Data model
    68
    69	### 5.1 Authoritative entities
    70
    71	Experiment, invocation configuration, attempt, model, checkpoint, lineage,
    72	continuation policy, continuation decision, and operational state.
    73
    74	### 5.2 Provenance and versions
    75
    76	Row ID, semantic identity, invocation identity, lineage, implementation
    77	provenance, and mutable state remain distinct per Volume I §5.
    78
    79	### 5.3 Invariants and legacy data
    80
    81	State/phase shapes, model ownership, lineage, attempt ownership, and terminal
    82	semantics require database enforcement. Legacy records are preserved and
    83	excluded where authoritative mapping is impossible.
    84
    85	## 6. Transactions
    86
    87	### 6.1 Read paths
    88
    89	Status/lineage inspection is read-only; scheduler eligibility reads a
    90	consistent set required by its claim operation.
    91
    92	### 6.2 Write paths
    93
    94	Creation, transition plus audit metadata, claim, completion, and continuation
    95	decision/child creation each have explicit atomic boundaries.
    96
    97	### 6.3 Failure semantics
    98
    99	No transition may leave status, phase, worker ownership, model linkage, or
   100	decision history partially updated.
   101
   102	## 7. Concurrency
   103
   104	### 7.1 Conflict domain
   105
   106	Conflicts are scoped to the same experiment, attempt, semantic duplicate key,
   107	or continuation decision.
   108
   109	### 7.2 Locking and serialization
   110
   111	Use row locks, conditional updates, unique constraints, or transaction advisory
   112	locks with stable ordering; unrelated experiments remain concurrent.
   113
   114	### 7.3 Winner, loser, and retry outcomes
   115
   116	Exactly one claimant owns a phase attempt. Duplicate creation and repeated
   117	terminal transitions return defined results and do not create partial lineage.
   118
   119	## 8. CLI
   120
   121	### 8.1 Commands and validation
   122
   123	Queue, status, pause/resume/cancel/retry, checkpoint, and continuation commands
   124	are explicit and mutually exclusive where state could conflict.
   125
   126	### 8.2 Machine output
   127
   128	Events identify experiment, phase, prior/resulting status, attempt, model,
   129	reason, and whether a write occurred.
   130
   131	### 8.3 Human output
   132
   133	Summaries distinguish requested state, persisted state, running work, and
   134	future eligibility.
   135
   136	## 9. Testing
   137
   138	### 9.1 Pure tests
   139
   140	Cover identity, transition, continuation planning, inheritance, and validation.
   141
   142	### 9.2 Persistence and migration tests
   143
   144	Cover state shapes, ownership, lineage, atomic transitions, and legacy rows.
   145
   146	### 9.3 Concurrency and integration tests
   147
   148	Exercise duplicate queues, simultaneous controls, claims, orphan recovery, and
   149	continuation child creation.
   150
   151	### 9.4 Regression boundaries
   152
   153	Lifecycle changes preserve training math, inference evidence, recommendations,
   154	and scheduler capacity unless explicitly scoped.
   155
   156	## 10. Operational safety
   157
   158	### 10.1 Runtime isolation
   159
   160	Tests never target real experiments; lifecycle mutations require exact IDs and
   161	operator authorization.
   162
   163	### 10.2 Permissions and destructive operations
   164
   165	Runtime roles receive transition-specific access; history and lineage are not
   166	silently deleted.
   167
   168	### 10.3 Observability and recovery
   169
   170	Durable attempts, PIDs, operation names, timestamps, exit status, and reasons
   171	support reconciliation without treating process presence as sole truth.
   172
   173	## 11. Future extensions
   174
   175	### 11.1 Approved extension points
   176
   177	Versioned configuration contracts, explicit lifecycle services, and scheduler
   178	claim interfaces.
   179
   180	### 11.2 Deferred capabilities
   181
   182	Recommendation conversion, experiment campaigns, budgets, and cross-machine
   183	attempt execution.
   184
   185	### 11.3 Required decisions
   186
   187	Each needs identity, authorization, capacity, idempotency, audit, and rollback
   188	ADRs before implementation.
   189
   190	## 12. References
   191
   192	- [Volume I §§5–10](Volume_I_Foundation.md)
   193	- [ADR-0002](adr/ADR-0002-deterministic-experiment-identity.md)
   194	- [Volume VIII](Volume_VIII_Recommendation_Engine.md)
   195	- [Volume XI](Volume_XI_Scheduler.md)
   196
   197	## 13. Revision history
   198
   199	| Version | Date | Change | ADR |
   200	|---|---|---|---|
   201	| 0.1.0 | 2026-07-15 | Established the experiment-lifecycle outline. | — |

exec
/bin/bash -lc "wc -l Sources/ExperimentScheduler.cpp Headers/ExperimentScheduler.hpp Database/migrations/005_experiment_scheduler.sql LSTM/main.cpp Sources/ExperimentRecommendationCampaign*.cpp Sources/ExperimentRecommendationCampaign*.hpp | sort -n | tail -n 50 && rg -n 'scheduler|Scheduler|SELECT.*experiment|FOR UPDATE|SKIP LOCKED|status|phase|pid|worker|capacity|cancel|pause|resume|recover|orphan|retry|continuation|claim|launch|fork|posix_spawn' Sources/ExperimentScheduler.cpp Headers/ExperimentScheduler.hpp Database/migrations/005_experiment_scheduler.sql | sed -n '1,1000p'" in /Volumes/Developer SSD/ExpertAdvisor
 succeeded in 0ms:
     139 Sources/ExperimentRecommendationCampaignProposalReviewRepository.cpp
     158 Sources/ExperimentRecommendationCampaignHandoff.hpp
     162 Sources/ExperimentRecommendationCampaignLaunchRepository.cpp
     163 Sources/ExperimentRecommendationCampaignExecutionRepository.cpp
     165 Sources/ExperimentRecommendationCampaignFollowUpProposalRatification.hpp
     167 Sources/ExperimentRecommendationCampaignFollowUpProposalPreview.cpp
     171 Sources/ExperimentRecommendationCampaignExecution.cpp
     179 Sources/ExperimentRecommendationCampaignFollowUpProposalReviewPresentation.cpp
     181 Sources/ExperimentRecommendationCampaignActivation.cpp
     209 Sources/ExperimentRecommendationCampaignPlanningService.cpp
     210 Sources/ExperimentRecommendationCampaignHandoffService.cpp
     213 Sources/ExperimentRecommendationCampaignPlanning.hpp
     228 Sources/ExperimentRecommendationCampaignExecutionService.cpp
     228 Sources/ExperimentRecommendationCampaignReviewService.cpp
     241 Sources/ExperimentRecommendationCampaignActivationRepository.cpp
     246 Sources/ExperimentRecommendationCampaignProposalReview.cpp
     257 Sources/ExperimentRecommendationCampaignPlanningRepository.cpp
     259 Sources/ExperimentRecommendationCampaignStatus.hpp
     261 Sources/ExperimentRecommendationCampaignActivationService.cpp
     264 Sources/ExperimentRecommendationCampaignStatusService.cpp
     272 Sources/ExperimentRecommendationCampaignHandoff.cpp
     272 Sources/ExperimentRecommendationCampaignMaterialization.cpp
     273 Sources/ExperimentRecommendationCampaignApprovalService.cpp
     281 Sources/ExperimentRecommendationCampaignFollowUpProposal.hpp
     288 Sources/ExperimentRecommendationCampaignFollowUpProposalReviewRepository.cpp
     307 Sources/ExperimentRecommendationCampaignProposalReviewService.cpp
     308 Sources/ExperimentRecommendationCampaignFollowUpProposalReview.cpp
     329 Sources/ExperimentRecommendationCampaignLaunchService.cpp
     331 Sources/ExperimentRecommendationCampaignApprovalRepository.cpp
     338 Sources/ExperimentRecommendationCampaignLaunch.cpp
     341 Sources/ExperimentRecommendationCampaignReview.cpp
     346 Sources/ExperimentRecommendationCampaignApproval.cpp
     356 Sources/ExperimentRecommendationCampaignFollowUpProposalRatificationRepository.cpp
     364 Sources/ExperimentRecommendationCampaignMaterializationService.cpp
     451 Sources/ExperimentRecommendationCampaignOutcomePolicy.hpp
     486 Sources/ExperimentRecommendationCampaignFollowUpProposalRatification.cpp
     529 Sources/ExperimentRecommendationCampaignFollowUpProposalRepository.cpp
     529 Sources/ExperimentRecommendationCampaignOutcomeAssessmentService.cpp
     531 Sources/ExperimentRecommendationCampaignOutcomeAssessment.hpp
     540 Sources/ExperimentRecommendationCampaignFollowUpProposal.cpp
     547 Sources/ExperimentRecommendationCampaignStatusRepository.cpp
     563 Sources/ExperimentRecommendationCampaignStatus.cpp
     596 Sources/ExperimentRecommendationCampaignMaterializationRepository.cpp
     907 Sources/ExperimentRecommendationCampaignOutcomeAssessmentRepository.cpp
     943 Sources/ExperimentRecommendationCampaignPlanning.cpp
    1314 Sources/ExperimentRecommendationCampaignOutcomePolicy.cpp
    1416 Sources/ExperimentRecommendationCampaignOutcomeAssessment.cpp
    7382 LSTM/main.cpp
   17408 Sources/ExperimentScheduler.cpp
   44799 total
Headers/ExperimentScheduler.hpp:3:namespace EA::ExperimentScheduler
Headers/ExperimentScheduler.hpp:6:bool IsExperimentSchedulerCommand(int argc, const char* argv[]);
Headers/ExperimentScheduler.hpp:7:int RunExperimentSchedulerCli(int argc, const char* argv[]);
Headers/ExperimentScheduler.hpp:9:} // namespace EA::ExperimentScheduler
Database/migrations/005_experiment_scheduler.sql:14:    status text NOT NULL DEFAULT 'pending'
Database/migrations/005_experiment_scheduler.sql:15:        CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
Database/migrations/005_experiment_scheduler.sql:16:    phase text NOT NULL DEFAULT 'train'
Database/migrations/005_experiment_scheduler.sql:17:        CHECK (phase IN ('train', 'infer', 'analyze', 'done')),
Database/migrations/005_experiment_scheduler.sql:19:    resume_model_id bigint,
Database/migrations/005_experiment_scheduler.sql:32:CREATE INDEX IF NOT EXISTS experiment_status_phase_idx
Database/migrations/005_experiment_scheduler.sql:33:    ON experiment (status, phase);
Database/migrations/005_experiment_scheduler.sql:54:        COALESCE(resume_model_id, -1),
Database/migrations/005_experiment_scheduler.sql:57:    WHERE status <> 'cancelled';
Database/migrations/005_experiment_scheduler.sql:59:GRANT SELECT, INSERT, UPDATE, DELETE ON experiment TO pqxx;
Database/migrations/005_experiment_scheduler.sql:60:GRANT USAGE, SELECT ON SEQUENCE experiment_experiment_id_seq TO pqxx;
Sources/ExperimentScheduler.cpp:32:#include "ExperimentScheduler.hpp"
Sources/ExperimentScheduler.cpp:58:#include "SchedulerChildStatus.hpp"
Sources/ExperimentScheduler.cpp:62:namespace EA::ExperimentScheduler
Sources/ExperimentScheduler.cpp:67:struct SchedulerOptions
Sources/ExperimentScheduler.cpp:81:    bool schedulerStatus = false;
Sources/ExperimentScheduler.cpp:90:    std::optional<long long> statusExperimentId;
Sources/ExperimentScheduler.cpp:93:    std::optional<long long> pauseExperimentId;
Sources/ExperimentScheduler.cpp:94:    std::optional<long long> resumeExperimentId;
Sources/ExperimentScheduler.cpp:95:    std::optional<long long> cancelExperimentId;
Sources/ExperimentScheduler.cpp:96:    std::optional<long long> retryFailedExperimentId;
Sources/ExperimentScheduler.cpp:97:    std::optional<long long> retryCheckpointEvalId;
Sources/ExperimentScheduler.cpp:104:    std::optional<long long> continuationStatusExperimentId;
Sources/ExperimentScheduler.cpp:235:    std::optional<long long> launchRecommendationCampaignMaterializationId;
Sources/ExperimentScheduler.cpp:276:    bool schedulerVerbose = false;
Sources/ExperimentScheduler.cpp:277:    bool schedulerOnce = false;
Sources/ExperimentScheduler.cpp:278:    bool recoverOrphansOnly = false;
Sources/ExperimentScheduler.cpp:281:    bool continuationDryRun = false;
Sources/ExperimentScheduler.cpp:282:    int continuationScanSeconds = 300;
Sources/ExperimentScheduler.cpp:283:    int continuationMaxQueuesPerScan = 1;
Sources/ExperimentScheduler.cpp:284:    bool continuationScanSecondsSpecified = false;
Sources/ExperimentScheduler.cpp:285:    bool continuationMaxQueuesPerScanSpecified = false;
Sources/ExperimentScheduler.cpp:289:    int schedulerPollSeconds = 30;
Sources/ExperimentScheduler.cpp:290:    std::string schedulerLogDir = "experiment_logs";
Sources/ExperimentScheduler.cpp:308:    std::optional<long long> resumeModelId;
Sources/ExperimentScheduler.cpp:364:    std::optional<long long> resumeModelId;
Sources/ExperimentScheduler.cpp:373:    bool launched = false;
Sources/ExperimentScheduler.cpp:379:    pid_t pid = -1;
Sources/ExperimentScheduler.cpp:383:struct SchedulerOwnedChild
Sources/ExperimentScheduler.cpp:385:    pid_t pid = -1;
Sources/ExperimentScheduler.cpp:387:    std::string phase;
Sources/ExperimentScheduler.cpp:393:    std::string launchedAt;
Sources/ExperimentScheduler.cpp:394:    double launchedEpoch = 0.0;
Sources/ExperimentScheduler.cpp:399:std::map<pid_t, SchedulerOwnedChild> gSchedulerOwnedChildren;
Sources/ExperimentScheduler.cpp:407:    std::string status;
Sources/ExperimentScheduler.cpp:408:    std::string phase;
Sources/ExperimentScheduler.cpp:409:    std::optional<int> workerPid;
Sources/ExperimentScheduler.cpp:418:    std::string phase;
Sources/ExperimentScheduler.cpp:419:    std::optional<int> workerPid;
Sources/ExperimentScheduler.cpp:433:struct SchedulerStatusJob
Sources/ExperimentScheduler.cpp:438:    std::string phase;
Sources/ExperimentScheduler.cpp:439:    std::string status;
Sources/ExperimentScheduler.cpp:473:    std::optional<int> pid;
Sources/ExperimentScheduler.cpp:488:struct SchedulerProcessResource
Sources/ExperimentScheduler.cpp:490:    int pid = -1;
Sources/ExperimentScheduler.cpp:496:struct SchedulerProcessInfo
Sources/ExperimentScheduler.cpp:498:    int pid = -1;
Sources/ExperimentScheduler.cpp:500:    SchedulerProcessResource resource;
Sources/ExperimentScheduler.cpp:503:struct SchedulerResourceAggregate
Sources/ExperimentScheduler.cpp:505:    int workers = 0;
Sources/ExperimentScheduler.cpp:511:struct SchedulerStatusCounts
Sources/ExperimentScheduler.cpp:514:    int paused = 0;
Sources/ExperimentScheduler.cpp:518:    int cancelled = 0;
Sources/ExperimentScheduler.cpp:521:struct SchedulerControlExperimentRow
Sources/ExperimentScheduler.cpp:524:    std::string status;
Sources/ExperimentScheduler.cpp:525:    std::string phase;
Sources/ExperimentScheduler.cpp:527:    std::optional<long long> resumeModelId;
Sources/ExperimentScheduler.cpp:532:struct SchedulerStopExperiment
Sources/ExperimentScheduler.cpp:535:    std::string status;
Sources/ExperimentScheduler.cpp:536:    std::string phase;
Sources/ExperimentScheduler.cpp:539:struct SchedulerStopCandidate
Sources/ExperimentScheduler.cpp:541:    SchedulerStopExperiment experiment;
Sources/ExperimentScheduler.cpp:542:    std::optional<int> pid;
Sources/ExperimentScheduler.cpp:546:struct SchedulerDetectedWorker
Sources/ExperimentScheduler.cpp:548:    int pid = -1;
Sources/ExperimentScheduler.cpp:551:    SchedulerProcessResource resource;
Sources/ExperimentScheduler.cpp:554:struct SchedulerUnmanagedWorker
Sources/ExperimentScheduler.cpp:556:    int pid = -1;
Sources/ExperimentScheduler.cpp:562:struct SchedulerStatusProcessSnapshot
Sources/ExperimentScheduler.cpp:565:    std::vector<SchedulerProcessInfo> processes;
Sources/ExperimentScheduler.cpp:566:    std::vector<int> schedulerPids;
Sources/ExperimentScheduler.cpp:567:    std::map<int, SchedulerProcessResource> resourcesByPid;
Sources/ExperimentScheduler.cpp:571:    std::vector<SchedulerDetectedWorker> workerProcesses;
Sources/ExperimentScheduler.cpp:572:    SchedulerResourceAggregate schedulerResources;
Sources/ExperimentScheduler.cpp:573:    SchedulerResourceAggregate trainResources;
Sources/ExperimentScheduler.cpp:574:    SchedulerResourceAggregate inferResources;
Sources/ExperimentScheduler.cpp:575:    SchedulerResourceAggregate analysisResources;
Sources/ExperimentScheduler.cpp:582:    std::optional<int> schedulerPollSeconds;
Sources/ExperimentScheduler.cpp:585:    std::optional<bool> continuationDryRun;
Sources/ExperimentScheduler.cpp:586:    std::optional<int> continuationScanSeconds;
Sources/ExperimentScheduler.cpp:587:    std::optional<int> continuationMaxQueuesPerScan;
Sources/ExperimentScheduler.cpp:593:struct SchedulerWorkerAccounting
Sources/ExperimentScheduler.cpp:601:    SchedulerResourceAggregate managedTrainResources;
Sources/ExperimentScheduler.cpp:602:    SchedulerResourceAggregate managedInferResources;
Sources/ExperimentScheduler.cpp:603:    SchedulerResourceAggregate managedAnalysisResources;
Sources/ExperimentScheduler.cpp:604:    SchedulerResourceAggregate unmanagedTrainResources;
Sources/ExperimentScheduler.cpp:605:    SchedulerResourceAggregate unmanagedInferResources;
Sources/ExperimentScheduler.cpp:606:    SchedulerResourceAggregate unmanagedAnalysisResources;
Sources/ExperimentScheduler.cpp:607:    std::vector<SchedulerUnmanagedWorker> unmanagedWorkers;
Sources/ExperimentScheduler.cpp:610:struct SchedulerIntelligenceRecord
Sources/ExperimentScheduler.cpp:623:struct SchedulerDominatedRecord
Sources/ExperimentScheduler.cpp:625:    SchedulerIntelligenceRecord dominated;
Sources/ExperimentScheduler.cpp:631:struct SchedulerIntelligenceSnapshot
Sources/ExperimentScheduler.cpp:633:    std::optional<SchedulerIntelligenceRecord> overallLeader;
Sources/ExperimentScheduler.cpp:634:    std::optional<SchedulerIntelligenceRecord> recentBest24h;
Sources/ExperimentScheduler.cpp:635:    std::vector<SchedulerIntelligenceRecord> leadersBySymbol;
Sources/ExperimentScheduler.cpp:636:    std::vector<SchedulerIntelligenceRecord> leadersByHorizon;
Sources/ExperimentScheduler.cpp:637:    std::vector<SchedulerIntelligenceRecord> top5;
Sources/ExperimentScheduler.cpp:638:    std::vector<SchedulerIntelligenceRecord> worst5;
Sources/ExperimentScheduler.cpp:639:    std::vector<SchedulerDominatedRecord> dominated;
Sources/ExperimentScheduler.cpp:685:    std::string phase;
Sources/ExperimentScheduler.cpp:688:    int launched = 0;
Sources/ExperimentScheduler.cpp:692:struct SchedulerEventLogState
Sources/ExperimentScheduler.cpp:740:    std::string status;
Sources/ExperimentScheduler.cpp:741:    std::string phase;
Sources/ExperimentScheduler.cpp:761:bool IsExperimentSchedulerCommandImpl(int argc, const char* argv[])
Sources/ExperimentScheduler.cpp:768:            arg == "--status" ||
Sources/ExperimentScheduler.cpp:775:            arg == "--scheduler-status" ||
Sources/ExperimentScheduler.cpp:785:            arg == "--pause-experiment" ||
Sources/ExperimentScheduler.cpp:786:            arg == "--resume-experiment" ||
Sources/ExperimentScheduler.cpp:787:            arg == "--cancel-experiment" ||
Sources/ExperimentScheduler.cpp:788:            arg == "--retry-failed-experiment" ||
Sources/ExperimentScheduler.cpp:789:            arg == "--retry-checkpoint-eval" ||
Sources/ExperimentScheduler.cpp:791:            arg == "--enable-continuation-policy" ||
Sources/ExperimentScheduler.cpp:792:            arg == "--disable-continuation-policy" ||
Sources/ExperimentScheduler.cpp:793:            arg == "--set-continuation-policy" ||
Sources/ExperimentScheduler.cpp:794:            arg == "--evaluate-continuation" ||
Sources/ExperimentScheduler.cpp:795:            arg == "--queue-continuation" ||
Sources/ExperimentScheduler.cpp:796:            arg == "--continuation-status" ||
Sources/ExperimentScheduler.cpp:799:            arg == "--recommendation-status" ||
Sources/ExperimentScheduler.cpp:801:            arg == "--recommendation-scan-status" ||
Sources/ExperimentScheduler.cpp:807:            arg == "--recommendation-status-filter" ||
Sources/ExperimentScheduler.cpp:812:            arg == "--recommendation-score-status" ||
Sources/ExperimentScheduler.cpp:814:            arg == "--recommendation-score-run-status" ||
Sources/ExperimentScheduler.cpp:825:            arg == "--recommendation-review-status" ||
Sources/ExperimentScheduler.cpp:837:            arg == "--recommendation-evaluation-status" ||
Sources/ExperimentScheduler.cpp:840:            arg == "--recommendation-evaluation-run-status" ||
Sources/ExperimentScheduler.cpp:855:            arg == "--recommendation-ranking-status" ||
Sources/ExperimentScheduler.cpp:857:            arg == "--recommendation-ranking-member-status" ||
Sources/ExperimentScheduler.cpp:868:            arg == "--list-conversion-proposals-by-review-status" ||
Sources/ExperimentScheduler.cpp:871:            arg == "--conversion-proposal-execution-status" ||
Sources/ExperimentScheduler.cpp:873:            arg == "--recommendation-conversion-activation-status" ||
Sources/ExperimentScheduler.cpp:905:            arg == "--launch-recommendation-campaign-materialization" ||
Sources/ExperimentScheduler.cpp:906:            arg == "--recommendation-campaign-status" ||
Sources/ExperimentScheduler.cpp:922:            arg == "--campaign-reconsider-cancelled" ||
Sources/ExperimentScheduler.cpp:923:            arg == "--auto-evaluate-continuations" ||
Sources/ExperimentScheduler.cpp:924:            arg == "--auto-queue-continuations" ||
Sources/ExperimentScheduler.cpp:925:            arg == "--continuation-scan-seconds" ||
Sources/ExperimentScheduler.cpp:926:            arg == "--continuation-max-queues-per-scan" ||
Sources/ExperimentScheduler.cpp:927:            arg == "--continuation-dry-run" ||
Sources/ExperimentScheduler.cpp:947:            arg.rfind("--pause-experiment=", 0) == 0 ||
Sources/ExperimentScheduler.cpp:948:            arg.rfind("--resume-experiment=", 0) == 0 ||
Sources/ExperimentScheduler.cpp:949:            arg.rfind("--cancel-experiment=", 0) == 0 ||
Sources/ExperimentScheduler.cpp:950:            arg.rfind("--retry-failed-experiment=", 0) == 0 ||
Sources/ExperimentScheduler.cpp:951:            arg.rfind("--retry-checkpoint-eval=", 0) == 0 ||
Sources/ExperimentScheduler.cpp:953:            arg.rfind("--enable-continuation-policy=", 0) == 0 ||
Sources/ExperimentScheduler.cpp:954:            arg.rfind("--disable-continuation-policy=", 0) == 0 ||
Sources/ExperimentScheduler.cpp:955:            arg.rfind("--set-continuation-policy=", 0) == 0 ||
Sources/ExperimentScheduler.cpp:956:            arg.rfind("--evaluate-continuation=", 0) == 0 ||
Sources/ExperimentScheduler.cpp:957:            arg.rfind("--queue-continuation=", 0) == 0 ||
Sources/ExperimentScheduler.cpp:958:            arg.rfind("--continuation-status=", 0) == 0 ||
Sources/ExperimentScheduler.cpp:959:            arg.rfind("--recommendation-status=", 0) == 0 ||
Sources/ExperimentScheduler.cpp:960:            arg.rfind("--recommendation-scan-status=", 0) == 0 ||
Sources/ExperimentScheduler.cpp:966:            arg.rfind("--recommendation-status-filter=", 0) == 0 ||
Sources/ExperimentScheduler.cpp:969:            arg.rfind("--recommendation-score-status=", 0) == 0 ||
Sources/ExperimentScheduler.cpp:970:            arg.rfind("--recommendation-score-run-status=", 0) == 0 ||
Sources/ExperimentScheduler.cpp:980:            arg.rfind("--recommendation-review-status=", 0) == 0 ||
Sources/ExperimentScheduler.cpp:990:            arg.rfind("--recommendation-evaluation-status=", 0) == 0 ||
Sources/ExperimentScheduler.cpp:992:            arg.rfind("--recommendation-evaluation-run-status=", 0) == 0 ||
Sources/ExperimentScheduler.cpp:1002:            arg.rfind("--recommendation-ranking-status=", 0) == 0 ||
Sources/ExperimentScheduler.cpp:1004:            arg.rfind("--recommendation-ranking-member-status=", 0) == 0 ||
Sources/ExperimentScheduler.cpp:1015:            arg.rfind("--list-conversion-proposals-by-review-status=", 0) == 0 ||
Sources/ExperimentScheduler.cpp:1018:            arg.rfind("--conversion-proposal-execution-status=", 0) == 0 ||
Sources/ExperimentScheduler.cpp:1022:                "--recommendation-conversion-activation-status=", 0) == 0 ||
Sources/ExperimentScheduler.cpp:1061:                "--launch-recommendation-campaign-materialization=", 0) == 0 ||
Sources/ExperimentScheduler.cpp:1062:            arg.rfind("--recommendation-campaign-status=", 0) == 0 ||
Sources/ExperimentScheduler.cpp:1065:            arg.rfind("--continuation-scan-seconds=", 0) == 0 ||
Sources/ExperimentScheduler.cpp:1066:            arg.rfind("--continuation-max-queues-per-scan=", 0) == 0 ||
Sources/ExperimentScheduler.cpp:1131:    const pid_t pid = ::fork();
Sources/ExperimentScheduler.cpp:1132:    if (pid < 0)
Sources/ExperimentScheduler.cpp:1133:        throw std::runtime_error("fork failed for " + args.front() + ": errno=" + std::to_string(errno));
Sources/ExperimentScheduler.cpp:1135:    if (pid == 0)
Sources/ExperimentScheduler.cpp:1141:    int status = 0;
Sources/ExperimentScheduler.cpp:1142:    while (::waitpid(pid, &status, 0) < 0)
Sources/ExperimentScheduler.cpp:1146:        throw std::runtime_error("waitpid failed for " + args.front() + ": errno=" + std::to_string(errno));
Sources/ExperimentScheduler.cpp:1149:    if (WIFEXITED(status))
Sources/ExperimentScheduler.cpp:1150:        return WEXITSTATUS(status);
Sources/ExperimentScheduler.cpp:1151:    if (WIFSIGNALED(status))
Sources/ExperimentScheduler.cpp:1152:        return 128 + WTERMSIG(status);
Sources/ExperimentScheduler.cpp:1466:void ValidateCheckpointPolicyConfig(const SchedulerOptions& options)
Sources/ExperimentScheduler.cpp:1480:SchedulerOptions ParseSchedulerArgs(int argc, const char* argv[])
Sources/ExperimentScheduler.cpp:1482:    SchedulerOptions options;
Sources/ExperimentScheduler.cpp:1492:        else if (arg == "--status")
Sources/ExperimentScheduler.cpp:1502:        else if (arg == "--auto-resume")
Sources/ExperimentScheduler.cpp:1510:        else if (arg == "--scheduler-status")
Sources/ExperimentScheduler.cpp:1511:            options.schedulerStatus = true;
Sources/ExperimentScheduler.cpp:1533:            options.statusExperimentId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
Sources/ExperimentScheduler.cpp:1542:        else if (arg == "--scheduler-verbose")
Sources/ExperimentScheduler.cpp:1543:            options.schedulerVerbose = true;
Sources/ExperimentScheduler.cpp:1548:        else if (arg == "--scheduler-once")
Sources/ExperimentScheduler.cpp:1549:            options.schedulerOnce = true;
Sources/ExperimentScheduler.cpp:1550:        else if (arg == "--recover-orphans-only")
Sources/ExperimentScheduler.cpp:1551:            options.recoverOrphansOnly = true;
Sources/ExperimentScheduler.cpp:1552:        else if (arg == "--auto-evaluate-continuations")
Sources/ExperimentScheduler.cpp:1554:        else if (arg == "--auto-queue-continuations")
Sources/ExperimentScheduler.cpp:1556:        else if (arg == "--continuation-dry-run")
Sources/ExperimentScheduler.cpp:1557:            options.continuationDryRun = true;
Sources/ExperimentScheduler.cpp:1558:        else if (arg == "--continuation-scan-seconds")
Sources/ExperimentScheduler.cpp:1560:            options.continuationScanSeconds =
Sources/ExperimentScheduler.cpp:1562:            options.continuationScanSecondsSpecified = true;
Sources/ExperimentScheduler.cpp:1564:        else if (arg == "--continuation-max-queues-per-scan")
Sources/ExperimentScheduler.cpp:1566:            options.continuationMaxQueuesPerScan =
Sources/ExperimentScheduler.cpp:1568:            options.continuationMaxQueuesPerScanSpecified = true;
Sources/ExperimentScheduler.cpp:1578:        else if (arg == "--pause-experiment")
Sources/ExperimentScheduler.cpp:1579:            options.pauseExperimentId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
Sources/ExperimentScheduler.cpp:1580:        else if (arg == "--resume-experiment")
Sources/ExperimentScheduler.cpp:1581:            options.resumeExperimentId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
Sources/ExperimentScheduler.cpp:1582:        else if (arg == "--cancel-experiment")
Sources/ExperimentScheduler.cpp:1583:            options.cancelExperimentId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
Sources/ExperimentScheduler.cpp:1584:        else if (arg == "--retry-failed-experiment")
Sources/ExperimentScheduler.cpp:1585:            options.retryFailedExperimentId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
Sources/ExperimentScheduler.cpp:1586:        else if (arg == "--retry-checkpoint-eval")
Sources/ExperimentScheduler.cpp:1587:            options.retryCheckpointEvalId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
Sources/ExperimentScheduler.cpp:1590:        else if (arg == "--enable-continuation-policy")
Sources/ExperimentScheduler.cpp:1592:        else if (arg == "--disable-continuation-policy")
Sources/ExperimentScheduler.cpp:1594:        else if (arg == "--set-continuation-policy")
Sources/ExperimentScheduler.cpp:1596:        else if (arg == "--evaluate-continuation")
Sources/ExperimentScheduler.cpp:1598:        else if (arg == "--queue-continuation")
Sources/ExperimentScheduler.cpp:1600:        else if (arg == "--continuation-status")
Sources/ExperimentScheduler.cpp:1601:            options.continuationStatusExperimentId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
Sources/ExperimentScheduler.cpp:1606:        else if (arg == "--recommendation-status")
Sources/ExperimentScheduler.cpp:1610:        else if (arg == "--recommendation-scan-status")
Sources/ExperimentScheduler.cpp:1622:        else if (arg == "--recommendation-status-filter")
Sources/ExperimentScheduler.cpp:1635:        else if (arg == "--recommendation-score-status")
Sources/ExperimentScheduler.cpp:1639:        else if (arg == "--recommendation-score-run-status")
Sources/ExperimentScheduler.cpp:1667:        else if (arg == "--recommendation-review-status")
Sources/ExperimentScheduler.cpp:1702:        else if (arg == "--recommendation-evaluation-status")
Sources/ExperimentScheduler.cpp:1710:        else if (arg == "--recommendation-evaluation-run-status")
Sources/ExperimentScheduler.cpp:1756:        else if (arg == "--recommendation-ranking-status")
Sources/ExperimentScheduler.cpp:1762:        else if (arg == "--recommendation-ranking-member-status")
Sources/ExperimentScheduler.cpp:1795:        else if (arg == "--list-conversion-proposals-by-review-status")
Sources/ExperimentScheduler.cpp:1807:        else if (arg == "--conversion-proposal-execution-status")
Sources/ExperimentScheduler.cpp:1814:        else if (arg == "--recommendation-conversion-activation-status")
Sources/ExperimentScheduler.cpp:1986:        else if (arg == "--launch-recommendation-campaign-materialization")
Sources/ExperimentScheduler.cpp:1990:                    "duplicate --launch-recommendation-campaign-materialization");
Sources/ExperimentScheduler.cpp:1992:            options.launchRecommendationCampaignMaterializationId =
Sources/ExperimentScheduler.cpp:1996:        else if (arg == "--recommendation-campaign-status")
Sources/ExperimentScheduler.cpp:2000:                    "duplicate --recommendation-campaign-status");
Sources/ExperimentScheduler.cpp:2101:        else if (arg == "--campaign-reconsider-cancelled")
Sources/ExperimentScheduler.cpp:2142:        else if (arg == "--continuation-candidate-excluded")
Sources/ExperimentScheduler.cpp:2192:        else if (arg == "--resume-model-id")
Sources/ExperimentScheduler.cpp:2193:            options.resumeModelId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
Sources/ExperimentScheduler.cpp:2200:        else if (arg == "--scheduler-poll-seconds")
Sources/ExperimentScheduler.cpp:2201:            options.schedulerPollSeconds = ParsePositiveInt(arg, RequireNextArg(argc, argv, i, arg));
Sources/ExperimentScheduler.cpp:2202:        else if (arg == "--scheduler-log-dir")
Sources/ExperimentScheduler.cpp:2203:            options.schedulerLogDir = RequireNextArg(argc, argv, i, arg);
Sources/ExperimentScheduler.cpp:2246:        else if (SplitOptionWithValue(arg, "--resume-model-id", value))
Sources/ExperimentScheduler.cpp:2247:            options.resumeModelId = ParsePositiveLongLong("--resume-model-id", value);
Sources/ExperimentScheduler.cpp:2254:        else if (SplitOptionWithValue(arg, "--scheduler-poll-seconds", value))
Sources/ExperimentScheduler.cpp:2255:            options.schedulerPollSeconds = ParsePositiveInt("--scheduler-poll-seconds", value);
Sources/ExperimentScheduler.cpp:2256:        else if (SplitOptionWithValue(arg, "--continuation-scan-seconds", value))
Sources/ExperimentScheduler.cpp:2258:            options.continuationScanSeconds = ParsePositiveInt("--continuation-scan-seconds", value);
Sources/ExperimentScheduler.cpp:2259:            options.continuationScanSecondsSpecified = true;
Sources/ExperimentScheduler.cpp:2261:        else if (SplitOptionWithValue(arg, "--continuation-max-queues-per-scan", value))
Sources/ExperimentScheduler.cpp:2263:            options.continuationMaxQueuesPerScan =
Sources/ExperimentScheduler.cpp:2264:                ParsePositiveInt("--continuation-max-queues-per-scan", value);
Sources/ExperimentScheduler.cpp:2265:            options.continuationMaxQueuesPerScanSpecified = true;
Sources/ExperimentScheduler.cpp:2267:        else if (SplitOptionWithValue(arg, "--scheduler-log-dir", value))
Sources/ExperimentScheduler.cpp:2268:            options.schedulerLogDir = value;
Sources/ExperimentScheduler.cpp:2275:        else if (SplitOptionWithValue(arg, "--pause-experiment", value))
Sources/ExperimentScheduler.cpp:2276:            options.pauseExperimentId = ParsePositiveLongLong("--pause-experiment", value);
Sources/ExperimentScheduler.cpp:2277:        else if (SplitOptionWithValue(arg, "--resume-experiment", value))
Sources/ExperimentScheduler.cpp:2278:            options.resumeExperimentId = ParsePositiveLongLong("--resume-experiment", value);
Sources/ExperimentScheduler.cpp:2279:        else if (SplitOptionWithValue(arg, "--cancel-experiment", value))
Sources/ExperimentScheduler.cpp:2280:            options.cancelExperimentId = ParsePositiveLongLong("--cancel-experiment", value);
Sources/ExperimentScheduler.cpp:2281:        else if (SplitOptionWithValue(arg, "--retry-failed-experiment", value))
Sources/ExperimentScheduler.cpp:2282:            options.retryFailedExperimentId = ParsePositiveLongLong("--retry-failed-experiment", value);
Sources/ExperimentScheduler.cpp:2283:        else if (SplitOptionWithValue(arg, "--retry-checkpoint-eval", value))
Sources/ExperimentScheduler.cpp:2284:            options.retryCheckpointEvalId = ParsePositiveLongLong("--retry-checkpoint-eval", value);
Sources/ExperimentScheduler.cpp:2287:        else if (SplitOptionWithValue(arg, "--enable-continuation-policy", value))
Sources/ExperimentScheduler.cpp:2288:            options.enableContinuationPolicyExperimentId = ParsePositiveLongLong("--enable-continuation-policy", value);
Sources/ExperimentScheduler.cpp:2289:        else if (SplitOptionWithValue(arg, "--disable-continuation-policy", value))
Sources/ExperimentScheduler.cpp:2290:            options.disableContinuationPolicyExperimentId = ParsePositiveLongLong("--disable-continuation-policy", value);
Sources/ExperimentScheduler.cpp:2291:        else if (SplitOptionWithValue(arg, "--set-continuation-policy", value))
Sources/ExperimentScheduler.cpp:2292:            options.setContinuationPolicy = ParseExperimentConfigPair("--set-continuation-policy", value);
Sources/ExperimentScheduler.cpp:2293:        else if (SplitOptionWithValue(arg, "--evaluate-continuation", value))
Sources/ExperimentScheduler.cpp:2294:            options.evaluateContinuationExperimentId = ParsePositiveLongLong("--evaluate-continuation", value);
Sources/ExperimentScheduler.cpp:2295:        else if (SplitOptionWithValue(arg, "--queue-continuation", value))
Sources/ExperimentScheduler.cpp:2296:            options.queueContinuationExperimentId = ParsePositiveLongLong("--queue-continuation", value);
Sources/ExperimentScheduler.cpp:2297:        else if (SplitOptionWithValue(arg, "--continuation-status", value))
Sources/ExperimentScheduler.cpp:2298:            options.continuationStatusExperimentId = ParsePositiveLongLong("--continuation-status", value);
Sources/ExperimentScheduler.cpp:2299:        else if (SplitOptionWithValue(arg, "--recommendation-status", value))
Sources/ExperimentScheduler.cpp:2300:            options.recommendationStatusId = ParsePositiveLongLong("--recommendation-status", value);
Sources/ExperimentScheduler.cpp:2301:        else if (SplitOptionWithValue(arg, "--recommendation-scan-status", value))
Sources/ExperimentScheduler.cpp:2302:            options.recommendationScanStatusId = ParsePositiveLongLong("--recommendation-scan-status", value);
Sources/ExperimentScheduler.cpp:2313:        else if (SplitOptionWithValue(arg, "--recommendation-status-filter", value))
Sources/ExperimentScheduler.cpp:2322:        else if (SplitOptionWithValue(arg, "--recommendation-score-status", value))
Sources/ExperimentScheduler.cpp:2323:            options.recommendationScoreStatusId = ParsePositiveLongLong("--recommendation-score-status", value);
Sources/ExperimentScheduler.cpp:2324:        else if (SplitOptionWithValue(arg, "--recommendation-score-run-status", value))
Sources/ExperimentScheduler.cpp:2325:            options.recommendationScoreRunStatusId = ParsePositiveLongLong("--recommendation-score-run-status", value);
Sources/ExperimentScheduler.cpp:2354:                     arg, "--recommendation-review-status", value))
Sources/ExperimentScheduler.cpp:2356:                "--recommendation-review-status", value);
Sources/ExperimentScheduler.cpp:2391:                     arg, "--recommendation-evaluation-status", value))
Sources/ExperimentScheduler.cpp:2393:                "--recommendation-evaluation-status", value);
Sources/ExperimentScheduler.cpp:2399:                     arg, "--recommendation-evaluation-run-status", value))
Sources/ExperimentScheduler.cpp:2401:                "--recommendation-evaluation-run-status", value);
Sources/ExperimentScheduler.cpp:2441:                     arg, "--recommendation-ranking-status", value))
Sources/ExperimentScheduler.cpp:2443:                "--recommendation-ranking-status", value);
Sources/ExperimentScheduler.cpp:2449:                     arg, "--recommendation-ranking-member-status", value))
Sources/ExperimentScheduler.cpp:2451:                "--recommendation-ranking-member-status", value);
Sources/ExperimentScheduler.cpp:2489:                     arg, "--list-conversion-proposals-by-review-status", value))
Sources/ExperimentScheduler.cpp:2503:                     arg, "--conversion-proposal-execution-status", value))
Sources/ExperimentScheduler.cpp:2505:                "--conversion-proposal-execution-status", value);
Sources/ExperimentScheduler.cpp:2515:                     "--recommendation-conversion-activation-status",
Sources/ExperimentScheduler.cpp:2519:                    "--recommendation-conversion-activation-status", value);
Sources/ExperimentScheduler.cpp:2762:                     "--launch-recommendation-campaign-materialization",
Sources/ExperimentScheduler.cpp:2767:                    "duplicate --launch-recommendation-campaign-materialization");
Sources/ExperimentScheduler.cpp:2769:            options.launchRecommendationCampaignMaterializationId =
Sources/ExperimentScheduler.cpp:2771:                    "--launch-recommendation-campaign-materialization", value);
Sources/ExperimentScheduler.cpp:2774:                     arg, "--recommendation-campaign-status", value))
Sources/ExperimentScheduler.cpp:2778:                    "duplicate --recommendation-campaign-status");
Sources/ExperimentScheduler.cpp:2781:                ParsePositiveLongLong("--recommendation-campaign-status", value);
Sources/ExperimentScheduler.cpp:2834:        else if (SplitOptionWithValue(arg, "--continuation-candidate-excluded", value))
Sources/ExperimentScheduler.cpp:2835:            options.queueContinuationCandidateExcluded = ParseBoolean("--continuation-candidate-excluded", value);
Sources/ExperimentScheduler.cpp:2883:            options.statusExperimentId = ParsePositiveLongLong("--experiment-id", value);
Sources/ExperimentScheduler.cpp:2885:            throw std::invalid_argument("unknown scheduler option '" + arg + "'");
Sources/ExperimentScheduler.cpp:2887:            throw std::invalid_argument("unexpected positional scheduler argument '" + arg + "'");
Sources/ExperimentScheduler.cpp:2904:        (options.schedulerStatus ? 1 : 0) +
Sources/ExperimentScheduler.cpp:2912:        (options.pauseExperimentId.has_value() ? 1 : 0) +
Sources/ExperimentScheduler.cpp:2913:        (options.resumeExperimentId.has_value() ? 1 : 0) +
Sources/ExperimentScheduler.cpp:2914:        (options.cancelExperimentId.has_value() ? 1 : 0) +
Sources/ExperimentScheduler.cpp:2915:        (options.retryFailedExperimentId.has_value() ? 1 : 0) +
Sources/ExperimentScheduler.cpp:2916:        (options.retryCheckpointEvalId.has_value() ? 1 : 0) +
Sources/ExperimentScheduler.cpp:2923:        (options.continuationStatusExperimentId.has_value() ? 1 : 0) +
Sources/ExperimentScheduler.cpp:2986:        (options.launchRecommendationCampaignMaterializationId.has_value()
Sources/ExperimentScheduler.cpp:3015:        throw std::invalid_argument("--continuation-candidate-excluded is only valid with --queue-experiment");
Sources/ExperimentScheduler.cpp:3031:        throw std::invalid_argument("recommendation status and scan filters require recommendation listing, scoring, or evaluation");
Sources/ExperimentScheduler.cpp:3035:        throw std::invalid_argument("recommendation scoring supports only proposed status");
Sources/ExperimentScheduler.cpp:3040:            "recommendation evaluation supports proposed status implicitly");
Sources/ExperimentScheduler.cpp:3057:        throw std::invalid_argument("invalid recommendation status filter");
Sources/ExperimentScheduler.cpp:3240:            "invalid conversion proposal review status");
Sources/ExperimentScheduler.cpp:3463:    if (options.launchRecommendationCampaignMaterializationId)
Sources/ExperimentScheduler.cpp:3467:                "campaign launch write requires --yes");
Sources/ExperimentScheduler.cpp:3469:            request{*options.launchRecommendationCampaignMaterializationId,
Sources/ExperimentScheduler.cpp:3478:                "--dry-run is not valid with --recommendation-campaign-status");
Sources/ExperimentScheduler.cpp:3481:                "--yes is not valid with --recommendation-campaign-status");
Sources/ExperimentScheduler.cpp:3504:        options.continuationDryRun ||
Sources/ExperimentScheduler.cpp:3505:        options.continuationScanSecondsSpecified ||
Sources/ExperimentScheduler.cpp:3506:        options.continuationMaxQueuesPerScanSpecified;
Sources/ExperimentScheduler.cpp:3508:        throw std::invalid_argument("automatic continuation options are only valid with --schedule-experiments");
Sources/ExperimentScheduler.cpp:3509:    if (options.continuationDryRun && !options.autoEvaluateContinuations)
Sources/ExperimentScheduler.cpp:3510:        throw std::invalid_argument("--continuation-dry-run requires --auto-evaluate-continuations or --auto-queue-continuations");
Sources/ExperimentScheduler.cpp:3535:        throw std::invalid_argument("expected exactly one experiment scheduler command");
Sources/ExperimentScheduler.cpp:3540:    if (!options.compactStatus && options.statusExperimentId.has_value())
Sources/ExperimentScheduler.cpp:3541:        throw std::invalid_argument("--experiment-id is only valid with --status");
Sources/ExperimentScheduler.cpp:3542:    if (options.recoverOrphansOnly && !options.scheduleExperiments)
Sources/ExperimentScheduler.cpp:3543:        throw std::invalid_argument("--recover-orphans-only requires --schedule-experiments");
Sources/ExperimentScheduler.cpp:3558:void EnsureRequiredEnqueueOptions(const SchedulerOptions& options)
Sources/ExperimentScheduler.cpp:3560:    if ((!options.symbol.has_value() && !options.resumeModelId.has_value()) ||
Sources/ExperimentScheduler.cpp:3567:        throw std::invalid_argument("--enqueue-experiment requires --symbol unless --resume-model-id is supplied, plus --prediction-horizon, --c-next-threshold, --target-epochs, --train-start, and --train-end");
Sources/ExperimentScheduler.cpp:3590:        return w.exec("SELECT count(*) FROM experiment;").one_row()[0].as<long long>();
Sources/ExperimentScheduler.cpp:3592:        return w.exec("SELECT count(*) FROM experiment_analysis_result;").one_row()[0].as<long long>();
Sources/ExperimentScheduler.cpp:3628:        throw std::runtime_error("resume model_id not found");
Sources/ExperimentScheduler.cpp:3635:        throw std::runtime_error("resume requires complete train_config_meta with 14 fields");
Sources/ExperimentScheduler.cpp:3656:              << ",resume_model_id=" << meta.modelId
Sources/ExperimentScheduler.cpp:3672:              << ",resume_model_id=" << modelId
Sources/ExperimentScheduler.cpp:3680:void MergeResumeMetaIntoQueueOptions(SchedulerOptions& options,
Sources/ExperimentScheduler.cpp:3741:                                                                 const SchedulerOptions& options)
Sources/ExperimentScheduler.cpp:3793:                                          const SchedulerOptions& options)
Sources/ExperimentScheduler.cpp:3836:              << ",resume_model_id=" << candidates.front().modelId
Sources/ExperimentScheduler.cpp:3917:        "SELECT 1 FROM model WHERE model_id = $1 AND experiment_id = $2 "
Sources/ExperimentScheduler.cpp:3930:bool RequireSchedulerTables(pqxx::work& w)
Sources/ExperimentScheduler.cpp:3938:        !ColumnExists(w, "experiment", "worker_started_at"))
Sources/ExperimentScheduler.cpp:3939:        missing.push_back("experiment.worker_started_at");
Sources/ExperimentScheduler.cpp:3998:                                                    const SchedulerOptions& options)
Sources/ExperimentScheduler.cpp:4000:    if (!options.resumeModelId.has_value())
Sources/ExperimentScheduler.cpp:4004:        TryLoadPersistedCanonicalSymbol(w, *options.resumeModelId);
Sources/ExperimentScheduler.cpp:4018:                  << ",model_id=" << *options.resumeModelId
Sources/ExperimentScheduler.cpp:4024:    PrintModelSymbolMissing(*options.resumeModelId);
Sources/ExperimentScheduler.cpp:4030:                  << ",model_id=" << *options.resumeModelId
Sources/ExperimentScheduler.cpp:4037:    throw std::runtime_error("resume-model-id is missing train_symbol_meta and --symbol was not supplied for legacy fallback");
Sources/ExperimentScheduler.cpp:4041:                                        const SchedulerOptions& options,
Sources/ExperimentScheduler.cpp:4058:        << " AND resume_model_id IS NOT DISTINCT FROM " << SqlNullable(w, options.resumeModelId)
Sources/ExperimentScheduler.cpp:4059:        << " AND status <> 'cancelled'";
Sources/ExperimentScheduler.cpp:4064:                                             const SchedulerOptions& options,
Sources/ExperimentScheduler.cpp:4074:        << " AND status NOT IN ('failed', 'cancelled')";
Sources/ExperimentScheduler.cpp:4085:                                        const SchedulerOptions& options,
Sources/ExperimentScheduler.cpp:4102:        ColumnExists(w, "experiment", "continuation_candidate_excluded");
Sources/ExperimentScheduler.cpp:4112:        throw std::runtime_error("continuation policy migration required; run ./migrate_lstm_db.sh");
Sources/ExperimentScheduler.cpp:4124:        << "resume_model_id, duplicate_nonce, status, phase, updated_at";
Sources/ExperimentScheduler.cpp:4148:        sql << ", continuation_candidate_excluded";
Sources/ExperimentScheduler.cpp:4163:        << SqlNullable(w, options.resumeModelId) << ","
Sources/ExperimentScheduler.cpp:4198:int EnqueueExperiment(const SchedulerOptions& options)
Sources/ExperimentScheduler.cpp:4228:    if (!RequireSchedulerTables(w))
Sources/ExperimentScheduler.cpp:4236:            "SELECT experiment_id FROM experiment WHERE " +
Sources/ExperimentScheduler.cpp:4263:SchedulerOptions ApplyQueueDefaults(SchedulerOptions options)
Sources/ExperimentScheduler.cpp:4287:void EnsureRequiredQueueOptions(const SchedulerOptions& options)
Sources/ExperimentScheduler.cpp:4292:        throw std::invalid_argument("--auto-resume is supported only with --queue-experiment");
Sources/ExperimentScheduler.cpp:4293:    if (options.autoResume && options.resumeModelId.has_value())
Sources/ExperimentScheduler.cpp:4294:        throw std::invalid_argument("--auto-resume cannot be combined with --resume-model-id");
Sources/ExperimentScheduler.cpp:4295:    if (options.queueSweep && options.resumeModelId.has_value())
Sources/ExperimentScheduler.cpp:4296:        throw std::invalid_argument("--queue-sweep does not support --resume-model-id");
Sources/ExperimentScheduler.cpp:4298:        throw std::invalid_argument("--queue-sweep does not support --auto-resume");
Sources/ExperimentScheduler.cpp:4301:    if (!options.resumeModelId.has_value() && !options.autoResume && !options.predictionHorizon.has_value())
Sources/ExperimentScheduler.cpp:4305:        if (!options.resumeModelId.has_value())
Sources/ExperimentScheduler.cpp:4306:            throw std::invalid_argument("--queue-experiment requires --symbol unless --resume-model-id is supplied");
Sources/ExperimentScheduler.cpp:4321:std::string CheckpointPolicyRuleText(const SchedulerOptions& options);
Sources/ExperimentScheduler.cpp:4324:                             const SchedulerOptions& options,
Sources/ExperimentScheduler.cpp:4334:              << ",resume_model_id=" << (options.resumeModelId.has_value() ? std::to_string(*options.resumeModelId) : "NULL")
Sources/ExperimentScheduler.cpp:4346:              << ",continuation_candidate_excluded="
Sources/ExperimentScheduler.cpp:4356:                                                   const SchedulerOptions& options,
Sources/ExperimentScheduler.cpp:4360:        "SELECT experiment_id FROM experiment WHERE " +
Sources/ExperimentScheduler.cpp:4369:                               const SchedulerOptions& options,
Sources/ExperimentScheduler.cpp:4385:int QueueExperiments(const SchedulerOptions& rawOptions)
Sources/ExperimentScheduler.cpp:4387:    SchedulerOptions options = rawOptions;
Sources/ExperimentScheduler.cpp:4389:    if (options.resumeModelId.has_value() && options.epochs.has_value())
Sources/ExperimentScheduler.cpp:4392:                                *options.resumeModelId);
Sources/ExperimentScheduler.cpp:4395:    if (options.dryRun && !options.resumeModelId.has_value() && !options.autoResume)
Sources/ExperimentScheduler.cpp:4411:    if (!RequireSchedulerTables(w))
Sources/ExperimentScheduler.cpp:4418:        options.resumeModelId = ResolveAutoResumeModelId(w, options);
Sources/ExperimentScheduler.cpp:4422:    if (options.resumeModelId.has_value())
Sources/ExperimentScheduler.cpp:4425:            ThrowQueueResumeInvalid("core_lr_override_not_allowed_in_resume", *options.resumeModelId);
Sources/ExperimentScheduler.cpp:4427:            ThrowQueueResumeInvalid("head_lr_override_not_allowed_in_resume", *options.resumeModelId);
Sources/ExperimentScheduler.cpp:4428:        const QueueResumeMeta meta = LoadQueueResumeMeta(w, *options.resumeModelId);
Sources/ExperimentScheduler.cpp:4453:            SchedulerOptions perSymbol = options;
Sources/ExperimentScheduler.cpp:4494:    if (!RequireSchedulerTables(w))
Sources/ExperimentScheduler.cpp:4500:            "SELECT 1 FROM experiment WHERE experiment_id = $1 LIMIT 1;",
Sources/ExperimentScheduler.cpp:4519:        "SELECT experiment_id, symbol, prediction_horizon, status, phase, "
Sources/ExperimentScheduler.cpp:4521:        "schema_version, scheduler_version, binary_name, invocation_mode, "
Sources/ExperimentScheduler.cpp:4539:              << ",status=" << row[3].as<std::string>()
Sources/ExperimentScheduler.cpp:4540:              << ",phase=" << row[4].as<std::string>()
Sources/ExperimentScheduler.cpp:4547:              << ",scheduler_version=" << FormatOptionalMetadataString(row, 11)
Sources/ExperimentScheduler.cpp:4561:        "AND status IN ('pending', 'running');");
Sources/ExperimentScheduler.cpp:4565:int BackfillExperimentMetadata(const SchedulerOptions& options)
Sources/ExperimentScheduler.cpp:4570:    if (!RequireSchedulerTables(w))
Sources/ExperimentScheduler.cpp:4657:int BackupDatabase(const SchedulerOptions& options)
Sources/ExperimentScheduler.cpp:4815:    experiment.resumeModelId = OptionalLongLongCell(row, 13);
Sources/ExperimentScheduler.cpp:4823:                                                        const std::string& phase)
Sources/ExperimentScheduler.cpp:4826:        "SELECT experiment_id, symbol, prediction_horizon, c_next_threshold, "
Sources/ExperimentScheduler.cpp:4829:        "last_model_id, resume_model_id, train_log_path, infer_log_path, analysis_log_path "
Sources/ExperimentScheduler.cpp:4831:        "WHERE status = 'pending' AND phase = $1 "
Sources/ExperimentScheduler.cpp:4833:        phase);
Sources/ExperimentScheduler.cpp:4845:        "SELECT experiment_id, symbol, prediction_horizon, c_next_threshold, "
Sources/ExperimentScheduler.cpp:4848:        "last_model_id, resume_model_id, train_log_path, infer_log_path, analysis_log_path, phase, worker_pid, "
Sources/ExperimentScheduler.cpp:4849:        "extract(epoch from COALESCE(worker_started_at, updated_at))::double precision "
Sources/ExperimentScheduler.cpp:4851:        "WHERE status = 'running' "
Sources/ExperimentScheduler.cpp:4866:                                      SchedulerEventLogState* logState,
Sources/ExperimentScheduler.cpp:4873:        "SELECT phase, status, count(*) "
Sources/ExperimentScheduler.cpp:4875:        "WHERE status IN ('pending', 'running') "
Sources/ExperimentScheduler.cpp:4876:        "AND phase IN ('train', 'infer', 'analyze') "
Sources/ExperimentScheduler.cpp:4877:        "GROUP BY phase, status;");
Sources/ExperimentScheduler.cpp:4880:        const std::string phase = row[0].as<std::string>();
Sources/ExperimentScheduler.cpp:4881:        const std::string status = row[1].as<std::string>();
Sources/ExperimentScheduler.cpp:4883:        if (phase == "train" && status == "pending")
Sources/ExperimentScheduler.cpp:4885:        else if (phase == "infer" && status == "pending")
Sources/ExperimentScheduler.cpp:4887:        else if (phase == "analyze" && status == "pending")
Sources/ExperimentScheduler.cpp:4889:        else if (phase == "train" && status == "running")
Sources/ExperimentScheduler.cpp:4891:        else if (phase == "infer" && status == "running")
Sources/ExperimentScheduler.cpp:4893:        else if (phase == "analyze" && status == "running")
Sources/ExperimentScheduler.cpp:4899:const char* SchedulerControlActionName(const SchedulerOptions& options)
Sources/ExperimentScheduler.cpp:4901:    if (options.pauseExperimentId.has_value())
Sources/ExperimentScheduler.cpp:4902:        return "pause";
Sources/ExperimentScheduler.cpp:4903:    if (options.resumeExperimentId.has_value())
Sources/ExperimentScheduler.cpp:4904:        return "resume";
Sources/ExperimentScheduler.cpp:4905:    if (options.cancelExperimentId.has_value())
Sources/ExperimentScheduler.cpp:4906:        return "cancel";
Sources/ExperimentScheduler.cpp:4907:    if (options.retryFailedExperimentId.has_value())
Sources/ExperimentScheduler.cpp:4908:        return "retry_failed";
Sources/ExperimentScheduler.cpp:4916:long long SchedulerControlExperimentId(const SchedulerOptions& options)
Sources/ExperimentScheduler.cpp:4918:    if (options.pauseExperimentId.has_value())
Sources/ExperimentScheduler.cpp:4919:        return *options.pauseExperimentId;
Sources/ExperimentScheduler.cpp:4920:    if (options.resumeExperimentId.has_value())
Sources/ExperimentScheduler.cpp:4921:        return *options.resumeExperimentId;
Sources/ExperimentScheduler.cpp:4922:    if (options.cancelExperimentId.has_value())
Sources/ExperimentScheduler.cpp:4923:        return *options.cancelExperimentId;
Sources/ExperimentScheduler.cpp:4924:    if (options.retryFailedExperimentId.has_value())
Sources/ExperimentScheduler.cpp:4925:        return *options.retryFailedExperimentId;
Sources/ExperimentScheduler.cpp:4930:    throw std::invalid_argument("missing scheduler control experiment id");
Sources/ExperimentScheduler.cpp:4933:std::optional<SchedulerControlExperimentRow> LoadSchedulerControlExperiment(pqxx::work& w,
Sources/ExperimentScheduler.cpp:4938:    sql << "SELECT experiment_id, status, phase, last_model_id, resume_model_id, symbol, prediction_horizon "
Sources/ExperimentScheduler.cpp:4941:        sql << " FOR UPDATE";
Sources/ExperimentScheduler.cpp:4948:    SchedulerControlExperimentRow row;
Sources/ExperimentScheduler.cpp:4950:    row.status = rows[0][1].as<std::string>();
Sources/ExperimentScheduler.cpp:4951:    row.phase = rows[0][2].as<std::string>();
Sources/ExperimentScheduler.cpp:4953:    row.resumeModelId = OptionalLongLongCell(rows[0], 4);
Sources/ExperimentScheduler.cpp:4959:std::optional<long long> ControlModelId(const SchedulerControlExperimentRow& row)
Sources/ExperimentScheduler.cpp:4963:    return row.resumeModelId;
Sources/ExperimentScheduler.cpp:4966:std::string RetryPhaseForExperiment(const SchedulerControlExperimentRow& row)
Sources/ExperimentScheduler.cpp:4968:    if (row.phase == "train" || row.phase == "infer" || row.phase == "analyze")
Sources/ExperimentScheduler.cpp:4969:        return row.phase;
Sources/ExperimentScheduler.cpp:4970:    if (row.phase == "done" && ControlModelId(row).has_value())
Sources/ExperimentScheduler.cpp:4975:void PrintSchedulerControlAttempt(const std::string& action,
Sources/ExperimentScheduler.cpp:4976:                                         const SchedulerControlExperimentRow& row)
Sources/ExperimentScheduler.cpp:4981:              << ",current_status=" << row.status
Sources/ExperimentScheduler.cpp:4982:              << ",current_phase=" << row.phase
Sources/ExperimentScheduler.cpp:4986:void PrintSchedulerControlRejected(const std::string& action,
Sources/ExperimentScheduler.cpp:4997:void PrintSchedulerControlTransition(const SchedulerControlExperimentRow& row,
Sources/ExperimentScheduler.cpp:5002:              << "Current: status=" << row.status << " phase=" << row.phase << "\n"
Sources/ExperimentScheduler.cpp:5003:              << "Requested: status=" << newStatus << " phase=" << newPhase << "\n";
Sources/ExperimentScheduler.cpp:5006:void PrintSchedulerControlApplied(const std::string& action,
Sources/ExperimentScheduler.cpp:5014:              << ",new_status=" << newStatus
Sources/ExperimentScheduler.cpp:5015:              << ",new_phase=" << newPhase
Sources/ExperimentScheduler.cpp:5019:std::optional<std::string> ValidateSchedulerControlTransition(const SchedulerOptions& options,
Sources/ExperimentScheduler.cpp:5020:                                                                     const SchedulerControlExperimentRow& row,
Sources/ExperimentScheduler.cpp:5024:    newStatus = row.status;
Sources/ExperimentScheduler.cpp:5025:    newPhase = row.phase;
Sources/ExperimentScheduler.cpp:5027:    if (options.pauseExperimentId.has_value())
Sources/ExperimentScheduler.cpp:5029:        if (row.status != "pending")
Sources/ExperimentScheduler.cpp:5030:            return "pause_requires_pending_status";
Sources/ExperimentScheduler.cpp:5031:        newStatus = "paused";
Sources/ExperimentScheduler.cpp:5034:    if (options.resumeExperimentId.has_value())
Sources/ExperimentScheduler.cpp:5036:        if (row.status != "paused")
Sources/ExperimentScheduler.cpp:5037:            return "resume_requires_paused_status";
Sources/ExperimentScheduler.cpp:5041:    if (options.cancelExperimentId.has_value())
Sources/ExperimentScheduler.cpp:5043:        if (row.status == "running")
Sources/ExperimentScheduler.cpp:5044:            return "running_experiment_cannot_be_cancelled";
Sources/ExperimentScheduler.cpp:5045:        if (row.status != "pending" && row.status != "paused")
Sources/ExperimentScheduler.cpp:5046:            return "cancel_requires_pending_or_paused_status";
Sources/ExperimentScheduler.cpp:5047:        newStatus = "cancelled";
Sources/ExperimentScheduler.cpp:5050:    if (options.retryFailedExperimentId.has_value())
Sources/ExperimentScheduler.cpp:5052:        if (row.status != "failed")
Sources/ExperimentScheduler.cpp:5053:            return "retry_requires_failed_status";
Sources/ExperimentScheduler.cpp:5060:        if (row.status == "running")
Sources/ExperimentScheduler.cpp:5070:        if (row.status == "running")
Sources/ExperimentScheduler.cpp:5079:    return "unknown_scheduler_control_action";
Sources/ExperimentScheduler.cpp:5082:void ApplySchedulerControlTransition(pqxx::work& w,
Sources/ExperimentScheduler.cpp:5084:                                            const SchedulerControlExperimentRow& row,
Sources/ExperimentScheduler.cpp:5089:    sql << "UPDATE experiment SET status = " << w.quote(newStatus)
Sources/ExperimentScheduler.cpp:5090:        << ", phase = " << w.quote(newPhase)
Sources/ExperimentScheduler.cpp:5093:    if (action == "cancel")
Sources/ExperimentScheduler.cpp:5097:    else if (action == "retry_failed" ||
Sources/ExperimentScheduler.cpp:5102:            << ", worker_started_at = NULL"
Sources/ExperimentScheduler.cpp:5106:            << ", worker_pid = NULL"
Sources/ExperimentScheduler.cpp:5114:int RunSchedulerControlCommand(const SchedulerOptions& options)
Sources/ExperimentScheduler.cpp:5116:    const std::string action = SchedulerControlActionName(options);
Sources/ExperimentScheduler.cpp:5117:    const long long experimentId = SchedulerControlExperimentId(options);
Sources/ExperimentScheduler.cpp:5124:    if (!RequireSchedulerTables(w))
Sources/ExperimentScheduler.cpp:5127:    std::optional<SchedulerControlExperimentRow> row =
Sources/ExperimentScheduler.cpp:5128:        LoadSchedulerControlExperiment(w, experimentId, willApply);
Sources/ExperimentScheduler.cpp:5131:        PrintSchedulerControlRejected(action, experimentId, "experiment_not_found");
Sources/ExperimentScheduler.cpp:5136:    PrintSchedulerControlAttempt(action, *row);
Sources/ExperimentScheduler.cpp:5141:            ValidateSchedulerControlTransition(options, *row, newStatus, newPhase);
Sources/ExperimentScheduler.cpp:5144:        PrintSchedulerControlRejected(action, experimentId, *rejection);
Sources/ExperimentScheduler.cpp:5149:    PrintSchedulerControlTransition(*row, newStatus, newPhase);
Sources/ExperimentScheduler.cpp:5156:                  << ",new_status=" << newStatus
Sources/ExperimentScheduler.cpp:5157:                  << ",new_phase=" << newPhase
Sources/ExperimentScheduler.cpp:5170:    ApplySchedulerControlTransition(w, action, *row, newStatus, newPhase);
Sources/ExperimentScheduler.cpp:5171:    PrintSchedulerControlApplied(action, experimentId, newStatus, newPhase);
Sources/ExperimentScheduler.cpp:5176:int RunRetryCheckpointEvalCommand(const SchedulerOptions& options)
Sources/ExperimentScheduler.cpp:5178:    if (!options.retryCheckpointEvalId.has_value())
Sources/ExperimentScheduler.cpp:5179:        throw std::invalid_argument("missing checkpoint_eval_id for retry");
Sources/ExperimentScheduler.cpp:5196:        << "SELECT checkpoint_eval_id, COALESCE(parent_experiment_id, experiment_id), "
Sources/ExperimentScheduler.cpp:5197:        << "checkpoint_model_id, checkpoint_epoch, status, phase "
Sources/ExperimentScheduler.cpp:5201:        selectSql << " FOR UPDATE";
Sources/ExperimentScheduler.cpp:5203:    pqxx::result rows = w.exec_params(selectSql.str(), *options.retryCheckpointEvalId);
Sources/ExperimentScheduler.cpp:5207:                  << ",checkpoint_eval_id=" << *options.retryCheckpointEvalId
Sources/ExperimentScheduler.cpp:5218:    const std::string status = rows[0][4].as<std::string>();
Sources/ExperimentScheduler.cpp:5219:    const std::string phase = rows[0][5].as<std::string>();
Sources/ExperimentScheduler.cpp:5220:    if (status != "failed")
Sources/ExperimentScheduler.cpp:5227:                  << ",status=" << status
Sources/ExperimentScheduler.cpp:5228:                  << ",phase=" << phase
Sources/ExperimentScheduler.cpp:5229:                  << ",reason=retry_requires_failed_status"
Sources/ExperimentScheduler.cpp:5234:    if (phase != "infer")
Sources/ExperimentScheduler.cpp:5241:                  << ",status=" << status
Sources/ExperimentScheduler.cpp:5242:                  << ",phase=" << phase
Sources/ExperimentScheduler.cpp:5243:                  << ",reason=retry_requires_failed_infer_phase"
Sources/ExperimentScheduler.cpp:5253:            "SET status = 'pending', phase = 'infer', worker_pid = NULL, "
Sources/ExperimentScheduler.cpp:5269:              << ",status=" << (options.dryRun ? "dry_run" : "pending")
Sources/ExperimentScheduler.cpp:5274:bool HasCheckpointControlCommand(const SchedulerOptions& options)
Sources/ExperimentScheduler.cpp:5289:long long CheckpointControlExperimentId(const SchedulerOptions& options)
Sources/ExperimentScheduler.cpp:5337:SchedulerOptions ParseCheckpointPolicyUpdateOptions(const std::string& config)
Sources/ExperimentScheduler.cpp:5339:    SchedulerOptions update;
Sources/ExperimentScheduler.cpp:5384:std::string CheckpointPolicyRuleText(const SchedulerOptions& options)
Sources/ExperimentScheduler.cpp:5457:        << "stop_after_checkpoint_epoch, status, phase, (";
Sources/ExperimentScheduler.cpp:5494:    config.status = rows[0][11].as<std::string>();
Sources/ExperimentScheduler.cpp:5495:    config.phase = rows[0][12].as<std::string>();
Sources/ExperimentScheduler.cpp:5500:int RunCheckpointControlCommand(const SchedulerOptions& options)
Sources/ExperimentScheduler.cpp:5506:    if (!RequireSchedulerTables(w))
Sources/ExperimentScheduler.cpp:5534:            "WHERE status = 'running' AND phase = 'train' "
Sources/ExperimentScheduler.cpp:5550:            "WHERE status = 'running' AND phase = 'train' "
Sources/ExperimentScheduler.cpp:5560:        "SELECT experiment_id, status, phase, checkpoint_interval, target_epochs "
Sources/ExperimentScheduler.cpp:5561:        "FROM experiment WHERE experiment_id = $1 FOR UPDATE;",
Sources/ExperimentScheduler.cpp:5573:    const std::string status = rows[0][1].as<std::string>();
Sources/ExperimentScheduler.cpp:5574:    const std::string phase = rows[0][2].as<std::string>();
Sources/ExperimentScheduler.cpp:5579:        (phase != "train" || status != "running"))
Sources/ExperimentScheduler.cpp:5583:                  << ",status=" << status
Sources/ExperimentScheduler.cpp:5584:                  << ",phase=" << phase
Sources/ExperimentScheduler.cpp:5591:    if (phase != "train" || (status != "pending" && status != "running" && status != "paused"))
Sources/ExperimentScheduler.cpp:5595:                  << ",status=" << status
Sources/ExperimentScheduler.cpp:5596:                  << ",phase=" << phase
Sources/ExperimentScheduler.cpp:5597:                  << ",reason=requires_train_pending_running_or_paused"
Sources/ExperimentScheduler.cpp:5643:                "SELECT checkpoint_policy_enabled FROM experiment WHERE experiment_id = $1;",
Sources/ExperimentScheduler.cpp:5748:        SchedulerOptions update = ParseCheckpointPolicyUpdateOptions(options.setCheckpointPolicy->second);
Sources/ExperimentScheduler.cpp:5819:                        SchedulerEventLogState* logState,
Sources/ExperimentScheduler.cpp:5838:int RunningCountForPhase(const QueueSnapshot& snapshot, const std::string& phase)
Sources/ExperimentScheduler.cpp:5840:    if (phase == "train")
Sources/ExperimentScheduler.cpp:5842:    if (phase == "infer")
Sources/ExperimentScheduler.cpp:5844:    if (phase == "analyze")
Sources/ExperimentScheduler.cpp:5849:void LogSkip(const std::string& phase,
Sources/ExperimentScheduler.cpp:5852:             SchedulerEventLogState* logState,
Sources/ExperimentScheduler.cpp:5855:    const std::string key = phase + "|" + std::to_string(experimentId) + "|" + reason;
Sources/ExperimentScheduler.cpp:5865:    std::cout << "SCHEDULER_SKIP_" << (phase == "train" ? "TRAIN" : phase == "infer" ? "INFER" : "ANALYZE")
Sources/ExperimentScheduler.cpp:5876:        << stats.launched << "|"
Sources/ExperimentScheduler.cpp:5882:                               SchedulerEventLogState* logState,
Sources/ExperimentScheduler.cpp:5888:        logState->previousPhaseKeys[stats.phase] == key)
Sources/ExperimentScheduler.cpp:5894:              << ",phase=" << stats.phase
Sources/ExperimentScheduler.cpp:5897:              << ",launched=" << stats.launched
Sources/ExperimentScheduler.cpp:5901:        logState->previousPhaseKeys[stats.phase] = key;
Sources/ExperimentScheduler.cpp:5904:int FailInvalidSchedulerPhases(pqxx::work& w)
Sources/ExperimentScheduler.cpp:5907:        "SELECT experiment_id, phase FROM experiment "
Sources/ExperimentScheduler.cpp:5908:        "WHERE status IN ('pending', 'running') "
Sources/ExperimentScheduler.cpp:5909:        "AND phase NOT IN ('train', 'infer', 'analyze', 'done') "
Sources/ExperimentScheduler.cpp:5915:        const std::string phase = row[1].as<std::string>();
Sources/ExperimentScheduler.cpp:5918:            "SET status = 'failed', "
Sources/ExperimentScheduler.cpp:5924:            "unknown_scheduler_phase:" + phase,
Sources/ExperimentScheduler.cpp:5928:                  << ",phase=" << phase
Sources/ExperimentScheduler.cpp:5929:                  << ",reason=unknown_scheduler_phase"
Sources/ExperimentScheduler.cpp:5941:std::string LogPathFor(const SchedulerOptions& options,
Sources/ExperimentScheduler.cpp:5943:                              const std::string& phase)
Sources/ExperimentScheduler.cpp:5946:    oss << options.schedulerLogDir
Sources/ExperimentScheduler.cpp:5949:        << "_" << phase << ".log";
Sources/ExperimentScheduler.cpp:5993:                                             const std::string& phase)
Sources/ExperimentScheduler.cpp:5995:    if (phase == "train")
Sources/ExperimentScheduler.cpp:6013:        if (phase == "infer")
Sources/ExperimentScheduler.cpp:6030:        else if (phase == "analyze")
Sources/ExperimentScheduler.cpp:6086:                           const SchedulerOptions& options)
Sources/ExperimentScheduler.cpp:6102:void PrintSchedulerExec(const std::vector<std::string>& argv)
Sources/ExperimentScheduler.cpp:6109:class SchedulerChildLaunchError : public std::runtime_error
Sources/ExperimentScheduler.cpp:6112:    SchedulerChildLaunchError(int errorNumber, const std::string& message)
Sources/ExperimentScheduler.cpp:6123:int SchedulerChildLaunchErrorNumber(const std::exception& error)
Sources/ExperimentScheduler.cpp:6125:    const auto* launchError = dynamic_cast<const SchedulerChildLaunchError*>(&error);
Sources/ExperimentScheduler.cpp:6126:    return launchError != nullptr ? launchError->ErrorNumber() : EIO;
Sources/ExperimentScheduler.cpp:6129:[[noreturn]] void ThrowSchedulerChildLaunchError(
Sources/ExperimentScheduler.cpp:6131:    const std::string& phase,
Sources/ExperimentScheduler.cpp:6136:    std::cout << SchedulerChildLaunchFailureDiagnostic(
Sources/ExperimentScheduler.cpp:6138:                     phase,
Sources/ExperimentScheduler.cpp:6142:    throw SchedulerChildLaunchError(
Sources/ExperimentScheduler.cpp:6147:pid_t LaunchChildProcess(const std::vector<std::string>& argv,
Sources/ExperimentScheduler.cpp:6150:                         const std::string& phase,
Sources/ExperimentScheduler.cpp:6155:        ThrowSchedulerChildLaunchError(
Sources/ExperimentScheduler.cpp:6156:            experimentId, phase, "", EINVAL, "empty child argv");
Sources/ExperimentScheduler.cpp:6159:    const double launchStartedEpoch =
Sources/ExperimentScheduler.cpp:6167:        ThrowSchedulerChildLaunchError(
Sources/ExperimentScheduler.cpp:6169:            phase,
Sources/ExperimentScheduler.cpp:6181:    const char* const phaseData = phase.data();
Sources/ExperimentScheduler.cpp:6182:    const size_t phaseLength = phase.size();
Sources/ExperimentScheduler.cpp:6184:    pid_t pid = ::fork();
Sources/ExperimentScheduler.cpp:6185:    if (pid < 0)
Sources/ExperimentScheduler.cpp:6187:        const int forkError = errno;
Sources/ExperimentScheduler.cpp:6189:        ThrowSchedulerChildLaunchError(
Sources/ExperimentScheduler.cpp:6190:            experimentId, phase, commandLine, forkError, "fork failed");
Sources/ExperimentScheduler.cpp:6193:    if (pid == 0)
Sources/ExperimentScheduler.cpp:6201:            WriteSchedulerChildExecFailureDiagnostic(
Sources/ExperimentScheduler.cpp:6202:                STDERR_FILENO, experimentId, phaseData, phaseLength, childErrno);
Sources/ExperimentScheduler.cpp:6216:        WriteSchedulerChildExecFailureDiagnostic(
Sources/ExperimentScheduler.cpp:6217:            STDERR_FILENO, experimentId, phaseData, phaseLength, childErrno);
Sources/ExperimentScheduler.cpp:6222:    SchedulerOwnedChild child;
Sources/ExperimentScheduler.cpp:6223:    child.pid = pid;
Sources/ExperimentScheduler.cpp:6225:    child.phase = phase;
Sources/ExperimentScheduler.cpp:6226:    child.operation = phase;
Sources/ExperimentScheduler.cpp:6231:    child.launchedAt = EA::RunMetadata::CurrentUtcTimestamp();
Sources/ExperimentScheduler.cpp:6232:    child.launchedEpoch = launchStartedEpoch;
Sources/ExperimentScheduler.cpp:6233:    gSchedulerOwnedChildren.emplace(pid, std::move(child));
Sources/ExperimentScheduler.cpp:6236:              << ",worker_pid=" << pid
Sources/ExperimentScheduler.cpp:6237:              << ",pid=" << pid
Sources/ExperimentScheduler.cpp:6238:              << ",phase=" << phase
Sources/ExperimentScheduler.cpp:6239:              << ",operation=" << phase
Sources/ExperimentScheduler.cpp:6243:    return pid;
Sources/ExperimentScheduler.cpp:6246:int WaitForChildProcess(pid_t pid)
Sources/ExperimentScheduler.cpp:6248:    if (pid <= 0)
Sources/ExperimentScheduler.cpp:6251:    int status = 0;
Sources/ExperimentScheduler.cpp:6252:    if (::waitpid(pid, &status, 0) < 0)
Sources/ExperimentScheduler.cpp:6257:    if (WIFEXITED(status))
Sources/ExperimentScheduler.cpp:6258:        return WEXITSTATUS(status);
Sources/ExperimentScheduler.cpp:6259:    else if (WIFSIGNALED(status))
Sources/ExperimentScheduler.cpp:6260:        return 128 + WTERMSIG(status);
Sources/ExperimentScheduler.cpp:6267:    const pid_t pid = LaunchChildProcess(argv, logPath, -1, "child");
Sources/ExperimentScheduler.cpp:6268:    result.launched = true;
Sources/ExperimentScheduler.cpp:6269:    result.exitCode = WaitForChildProcess(pid);
Sources/ExperimentScheduler.cpp:6270:    gSchedulerOwnedChildren.erase(pid);
Sources/ExperimentScheduler.cpp:6274:std::vector<std::string> BuildTrainCommand(const SchedulerOptions& options,
Sources/ExperimentScheduler.cpp:6283:    AddCliOption(argv, "--scheduler-experiment-id", std::to_string(experiment.experimentId));
Sources/ExperimentScheduler.cpp:6286:    const std::optional<long long> resumeFrom =
Sources/ExperimentScheduler.cpp:6287:        experiment.resumeModelId.has_value() ? experiment.resumeModelId : experiment.lastModelId;
Sources/ExperimentScheduler.cpp:6288:    if (resumeFrom.has_value())
Sources/ExperimentScheduler.cpp:6290:        AddCliOption(argv, "--resume-model-id", std::to_string(*resumeFrom));
Sources/ExperimentScheduler.cpp:6308:std::vector<std::string> BuildInferCommand(const SchedulerOptions& options,
Sources/ExperimentScheduler.cpp:6312:        throw std::runtime_error("infer phase has no last_model_id");
Sources/ExperimentScheduler.cpp:6314:        throw std::runtime_error("infer phase has no infer date range");
Sources/ExperimentScheduler.cpp:6320:    AddCliOption(argv, "--scheduler-experiment-id", std::to_string(experiment.experimentId));
Sources/ExperimentScheduler.cpp:6328:std::vector<std::string> BuildAnalyzeCommand(const SchedulerOptions& options,
Sources/ExperimentScheduler.cpp:6341:std::string CheckpointEvalLogPathFor(const SchedulerOptions& options,
Sources/ExperimentScheduler.cpp:6343:                                     const std::string& phase)
Sources/ExperimentScheduler.cpp:6345:    std::filesystem::path dir{options.schedulerLogDir};
Sources/ExperimentScheduler.cpp:6351:         << "_" << phase << ".log";
Sources/ExperimentScheduler.cpp:6355:std::vector<std::string> BuildCheckpointEvalInferCommand(const SchedulerOptions& options,
Sources/ExperimentScheduler.cpp:6365:    AddCliOption(argv, "--scheduler-checkpoint-eval-id", std::to_string(eval.checkpointEvalId));
Sources/ExperimentScheduler.cpp:6400:        << "SELECT e.experiment_id, tm.completed_epoch, m.model_id, "
Sources/ExperimentScheduler.cpp:6414:        << "AND e.phase = 'train' "
Sources/ExperimentScheduler.cpp:6415:        << "AND e.status IN ('pending', 'running', 'paused') "
Sources/ExperimentScheduler.cpp:6500:    eval.status = row[3].as<std::string>();
Sources/ExperimentScheduler.cpp:6501:    eval.phase = row[4].as<std::string>();
Sources/ExperimentScheduler.cpp:6502:    eval.workerPid = row[5].is_null() ? std::optional<int>{} : std::optional<int>{row[5].as<int>()};
Sources/ExperimentScheduler.cpp:6518:    eval.experiment.resumeModelId = OptionalLongLongCell(row, 20);
Sources/ExperimentScheduler.cpp:6527:                                                      const std::string& status,
Sources/ExperimentScheduler.cpp:6528:                                                      const std::string& phase)
Sources/ExperimentScheduler.cpp:6534:        "ce.status, ce.phase, ce.worker_pid, ce.infer_log_path, ce.analysis_log_path, "
Sources/ExperimentScheduler.cpp:6538:        "e.resume_model_id, e.train_log_path, "
Sources/ExperimentScheduler.cpp:6542:        "WHERE ce.status = $1 AND ce.phase = $2 "
Sources/ExperimentScheduler.cpp:6544:        status,
Sources/ExperimentScheduler.cpp:6545:        phase);
Sources/ExperimentScheduler.cpp:6561:        "ce.status, ce.phase, ce.worker_pid, ce.infer_log_path, ce.analysis_log_path, "
Sources/ExperimentScheduler.cpp:6565:        "e.resume_model_id, e.train_log_path, "
Sources/ExperimentScheduler.cpp:6607:    int resumedFromEpoch = 0;
Sources/ExperimentScheduler.cpp:6608:    const std::regex resumeRegex{R"(RESUME_COMPLETED_EPOCH=([0-9]+))"};
Sources/ExperimentScheduler.cpp:6609:    for (std::sregex_iterator it{logText.begin(), logText.end(), resumeRegex}, end; it != end; ++it)
Sources/ExperimentScheduler.cpp:6610:        resumedFromEpoch = std::stoi((*it)[1].str());
Sources/ExperimentScheduler.cpp:6617:    return std::max(checkpointEpoch, resumedFromEpoch + epochsCompletedInThisProcess);
Sources/ExperimentScheduler.cpp:6624:    sql << "SELECT experiment_id, train_log_path, current_epoch "
Sources/ExperimentScheduler.cpp:6626:        << "WHERE status = 'running' "
Sources/ExperimentScheduler.cpp:6627:        << "AND phase = 'train' "
Sources/ExperimentScheduler.cpp:6650:            "AND status = 'running' "
Sources/ExperimentScheduler.cpp:6651:            "AND phase = 'train' "
Sources/ExperimentScheduler.cpp:6658:void PersistDiscoveredRunningTrainingMetadata(const std::vector<SchedulerStatusJob>& jobs)
Sources/ExperimentScheduler.cpp:6665:        if (job.status != "running" || job.phase != "train")
Sources/ExperimentScheduler.cpp:6667:        if (!job.pid.has_value() && !job.currentEpoch.has_value())
Sources/ExperimentScheduler.cpp:6672:        if (job.pid.has_value())
Sources/ExperimentScheduler.cpp:6673:            sql << ", worker_pid = " << *job.pid;
Sources/ExperimentScheduler.cpp:6678:            << "AND status = 'running' "
Sources/ExperimentScheduler.cpp:6679:            << "AND phase = 'train';";
Sources/ExperimentScheduler.cpp:7056:        "AND status = 'completed' "
Sources/ExperimentScheduler.cpp:7104:        "AND status = 'completed' "
Sources/ExperimentScheduler.cpp:7134:        "AND status = 'completed' AND inference_scope = 'final' "
Sources/ExperimentScheduler.cpp:7163:        "AND status = 'completed' "
Sources/ExperimentScheduler.cpp:7184:        "AND inference_scope = 'checkpoint' AND status = 'completed' "
Sources/ExperimentScheduler.cpp:7204:        "AND status = 'completed' "
Sources/ExperimentScheduler.cpp:7238:        "AND analysis_status = 'completed' "
Sources/ExperimentScheduler.cpp:7254:        "SELECT 1 FROM experiment_analysis_result "
Sources/ExperimentScheduler.cpp:7257:        "AND analysis_status = 'completed' "
Sources/ExperimentScheduler.cpp:7455:        "SELECT e.experiment_id, a.model_id, a.symbol, a.prediction_horizon, "
Sources/ExperimentScheduler.cpp:7457:        "a.accept_accuracy, a.leader_score, a.analysis_status, e.completed_at::text "
Sources/ExperimentScheduler.cpp:7460:        "WHERE a.analysis_status = 'completed' "
Sources/ExperimentScheduler.cpp:7478:        "analysis_status",
Sources/ExperimentScheduler.cpp:7496:        "analysis_status",
Sources/ExperimentScheduler.cpp:7497:        "status",
Sources/ExperimentScheduler.cpp:7498:        "phase",
Sources/ExperimentScheduler.cpp:7516:        "status",
Sources/ExperimentScheduler.cpp:7517:        "phase",
Sources/ExperimentScheduler.cpp:7563:        " --queue-experiment --resume-model-id=' || COALESCE(a.model_id, e.last_model_id)::text || "
Sources/ExperimentScheduler.cpp:7574:    return ActionCliPrefixSql() + " --status --experiment-id=' || e.experiment_id::text";
Sources/ExperimentScheduler.cpp:7593:        "SELECT e.experiment_id, COALESCE(a.model_id, e.last_model_id), "
Sources/ExperimentScheduler.cpp:7597:        "COALESCE(a.analysis_status, 'n/a'), e.status, e.phase, e.completed_at::text, " +
Sources/ExperimentScheduler.cpp:7609:        "SELECT e.experiment_id, COALESCE(a.model_id, e.last_model_id), "
Sources/ExperimentScheduler.cpp:7612:        "e.status, e.phase, a.infer_accuracy, a.accept_rate, a.accept_accuracy, "
Sources/ExperimentScheduler.cpp:7653:    out << "This report is advisory only. It is generated from existing scheduler experiment "
Sources/ExperimentScheduler.cpp:7654:        << "and analysis rows and does not queue, update, cancel, pause, retry, stop, or "
Sources/ExperimentScheduler.cpp:7655:        << "launch experiments.\n\n";
Sources/ExperimentScheduler.cpp:7687:    out << "This report is advisory only. It identifies scheduler experiment rows and model "
Sources/ExperimentScheduler.cpp:7688:        << "references that may be obsolete, dominated, failed, cancelled, duplicate, "
Sources/ExperimentScheduler.cpp:7689:        << "or missing analysis. It does not delete, archive, cancel, pause, retry, "
Sources/ExperimentScheduler.cpp:7690:        << "resume, move, or mutate anything.\n\n";
Sources/ExperimentScheduler.cpp:7718:                "'Queue resumed extension from this model'",
Sources/ExperimentScheduler.cpp:7722:            "WHERE e.status = 'completed' "
Sources/ExperimentScheduler.cpp:7723:            "AND a.analysis_status = 'completed' "
Sources/ExperimentScheduler.cpp:7735:            "  SELECT e.experiment_id, a.model_id, a.symbol, a.prediction_horizon, "
Sources/ExperimentScheduler.cpp:7737:            "         a.accept_accuracy, a.leader_score, a.analysis_status, e.status, e.phase, "
Sources/ExperimentScheduler.cpp:7746:            "  WHERE e.status = 'completed' "
Sources/ExperimentScheduler.cpp:7747:            "  AND a.analysis_status = 'completed' "
Sources/ExperimentScheduler.cpp:7752:            "SELECT e.experiment_id, e.model_id, e.symbol, e.prediction_horizon, e.target_epochs, "
Sources/ExperimentScheduler.cpp:7754:            "e.analysis_status, e.status, e.phase, e.completed_at::text, "
Sources/ExperimentScheduler.cpp:7782:            "WHERE e.status = 'completed' "
Sources/ExperimentScheduler.cpp:7783:            "AND a.analysis_status = 'completed' "
Sources/ExperimentScheduler.cpp:7800:                "'Review logs; retry failed experiment if still relevant'",
Sources/ExperimentScheduler.cpp:7801:                ExperimentControlCommandSql("--retry-failed-experiment")) +
Sources/ExperimentScheduler.cpp:7805:            "WHERE e.status = 'failed' "
Sources/ExperimentScheduler.cpp:7814:            "  SELECT e.experiment_id, a.model_id, a.symbol, a.prediction_horizon, "
Sources/ExperimentScheduler.cpp:7816:            "         a.accept_accuracy, a.leader_score, a.analysis_status, e.status, e.phase, "
Sources/ExperimentScheduler.cpp:7820:            "  WHERE e.status = 'completed' "
Sources/ExperimentScheduler.cpp:7821:            "  AND a.analysis_status = 'completed' "
Sources/ExperimentScheduler.cpp:7830:            "SELECT e.experiment_id, COALESCE(a.model_id, e.last_model_id), "
Sources/ExperimentScheduler.cpp:7834:            "COALESCE(a.analysis_status, 'n/a'), e.status, e.phase, e.completed_at::text, "
Sources/ExperimentScheduler.cpp:7835:            "CASE WHEN e.status = 'running' THEN 'High' "
Sources/ExperimentScheduler.cpp:7836:            "     WHEN e.status = 'pending' THEN 'Medium' "
Sources/ExperimentScheduler.cpp:7840:            "CASE WHEN e.status = 'running' THEN 'Preview pause of running dominated experiment' "
Sources/ExperimentScheduler.cpp:7841:            "     WHEN e.status = 'pending' THEN 'Preview cancel of pending dominated experiment' "
Sources/ExperimentScheduler.cpp:7843:            "CASE WHEN e.status = 'running' THEN " + ExperimentControlCommandSql("--pause-experiment") + " "
Sources/ExperimentScheduler.cpp:7844:            "     WHEN e.status = 'pending' THEN " + ExperimentControlCommandSql("--cancel-experiment") + " "
Sources/ExperimentScheduler.cpp:7853:            "AND e.status IN ('pending', 'running', 'completed') "
Sources/ExperimentScheduler.cpp:7855:            "ORDER BY priority ASC, e.status DESC, e.symbol ASC, e.prediction_horizon ASC, e.experiment_id ASC "
Sources/ExperimentScheduler.cpp:7864:                "CASE WHEN e.status IN ('pending', 'running') THEN "
Sources/ExperimentScheduler.cpp:7865:                "'No action required; scheduler will process automatically' "
Sources/ExperimentScheduler.cpp:7867:                "CASE WHEN e.status IN ('pending', 'running') THEN 'No direct CLI action.' "
Sources/ExperimentScheduler.cpp:7872:            "  AND a.analysis_status = 'completed' "
Sources/ExperimentScheduler.cpp:7875:            "AND e.status <> 'cancelled' "
Sources/ExperimentScheduler.cpp:7886:                "CASE WHEN e.status IN ('pending', 'running') THEN "
Sources/ExperimentScheduler.cpp:7887:                "'No action required; scheduler will process automatically' "
Sources/ExperimentScheduler.cpp:7890:                "CASE WHEN e.status IN ('pending', 'running') THEN 'No direct CLI action.' "
Sources/ExperimentScheduler.cpp:7896:            "WHERE e.status = 'completed' "
Sources/ExperimentScheduler.cpp:7912:            "WHERE e.status = 'running' "
Sources/ExperimentScheduler.cpp:7927:            "WHERE e.status = 'pending' "
Sources/ExperimentScheduler.cpp:7935:            "  SELECT e.experiment_id, a.model_id, a.symbol, a.prediction_horizon, "
Sources/ExperimentScheduler.cpp:7937:            "         a.accept_accuracy, a.leader_score, a.analysis_status, e.status, e.phase, "
Sources/ExperimentScheduler.cpp:7944:            "  WHERE e.status = 'completed' "
Sources/ExperimentScheduler.cpp:7945:            "  AND a.analysis_status = 'completed' "
Sources/ExperimentScheduler.cpp:7948:            "SELECT e.experiment_id, e.model_id, e.symbol, e.prediction_horizon, "
Sources/ExperimentScheduler.cpp:7950:            "e.accept_accuracy, e.leader_score, e.analysis_status, e.status, e.phase, "
Sources/ExperimentScheduler.cpp:7994:            "WHERE e.status = 'completed' "
Sources/ExperimentScheduler.cpp:7995:            "AND a.analysis_status = 'completed' "
Sources/ExperimentScheduler.cpp:8003:            "  WHERE e2.status = 'completed' "
Sources/ExperimentScheduler.cpp:8004:            "  AND a2.analysis_status = 'completed' "
Sources/ExperimentScheduler.cpp:8016:            "  SELECT 1 FROM experiment ref "
Sources/ExperimentScheduler.cpp:8017:            "  WHERE ref.status IN ('pending', 'running') "
Sources/ExperimentScheduler.cpp:8019:            "       OR ref.resume_model_id = COALESCE(a.model_id, e.last_model_id))"
Sources/ExperimentScheduler.cpp:8029:            "  SELECT e.experiment_id, a.model_id, a.symbol, a.prediction_horizon, "
Sources/ExperimentScheduler.cpp:8031:            "         a.accept_accuracy, a.leader_score, e.status, e.phase, e.completed_at "
Sources/ExperimentScheduler.cpp:8034:            "  WHERE e.status = 'completed' "
Sources/ExperimentScheduler.cpp:8035:            "  AND a.analysis_status = 'completed' "
Sources/ExperimentScheduler.cpp:8040:            "SELECT d.experiment_id, d.model_id, d.symbol, d.prediction_horizon, "
Sources/ExperimentScheduler.cpp:8041:            "d.target_epochs, d.completed_epochs, d.status, d.phase, d.infer_accuracy, "
Sources/ExperimentScheduler.cpp:8049:            "  SELECT e2.experiment_id "
Sources/ExperimentScheduler.cpp:8074:            "WHERE e.status = 'failed' "
Sources/ExperimentScheduler.cpp:8082:                "'terminal cancelled run; no automatic cleanup action'",
Sources/ExperimentScheduler.cpp:8083:                "'No action required unless manually archiving old scheduler rows'",
Sources/ExperimentScheduler.cpp:8088:            "WHERE e.status = 'cancelled' "
Sources/ExperimentScheduler.cpp:8113:            "  WHERE e.status IN ('completed', 'failed', 'cancelled') "
Sources/ExperimentScheduler.cpp:8115:            "SELECT e.experiment_id, COALESCE(e.model_id, e.last_model_id), e.symbol, "
Sources/ExperimentScheduler.cpp:8116:            "e.prediction_horizon, e.target_epochs, e.completed_epochs, e.status, e.phase, "
Sources/ExperimentScheduler.cpp:8132:                "'pending for more than 7 days; verify still relevant before scheduler capacity is used'",
Sources/ExperimentScheduler.cpp:8133:                "'Inspect current experiment status; cancel only if intentionally obsolete'",
Sources/ExperimentScheduler.cpp:8138:            "WHERE e.status = 'pending' "
Sources/ExperimentScheduler.cpp:8148:                "CASE WHEN e.phase IN ('infer', 'analyze') AND e.status IN ('pending', 'running') THEN "
Sources/ExperimentScheduler.cpp:8149:                "'Wait for scheduler; missing metrics are expected while queued/running' "
Sources/ExperimentScheduler.cpp:8150:                "WHEN e.last_model_id IS NOT NULL THEN 'Inspect status and consider requeue-analysis if appropriate' "
Sources/ExperimentScheduler.cpp:8152:                "CASE WHEN e.status IN ('pending', 'running') THEN " + ExperimentStatusCommandSql() + " "
Sources/ExperimentScheduler.cpp:8158:            "WHERE e.status = 'completed' "
Sources/ExperimentScheduler.cpp:8178:            "'inspect live scheduler state before pruning decisions' AS reason, "
Sources/ExperimentScheduler.cpp:8179:            "'Review scheduler status' AS suggested_action, "
Sources/ExperimentScheduler.cpp:8180:            "" + StaticCommandSql("--scheduler-status") + " AS example_command;"));
Sources/ExperimentScheduler.cpp:8215:        "COUNT(*) FILTER (WHERE status = 'completed') AS completed_experiments, "
Sources/ExperimentScheduler.cpp:8216:        "COUNT(*) FILTER (WHERE status = 'failed') AS failed_experiments, "
Sources/ExperimentScheduler.cpp:8217:        "COUNT(*) FILTER (WHERE status IN ('pending', 'running')) AS pending_running_experiments "
Sources/ExperimentScheduler.cpp:8272:    out << "| phase | pending | running |\n";
Sources/ExperimentScheduler.cpp:8279:    out << "Unmanaged process detection is available from `--scheduler-status`; this index does not "
Sources/ExperimentScheduler.cpp:8298:        if (!RequireSchedulerTables(w))
Sources/ExperimentScheduler.cpp:8306:            "SELECT DISTINCT ON (a.symbol) e.experiment_id, a.model_id, a.symbol, a.prediction_horizon, "
Sources/ExperimentScheduler.cpp:8308:            "a.accept_accuracy, a.leader_score, a.analysis_status, e.completed_at::text "
Sources/ExperimentScheduler.cpp:8311:            "WHERE a.analysis_status = 'completed' "
Sources/ExperimentScheduler.cpp:8315:            "SELECT DISTINCT ON (a.prediction_horizon) e.experiment_id, a.model_id, a.symbol, a.prediction_horizon, "
Sources/ExperimentScheduler.cpp:8317:            "a.accept_accuracy, a.leader_score, a.analysis_status, e.completed_at::text "
Sources/ExperimentScheduler.cpp:8320:            "WHERE a.analysis_status = 'completed' "
Sources/ExperimentScheduler.cpp:8328:            "SELECT e.experiment_id, COALESCE(a.model_id, e.last_model_id), e.symbol, e.prediction_horizon, "
Sources/ExperimentScheduler.cpp:8330:            "a.accept_accuracy, a.leader_score, COALESCE(a.analysis_status, e.status), e.completed_at::text "
Sources/ExperimentScheduler.cpp:8334:            "WHERE e.status = 'failed' "
Sources/ExperimentScheduler.cpp:8374:void TryGenerateExperimentReports(const SchedulerOptions& options)
Sources/ExperimentScheduler.cpp:8443:        << "best_metric_name, best_metric_value, leader_score, analysis_status, analysis_notes, "
Sources/ExperimentScheduler.cpp:8540:        << "analysis_status = EXCLUDED.analysis_status, "
Sources/ExperimentScheduler.cpp:8579:        "stop_after_checkpoint_epoch, status, phase, "
Sources/ExperimentScheduler.cpp:8583:        "FOR UPDATE;",
Sources/ExperimentScheduler.cpp:8603:    config.status = rows[0][11].as<std::string>();
Sources/ExperimentScheduler.cpp:8604:    config.phase = rows[0][12].as<std::string>();
Sources/ExperimentScheduler.cpp:8623:        "AND ce.status = 'completed' "
Sources/ExperimentScheduler.cpp:8624:        "AND ce.phase = 'done' "
Sources/ExperimentScheduler.cpp:8629:        "AND a.analysis_status = 'completed' "
Sources/ExperimentScheduler.cpp:8648:            "SELECT status, phase, analysis_id "
Sources/ExperimentScheduler.cpp:8682:        "AND ce.status = 'completed' "
Sources/ExperimentScheduler.cpp:8683:        "AND ce.phase = 'done' "
Sources/ExperimentScheduler.cpp:8685:        "AND a.analysis_status = 'completed' "
Sources/ExperimentScheduler.cpp:8710:        << "  WHERE ce.status = 'completed' "
Sources/ExperimentScheduler.cpp:8711:        << "  AND ce.phase = 'done' "
Sources/ExperimentScheduler.cpp:8713:        << "  AND a.analysis_status = 'completed' "
Sources/ExperimentScheduler.cpp:8843:        "SELECT parent_experiment_id, checkpoint_epoch, checkpoint_model_id "
Sources/ExperimentScheduler.cpp:8845:        "WHERE checkpoint_eval_id = $1 FOR UPDATE;",
Sources/ExperimentScheduler.cpp:8906:        "AND status = 'running' "
Sources/ExperimentScheduler.cpp:8907:        "AND phase = 'train' "
Sources/ExperimentScheduler.cpp:8972:    if (config->phase != "train" || config->status != "running")
Sources/ExperimentScheduler.cpp:9050:int RunEvaluateCheckpointPolicyCommand(const SchedulerOptions& options)
Sources/ExperimentScheduler.cpp:9139:        " AND ir.inference_scope = 'checkpoint' AND ir.status = 'completed' "
Sources/ExperimentScheduler.cpp:9141:        "AND ce.status = 'completed' AND ce.phase = 'done' "
Sources/ExperimentScheduler.cpp:9143:        "AND a.analysis_scope = 'checkpoint' AND a.analysis_status = 'completed' "
Sources/ExperimentScheduler.cpp:9164:        "  AND ier.inference_scope = 'final' AND ier.status = 'completed' "
Sources/ExperimentScheduler.cpp:9168:        "AND a.analysis_scope = 'final' AND a.analysis_status = 'completed' "
Sources/ExperimentScheduler.cpp:9352:        "SELECT experiment_id FROM model WHERE model_id = $1;",
Sources/ExperimentScheduler.cpp:9388:        reason = "source_model_resume_metadata_mismatch";
Sources/ExperimentScheduler.cpp:9403:        "SELECT experiment_id FROM experiment "
Sources/ExperimentScheduler.cpp:9405:        "AND resume_model_id = $2 "
Sources/ExperimentScheduler.cpp:9407:        "ORDER BY (continuation_source_model_id IS NOT NULL) DESC, experiment_id ASC "
Sources/ExperimentScheduler.cpp:9460:    sql << "SELECT experiment_id FROM experiment "
Sources/ExperimentScheduler.cpp:9461:        << "WHERE status = 'completed' AND phase = 'done' ";
Sources/ExperimentScheduler.cpp:9463:        sql << "AND continuation_candidate_excluded = false ";
Sources/ExperimentScheduler.cpp:9513:        std::string resumeReason;
Sources/ExperimentScheduler.cpp:9519:                resumeReason))
Sources/ExperimentScheduler.cpp:9660:              << ",continuation_decision_id="
Sources/ExperimentScheduler.cpp:9696:    if (decision == "already_continued" || decision == "continuation_queued")
Sources/ExperimentScheduler.cpp:9711:        "SELECT continuation_decision_id, source_model_id, source_analysis_id, "
Sources/ExperimentScheduler.cpp:9716:        "FROM experiment_continuation_decision "
Sources/ExperimentScheduler.cpp:9719:        sql += " FOR UPDATE";
Sources/ExperimentScheduler.cpp:9763:        sql << "INSERT INTO experiment_continuation_decision ("
Sources/ExperimentScheduler.cpp:9789:            << "RETURNING continuation_decision_id;";
Sources/ExperimentScheduler.cpp:9793:        sql << "UPDATE experiment_continuation_decision SET "
Sources/ExperimentScheduler.cpp:9812:            << "WHERE continuation_decision_id=" << *existingDecisionId
Sources/ExperimentScheduler.cpp:9814:            << "RETURNING continuation_decision_id;";
Sources/ExperimentScheduler.cpp:9819:        throw std::runtime_error("continuation decision changed concurrently or is already queued");
Sources/ExperimentScheduler.cpp:9824:        "continuation_policy_last_decision = $1, "
Sources/ExperimentScheduler.cpp:9825:        "continuation_policy_last_decision_at = now(), "
Sources/ExperimentScheduler.cpp:9826:        "continuation_policy_last_reason = $2, "
Sources/ExperimentScheduler.cpp:9827:        "continuation_policy_selected_model_id = $3, "
Sources/ExperimentScheduler.cpp:9828:        "continuation_policy_queued_experiment_id = $4, "
Sources/ExperimentScheduler.cpp:9847:        persistDecision ? "continuation_evaluate" : "continuation_dry_run",
Sources/ExperimentScheduler.cpp:9872:        evaluation.reason = "continuation_policy_disabled";
Sources/ExperimentScheduler.cpp:9923:    std::string resumeReason;
Sources/ExperimentScheduler.cpp:9929:            resumeReason))
Sources/ExperimentScheduler.cpp:9931:        evaluation.reason = resumeReason;
Sources/ExperimentScheduler.cpp:9966:        evaluation.reason = "continuation_already_queued";
Sources/ExperimentScheduler.cpp:9979:        evaluation.reason = "equivalent_resume_experiment_exists";
Sources/ExperimentScheduler.cpp:10123:long long ContinuationControlExperimentId(const SchedulerOptions& options)
Sources/ExperimentScheduler.cpp:10131:    throw std::invalid_argument("missing continuation policy experiment id");
Sources/ExperimentScheduler.cpp:10134:bool HasContinuationControlCommand(const SchedulerOptions& options)
Sources/ExperimentScheduler.cpp:10141:int RunContinuationPolicyControlCommand(const SchedulerOptions& options)
Sources/ExperimentScheduler.cpp:10149:        std::cerr << "DATABASE_MIGRATION_REQUIRED,command=./migrate_lstm_db.sh,missing=continuation_policy_schema"
Sources/ExperimentScheduler.cpp:10189:            "UPDATE experiment SET continuation_policy_enabled = $1, updated_at = now() "
Sources/ExperimentScheduler.cpp:10279:        sql << ", continuation_policy_target_epochs = " << SqlNullable(w, update.targetEpochs);
Sources/ExperimentScheduler.cpp:10281:        sql << ", continuation_policy_min_evals = " << update.minEvals;
Sources/ExperimentScheduler.cpp:10283:        sql << ", continuation_policy_patience = " << update.patience;
Sources/ExperimentScheduler.cpp:10285:        sql << ", continuation_policy_min_leader_score = " << SqlNullable(w, update.minLeaderScore);
Sources/ExperimentScheduler.cpp:10287:        sql << ", continuation_policy_min_infer_accuracy = " << SqlNullable(w, update.minInferAccuracy);
Sources/ExperimentScheduler.cpp:10289:        sql << ", continuation_policy_min_improvement = " << SqlNullable(w, update.minImprovement);
Sources/ExperimentScheduler.cpp:10291:        sql << ", continuation_policy_max_degradation = " << SqlNullable(w, update.maxDegradation);
Sources/ExperimentScheduler.cpp:10293:        sql << ", continuation_policy_top_n = " << SqlNullable(w, update.topN);
Sources/ExperimentScheduler.cpp:10295:        sql << ", continuation_policy_scope = " << w.quote(update.scope);
Sources/ExperimentScheduler.cpp:10297:        sql << ", continuation_policy_trend_mode = " << w.quote(update.trendMode);
Sources/ExperimentScheduler.cpp:10299:        sql << ", continuation_policy_source_mode = " << w.quote(update.sourceMode);
Sources/ExperimentScheduler.cpp:10301:        sql << ", continuation_policy_include_excluded = "
Sources/ExperimentScheduler.cpp:10304:        sql << ", continuation_candidate_excluded = "
Sources/ExperimentScheduler.cpp:10307:        sql << ", continuation_policy_inherit_to_child = "
Sources/ExperimentScheduler.cpp:10310:        sql << ", continuation_policy_progression_mode = "
Sources/ExperimentScheduler.cpp:10313:        sql << ", continuation_policy_target_increment = "
Sources/ExperimentScheduler.cpp:10316:        sql << ", continuation_policy_max_target_epochs = "
Sources/ExperimentScheduler.cpp:10319:        sql << ", continuation_policy_target_sequence = "
Sources/ExperimentScheduler.cpp:10322:        sql << ", continuation_policy_revision = continuation_policy_revision + 1";
Sources/ExperimentScheduler.cpp:10324:        << " RETURNING continuation_policy_revision;";
Sources/ExperimentScheduler.cpp:10337:int RunEvaluateContinuationCommand(const SchedulerOptions& options)
Sources/ExperimentScheduler.cpp:10371:    SchedulerOptions& child,
Sources/ExperimentScheduler.cpp:10372:    int continuationSourceEpoch)
Sources/ExperimentScheduler.cpp:10419:            static_cast<long long>(continuationSourceEpoch) + 1,
Sources/ExperimentScheduler.cpp:10444:        << "continuation_policy_enabled = " << (plan.terminal ? "false" : "true") << ", "
Sources/ExperimentScheduler.cpp:10445:        << "continuation_policy_target_epochs = "
Sources/ExperimentScheduler.cpp:10447:        << "continuation_policy_min_evals = " << sourceConfig.minEvals << ", "
Sources/ExperimentScheduler.cpp:10448:        << "continuation_policy_patience = " << sourceConfig.patience << ", "
Sources/ExperimentScheduler.cpp:10449:        << "continuation_policy_min_leader_score = " << SqlNullable(w, sourceConfig.minLeaderScore) << ", "
Sources/ExperimentScheduler.cpp:10450:        << "continuation_policy_min_infer_accuracy = " << SqlNullable(w, sourceConfig.minInferAccuracy) << ", "
Sources/ExperimentScheduler.cpp:10451:        << "continuation_policy_min_improvement = " << SqlNullable(w, sourceConfig.minImprovement) << ", "
Sources/ExperimentScheduler.cpp:10452:        << "continuation_policy_max_degradation = " << SqlNullable(w, sourceConfig.maxDegradation) << ", "
Sources/ExperimentScheduler.cpp:10453:        << "continuation_policy_top_n = " << SqlNullable(w, sourceConfig.topN) << ", "
Sources/ExperimentScheduler.cpp:10454:        << "continuation_policy_scope = " << w.quote(sourceConfig.scope) << ", "
Sources/ExperimentScheduler.cpp:10455:        << "continuation_policy_trend_mode = " << w.quote(sourceConfig.trendMode) << ", "
Sources/ExperimentScheduler.cpp:10456:        << "continuation_policy_source_mode = " << w.quote(sourceConfig.sourceMode) << ", "
Sources/ExperimentScheduler.cpp:10457:        << "continuation_policy_include_excluded = " << (sourceConfig.includeExcluded ? "true" : "false") << ", "
Sources/ExperimentScheduler.cpp:10458:        << "continuation_candidate_excluded = " << (sourceConfig.candidateExcluded ? "true" : "false") << ", "
Sources/ExperimentScheduler.cpp:10459:        << "continuation_policy_inherit_to_child = " << (plan.terminal ? "false" : "true") << ", "
Sources/ExperimentScheduler.cpp:10460:        << "continuation_policy_progression_mode = "
Sources/ExperimentScheduler.cpp:10462:        << "continuation_policy_target_increment = "
Sources/ExperimentScheduler.cpp:10464:        << "continuation_policy_max_target_epochs = " << SqlNullable(w, sourceConfig.maxTargetEpochs) << ", "
Sources/ExperimentScheduler.cpp:10465:        << "continuation_policy_target_sequence = "
Sources/ExperimentScheduler.cpp:10467:        << "continuation_policy_revision = 1, "
Sources/ExperimentScheduler.cpp:10468:        << "continuation_policy_last_decision = NULL, "
Sources/ExperimentScheduler.cpp:10469:        << "continuation_policy_last_decision_at = NULL, "
Sources/ExperimentScheduler.cpp:10470:        << "continuation_policy_last_reason = NULL, "
Sources/ExperimentScheduler.cpp:10471:        << "continuation_policy_selected_model_id = NULL, "
Sources/ExperimentScheduler.cpp:10472:        << "continuation_policy_queued_experiment_id = NULL, "
Sources/ExperimentScheduler.cpp:10473:        << "continuation_policy_inherited = true, "
Sources/ExperimentScheduler.cpp:10474:        << "continuation_policy_inherited_from_experiment_id = " << sourceConfig.sourceExperimentId << ", "
Sources/ExperimentScheduler.cpp:10475:        << "continuation_policy_inherited_from_revision = " << sourceConfig.policyRevision << ", "
Sources/ExperimentScheduler.cpp:10476:        << "continuation_policy_inherited_from_hash = " << w.quote(plan.sourcePolicyHash) << ", "
Sources/ExperimentScheduler.cpp:10477:        << "continuation_policy_inheritance_status = "
Sources/ExperimentScheduler.cpp:10484:int RunQueueContinuationCommand(const SchedulerOptions& options)
Sources/ExperimentScheduler.cpp:10508:            evaluation.reason = "continuation_evaluation_not_persisted";
Sources/ExperimentScheduler.cpp:10522:            "SELECT decision, policy_revision, policy_hash, evidence_watermark, queued_experiment_id "
Sources/ExperimentScheduler.cpp:10523:            "FROM experiment_continuation_decision "
Sources/ExperimentScheduler.cpp:10524:            "WHERE continuation_decision_id = $1 FOR UPDATE;",
Sources/ExperimentScheduler.cpp:10533:            throw std::runtime_error("continuation decision changed before queueing");
Sources/ExperimentScheduler.cpp:10544:            evaluation.reason = "equivalent_resume_experiment_exists";
Sources/ExperimentScheduler.cpp:10550:        QueueResumeMeta resumeMeta;
Sources/ExperimentScheduler.cpp:10551:        std::string resumeReason;
Sources/ExperimentScheduler.cpp:10556:                &resumeMeta,
Sources/ExperimentScheduler.cpp:10557:                resumeReason))
Sources/ExperimentScheduler.cpp:10559:            throw std::runtime_error(resumeReason);
Sources/ExperimentScheduler.cpp:10562:        SchedulerOptions child;
Sources/ExperimentScheduler.cpp:10565:        child.resumeModelId = evaluation.selected.modelId;
Sources/ExperimentScheduler.cpp:10571:        MergeResumeMetaIntoQueueOptions(child, resumeMeta);
Sources/ExperimentScheduler.cpp:10585:        const int childGeneration = config.continuationSourceExperimentId.has_value()
Sources/ExperimentScheduler.cpp:10586:            ? config.continuationGeneration + 1
Sources/ExperimentScheduler.cpp:10591:            "continuation_source_experiment_id = $1, "
Sources/ExperimentScheduler.cpp:10592:            "continuation_source_model_id = $2, "
Sources/ExperimentScheduler.cpp:10593:            "continuation_source_epoch = $3, "
Sources/ExperimentScheduler.cpp:10594:            "continuation_decision_id = $4, "
Sources/ExperimentScheduler.cpp:10595:            "continuation_generation = $5, "

exec
/bin/bash -lc "rg -n '"'^([[:alnum:]_:<> ,&*]+ )?(Claim|Launch|Recover|Reconcile|RunScheduler|TryLaunch|Poll|Mark|Complete|Update|Process|Handle|Start|Stop)[[:alnum:]_]*'"\\(' Sources/ExperimentScheduler.cpp | sed -n '1,400p' && rg -n 'FOR UPDATE SKIP LOCKED|SKIP LOCKED|status = .running.|worker_pid|worker_started_at|worker_heartbeat|worker_attempt|scheduler' Sources/ExperimentScheduler.cpp | sed -n '1,700p'" in /Volumes/Developer SSD/ExpertAdvisor
 succeeded in 0ms:
4865:int RecoverOrphanedRunningExperiments(pqxx::work& w,
5114:int RunSchedulerControlCommand(const SchedulerOptions& options)
6147:pid_t LaunchChildProcess(const std::vector<std::string>& argv,
7343:std::string MarkdownCell(std::string value)
11344:void MarkAnalyzeFailed(pqxx::work& w,
11368:    MarkAnalyzeFailed(w, experimentId, errorMessage);
11409:        MarkAnalyzeFailed(w, experiment.experimentId, "analyze_missing_last_model_id");
11532:void MarkExperimentRunning(pqxx::work& w,
11557:    MarkExperimentRunning(w, experiment, phase, logPath);
11590:void MarkExperimentPendingPhase(pqxx::work& w,
11611:void MarkExperimentDone(pqxx::work& w,
11627:void MarkExperimentFailed(pqxx::work& w,
11641:void UpdateInferLogPath(pqxx::work& w,
11660:        UpdateInferLogPath(w, experiment, *recoveredLogPath);
11673:    MarkExperimentPendingPhase(w, experiment, "infer", "analyze", 0);
11706:        MarkExperimentDone(w, updatedExperiment, fromPhase);
11714:        MarkExperimentPendingPhase(w, updatedExperiment, fromPhase, "analyze", exitCode);
11722:        MarkExperimentPendingPhase(w, updatedExperiment, fromPhase, "infer", exitCode);
11730:        MarkExperimentFailed(w, updatedExperiment, "train_completed_missing_inference_range", exitCode);
11756:bool RecoverTrainOrphanFromModel(pqxx::work& w,
11863:int RecoverOrphanedRunningExperiments(pqxx::work& w,
11981:                MarkExperimentFailed(w, experiment, error);
12001:                MarkExperimentDone(w, experiment, "analyze");
12006:                MarkExperimentFailed(w, experiment, error);
12018:            MarkExperimentFailed(w, experiment, error);
12041:                MarkExperimentFailed(w, experiment, error + ";model_id=" + std::to_string(*linkedModelId));
12068:                MarkExperimentFailed(w, experiment, error + ";model_id=" + std::to_string(*finalModelId));
12094:                MarkExperimentFailed(w, experiment, error + ";checkpoint_model_id=" + std::to_string(*checkpointModelId));
12104:        MarkExperimentFailed(w, experiment, error);
12120:        MarkExperimentFailed(w, experiment, "infer_missing_last_model_id", -1);
12132:            MarkExperimentDone(w, experiment, "infer");
12136:            MarkExperimentPendingPhase(w, experiment, "infer", "analyze", exitCode);
12148:        MarkExperimentFailed(w, experiment, errorMessage, exitCode);
12257:        MarkExperimentDone(w, experiment, "analyze");
12308:    MarkExperimentFailed(w, experiment, error, exitCode);
12412:            MarkChildTerminationObserved(child.diagnosticState);
12520:                MarkExperimentFailed(w, job, "train_model_not_found");
12529:                MarkExperimentRunning(w, job, "train", LogPathFor(options, job, "train"));
12604:            MarkExperimentFailed(
12646:                MarkExperimentFailed(w, job, "infer_missing_last_model_id");
12654:                MarkExperimentFailed(w, job, "infer_model_not_found");
12673:                    MarkExperimentDone(w, job, "infer");
12677:                    MarkExperimentPendingPhase(w, job, "infer", "analyze");
12722:                MarkExperimentRunning(w, job, "infer", LogPathFor(options, job, "infer"));
12790:            MarkExperimentFailed(
12833:                MarkAnalyzeFailed(w, job.experimentId, "analyze_missing_last_model_id");
12841:                MarkAnalyzeFailed(w, job.experimentId, "analyze_model_not_found");
12854:                MarkExperimentDone(w, job, "analyze");
12871:                MarkExperimentRunning(w, job, "analyze", LogPathFor(options, job, "analysis"));
12943:            MarkAnalyzeFailed(
13327:int RunSchedulerOnce(const SchedulerOptions& options,
13344:            RecoverOrphanedRunningExperiments(w, logState, options.schedulerVerbose);
13362:int RunScheduler(const SchedulerOptions& options)
13957:bool ProcessStillExists(pid_t pid)
14223:void MarkExperimentStopped(pqxx::work& w,
14284:    MarkExperimentStopped(w, experimentId, message, usedSigkill ? 137 : 143);
17262:            return RunScheduler(options);
17371:            return RunSchedulerControlCommand(options);
81:    bool schedulerStatus = false;
276:    bool schedulerVerbose = false;
277:    bool schedulerOnce = false;
289:    int schedulerPollSeconds = 30;
290:    std::string schedulerLogDir = "experiment_logs";
566:    std::vector<int> schedulerPids;
572:    SchedulerResourceAggregate schedulerResources;
582:    std::optional<int> schedulerPollSeconds;
775:            arg == "--scheduler-status" ||
1510:        else if (arg == "--scheduler-status")
1511:            options.schedulerStatus = true;
1542:        else if (arg == "--scheduler-verbose")
1543:            options.schedulerVerbose = true;
1548:        else if (arg == "--scheduler-once")
1549:            options.schedulerOnce = true;
2200:        else if (arg == "--scheduler-poll-seconds")
2201:            options.schedulerPollSeconds = ParsePositiveInt(arg, RequireNextArg(argc, argv, i, arg));
2202:        else if (arg == "--scheduler-log-dir")
2203:            options.schedulerLogDir = RequireNextArg(argc, argv, i, arg);
2254:        else if (SplitOptionWithValue(arg, "--scheduler-poll-seconds", value))
2255:            options.schedulerPollSeconds = ParsePositiveInt("--scheduler-poll-seconds", value);
2267:        else if (SplitOptionWithValue(arg, "--scheduler-log-dir", value))
2268:            options.schedulerLogDir = value;
2885:            throw std::invalid_argument("unknown scheduler option '" + arg + "'");
2887:            throw std::invalid_argument("unexpected positional scheduler argument '" + arg + "'");
2904:        (options.schedulerStatus ? 1 : 0) +
3535:        throw std::invalid_argument("expected exactly one experiment scheduler command");
3938:        !ColumnExists(w, "experiment", "worker_started_at"))
3939:        missing.push_back("experiment.worker_started_at");
4521:        "schema_version, scheduler_version, binary_name, invocation_mode, "
4547:              << ",scheduler_version=" << FormatOptionalMetadataString(row, 11)
4848:        "last_model_id, resume_model_id, train_log_path, infer_log_path, analysis_log_path, phase, worker_pid, "
4849:        "extract(epoch from COALESCE(worker_started_at, updated_at))::double precision "
4851:        "WHERE status = 'running' "
4930:    throw std::invalid_argument("missing scheduler control experiment id");
5079:    return "unknown_scheduler_control_action";
5102:            << ", worker_started_at = NULL"
5106:            << ", worker_pid = NULL"
5253:            "SET status = 'pending', phase = 'infer', worker_pid = NULL, "
5534:            "WHERE status = 'running' AND phase = 'train' "
5550:            "WHERE status = 'running' AND phase = 'train' "
5924:            "unknown_scheduler_phase:" + phase,
5929:                  << ",reason=unknown_scheduler_phase"
5946:    oss << options.schedulerLogDir
6236:              << ",worker_pid=" << pid
6283:    AddCliOption(argv, "--scheduler-experiment-id", std::to_string(experiment.experimentId));
6320:    AddCliOption(argv, "--scheduler-experiment-id", std::to_string(experiment.experimentId));
6345:    std::filesystem::path dir{options.schedulerLogDir};
6365:    AddCliOption(argv, "--scheduler-checkpoint-eval-id", std::to_string(eval.checkpointEvalId));
6534:        "ce.status, ce.phase, ce.worker_pid, ce.infer_log_path, ce.analysis_log_path, "
6561:        "ce.status, ce.phase, ce.worker_pid, ce.infer_log_path, ce.analysis_log_path, "
6626:        << "WHERE status = 'running' "
6650:            "AND status = 'running' "
6673:            sql << ", worker_pid = " << *job.pid;
6678:            << "AND status = 'running' "
7653:    out << "This report is advisory only. It is generated from existing scheduler experiment "
7687:    out << "This report is advisory only. It identifies scheduler experiment rows and model "
7835:            "CASE WHEN e.status = 'running' THEN 'High' "
7840:            "CASE WHEN e.status = 'running' THEN 'Preview pause of running dominated experiment' "
7843:            "CASE WHEN e.status = 'running' THEN " + ExperimentControlCommandSql("--pause-experiment") + " "
7865:                "'No action required; scheduler will process automatically' "
7887:                "'No action required; scheduler will process automatically' "
7912:            "WHERE e.status = 'running' "
8083:                "'No action required unless manually archiving old scheduler rows'",
8132:                "'pending for more than 7 days; verify still relevant before scheduler capacity is used'",
8149:                "'Wait for scheduler; missing metrics are expected while queued/running' "
8178:            "'inspect live scheduler state before pruning decisions' AS reason, "
8179:            "'Review scheduler status' AS suggested_action, "
8180:            "" + StaticCommandSql("--scheduler-status") + " AS example_command;"));
8279:    out << "Unmanaged process detection is available from `--scheduler-status`; this index does not "
8906:        "AND status = 'running' "
11353:        "worker_pid = NULL, "
11443:        "SET status = 'completed', phase = 'done', worker_pid = NULL, completed_at = COALESCE(completed_at, now()), updated_at = now() "
11541:        "UPDATE experiment SET status = 'running', started_at = COALESCE(started_at, now()), " +
11544:        ", worker_started_at = now()" +
11546:        ", worker_pid = NULL, updated_at = now() "
11570:        "SET worker_pid = $1, current_operation = $2, updated_at = now() "
11571:        "WHERE experiment_id = $3 AND status = 'running' AND phase = $4;",
11600:        << ", error_message = NULL, worker_pid = NULL, updated_at = now() "
11617:        "SET status = 'completed', phase = 'done', worker_pid = NULL, completed_at = COALESCE(completed_at, now()), updated_at = now() "
11634:        "SET status = 'failed', exit_code = $1, error_message = $2, worker_pid = NULL, completed_at = now(), updated_at = now() "
11942:                      << ",ownership=previous_scheduler_or_unknown"
12204:        "UPDATE experiment SET exit_code = $1, error_message = $2, worker_pid = NULL, updated_at = now() "
12205:        "WHERE experiment_id = $3 AND (worker_pid = $4 OR worker_pid IS NULL) "
12224:        "status, phase, worker_pid "
12349:        "UPDATE experiment_checkpoint_eval SET status = 'failed', worker_pid = NULL, completed_at = now(), "
12351:        "AND status = 'running' AND phase = 'infer';",
12366:            "UPDATE experiment_checkpoint_eval SET status = 'failed', worker_pid = NULL, "
12368:            "WHERE checkpoint_eval_id = $2 AND status = 'running';",
12375:        "worker_pid = NULL, completed_at = now(), updated_at = now() "
12376:        "WHERE experiment_id = $2 AND status = 'running' "
12377:        "AND (worker_pid = $3 OR worker_pid IS NULL);",
12402:                          << ",worker_pid=" << child.pid
12425:                          << ",worker_pid=" << child.pid
12440:                          << ",worker_pid=" << child.pid
12457:                      << ",worker_pid=" << child.pid
12510:    EnsureLogDir(options.schedulerLogDir);
12518:            LogSkip("train", job.experimentId, "model_not_found", logState, options.schedulerVerbose);
12526:            LogSkip("train", job.experimentId, "already_running", logState, options.schedulerVerbose);
12538:            LogSkip("train", job.experimentId, "train_slots_full", logState, options.schedulerVerbose);
12588:                          << ",worker_pid=" << pid
12614:    PrintPhaseSchedulingStats(stats, logState, options.schedulerVerbose);
12638:    EnsureLogDir(options.schedulerLogDir);
12644:            LogSkip("infer", job.experimentId, "missing_model", logState, options.schedulerVerbose);
12652:            LogSkip("infer", job.experimentId, "model_not_found", logState, options.schedulerVerbose);
12660:            LogSkip("infer", job.experimentId, "existing_inference", logState, options.schedulerVerbose);
12689:            LogSkip("infer", job.experimentId, "valid_infer_log_path", logState, options.schedulerVerbose);
12704:            LogSkip("infer", job.experimentId, "discovered_valid_infer_log", logState, options.schedulerVerbose);
12719:            LogSkip("infer", job.experimentId, "already_running", logState, options.schedulerVerbose);
12731:            LogSkip("infer", job.experimentId, "infer_slots_full", logState, options.schedulerVerbose);
12774:                          << ",worker_pid=" << pid
12800:    PrintPhaseSchedulingStats(stats, logState, options.schedulerVerbose);
12825:    EnsureLogDir(options.schedulerLogDir);
12831:            LogSkip("analyze", job.experimentId, "missing_model", logState, options.schedulerVerbose);
12839:            LogSkip("analyze", job.experimentId, "model_not_found", logState, options.schedulerVerbose);
12847:            LogSkip("analyze", job.experimentId, "existing_analysis", logState, options.schedulerVerbose);
12862:            LogSkip("analyze", job.experimentId, "infer_log_missing", logState, options.schedulerVerbose);
12868:            LogSkip("analyze", job.experimentId, "already_running", logState, options.schedulerVerbose);
12880:            LogSkip("analyze", job.experimentId, "analyze_slots_full", logState, options.schedulerVerbose);
12927:                          << ",worker_pid=" << pid
12953:    PrintPhaseSchedulingStats(stats, logState, options.schedulerVerbose);
12997:        << "SET status = 'pending', phase = 'analyze', worker_pid = NULL, "
13064:                "SET status = 'failed', worker_pid = NULL, completed_at = now(), updated_at = now(), "
13085:    const int runningFinalInfer = CountRows(w, "SELECT count(*) FROM experiment WHERE status = 'running' AND phase = 'infer';");
13086:    const int runningCheckpointInfer = CountRows(w, "SELECT count(*) FROM experiment_checkpoint_eval WHERE status = 'running' AND phase = 'infer';");
13094:    EnsureLogDir(options.schedulerLogDir);
13179:            << "SET status = 'running', phase = 'infer', worker_pid = $1, infer_log_path = $2, "
13197:                      << ",worker_pid=" << pid
13231:    const int runningFinalAnalyze = CountRows(w, "SELECT count(*) FROM experiment WHERE status = 'running' AND phase = 'analyze';");
13259:                << "SET status = 'running', phase = 'analyze', updated_at = now(), error_message = NULL";
13342:            EA::RunMetadata::BackfillMissingExperimentRunMetadata(w, options.selfPath, "scheduler_start");
13344:            RecoverOrphanedRunningExperiments(w, logState, options.schedulerVerbose);
13349:        PrintQueueSnapshot(snapshot, logState, options.schedulerVerbose);
13374:            EA::RunMetadata::BackfillMissingExperimentRunMetadata(w, options.selfPath, "scheduler_start");
13376:            recoveryCount = RecoverOrphanedRunningExperiments(w, &logState, options.schedulerVerbose);
13431:        if (options.schedulerOnce)
13433:        ::sleep(static_cast<unsigned int>(options.schedulerPollSeconds));
13878:            snapshot.schedulerPids.push_back(pid);
13879:            AddResourceToAggregate(snapshot.schedulerResources, resource);
13886:            if (!snapshot.schedulerPollSeconds.has_value())
13887:                snapshot.schedulerPollSeconds = ExtractCommandIntOption(command, "--scheduler-poll-seconds");
13931:    snapshot.totalCpuPercent = snapshot.schedulerResources.cpuPercent +
13985:        command.find("--scheduler-status") != std::string::npos)
14075:        "WHERE status = 'running' AND phase IN ('train', 'infer', 'analyze') "
14247:        failureReason = "scheduler_tables_missing";
15075:            << "         count(*) FILTER (WHERE status = 'running') AS running_count, "
15091:        << "e.current_epoch, e.worker_pid, e.current_operation, ";
15193:            << "         count(*) FILTER (WHERE status = 'running') AS running_count, "
15209:        << "e.current_epoch, e.worker_pid, e.current_operation, ";
15801:    if (processes.schedulerPids.size() > 1)
15802:        warnings.push_back("multiple scheduler processes detected: " + std::to_string(processes.schedulerPids.size()));
15844:    std::cout << "  scheduler " << FormatAggregateResource(processes.schedulerResources) << "\n";
16206:    const bool schedulerRunning = !processes.schedulerPids.empty();
16207:    const std::string schedulerPid =
16208:        schedulerRunning ? std::to_string(processes.schedulerPids.front()) : "unknown";
16216:    if (schedulerRunning)
16220:                  << " pid=" << schedulerPid;
16221:        if (processes.schedulerPids.size() > 1)
16222:            std::cout << " additional_pids=" << (processes.schedulerPids.size() - 1);
16235:              << (processes.schedulerPollSeconds.has_value()
16236:                      ? std::to_string(*processes.schedulerPollSeconds) + "s"
16259:              << " counts=unavailable (scheduler process memory only)\n";
16304:                  << ",running=" << (schedulerRunning ? "1" : "0")
16305:                  << ",pid=" << (schedulerRunning ? schedulerPid : "unknown")
16315:                  << ",poll_seconds=" << OptionalIntText(processes.schedulerPollSeconds)
16324:                  << ",scheduler_cpu_percent=" << OptionalDoubleText(processes.schedulerResources.cpuPercent, 1)
16325:                  << ",scheduler_rss_mb=" << OptionalDoubleText(processes.schedulerResources.rssMb, 0)
16326:                  << ",scheduler_mem_percent=" << OptionalDoubleText(processes.schedulerResources.memPercent, 1)
16453:        << "[--max-analyze-procs=N] [--scheduler-poll-seconds=N] [--scheduler-once] "
16454:        << "[--scheduler-log-dir=PATH] [--auto-generate-reports] [--experiment-report-dir=PATH] "
16456:        << "[--scheduler-verbose] [--dry-run] [--recover-orphans-only] "
16462:        << "enter the normal pending/train queue and obey ordinary scheduler capacity. Defaults: scan "
16466:        << " --scheduler-status [--log-level=quiet|summary|diagnostic]\n"
16564:        << "Phase 4B evaluation is advisory only: it never creates or queues experiments or changes scheduler state.\n"
16582:        << "Phase 4B ranking and comparison are advisory only: they never create or queue experiments or change scheduler state.\n"
16596:        << "or queues an experiment or changes scheduler state.\n"
16606:        << "pending/train. It starts no worker and does not bypass the scheduler.\n"
16613:        << "experiment, audit record, worker, or scheduler state.\n"
16631:        << "experiment and never starts the scheduler or a worker.\n"
16680:        << "or starts the scheduler or workers. --yes is required unless "
16688:        << "the scheduler or workers. --yes is required unless --dry-run is "
16695:        << "persisted materialization member. It starts neither scheduler nor "
16704:        << "It starts neither scheduler nor workers and performs no automatic "
16711:        << "controls the scheduler or workers.\n"
17351:        if (options.schedulerStatus)

exec
/bin/bash -lc "nl -ba Sources/ExperimentScheduler.cpp | sed -n '11320,11750p' && nl -ba Sources/ExperimentScheduler.cpp | sed -n '11840,12340p' && nl -ba Sources/ExperimentScheduler.cpp | sed -n '12480,13460p'" in /Volumes/Developer SSD/ExpertAdvisor
 succeeded in 0ms:
 11320	            {
 11321	                ++counts.errors;
 11322	                candidate.evaluation.reason = "automatic_queue_failed";
 11323	                PrintContinuationAutoEvaluationLog(
 11324	                    "CONTINUATION_AUTO_ERROR",
 11325	                    candidate.config,
 11326	                    candidate.evaluation,
 11327	                    false);
 11328	            }
 11329	        }
 11330	    }
 11331
 11332	    std::cout << "CONTINUATION_AUTO_SCAN_COMPLETED"
 11333	              << ",scan_candidates=" << counts.candidates
 11334	              << ",scan_evaluated=" << counts.evaluated
 11335	              << ",scan_already_satisfied=" << counts.alreadySatisfied
 11336	              << ",scan_eligible=" << counts.eligible
 11337	              << ",scan_queued=" << counts.queued
 11338	              << ",scan_errors=" << counts.errors
 11339	              << ",dry_run=" << (dryRun ? "1" : "0")
 11340	              << std::endl;
 11341	    return counts;
 11342	}
 11343
 11344	void MarkAnalyzeFailed(pqxx::work& w,
 11345	                              long long experimentId,
 11346	                              const std::string& errorMessage)
 11347	{
 11348	    w.exec_params(
 11349	        "UPDATE experiment "
 11350	        "SET status = 'failed', "
 11351	        "exit_code = -1, "
 11352	        "error_message = $1, "
 11353	        "worker_pid = NULL, "
 11354	        "completed_at = now(), "
 11355	        "updated_at = now() "
 11356	        "WHERE experiment_id = $2;",
 11357	        errorMessage,
 11358	        experimentId);
 11359	}
 11360
 11361	[[maybe_unused]] void MarkAnalyzeFailedById(long long experimentId,
 11362	                                  const std::optional<long long>& modelId,
 11363	                                  const std::string& errorMessage)
 11364	{
 11365	    pqxx::connection c{LstmDbConnectionString()};
 11366	    pqxx::work w{c};
 11367	    SetTransactionReadWrite(w);
 11368	    MarkAnalyzeFailed(w, experimentId, errorMessage);
 11369	    w.commit();
 11370	    std::cerr << "SCHEDULER_ANALYZE_FAILED"
 11371	              << ",experiment_id=" << experimentId
 11372	              << ",model_id=" << (modelId.has_value() ? std::to_string(*modelId) : "none")
 11373	              << ",error=" << errorMessage
 11374	              << std::endl;
 11375	}
 11376
 11377	int AnalyzeExperimentById(long long experimentId, const SchedulerOptions& options)
 11378	{
 11379	    LogWorkerStarted(
 11380	        "ANALYSIS_WORKER_STARTED",
 11381	        "analyze",
 11382	        experimentId);
 11383
 11384	    pqxx::connection c{LstmDbConnectionString()};
 11385	    pqxx::work w{c};
 11386	    SetTransactionReadWrite(w);
 11387	    if (!RequireSchedulerTables(w))
 11388	        return 1;
 11389
 11390	    pqxx::result rows = w.exec_params(
 11391	        "SELECT experiment_id, symbol, prediction_horizon, c_next_threshold, "
 11392	        "core_lr_mult, head_lr_mult, target_epochs, checkpoint_interval, "
 11393	        "train_start::text, train_end::text, infer_start::text, infer_end::text, "
 11394	        "last_model_id, resume_model_id, train_log_path, infer_log_path, analysis_log_path "
 11395	        "FROM experiment WHERE experiment_id = $1;",
 11396	        experimentId);
 11397	    if (rows.empty())
 11398	    {
 11399	        std::cerr << "EXPERIMENT_ANALYSIS_FAILED"
 11400	                  << ",experiment_id=" << experimentId
 11401	                  << ",reason=not_found"
 11402	                  << std::endl;
 11403	        return 1;
 11404	    }
 11405
 11406	    ExperimentRow experiment = RowToExperiment(rows[0]);
 11407	    if (!experiment.lastModelId.has_value())
 11408	    {
 11409	        MarkAnalyzeFailed(w, experiment.experimentId, "analyze_missing_last_model_id");
 11410	        w.commit();
 11411	        std::cerr << "SCHEDULER_ANALYZE_FAILED"
 11412	                  << ",experiment_id=" << experiment.experimentId
 11413	                  << ",model_id=none"
 11414	                  << ",error=analyze_missing_last_model_id"
 11415	                  << std::endl;
 11416	        return 1;
 11417	    }
 11418
 11419	    const long long modelId = *experiment.lastModelId;
 11420	    std::cout << "SCHEDULER_ANALYZE_STARTED"
 11421	              << ",experiment_id=" << experiment.experimentId
 11422	              << ",model_id=" << modelId
 11423	              << std::endl;
 11424	    std::cout << "EXPERIMENT_ANALYSIS_STARTED"
 11425	              << ",experiment_id=" << experiment.experimentId
 11426	              << ",model_id=" << modelId
 11427	              << std::endl;
 11428
 11429	    ParsedMetrics metrics = ParseMetricsFromLogs(experiment);
 11430	    metrics.modelId = modelId;
 11431	    ApplyPersistedSymbolToAnalysisExperiment(w, experiment, metrics);
 11432	    const bool usedStructuredInferenceMetrics = ApplyStructuredInferenceMetrics(w, experiment, metrics);
 11433	    if (usedStructuredInferenceMetrics)
 11434	    {
 11435	        std::cout << "SCHEDULER_ANALYZE_USING_EXISTING_INFERENCE"
 11436	                  << ",model_id=" << modelId
 11437	                  << std::endl;
 11438	    }
 11439	    const std::optional<double> leaderScore = ComputeLeaderScore(metrics);
 11440	    UpsertAnalysisResult(w, experiment, metrics, leaderScore);
 11441	    w.exec_params(
 11442	        "UPDATE experiment "
 11443	        "SET status = 'completed', phase = 'done', worker_pid = NULL, completed_at = COALESCE(completed_at, now()), updated_at = now() "
 11444	        "WHERE experiment_id = $1;",
 11445	        experiment.experimentId);
 11446	    w.commit();
 11447
 11448	    std::cout << "SCHEDULER_PHASE_TRANSITION"
 11449	              << ",experiment_id=" << experiment.experimentId
 11450	              << ",from_phase=analyze"
 11451	              << ",to_phase=done"
 11452	              << std::endl;
 11453	    std::cout << "SCHEDULER_PIPELINE_DONE"
 11454	              << ",experiment_id=" << experiment.experimentId
 11455	              << ",model_id=" << modelId
 11456	              << std::endl;
 11457	    std::cout << "EXPERIMENT_ANALYSIS_SOURCE"
 11458	              << ",experiment_id=" << experiment.experimentId
 11459	              << ",source=" << (usedStructuredInferenceMetrics ? "inference_eval_result" : "logs")
 11460	              << ",train_log=" << (experiment.trainLogPath.has_value() ? *experiment.trainLogPath : "none")
 11461	              << ",infer_log=" << (experiment.inferLogPath.has_value() ? *experiment.inferLogPath : "none")
 11462	              << std::endl;
 11463	    std::cout << "EXPERIMENT_ANALYSIS_METRIC"
 11464	              << ",experiment_id=" << experiment.experimentId
 11465	              << ",metric=infer_accuracy"
 11466	              << ",value=" << (metrics.inferAccuracy.has_value() ? FormatDouble(*metrics.inferAccuracy) : "NULL")
 11467	              << std::endl;
 11468	    std::cout << "EXPERIMENT_LEADER_SCORE"
 11469	              << ",experiment_id=" << experiment.experimentId
 11470	              << ",model_id=" << (metrics.modelId.has_value() ? std::to_string(*metrics.modelId) : "none")
 11471	              << ",leader_score=" << (leaderScore.has_value() ? FormatDouble(*leaderScore) : "NULL")
 11472	              << std::endl;
 11473	    std::cout << "EXPERIMENT_LEADER_UPDATED"
 11474	              << ",experiment_id=" << experiment.experimentId
 11475	              << ",model_id=" << (metrics.modelId.has_value() ? std::to_string(*metrics.modelId) : "none")
 11476	              << std::endl;
 11477	    std::cout << "EXPERIMENT_ANALYSIS_COMPLETED"
 11478	              << ",experiment_id=" << experiment.experimentId
 11479	              << ",model_id=" << (metrics.modelId.has_value() ? std::to_string(*metrics.modelId) : "none")
 11480	              << std::endl;
 11481	    std::cout << "SCHEDULER_ANALYZE_COMPLETED"
 11482	              << ",experiment_id=" << experiment.experimentId
 11483	              << ",model_id=" << modelId
 11484	              << std::endl;
 11485	    std::cout.flush();
 11486	    std::cerr.flush();
 11487	    TryGenerateExperimentReports(options);
 11488	    std::cout.flush();
 11489	    std::cerr.flush();
 11490	    if (experiment.analysisLogPath.has_value())
 11491	    {
 11492	        std::ostringstream analysisLog;
 11493	        analysisLog << "EXPERIMENT_ANALYSIS_METRIC"
 11494	                    << ",experiment_id=" << experiment.experimentId
 11495	                    << ",metric=infer_accuracy"
 11496	                    << ",value=" << (metrics.inferAccuracy.has_value() ? FormatDouble(*metrics.inferAccuracy) : "NULL")
 11497	                    << "\n";
 11498	        analysisLog << "EXPERIMENT_LEADER_SCORE"
 11499	                    << ",experiment_id=" << experiment.experimentId
 11500	                    << ",model_id=" << (metrics.modelId.has_value() ? std::to_string(*metrics.modelId) : "none")
 11501	                    << ",leader_score=" << (leaderScore.has_value() ? FormatDouble(*leaderScore) : "NULL")
 11502	                    << "\n";
 11503	        AppendTextFile(*experiment.analysisLogPath, analysisLog.str());
 11504	    }
 11505	    return 0;
 11506	}
 11507
 11508	int AnalyzeCompletedExperiments()
 11509	{
 11510	    pqxx::connection c{LstmDbConnectionString()};
 11511	    pqxx::work w{c};
 11512	    if (!RequireSchedulerTables(w))
 11513	        return 1;
 11514
 11515	    pqxx::result rows = w.exec(
 11516	        "SELECT experiment_id FROM experiment "
 11517	        "WHERE phase = 'analyze' AND status = 'pending' "
 11518	        "ORDER BY updated_at ASC, experiment_id ASC;");
 11519	    std::vector<long long> ids;
 11520	    ids.reserve(rows.size());
 11521	    for (const auto& row : rows)
 11522	        ids.push_back(row[0].as<long long>());
 11523	    w.commit();
 11524
 11525	    int rc = 0;
 11526	    SchedulerOptions unused;
 11527	    for (long long id : ids)
 11528	        rc |= AnalyzeExperimentById(id, unused);
 11529	    return rc;
 11530	}
 11531
 11532	void MarkExperimentRunning(pqxx::work& w,
 11533	                                  const ExperimentRow& experiment,
 11534	                                  const std::string& phase,
 11535	                                  const std::string& logPath)
 11536	{
 11537	    const char* logColumn = phase == "train" ? "train_log_path" :
 11538	                            phase == "infer" ? "infer_log_path" :
 11539	                            "analysis_log_path";
 11540	    w.exec(
 11541	        "UPDATE experiment SET status = 'running', started_at = COALESCE(started_at, now()), " +
 11542	        std::string{logColumn} + " = " + w.quote(logPath) +
 11543	        ", current_operation = " + w.quote(phase) +
 11544	        ", worker_started_at = now()" +
 11545	        (phase == "train" ? ", current_epoch = NULL" : "") +
 11546	        ", worker_pid = NULL, updated_at = now() "
 11547	        "WHERE experiment_id = " + std::to_string(experiment.experimentId) + ";");
 11548	}
 11549
 11550	void PersistExperimentRunningBeforeLaunch(const ExperimentRow& experiment,
 11551	                                                 const std::string& phase,
 11552	                                                 const std::string& logPath)
 11553	{
 11554	    pqxx::connection c{LstmDbConnectionString()};
 11555	    pqxx::work w{c};
 11556	    SetTransactionReadWrite(w);
 11557	    MarkExperimentRunning(w, experiment, phase, logPath);
 11558	    w.commit();
 11559	}
 11560
 11561	void PersistExperimentWorkerPid(const ExperimentRow& experiment,
 11562	                                const std::string& phase,
 11563	                                pid_t pid)
 11564	{
 11565	    pqxx::connection c{LstmDbConnectionString()};
 11566	    pqxx::work w{c};
 11567	    SetTransactionReadWrite(w);
 11568	    w.exec_params(
 11569	        "UPDATE experiment "
 11570	        "SET worker_pid = $1, current_operation = $2, updated_at = now() "
 11571	        "WHERE experiment_id = $3 AND status = 'running' AND phase = $4;",
 11572	        static_cast<int>(pid),
 11573	        phase,
 11574	        experiment.experimentId,
 11575	        phase);
 11576	    w.commit();
 11577	}
 11578
 11579	void LogPhaseTransition(long long experimentId,
 11580	                               const std::string& fromPhase,
 11581	                               const std::string& toPhase)
 11582	{
 11583	    std::cout << "SCHEDULER_PHASE_TRANSITION"
 11584	              << ",experiment_id=" << experimentId
 11585	              << ",from_phase=" << fromPhase
 11586	              << ",to_phase=" << toPhase
 11587	              << std::endl;
 11588	}
 11589
 11590	void MarkExperimentPendingPhase(pqxx::work& w,
 11591	                                       const ExperimentRow& experiment,
 11592	                                       const std::string& fromPhase,
 11593	                                       const std::string& toPhase,
 11594	                                       const std::optional<int>& exitCode = std::nullopt)
 11595	{
 11596	    std::ostringstream sql;
 11597	    sql << "UPDATE experiment "
 11598	        << "SET status = 'pending', phase = " << w.quote(toPhase)
 11599	        << ", exit_code = " << (exitCode.has_value() ? std::to_string(*exitCode) : "NULL")
 11600	        << ", error_message = NULL, worker_pid = NULL, updated_at = now() "
 11601	        << "WHERE experiment_id = " << experiment.experimentId << ";";
 11602	    w.exec(sql.str());
 11603	    LogPhaseTransition(experiment.experimentId, fromPhase, toPhase);
 11604	    std::cout << "EXPERIMENT_PHASE_CHANGED"
 11605	              << ",experiment_id=" << experiment.experimentId
 11606	              << ",phase=" << toPhase
 11607	              << ",status=pending"
 11608	              << std::endl;
 11609	}
 11610
 11611	void MarkExperimentDone(pqxx::work& w,
 11612	                               const ExperimentRow& experiment,
 11613	                               const std::string& fromPhase)
 11614	{
 11615	    w.exec_params(
 11616	        "UPDATE experiment "
 11617	        "SET status = 'completed', phase = 'done', worker_pid = NULL, completed_at = COALESCE(completed_at, now()), updated_at = now() "
 11618	        "WHERE experiment_id = $1;",
 11619	        experiment.experimentId);
 11620	    LogPhaseTransition(experiment.experimentId, fromPhase, "done");
 11621	    std::cout << "SCHEDULER_PIPELINE_DONE"
 11622	              << ",experiment_id=" << experiment.experimentId
 11623	              << ",model_id=" << (experiment.lastModelId.has_value() ? std::to_string(*experiment.lastModelId) : "none")
 11624	              << std::endl;
 11625	}
 11626
 11627	void MarkExperimentFailed(pqxx::work& w,
 11628	                                 const ExperimentRow& experiment,
 11629	                                 const std::string& errorMessage,
 11630	                                 int exitCode = -1)
 11631	{
 11632	    w.exec_params(
 11633	        "UPDATE experiment "
 11634	        "SET status = 'failed', exit_code = $1, error_message = $2, worker_pid = NULL, completed_at = now(), updated_at = now() "
 11635	        "WHERE experiment_id = $3;",
 11636	        exitCode,
 11637	        errorMessage,
 11638	        experiment.experimentId);
 11639	}
 11640
 11641	void UpdateInferLogPath(pqxx::work& w,
 11642	                               const ExperimentRow& experiment,
 11643	                               const std::string& inferLogPath)
 11644	{
 11645	    w.exec_params(
 11646	        "UPDATE experiment "
 11647	        "SET infer_log_path = $1, updated_at = now() "
 11648	        "WHERE experiment_id = $2;",
 11649	        inferLogPath,
 11650	        experiment.experimentId);
 11651	}
 11652
 11653	void TransitionRecoveredInferenceToAnalyze(pqxx::work& w,
 11654	                                                  const ExperimentRow& experiment,
 11655	                                                  const std::string& sourceMarker,
 11656	                                                  const std::string& reason,
 11657	                                                  const std::optional<std::string>& recoveredLogPath = std::nullopt)
 11658	{
 11659	    if (recoveredLogPath.has_value())
 11660	        UpdateInferLogPath(w, experiment, *recoveredLogPath);
 11661
 11662	    std::cout << sourceMarker
 11663	              << ",experiment_id=" << experiment.experimentId
 11664	              << ",model_id=" << (experiment.lastModelId.has_value() ? std::to_string(*experiment.lastModelId) : "none")
 11665	              << ",phase=infer"
 11666	              << ",reason=" << reason;
 11667	    if (recoveredLogPath.has_value())
 11668	        std::cout << ",infer_log_path=" << *recoveredLogPath;
 11669	    else if (experiment.inferLogPath.has_value())
 11670	        std::cout << ",infer_log_path=" << *experiment.inferLogPath;
 11671	    std::cout << std::endl;
 11672
 11673	    MarkExperimentPendingPhase(w, experiment, "infer", "analyze", 0);
 11674	    std::cout << "SCHEDULER_ENQUEUE_ANALYZE"
 11675	              << ",experiment_id=" << experiment.experimentId
 11676	              << ",model_id=" << (experiment.lastModelId.has_value() ? std::to_string(*experiment.lastModelId) : "none")
 11677	              << std::endl;
 11678	}
 11679
 11680	bool TransitionAfterTrainModelAvailable(pqxx::work& w,
 11681	                                               const ExperimentRow& experiment,
 11682	                                               long long modelId,
 11683	                                               int exitCode,
 11684	                                               const std::string& fromPhase)
 11685	{
 11686	    ExperimentRow updatedExperiment = experiment;
 11687	    updatedExperiment.lastModelId = modelId;
 11688	    w.exec_params(
 11689	        "UPDATE experiment "
 11690	        "SET last_model_id = $1, exit_code = $2, error_message = NULL, updated_at = now() "
 11691	        "WHERE experiment_id = $3;",
 11692	        modelId,
 11693	        exitCode,
 11694	        experiment.experimentId);
 11695	    std::cout << "EXPERIMENT_LAST_MODEL_ID"
 11696	              << ",experiment_id=" << experiment.experimentId
 11697	              << ",model_id=" << modelId
 11698	              << std::endl;
 11699
 11700	    if (HasCompletedAnalysisResult(w, updatedExperiment))
 11701	    {
 11702	        std::cout << "SCHEDULER_SKIP_EXISTING_ANALYSIS"
 11703	                  << ",experiment_id=" << experiment.experimentId
 11704	                  << ",model_id=" << modelId
 11705	                  << std::endl;
 11706	        MarkExperimentDone(w, updatedExperiment, fromPhase);
 11707	    }
 11708	    else if (HasCompletedInferenceResult(w, updatedExperiment))
 11709	    {
 11710	        std::cout << "SCHEDULER_SKIP_EXISTING_INFERENCE"
 11711	                  << ",experiment_id=" << experiment.experimentId
 11712	                  << ",model_id=" << modelId
 11713	                  << std::endl;
 11714	        MarkExperimentPendingPhase(w, updatedExperiment, fromPhase, "analyze", exitCode);
 11715	        std::cout << "SCHEDULER_ENQUEUE_ANALYZE"
 11716	                  << ",experiment_id=" << experiment.experimentId
 11717	                  << ",model_id=" << modelId
 11718	                  << std::endl;
 11719	    }
 11720	    else if (updatedExperiment.inferStart.has_value() && updatedExperiment.inferEnd.has_value())
 11721	    {
 11722	        MarkExperimentPendingPhase(w, updatedExperiment, fromPhase, "infer", exitCode);
 11723	        std::cout << "SCHEDULER_ENQUEUE_INFER"
 11724	                  << ",experiment_id=" << experiment.experimentId
 11725	                  << ",model_id=" << modelId
 11726	                  << std::endl;
 11727	    }
 11728	    else
 11729	    {
 11730	        MarkExperimentFailed(w, updatedExperiment, "train_completed_missing_inference_range", exitCode);
 11731	        return false;
 11732	    }
 11733	    return true;
 11734	}
 11735
 11736	void RequeueTrainOrphanFromCheckpoint(pqxx::work& w,
 11737	                                      const ExperimentRow& experiment,
 11738	                                      const QueueResumeMeta& meta)
 11739	{
 11740	    w.exec_params(
 11741	        "UPDATE experiment "
 11742	        "SET status = 'pending', phase = 'train', last_model_id = $1, resume_model_id = $1, "
 11743	        "exit_code = NULL, error_message = NULL, updated_at = updated_at "
 11744	        "WHERE experiment_id = $2;",
 11745	        meta.modelId,
 11746	        experiment.experimentId);
 11747	    std::cout << "SCHEDULER_ORPHAN_RECOVERED"
 11748	              << ",experiment_id=" << experiment.experimentId
 11749	              << ",resume_model_id=" << meta.modelId
 11750	              << ",completed_epochs=" << meta.completedEpochs
 11840	        logState->previousRunningPresentKeys.find(key) != logState->previousRunningPresentKeys.end())
 11841	    {
 11842	        return;
 11843	    }
 11844
 11845	    if (phase == "infer")
 11846	    {
 11847	        std::cout << "SCHEDULER_ORPHAN_INFER_STILL_RUNNING"
 11848	                  << ",experiment_id=" << experiment.experimentId
 11849	                  << ",model_id=" << (experiment.lastModelId.has_value() ? std::to_string(*experiment.lastModelId) : "none")
 11850	                  << ",phase=infer"
 11851	                  << ",reason=matching_process_exists"
 11852	                  << std::endl;
 11853	    }
 11854	    else
 11855	    {
 11856	        std::cout << "SCHEDULER_RUNNING_EXPERIMENT_PRESENT"
 11857	                  << ",experiment_id=" << experiment.experimentId
 11858	                  << ",phase=" << phase
 11859	                  << std::endl;
 11860	    }
 11861	}
 11862
 11863	int RecoverOrphanedRunningExperiments(pqxx::work& w,
 11864	                                      SchedulerEventLogState* logState,
 11865	                                      bool verbose)
 11866	{
 11867	    int recoveredOrFailed = 0;
 11868	    const std::vector<RunningExperimentState> runningExperiments = LoadRunningExperiments(w);
 11869	    for (const auto& state : runningExperiments)
 11870	    {
 11871	        const ExperimentRow& experiment = state.experiment;
 11872	        const std::string& phase = state.phase;
 11873	        const auto ownedChild = std::find_if(
 11874	            gSchedulerOwnedChildren.begin(),
 11875	            gSchedulerOwnedChildren.end(),
 11876	            [&](const auto& entry) {
 11877	                return entry.second.experimentId == experiment.experimentId &&
 11878	                       entry.second.phase == phase;
 11879	            });
 11880	        if (ownedChild != gSchedulerOwnedChildren.end())
 11881	        {
 11882	            LogRunningExperimentPresent(experiment, phase, logState, verbose);
 11883	            continue;
 11884	        }
 11885	        if (RunningProcessExistsForExperiment(experiment, phase))
 11886	        {
 11887	            LogRunningExperimentPresent(experiment, phase, logState, verbose);
 11888	            continue;
 11889	        }
 11890
 11891	        bool successfulCompletionRecorded = false;
 11892	        if (phase == "infer")
 11893	        {
 11894	            successfulCompletionRecorded =
 11895	                HasCompletedInferenceResultForAttempt(
 11896	                    w, experiment, state.attemptStartedEpoch) ||
 11897	                HasValidInferenceLogPathForAttempt(
 11898	                    experiment, state.attemptStartedEpoch) ||
 11899	                DiscoverValidInferenceLogForAttempt(
 11900	                    experiment, state.attemptStartedEpoch).has_value();
 11901	        }
 11902	        else if (phase == "analyze")
 11903	        {
 11904	            successfulCompletionRecorded =
 11905	                HasCompletedAnalysisResultForAttempt(
 11906	                    w, experiment, state.attemptStartedEpoch);
 11907	        }
 11908	        else if (phase == "train")
 11909	        {
 11910	            successfulCompletionRecorded =
 11911	                FindLatestModelForExperimentSince(
 11912	                    w,
 11913	                    experiment.experimentId,
 11914	                    state.attemptStartedEpoch).has_value();
 11915	            if (!successfulCompletionRecorded)
 11916	            {
 11917	                const std::string existingTrainLog = ReadFileIfExists(experiment.trainLogPath);
 11918	                std::optional<long long> loggedModelId =
 11919	                    ExtractCompletedTrainModelId(existingTrainLog);
 11920	                if (!loggedModelId.has_value())
 11921	                    loggedModelId = ExtractCheckpointModelId(existingTrainLog);
 11922	                successfulCompletionRecorded =
 11923	                    loggedModelId.has_value() &&
 11924	                    ModelWasCreatedForAttempt(
 11925	                        w,
 11926	                        *loggedModelId,
 11927	                        experiment.experimentId,
 11928	                        state.attemptStartedEpoch);
 11929	            }
 11930	        }
 11931
 11932	        ++recoveredOrFailed;
 11933	        if (ShouldReconcileSchedulerOrphan(
 11934	                false,
 11935	                false,
 11936	                successfulCompletionRecorded))
 11937	        {
 11938	            std::cout << "SCHEDULER_ORPHAN_DETECTED"
 11939	                      << ",experiment_id=" << experiment.experimentId
 11940	                      << ",phase=" << phase
 11941	                      << ",pid=" << (state.workerPid.has_value() ? std::to_string(*state.workerPid) : "none")
 11942	                      << ",ownership=previous_scheduler_or_unknown"
 11943	                      << std::endl;
 11944	        }
 11945
 11946	        if (phase == "infer")
 11947	        {
 11948	            if (HasCompletedInferenceResultForAttempt(
 11949	                    w, experiment, state.attemptStartedEpoch))
 11950	            {
 11951	                TransitionRecoveredInferenceToAnalyze(
 11952	                    w,
 11953	                    experiment,
 11954	                    "SCHEDULER_ORPHAN_RECOVERED_INFERENCE_RESULT",
 11955	                    "completed_inference_eval_result");
 11956	            }
 11957	            else if (HasValidInferenceLogPathForAttempt(
 11958	                         experiment, state.attemptStartedEpoch))
 11959	            {
 11960	                TransitionRecoveredInferenceToAnalyze(
 11961	                    w,
 11962	                    experiment,
 11963	                    "SCHEDULER_ORPHAN_RECOVERED_INFER_LOG",
 11964	                    "valid_infer_log_path");
 11965	            }
 11966	            else if (const std::optional<std::string> discoveredLog =
 11967	                         DiscoverValidInferenceLogForAttempt(
 11968	                             experiment, state.attemptStartedEpoch);
 11969	                     discoveredLog.has_value())
 11970	            {
 11971	                TransitionRecoveredInferenceToAnalyze(
 11972	                    w,
 11973	                    experiment,
 11974	                    "SCHEDULER_ORPHAN_RECOVERED_INFER_LOG",
 11975	                    "discovered_valid_infer_log",
 11976	                    discoveredLog);
 11977	            }
 11978	            else
 11979	            {
 11980	                const std::string error = "orphaned_running_infer_no_process_no_result";
 11981	                MarkExperimentFailed(w, experiment, error);
 11982	                std::cout << "SCHEDULER_ORPHAN_MARKED_FAILED"
 11983	                          << ",experiment_id=" << experiment.experimentId
 11984	                          << ",model_id=" << (experiment.lastModelId.has_value() ? std::to_string(*experiment.lastModelId) : "none")
 11985	                          << ",phase=infer"
 11986	                          << ",error=" << error
 11987	                          << std::endl;
 11988	            }
 11989	            continue;
 11990	        }
 11991
 11992	        if (phase == "analyze")
 11993	        {
 11994	            if (HasCompletedAnalysisResultForAttempt(
 11995	                    w, experiment, state.attemptStartedEpoch))
 11996	            {
 11997	                std::cout << "SCHEDULER_ORPHAN_RECOVERED_ANALYSIS"
 11998	                          << ",experiment_id=" << experiment.experimentId
 11999	                          << ",model_id=" << (experiment.lastModelId.has_value() ? std::to_string(*experiment.lastModelId) : "none")
 12000	                          << std::endl;
 12001	                MarkExperimentDone(w, experiment, "analyze");
 12002	            }
 12003	            else
 12004	            {
 12005	                const std::string error = "orphaned_running_analyze_no_process_no_result";
 12006	                MarkExperimentFailed(w, experiment, error);
 12007	                std::cout << "SCHEDULER_ORPHAN_MARKED_FAILED"
 12008	                          << ",experiment_id=" << experiment.experimentId
 12009	                          << ",error=" << error
 12010	                          << std::endl;
 12011	            }
 12012	            continue;
 12013	        }
 12014
 12015	        if (phase != "train")
 12016	        {
 12017	            const std::string error = "orphaned_running_invalid_phase";
 12018	            MarkExperimentFailed(w, experiment, error);
 12019	            std::cout << "SCHEDULER_ORPHAN_MARKED_FAILED"
 12020	                      << ",experiment_id=" << experiment.experimentId
 12021	                      << ",error=" << error
 12022	                      << std::endl;
 12023	            continue;
 12024	        }
 12025
 12026	        const std::optional<long long> linkedModelId =
 12027	            FindLatestModelForExperimentSince(
 12028	                w,
 12029	                experiment.experimentId,
 12030	                state.attemptStartedEpoch);
 12031	        if (linkedModelId.has_value())
 12032	        {
 12033	            std::cout << "SCHEDULER_ORPHAN_RECOVERED_MODEL"
 12034	                      << ",experiment_id=" << experiment.experimentId
 12035	                      << ",model_id=" << *linkedModelId
 12036	                      << ",source=model_experiment_id"
 12037	                      << std::endl;
 12038	            if (!RecoverTrainOrphanFromModel(w, experiment, *linkedModelId))
 12039	            {
 12040	                const std::string error = "orphaned_running_train_model_unusable";
 12041	                MarkExperimentFailed(w, experiment, error + ";model_id=" + std::to_string(*linkedModelId));
 12042	                std::cout << "SCHEDULER_ORPHAN_FAILED"
 12043	                          << ",experiment_id=" << experiment.experimentId
 12044	                          << ",reason=model_unusable"
 12045	                          << std::endl;
 12046	            }
 12047	            continue;
 12048	        }
 12049
 12050	        const std::string logText = ReadFileIfExists(experiment.trainLogPath);
 12051	        std::optional<long long> finalModelId = ExtractCompletedTrainModelId(logText);
 12052	        if (finalModelId.has_value() &&
 12053	            !ModelWasCreatedForAttempt(
 12054	                w,
 12055	                *finalModelId,
 12056	                experiment.experimentId,
 12057	                state.attemptStartedEpoch))
 12058	            finalModelId.reset();
 12059	        if (finalModelId.has_value())
 12060	        {
 12061	            std::cout << "SCHEDULER_ORPHAN_RECOVERED_MODEL"
 12062	                      << ",experiment_id=" << experiment.experimentId
 12063	                      << ",model_id=" << *finalModelId
 12064	                      << std::endl;
 12065	            if (!RecoverTrainOrphanFromModel(w, experiment, *finalModelId))
 12066	            {
 12067	                const std::string error = "orphaned_running_train_model_unusable";
 12068	                MarkExperimentFailed(w, experiment, error + ";model_id=" + std::to_string(*finalModelId));
 12069	                std::cout << "SCHEDULER_ORPHAN_FAILED"
 12070	                          << ",experiment_id=" << experiment.experimentId
 12071	                          << ",reason=model_unusable"
 12072	                          << std::endl;
 12073	            }
 12074	            continue;
 12075	        }
 12076
 12077	        std::optional<long long> checkpointModelId = ExtractCheckpointModelId(logText);
 12078	        if (checkpointModelId.has_value() &&
 12079	            !ModelWasCreatedForAttempt(
 12080	                w,
 12081	                *checkpointModelId,
 12082	                experiment.experimentId,
 12083	                state.attemptStartedEpoch))
 12084	            checkpointModelId.reset();
 12085	        if (checkpointModelId.has_value())
 12086	        {
 12087	            std::cout << "SCHEDULER_ORPHAN_RECOVERED_CHECKPOINT"
 12088	                      << ",experiment_id=" << experiment.experimentId
 12089	                      << ",checkpoint_model_id=" << *checkpointModelId
 12090	                      << std::endl;
 12091	            if (!RecoverTrainOrphanFromModel(w, experiment, *checkpointModelId))
 12092	            {
 12093	                const std::string error = "orphaned_running_train_checkpoint_unusable";
 12094	                MarkExperimentFailed(w, experiment, error + ";checkpoint_model_id=" + std::to_string(*checkpointModelId));
 12095	                std::cout << "SCHEDULER_ORPHAN_FAILED"
 12096	                          << ",experiment_id=" << experiment.experimentId
 12097	                          << ",reason=checkpoint_unusable"
 12098	                          << std::endl;
 12099	            }
 12100	            continue;
 12101	        }
 12102
 12103	        const std::string error = "orphaned_running_train_no_process_no_model";
 12104	        MarkExperimentFailed(w, experiment, error);
 12105	        std::cout << "SCHEDULER_ORPHAN_FAILED"
 12106	                  << ",experiment_id=" << experiment.experimentId
 12107	                  << ",reason=no_recoverable_checkpoint"
 12108	                  << std::endl;
 12109	    }
 12110	    return recoveredOrFailed;
 12111	}
 12112
 12113	[[maybe_unused]] void CompleteInferPhase(pqxx::work& w,
 12114	                               const ExperimentRow& experiment,
 12115	                               int exitCode,
 12116	                               const std::string& commandDisplay)
 12117	{
 12118	    if (!experiment.lastModelId.has_value())
 12119	    {
 12120	        MarkExperimentFailed(w, experiment, "infer_missing_last_model_id", -1);
 12121	        return;
 12122	    }
 12123
 12124	    if (exitCode == 0)
 12125	    {
 12126	        if (HasCompletedAnalysisResult(w, experiment))
 12127	        {
 12128	            std::cout << "SCHEDULER_SKIP_EXISTING_ANALYSIS"
 12129	                      << ",experiment_id=" << experiment.experimentId
 12130	                      << ",model_id=" << *experiment.lastModelId
 12131	                      << std::endl;
 12132	            MarkExperimentDone(w, experiment, "infer");
 12133	        }
 12134	        else
 12135	        {
 12136	            MarkExperimentPendingPhase(w, experiment, "infer", "analyze", exitCode);
 12137	            std::cout << "SCHEDULER_ENQUEUE_ANALYZE"
 12138	                      << ",experiment_id=" << experiment.experimentId
 12139	                      << ",model_id=" << *experiment.lastModelId
 12140	                      << std::endl;
 12141	        }
 12142	    }
 12143	    else
 12144	    {
 12145	        const std::string errorMessage =
 12146	            "infer_failed;exit_code=" + std::to_string(exitCode) +
 12147	            ";command=" + commandDisplay;
 12148	        MarkExperimentFailed(w, experiment, errorMessage, exitCode);
 12149	    }
 12150	}
 12151
 12152	const char* SchedulerSignalName(int signalNumber)
 12153	{
 12154	    switch (signalNumber)
 12155	    {
 12156	        case SIGHUP: return "SIGHUP";
 12157	        case SIGINT: return "SIGINT";
 12158	        case SIGQUIT: return "SIGQUIT";
 12159	        case SIGILL: return "SIGILL";
 12160	        case SIGABRT: return "SIGABRT";
 12161	        case SIGFPE: return "SIGFPE";
 12162	        case SIGKILL: return "SIGKILL";
 12163	        case SIGSEGV: return "SIGSEGV";
 12164	        case SIGPIPE: return "SIGPIPE";
 12165	        case SIGALRM: return "SIGALRM";
 12166	        case SIGTERM: return "SIGTERM";
 12167	#ifdef SIGBUS
 12168	        case SIGBUS: return "SIGBUS";
 12169	#endif
 12170	        default: return "UNKNOWN";
 12171	    }
 12172	}
 12173
 12174	std::string ObservedChildError(const SchedulerOwnedChild& child,
 12175	                               int exitCode,
 12176	                               const std::optional<int>& signalNumber,
 12177	                               bool coreDumped)
 12178	{
 12179	    std::ostringstream error;
 12180	    if (signalNumber.has_value())
 12181	    {
 12182	        error << "child_signal_" << *signalNumber
 12183	              << ";phase=" << child.phase
 12184	              << ";signal=" << *signalNumber
 12185	              << ";signal_name=" << SchedulerSignalName(*signalNumber)
 12186	              << ";core_dumped=" << (coreDumped ? 1 : 0);
 12187	    }
 12188	    else
 12189	    {
 12190	        error << "child_exit_code_" << exitCode
 12191	              << ";phase=" << child.phase
 12192	              << ";exit_code=" << exitCode;
 12193	    }
 12194	    return error.str();
 12195	}
 12196
 12197	void PersistObservedExitFields(pqxx::work& w,
 12198	                               const SchedulerOwnedChild& child,
 12199	                               int exitCode,
 12200	                               const std::string& error,
 12201	                               bool preserveSuccessState)
 12202	{
 12203	    w.exec_params(
 12204	        "UPDATE experiment SET exit_code = $1, error_message = $2, worker_pid = NULL, updated_at = now() "
 12205	        "WHERE experiment_id = $3 AND (worker_pid = $4 OR worker_pid IS NULL) "
 12206	        "AND status <> 'running';",
 12207	        exitCode,
 12208	        preserveSuccessState && exitCode == 0 ? std::optional<std::string>{} : std::optional<std::string>{error},
 12209	        child.experimentId,
 12210	        static_cast<int>(child.pid));
 12211	}
 12212
 12213	void PersistObservedExperimentChild(pqxx::work& w,
 12214	                                    const SchedulerOwnedChild& child,
 12215	                                    int exitCode,
 12216	                                    const std::optional<int>& signalNumber,
 12217	                                    bool coreDumped)
 12218	{
 12219	    pqxx::result rows = w.exec_params(
 12220	        "SELECT experiment_id, symbol, prediction_horizon, c_next_threshold, "
 12221	        "core_lr_mult, head_lr_mult, target_epochs, checkpoint_interval, "
 12222	        "train_start::text, train_end::text, infer_start::text, infer_end::text, "
 12223	        "last_model_id, resume_model_id, train_log_path, infer_log_path, analysis_log_path, "
 12224	        "status, phase, worker_pid "
 12225	        "FROM experiment WHERE experiment_id = $1;",
 12226	        child.experimentId);
 12227	    if (rows.empty())
 12228	        return;
 12229
 12230	    ExperimentRow experiment = RowToExperiment(rows[0]);
 12231	    const std::string status = rows[0][17].as<std::string>();
 12232	    const std::string phase = rows[0][18].as<std::string>();
 12233	    const std::optional<int> workerPid = rows[0][19].is_null()
 12234	        ? std::nullopt
 12235	        : std::optional<int>{rows[0][19].as<int>()};
 12236	    const std::string error = ObservedChildError(child, exitCode, signalNumber, coreDumped);
 12237
 12238	    if (status != "running" || phase != child.phase ||
 12239	        (workerPid.has_value() && *workerPid != child.pid))
 12240	    {
 12241	        PersistObservedExitFields(w, child, exitCode, error, true);
 12242	        std::cout << "SCHEDULER_CHILD_RESULT_ALREADY_PERSISTED"
 12243	                  << ",experiment_id=" << child.experimentId
 12244	                  << ",phase=" << child.phase
 12245	                  << ",pid=" << child.pid
 12246	                  << ",status=" << status
 12247	                  << ",database_phase=" << phase
 12248	                  << ",exit_code=" << exitCode
 12249	                  << std::endl;
 12250	        return;
 12251	    }
 12252
 12253	    if (child.phase == "analyze" &&
 12254	        HasCompletedAnalysisResultForAttempt(
 12255	            w, experiment, child.launchedEpoch))
 12256	    {
 12257	        MarkExperimentDone(w, experiment, "analyze");
 12258	        PersistObservedExitFields(w, child, exitCode, error, true);
 12259	        return;
 12260	    }
 12261
 12262	    if (child.phase == "infer")
 12263	    {
 12264	        if (HasCompletedInferenceResultForAttempt(
 12265	                w, experiment, child.launchedEpoch))
 12266	        {
 12267	            TransitionRecoveredInferenceToAnalyze(
 12268	                w, experiment, "SCHEDULER_CHILD_RESULT_PERSISTED", "completed_inference_eval_result");
 12269	            PersistObservedExitFields(w, child, exitCode, error, true);
 12270	            return;
 12271	        }
 12272	        if (HasValidInferenceLogPathForAttempt(
 12273	                experiment, child.launchedEpoch))
 12274	        {
 12275	            TransitionRecoveredInferenceToAnalyze(
 12276	                w, experiment, "SCHEDULER_CHILD_RESULT_PERSISTED", "valid_infer_log_path");
 12277	            PersistObservedExitFields(w, child, exitCode, error, true);
 12278	            return;
 12279	        }
 12280	        if (const std::optional<std::string> discoveredLog =
 12281	                DiscoverValidInferenceLogForAttempt(
 12282	                    experiment, child.launchedEpoch);
 12283	            discoveredLog.has_value())
 12284	        {
 12285	            TransitionRecoveredInferenceToAnalyze(
 12286	                w, experiment, "SCHEDULER_CHILD_RESULT_PERSISTED", "discovered_valid_infer_log", discoveredLog);
 12287	            PersistObservedExitFields(w, child, exitCode, error, true);
 12288	            return;
 12289	        }
 12290	    }
 12291
 12292	    if (child.phase == "train")
 12293	    {
 12294	        if (const std::optional<long long> modelId =
 12295	                FindLatestModelForExperimentSince(
 12296	                    w,
 12297	                    experiment.experimentId,
 12298	                    child.launchedEpoch);
 12299	            modelId.has_value() &&
 12300	            (!child.expectedModelId.has_value() || *modelId != *child.expectedModelId))
 12301	        {
 12302	            (void)TransitionAfterTrainModelAvailable(w, experiment, *modelId, exitCode, "train");
 12303	            PersistObservedExitFields(w, child, exitCode, error, true);
 12304	            return;
 12305	        }
 12306	    }
 12307
 12308	    MarkExperimentFailed(w, experiment, error, exitCode);
 12309	}
 12310
 12311	void AdvanceCheckpointEvalToAnalyze(pqxx::work& w,
 12312	                                    const CheckpointEvalRow& eval,
 12313	                                    long long inferenceResultId,
 12314	                                    bool hasInferCompletedAt);
 12315
 12316	void PersistObservedCheckpointChild(pqxx::work& w,
 12317	                                    const SchedulerOwnedChild& child,
 12318	                                    int exitCode,
 12319	                                    const std::optional<int>& signalNumber,
 12320	                                    bool coreDumped)
 12321	{
 12322	    if (!child.checkpointEvalId.has_value())
 12323	        return;
 12324	    const std::string error = ObservedChildError(child, exitCode, signalNumber, coreDumped);
 12325	    CheckpointEvalRow eval;
 12326	    eval.checkpointEvalId = *child.checkpointEvalId;
 12327	    eval.experiment.experimentId = child.experimentId;
 12328	    eval.checkpointModelId = child.expectedModelId.value_or(-1);
 12329	    const std::optional<long long> resultId =
 12330	        FindCompletedCheckpointInferenceResultIdForAttempt(
 12331	            w, eval, child.launchedEpoch);
 12332
 12333	    if (resultId.has_value())
 12334	    {
 12335	        const bool hasInferCompletedAt =
 12336	            ColumnExists(w, "experiment_checkpoint_eval", "infer_completed_at");
 12337	        AdvanceCheckpointEvalToAnalyze(w, eval, *resultId, hasInferCompletedAt);
 12338	        if (exitCode != 0)
 12339	        {
 12340	            w.exec_params(
 12480	    logState->currentRunningPresentKeys.clear();
 12481	}
 12482
 12483	void FinishSchedulerPollLogging(SchedulerEventLogState* logState)
 12484	{
 12485	    if (logState == nullptr)
 12486	        return;
 12487	    logState->previousSkipKeys = logState->currentSkipKeys;
 12488	    logState->previousRunningPresentKeys = logState->currentRunningPresentKeys;
 12489	}
 12490
 12491	int RunTrainJobs(const SchedulerOptions& options,
 12492	                 const QueueSnapshot& snapshot,
 12493	                 SchedulerEventLogState* logState)
 12494	{
 12495	    pqxx::connection c{LstmDbConnectionString()};
 12496	    pqxx::work w{c};
 12497	    if (!options.dryRun)
 12498	        SetTransactionReadWrite(w);
 12499	    if (!RequireSchedulerTables(w))
 12500	        return 1;
 12501
 12502	    const std::vector<ExperimentRow> jobs = LoadPendingExperiments(w, "train");
 12503	    PhaseSchedulingStats stats;
 12504	    stats.phase = "train";
 12505	    stats.examined = static_cast<int>(jobs.size());
 12506	    stats.freeSlots = std::max(0, options.maxTrainProcs - RunningCountForPhase(snapshot, "train"));
 12507	    int freeSlots = stats.freeSlots;
 12508	    int rc = 0;
 12509
 12510	    EnsureLogDir(options.schedulerLogDir);
 12511	    for (const auto& job : jobs)
 12512	    {
 12513	        const std::optional<long long> resumeFrom =
 12514	            job.resumeModelId.has_value() ? job.resumeModelId : job.lastModelId;
 12515	        if (resumeFrom.has_value() && !ModelExists(w, *resumeFrom))
 12516	        {
 12517	            ++stats.skipped;
 12518	            LogSkip("train", job.experimentId, "model_not_found", logState, options.schedulerVerbose);
 12519	            if (!options.dryRun)
 12520	                MarkExperimentFailed(w, job, "train_model_not_found");
 12521	            continue;
 12522	        }
 12523	        if (RunningProcessExistsForExperiment(job, "train"))
 12524	        {
 12525	            ++stats.skipped;
 12526	            LogSkip("train", job.experimentId, "already_running", logState, options.schedulerVerbose);
 12527	            if (!options.dryRun)
 12528	            {
 12529	                MarkExperimentRunning(w, job, "train", LogPathFor(options, job, "train"));
 12530	                if (freeSlots > 0)
 12531	                    --freeSlots;
 12532	            }
 12533	            continue;
 12534	        }
 12535	        if (freeSlots <= 0)
 12536	        {
 12537	            ++stats.skipped;
 12538	            LogSkip("train", job.experimentId, "train_slots_full", logState, options.schedulerVerbose);
 12539	            continue;
 12540	        }
 12541
 12542	        const std::string logPath = LogPathFor(options, job, "train");
 12543	        const std::vector<std::string> command = BuildTrainCommand(options, job);
 12544	        const std::string commandDisplay = CommandForDisplay(command);
 12545	        if (options.dryRun)
 12546	        {
 12547	            std::cout << "EXPERIMENT_CHILD_COMMAND"
 12548	                      << ",experiment_id=" << job.experimentId
 12549	                      << ",phase=train"
 12550	                      << ",dry_run=1"
 12551	                      << ",argv=" << commandDisplay
 12552	                      << std::endl;
 12553	            --freeSlots;
 12554	            ++stats.launched;
 12555	            continue;
 12556	        }
 12557
 12558	        PersistExperimentRunningBeforeLaunch(job, "train", logPath);
 12559	        if (job.lastModelId.has_value() && !job.resumeModelId.has_value())
 12560	        {
 12561	            std::cout << "SCHEDULER_RESUME_FROM_LAST_MODEL"
 12562	                      << ",experiment_id=" << job.experimentId
 12563	                      << ",model_id=" << *job.lastModelId
 12564	                      << std::endl;
 12565	        }
 12566	        std::cout << "EXPERIMENT_STARTED"
 12567	                  << ",experiment_id=" << job.experimentId
 12568	                  << ",phase=train"
 12569	                  << std::endl;
 12570	        std::cout << "EXPERIMENT_CHILD_COMMAND"
 12571	                  << ",experiment_id=" << job.experimentId
 12572	                  << ",phase=train"
 12573	                  << ",argv=" << commandDisplay
 12574	                  << std::endl;
 12575	        PrintSchedulerExec(command);
 12576	        try
 12577	        {
 12578	            const pid_t pid = LaunchChildProcess(
 12579	                command, logPath, job.experimentId, "train", job.lastModelId);
 12580	            try
 12581	            {
 12582	                PersistExperimentWorkerPid(job, "train", pid);
 12583	            }
 12584	            catch (const std::exception& e)
 12585	            {
 12586	                std::cerr << "SCHEDULER_CHILD_PID_PERSIST_FAILED"
 12587	                          << ",experiment_id=" << job.experimentId
 12588	                          << ",worker_pid=" << pid
 12589	                          << ",phase=train"
 12590	                          << ",operation=train"
 12591	                          << ",error=" << e.what()
 12592	                          << std::endl;
 12593	                --freeSlots;
 12594	                ++stats.launched;
 12595	                rc = 1;
 12596	                continue;
 12597	            }
 12598	            --freeSlots;
 12599	            ++stats.launched;
 12600	        }
 12601	        catch (const std::exception& e)
 12602	        {
 12603	            const int launchError = SchedulerChildLaunchErrorNumber(e);
 12604	            MarkExperimentFailed(
 12605	                w, job, SchedulerLaunchFailureError(launchError));
 12606	            std::cout << "EXPERIMENT_FAILED"
 12607	                      << ",experiment_id=" << job.experimentId
 12608	                      << ",phase=train"
 12609	                      << ",error=train_launch_failed"
 12610	                      << std::endl;
 12611	            rc = 1;
 12612	        }
 12613	    }
 12614	    PrintPhaseSchedulingStats(stats, logState, options.schedulerVerbose);
 12615	    w.commit();
 12616	    return rc;
 12617	}
 12618
 12619	int RunInferJobs(const SchedulerOptions& options,
 12620	                 const QueueSnapshot& snapshot,
 12621	                 SchedulerEventLogState* logState)
 12622	{
 12623	    pqxx::connection c{LstmDbConnectionString()};
 12624	    pqxx::work w{c};
 12625	    if (!options.dryRun)
 12626	        SetTransactionReadWrite(w);
 12627	    if (!RequireSchedulerTables(w))
 12628	        return 1;
 12629
 12630	    const std::vector<ExperimentRow> jobs = LoadPendingExperiments(w, "infer");
 12631	    PhaseSchedulingStats stats;
 12632	    stats.phase = "infer";
 12633	    stats.examined = static_cast<int>(jobs.size());
 12634	    stats.freeSlots = std::max(0, options.maxInferProcs - RunningCountForPhase(snapshot, "infer"));
 12635	    int freeSlots = stats.freeSlots;
 12636	    int rc = 0;
 12637
 12638	    EnsureLogDir(options.schedulerLogDir);
 12639	    for (const auto& job : jobs)
 12640	    {
 12641	        if (!job.lastModelId.has_value())
 12642	        {
 12643	            ++stats.skipped;
 12644	            LogSkip("infer", job.experimentId, "missing_model", logState, options.schedulerVerbose);
 12645	            if (!options.dryRun)
 12646	                MarkExperimentFailed(w, job, "infer_missing_last_model_id");
 12647	            continue;
 12648	        }
 12649	        if (!ModelExists(w, *job.lastModelId))
 12650	        {
 12651	            ++stats.skipped;
 12652	            LogSkip("infer", job.experimentId, "model_not_found", logState, options.schedulerVerbose);
 12653	            if (!options.dryRun)
 12654	                MarkExperimentFailed(w, job, "infer_model_not_found");
 12655	            continue;
 12656	        }
 12657	        if (HasCompletedInferenceResult(w, job))
 12658	        {
 12659	            ++stats.skipped;
 12660	            LogSkip("infer", job.experimentId, "existing_inference", logState, options.schedulerVerbose);
 12661	            if (!options.dryRun)
 12662	            {
 12663	                std::cout << "SCHEDULER_SKIP_EXISTING_INFERENCE"
 12664	                          << ",experiment_id=" << job.experimentId
 12665	                          << ",model_id=" << *job.lastModelId
 12666	                          << std::endl;
 12667	                if (HasCompletedAnalysisResult(w, job))
 12668	                {
 12669	                    std::cout << "SCHEDULER_SKIP_EXISTING_ANALYSIS"
 12670	                              << ",experiment_id=" << job.experimentId
 12671	                              << ",model_id=" << *job.lastModelId
 12672	                              << std::endl;
 12673	                    MarkExperimentDone(w, job, "infer");
 12674	                }
 12675	                else
 12676	                {
 12677	                    MarkExperimentPendingPhase(w, job, "infer", "analyze");
 12678	                    std::cout << "SCHEDULER_ENQUEUE_ANALYZE"
 12679	                              << ",experiment_id=" << job.experimentId
 12680	                              << ",model_id=" << *job.lastModelId
 12681	                              << std::endl;
 12682	                }
 12683	            }
 12684	            continue;
 12685	        }
 12686	        if (HasValidInferenceLogPath(job))
 12687	        {
 12688	            ++stats.skipped;
 12689	            LogSkip("infer", job.experimentId, "valid_infer_log_path", logState, options.schedulerVerbose);
 12690	            if (!options.dryRun)
 12691	            {
 12692	                TransitionRecoveredInferenceToAnalyze(
 12693	                    w,
 12694	                    job,
 12695	                    "SCHEDULER_ORPHAN_RECOVERED_INFER_LOG",
 12696	                    "valid_infer_log_path");
 12697	            }
 12698	            continue;
 12699	        }
 12700	        if (const std::optional<std::string> discoveredLog = DiscoverValidInferenceLog(job);
 12701	            discoveredLog.has_value())
 12702	        {
 12703	            ++stats.skipped;
 12704	            LogSkip("infer", job.experimentId, "discovered_valid_infer_log", logState, options.schedulerVerbose);
 12705	            if (!options.dryRun)
 12706	            {
 12707	                TransitionRecoveredInferenceToAnalyze(
 12708	                    w,
 12709	                    job,
 12710	                    "SCHEDULER_ORPHAN_RECOVERED_INFER_LOG",
 12711	                    "discovered_valid_infer_log",
 12712	                    discoveredLog);
 12713	            }
 12714	            continue;
 12715	        }
 12716	        if (RunningProcessExistsForExperiment(job, "infer"))
 12717	        {
 12718	            ++stats.skipped;
 12719	            LogSkip("infer", job.experimentId, "already_running", logState, options.schedulerVerbose);
 12720	            if (!options.dryRun)
 12721	            {
 12722	                MarkExperimentRunning(w, job, "infer", LogPathFor(options, job, "infer"));
 12723	                if (freeSlots > 0)
 12724	                    --freeSlots;
 12725	            }
 12726	            continue;
 12727	        }
 12728	        if (freeSlots <= 0)
 12729	        {
 12730	            ++stats.skipped;
 12731	            LogSkip("infer", job.experimentId, "infer_slots_full", logState, options.schedulerVerbose);
 12732	            continue;
 12733	        }
 12734
 12735	        const std::string logPath = LogPathFor(options, job, "infer");
 12736	        const std::vector<std::string> command = BuildInferCommand(options, job);
 12737	        const std::string commandDisplay = CommandForDisplay(command);
 12738	        if (options.dryRun)
 12739	        {
 12740	            std::cout << "EXPERIMENT_CHILD_COMMAND"
 12741	                      << ",experiment_id=" << job.experimentId
 12742	                      << ",phase=infer"
 12743	                      << ",dry_run=1"
 12744	                      << ",argv=" << commandDisplay
 12745	                      << std::endl;
 12746	            --freeSlots;
 12747	            ++stats.launched;
 12748	            continue;
 12749	        }
 12750
 12751	        PersistExperimentRunningBeforeLaunch(job, "infer", logPath);
 12752	        std::cout << "EXPERIMENT_STARTED"
 12753	                  << ",experiment_id=" << job.experimentId
 12754	                  << ",phase=infer"
 12755	                  << std::endl;
 12756	        std::cout << "EXPERIMENT_CHILD_COMMAND"
 12757	                  << ",experiment_id=" << job.experimentId
 12758	                  << ",phase=infer"
 12759	                  << ",argv=" << commandDisplay
 12760	                  << std::endl;
 12761	        PrintSchedulerExec(command);
 12762	        try
 12763	        {
 12764	            const pid_t pid = LaunchChildProcess(
 12765	                command, logPath, job.experimentId, "infer", job.lastModelId);
 12766	            try
 12767	            {
 12768	                PersistExperimentWorkerPid(job, "infer", pid);
 12769	            }
 12770	            catch (const std::exception& e)
 12771	            {
 12772	                std::cerr << "SCHEDULER_CHILD_PID_PERSIST_FAILED"
 12773	                          << ",experiment_id=" << job.experimentId
 12774	                          << ",worker_pid=" << pid
 12775	                          << ",phase=infer"
 12776	                          << ",operation=infer"
 12777	                          << ",error=" << e.what()
 12778	                          << std::endl;
 12779	                --freeSlots;
 12780	                ++stats.launched;
 12781	                rc = 1;
 12782	                continue;
 12783	            }
 12784	            --freeSlots;
 12785	            ++stats.launched;
 12786	        }
 12787	        catch (const std::exception& e)
 12788	        {
 12789	            const int launchError = SchedulerChildLaunchErrorNumber(e);
 12790	            MarkExperimentFailed(
 12791	                w, job, SchedulerLaunchFailureError(launchError));
 12792	            std::cout << "EXPERIMENT_FAILED"
 12793	                      << ",experiment_id=" << job.experimentId
 12794	                      << ",phase=infer"
 12795	                      << ",error=infer_launch_failed"
 12796	                      << std::endl;
 12797	            rc = 1;
 12798	        }
 12799	    }
 12800	    PrintPhaseSchedulingStats(stats, logState, options.schedulerVerbose);
 12801	    w.commit();
 12802	    return rc;
 12803	}
 12804
 12805	int RunAnalyzeJobs(const SchedulerOptions& options,
 12806	                   const QueueSnapshot& snapshot,
 12807	                   SchedulerEventLogState* logState)
 12808	{
 12809	    pqxx::connection c{LstmDbConnectionString()};
 12810	    pqxx::work w{c};
 12811	    if (!options.dryRun)
 12812	        SetTransactionReadWrite(w);
 12813	    if (!RequireSchedulerTables(w))
 12814	        return 1;
 12815
 12816	    const std::vector<ExperimentRow> jobs = LoadPendingExperiments(w, "analyze");
 12817	    PhaseSchedulingStats stats;
 12818	    stats.phase = "analyze";
 12819	    stats.examined = static_cast<int>(jobs.size());
 12820	    stats.freeSlots = std::max(0, options.maxAnalyzeProcs - RunningCountForPhase(snapshot, "analyze"));
 12821	    int freeSlots = stats.freeSlots;
 12822	    int rc = 0;
 12823	    bool reportsNeeded = false;
 12824
 12825	    EnsureLogDir(options.schedulerLogDir);
 12826	    for (const auto& job : jobs)
 12827	    {
 12828	        if (!job.lastModelId.has_value())
 12829	        {
 12830	            ++stats.skipped;
 12831	            LogSkip("analyze", job.experimentId, "missing_model", logState, options.schedulerVerbose);
 12832	            if (!options.dryRun)
 12833	                MarkAnalyzeFailed(w, job.experimentId, "analyze_missing_last_model_id");
 12834	            continue;
 12835	        }
 12836	        if (!ModelExists(w, *job.lastModelId))
 12837	        {
 12838	            ++stats.skipped;
 12839	            LogSkip("analyze", job.experimentId, "model_not_found", logState, options.schedulerVerbose);
 12840	            if (!options.dryRun)
 12841	                MarkAnalyzeFailed(w, job.experimentId, "analyze_model_not_found");
 12842	            continue;
 12843	        }
 12844	        if (HasCompletedAnalysisResult(w, job))
 12845	        {
 12846	            ++stats.skipped;
 12847	            LogSkip("analyze", job.experimentId, "existing_analysis", logState, options.schedulerVerbose);
 12848	            if (!options.dryRun)
 12849	            {
 12850	                std::cout << "SCHEDULER_SKIP_EXISTING_ANALYSIS"
 12851	                          << ",experiment_id=" << job.experimentId
 12852	                          << ",model_id=" << *job.lastModelId
 12853	                          << std::endl;
 12854	                MarkExperimentDone(w, job, "analyze");
 12855	                reportsNeeded = true;
 12856	            }
 12857	            continue;
 12858	        }
 12859	        if (!HasCompletedInferenceResult(w, job) && !HasValidInferenceLogPath(job))
 12860	        {
 12861	            ++stats.skipped;
 12862	            LogSkip("analyze", job.experimentId, "infer_log_missing", logState, options.schedulerVerbose);
 12863	            continue;
 12864	        }
 12865	        if (RunningProcessExistsForExperiment(job, "analyze"))
 12866	        {
 12867	            ++stats.skipped;
 12868	            LogSkip("analyze", job.experimentId, "already_running", logState, options.schedulerVerbose);
 12869	            if (!options.dryRun)
 12870	            {
 12871	                MarkExperimentRunning(w, job, "analyze", LogPathFor(options, job, "analysis"));
 12872	                if (freeSlots > 0)
 12873	                    --freeSlots;
 12874	            }
 12875	            continue;
 12876	        }
 12877	        if (freeSlots <= 0)
 12878	        {
 12879	            ++stats.skipped;
 12880	            LogSkip("analyze", job.experimentId, "analyze_slots_full", logState, options.schedulerVerbose);
 12881	            continue;
 12882	        }
 12883
 12884	        const std::string logPath = LogPathFor(options, job, "analysis");
 12885	        const std::vector<std::string> command = BuildAnalyzeCommand(options, job);
 12886	        const std::string commandDisplay = CommandForDisplay(command);
 12887	        if (options.dryRun)
 12888	        {
 12889	            std::cout << "EXPERIMENT_ANALYSIS_STARTED"
 12890	                      << ",experiment_id=" << job.experimentId
 12891	                      << ",dry_run=1"
 12892	                      << std::endl;
 12893	            std::cout << "EXPERIMENT_CHILD_COMMAND"
 12894	                      << ",experiment_id=" << job.experimentId
 12895	                      << ",phase=analyze"
 12896	                      << ",dry_run=1"
 12897	                      << ",argv=" << commandDisplay
 12898	                      << std::endl;
 12899	            --freeSlots;
 12900	            ++stats.launched;
 12901	            continue;
 12902	        }
 12903
 12904	        PersistExperimentRunningBeforeLaunch(job, "analyze", logPath);
 12905	        std::cout << "EXPERIMENT_STARTED"
 12906	                  << ",experiment_id=" << job.experimentId
 12907	                  << ",phase=analyze"
 12908	                  << std::endl;
 12909	        std::cout << "EXPERIMENT_CHILD_COMMAND"
 12910	                  << ",experiment_id=" << job.experimentId
 12911	                  << ",phase=analyze"
 12912	                  << ",argv=" << commandDisplay
 12913	                  << std::endl;
 12914	        PrintSchedulerExec(command);
 12915	        try
 12916	        {
 12917	            const pid_t pid = LaunchChildProcess(
 12918	                command, logPath, job.experimentId, "analyze", job.lastModelId);
 12919	            try
 12920	            {
 12921	                PersistExperimentWorkerPid(job, "analyze", pid);
 12922	            }
 12923	            catch (const std::exception& e)
 12924	            {
 12925	                std::cerr << "SCHEDULER_CHILD_PID_PERSIST_FAILED"
 12926	                          << ",experiment_id=" << job.experimentId
 12927	                          << ",worker_pid=" << pid
 12928	                          << ",phase=analyze"
 12929	                          << ",operation=analyze"
 12930	                          << ",error=" << e.what()
 12931	                          << std::endl;
 12932	                --freeSlots;
 12933	                ++stats.launched;
 12934	                rc = 1;
 12935	                continue;
 12936	            }
 12937	            --freeSlots;
 12938	            ++stats.launched;
 12939	        }
 12940	        catch (const std::exception& e)
 12941	        {
 12942	            const int launchError = SchedulerChildLaunchErrorNumber(e);
 12943	            MarkAnalyzeFailed(
 12944	                w, job.experimentId, SchedulerLaunchFailureError(launchError));
 12945	            std::cout << "EXPERIMENT_FAILED"
 12946	                      << ",experiment_id=" << job.experimentId
 12947	                      << ",phase=analyze"
 12948	                      << ",error=analyze_launch_failed"
 12949	                      << std::endl;
 12950	            rc = 1;
 12951	        }
 12952	    }
 12953	    PrintPhaseSchedulingStats(stats, logState, options.schedulerVerbose);
 12954	    w.commit();
 12955	    if (reportsNeeded)
 12956	        TryGenerateExperimentReports(options);
 12957	    return rc;
 12958	}
 12959
 12960	bool SchedulerPidStillExists(std::optional<int> pid)
 12961	{
 12962	    if (!pid.has_value() || *pid <= 0)
 12963	        return false;
 12964	    if (::kill(static_cast<pid_t>(*pid), 0) == 0)
 12965	        return true;
 12966	    return errno == EPERM;
 12967	}
 12968
 12969	int CountRows(pqxx::work& w, const std::string& sql)
 12970	{
 12971	    return w.exec(sql).one_row()[0].as<int>();
 12972	}
 12973
 12974	void LogCheckpointEvalInferenceResultFound(const CheckpointEvalRow& eval,
 12975	                                           long long inferenceResultId)
 12976	{
 12977	    std::cout << "CHECKPOINT_EVAL_INFER_RESULT_FOUND"
 12978	              << ",checkpoint_eval_id=" << eval.checkpointEvalId
 12979	              << ",parent_experiment_id=" << eval.experiment.experimentId
 12980	              << ",checkpoint_model_id=" << eval.checkpointModelId
 12981	              << ",checkpoint_epoch=" << eval.checkpointEpoch
 12982	              << ",model_id=" << eval.checkpointModelId
 12983	              << ",inference_scope=checkpoint"
 12984	              << ",status=completed"
 12985	              << ",inference_result_id=" << inferenceResultId
 12986	              << std::endl;
 12987	}
 12988
 12989	void AdvanceCheckpointEvalToAnalyze(pqxx::work& w,
 12990	                                    const CheckpointEvalRow& eval,
 12991	                                    long long inferenceResultId,
 12992	                                    bool hasInferCompletedAt)
 12993	{
 12994	    LogCheckpointEvalInferenceResultFound(eval, inferenceResultId);
 12995	    std::ostringstream sql;
 12996	    sql << "UPDATE experiment_checkpoint_eval "
 12997	        << "SET status = 'pending', phase = 'analyze', worker_pid = NULL, "
 12998	        << "completed_at = NULL, error_message = NULL, updated_at = now()";
 12999	    if (hasInferCompletedAt)
 13000	        sql << ", infer_completed_at = COALESCE(infer_completed_at, now())";
 13001	    sql << " WHERE checkpoint_eval_id = $1;";
 13002	    w.exec_params(sql.str(), eval.checkpointEvalId);
 13003
 13004	    std::cout << "CHECKPOINT_EVAL_ADVANCED_TO_ANALYZE"
 13005	              << ",checkpoint_eval_id=" << eval.checkpointEvalId
 13006	              << ",parent_experiment_id=" << eval.experiment.experimentId
 13007	              << ",checkpoint_model_id=" << eval.checkpointModelId
 13008	              << ",checkpoint_epoch=" << eval.checkpointEpoch
 13009	              << ",model_id=" << eval.checkpointModelId
 13010	              << ",inference_scope=checkpoint"
 13011	              << ",status=pending"
 13012	              << ",inference_result_id=" << inferenceResultId
 13013	              << std::endl;
 13014	}
 13015
 13016	void LogCheckpointEvalInferenceResultMissing(const CheckpointEvalRow& eval,
 13017	                                             const std::string& error)
 13018	{
 13019	    std::cout << "CHECKPOINT_EVAL_INFER_RESULT_MISSING"
 13020	              << ",checkpoint_eval_id=" << eval.checkpointEvalId
 13021	              << ",parent_experiment_id=" << eval.experiment.experimentId
 13022	              << ",checkpoint_model_id=" << eval.checkpointModelId
 13023	              << ",checkpoint_epoch=" << eval.checkpointEpoch
 13024	              << ",model_id=" << eval.checkpointModelId
 13025	              << ",inference_scope=checkpoint"
 13026	              << ",status=missing"
 13027	              << ",error=" << error
 13028	              << std::endl;
 13029	}
 13030
 13031	int RunCheckpointEvalInferJobs(const SchedulerOptions& options)
 13032	{
 13033	    pqxx::connection c{LstmDbConnectionString()};
 13034	    pqxx::work w{c};
 13035	    if (!options.dryRun)
 13036	        SetTransactionReadWrite(w);
 13037	    if (!CheckpointEvalTableExists(w))
 13038	        return 0;
 13039	    const bool hasInferStartedAt = ColumnExists(w, "experiment_checkpoint_eval", "infer_started_at");
 13040	    const bool hasInferCompletedAt = ColumnExists(w, "experiment_checkpoint_eval", "infer_completed_at");
 13041
 13042	    int rc = 0;
 13043	    for (const auto& eval : LoadCheckpointEvalRows(w, "running", "infer"))
 13044	    {
 13045	        if (SchedulerPidStillExists(eval.workerPid))
 13046	            continue;
 13047	        const std::optional<long long> inferenceResultId =
 13048	            FindCompletedCheckpointInferenceResultIdForAttempt(
 13049	                w, eval, eval.inferStartedEpoch);
 13050	        if (inferenceResultId.has_value())
 13051	        {
 13052	            AdvanceCheckpointEvalToAnalyze(w,
 13053	                                           eval,
 13054	                                           *inferenceResultId,
 13055	                                           hasInferCompletedAt);
 13056	        }
 13057	        else
 13058	        {
 13059	            LogCheckpointEvalInferenceResultMissing(
 13060	                eval,
 13061	                "infer_worker_exited_without_completed_result");
 13062	            w.exec_params(
 13063	                "UPDATE experiment_checkpoint_eval "
 13064	                "SET status = 'failed', worker_pid = NULL, completed_at = now(), updated_at = now(), "
 13065	                "error_message = 'infer_worker_exited_without_completed_result' "
 13066	                "WHERE checkpoint_eval_id = $1;",
 13067	                eval.checkpointEvalId);
 13068	            std::cout << "CHECKPOINT_EVAL_FAILED"
 13069	                      << " experiment_id=" << eval.experiment.experimentId
 13070	                      << " phase=infer"
 13071	                      << " epoch=" << eval.checkpointEpoch
 13072	                      << " model_id=" << eval.checkpointModelId
 13073	                      << " error=infer_worker_exited_without_completed_result"
 13074	                      << std::endl;
 13075	        }
 13076	    }
 13077
 13078	    const int pendingFinalInfer = CountRows(w, "SELECT count(*) FROM experiment WHERE status = 'pending' AND phase = 'infer';");
 13079	    if (pendingFinalInfer > 0)
 13080	    {
 13081	        w.commit();
 13082	        return 0;
 13083	    }
 13084
 13085	    const int runningFinalInfer = CountRows(w, "SELECT count(*) FROM experiment WHERE status = 'running' AND phase = 'infer';");
 13086	    const int runningCheckpointInfer = CountRows(w, "SELECT count(*) FROM experiment_checkpoint_eval WHERE status = 'running' AND phase = 'infer';");
 13087	    int freeSlots = std::max(0, options.maxInferProcs - runningFinalInfer - runningCheckpointInfer);
 13088	    if (freeSlots <= 0)
 13089	    {
 13090	        w.commit();
 13091	        return 0;
 13092	    }
 13093
 13094	    EnsureLogDir(options.schedulerLogDir);
 13095	    for (const auto& eval : LoadCheckpointEvalRows(w, "pending", "infer"))
 13096	    {
 13097	        if (freeSlots <= 0)
 13098	            break;
 13099	        if (!eval.experiment.inferStart.has_value() || !eval.experiment.inferEnd.has_value())
 13100	        {
 13101	            w.exec_params(
 13102	                "UPDATE experiment_checkpoint_eval "
 13103	                "SET status = 'skipped', completed_at = now(), updated_at = now(), error_message = 'missing_infer_range' "
 13104	                "WHERE checkpoint_eval_id = $1;",
 13105	                eval.checkpointEvalId);
 13106	            std::cout << "CHECKPOINT_EVAL_SKIPPED"
 13107	                      << " experiment_id=" << eval.experiment.experimentId
 13108	                      << " epoch=" << eval.checkpointEpoch
 13109	                      << " reason=missing_infer_range"
 13110	                      << std::endl;
 13111	            continue;
 13112	        }
 13113	        const std::optional<long long> inferenceResultId =
 13114	            FindCompletedCheckpointInferenceResultId(w, eval);
 13115	        if (inferenceResultId.has_value())
 13116	        {
 13117	            AdvanceCheckpointEvalToAnalyze(w,
 13118	                                           eval,
 13119	                                           *inferenceResultId,
 13120	                                           hasInferCompletedAt);
 13121	            continue;
 13122	        }
 13123
 13124	        const std::string logPath = CheckpointEvalLogPathFor(options, eval, "infer");
 13125	        const std::vector<std::string> command = BuildCheckpointEvalInferCommand(options, eval);
 13126	        const std::string commandDisplay = CommandForDisplay(command);
 13127	        if (options.dryRun)
 13128	        {
 13129	            std::cout << "CHECKPOINT_EVAL_STARTED"
 13130	                      << " experiment_id=" << eval.experiment.experimentId
 13131	                      << " phase=infer"
 13132	                      << " epoch=" << eval.checkpointEpoch
 13133	                      << " model_id=" << eval.checkpointModelId
 13134	                      << " dry_run=1"
 13135	                      << std::endl;
 13136	            std::cout << "EXPERIMENT_CHILD_COMMAND"
 13137	                      << ",experiment_id=" << eval.experiment.experimentId
 13138	                      << ",phase=checkpoint_infer"
 13139	                      << ",dry_run=1"
 13140	                      << ",argv=" << commandDisplay
 13141	                      << std::endl;
 13142	            --freeSlots;
 13143	            continue;
 13144	        }
 13145
 13146	        pid_t pid = -1;
 13147	        try
 13148	        {
 13149	            pid = LaunchChildProcess(
 13150	                command,
 13151	                logPath,
 13152	                eval.experiment.experimentId,
 13153	                "checkpoint_infer",
 13154	                eval.checkpointModelId,
 13155	                eval.checkpointEvalId);
 13156	        }
 13157	        catch (const std::exception& e)
 13158	        {
 13159	            const int launchError = SchedulerChildLaunchErrorNumber(e);
 13160	            w.exec_params(
 13161	                "UPDATE experiment_checkpoint_eval "
 13162	                "SET status = 'failed', completed_at = now(), updated_at = now(), error_message = $1 "
 13163	                "WHERE checkpoint_eval_id = $2;",
 13164	                SchedulerLaunchFailureError(launchError),
 13165	                eval.checkpointEvalId);
 13166	            std::cout << "CHECKPOINT_EVAL_FAILED"
 13167	                      << " experiment_id=" << eval.experiment.experimentId
 13168	                      << " phase=infer"
 13169	                      << " epoch=" << eval.checkpointEpoch
 13170	                      << " model_id=" << eval.checkpointModelId
 13171	                      << " error=infer_launch_failed"
 13172	                      << std::endl;
 13173	            rc = 1;
 13174	            continue;
 13175	        }
 13176
 13177	        std::ostringstream sql;
 13178	        sql << "UPDATE experiment_checkpoint_eval "
 13179	            << "SET status = 'running', phase = 'infer', worker_pid = $1, infer_log_path = $2, "
 13180	            << "started_at = COALESCE(started_at, now()), updated_at = now(), error_message = NULL";
 13181	        if (hasInferStartedAt)
 13182	            sql << ", infer_started_at = COALESCE(infer_started_at, now())";
 13183	        sql << " WHERE checkpoint_eval_id = $3;";
 13184	        try
 13185	        {
 13186	            w.exec_params(sql.str(),
 13187	                          static_cast<int>(pid),
 13188	                          logPath,
 13189	                          eval.checkpointEvalId);
 13190	        }
 13191	        catch (const std::exception& e)
 13192	        {
 13193	            const int killResult = ::kill(pid, SIGTERM);
 13194	            std::cerr << "SCHEDULER_CHILD_PID_PERSIST_FAILED"
 13195	                      << ",experiment_id=" << eval.experiment.experimentId
 13196	                      << ",checkpoint_eval_id=" << eval.checkpointEvalId
 13197	                      << ",worker_pid=" << pid
 13198	                      << ",phase=checkpoint_infer"
 13199	                      << ",operation=checkpoint_infer"
 13200	                      << ",child_termination_requested="
 13201	                      << (killResult == 0 ? "1" : "0")
 13202	                      << ",error=" << e.what()
 13203	                      << std::endl;
 13204	            throw;
 13205	        }
 13206	        std::cout << "CHECKPOINT_EVAL_STARTED"
 13207	                  << " experiment_id=" << eval.experiment.experimentId
 13208	                  << " phase=infer"
 13209	                  << " epoch=" << eval.checkpointEpoch
 13210	                  << " model_id=" << eval.checkpointModelId
 13211	                  << std::endl;
 13212	        PrintSchedulerExec(command);
 13213	        --freeSlots;
 13214	    }
 13215	    w.commit();
 13216	    return rc;
 13217	}
 13218
 13219	int RunCheckpointEvalAnalyzeJobs(const SchedulerOptions& options)
 13220	{
 13221	    pqxx::connection c{LstmDbConnectionString()};
 13222	    pqxx::work w{c};
 13223	    SetTransactionReadWrite(w);
 13224	    if (!CheckpointEvalTableExists(w))
 13225	        return 0;
 13226	    const bool hasAnalyzeStartedAt = ColumnExists(w, "experiment_checkpoint_eval", "analyze_started_at");
 13227	    const bool hasAnalyzeCompletedAt = ColumnExists(w, "experiment_checkpoint_eval", "analyze_completed_at");
 13228	    const bool hasAnalysisId = ColumnExists(w, "experiment_checkpoint_eval", "analysis_id");
 13229
 13230	    const int pendingFinalAnalyze = CountRows(w, "SELECT count(*) FROM experiment WHERE status = 'pending' AND phase = 'analyze';");
 13231	    const int runningFinalAnalyze = CountRows(w, "SELECT count(*) FROM experiment WHERE status = 'running' AND phase = 'analyze';");
 13232	    if (pendingFinalAnalyze > 0 || runningFinalAnalyze >= options.maxAnalyzeProcs)
 13233	    {
 13234	        w.commit();
 13235	        return 0;
 13236	    }
 13237
 13238	    int freeSlots = std::max(0, options.maxAnalyzeProcs - runningFinalAnalyze);
 13239	    int rc = 0;
 13240	    bool reportsNeeded = false;
 13241	    for (auto eval : LoadCheckpointEvalRows(w, "pending", "analyze"))
 13242	    {
 13243	        if (freeSlots <= 0)
 13244	            break;
 13245	        LogWorkerStarted(
 13246	            "CHECKPOINT_ANALYSIS_WORKER_STARTED",
 13247	            "checkpoint_analyze",
 13248	            eval.experiment.experimentId,
 13249	            eval.checkpointModelId,
 13250	            eval.checkpointEvalId);
 13251	        std::cout << "CHECKPOINT_EVAL_ANALYSIS_STARTED"
 13252	                  << " experiment_id=" << eval.experiment.experimentId
 13253	                  << " epoch=" << eval.checkpointEpoch
 13254	                  << " model_id=" << eval.checkpointModelId
 13255	                  << std::endl;
 13256	        {
 13257	            std::ostringstream sql;
 13258	            sql << "UPDATE experiment_checkpoint_eval "
 13259	                << "SET status = 'running', phase = 'analyze', updated_at = now(), error_message = NULL";
 13260	            if (hasAnalyzeStartedAt)
 13261	                sql << ", analyze_started_at = COALESCE(analyze_started_at, now())";
 13262	            sql << " WHERE checkpoint_eval_id = $1;";
 13263	            w.exec_params(sql.str(), eval.checkpointEvalId);
 13264	        }
 13265	        const std::optional<long long> inferenceResultId =
 13266	            FindCompletedCheckpointInferenceResultId(w, eval);
 13267	        if (!inferenceResultId.has_value())
 13268	        {
 13269	            LogCheckpointEvalInferenceResultMissing(eval, "missing_completed_checkpoint_inference");
 13270	            std::ostringstream sql;
 13271	            sql << "UPDATE experiment_checkpoint_eval "
 13272	                << "SET status = 'failed', completed_at = now(), updated_at = now(), error_message = 'missing_completed_inference'";
 13273	            if (hasAnalyzeCompletedAt)
 13274	                sql << ", analyze_completed_at = now()";
 13275	            sql << " WHERE checkpoint_eval_id = $1;";
 13276	            w.exec_params(sql.str(), eval.checkpointEvalId);
 13277	            std::cout << "CHECKPOINT_EVAL_FAILED"
 13278	                      << " experiment_id=" << eval.experiment.experimentId
 13279	                      << " phase=analyze"
 13280	                      << " epoch=" << eval.checkpointEpoch
 13281	                      << " model_id=" << eval.checkpointModelId
 13282	                      << " error=missing_completed_inference"
 13283	                      << std::endl;
 13284	            rc = 1;
 13285	            continue;
 13286	        }
 13287
 13288	        ParsedMetrics metrics;
 13289	        metrics.modelId = eval.checkpointModelId;
 13290	        ApplyPersistedSymbolToAnalysisExperiment(w, eval.experiment, metrics);
 13291	        if (!ApplyStructuredCheckpointInferenceMetrics(w, eval, metrics))
 13292	            throw std::runtime_error("checkpoint inference result disappeared during analysis");
 13293	        const std::optional<double> leaderScore = ComputeLeaderScore(metrics);
 13294	        AnalysisScopeOptions analysisScope;
 13295	        analysisScope.scope = "checkpoint";
 13296	        analysisScope.checkpointEvalId = eval.checkpointEvalId;
 13297	        analysisScope.checkpointEpoch = eval.checkpointEpoch;
 13298	        analysisScope.parentExperimentId = eval.experiment.experimentId;
 13299	        UpsertAnalysisResult(w, eval.experiment, metrics, leaderScore, analysisScope);
 13300	        const std::optional<long long> analysisId = FindCheckpointAnalysisResultId(w, eval.checkpointEvalId);
 13301	        {
 13302	            std::ostringstream sql;
 13303	            sql << "UPDATE experiment_checkpoint_eval "
 13304	                << "SET status = 'completed', phase = 'done', completed_at = now(), updated_at = now(), error_message = NULL";
 13305	            if (hasAnalyzeCompletedAt)
 13306	                sql << ", analyze_completed_at = now()";
 13307	            if (hasAnalysisId)
 13308	                sql << ", analysis_id = " << (analysisId.has_value() ? std::to_string(*analysisId) : "NULL");
 13309	            sql << " WHERE checkpoint_eval_id = $1;";
 13310	            w.exec_params(sql.str(), eval.checkpointEvalId);
 13311	        }
 13312	        (void)EvaluateCheckpointPolicyAfterAnalysis(w, eval);
 13313	        std::cout << "CHECKPOINT_EVAL_ANALYSIS_COMPLETED"
 13314	                  << " experiment_id=" << eval.experiment.experimentId
 13315	                  << " epoch=" << eval.checkpointEpoch
 13316	                  << " model_id=" << eval.checkpointModelId
 13317	                  << std::endl;
 13318	        reportsNeeded = true;
 13319	        --freeSlots;
 13320	    }
 13321	    w.commit();
 13322	    if (reportsNeeded)
 13323	        TryGenerateExperimentReports(options);
 13324	    return rc;
 13325	}
 13326
 13327	int RunSchedulerOnce(const SchedulerOptions& options,
 13328	                     SchedulerEventLogState* logState)
 13329	{
 13330	    int rc = 0;
 13331	    QueueSnapshot snapshot;
 13332	    BeginSchedulerPollLogging(logState);
 13333	    {
 13334	        pqxx::connection c{LstmDbConnectionString()};
 13335	        pqxx::work w{c};
 13336	        if (!options.dryRun)
 13337	            SetTransactionReadWrite(w);
 13338	        if (!RequireSchedulerTables(w))
 13339	            return 1;
 13340	        if (!options.dryRun)
 13341	        {
 13342	            EA::RunMetadata::BackfillMissingExperimentRunMetadata(w, options.selfPath, "scheduler_start");
 13343	            ReapSchedulerOwnedChildren(w);
 13344	            RecoverOrphanedRunningExperiments(w, logState, options.schedulerVerbose);
 13345	            EnqueueCheckpointEvalRows(w);
 13346	            rc |= FailInvalidSchedulerPhases(w);
 13347	        }
 13348	        snapshot = LoadQueueSnapshot(w);
 13349	        PrintQueueSnapshot(snapshot, logState, options.schedulerVerbose);
 13350	        w.commit();
 13351	    }
 13352
 13353	    rc |= RunTrainJobs(options, snapshot, logState);
 13354	    rc |= RunInferJobs(options, snapshot, logState);
 13355	    rc |= RunAnalyzeJobs(options, snapshot, logState);
 13356	    rc |= RunCheckpointEvalInferJobs(options);
 13357	    rc |= RunCheckpointEvalAnalyzeJobs(options);
 13358	    FinishSchedulerPollLogging(logState);
 13359	    return rc;
 13360	}
 13361
 13362	int RunScheduler(const SchedulerOptions& options)
 13363	{
 13364	    int recoveryCount = 0;
 13365	    SchedulerEventLogState logState;
 13366	    {
 13367	        pqxx::connection c{LstmDbConnectionString()};
 13368	        pqxx::work w{c};
 13369	        SetTransactionReadWrite(w);
 13370	        if (!RequireSchedulerTables(w))
 13371	            return 1;
 13372	        if (!options.dryRun)
 13373	        {
 13374	            EA::RunMetadata::BackfillMissingExperimentRunMetadata(w, options.selfPath, "scheduler_start");
 13375	            BeginSchedulerPollLogging(&logState);
 13376	            recoveryCount = RecoverOrphanedRunningExperiments(w, &logState, options.schedulerVerbose);
 13377	            FinishSchedulerPollLogging(&logState);
 13378	            if (FailInvalidSchedulerPhases(w) != 0)
 13379	            {
 13380	                w.commit();
 13381	                return 1;
 13382	            }
 13383	        }
 13384	        w.commit();
 13385	    }
 13386
 13387	    std::cout << "SCHEDULER_START"
 13388	              << ",dry_run=" << (options.dryRun ? "1" : "0")
 13389	              << ",max_train_procs=" << options.maxTrainProcs
 13390	              << ",max_infer_procs=" << options.maxInferProcs
 13391	              << ",max_analyze_procs=" << options.maxAnalyzeProcs
 13392	              << ",auto_evaluate_continuations="
 13393	              << (options.autoEvaluateContinuations ? "1" : "0")
 13394	              << ",auto_queue_continuations="
 13395	              << (options.autoQueueContinuations ? "1" : "0")
 13396	              << ",continuation_scan_seconds=" << options.continuationScanSeconds
 13397	              << ",continuation_max_queues_per_scan="
 13398	              << options.continuationMaxQueuesPerScan
 13399	              << ",continuation_dry_run="
 13400	              << ((options.continuationDryRun || options.dryRun) ? "1" : "0")
 13401	              << ",recover_orphans_only=" << (options.recoverOrphansOnly ? "1" : "0")
 13402	              << std::endl;
 13403	    if (options.dryRun)
 13404	        std::cout << "SCHEDULER_DRY_RUN=1" << std::endl;
 13405	    if (options.recoverOrphansOnly)
 13406	    {
 13407	        std::cout << "SCHEDULER_ORPHAN_RECOVERY_DONE"
 13408	                  << ",recovered_or_failed=" << recoveryCount
 13409	                  << std::endl;
 13410	        std::cout << "SCHEDULER_STOP"
 13411	                  << ",exit_code=0"
 13412	                  << std::endl;
 13413	        return 0;
 13414	    }
 13415
 13416	    int rc = 0;
 13417	    ContinuationAutoScanState continuationScanState;
 13418	    do
 13419	    {
 13420	        rc |= RunSchedulerOnce(options, &logState);
 13421	        if (options.autoEvaluateContinuations &&
 13422	            std::chrono::steady_clock::now() >= continuationScanState.nextScan)
 13423	        {
 13424	            continuationScanState.lastCounts = RunAutomaticContinuationScan(options);
 13425	            continuationScanState.hasRun = true;
 13426	            continuationScanState.lastScanAt = EA::RunMetadata::CurrentUtcTimestamp();
 13427	            continuationScanState.nextScan =
 13428	                std::chrono::steady_clock::now() +
 13429	                std::chrono::seconds(options.continuationScanSeconds);
 13430	        }
 13431	        if (options.schedulerOnce)
 13432	            break;
 13433	        ::sleep(static_cast<unsigned int>(options.schedulerPollSeconds));
 13434	    } while (true);
 13435
 13436	    std::cout << "SCHEDULER_STOP"
 13437	              << ",exit_code=" << rc
 13438	              << std::endl;
 13439	    return rc;
 13440	}
 13441
 13442	int PrintLeaderboard(const SchedulerOptions& options)
 13443	{
 13444	    pqxx::connection c{LstmDbConnectionString()};
 13445	    pqxx::work w{c};
 13446	    if (!RequireSchedulerTables(w))
 13447	        return 1;
 13448
 13449	    std::ostringstream sql;
 13450	    sql << "SELECT e.experiment_id, a.model_id, a.symbol, a.prediction_horizon, "
 13451	        << "a.target_epochs, a.infer_accuracy, a.accept_accuracy, a.accept_rate, a.leader_score "
 13452	        << "FROM experiment_analysis_result a "
 13453	        << "JOIN experiment e ON e.experiment_id = a.experiment_id "
 13454	        << "WHERE COALESCE(a.analysis_scope, 'final') = 'final' ";
 13455	    if (options.leaderboardSymbol.has_value())
 13456	        sql << "AND a.symbol = " << w.quote(*options.leaderboardSymbol) << " ";
 13457	    if (options.leaderboardHorizon.has_value())
 13458	        sql << "AND a.prediction_horizon = " << *options.leaderboardHorizon << " ";
 13459	    sql << "ORDER BY a.leader_score DESC NULLS LAST LIMIT " << options.leaderboardLimit << ";";
 13460

codex
Two repository facts now materially constrain the review: Phase 6D is implemented at `d78771f`, but its governing ADR is still `Proposed`; and the accepted architecture does not establish that the ratification’s “next separately controlled phase” is Campaign Operations. A separate read-only recommendation artifact proposes a Phase 6E planning bridge, so “no meaningful Phase 6E remains” is not presently an authoritative repository fact.
exec
/bin/bash -lc "nl -ba docs/architecture/adr/README.md | sed -n '1,120p' && nl -ba docs/architecture/Volume_XII_Database.md | sed -n '1,180p' && nl -ba docs/architecture/Volume_XII_Database.md | sed -n '180,320p' && git show --stat --oneline --decorate d78771f && git show --format= --name-only d78771f | sort" in /Volumes/Developer SSD/ExpertAdvisor
 succeeded in 199ms:
d78771f (HEAD -> campaign-operations, phase6e, phase6d) Implement Phase 6D governance ratification
 Database/README.md                                 |   19 +
 ...on_campaign_follow_up_proposal_ratification.sql |  189 ++++
 ExpertAdvisor.xcodeproj/project.pbxproj            |   24 +
 ...ndationCampaignFollowUpProposalRatification.cpp |  486 ++++++++
 ...ndationCampaignFollowUpProposalRatification.hpp |  165 +++
 ...paignFollowUpProposalRatificationRepository.cpp |  356 ++++++
 ...paignFollowUpProposalRatificationRepository.hpp |  102 ++
 ...CampaignFollowUpProposalRatificationService.cpp |   88 ++
 ...CampaignFollowUpProposalRatificationService.hpp |   29 +
 ...nFollowUpProposalRatificationMigrationTests.sql |  237 ++++
 ...FollowUpProposalRatificationRepositoryTests.cpp | 1168 ++++++++++++++++++++
 ...onCampaignFollowUpProposalRatificationTests.cpp |  391 +++++++
 ...ndationCampaignFollowUpProposalRatification.rst |  138 +++
 .../Volume_VIII_Recommendation_Engine.md           |   86 +-
 docs/architecture/Volume_XII_Database.md           |   42 +-
 ...d-follow-up-proposal-governance-ratification.md |  155 +++
 docs/architecture/adr/README.md                    |    2 +
 17 files changed, 3667 insertions(+), 10 deletions(-)
Database/README.md
Database/migrations/044_experiment_recommendation_campaign_follow_up_proposal_ratification.sql
ExpertAdvisor.xcodeproj/project.pbxproj
Sources/ExperimentRecommendationCampaignFollowUpProposalRatification.cpp
Sources/ExperimentRecommendationCampaignFollowUpProposalRatification.hpp
Sources/ExperimentRecommendationCampaignFollowUpProposalRatificationRepository.cpp
Sources/ExperimentRecommendationCampaignFollowUpProposalRatificationRepository.hpp
Sources/ExperimentRecommendationCampaignFollowUpProposalRatificationService.cpp
Sources/ExperimentRecommendationCampaignFollowUpProposalRatificationService.hpp
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationMigrationTests.sql
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationTests.cpp
docs/Phase6DRecommendationCampaignFollowUpProposalRatification.rst
docs/architecture/Volume_VIII_Recommendation_Engine.md
docs/architecture/Volume_XII_Database.md
docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md
docs/architecture/adr/README.md

exec
/bin/bash -lc "nl -ba Sources/ExperimentRecommendationCampaignFollowUpProposalRatificationService.cpp && nl -ba Sources/ExperimentRecommendationCampaignFollowUpProposalRatificationRepository.hpp && nl -ba Database/migrations/044_experiment_recommendation_campaign_follow_up_proposal_ratification.sql && rg -n 'phase_6e|follow_up_author|execution_author|scheduler|worker|experiment|ratified|separation|existing_identical|advisory|FOR UPDATE|INSERT|UPDATE|DELETE|TRUNCATE|GRANT|REVOKE' Sources/ExperimentRecommendationCampaignFollowUpProposalRatification.cpp Sources/ExperimentRecommendationCampaignFollowUpProposalRatificationRepository.cpp Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationMigrationTests.sql | sed -n '1,800p'" in /Volumes/Developer SSD/ExpertAdvisor
 succeeded in 0ms:
     1	#include "ExperimentRecommendationCampaignFollowUpProposalRatificationService.hpp"
     2
     3	#include "ExperimentRecommendationCampaignFollowUpProposalReviewRepository.hpp"
     4
     5	#include <algorithm>
     6	#include <stdexcept>
     7	#include <string_view>
     8
     9	namespace EA::ExperimentRecommendation
    10	{
    11	namespace
    12	{
    13
    14	bool IsValidCanonicalHash(const std::string& value)
    15	{
    16	    constexpr std::string_view prefix = "fnv1a64:";
    17	    return value.size() == prefix.size() + 16U &&
    18	        value.compare(0, prefix.size(), prefix) == 0 &&
    19	        std::all_of(value.begin() +
    20	                static_cast<std::ptrdiff_t>(prefix.size()),
    21	            value.end(),
    22	            [](unsigned char character)
    23	            {
    24	                return (character >= '0' && character <= '9') ||
    25	                    (character >= 'a' && character <= 'f');
    26	            });
    27	}
    28
    29	} // namespace
    30
    31	RecommendationCampaignFollowUpProposalRatificationRequest
    32	ValidateRecommendationCampaignFollowUpProposalRatificationRequest(
    33	    const RecommendationCampaignFollowUpProposalRatificationRequest& request)
    34	{
    35	    if (request.reviewEventId <= 0)
    36	        throw std::invalid_argument(
    37	            "recommendation_campaign_follow_up_proposal_ratification_review_event_id_invalid");
    38	    if (!IsValidCanonicalHash(request.expectedReviewIdentityHash))
    39	        throw std::invalid_argument(
    40	            "recommendation_campaign_follow_up_proposal_ratification_expected_review_hash_invalid");
    41	    ValidateRecommendationCampaignFollowUpProposalRatifierIdentity(
    42	        request.ratifierIdentity);
    43	    ValidateRecommendationCampaignFollowUpProposalRatificationBasis(
    44	        request.ratificationBasis);
    45	    return request;
    46	}
    47
    48	RecommendationCampaignFollowUpProposalRatificationPersistResult
    49	RatifyRecommendationCampaignFollowUpProposal(
    50	    pqxx::connection& connection,
    51	    const RecommendationCampaignFollowUpProposalRatificationRequest& request)
    52	{
    53	    const auto validated =
    54	        ValidateRecommendationCampaignFollowUpProposalRatificationRequest(request);
    55	    pqxx::work transaction{connection};
    56	    if (!RecommendationCampaignFollowUpProposalReviewSchemaExists(transaction) ||
    57	        !RecommendationCampaignFollowUpProposalRatificationSchemaExists(transaction))
    58	        throw std::runtime_error(
    59	            "recommendation_campaign_follow_up_proposal_ratification_schemas_required");
    60	    const auto persistedReview =
    61	        FindRecommendationCampaignFollowUpProposalReview(
    62	            transaction, validated.reviewEventId);
    63	    if (!persistedReview)
    64	        throw std::runtime_error(
    65	            "recommendation_campaign_follow_up_proposal_ratification_review_not_found");
    66	    if (persistedReview->review.identity.hash !=
    67	        validated.expectedReviewIdentityHash)
    68	        throw std::runtime_error(
    69	            "recommendation_campaign_follow_up_proposal_ratification_review_identity_mismatch");
    70	    if (persistedReview->review.decision !=
    71	        RecommendationCampaignFollowUpProposalReviewDecision::approved)
    72	        throw std::runtime_error(
    73	            "recommendation_campaign_follow_up_proposal_ratification_review_not_eligible");
    74	    if (persistedReview->review.reviewerIdentity ==
    75	        validated.ratifierIdentity)
    76	        throw std::runtime_error(
    77	            "recommendation_campaign_follow_up_proposal_ratification_separation_of_duties_violation");
    78	    const auto ratification =
    79	        BuildRecommendationCampaignFollowUpProposalRatification(
    80	            persistedReview->reviewEventId, persistedReview->review,
    81	            validated.ratifierIdentity, validated.ratificationBasis);
    82	    auto result = PersistRecommendationCampaignFollowUpProposalRatification(
    83	        transaction, ratification);
    84	    transaction.commit();
    85	    return result;
    86	}
    87
    88	} // namespace EA::ExperimentRecommendation
     1	#pragma once
     2
     3	#include "ExperimentRecommendationCampaignFollowUpProposalRatification.hpp"
     4
     5	#include <optional>
     6	#include <string>
     7	#include <vector>
     8
     9	#include <pqxx/pqxx>
    10
    11	namespace EA::ExperimentRecommendation
    12	{
    13
    14	inline constexpr int
    15	    kMaximumRecommendationCampaignFollowUpProposalRatificationListLimit = 1000;
    16
    17	struct PersistedRecommendationCampaignFollowUpProposalRatification
    18	{
    19	    const long long ratificationEventId;
    20	    const RecommendationCampaignFollowUpProposalRatification ratification;
    21	    const std::string createdAt;
    22
    23	    PersistedRecommendationCampaignFollowUpProposalRatification(
    24	        long long ratificationEventId,
    25	        RecommendationCampaignFollowUpProposalRatification ratification,
    26	        std::string createdAt);
    27	    PersistedRecommendationCampaignFollowUpProposalRatification(
    28	        const PersistedRecommendationCampaignFollowUpProposalRatification&) =
    29	        default;
    30	    PersistedRecommendationCampaignFollowUpProposalRatification(
    31	        PersistedRecommendationCampaignFollowUpProposalRatification&&) = default;
    32	    bool operator==(
    33	        const PersistedRecommendationCampaignFollowUpProposalRatification&) const =
    34	        default;
    35	};
    36
    37	enum class RecommendationCampaignFollowUpProposalRatificationPersistOutcome
    38	{
    39	    recorded,
    40	    existingIdentical
    41	};
    42
    43	std::string RecommendationCampaignFollowUpProposalRatificationPersistOutcomeText(
    44	    RecommendationCampaignFollowUpProposalRatificationPersistOutcome outcome);
    45
    46	struct RecommendationCampaignFollowUpProposalRatificationPersistResult
    47	{
    48	    const RecommendationCampaignFollowUpProposalRatificationPersistOutcome outcome;
    49	    const PersistedRecommendationCampaignFollowUpProposalRatification persisted;
    50
    51	    RecommendationCampaignFollowUpProposalRatificationPersistResult(
    52	        RecommendationCampaignFollowUpProposalRatificationPersistOutcome outcome,
    53	        PersistedRecommendationCampaignFollowUpProposalRatification persisted);
    54	};
    55
    56	bool RecommendationCampaignFollowUpProposalRatificationSchemaExists(
    57	    pqxx::connection& connection);
    58	bool RecommendationCampaignFollowUpProposalRatificationSchemaExists(
    59	    pqxx::transaction_base& transaction);
    60
    61	RecommendationCampaignFollowUpProposalRatificationPersistResult
    62	PersistRecommendationCampaignFollowUpProposalRatification(
    63	    pqxx::transaction_base& transaction,
    64	    const RecommendationCampaignFollowUpProposalRatification& ratification);
    65
    66	std::optional<PersistedRecommendationCampaignFollowUpProposalRatification>
    67	FindRecommendationCampaignFollowUpProposalRatification(
    68	    pqxx::connection& connection,
    69	    long long ratificationEventId);
    70	std::optional<PersistedRecommendationCampaignFollowUpProposalRatification>
    71	FindRecommendationCampaignFollowUpProposalRatification(
    72	    pqxx::transaction_base& transaction,
    73	    long long ratificationEventId);
    74
    75	std::optional<PersistedRecommendationCampaignFollowUpProposalRatification>
    76	FindRecommendationCampaignFollowUpProposalRatificationByReviewEventId(
    77	    pqxx::connection& connection,
    78	    long long reviewEventId);
    79	std::optional<PersistedRecommendationCampaignFollowUpProposalRatification>
    80	FindRecommendationCampaignFollowUpProposalRatificationByReviewEventId(
    81	    pqxx::transaction_base& transaction,
    82	    long long reviewEventId);
    83
    84	std::optional<PersistedRecommendationCampaignFollowUpProposalRatification>
    85	FindRecommendationCampaignFollowUpProposalRatificationByProposalId(
    86	    pqxx::connection& connection,
    87	    long long followUpProposalId);
    88	std::optional<PersistedRecommendationCampaignFollowUpProposalRatification>
    89	FindRecommendationCampaignFollowUpProposalRatificationByProposalId(
    90	    pqxx::transaction_base& transaction,
    91	    long long followUpProposalId);
    92
    93	std::vector<PersistedRecommendationCampaignFollowUpProposalRatification>
    94	ListRecommendationCampaignFollowUpProposalRatifications(
    95	    pqxx::connection& connection,
    96	    int limit);
    97	std::vector<PersistedRecommendationCampaignFollowUpProposalRatification>
    98	ListRecommendationCampaignFollowUpProposalRatifications(
    99	    pqxx::transaction_base& transaction,
   100	    int limit);
   101
   102	} // namespace EA::ExperimentRecommendation
     1	-- Phase 6D: one immutable governance ratification for advancement of one exact
     2	-- approved Phase 6C review into the next separately controlled phase. The
     3	-- ratifier must differ from the Phase 6C reviewer. This evidence grants no
     4	-- Phase 6E capability or operational authority.
     5
     6	CREATE TABLE IF NOT EXISTS
     7	experiment_recommendation_campaign_follow_up_ratification_event (
     8	    recommendation_campaign_follow_up_ratification_event_id bigserial
     9	        PRIMARY KEY CHECK (
    10	            recommendation_campaign_follow_up_ratification_event_id > 0),
    11	    ratification_contract_version integer NOT NULL CHECK (
    12	        ratification_contract_version = 1),
    13	    recommendation_campaign_follow_up_proposal_review_event_id bigint NOT NULL,
    14	    review_contract_version integer NOT NULL CHECK (
    15	        review_contract_version = 1),
    16	    review_identity_canonical text COLLATE "C" NOT NULL CHECK (
    17	        review_identity_canonical <> '' AND
    18	        octet_length(review_identity_canonical) <= 1056768),
    19	    review_identity_hash text NOT NULL CHECK (
    20	        review_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    21	    review_decision text NOT NULL CHECK (review_decision = 'approved'),
    22	    reviewer_identity text COLLATE "C" NOT NULL CHECK (
    23	        reviewer_identity ~
    24	            '^[A-Za-z0-9][A-Za-z0-9._@:/+\-]{0,127}$'),
    25	    recommendation_campaign_follow_up_proposal_id bigint NOT NULL,
    26	    proposal_contract_version integer NOT NULL CHECK (
    27	        proposal_contract_version = 1),
    28	    proposal_identity_canonical text COLLATE "C" NOT NULL CHECK (
    29	        proposal_identity_canonical <> '' AND
    30	        octet_length(proposal_identity_canonical) <= 1048576),
    31	    proposal_identity_hash text NOT NULL CHECK (
    32	        proposal_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    33	    ratification_authority_role text COLLATE "C" NOT NULL CHECK (
    34	        ratification_authority_role = 'follow_up_governance_ratifier'),
    35	    ratification_decision text NOT NULL CHECK (
    36	        ratification_decision = 'ratified'),
    37	    ratifier_identity text COLLATE "C" NOT NULL CHECK (
    38	        ratifier_identity ~
    39	            '^[A-Za-z0-9][A-Za-z0-9._@:/+\-]{0,127}$'),
    40	    ratification_basis text COLLATE "C" NOT NULL CHECK (
    41	        ratification_basis <> '' AND octet_length(ratification_basis) <= 4096 AND
    42	        ratification_basis ~ E'[^ \t\r\n]' AND
    43	        translate(ratification_basis, E'\t\r\n', '') !~ '[[:cntrl:]]'),
    44	    ratification_identity_canonical text COLLATE "C" NOT NULL CHECK (
    45	        ratification_identity_canonical <> '' AND
    46	        octet_length(ratification_identity_canonical) <= 2113536),
    47	    ratification_identity_hash text NOT NULL CHECK (
    48	        ratification_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    49	    created_at timestamptz NOT NULL DEFAULT now(),
    50	    CONSTRAINT follow_up_ratification_separation_of_duties_check
    51	        CHECK (ratifier_identity <> reviewer_identity),
    52	    CONSTRAINT follow_up_ratification_review_fk
    53	        FOREIGN KEY (
    54	            recommendation_campaign_follow_up_proposal_review_event_id)
    55	        REFERENCES
    56	            experiment_recommendation_campaign_follow_up_proposal_review_event(
    57	                recommendation_campaign_follow_up_proposal_review_event_id)
    58	        ON DELETE RESTRICT,
    59	    CONSTRAINT follow_up_ratification_proposal_fk
    60	        FOREIGN KEY (recommendation_campaign_follow_up_proposal_id)
    61	        REFERENCES experiment_recommendation_campaign_follow_up_proposal(
    62	            recommendation_campaign_follow_up_proposal_id)
    63	        ON DELETE RESTRICT,
    64	    CONSTRAINT follow_up_ratification_review_uidx
    65	        UNIQUE (recommendation_campaign_follow_up_proposal_review_event_id),
    66	    CONSTRAINT follow_up_ratification_proposal_uidx
    67	        UNIQUE (recommendation_campaign_follow_up_proposal_id)
    68	);
    69
    70	CREATE INDEX IF NOT EXISTS
    71	follow_up_ratification_identity_hash_idx
    72	    ON experiment_recommendation_campaign_follow_up_ratification_event(
    73	        ratification_identity_hash);
    74
    75	CREATE OR REPLACE FUNCTION
    76	enforce_follow_up_ratification_provenance()
    77	RETURNS trigger
    78	LANGUAGE plpgsql
    79	SECURITY INVOKER
    80	AS $$
    81	BEGIN
    82	    PERFORM 1
    83	    FROM experiment_recommendation_campaign_follow_up_proposal_review_event
    84	        review
    85	    JOIN experiment_recommendation_campaign_follow_up_proposal proposal
    86	      ON proposal.recommendation_campaign_follow_up_proposal_id =
    87	             review.recommendation_campaign_follow_up_proposal_id
    88	    WHERE review.recommendation_campaign_follow_up_proposal_review_event_id =
    89	              NEW.recommendation_campaign_follow_up_proposal_review_event_id
    90	      AND review.review_contract_version = NEW.review_contract_version
    91	      AND review.review_identity_canonical = NEW.review_identity_canonical
    92	      AND review.review_identity_hash = NEW.review_identity_hash
    93	      AND review.review_decision = 'approved'
    94	      AND NEW.review_decision = review.review_decision
    95	      AND review.reviewer_identity = NEW.reviewer_identity
    96	      AND NEW.ratifier_identity <> review.reviewer_identity
    97	      AND review.recommendation_campaign_follow_up_proposal_id =
    98	              NEW.recommendation_campaign_follow_up_proposal_id
    99	      AND review.proposal_contract_version = NEW.proposal_contract_version
   100	      AND review.proposal_identity_canonical =
   101	              NEW.proposal_identity_canonical
   102	      AND review.proposal_identity_hash = NEW.proposal_identity_hash
   103	      AND proposal.proposal_contract_version = NEW.proposal_contract_version
   104	      AND proposal.proposal_identity_canonical =
   105	              NEW.proposal_identity_canonical
   106	      AND proposal.proposal_identity_hash = NEW.proposal_identity_hash;
   107	    IF NOT FOUND THEN
   108	        RAISE EXCEPTION 'follow-up proposal ratification provenance mismatch'
   109	            USING ERRCODE = '23514',
   110	                  CONSTRAINT =
   111	                      'follow_up_ratification_provenance_check';
   112	    END IF;
   113	    RETURN NEW;
   114	END;
   115	$$;
   116
   117	DO $$
   118	DECLARE ratification_schema text := current_schema();
   119	BEGIN
   120	    EXECUTE format(
   121	        'ALTER FUNCTION %I.enforce_follow_up_ratification_provenance() '
   122	        'SET search_path TO pg_catalog, %I, pg_temp',
   123	        ratification_schema, ratification_schema);
   124	END $$;
   125
   126	DROP TRIGGER IF EXISTS
   127	follow_up_ratification_provenance_trigger
   128	ON experiment_recommendation_campaign_follow_up_ratification_event;
   129	CREATE TRIGGER
   130	follow_up_ratification_provenance_trigger
   131	BEFORE INSERT ON
   132	    experiment_recommendation_campaign_follow_up_ratification_event
   133	FOR EACH ROW EXECUTE FUNCTION
   134	    enforce_follow_up_ratification_provenance();
   135
   136	COMMENT ON TABLE
   137	experiment_recommendation_campaign_follow_up_ratification_event IS
   138	    'Immutable Phase 6D governance ratification of advancement only; never Phase 6E capability, operational authorization, or action.';
   139	COMMENT ON COLUMN
   140	experiment_recommendation_campaign_follow_up_ratification_event.ratification_identity_canonical IS
   141	    'Authoritative byte-exact Phase 6D ratification identity; the tagged hash is only an accelerator.';
   142	COMMENT ON COLUMN
   143	experiment_recommendation_campaign_follow_up_ratification_event.ratification_decision IS
   144	    'Governance ratification of an eligible approved Phase 6C review for entry into the next separately controlled phase; grants no Phase 6E or action authority.';
   145
   146	REVOKE ALL PRIVILEGES ON
   147	    experiment_recommendation_campaign_follow_up_ratification_event
   148	    FROM PUBLIC, pqxx;
   149	REVOKE ALL PRIVILEGES ON FUNCTION
   150	    enforce_follow_up_ratification_provenance()
   151	    FROM PUBLIC, pqxx;
   152
   153	GRANT SELECT ON
   154	    experiment_recommendation_campaign_follow_up_ratification_event
   155	    TO pqxx;
   156	GRANT INSERT (
   157	    ratification_contract_version,
   158	    recommendation_campaign_follow_up_proposal_review_event_id,
   159	    review_contract_version,
   160	    review_identity_canonical,
   161	    review_identity_hash,
   162	    review_decision,
   163	    reviewer_identity,
   164	    recommendation_campaign_follow_up_proposal_id,
   165	    proposal_contract_version,
   166	    proposal_identity_canonical,
   167	    proposal_identity_hash,
   168	    ratification_authority_role,
   169	    ratification_decision,
   170	    ratifier_identity,
   171	    ratification_basis,
   172	    ratification_identity_canonical,
   173	    ratification_identity_hash)
   174	ON experiment_recommendation_campaign_follow_up_ratification_event
   175	TO pqxx;
   176
   177	DO $$
   178	DECLARE sequence_name text;
   179	BEGIN
   180	    sequence_name := pg_get_serial_sequence(
   181	        'experiment_recommendation_campaign_follow_up_ratification_event',
   182	        'recommendation_campaign_follow_up_ratification_event_id');
   183	    EXECUTE format(
   184	        'REVOKE ALL PRIVILEGES ON SEQUENCE %s FROM PUBLIC', sequence_name);
   185	    EXECUTE format(
   186	        'REVOKE ALL PRIVILEGES ON SEQUENCE %s FROM pqxx', sequence_name);
   187	    EXECUTE format(
   188	        'GRANT USAGE ON SEQUENCE %s TO pqxx', sequence_name);
   189	END $$;
Sources/ExperimentRecommendationCampaignFollowUpProposalRatificationRepository.cpp:59:            "recommendation_campaign_follow_up_proposal_ratification_separation_of_duties_violation");
Sources/ExperimentRecommendationCampaignFollowUpProposalRatificationRepository.cpp:118:        "experiment_recommendation_campaign_follow_up_ratification_event "
Sources/ExperimentRecommendationCampaignFollowUpProposalRatificationRepository.cpp:161:            return "existing_identical";
Sources/ExperimentRecommendationCampaignFollowUpProposalRatificationRepository.cpp:179:        "SELECT to_regclass('experiment_recommendation_campaign_follow_up_"
Sources/ExperimentRecommendationCampaignFollowUpProposalRatificationRepository.cpp:191:        "SELECT pg_advisory_xact_lock(hashtextextended($1, "
Sources/ExperimentRecommendationCampaignFollowUpProposalRatificationRepository.cpp:208:        "INSERT INTO experiment_recommendation_campaign_follow_up_"
Sources/ExperimentRecommendationCampaignFollowUpProposalRatificationRepository.cpp:264:        "experiment_recommendation_campaign_follow_up_ratification_event "
Sources/ExperimentRecommendationCampaignFollowUpProposalRatificationRepository.cpp:312:        "experiment_recommendation_campaign_follow_up_ratification_event "
Sources/ExperimentRecommendationCampaignFollowUpProposalRatificationRepository.cpp:344:        "experiment_recommendation_campaign_follow_up_ratification_event "
Sources/ExperimentRecommendationCampaignFollowUpProposalRatification.cpp:204:        << "experiment_recommendation_campaign_follow_up_proposal_ratification_v1"
Sources/ExperimentRecommendationCampaignFollowUpProposalRatification.cpp:213:           ";separation_of_duties_required=true"
Sources/ExperimentRecommendationCampaignFollowUpProposalRatification.cpp:214:           ";phase_6e_capability_granted=false"
Sources/ExperimentRecommendationCampaignFollowUpProposalRatification.cpp:216:           ";execution_authorized=false"
Sources/ExperimentRecommendationCampaignFollowUpProposalRatification.cpp:217:           ";follow_up_authorized=false"
Sources/ExperimentRecommendationCampaignFollowUpProposalRatification.cpp:220:           ";scheduler_started=false"
Sources/ExperimentRecommendationCampaignFollowUpProposalRatification.cpp:221:           ";scheduler_signaled=false"
Sources/ExperimentRecommendationCampaignFollowUpProposalRatification.cpp:222:           ";workers_started=false"
Sources/ExperimentRecommendationCampaignFollowUpProposalRatification.cpp:223:           ";experiments_created=false"
Sources/ExperimentRecommendationCampaignFollowUpProposalRatification.cpp:224:           ";experiments_modified=false"
Sources/ExperimentRecommendationCampaignFollowUpProposalRatification.cpp:331:                "ratification_separation_of_duties_violation");
Sources/ExperimentRecommendationCampaignFollowUpProposalRatification.cpp:446:        RecommendationCampaignFollowUpProposalRatificationDecision::ratified,
Sources/ExperimentRecommendationCampaignFollowUpProposalRatification.cpp:471:        case RecommendationCampaignFollowUpProposalRatificationDecision::ratified:
Sources/ExperimentRecommendationCampaignFollowUpProposalRatification.cpp:472:            return "ratified";
Sources/ExperimentRecommendationCampaignFollowUpProposalRatification.cpp:481:    if (text == "ratified")
Sources/ExperimentRecommendationCampaignFollowUpProposalRatification.cpp:482:        return RecommendationCampaignFollowUpProposalRatificationDecision::ratified;
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationMigrationTests.sql:4:        '%I.experiment_recommendation_campaign_follow_up_ratification_event',
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationMigrationTests.sql:7:        '%I.experiment_recommendation_campaign_follow_up_proposal_review_event',
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationMigrationTests.sql:10:        '%I.experiment_recommendation_campaign_follow_up_proposal',
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationMigrationTests.sql:51:              'experiment_recommendation_campaign_follow_up_ratification_event'
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationMigrationTests.sql:85:              'experiment_recommendation_campaign_follow_up_ratification_event'
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationMigrationTests.sql:87:              'activated', 'execution_authorized', 'follow_up_authorized',
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationMigrationTests.sql:88:              'queued', 'scheduled', 'scheduler_started',
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationMigrationTests.sql:89:              'scheduler_signaled', 'workers_started',
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationMigrationTests.sql:90:              'experiments_created', 'experiments_modified',
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationMigrationTests.sql:105:              '%ratification_decision = ''ratified''%') OR
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationMigrationTests.sql:110:              'follow_up_ratification_separation_of_duties_check'
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationMigrationTests.sql:124:    IF has_table_privilege('pqxx', ratification_table, 'INSERT') THEN
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationMigrationTests.sql:125:        RAISE EXCEPTION 'follow-up proposal ratification table INSERT too broad';
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationMigrationTests.sql:128:            'pqxx', ratification_table, 'ratification_contract_version', 'INSERT') OR
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationMigrationTests.sql:130:            'pqxx', ratification_table, 'review_identity_canonical', 'INSERT') OR
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationMigrationTests.sql:132:            'pqxx', ratification_table, 'reviewer_identity', 'INSERT') OR
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationMigrationTests.sql:134:            'pqxx', ratification_table, 'ratification_authority_role', 'INSERT') OR
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationMigrationTests.sql:136:            'pqxx', ratification_table, 'ratification_identity_canonical', 'INSERT') THEN
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationMigrationTests.sql:137:        RAISE EXCEPTION 'follow-up proposal ratification payload INSERT missing';
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationMigrationTests.sql:142:            'INSERT') OR
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationMigrationTests.sql:144:            'pqxx', ratification_table, 'created_at', 'INSERT') THEN
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationMigrationTests.sql:147:    IF has_table_privilege('pqxx', ratification_table, 'UPDATE') OR
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationMigrationTests.sql:148:       has_table_privilege('pqxx', ratification_table, 'DELETE') OR
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationMigrationTests.sql:149:       has_table_privilege('pqxx', ratification_table, 'TRUNCATE') THEN
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationMigrationTests.sql:201:       has_sequence_privilege('pqxx', ratification_sequence, 'UPDATE') THEN
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:125:CREATE TABLE experiment_recommendation_campaign_approval(
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:129:CREATE TABLE experiment_recommendation_campaign_materialization(
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:137:CREATE TABLE experiment_recommendation_campaign_materialization_member(
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:143:    source_experiment_id bigint NOT NULL,
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:145:CREATE TABLE experiment(placeholder text PRIMARY KEY);
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:146:CREATE TABLE experiment_scheduler(placeholder text PRIMARY KEY);
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:147:INSERT INTO experiment VALUES('unchanged');
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:148:INSERT INTO experiment_scheduler VALUES('unchanged');
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:150:    setup.exec("GRANT USAGE ON SCHEMA " + setup.quote_name(schema) +
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:151:        " TO pqxx; GRANT SELECT ON ALL TABLES IN SCHEMA " +
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:249:            "INSERT INTO experiment_recommendation_campaign_approval "
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:255:            "INSERT INTO experiment_recommendation_campaign_materialization "
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:267:            "INSERT INTO experiment_recommendation_campaign_"
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:344:        "pg_get_serial_sequence('experiment_recommendation_campaign_follow_"
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:468:            "Database/migrations/042_experiment_recommendation_campaign_follow_up_proposal.sql");
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:470:            "Database/migrations/043_experiment_recommendation_campaign_follow_up_proposal_review.sql");
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:472:            "Database/migrations/044_experiment_recommendation_campaign_follow_up_proposal_ratification.sql");
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:474:            "Database/migrations/044_experiment_recommendation_campaign_follow_up_proposal_ratification.sql");
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:477:                   "experiment_recommendation_campaign_follow_up_"
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:484:            "Database/migrations/042_experiment_recommendation_campaign_follow_up_proposal.sql");
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:486:            "Database/migrations/043_experiment_recommendation_campaign_follow_up_proposal_review.sql");
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:516:            "experiment_recommendation_campaign_follow_up_proposal",
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:520:            "experiment_recommendation_campaign_follow_up_proposal_member",
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:524:            "experiment_recommendation_campaign_follow_up_proposal_review_event",
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:526:        const std::string experimentSnapshotBefore = TableSnapshot(
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:527:            owner, upgradeSchema, "experiment", "placeholder");
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:528:        const std::string schedulerSnapshotBefore = TableSnapshot(
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:529:            owner, upgradeSchema, "experiment_scheduler", "placeholder");
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:532:            "Database/migrations/044_experiment_recommendation_campaign_follow_up_proposal_ratification.sql");
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:534:            "Database/migrations/044_experiment_recommendation_campaign_follow_up_proposal_ratification.sql");
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:601:                       replay.persisted.ratification.decision) == "ratified");
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:628:        }, "recommendation_campaign_follow_up_proposal_ratification_separation_of_duties_violation");
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:750:                    "INSERT INTO experiment_recommendation_campaign_follow_up_"
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:769:                        authorityRole, "ratified", ratifierIdentity, basis,
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:797:                "INSERT INTO experiment_recommendation_campaign_follow_up_"
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:816:                    value.ratificationAuthorityRole, "ratified",
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:834:            "experiment_recommendation_campaign_follow_up_ratification_event");
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:844:                   "experiment_recommendation_campaign_follow_up_"
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:875:                    "UPDATE experiment_recommendation_campaign_follow_up_"
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:885:                    "UPDATE experiment_recommendation_campaign_follow_up_"
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:895:                    "UPDATE experiment_recommendation_campaign_follow_up_"
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:926:                        RecommendationCampaignFollowUpProposalRatificationDecision::ratified,
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:930:                    "UPDATE experiment_recommendation_campaign_follow_up_"
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:959:                    "UPDATE experiment_recommendation_campaign_follow_up_"
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:974:                "INSERT INTO experiment_recommendation_campaign_follow_up_"
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:992:                    value.ratificationAuthorityRole, "ratified",
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:1008:                "UPDATE experiment_recommendation_campaign_follow_up_"
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:1022:                "DELETE FROM experiment_recommendation_campaign_follow_up_"
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:1036:                "INSERT INTO experiment_recommendation_campaign_follow_up_"
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:1051:                "INSERT INTO experiment_recommendation_campaign_follow_up_"
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:1066:                "DELETE FROM experiment_recommendation_campaign_follow_up_"
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:1106:                       "('advisory','tuple');")
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:1129:                   "experiment_recommendation_campaign_follow_up_proposal",
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:1133:                   "experiment_recommendation_campaign_follow_up_proposal_member",
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:1137:                   "experiment_recommendation_campaign_follow_up_proposal_review_event",
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:1140:        assert(TableSnapshot(owner, upgradeSchema, "experiment", "placeholder") ==
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:1141:            experimentSnapshotBefore);
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:1143:                   owner, upgradeSchema, "experiment_scheduler", "placeholder") ==
Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationRepositoryTests.cpp:1144:            schedulerSnapshotBefore);

exec
/bin/bash -lc "nl -ba docs/Phase4DExperimentRecommendationCampaignMaterialization.rst && nl -ba docs/Phase5ExperimentRecommendationCampaignLaunch.rst && nl -ba docs/Phase5ExperimentRecommendationCampaignStatus.rst && nl -ba docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst | sed -n '1,240p'" in /Volumes/Developer SSD/ExpertAdvisor
 succeeded in 0ms:
     1	Phase 4D Step 4 — Approved Campaign Materialization
     2	===================================================
     3
     4	Purpose and boundary
     5	--------------------
     6
     7	Step 4 translates one explicit ``approved`` Phase 4D campaign decision into
     8	the exact ordered set of Phase 4C conversion proposals selected by the
     9	authorized plan. Materialization is not campaign execution: it creates no
    10	experiment, conversion review, conversion execution, or activation and does
    11	not contact the scheduler or a worker.
    12
    13	Authoritative reconstruction
    14	----------------------------
    15
    16	The command accepts an approval primary key plus explicit operator and reason
    17	text. In one PostgreSQL transaction it validates the approval, parses its exact
    18	canonical planning policy and scope, reloads the ranking snapshot and Phase 4C
    19	workflow evidence, and reconstructs the Step 1 plan and Step 2 review. The
    20	reconstructed approval canonical text and hash must equal the persisted Step 3
    21	evidence. Rejected, stale, malformed, or empty approvals fail before any
    22	proposal insert.
    23
    24	Every selected member then passes through the existing Phase 4C conversion
    25	domain contract. That contract still requires effective manual recommendation
    26	approval, completed valid evaluation and score evidence, one supported
    27	mutation, and an exact source-configuration match. Campaign approval is not a
    28	substitute for Phase 4C eligibility.
    29
    30	Identity and persistence
    31	------------------------
    32
    33	Materialization contract version 1 length-frames variable text and binds the
    34	exact campaign approval; ranking snapshot; planning policy and scope; plan and
    35	review identities; approval decision, reviewer, and reason; explicit
    36	materialization operator and reason; ordered selected-member identities; and
    37	every Phase 4C proposal canonical identity and hash. Timestamps are descriptive
    38	and excluded. Canonical text is authoritative; hashes are accelerators only.
    39
    40	Migration ``041_experiment_recommendation_campaign_materialization.sql`` adds
    41	one immutable manifest per approval and ordered member rows. Each member links
    42	the exact ranking member and Phase 4C proposal. Foreign keys are restrictive.
    43	Invoker-rights enforcement triggers require the manifest's copied approval
    44	fields to equal the referenced approved decision, require every member's
    45	snapshot/rank and proposal provenance to match its source rows, and reject a
    46	transaction whose final member count or one-based ordinal range is incomplete.
    47	Runtime ``pqxx`` has only ``SELECT``, column-limited ``INSERT``, and sequence
    48	``USAGE``; it cannot supply generated IDs/timestamps or mutate history.
    49
    50	Atomicity, replay, and concurrency
    51	----------------------------------
    52
    53	Approval validation, reconstruction, proposal lookup/insertion, manifest
    54	insertion, and every member link share one transaction. Any failure rolls back
    55	new proposals and materialization evidence together. Pre-existing exact Phase
    56	4C proposals are reused and linked explicitly.
    57
    58	A transaction-scoped advisory lock is keyed by approval identity and one
    59	manifest is permitted per approval. Exact replay returns the existing manifest;
    60	changed operator, reason, selected membership, proposal provenance, or approval
    61	evidence conflicts. Proposals are persisted in sorted hash/canonical order to
    62	retain deterministic collision-lock ordering, while manifest order remains
    63	campaign rank order.
    64
    65	An exact retry first validates the existing manifest's authoritative canonical
    66	identity against the immutable approval, ordered member links, and normalized
    67	operator request. It does not reconstruct the current plan on replay because
    68	the proposals created by the original materialization are themselves durable
    69	workflow evidence that would intentionally change a newly planned campaign.
    70	First-time materialization always performs the full stale-evidence
    71	reconstruction described above.
    72
    73	CLI
    74	---
    75
    76	::
    77
    78	   LSTM_Release --materialize-recommendation-campaign \
    79	     --campaign-approval-id=123 \
    80	     --campaign-materialized-by='operator@example' \
    81	     --campaign-materialization-reason='Create exact Phase 4C proposal set.'
    82
    83	Read-only inspection::
    84
    85	   LSTM_Release --show-recommendation-campaign-materialization=1
    86	   LSTM_Release --list-recommendation-campaign-materializations \
    87	     --campaign-approval-id=123 --campaign-materialization-limit=100
    88
    89	Output distinguishes a newly recorded manifest from exact replay and reports
    90	ordered member/proposal links plus exact created/reused proposal counts. It
    91	always reports conversion reviews, executions, activations, experiments,
    92	scheduler, and workers as unchanged.
    93
    94	Verification and deferred work
    95	------------------------------
    96
    97	Use only an explicit disposable non-``LSTM`` database::
    98
    99	   createdb phase4d_campaign_test 2>/dev/null || true
   100	   LSTM_TEST_DB_NAME=phase4d_campaign_test \
   101	     /tmp/ExperimentRecommendationCampaignMaterializationRepositoryTests
   102
   103	Phase 4C proposal review, execution to a paused experiment, and explicit
   104	activation remain separate actions. Phase 4D Step 6 provides an explicit atomic
   105	operator convenience for reviewing every exact persisted member through
   106	ordinary Phase 4C review rows. Campaign execution, automatic review or
   107	activation, background scans, scheduler integration, and worker launch remain
   108	outside Step 4.
   109
   110	Phase 4D Step 5 adds a separate read-only handoff projection over the immutable
   111	manifest membership and current Phase 4C lifecycle evidence. It does not alter
   112	the materialization or advance any linked proposal.
     1	Phase 5 Step 3 — Atomic Campaign Conversion Launch
     2	====================================================
     3
     4	Purpose and authority
     5	---------------------
     6
     7	Phase 5 Step 3 is one explicitly confirmed orchestration convenience over the
     8	existing Phase 4C conversion-execution and conversion-activation authorities.
     9	For one persisted Phase 4D campaign materialization, it establishes the
    10	ordinary execution and activation outcome for every exact member in one
    11	PostgreSQL transaction. A successful first launch leaves every resulting
    12	experiment in the ordinary ``pending/train`` state. It does not start or signal
    13	the scheduler, poll for work, launch a worker, or start training.
    14
    15	The operation adds no launch table, campaign status, lifecycle authority,
    16	scheduler queue, privilege, or migration. Durable authority remains the
    17	immutable Phase 4D materialization, ordinary Phase 4C proposal reviews,
    18	executions, activations, and experiment rows.
    19
    20	Membership and identity
    21	-----------------------
    22
    23	The sole member set is the persisted ordinal order of
    24	``experiment_recommendation_campaign_materialization_member`` rows for the
    25	requested ``experiment_recommendation_campaign_materialization``. Existing
    26	Phase 4D loaders and validators establish the supported materialization
    27	contract, canonical identity and hash, exact count, contiguous ordinals,
    28	unique links, and proposal provenance. No recommendation, ranking, score,
    29	symbol, horizon, policy, or current query can substitute membership.
    30
    31	The version-1 launch operation identity binds the operation type and contract,
    32	the exact materialization ID, contract, canonical identity and hash, ordered
    33	member count, and launch-semantics version. The validated immutable
    34	materialization identity already binds the ordered exact members. The launch
    35	identity contains no timestamp, process value, random input, or sequence ID;
    36	ordinary Phase 4C execution and activation rows remain the durable retry
    37	authority.
    38
    39	Transaction and lock hierarchy
    40	------------------------------
    41
    42	One write transaction owns the complete launch. It:
    43
    44	1. acquires existing proposal review/execution advisory locks in ascending
    45	   proposal-ID order;
    46	2. reloads and validates the complete execution phase;
    47	3. creates every required paused experiment and ordinary execution row;
    48	4. resolves the complete execution and experiment set;
    49	5. acquires existing activation advisory locks in ascending execution-ID
    50	   order, then experiment row locks in ascending experiment-ID order;
    51	6. reloads and validates the complete activation phase;
    52	7. creates every required activation row and applies the existing
    53	   ``paused/train`` to ``pending/train`` transition; and
    54	8. commits once.
    55
    56	The reused Step 1, Step 2, and Phase 4C transaction-bound primitives begin and
    57	commit no transaction. No activation mutation occurs before every member
    58	passes activation validation. An execution-stage or activation-stage failure
    59	rolls back every experiment, execution, activation, and lifecycle update made
    60	by the request. The shared lock domains serialize direct Phase 4C review,
    61	execution, and activation commands and overlapping campaign commands; locks
    62	are not taken in materialization order.
    63
    64	Supported state matrix
    65	----------------------
    66
    67	* No executions and no activations: create all ordinary executions and
    68	  activations, returning ``newly_launched``.
    69	* All valid executions and no activations: reuse the immutable executions,
    70	  activate all experiments, and return ``activated_existing_executions``.
    71	* All valid executions and activations, with all experiments still exactly
    72	  ``pending/train``: change nothing and return ``already_satisfied``.
    73	* Some but not all executions: reject with a partial-execution conflict.
    74	* Some but not all activations: reject with a partial-activation conflict.
    75	* Activation without exact execution provenance, malformed evidence, or an
    76	  invalid lifecycle state: fail closed without repair.
    77	* An activated experiment that has progressed beyond ``pending/train`` is not
    78	  rewound and is not reported as an ordinary idempotent retry.
    79
    80	An existing execution keeps its original approving review as immutable
    81	authorization even if a later ordinary review reverses current disposition.
    82	A proposal that needs first execution must still have a current valid approve
    83	decision.
    84
    85	CLI and dry run
    86	---------------
    87
    88	Write::
    89
    90	   LSTM_Release \
    91	     --launch-recommendation-campaign-materialization=MATERIALIZATION_ID \
    92	     --yes
    93
    94	Dry run::
    95
    96	   LSTM_Release \
    97	     --launch-recommendation-campaign-materialization MATERIALIZATION_ID \
    98	     --dry-run
    99
   100	Both option forms require one positive ID. Duplicate or malformed values and
   101	combinations with another standalone operation are rejected before database
   102	dispatch. ``--yes`` is required for a write and is not required for dry run.
   103
   104	Dry run uses one repeatable-read, read-only transaction and takes no advisory
   105	or row lock. It writes nothing and advances no sequence. When executions do not
   106	yet exist, their future execution, experiment, and activation IDs are reported
   107	as ``null`` rather than allocated. Results are advisory because another
   108	explicit operation may change workflow evidence after the snapshot.
   109
   110	Output and operational boundary
   111	-------------------------------
   112
   113	``RECOMMENDATION_CAMPAIGN_LAUNCH`` is followed by one
   114	``RECOMMENDATION_CAMPAIGN_LAUNCH_MEMBER`` record per persisted ordinal. The
   115	records contain materialization and operation hashes, aggregate create/reuse
   116	counts, exact provenance IDs when available, execution and activation
   117	dispositions, pre/post lifecycle state, diagnostics, dry-run state, and
   118	explicit all-or-nothing and safety fields. Optional identifiers use ``null``;
   119	variable text uses the production machine-text encoder.
   120
   121	Step 1 remains execute-only and leaves experiments ``paused/train``. Step 2
   122	remains activate-only and requires existing executions. Step 3 combines those
   123	existing authorities atomically without calling either public command and
   124	without adding campaign lifecycle state. Pending experiments are merely
   125	eligible for a later claim by the existing scheduler; launch does not imply
   126	that training has started.
   127	Phase 5 Step 4 is the separate read-only operational projection of those exact
   128	members. It observes persisted lifecycle evidence but never launches, repairs,
   129	or controls it.
   130
   131	References
   132	----------
   133
   134	* ``docs/Phase4CExperimentRecommendationConversionExecution.rst``
   135	* ``docs/Phase4CExperimentRecommendationConversionActivation.rst``
   136	* ``docs/Phase4DExperimentRecommendationCampaignHandoff.rst``
   137	* ``docs/Phase5ExperimentRecommendationCampaignExecution.rst``
   138	* ``docs/Phase5ExperimentRecommendationCampaignActivation.rst``
   139	* ``docs/architecture/Volume_VIII_Recommendation_Engine.md``
   140	* ``docs/Phase5ExperimentRecommendationCampaignStatus.rst``
     1	Phase 5 Step 4 — Recommendation Campaign Operational Status
     2	=============================================================
     3
     4	Purpose and authority
     5	---------------------
     6
     7	Phase 5 Step 4 is a read-only point-in-time projection for one persisted Phase
     8	4D campaign materialization. It reports every exact member's ordinary Phase 4C
     9	review, execution, activation, experiment lifecycle, worker metadata, model,
    10	final inference, and final analysis evidence. It creates no campaign status or
    11	lifecycle authority and persists nothing.
    12
    13	The immutable ordered rows in
    14	``experiment_recommendation_campaign_materialization_member`` are the only
    15	membership source. The production Phase 4D loader validates the manifest,
    16	canonical identity and hash, selected count, contiguous ordinals, unique links,
    17	and exact proposal provenance before downstream status is projected.
    18
    19	Snapshot and query contract
    20	---------------------------
    21
    22	One PostgreSQL ``REPEATABLE READ``, ``READ ONLY`` transaction owns the complete
    23	snapshot. The operation takes no advisory or row locks, performs no write or
    24	sequence call, and uses bounded exact-ID queries for the materialized proposals
    25	and their linked experiments. Unrelated workflow and experiment rows cannot
    26	become campaign members.
    27
    28	``observed_at`` is one PostgreSQL transaction timestamp shared by every output
    29	row. It is metadata, not identity. The deterministic version-1 snapshot
    30	identity binds the validated materialization and every ordered member's durable
    31	review, execution, activation, experiment lifecycle, worker, model, phase-
    32	completion, exact completed/failed final-result counts, consistency, and
    33	diagnostic projection.
    34	Persisted lifecycle timestamps are evidence and are bound; ``observed_at`` and
    35	local wall-clock time are not identity inputs. The identity contains no random
    36	value, local command process ID, transaction ID, or unordered iteration. A
    37	persisted ``worker_pid`` is bound as database evidence but is not treated as a
    38	live-process observation.
    39
    40	Member and aggregate status
    41	---------------------------
    42
    43	Execution, activation, operational state, terminal result, and consistency are
    44	separate dimensions. Operational states distinguish proposal-only, executed
    45	paused, pending scheduler claim, running train/infer/analyze, completed, failed,
    46	cancelled, paused after progress, and inconsistent evidence. An execution keeps
    47	its immutable original authorizing approval even when the greatest current
    48	review-decision ID later records rejection.
    49
    50	Final inference is configured only when both persisted inference dates exist.
    51	For ordinary recommendation-conversion experiments, the existing scheduler
    52	requires that range after training and advances completed final inference to
    53	final analysis. Successful ``completed/done`` therefore requires the exact
    54	final inference row selected by the scheduler (current model, symbol, horizon,
    55	threshold, and dates) and completed final analysis for the current model. A
    56	persisted ``completed/done`` recommendation-conversion experiment without that
    57	configured pipeline is reported as inconsistent rather than treated as a
    58	train-only success.
    59
    60	Unknown lifecycle values, invalid cardinality or provenance, activation without
    61	execution, missing experiments, incompatible activation/lifecycle state,
    62	incoherent worker evidence, and missing configured terminal results remain
    63	visible as deterministic member diagnostics. A valid materialization still
    64	returns every ordered member when downstream evidence is malformed. An invalid
    65	materialization rejects the command because exact membership cannot be trusted.
    66	Any inconsistent member forces aggregate ``inconsistent`` status.
    67
    68	CLI and output
    69	--------------
    70
    71	::
    72
    73	   LSTM_Release --recommendation-campaign-status=MATERIALIZATION_ID
    74	   LSTM_Release --recommendation-campaign-status MATERIALIZATION_ID
    75
    76	The positive ID is required. Duplicate options, ``--dry-run``, ``--yes``, and
    77	combinations with any other standalone operation are rejected before database
    78	dispatch because the command is intrinsically read-only.
    79
    80	``RECOMMENDATION_CAMPAIGN_STATUS`` reports snapshot/materialization identities,
    81	aggregate state and counts, percentages and epoch aggregates where meaningful,
    82	and explicit no-write/no-lock/no-scheduler safety fields. It is followed by one
    83	``RECOMMENDATION_CAMPAIGN_STATUS_MEMBER`` record per persisted ordinal with
    84	exact provenance IDs, independent state dimensions, persisted lifecycle and
    85	worker fields, phase-completion flags, consistency, and diagnostics. Optional
    86	values use ``null`` and variable text uses the production percent encoder.
    87
    88	The snapshot is advisory and may become stale immediately after its transaction
    89	ends. It observes persisted scheduler/worker evidence only; it does not inspect
    90	processes, poll or signal the scheduler, launch workers, or control experiments.
    91
    92	Schema and boundaries
    93	---------------------
    94
    95	No migration or privilege change is required. Runtime access uses existing
    96	``SELECT`` privileges. Step 1 remains execute-only, Step 2 activate-only, Step 3
    97	atomic execute-and-activate, and Step 4 observational only. Step 4 invokes none
    98	of the preceding mutation commands and performs no automatic follow-up.
    99
   100	References
   101	----------
   102
   103	* ``docs/Phase4DExperimentRecommendationCampaignHandoff.rst``
   104	* ``docs/Phase5ExperimentRecommendationCampaignExecution.rst``
   105	* ``docs/Phase5ExperimentRecommendationCampaignActivation.rst``
   106	* ``docs/Phase5ExperimentRecommendationCampaignLaunch.rst``
   107	* ``docs/architecture/Volume_VIII_Recommendation_Engine.md``
     1	Phase 5 Recommendation Campaign Outcome Assessment
     2	==================================================
     3
     4	Purpose and boundary
     5	--------------------
     6
     7	Step 5a defines a pure, immutable, point-in-time domain assessment of the
     8	scientific outcomes associated with an exact recommendation-campaign
     9	materialization.  It is read-only, non-persistent, non-authoritative, and does
    10	not declare campaign success or authorize follow-up work.
    11
    12	Lifecycle, consistency, and comparison
    13	--------------------------------------
    14
    15	Lifecycle is explicit (``not_terminal``, ``succeeded``, ``failed``,
    16	``cancelled``, or ``unknown``).  Consistency is independent
    17	(``consistent`` or ``inconsistent``), and inconsistent downstream evidence is
    18	retained with deterministic diagnostic codes.  Callers can also carry a known
    19	inconsistent lifecycle or workflow observation without changing or discarding
    20	the exact materialized-member identity.  Scientific comparisons are derived
    21	only for consistent succeeded members.  Member outcomes distinguish not-ready,
    22	comparable success, context-changed success, metric-gap success, terminal
    23	failure, terminal cancellation, and inconsistency.
    24
    25	Multi-metric evidence
    26	---------------------
    27
    28	Source and result evidence contain metric collections.  Collections reject a
    29	duplicate metric identity and are canonicalized by metric identity, so input
    30	order does not affect output or assessment identity.  A comparison is derived
    31	for every metric identity in the source/result union.  Member outcome counts
    32	remain separate from metric-comparison and delta counts.
    33
    34	Provenance and deterministic identity
    35	-------------------------------------
    36
    37	Each materialized member may identify its expected downstream experiment.
    38	Succeeded-result evidence is checked against that exact experiment.  Terminal
    39	members require an expected experiment, and expected experiments, observed
    40	result experiments, result models, and typed result identities may not be
    41	reused across members.  Unexpected result evidence is retained and diagnosed
    42	rather than discarded.  Provenance defects remain inconsistent members when
    43	top-level materialization membership is trustworthy; invalid or ambiguous
    44	member identity fails the assessment.  Campaign and materialization inputs
    45	must agree on both the persisted campaign-approval ID and its canonical
    46	identity hash.
    47
    48	Canonical text and hashing bind lifecycle, consistency, diagnostics, canonical
    49	source and result metrics, comparisons, exact provenance, and aggregate member
    50	and metric summaries.  Authoritative campaign and materialization canonical
    51	texts are length-framed into the assessment identity alongside their compact
    52	hashes.  Canonical number formatting is locale-independent.  ``observed_at``
    53	is intentionally excluded from evidence identity.
    54
    55	Non-goals
    56	---------
    57
    58	Step 5a does not provide a repository, database access, persistence, service,
    59	CLI, scheduler, worker, filesystem behavior, campaign-success policy, or
    60	follow-up authorization.
    61
    62	Step 5b authoritative-evidence integration
    63	------------------------------------------
    64
    65	Step 5b supplies the read-only integration without changing the Step 5a domain
    66	boundary.  One PostgreSQL ``REPEATABLE READ``, ``READ ONLY`` transaction loads
    67	authoritative persisted evidence: the exact materialization through the Phase
    68	4D production loader, its exact campaign approval, and the evidence underlying
    69	lifecycle and workflow consistency from the existing Phase 5 campaign-status
    70	projection.  Step 5b builds a non-authoritative, point-in-time assessment from
    71	that evidence.  It does not reconstruct a second lifecycle classifier.
    72
    73	The Step 5a campaign identity is the persisted approval canonical text and
    74	hash.  Its materialization input receives the materialization row's persisted
    75	``approval_identity_hash`` unchanged, so Step 5a verifies the approval-to-
    76	materialization binding.  Exact materialization member ordinal, member ID,
    77	ranking-member ID, recommendation ID, source-experiment ID, proposal ID, and
    78	the execution-linked expected experiment ID flow from the validated status
    79	snapshot.  A status member reported as inconsistent is passed to Step 5a as
    80	``inputConsistency=inconsistent``; Step 5b neither repairs nor normalizes the
    81	upstream evidence.
    82
    83	Scientific evidence
    84	-------------------
    85
    86	Source evidence follows the materialized recommendation's persisted source
    87	experiment, model, and analysis identities.  Source final-inference selection
    88	uses the immutable source invocation canonical text captured by the exact
    89	persisted Phase 4C proposal; it never substitutes the source experiment's
    90	current configuration.  Result evidence follows the exact conversion-
    91	execution experiment, its current linked model, and its final inference and
    92	final analysis rows.  The comparison context is taken from the selected
    93	persisted final inference identity: canonical symbol, prediction horizon,
    94	threshold, persisted window size, the versioned
    95	``inference_eval_result_label_v1`` representation of the
    96	``label_rule_id``/``target_type`` pair, and canonical ``YYYY-MM-DD`` inference
    97	range.
    98
    99	The version-1 Step 5b mapping carries ``inference_accuracy`` and
   100	``leader_score`` with ``NumericDelta`` support.  The source values are the
   101	immutable values captured on the recommendation; result values are the final
   102	analysis values.  Missing analysis or metric evidence remains absent or null,
   103	and changed contexts remain non-comparable.  No metric is recomputed and no
   104	campaign-success threshold is applied.
   105
   106	Other persisted analysis fields were deliberately not mapped in this version:
   107	the immutable recommendation does not retain a directly corresponding value
   108	with the same scientific meaning.  In particular, Step 5b does not reconstruct
   109	predicted-neutral proportions or evidence totals from result counts.  Such
   110	values therefore cannot be mistaken for a valid numeric delta.
   111
   112	CLI and output
   113	--------------
   114
   115	::
   116
   117	   LSTM_Release \
   118	     --recommendation-campaign-outcome-assessment=MATERIALIZATION_ID
   119
   120	The separated option form is also accepted.  The command is intrinsically
   121	read-only, so ``--yes`` and ``--dry-run`` are rejected.  Output begins with one
   122	``RECOMMENDATION_CAMPAIGN_OUTCOME_ASSESSMENT`` record and then one
   123	``RECOMMENDATION_CAMPAIGN_OUTCOME_ASSESSMENT_MEMBER`` record per exact
   124	materialized ordinal.  Records expose contract and identity hashes, the exact
   125	Step 5a aggregate classification and counts, lifecycle, separate input and
   126	final consistency, diagnostics, provenance, contexts, metrics, comparisons,
   127	explicit evidence/context-presence booleans and ``null`` values,
   128	percent-encoded variable text, the materialization's persisted campaign-
   129	approval hash, and explicit no-write/no-scheduler/no-success/no-follow-up
   130	safety fields.
   131
   132	No schema migration or privilege change is required.  The assessment remains
   133	point-in-time and non-persistent; Step 5b inserts, updates, deletes, activates,
   134	queues, retries, and launches nothing and authorizes no follow-up.
   135
   136	Step 5c pure outcome policy
   137	----------------------------
   138
   139	Step 5c adds only a database-free policy and interpretation contract over one
   140	immutable Step 5a assessment.  It does not reopen source or result evidence,
   141	recompute a metric or delta, or change any Step 5a lifecycle, consistency,
   142	member-outcome, comparison, diagnostic, or aggregate classification.  Evidence
   143	classification and policy judgment remain separate fields.
   144
   145	Policy contract version 1 has an explicit canonical identity.  Its default
   146	minimum comparable-member evidence coverage is one ``succeeded_comparable``
   147	member, and its required metric rules are ``inference_accuracy`` and
   148	``leader_score``, each with the explicit ``higher_is_favorable`` direction.
   149	Required rules are sorted by metric identity and duplicates are rejected.  A
   150	caller may construct another version-1 policy with a different positive
   151	minimum or deterministic set of metric-direction rules; changing the policy
   152	changes both policy and decision identity.  The canonical policy binds the
   153	contract version, coverage minimum, sorted metric identities and directions,
   154	fixed all-member consistency, terminal, and successful comparability
   155	requirements, favorable-interpretation requirement for operator-review
   156	eligibility, and fixed non-authorizing semantics.
   157
   158	``minimumComparableMemberCount`` is only an advisory evidence-coverage
   159	threshold.  It is not statistical significance, confidence, causal evidence,
   160	profitability evidence, a guarantee of repeatability, or campaign success.  No
   161	materiality threshold or statistical, causal, profitability, or weighted
   162	campaign score is inferred.
   163
   164	For every exact member and required metric, the policy output preserves the
   165	Step 5a comparison classification and delta as optional evidence, then records
   166	a separate ``favorable``, ``neutral``, ``unfavorable``, or ``not_evaluable``
   167	judgment.  It never subtracts result and source values again.  Positive and
   168	negative meaning is determined only by the versioned metric-direction rule;
   169	an exact zero remains neutral.
   170
   171	Evidence sufficiency is separate from campaign interpretation.  Inconsistent,
   172	not-ready, cancelled, context-changed, metric-gap, missing-source,
   173	missing-result, unavailable, unsupported, missing-required-metric, and
   174	minimum-comparable-member conditions remain explicit conservative reasons.
   175	A terminal failure is adverse lifecycle evidence rather than a scientific
   176	metric gap.  Thus an all-failed campaign is interpreted ``unfavorable`` even
   177	though it lacks the minimum comparable-member evidence coverage; a failed
   178	member mixed with complete favorable evidence is ``mixed``.  Cancellation,
   179	inconsistency, unfinished lifecycle, or non-comparable successful evidence
   180	makes the campaign interpretation ``inconclusive``.  Complete comparable
   181	metric judgments yield ``favorable``, ``neutral``, ``unfavorable``, or
   182	``mixed`` without declaring the campaign successful.
   183
   184	Follow-up eligibility is only ``eligible_for_operator_review`` when every
   185	exact member is a comparable successful outcome, the minimum evidence coverage
   186	is met, all required metrics are comparable, and the campaign interpretation
   187	is ``favorable``.  Neutral, unfavorable, mixed, and inconclusive campaigns are
   188	``not_eligible``; sufficient evidence alone is not enough.  Failure,
   189	cancellation, incompleteness, inconsistency, context change, or a metric gap
   190	also leaves it ``not_eligible``.  This is an advisory screening result, not a
   191	recommendation, approval, or authorization: the immutable result always has
   192	``follow_up_authorized=false`` and ``follow_up_authorizing=false``.
   193
   194	Decision contract version 1 binds the complete policy canonical text and hash,
   195	the exact Step 5a assessment canonical text and hash, complete campaign
   196	approval and materialization identities, ordered member identities including
   197	the expected experiment, assessment lifecycle/consistency/outcome/diagnostics,
   198	evidence classifications, metric judgments, interpretations, reasons, counts,
   199	and explicit non-authorization result.  The complete upstream assessment
   200	canonical already binds Step 5a provenance; the direct fields are deliberately
   201	included as well because the decision carries and exposes those identities and
   202	assessment classifications.  ``observed_at`` remains display metadata and is
   203	excluded through the upstream assessment identity.  Unsupported policy or
   204	assessment contracts, invalid enums, duplicate policy metrics, duplicate or
   205	ambiguous top-level member identities, malformed identity/hash pairs, and
   206	invalid counts fail closed.
   207
   208	Step 5c adds no repository, service, formatter, CLI, migration, privilege,
   209	table, sequence, lock, write, scheduler call, worker behavior, experiment or
   210	campaign mutation, persistence, operator-review event, follow-up generation,
   211	approval, queueing, activation, retry, continuation, or automatic action.  The
   212	existing Step 5b CLI continues to report only the non-authoritative assessment
   213	built from authoritative persisted evidence and does not silently apply outcome
   214	policy.

exec
/bin/bash -lc "nl -ba CampaignOperations_Architecture_Output.txt | sed -n '1,145p' && nl -ba CampaignOperations_Architecture_Output.txt | sed -n '620,680p' && rg -n '"'^final$|''^assistant$|''^## Executive|''^# Campaign|''^## 1'"\\.|"'^## 2'"\\.|"'^Executive summary|''^Executive Summary'"' CampaignOperations_Architecture_Output.txt || true" in /Volumes/Developer SSD/ExpertAdvisor
 succeeded in 0ms:
     1	Reading prompt from stdin...
     2	OpenAI Codex v0.145.0
     3	--------
     4	workdir: /Volumes/Developer SSD/ExpertAdvisor
     5	model: gpt-5.6-sol
     6	provider: openai
     7	approval: never
     8	sandbox: danger-full-access
     9	reasoning effort: xhigh
    10	reasoning summaries: none
    11	session id: 019f89e9-56ac-7783-bb06-4a7fc6cfef79
    12	--------
    13	user
    14	Using High reasoning, perform a complete architectural design for the new Campaign Operations subsystem.
    15
    16	GOAL
    17
    18	Design the architecture that follows the completed Phase 6 governance work.
    19
    20	Do not implement code.
    21
    22	Do not modify existing production behavior.
    23
    24	Do not redesign completed Phases 4–6.
    25
    26	Treat all completed authority chains as fixed architecture.
    27
    28	BACKGROUND
    29
    30	The repository currently provides:
    31
    32	Planning
    33	→ Review
    34	→ Approval
    35	→ Materialization
    36	→ Execution
    37	→ Follow-up Review
    38	→ Governance
    39
    40	The governance chain is complete.
    41
    42	The next architectural concern is long-lived campaign operations.
    43
    44	Campaign Operations must remain a separate subsystem and must not transfer policy authority into the scheduler.
    45
    46	OBJECTIVES
    47
    48	Design the complete Campaign Operations architecture, including:
    49
    50	• subsystem responsibilities
    51	• ownership boundaries
    52	• lifecycle state machine
    53	• operational entities
    54	• database model
    55	• repository interfaces
    56	• service layer responsibilities
    57	• scheduler interaction contract
    58	• operator interaction
    59	• reconciliation
    60	• restart recovery
    61	• audit requirements
    62	• budgeting concepts
    63	• reservation concepts
    64	• execution requests
    65	• completion semantics
    66	• cancellation semantics
    67	• deterministic replay
    68	• failure recovery
    69	• concurrency model
    70	• transactional boundaries
    71	• privilege model
    72
    73	Clearly identify:
    74
    75	• what belongs inside Campaign Operations
    76	• what remains inside the scheduler
    77	• what remains inside recommendation governance
    78	• what belongs nowhere
    79
    80	REQUIREMENTS
    81
    82	Do not implement code.
    83
    84	Do not create migrations.
    85
    86	Do not design CLI commands.
    87
    88	Do not invent APIs unless required to explain architecture.
    89
    90	Do not redesign existing repository components.
    91
    92	Do not introduce automatic experiment creation.
    93
    94	Do not introduce autonomous decision making.
    95
    96	Do not change completed authority boundaries.
    97
    98	Produce:
    99
   100	1. Executive summary.
   101	2. Architectural rationale.
   102	3. Complete subsystem decomposition.
   103	4. State machine.
   104	5. Component interaction diagrams (textual).
   105	6. Data ownership.
   106	7. Transaction boundaries.
   107	8. Failure recovery model.
   108	9. Concurrency analysis.
   109	10. Security / privilege analysis.
   110	11. ADR recommendations.
   111	12. Recommended implementation phases.
   112	13. Risks.
   113	14. Open architectural questions.
   114
   115	The deliverable should be an architecture document suitable for implementation planning.
   116
   117	codex
   118	I’ll ground the design in the repository’s existing Phase 4–6 authority chains, schemas, and workflow terminology before drafting it. I’ll keep this strictly read-only: no code, migrations, CLI design, or repository edits.
   119	  → Map existing Phase 4–6 architecture, entities, workflows, and scheduler boundaries
   120	  • Derive Campaign Operations ownership, lifecycle, persistence, and interaction contracts
   121	  • Validate design against authority, concurrency, recovery, and audit constraints
   122	  • Deliver implementation-planning architecture document with ADRs, phases, risks, and open questions
   123	exec
   124	/bin/bash -lc "pwd && rg --files -g 'AGENTS.md' -g '*.md' -g '*.hpp' -g '*.h' -g '*.cpp' -g '*.mm' -g '*.sql' -g '*.xcodeproj/**' | sed -n '1,240p'" in /Volumes/Developer SSD/ExpertAdvisor
   125	 succeeded in 0ms:
   126	/Volumes/Developer SSD/ExpertAdvisor
   127	LSTM/ExperimentMetaAnalyzer.cpp
   128	LSTM/main.cpp
   129	LSTM/Tensor.cpp
   130	LSTM/LSTM.cpp
   131	meta_analysis_report.md
   132	MetaNN/MetaNN/facilities/_.h
   133	MetaNN/MetaNN/facilities/var_type_dict.h
   134	MetaNN/MetaNN/facilities/cont_metafuns/_.h
   135	MetaNN/MetaNN/facilities/cont_metafuns/set.h
   136	MetaNN/MetaNN/facilities/cont_metafuns/helpers.h
   137	MetaNN/MetaNN/facilities/cont_metafuns/value_sequential.h
   138	MetaNN/MetaNN/facilities/cont_metafuns/sequential.h
   139	MetaNN/MetaNN/facilities/cont_metafuns/multi_map.h
   140	MetaNN/MetaNN/facilities/cont_metafuns/map.h
   141	MetaNN/MetaNN/facilities/type_id.h
   142	MetaNN/MetaNN/facilities/null_param.h
   143	MetaNN/MetaNN/facilities/traits.h
   144	MetaNN/MetaNN/data/_.h
   145	MetaNN/MetaNN/data/bias_vector.h
   620	docs/Phase4DExperimentRecommendationCampaignPlanning.rst
   621	docs/Phase4DExperimentRecommendationCampaignProposalReview.rst
   622	docs/Phase4DExperimentRecommendationCampaignReview.rst
   623	docs/Phase5ExperimentRecommendationCampaignActivation.rst
   624	docs/Phase5ExperimentRecommendationCampaignExecution.rst
   625	docs/Phase5ExperimentRecommendationCampaignLaunch.rst
   626	docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst
   627	docs/Phase5ExperimentRecommendationCampaignStatus.rst
   628	docs/Phase6ARecommendationCampaignFollowUpProposal.rst
   629	docs/Phase6BRecommendationCampaignFollowUpProposalPersistence.rst
   630	docs/Phase6CRecommendationCampaignFollowUpProposalReview.rst
   631	docs/Phase6DRecommendationCampaignFollowUpProposalRatification.rst
   632	docs/architecture/Volume_VIII_Recommendation_Engine.md
   633	docs/architecture/Volume_XI_Scheduler.md
   634	docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md
   635	docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md
   636	docs/architecture/adr/ADR-0005-manual-recommendation-conversion.md
   637	docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md
   638
   639	time: command terminated abnormally
   640	       23.56 real         0.66 user         1.14 sys
   641	           147062784  maximum resident set size
   642	                   0  average shared memory size
   643	                   0  average unshared data size
   644	                   0  average unshared stack size
   645	               61270  page reclaims
   646	               12329  page faults
   647	                   0  swaps
   648	                   0  block input operations
   649	                   0  block output operations
   650	                 398  messages sent
   651	                5162  messages received
   652	                  26  signals received
   653	                2199  voluntary context switches
   654	               40308  involuntary context switches
   655	          7893238807  instructions retired
   656	          4854616510  cycles elapsed
   657	           119080760  peak memory footprint
   658	time: signal: Invalid argument

exec
/bin/bash -lc "nl -ba docs/architecture/Volume_I_Foundation.md | sed -n '300,390p' && nl -ba docs/architecture/Volume_I_Foundation.md | sed -n '500,610p' && nl -ba docs/architecture/README.md | sed -n '1,145p'" in /Volumes/Developer SSD/ExpertAdvisor
 succeeded in 0ms:
   300	Database transactions MUST NOT remain open while launching or waiting for
   301	workers. Process launch and database state transitions require an explicit
   302	claim/attempt protocol owned by the scheduler volume.
   303
   304	## 9. Concurrency philosophy
   305
   306	### 9.1 Narrow serialization
   307
   308	Concurrency control SHOULD serialize only records or semantic keys that can
   309	conflict. Global locks require exceptional justification.
   310
   311	### 9.2 Database authority
   312
   313	Correctness MUST rely on database-visible mechanisms such as row locks,
   314	constraints, unique indexes, transaction-scoped advisory locks, or compare-and-
   315	set updates—not process-local mutexes alone.
   316
   317	### 9.3 Defined outcomes
   318
   319	Every concurrent mutation contract MUST define winners, losers, retry
   320	behavior, idempotency, and resulting row counts. A conflict MUST be observable
   321	and deterministic; it MUST NOT create partial or duplicate history.
   322
   323	### 9.4 Idempotency
   324
   325	Idempotency is explicit, not assumed. A retry is idempotent only when the
   326	system can prove the request and persisted immutable result are equivalent.
   327	Otherwise, repeated terminal transitions fail or require an authoritative
   328	request identifier.
   329
   330	### 9.5 Lock order
   331
   332	Operations that lock multiple rows or tables MUST define a stable lock order
   333	and test deadlock-sensitive paths. Locks are released by transaction end, not
   334	by ad hoc process conventions.
   335
   336	## 10. Scheduler interaction rules
   337
   338	### 10.1 Scheduler ownership
   339
   340	The scheduler owns experiment lifecycle polling, capacity allocation, worker
   341	claiming, launch, recovery, and persisted operational transitions assigned in
   342	Volume XI. It does not own pure research policy merely because policy results
   343	may later influence work.
   344
   345	### 10.2 Explicit integration
   346
   347	A subsystem enters scheduler polling only through an accepted architectural
   348	change defining:
   349
   350	- trigger and eligibility;
   351	- durable claim and attempt state;
   352	- capacity and fairness;
   353	- retry and orphan recovery;
   354	- idempotency and concurrency;
   355	- operator controls and dry-run behavior;
   356	- observability and shutdown; and
   357	- regression boundaries for existing work.
   358
   359	### 10.3 Advisory isolation
   360
   361	Recommendation generation, scoring, ranking, and review are advisory unless a
   362	later accepted ADR defines a separate conversion boundary. Approval alone is
   363	not execution authorization, scheduler eligibility, queueing, or experiment
   364	creation.
   365
   366	### 10.4 Operational safety
   367
   368	Development and verification MUST NOT disturb running experiments. Tests use
   369	isolated schemas, rollback, mocks, or exact disposable IDs. Process signals,
   370	scheduler launch, experiment creation, and live-row mutation require explicit
   371	task authorization.
   372
   373	## 11. Testing philosophy
   374
   375	### 11.1 Test layers
   376
   377	Material features SHOULD include, as applicable:
   378
   379	- pure parsing, validation, identity, and calculation tests;
   380	- schema and migration tests;
   381	- repository mapping and transaction tests;
   382	- concurrency tests with independent connections;
   383	- service orchestration and structured-output tests;
   384	- CLI parsing, mutual-exclusion, and dispatch tests;
   385	- controlled integration tests with disposable fixtures; and
   386	- regression tests for adjacent subsystems.
   387
   388	### 11.2 Invariant-first tests
   389
   390	Tests SHOULD prove externally meaningful invariants, including unchanged state
   500	## 15. Versioning policy
   501
   502	### 15.1 Architecture versions
   503
   504	Architecture uses semantic versions:
   505
   506	- major: incompatible constitutional or domain-contract change;
   507	- minor: backward-compatible architectural capability or material extension;
   508	- patch: clarification or correction that does not change behavior.
   509
   510	### 15.2 Contract versions
   511
   512	Canonical identities, persisted policy grammars, schemas, serialized models,
   513	machine-output records, and other durable contracts carry their own explicit
   514	versions where compatibility requires them. Architecture version does not
   515	replace those versions.
   516
   517	### 15.3 Database versions
   518
   519	Database versioning is the ordered migration ledger. Migration filenames and
   520	checksums are permanent after application. Architecture records intent but
   521	does not renumber or rewrite history.
   522
   523	### 15.4 CLI compatibility
   524
   525	Existing command meanings and machine events SHOULD remain backward-compatible.
   526	Breaking changes require an ADR, migration/deprecation plan, version impact,
   527	and updates to tests and documentation.
   528
   529	## 16. Architecture governance
   530
   531	### 16.1 Changes requiring an ADR
   532
   533	An ADR is required for material changes to:
   534
   535	- identity or canonicalization;
   536	- durable ownership or schema strategy;
   537	- transaction or concurrency semantics;
   538	- scheduler responsibilities;
   539	- experiment lifecycle or status transitions;
   540	- recommendation-to-experiment conversion;
   541	- model/data compatibility contracts;
   542	- public CLI or machine-output compatibility;
   543	- security or runtime privilege boundaries; or
   544	- this constitution.
   545
   546	### 16.2 Decision lifecycle
   547
   548	ADRs progress through the statuses in `adr/README.md`. Accepted ADRs are
   549	immutable records; later decisions supersede rather than rewrite them, apart
   550	from clearly marked editorial corrections.
   551
   552	### 16.3 Review and approval
   553
   554	Architecture review evaluates alternatives, compatibility, migrations,
   555	operational risk, observability, testing, and rollback. Implementation SHOULD
   556	not begin while a required ADR remains merely proposed.
   557
   558	### 16.4 Emergency corrections
   559
   560	An urgent safety fix MAY precede complete documentation when delaying it would
   561	increase harm. The change MUST remain narrowly scoped, preserve evidence, and
   562	receive a follow-up ADR and volume update promptly.
   563
   564	## 17. Future extension policy
   565
   566	### 17.1 Capability gates
   567
   568	Future capabilities begin as documented proposals. They become executable only
   569	after authoritative inputs, identity, persistence, transaction behavior,
   570	concurrency, CLI, tests, and operational controls are defined.
   571
   572	### 17.2 No authority by implication
   573
   574	The presence of a future volume, schema placeholder, status, score, or approved
   575	ADR does not by itself enable runtime behavior. Implementation and operator
   576	authorization remain separate gates.
   577
   578	### 17.3 Compatibility-first extension
   579
   580	Extensions SHOULD add new versioned fields, tables, commands, or adapters while
   581	preserving existing contracts. If compatibility is impossible, the proposal
   582	must define migration, coexistence, rollback, and observability explicitly.
   583
   584	### 17.4 Autonomous research
   585
   586	Autonomous experimentation, recommendation conversion, profitability-driven
   587	selection, and self-modifying policies require explicit budgets, safety limits,
   588	audit trails, kill switches, and scheduler ownership. They MUST NOT emerge by
   589	connecting existing advisory components informally.
   590
   591	### 17.5 Data and model evolution
   592
   593	New features, labels, architectures, attention mechanisms, optimizers, and data
   594	sources require versioned semantic contracts. Git commit or binary name alone
   595	is insufficient as semantic identity.
   596
   597	## 18. Glossary
   598
   599	**ADR**
   600	Architecture Decision Record: an immutable explanation of a consequential
   601	decision, its context, and consequences.
   602
   603	**Advisory**
   604	Evidence or disposition that does not authorize execution.
   605
   606	**Canonical text**
   607	Versioned, deterministic, authoritative textual representation used for
   608	semantic equality.
   609
   610	**Continuation**
     1	# LSTM Research Platform Architecture
     2
     3	Status: Authoritative documentation framework
     4	Architecture version: 1.0.0
     5	Last revised: 2026-07-15
     6
     7	## 1. Purpose and authority
     8
     9	This directory is the durable architecture reference for the LSTM Research
    10	Platform. It explains the system's intended boundaries, invariants, ownership,
    11	and extension rules. It complements the source code, database migrations,
    12	tests, and operational runbooks; it does not replace them.
    13
    14	[Volume I](Volume_I_Foundation.md) is constitutional. Its rules apply across
    15	all platform components unless an accepted Architecture Decision Record (ADR)
    16	explicitly amends them. Volumes II–XII specify individual domains. ADRs record
    17	why consequential decisions were made and how later changes relate to them.
    18
    19	Normative words such as **MUST**, **MUST NOT**, **SHOULD**, and **MAY** have the
    20	meaning defined in Volume I §1.3.
    21
    22	## 2. Document organization
    23
    24	| Volume | Domain | Initial status |
    25	|---|---|---|
    26	| [I](Volume_I_Foundation.md) | Foundation and engineering constitution | Authoritative |
    27	| [II](Volume_II_Data_Pipeline.md) | Market-data ingestion, transformation, and feature inputs | Foundation outline |
    28	| [III](Volume_III_Label_Generation.md) | Targets, labels, horizons, and label provenance | Foundation outline |
    29	| [IV](Volume_IV_Model_Architecture.md) | Model topology, compatibility, and serialization | Foundation outline |
    30	| [V](Volume_V_Training_Engine.md) | Training, optimization, checkpoints, and resumption | Foundation outline |
    31	| [VI](Volume_VI_Inference_Evaluation.md) | Inference, evaluation, analysis, and evidence | Foundation outline |
    32	| [VII](Volume_VII_Experiment_Lifecycle.md) | Experiment identity, state, lineage, and continuation | Foundation outline |
    33	| [VIII](Volume_VIII_Recommendation_Engine.md) | Advisory recommendation generation, scoring, persistence, and review | Foundation outline |
    34	| [IX](Volume_IX_Trading_Profitability.md) | Trading simulation and profitability evidence | Reserved outline |
    35	| [X](Volume_X_Research_Automation.md) | Controlled research automation | Reserved outline |
    36	| [XI](Volume_XI_Scheduler.md) | Scheduler ownership, workers, recovery, and capacity | Foundation outline |
    37	| [XII](Volume_XII_Database.md) | PostgreSQL ownership, schema, migrations, and data integrity | Foundation outline |
    38
    39	The reusable [volume template](VOLUME_TEMPLATE.md) fixes the standard section
    40	order for Volumes II–XII. Stable headings permit references such as
    41	“Volume VIII §7.3.” Sections may grow subordinate headings without renumbering
    42	the standard top-level sections.
    43
    44	The [ADR index](adr/README.md) defines the ADR lifecycle and numbering scheme.
    45
    46	## 3. Architecture and implementation
    47
    48	Architecture describes intended contracts. Implementation demonstrates the
    49	currently shipped behavior. Neither silently overrides the other:
    50
    51	1. An implementation change MUST conform to accepted architecture.
    52	2. A discovered mismatch MUST be classified as an implementation defect, a
    53	   documentation defect, or a proposed architectural change.
    54	3. A material architectural change MUST be accepted in an ADR and reflected
    55	   in the affected volume before or with its implementation.
    56	4. Database migrations and tests remain executable evidence. Illustrative SQL
    57	   in these documents is non-executable unless explicitly identified as a
    58	   migration file.
    59	5. Historical design documents remain evidence for the phases they describe.
    60	   This hierarchy supplies the durable cross-phase organization.
    61
    62	Architecture documents MUST distinguish among:
    63
    64	- current implemented behavior;
    65	- required invariants;
    66	- planned but unimplemented behavior; and
    67	- rejected or deliberately deferred behavior.
    68
    69	An outline or future-extension section is not implementation authorization.
    70
    71	## 4. Proposing an architectural change
    72
    73	An architectural proposal follows this sequence:
    74
    75	1. Identify the affected volume sections and existing ADRs.
    76	2. State the problem and evidence without presupposing a solution.
    77	3. Draft a new numbered ADR with status `Proposed`.
    78	4. Describe compatibility, data migration, concurrency, operational safety,
    79	   rollback, testing, and documentation effects.
    80	5. Review the proposal against Volume I.
    81	6. Mark the ADR `Accepted`, `Rejected`, or retain it as `Proposed`.
    82	7. Update affected volume text and revision histories.
    83	8. Implement only the accepted scope, with tests that demonstrate its stated
    84	   invariants.
    85
    86	Editorial corrections that do not alter behavior, ownership, identity,
    87	transactions, concurrency, or compatibility do not require an ADR. They still
    88	update the affected document's revision history when material.
    89
    90	## 5. Review expectations
    91
    92	Architecture review MUST ask:
    93
    94	- Is the authoritative owner of each state transition clear?
    95	- Is semantic identity defined independently of row IDs and hash shortcuts?
    96	- Are transaction boundaries and concurrency outcomes explicit?
    97	- Does the database enforce every invariant it can enforce safely?
    98	- Are scheduler and worker responsibilities separated?
    99	- Are machine interfaces stable, escaped, and testable?
   100	- Are legacy rows and migration ledgers preserved deliberately?
   101	- Is deferred behavior clearly non-executable?
   102	- Can tests prove the contract without disturbing real experiments?
   103
   104	## 6. Relationship to Codex implementation prompts
   105
   106	Future Codex prompts SHOULD cite exact architecture sections and relevant ADRs,
   107	for example:
   108
   109	> Implement the approved change in Volume VIII §7.3 and ADR-0012. Preserve
   110	> Volume I §§5–10 and do not expand the scope described by ADR-0012.
   111
   112	Prompts MUST NOT redefine settled architecture casually. If a prompt conflicts
   113	with an accepted volume or ADR, the conflict must be reported before code or
   114	schema changes begin. A prompt may request an architectural amendment, but the
   115	ADR and volume update are part of that request and precede behavioral work.
   116
   117	Codex implementation reports SHOULD identify:
   118
   119	- architecture sections implemented or preserved;
   120	- ADRs followed or introduced;
   121	- migrations and compatibility decisions;
   122	- transaction and concurrency verification;
   123	- regression boundaries; and
   124	- deliberately deferred behavior.
   125
   126	Architecture references constrain implementation scope; they do not grant
   127	permission to operate schedulers, mutate production data, create experiments,
   128	or perform external actions unless the prompt separately authorizes them.
   129
   130	## 7. Document maintenance
   131
   132	- Markdown is the canonical format for this hierarchy.
   133	- Filenames and volume numbers are stable public references.
   134	- Headings use explicit section numbers.
   135	- Links SHOULD be relative within this directory.
   136	- Every volume and ADR includes a revision history.
   137	- Renaming or splitting a volume requires an accepted ADR and compatibility
   138	  aliases or clear replacement references.
   139	- Superseded text remains discoverable through Git history and ADR links.
   140
   141	## 8. Existing implementation references
   142
   143	The Phase 4 recommendation documents remain detailed implementation references:
   144
   145	- [Recommendation identity and candidate generation](../Phase4AExperimentRecommendationFoundation.rst)

exec
/bin/bash -lc 'git ls-files --error-unmatch CampaignOperations_Architecture_Output.txt >/dev/null 2>&1; echo target_tracked=$?; git ls-files --error-unmatch Phase6_Next_Architecture_Recommendation_Output.txt >/dev/null 2>&1; echo phase6_next_tracked=$?; git check-ignore -v CampaignOperations_Architecture_Output.txt Phase6_Next_Architecture_Recommendation_Output.txt CampaignOperations_Architecture_Recommendation_CEE_Output.txt || true; git tag --points-at HEAD; git branch --contains HEAD' in /Volumes/Developer SSD/ExpertAdvisor
 succeeded in 315ms:
target_tracked=1
phase6_next_tracked=1
.gitignore:6:*.txt	CampaignOperations_Architecture_Output.txt
.gitignore:6:*.txt	Phase6_Next_Architecture_Recommendation_Output.txt
.gitignore:6:*.txt	CampaignOperations_Architecture_Recommendation_CEE_Output.txt
* campaign-operations
  phase6d
  phase6e

exec
/bin/bash -lc "nl -ba Sources/ExperimentScheduler.cpp | sed -n '6450,6710p' && nl -ba Sources/ExperimentScheduler.cpp | sed -n '12470,12980p'" in /Volumes/Developer SSD/ExpertAdvisor
 succeeded in 0ms:
  6450	        insertSql << ") VALUES ($1";
  6451	        int param = 2;
  6452	        if (hasParentExperimentId)
  6453	            insertSql << ", $" << param++;
  6454	        insertSql << ", $" << param++ << ", $" << param++;
  6455	        if (hasSymbol)
  6456	            insertSql << ", $" << param++;
  6457	        if (hasPredictionHorizon)
  6458	            insertSql << ", $" << param++;
  6459	        insertSql << ") ON CONFLICT ";
  6460	        if (hasParentExperimentId)
  6461	            insertSql << "(parent_experiment_id, checkpoint_model_id, checkpoint_epoch) ";
  6462	        else
  6463	            insertSql << "(experiment_id, checkpoint_epoch, checkpoint_model_id) ";
  6464	        insertSql << "DO NOTHING RETURNING checkpoint_eval_id;";
  6465
  6466	        pqxx::result inserted;
  6467	        if (hasParentExperimentId && hasSymbol && hasPredictionHorizon)
  6468	            inserted = w.exec_params(insertSql.str(), experimentId, experimentId, epoch, modelId, symbol, horizon);
  6469	        else if (hasParentExperimentId && hasSymbol)
  6470	            inserted = w.exec_params(insertSql.str(), experimentId, experimentId, epoch, modelId, symbol);
  6471	        else if (hasParentExperimentId && hasPredictionHorizon)
  6472	            inserted = w.exec_params(insertSql.str(), experimentId, experimentId, epoch, modelId, horizon);
  6473	        else if (hasParentExperimentId)
  6474	            inserted = w.exec_params(insertSql.str(), experimentId, experimentId, epoch, modelId);
  6475	        else if (hasSymbol && hasPredictionHorizon)
  6476	            inserted = w.exec_params(insertSql.str(), experimentId, epoch, modelId, symbol, horizon);
  6477	        else if (hasSymbol)
  6478	            inserted = w.exec_params(insertSql.str(), experimentId, epoch, modelId, symbol);
  6479	        else if (hasPredictionHorizon)
  6480	            inserted = w.exec_params(insertSql.str(), experimentId, epoch, modelId, horizon);
  6481	        else
  6482	            inserted = w.exec_params(insertSql.str(), experimentId, epoch, modelId);
  6483	        if (!inserted.empty())
  6484	        {
  6485	            std::cout << "CHECKPOINT_EVAL_ENQUEUED"
  6486	                      << " experiment_id=" << experimentId
  6487	                      << " epoch=" << epoch
  6488	                      << " model_id=" << modelId
  6489	                      << std::endl;
  6490	        }
  6491	    }
  6492	}
  6493
  6494	CheckpointEvalRow RowToCheckpointEval(const pqxx::row& row)
  6495	{
  6496	    CheckpointEvalRow eval;
  6497	    eval.checkpointEvalId = row[0].as<long long>();
  6498	    eval.checkpointEpoch = row[1].as<int>();
  6499	    eval.checkpointModelId = row[2].as<long long>();
  6500	    eval.status = row[3].as<std::string>();
  6501	    eval.phase = row[4].as<std::string>();
  6502	    eval.workerPid = row[5].is_null() ? std::optional<int>{} : std::optional<int>{row[5].as<int>()};
  6503	    eval.inferLogPath = OptionalStringCell(row, 6);
  6504	    eval.analysisLogPath = OptionalStringCell(row, 7);
  6505	    eval.experiment.experimentId = row[8].as<long long>();
  6506	    eval.experiment.symbol = EA::CanonicalSymbol::Normalize(row[9].as<std::string>());
  6507	    eval.experiment.predictionHorizon = row[10].as<int>();
  6508	    eval.experiment.cNextThreshold = row[11].as<double>();
  6509	    eval.experiment.coreLrMult = OptionalDoubleCell(row, 12);
  6510	    eval.experiment.headLrMult = OptionalDoubleCell(row, 13);
  6511	    eval.experiment.targetEpochs = row[14].as<int>();
  6512	    eval.experiment.checkpointInterval = row[15].as<int>();
  6513	    eval.experiment.trainStart = row[16].as<std::string>();
  6514	    eval.experiment.trainEnd = row[17].as<std::string>();
  6515	    eval.experiment.inferStart = OptionalStringCell(row, 18);
  6516	    eval.experiment.inferEnd = OptionalStringCell(row, 19);
  6517	    eval.experiment.lastModelId = eval.checkpointModelId;
  6518	    eval.experiment.resumeModelId = OptionalLongLongCell(row, 20);
  6519	    eval.experiment.trainLogPath = OptionalStringCell(row, 21);
  6520	    eval.inferStartedEpoch = row[22].is_null() ? 0.0 : row[22].as<double>();
  6521	    eval.experiment.inferLogPath = eval.inferLogPath;
  6522	    eval.experiment.analysisLogPath = eval.analysisLogPath;
  6523	    return eval;
  6524	}
  6525
  6526	std::vector<CheckpointEvalRow> LoadCheckpointEvalRows(pqxx::work& w,
  6527	                                                      const std::string& status,
  6528	                                                      const std::string& phase)
  6529	{
  6530	    if (!CheckpointEvalTableExists(w))
  6531	        return {};
  6532	    pqxx::result rows = w.exec_params(
  6533	        "SELECT ce.checkpoint_eval_id, ce.checkpoint_epoch, ce.checkpoint_model_id, "
  6534	        "ce.status, ce.phase, ce.worker_pid, ce.infer_log_path, ce.analysis_log_path, "
  6535	        "e.experiment_id, e.symbol, e.prediction_horizon, e.c_next_threshold, "
  6536	        "e.core_lr_mult, e.head_lr_mult, e.target_epochs, e.checkpoint_interval, "
  6537	        "e.train_start::text, e.train_end::text, e.infer_start::text, e.infer_end::text, "
  6538	        "e.resume_model_id, e.train_log_path, "
  6539	        "extract(epoch from COALESCE(ce.infer_started_at, ce.started_at, ce.updated_at))::double precision "
  6540	        "FROM experiment_checkpoint_eval ce "
  6541	        "JOIN experiment e ON e.experiment_id = COALESCE(ce.parent_experiment_id, ce.experiment_id) "
  6542	        "WHERE ce.status = $1 AND ce.phase = $2 "
  6543	        "ORDER BY ce.created_at ASC, ce.checkpoint_eval_id ASC;",
  6544	        status,
  6545	        phase);
  6546
  6547	    std::vector<CheckpointEvalRow> evals;
  6548	    evals.reserve(rows.size());
  6549	    for (const auto& row : rows)
  6550	        evals.push_back(RowToCheckpointEval(row));
  6551	    return evals;
  6552	}
  6553
  6554	std::optional<CheckpointEvalRow> LoadCheckpointEvalById(pqxx::work& w,
  6555	                                                        long long checkpointEvalId)
  6556	{
  6557	    if (!CheckpointEvalTableExists(w))
  6558	        return std::nullopt;
  6559	    pqxx::result rows = w.exec_params(
  6560	        "SELECT ce.checkpoint_eval_id, ce.checkpoint_epoch, ce.checkpoint_model_id, "
  6561	        "ce.status, ce.phase, ce.worker_pid, ce.infer_log_path, ce.analysis_log_path, "
  6562	        "e.experiment_id, e.symbol, e.prediction_horizon, e.c_next_threshold, "
  6563	        "e.core_lr_mult, e.head_lr_mult, e.target_epochs, e.checkpoint_interval, "
  6564	        "e.train_start::text, e.train_end::text, e.infer_start::text, e.infer_end::text, "
  6565	        "e.resume_model_id, e.train_log_path, "
  6566	        "extract(epoch from COALESCE(ce.infer_started_at, ce.started_at, ce.updated_at))::double precision "
  6567	        "FROM experiment_checkpoint_eval ce "
  6568	        "JOIN experiment e ON e.experiment_id = COALESCE(ce.parent_experiment_id, ce.experiment_id) "
  6569	        "WHERE ce.checkpoint_eval_id = $1;",
  6570	        checkpointEvalId);
  6571	    if (rows.empty())
  6572	        return std::nullopt;
  6573	    return RowToCheckpointEval(rows[0]);
  6574	}
  6575
  6576	std::string ReadFileIfExists(const std::optional<std::string>& path)
  6577	{
  6578	    if (!path.has_value())
  6579	        return {};
  6580	    std::ifstream in{*path};
  6581	    if (!in)
  6582	        return {};
  6583	    std::ostringstream ss;
  6584	    ss << in.rdbuf();
  6585	    return ss.str();
  6586	}
  6587
  6588	std::string ReadFileIfExists(const std::string& path)
  6589	{
  6590	    std::ifstream in{path};
  6591	    if (!in)
  6592	        return {};
  6593	    std::ostringstream ss;
  6594	    ss << in.rdbuf();
  6595	    return ss.str();
  6596	}
  6597
  6598	int CountCompletedTrainingEpochsFromLog(const std::string& logText)
  6599	{
  6600	    if (logText.empty())
  6601	        return 0;
  6602	    const std::regex epochDoneRegex{R"(\bEPOCH_3CLASS_ACCURACY\b)"};
  6603	    const int epochsCompletedInThisProcess =
  6604	        static_cast<int>(std::distance(std::sregex_iterator(logText.begin(), logText.end(), epochDoneRegex),
  6605	                                       std::sregex_iterator()));
  6606
  6607	    int resumedFromEpoch = 0;
  6608	    const std::regex resumeRegex{R"(RESUME_COMPLETED_EPOCH=([0-9]+))"};
  6609	    for (std::sregex_iterator it{logText.begin(), logText.end(), resumeRegex}, end; it != end; ++it)
  6610	        resumedFromEpoch = std::stoi((*it)[1].str());
  6611
  6612	    int checkpointEpoch = 0;
  6613	    const std::regex checkpointRegex{R"(CHECKPOINT_SAVE_DONE[^[:cntrl:]]*epoch=([0-9]+))"};
  6614	    for (std::sregex_iterator it{logText.begin(), logText.end(), checkpointRegex}, end; it != end; ++it)
  6615	        checkpointEpoch = std::stoi((*it)[1].str());
  6616
  6617	    return std::max(checkpointEpoch, resumedFromEpoch + epochsCompletedInThisProcess);
  6618	}
  6619
  6620	void BackfillRunningTrainingProgressFromLogs(pqxx::work& w,
  6621	                                             const std::optional<long long>& experimentId)
  6622	{
  6623	    std::ostringstream sql;
  6624	    sql << "SELECT experiment_id, train_log_path, current_epoch "
  6625	        << "FROM experiment "
  6626	        << "WHERE status = 'running' "
  6627	        << "AND phase = 'train' "
  6628	        << "AND train_log_path IS NOT NULL ";
  6629	    if (experimentId.has_value())
  6630	        sql << "AND experiment_id = " << *experimentId << " ";
  6631	    sql << "ORDER BY experiment_id ASC;";
  6632
  6633	    pqxx::result rows = w.exec(sql.str());
  6634	    for (const auto& row : rows)
  6635	    {
  6636	        const long long id = row[0].as<long long>();
  6637	        const std::string logPath = row[1].as<std::string>();
  6638	        const std::optional<int> currentEpoch =
  6639	            row[2].is_null() ? std::optional<int>{} : std::optional<int>{row[2].as<int>()};
  6640	        const int completedEpochs = CountCompletedTrainingEpochsFromLog(ReadFileIfExists(logPath));
  6641	        if (completedEpochs <= 0)
  6642	            continue;
  6643	        if (currentEpoch.has_value() && *currentEpoch >= completedEpochs)
  6644	            continue;
  6645
  6646	        w.exec_params(
  6647	            "UPDATE experiment "
  6648	            "SET current_epoch = $1, current_operation = 'training', updated_at = now() "
  6649	            "WHERE experiment_id = $2 "
  6650	            "AND status = 'running' "
  6651	            "AND phase = 'train' "
  6652	            "AND (current_epoch IS NULL OR current_epoch < $1);",
  6653	            completedEpochs,
  6654	            id);
  6655	    }
  6656	}
  6657
  6658	void PersistDiscoveredRunningTrainingMetadata(const std::vector<SchedulerStatusJob>& jobs)
  6659	{
  6660	    pqxx::connection c{LstmDbConnectionString()};
  6661	    pqxx::work w{c};
  6662	    SetTransactionReadWrite(w);
  6663	    for (const auto& job : jobs)
  6664	    {
  6665	        if (job.status != "running" || job.phase != "train")
  6666	            continue;
  6667	        if (!job.pid.has_value() && !job.currentEpoch.has_value())
  6668	            continue;
  6669
  6670	        std::ostringstream sql;
  6671	        sql << "UPDATE experiment SET current_operation = 'training'";
  6672	        if (job.pid.has_value())
  6673	            sql << ", worker_pid = " << *job.pid;
  6674	        if (job.currentEpoch.has_value())
  6675	            sql << ", current_epoch = GREATEST(COALESCE(current_epoch, 0), " << *job.currentEpoch << ")";
  6676	        sql << ", updated_at = now() "
  6677	            << "WHERE experiment_id = " << job.experimentId << " "
  6678	            << "AND status = 'running' "
  6679	            << "AND phase = 'train';";
  6680	        w.exec(sql.str());
  6681	    }
  6682	    w.commit();
  6683	}
  6684
  6685	void WriteTextFile(const std::string& path, const std::string& text)
  6686	{
  6687	    std::ofstream out{path};
  6688	    out << text;
  6689	}
  6690
  6691	void AppendTextFile(const std::string& path, const std::string& text)
  6692	{
  6693	    std::ofstream out{path, std::ios::app};
  6694	    out << text;
  6695	    out.flush();
  6696	}
  6697
  6698	std::optional<long long> ExtractLastModelId(const std::string& text)
  6699	{
  6700	    std::regex idRegex{
  6701	        "(Saved model with model_id=|Created new model_id=|RESUME_SAVED_NEW_MODEL_ID=|CHECKPOINT_SAVE_DONE[^\\n]*model_id=)([0-9]+)"};
  6702	    std::optional<long long> last;
  6703	    for (auto it = std::sregex_iterator(text.begin(), text.end(), idRegex);
  6704	         it != std::sregex_iterator();
  6705	         ++it)
  6706	    {
  6707	        last = std::stoll((*it)[2].str());
  6708	    }
  6709	    return last;
  6710	}
 12470	            PersistObservedExperimentChild(w, child, exitCode, signalNumber, coreDumped);
 12471	        it = gSchedulerOwnedChildren.erase(it);
 12472	    }
 12473	}
 12474
 12475	void BeginSchedulerPollLogging(SchedulerEventLogState* logState)
 12476	{
 12477	    if (logState == nullptr)
 12478	        return;
 12479	    logState->currentSkipKeys.clear();
 12480	    logState->currentRunningPresentKeys.clear();
 12481	}
 12482
 12483	void FinishSchedulerPollLogging(SchedulerEventLogState* logState)
 12484	{
 12485	    if (logState == nullptr)
 12486	        return;
 12487	    logState->previousSkipKeys = logState->currentSkipKeys;
 12488	    logState->previousRunningPresentKeys = logState->currentRunningPresentKeys;
 12489	}
 12490
 12491	int RunTrainJobs(const SchedulerOptions& options,
 12492	                 const QueueSnapshot& snapshot,
 12493	                 SchedulerEventLogState* logState)
 12494	{
 12495	    pqxx::connection c{LstmDbConnectionString()};
 12496	    pqxx::work w{c};
 12497	    if (!options.dryRun)
 12498	        SetTransactionReadWrite(w);
 12499	    if (!RequireSchedulerTables(w))
 12500	        return 1;
 12501
 12502	    const std::vector<ExperimentRow> jobs = LoadPendingExperiments(w, "train");
 12503	    PhaseSchedulingStats stats;
 12504	    stats.phase = "train";
 12505	    stats.examined = static_cast<int>(jobs.size());
 12506	    stats.freeSlots = std::max(0, options.maxTrainProcs - RunningCountForPhase(snapshot, "train"));
 12507	    int freeSlots = stats.freeSlots;
 12508	    int rc = 0;
 12509
 12510	    EnsureLogDir(options.schedulerLogDir);
 12511	    for (const auto& job : jobs)
 12512	    {
 12513	        const std::optional<long long> resumeFrom =
 12514	            job.resumeModelId.has_value() ? job.resumeModelId : job.lastModelId;
 12515	        if (resumeFrom.has_value() && !ModelExists(w, *resumeFrom))
 12516	        {
 12517	            ++stats.skipped;
 12518	            LogSkip("train", job.experimentId, "model_not_found", logState, options.schedulerVerbose);
 12519	            if (!options.dryRun)
 12520	                MarkExperimentFailed(w, job, "train_model_not_found");
 12521	            continue;
 12522	        }
 12523	        if (RunningProcessExistsForExperiment(job, "train"))
 12524	        {
 12525	            ++stats.skipped;
 12526	            LogSkip("train", job.experimentId, "already_running", logState, options.schedulerVerbose);
 12527	            if (!options.dryRun)
 12528	            {
 12529	                MarkExperimentRunning(w, job, "train", LogPathFor(options, job, "train"));
 12530	                if (freeSlots > 0)
 12531	                    --freeSlots;
 12532	            }
 12533	            continue;
 12534	        }
 12535	        if (freeSlots <= 0)
 12536	        {
 12537	            ++stats.skipped;
 12538	            LogSkip("train", job.experimentId, "train_slots_full", logState, options.schedulerVerbose);
 12539	            continue;
 12540	        }
 12541
 12542	        const std::string logPath = LogPathFor(options, job, "train");
 12543	        const std::vector<std::string> command = BuildTrainCommand(options, job);
 12544	        const std::string commandDisplay = CommandForDisplay(command);
 12545	        if (options.dryRun)
 12546	        {
 12547	            std::cout << "EXPERIMENT_CHILD_COMMAND"
 12548	                      << ",experiment_id=" << job.experimentId
 12549	                      << ",phase=train"
 12550	                      << ",dry_run=1"
 12551	                      << ",argv=" << commandDisplay
 12552	                      << std::endl;
 12553	            --freeSlots;
 12554	            ++stats.launched;
 12555	            continue;
 12556	        }
 12557
 12558	        PersistExperimentRunningBeforeLaunch(job, "train", logPath);
 12559	        if (job.lastModelId.has_value() && !job.resumeModelId.has_value())
 12560	        {
 12561	            std::cout << "SCHEDULER_RESUME_FROM_LAST_MODEL"
 12562	                      << ",experiment_id=" << job.experimentId
 12563	                      << ",model_id=" << *job.lastModelId
 12564	                      << std::endl;
 12565	        }
 12566	        std::cout << "EXPERIMENT_STARTED"
 12567	                  << ",experiment_id=" << job.experimentId
 12568	                  << ",phase=train"
 12569	                  << std::endl;
 12570	        std::cout << "EXPERIMENT_CHILD_COMMAND"
 12571	                  << ",experiment_id=" << job.experimentId
 12572	                  << ",phase=train"
 12573	                  << ",argv=" << commandDisplay
 12574	                  << std::endl;
 12575	        PrintSchedulerExec(command);
 12576	        try
 12577	        {
 12578	            const pid_t pid = LaunchChildProcess(
 12579	                command, logPath, job.experimentId, "train", job.lastModelId);
 12580	            try
 12581	            {
 12582	                PersistExperimentWorkerPid(job, "train", pid);
 12583	            }
 12584	            catch (const std::exception& e)
 12585	            {
 12586	                std::cerr << "SCHEDULER_CHILD_PID_PERSIST_FAILED"
 12587	                          << ",experiment_id=" << job.experimentId
 12588	                          << ",worker_pid=" << pid
 12589	                          << ",phase=train"
 12590	                          << ",operation=train"
 12591	                          << ",error=" << e.what()
 12592	                          << std::endl;
 12593	                --freeSlots;
 12594	                ++stats.launched;
 12595	                rc = 1;
 12596	                continue;
 12597	            }
 12598	            --freeSlots;
 12599	            ++stats.launched;
 12600	        }
 12601	        catch (const std::exception& e)
 12602	        {
 12603	            const int launchError = SchedulerChildLaunchErrorNumber(e);
 12604	            MarkExperimentFailed(
 12605	                w, job, SchedulerLaunchFailureError(launchError));
 12606	            std::cout << "EXPERIMENT_FAILED"
 12607	                      << ",experiment_id=" << job.experimentId
 12608	                      << ",phase=train"
 12609	                      << ",error=train_launch_failed"
 12610	                      << std::endl;
 12611	            rc = 1;
 12612	        }
 12613	    }
 12614	    PrintPhaseSchedulingStats(stats, logState, options.schedulerVerbose);
 12615	    w.commit();
 12616	    return rc;
 12617	}
 12618
 12619	int RunInferJobs(const SchedulerOptions& options,
 12620	                 const QueueSnapshot& snapshot,
 12621	                 SchedulerEventLogState* logState)
 12622	{
 12623	    pqxx::connection c{LstmDbConnectionString()};
 12624	    pqxx::work w{c};
 12625	    if (!options.dryRun)
 12626	        SetTransactionReadWrite(w);
 12627	    if (!RequireSchedulerTables(w))
 12628	        return 1;
 12629
 12630	    const std::vector<ExperimentRow> jobs = LoadPendingExperiments(w, "infer");
 12631	    PhaseSchedulingStats stats;
 12632	    stats.phase = "infer";
 12633	    stats.examined = static_cast<int>(jobs.size());
 12634	    stats.freeSlots = std::max(0, options.maxInferProcs - RunningCountForPhase(snapshot, "infer"));
 12635	    int freeSlots = stats.freeSlots;
 12636	    int rc = 0;
 12637
 12638	    EnsureLogDir(options.schedulerLogDir);
 12639	    for (const auto& job : jobs)
 12640	    {
 12641	        if (!job.lastModelId.has_value())
 12642	        {
 12643	            ++stats.skipped;
 12644	            LogSkip("infer", job.experimentId, "missing_model", logState, options.schedulerVerbose);
 12645	            if (!options.dryRun)
 12646	                MarkExperimentFailed(w, job, "infer_missing_last_model_id");
 12647	            continue;
 12648	        }
 12649	        if (!ModelExists(w, *job.lastModelId))
 12650	        {
 12651	            ++stats.skipped;
 12652	            LogSkip("infer", job.experimentId, "model_not_found", logState, options.schedulerVerbose);
 12653	            if (!options.dryRun)
 12654	                MarkExperimentFailed(w, job, "infer_model_not_found");
 12655	            continue;
 12656	        }
 12657	        if (HasCompletedInferenceResult(w, job))
 12658	        {
 12659	            ++stats.skipped;
 12660	            LogSkip("infer", job.experimentId, "existing_inference", logState, options.schedulerVerbose);
 12661	            if (!options.dryRun)
 12662	            {
 12663	                std::cout << "SCHEDULER_SKIP_EXISTING_INFERENCE"
 12664	                          << ",experiment_id=" << job.experimentId
 12665	                          << ",model_id=" << *job.lastModelId
 12666	                          << std::endl;
 12667	                if (HasCompletedAnalysisResult(w, job))
 12668	                {
 12669	                    std::cout << "SCHEDULER_SKIP_EXISTING_ANALYSIS"
 12670	                              << ",experiment_id=" << job.experimentId
 12671	                              << ",model_id=" << *job.lastModelId
 12672	                              << std::endl;
 12673	                    MarkExperimentDone(w, job, "infer");
 12674	                }
 12675	                else
 12676	                {
 12677	                    MarkExperimentPendingPhase(w, job, "infer", "analyze");
 12678	                    std::cout << "SCHEDULER_ENQUEUE_ANALYZE"
 12679	                              << ",experiment_id=" << job.experimentId
 12680	                              << ",model_id=" << *job.lastModelId
 12681	                              << std::endl;
 12682	                }
 12683	            }
 12684	            continue;
 12685	        }
 12686	        if (HasValidInferenceLogPath(job))
 12687	        {
 12688	            ++stats.skipped;
 12689	            LogSkip("infer", job.experimentId, "valid_infer_log_path", logState, options.schedulerVerbose);
 12690	            if (!options.dryRun)
 12691	            {
 12692	                TransitionRecoveredInferenceToAnalyze(
 12693	                    w,
 12694	                    job,
 12695	                    "SCHEDULER_ORPHAN_RECOVERED_INFER_LOG",
 12696	                    "valid_infer_log_path");
 12697	            }
 12698	            continue;
 12699	        }
 12700	        if (const std::optional<std::string> discoveredLog = DiscoverValidInferenceLog(job);
 12701	            discoveredLog.has_value())
 12702	        {
 12703	            ++stats.skipped;
 12704	            LogSkip("infer", job.experimentId, "discovered_valid_infer_log", logState, options.schedulerVerbose);
 12705	            if (!options.dryRun)
 12706	            {
 12707	                TransitionRecoveredInferenceToAnalyze(
 12708	                    w,
 12709	                    job,
 12710	                    "SCHEDULER_ORPHAN_RECOVERED_INFER_LOG",
 12711	                    "discovered_valid_infer_log",
 12712	                    discoveredLog);
 12713	            }
 12714	            continue;
 12715	        }
 12716	        if (RunningProcessExistsForExperiment(job, "infer"))
 12717	        {
 12718	            ++stats.skipped;
 12719	            LogSkip("infer", job.experimentId, "already_running", logState, options.schedulerVerbose);
 12720	            if (!options.dryRun)
 12721	            {
 12722	                MarkExperimentRunning(w, job, "infer", LogPathFor(options, job, "infer"));
 12723	                if (freeSlots > 0)
 12724	                    --freeSlots;
 12725	            }
 12726	            continue;
 12727	        }
 12728	        if (freeSlots <= 0)
 12729	        {
 12730	            ++stats.skipped;
 12731	            LogSkip("infer", job.experimentId, "infer_slots_full", logState, options.schedulerVerbose);
 12732	            continue;
 12733	        }
 12734
 12735	        const std::string logPath = LogPathFor(options, job, "infer");
 12736	        const std::vector<std::string> command = BuildInferCommand(options, job);
 12737	        const std::string commandDisplay = CommandForDisplay(command);
 12738	        if (options.dryRun)
 12739	        {
 12740	            std::cout << "EXPERIMENT_CHILD_COMMAND"
 12741	                      << ",experiment_id=" << job.experimentId
 12742	                      << ",phase=infer"
 12743	                      << ",dry_run=1"
 12744	                      << ",argv=" << commandDisplay
 12745	                      << std::endl;
 12746	            --freeSlots;
 12747	            ++stats.launched;
 12748	            continue;
 12749	        }
 12750
 12751	        PersistExperimentRunningBeforeLaunch(job, "infer", logPath);
 12752	        std::cout << "EXPERIMENT_STARTED"
 12753	                  << ",experiment_id=" << job.experimentId
 12754	                  << ",phase=infer"
 12755	                  << std::endl;
 12756	        std::cout << "EXPERIMENT_CHILD_COMMAND"
 12757	                  << ",experiment_id=" << job.experimentId
 12758	                  << ",phase=infer"
 12759	                  << ",argv=" << commandDisplay
 12760	                  << std::endl;
 12761	        PrintSchedulerExec(command);
 12762	        try
 12763	        {
 12764	            const pid_t pid = LaunchChildProcess(
 12765	                command, logPath, job.experimentId, "infer", job.lastModelId);
 12766	            try
 12767	            {
 12768	                PersistExperimentWorkerPid(job, "infer", pid);
 12769	            }
 12770	            catch (const std::exception& e)
 12771	            {
 12772	                std::cerr << "SCHEDULER_CHILD_PID_PERSIST_FAILED"
 12773	                          << ",experiment_id=" << job.experimentId
 12774	                          << ",worker_pid=" << pid
 12775	                          << ",phase=infer"
 12776	                          << ",operation=infer"
 12777	                          << ",error=" << e.what()
 12778	                          << std::endl;
 12779	                --freeSlots;
 12780	                ++stats.launched;
 12781	                rc = 1;
 12782	                continue;
 12783	            }
 12784	            --freeSlots;
 12785	            ++stats.launched;
 12786	        }
 12787	        catch (const std::exception& e)
 12788	        {
 12789	            const int launchError = SchedulerChildLaunchErrorNumber(e);
 12790	            MarkExperimentFailed(
 12791	                w, job, SchedulerLaunchFailureError(launchError));
 12792	            std::cout << "EXPERIMENT_FAILED"
 12793	                      << ",experiment_id=" << job.experimentId
 12794	                      << ",phase=infer"
 12795	                      << ",error=infer_launch_failed"
 12796	                      << std::endl;
 12797	            rc = 1;
 12798	        }
 12799	    }
 12800	    PrintPhaseSchedulingStats(stats, logState, options.schedulerVerbose);
 12801	    w.commit();
 12802	    return rc;
 12803	}
 12804
 12805	int RunAnalyzeJobs(const SchedulerOptions& options,
 12806	                   const QueueSnapshot& snapshot,
 12807	                   SchedulerEventLogState* logState)
 12808	{
 12809	    pqxx::connection c{LstmDbConnectionString()};
 12810	    pqxx::work w{c};
 12811	    if (!options.dryRun)
 12812	        SetTransactionReadWrite(w);
 12813	    if (!RequireSchedulerTables(w))
 12814	        return 1;
 12815
 12816	    const std::vector<ExperimentRow> jobs = LoadPendingExperiments(w, "analyze");
 12817	    PhaseSchedulingStats stats;
 12818	    stats.phase = "analyze";
 12819	    stats.examined = static_cast<int>(jobs.size());
 12820	    stats.freeSlots = std::max(0, options.maxAnalyzeProcs - RunningCountForPhase(snapshot, "analyze"));
 12821	    int freeSlots = stats.freeSlots;
 12822	    int rc = 0;
 12823	    bool reportsNeeded = false;
 12824
 12825	    EnsureLogDir(options.schedulerLogDir);
 12826	    for (const auto& job : jobs)
 12827	    {
 12828	        if (!job.lastModelId.has_value())
 12829	        {
 12830	            ++stats.skipped;
 12831	            LogSkip("analyze", job.experimentId, "missing_model", logState, options.schedulerVerbose);
 12832	            if (!options.dryRun)
 12833	                MarkAnalyzeFailed(w, job.experimentId, "analyze_missing_last_model_id");
 12834	            continue;
 12835	        }
 12836	        if (!ModelExists(w, *job.lastModelId))
 12837	        {
 12838	            ++stats.skipped;
 12839	            LogSkip("analyze", job.experimentId, "model_not_found", logState, options.schedulerVerbose);
 12840	            if (!options.dryRun)
 12841	                MarkAnalyzeFailed(w, job.experimentId, "analyze_model_not_found");
 12842	            continue;
 12843	        }
 12844	        if (HasCompletedAnalysisResult(w, job))
 12845	        {
 12846	            ++stats.skipped;
 12847	            LogSkip("analyze", job.experimentId, "existing_analysis", logState, options.schedulerVerbose);
 12848	            if (!options.dryRun)
 12849	            {
 12850	                std::cout << "SCHEDULER_SKIP_EXISTING_ANALYSIS"
 12851	                          << ",experiment_id=" << job.experimentId
 12852	                          << ",model_id=" << *job.lastModelId
 12853	                          << std::endl;
 12854	                MarkExperimentDone(w, job, "analyze");
 12855	                reportsNeeded = true;
 12856	            }
 12857	            continue;
 12858	        }
 12859	        if (!HasCompletedInferenceResult(w, job) && !HasValidInferenceLogPath(job))
 12860	        {
 12861	            ++stats.skipped;
 12862	            LogSkip("analyze", job.experimentId, "infer_log_missing", logState, options.schedulerVerbose);
 12863	            continue;
 12864	        }
 12865	        if (RunningProcessExistsForExperiment(job, "analyze"))
 12866	        {
 12867	            ++stats.skipped;
 12868	            LogSkip("analyze", job.experimentId, "already_running", logState, options.schedulerVerbose);
 12869	            if (!options.dryRun)
 12870	            {
 12871	                MarkExperimentRunning(w, job, "analyze", LogPathFor(options, job, "analysis"));
 12872	                if (freeSlots > 0)
 12873	                    --freeSlots;
 12874	            }
 12875	            continue;
 12876	        }
 12877	        if (freeSlots <= 0)
 12878	        {
 12879	            ++stats.skipped;
 12880	            LogSkip("analyze", job.experimentId, "analyze_slots_full", logState, options.schedulerVerbose);
 12881	            continue;
 12882	        }
 12883
 12884	        const std::string logPath = LogPathFor(options, job, "analysis");
 12885	        const std::vector<std::string> command = BuildAnalyzeCommand(options, job);
 12886	        const std::string commandDisplay = CommandForDisplay(command);
 12887	        if (options.dryRun)
 12888	        {
 12889	            std::cout << "EXPERIMENT_ANALYSIS_STARTED"
 12890	                      << ",experiment_id=" << job.experimentId
 12891	                      << ",dry_run=1"
 12892	                      << std::endl;
 12893	            std::cout << "EXPERIMENT_CHILD_COMMAND"
 12894	                      << ",experiment_id=" << job.experimentId
 12895	                      << ",phase=analyze"
 12896	                      << ",dry_run=1"
 12897	                      << ",argv=" << commandDisplay
 12898	                      << std::endl;
 12899	            --freeSlots;
 12900	            ++stats.launched;
 12901	            continue;
 12902	        }
 12903
 12904	        PersistExperimentRunningBeforeLaunch(job, "analyze", logPath);
 12905	        std::cout << "EXPERIMENT_STARTED"
 12906	                  << ",experiment_id=" << job.experimentId
 12907	                  << ",phase=analyze"
 12908	                  << std::endl;
 12909	        std::cout << "EXPERIMENT_CHILD_COMMAND"
 12910	                  << ",experiment_id=" << job.experimentId
 12911	                  << ",phase=analyze"
 12912	                  << ",argv=" << commandDisplay
 12913	                  << std::endl;
 12914	        PrintSchedulerExec(command);
 12915	        try
 12916	        {
 12917	            const pid_t pid = LaunchChildProcess(
 12918	                command, logPath, job.experimentId, "analyze", job.lastModelId);
 12919	            try
 12920	            {
 12921	                PersistExperimentWorkerPid(job, "analyze", pid);
 12922	            }
 12923	            catch (const std::exception& e)
 12924	            {
 12925	                std::cerr << "SCHEDULER_CHILD_PID_PERSIST_FAILED"
 12926	                          << ",experiment_id=" << job.experimentId
 12927	                          << ",worker_pid=" << pid
 12928	                          << ",phase=analyze"
 12929	                          << ",operation=analyze"
 12930	                          << ",error=" << e.what()
 12931	                          << std::endl;
 12932	                --freeSlots;
 12933	                ++stats.launched;
 12934	                rc = 1;
 12935	                continue;
 12936	            }
 12937	            --freeSlots;
 12938	            ++stats.launched;
 12939	        }
 12940	        catch (const std::exception& e)
 12941	        {
 12942	            const int launchError = SchedulerChildLaunchErrorNumber(e);
 12943	            MarkAnalyzeFailed(
 12944	                w, job.experimentId, SchedulerLaunchFailureError(launchError));
 12945	            std::cout << "EXPERIMENT_FAILED"
 12946	                      << ",experiment_id=" << job.experimentId
 12947	                      << ",phase=analyze"
 12948	                      << ",error=analyze_launch_failed"
 12949	                      << std::endl;
 12950	            rc = 1;
 12951	        }
 12952	    }
 12953	    PrintPhaseSchedulingStats(stats, logState, options.schedulerVerbose);
 12954	    w.commit();
 12955	    if (reportsNeeded)
 12956	        TryGenerateExperimentReports(options);
 12957	    return rc;
 12958	}
 12959
 12960	bool SchedulerPidStillExists(std::optional<int> pid)
 12961	{
 12962	    if (!pid.has_value() || *pid <= 0)
 12963	        return false;
 12964	    if (::kill(static_cast<pid_t>(*pid), 0) == 0)
 12965	        return true;
 12966	    return errno == EPERM;
 12967	}
 12968
 12969	int CountRows(pqxx::work& w, const std::string& sql)
 12970	{
 12971	    return w.exec(sql).one_row()[0].as<int>();
 12972	}
 12973
 12974	void LogCheckpointEvalInferenceResultFound(const CheckpointEvalRow& eval,
 12975	                                           long long inferenceResultId)
 12976	{
 12977	    std::cout << "CHECKPOINT_EVAL_INFER_RESULT_FOUND"
 12978	              << ",checkpoint_eval_id=" << eval.checkpointEvalId
 12979	              << ",parent_experiment_id=" << eval.experiment.experimentId
 12980	              << ",checkpoint_model_id=" << eval.checkpointModelId

exec
/bin/bash -lc "rg -n '"'^std::vector<ExperimentRow> LoadPendingExperiments|LoadPendingExperiments'"\\(' Sources/ExperimentScheduler.cpp && nl -ba Sources/ExperimentScheduler.cpp | sed -n '5770,5965p' && rg -n 'experiment_worker|worker_attempt|attempt_id|claim' Database/migrations/*.sql Sources/ExperimentScheduler.cpp Headers/ExperimentScheduler.hpp | sed -n '1,500p'" in /Volumes/Developer SSD/ExpertAdvisor
 succeeded in 0ms:
4822:std::vector<ExperimentRow> LoadPendingExperiments(pqxx::work& w,
12502:    const std::vector<ExperimentRow> jobs = LoadPendingExperiments(w, "train");
12630:    const std::vector<ExperimentRow> jobs = LoadPendingExperiments(w, "infer");
12816:    const std::vector<ExperimentRow> jobs = LoadPendingExperiments(w, "analyze");
  5770	        {
  5771	            std::cerr << "CHECKPOINT_POLICY_REJECTED"
  5772	                      << ",experiment_id=" << experimentId
  5773	                      << ",reason=" << *configError
  5774	                      << std::endl;
  5775	            w.commit();
  5776	            return 1;
  5777	        }
  5778	        std::ostringstream sql;
  5779	        sql << "UPDATE experiment SET updated_at = now()";
  5780	        if (update.checkpointPolicySetKeys.count("min_leader_score"))
  5781	            sql << ", checkpoint_policy_min_leader_score = " << FormatDouble(*update.checkpointPolicyMinLeaderScore);
  5782	        if (update.checkpointPolicySetKeys.count("min_infer_accuracy"))
  5783	            sql << ", checkpoint_policy_min_infer_accuracy = " << FormatDouble(*update.checkpointPolicyMinInferAccuracy);
  5784	        if (update.checkpointPolicySetKeys.count("top_n"))
  5785	            sql << ", checkpoint_policy_top_n = " << *update.checkpointPolicyTopN;
  5786	        if (update.checkpointPolicySetKeys.count("scope"))
  5787	            sql << ", checkpoint_policy_scope = " << w.quote(update.checkpointPolicyScope);
  5788	        if (update.checkpointPolicySetKeys.count("stop_mode"))
  5789	            sql << ", checkpoint_policy_stop_mode = " << w.quote(update.checkpointPolicyStopMode);
  5790	        if (update.checkpointPolicySetKeys.count("grace_evals"))
  5791	            sql << ", checkpoint_policy_grace_evals = " << update.checkpointPolicyGraceEvals;
  5792	        sql << " WHERE experiment_id = " << experimentId << ";";
  5793	        w.exec(sql.str());
  5794	        w.commit();
  5795	        std::cout << "CHECKPOINT_POLICY_SET"
  5796	                  << ",experiment_id=" << experimentId
  5797	                  << ",rules=" << CheckpointPolicyRuleText(resultingConfig)
  5798	                  << std::endl;
  5799	        return 0;
  5800	    }
  5801
  5802	    w.commit();
  5803	    return 1;
  5804	}
  5805
  5806	std::string QueueSnapshotKey(const QueueSnapshot& snapshot)
  5807	{
  5808	    std::ostringstream key;
  5809	    key << snapshot.pendingTrain << "|"
  5810	        << snapshot.pendingInfer << "|"
  5811	        << snapshot.pendingAnalyze << "|"
  5812	        << snapshot.runningTrain << "|"
  5813	        << snapshot.runningInfer << "|"
  5814	        << snapshot.runningAnalyze;
  5815	    return key.str();
  5816	}
  5817
  5818	void PrintQueueSnapshot(const QueueSnapshot& snapshot,
  5819	                        SchedulerEventLogState* logState,
  5820	                        bool verbose)
  5821	{
  5822	    const std::string key = QueueSnapshotKey(snapshot);
  5823	    if (!verbose && logState != nullptr && logState->previousQueueKey == key)
  5824	        return;
  5825
  5826	    std::cout << "SCHEDULER_QUEUE"
  5827	              << ",pending_train=" << snapshot.pendingTrain
  5828	              << ",pending_infer=" << snapshot.pendingInfer
  5829	              << ",pending_analyze=" << snapshot.pendingAnalyze
  5830	              << ",running_train=" << snapshot.runningTrain
  5831	              << ",running_infer=" << snapshot.runningInfer
  5832	              << ",running_analyze=" << snapshot.runningAnalyze
  5833	              << std::endl;
  5834	    if (logState != nullptr)
  5835	        logState->previousQueueKey = key;
  5836	}
  5837
  5838	int RunningCountForPhase(const QueueSnapshot& snapshot, const std::string& phase)
  5839	{
  5840	    if (phase == "train")
  5841	        return snapshot.runningTrain;
  5842	    if (phase == "infer")
  5843	        return snapshot.runningInfer;
  5844	    if (phase == "analyze")
  5845	        return snapshot.runningAnalyze;
  5846	    return 0;
  5847	}
  5848
  5849	void LogSkip(const std::string& phase,
  5850	             long long experimentId,
  5851	             const std::string& reason,
  5852	             SchedulerEventLogState* logState,
  5853	             bool verbose)
  5854	{
  5855	    const std::string key = phase + "|" + std::to_string(experimentId) + "|" + reason;
  5856	    if (logState != nullptr)
  5857	        logState->currentSkipKeys.insert(key);
  5858	    if (!verbose &&
  5859	        logState != nullptr &&
  5860	        logState->previousSkipKeys.find(key) != logState->previousSkipKeys.end())
  5861	    {
  5862	        return;
  5863	    }
  5864
  5865	    std::cout << "SCHEDULER_SKIP_" << (phase == "train" ? "TRAIN" : phase == "infer" ? "INFER" : "ANALYZE")
  5866	              << ",experiment_id=" << experimentId
  5867	              << ",reason=" << reason
  5868	              << std::endl;
  5869	}
  5870
  5871	std::string PhaseSchedulingStatsKey(const PhaseSchedulingStats& stats)
  5872	{
  5873	    std::ostringstream key;
  5874	    key << stats.examined << "|"
  5875	        << stats.skipped << "|"
  5876	        << stats.launched << "|"
  5877	        << stats.freeSlots;
  5878	    return key.str();
  5879	}
  5880
  5881	void PrintPhaseSchedulingStats(const PhaseSchedulingStats& stats,
  5882	                               SchedulerEventLogState* logState,
  5883	                               bool verbose)
  5884	{
  5885	    const std::string key = PhaseSchedulingStatsKey(stats);
  5886	    if (!verbose &&
  5887	        logState != nullptr &&
  5888	        logState->previousPhaseKeys[stats.phase] == key)
  5889	    {
  5890	        return;
  5891	    }
  5892
  5893	    std::cout << "SCHEDULER_QUEUE_PHASE"
  5894	              << ",phase=" << stats.phase
  5895	              << ",examined=" << stats.examined
  5896	              << ",skipped=" << stats.skipped
  5897	              << ",launched=" << stats.launched
  5898	              << ",free_slots=" << stats.freeSlots
  5899	              << std::endl;
  5900	    if (logState != nullptr)
  5901	        logState->previousPhaseKeys[stats.phase] = key;
  5902	}
  5903
  5904	int FailInvalidSchedulerPhases(pqxx::work& w)
  5905	{
  5906	    pqxx::result rows = w.exec(
  5907	        "SELECT experiment_id, phase FROM experiment "
  5908	        "WHERE status IN ('pending', 'running') "
  5909	        "AND phase NOT IN ('train', 'infer', 'analyze', 'done') "
  5910	        "ORDER BY updated_at ASC, experiment_id ASC;");
  5911
  5912	    for (const auto& row : rows)
  5913	    {
  5914	        const long long experimentId = row[0].as<long long>();
  5915	        const std::string phase = row[1].as<std::string>();
  5916	        w.exec_params(
  5917	            "UPDATE experiment "
  5918	            "SET status = 'failed', "
  5919	            "exit_code = -1, "
  5920	            "error_message = $1, "
  5921	            "completed_at = now(), "
  5922	            "updated_at = now() "
  5923	            "WHERE experiment_id = $2;",
  5924	            "unknown_scheduler_phase:" + phase,
  5925	            experimentId);
  5926	        std::cout << "EXPERIMENT_FAILED"
  5927	                  << ",experiment_id=" << experimentId
  5928	                  << ",phase=" << phase
  5929	                  << ",reason=unknown_scheduler_phase"
  5930	                  << std::endl;
  5931	    }
  5932
  5933	    return rows.empty() ? 0 : 1;
  5934	}
  5935
  5936	void EnsureLogDir(const std::string& logDir)
  5937	{
  5938	    std::filesystem::create_directories(logDir);
  5939	}
  5940
  5941	std::string LogPathFor(const SchedulerOptions& options,
  5942	                              const ExperimentRow& experiment,
  5943	                              const std::string& phase)
  5944	{
  5945	    std::ostringstream oss;
  5946	    oss << options.schedulerLogDir
  5947	        << "/experiment_" << experiment.experimentId
  5948	        << "_" << EA::CanonicalSymbol::Normalize(experiment.symbol)
  5949	        << "_" << phase << ".log";
  5950	    return oss.str();
  5951	}
  5952
  5953	std::string BaseModelName(const ExperimentRow& experiment)
  5954	{
  5955	    std::ostringstream oss;
  5956	    oss << EA::CanonicalSymbol::Normalize(experiment.symbol)
  5957	        << "-experiment" << experiment.experimentId
  5958	        << "_h" << experiment.predictionHorizon
  5959	        << "_e" << experiment.targetEpochs;
  5960	    return oss.str();
  5961	}
  5962
  5963	bool RunningTrainProcessExistsForExperiment(const ExperimentRow& experiment)
  5964	{
  5965	    FILE* pipe = ::popen("ps -axo command", "r");
Database/migrations/011_experiment_live_progress.sql:6:CREATE INDEX IF NOT EXISTS experiment_worker_pid_idx

exec
/bin/bash -lc "nl -ba Sources/ExperimentScheduler.cpp | sed -n '4790,4895p' && nl -ba Sources/ExperimentScheduler.cpp | sed -n '11840,12130p'" in /Volumes/Developer SSD/ExpertAdvisor
 succeeded in 0ms:
  4790	}
  4791
  4792	std::optional<std::string> OptionalStringCell(const pqxx::row& row, int index)
  4793	{
  4794	    if (row[index].is_null())
  4795	        return std::nullopt;
  4796	    return row[index].as<std::string>();
  4797	}
  4798
  4799	ExperimentRow RowToExperiment(const pqxx::row& row)
  4800	{
  4801	    ExperimentRow experiment;
  4802	    experiment.experimentId = row[0].as<long long>();
  4803	    experiment.symbol = EA::CanonicalSymbol::Normalize(row[1].as<std::string>());
  4804	    experiment.predictionHorizon = row[2].as<int>();
  4805	    experiment.cNextThreshold = row[3].as<double>();
  4806	    experiment.coreLrMult = OptionalDoubleCell(row, 4);
  4807	    experiment.headLrMult = OptionalDoubleCell(row, 5);
  4808	    experiment.targetEpochs = row[6].as<int>();
  4809	    experiment.checkpointInterval = row[7].as<int>();
  4810	    experiment.trainStart = row[8].as<std::string>();
  4811	    experiment.trainEnd = row[9].as<std::string>();
  4812	    experiment.inferStart = OptionalStringCell(row, 10);
  4813	    experiment.inferEnd = OptionalStringCell(row, 11);
  4814	    experiment.lastModelId = OptionalLongLongCell(row, 12);
  4815	    experiment.resumeModelId = OptionalLongLongCell(row, 13);
  4816	    experiment.trainLogPath = OptionalStringCell(row, 14);
  4817	    experiment.inferLogPath = OptionalStringCell(row, 15);
  4818	    experiment.analysisLogPath = OptionalStringCell(row, 16);
  4819	    return experiment;
  4820	}
  4821
  4822	std::vector<ExperimentRow> LoadPendingExperiments(pqxx::work& w,
  4823	                                                        const std::string& phase)
  4824	{
  4825	    pqxx::result rows = w.exec_params(
  4826	        "SELECT experiment_id, symbol, prediction_horizon, c_next_threshold, "
  4827	        "core_lr_mult, head_lr_mult, target_epochs, checkpoint_interval, "
  4828	        "train_start::text, train_end::text, infer_start::text, infer_end::text, "
  4829	        "last_model_id, resume_model_id, train_log_path, infer_log_path, analysis_log_path "
  4830	        "FROM experiment "
  4831	        "WHERE status = 'pending' AND phase = $1 "
  4832	        "ORDER BY updated_at ASC, experiment_id ASC;",
  4833	        phase);
  4834
  4835	    std::vector<ExperimentRow> experiments;
  4836	    experiments.reserve(rows.size());
  4837	    for (const auto& row : rows)
  4838	        experiments.push_back(RowToExperiment(row));
  4839	    return experiments;
  4840	}
  4841
  4842	std::vector<RunningExperimentState> LoadRunningExperiments(pqxx::work& w)
  4843	{
  4844	    pqxx::result rows = w.exec(
  4845	        "SELECT experiment_id, symbol, prediction_horizon, c_next_threshold, "
  4846	        "core_lr_mult, head_lr_mult, target_epochs, checkpoint_interval, "
  4847	        "train_start::text, train_end::text, infer_start::text, infer_end::text, "
  4848	        "last_model_id, resume_model_id, train_log_path, infer_log_path, analysis_log_path, phase, worker_pid, "
  4849	        "extract(epoch from COALESCE(worker_started_at, updated_at))::double precision "
  4850	        "FROM experiment "
  4851	        "WHERE status = 'running' "
  4852	        "ORDER BY updated_at ASC, experiment_id ASC;");
  4853
  4854	    std::vector<RunningExperimentState> experiments;
  4855	    experiments.reserve(rows.size());
  4856	    for (const auto& row : rows)
  4857	        experiments.push_back(RunningExperimentState{
  4858	            RowToExperiment(row),
  4859	            row[17].as<std::string>(),
  4860	            row[18].is_null() ? std::nullopt : std::optional<int>{row[18].as<int>()},
  4861	            row[19].as<double>()});
  4862	    return experiments;
  4863	}
  4864
  4865	int RecoverOrphanedRunningExperiments(pqxx::work& w,
  4866	                                      SchedulerEventLogState* logState,
  4867	                                      bool verbose);
  4868
  4869	QueueSnapshot LoadQueueSnapshot(pqxx::work& w)
  4870	{
  4871	    QueueSnapshot snapshot;
  4872	    pqxx::result rows = w.exec(
  4873	        "SELECT phase, status, count(*) "
  4874	        "FROM experiment "
  4875	        "WHERE status IN ('pending', 'running') "
  4876	        "AND phase IN ('train', 'infer', 'analyze') "
  4877	        "GROUP BY phase, status;");
  4878	    for (const auto& row : rows)
  4879	    {
  4880	        const std::string phase = row[0].as<std::string>();
  4881	        const std::string status = row[1].as<std::string>();
  4882	        const int count = row[2].as<int>();
  4883	        if (phase == "train" && status == "pending")
  4884	            snapshot.pendingTrain = count;
  4885	        else if (phase == "infer" && status == "pending")
  4886	            snapshot.pendingInfer = count;
  4887	        else if (phase == "analyze" && status == "pending")
  4888	            snapshot.pendingAnalyze = count;
  4889	        else if (phase == "train" && status == "running")
  4890	            snapshot.runningTrain = count;
  4891	        else if (phase == "infer" && status == "running")
  4892	            snapshot.runningInfer = count;
  4893	        else if (phase == "analyze" && status == "running")
  4894	            snapshot.runningAnalyze = count;
  4895	    }
 11840	        logState->previousRunningPresentKeys.find(key) != logState->previousRunningPresentKeys.end())
 11841	    {
 11842	        return;
 11843	    }
 11844
 11845	    if (phase == "infer")
 11846	    {
 11847	        std::cout << "SCHEDULER_ORPHAN_INFER_STILL_RUNNING"
 11848	                  << ",experiment_id=" << experiment.experimentId
 11849	                  << ",model_id=" << (experiment.lastModelId.has_value() ? std::to_string(*experiment.lastModelId) : "none")
 11850	                  << ",phase=infer"
 11851	                  << ",reason=matching_process_exists"
 11852	                  << std::endl;
 11853	    }
 11854	    else
 11855	    {
 11856	        std::cout << "SCHEDULER_RUNNING_EXPERIMENT_PRESENT"
 11857	                  << ",experiment_id=" << experiment.experimentId
 11858	                  << ",phase=" << phase
 11859	                  << std::endl;
 11860	    }
 11861	}
 11862
 11863	int RecoverOrphanedRunningExperiments(pqxx::work& w,
 11864	                                      SchedulerEventLogState* logState,
 11865	                                      bool verbose)
 11866	{
 11867	    int recoveredOrFailed = 0;
 11868	    const std::vector<RunningExperimentState> runningExperiments = LoadRunningExperiments(w);
 11869	    for (const auto& state : runningExperiments)
 11870	    {
 11871	        const ExperimentRow& experiment = state.experiment;
 11872	        const std::string& phase = state.phase;
 11873	        const auto ownedChild = std::find_if(
 11874	            gSchedulerOwnedChildren.begin(),
 11875	            gSchedulerOwnedChildren.end(),
 11876	            [&](const auto& entry) {
 11877	                return entry.second.experimentId == experiment.experimentId &&
 11878	                       entry.second.phase == phase;
 11879	            });
 11880	        if (ownedChild != gSchedulerOwnedChildren.end())
 11881	        {
 11882	            LogRunningExperimentPresent(experiment, phase, logState, verbose);
 11883	            continue;
 11884	        }
 11885	        if (RunningProcessExistsForExperiment(experiment, phase))
 11886	        {
 11887	            LogRunningExperimentPresent(experiment, phase, logState, verbose);
 11888	            continue;
 11889	        }
 11890
 11891	        bool successfulCompletionRecorded = false;
 11892	        if (phase == "infer")
 11893	        {
 11894	            successfulCompletionRecorded =
 11895	                HasCompletedInferenceResultForAttempt(
 11896	                    w, experiment, state.attemptStartedEpoch) ||
 11897	                HasValidInferenceLogPathForAttempt(
 11898	                    experiment, state.attemptStartedEpoch) ||
 11899	                DiscoverValidInferenceLogForAttempt(
 11900	                    experiment, state.attemptStartedEpoch).has_value();
 11901	        }
 11902	        else if (phase == "analyze")
 11903	        {
 11904	            successfulCompletionRecorded =
 11905	                HasCompletedAnalysisResultForAttempt(
 11906	                    w, experiment, state.attemptStartedEpoch);
 11907	        }
 11908	        else if (phase == "train")
 11909	        {
 11910	            successfulCompletionRecorded =
 11911	                FindLatestModelForExperimentSince(
 11912	                    w,
 11913	                    experiment.experimentId,
 11914	                    state.attemptStartedEpoch).has_value();
 11915	            if (!successfulCompletionRecorded)
 11916	            {
 11917	                const std::string existingTrainLog = ReadFileIfExists(experiment.trainLogPath);
 11918	                std::optional<long long> loggedModelId =
 11919	                    ExtractCompletedTrainModelId(existingTrainLog);
 11920	                if (!loggedModelId.has_value())
 11921	                    loggedModelId = ExtractCheckpointModelId(existingTrainLog);
 11922	                successfulCompletionRecorded =
 11923	                    loggedModelId.has_value() &&
 11924	                    ModelWasCreatedForAttempt(
 11925	                        w,
 11926	                        *loggedModelId,
 11927	                        experiment.experimentId,
 11928	                        state.attemptStartedEpoch);
 11929	            }
 11930	        }
 11931
 11932	        ++recoveredOrFailed;
 11933	        if (ShouldReconcileSchedulerOrphan(
 11934	                false,
 11935	                false,
 11936	                successfulCompletionRecorded))
 11937	        {
 11938	            std::cout << "SCHEDULER_ORPHAN_DETECTED"
 11939	                      << ",experiment_id=" << experiment.experimentId
 11940	                      << ",phase=" << phase
 11941	                      << ",pid=" << (state.workerPid.has_value() ? std::to_string(*state.workerPid) : "none")
 11942	                      << ",ownership=previous_scheduler_or_unknown"
 11943	                      << std::endl;
 11944	        }
 11945
 11946	        if (phase == "infer")
 11947	        {
 11948	            if (HasCompletedInferenceResultForAttempt(
 11949	                    w, experiment, state.attemptStartedEpoch))
 11950	            {
 11951	                TransitionRecoveredInferenceToAnalyze(
 11952	                    w,
 11953	                    experiment,
 11954	                    "SCHEDULER_ORPHAN_RECOVERED_INFERENCE_RESULT",
 11955	                    "completed_inference_eval_result");
 11956	            }
 11957	            else if (HasValidInferenceLogPathForAttempt(
 11958	                         experiment, state.attemptStartedEpoch))
 11959	            {
 11960	                TransitionRecoveredInferenceToAnalyze(
 11961	                    w,
 11962	                    experiment,
 11963	                    "SCHEDULER_ORPHAN_RECOVERED_INFER_LOG",
 11964	                    "valid_infer_log_path");
 11965	            }
 11966	            else if (const std::optional<std::string> discoveredLog =
 11967	                         DiscoverValidInferenceLogForAttempt(
 11968	                             experiment, state.attemptStartedEpoch);
 11969	                     discoveredLog.has_value())
 11970	            {
 11971	                TransitionRecoveredInferenceToAnalyze(
 11972	                    w,
 11973	                    experiment,
 11974	                    "SCHEDULER_ORPHAN_RECOVERED_INFER_LOG",
 11975	                    "discovered_valid_infer_log",
 11976	                    discoveredLog);
 11977	            }
 11978	            else
 11979	            {
 11980	                const std::string error = "orphaned_running_infer_no_process_no_result";
 11981	                MarkExperimentFailed(w, experiment, error);
 11982	                std::cout << "SCHEDULER_ORPHAN_MARKED_FAILED"
 11983	                          << ",experiment_id=" << experiment.experimentId
 11984	                          << ",model_id=" << (experiment.lastModelId.has_value() ? std::to_string(*experiment.lastModelId) : "none")
 11985	                          << ",phase=infer"
 11986	                          << ",error=" << error
 11987	                          << std::endl;
 11988	            }
 11989	            continue;
 11990	        }
 11991
 11992	        if (phase == "analyze")
 11993	        {
 11994	            if (HasCompletedAnalysisResultForAttempt(
 11995	                    w, experiment, state.attemptStartedEpoch))
 11996	            {
 11997	                std::cout << "SCHEDULER_ORPHAN_RECOVERED_ANALYSIS"
 11998	                          << ",experiment_id=" << experiment.experimentId
 11999	                          << ",model_id=" << (experiment.lastModelId.has_value() ? std::to_string(*experiment.lastModelId) : "none")
 12000	                          << std::endl;
 12001	                MarkExperimentDone(w, experiment, "analyze");
 12002	            }
 12003	            else
 12004	            {
 12005	                const std::string error = "orphaned_running_analyze_no_process_no_result";
 12006	                MarkExperimentFailed(w, experiment, error);
 12007	                std::cout << "SCHEDULER_ORPHAN_MARKED_FAILED"
 12008	                          << ",experiment_id=" << experiment.experimentId
 12009	                          << ",error=" << error
 12010	                          << std::endl;
 12011	            }
 12012	            continue;
 12013	        }
 12014
 12015	        if (phase != "train")
 12016	        {
 12017	            const std::string error = "orphaned_running_invalid_phase";
 12018	            MarkExperimentFailed(w, experiment, error);
 12019	            std::cout << "SCHEDULER_ORPHAN_MARKED_FAILED"
 12020	                      << ",experiment_id=" << experiment.experimentId
 12021	                      << ",error=" << error
 12022	                      << std::endl;
 12023	            continue;
 12024	        }
 12025
 12026	        const std::optional<long long> linkedModelId =
 12027	            FindLatestModelForExperimentSince(
 12028	                w,
 12029	                experiment.experimentId,
 12030	                state.attemptStartedEpoch);
 12031	        if (linkedModelId.has_value())
 12032	        {
 12033	            std::cout << "SCHEDULER_ORPHAN_RECOVERED_MODEL"
 12034	                      << ",experiment_id=" << experiment.experimentId
 12035	                      << ",model_id=" << *linkedModelId
 12036	                      << ",source=model_experiment_id"
 12037	                      << std::endl;
 12038	            if (!RecoverTrainOrphanFromModel(w, experiment, *linkedModelId))
 12039	            {
 12040	                const std::string error = "orphaned_running_train_model_unusable";
 12041	                MarkExperimentFailed(w, experiment, error + ";model_id=" + std::to_string(*linkedModelId));
 12042	                std::cout << "SCHEDULER_ORPHAN_FAILED"
 12043	                          << ",experiment_id=" << experiment.experimentId
 12044	                          << ",reason=model_unusable"
 12045	                          << std::endl;
 12046	            }
 12047	            continue;
 12048	        }
 12049
 12050	        const std::string logText = ReadFileIfExists(experiment.trainLogPath);
 12051	        std::optional<long long> finalModelId = ExtractCompletedTrainModelId(logText);
 12052	        if (finalModelId.has_value() &&
 12053	            !ModelWasCreatedForAttempt(
 12054	                w,
 12055	                *finalModelId,
 12056	                experiment.experimentId,
 12057	                state.attemptStartedEpoch))
 12058	            finalModelId.reset();
 12059	        if (finalModelId.has_value())
 12060	        {
 12061	            std::cout << "SCHEDULER_ORPHAN_RECOVERED_MODEL"
 12062	                      << ",experiment_id=" << experiment.experimentId
 12063	                      << ",model_id=" << *finalModelId
 12064	                      << std::endl;
 12065	            if (!RecoverTrainOrphanFromModel(w, experiment, *finalModelId))
 12066	            {
 12067	                const std::string error = "orphaned_running_train_model_unusable";
 12068	                MarkExperimentFailed(w, experiment, error + ";model_id=" + std::to_string(*finalModelId));
 12069	                std::cout << "SCHEDULER_ORPHAN_FAILED"
 12070	                          << ",experiment_id=" << experiment.experimentId
 12071	                          << ",reason=model_unusable"
 12072	                          << std::endl;
 12073	            }
 12074	            continue;
 12075	        }
 12076
 12077	        std::optional<long long> checkpointModelId = ExtractCheckpointModelId(logText);
 12078	        if (checkpointModelId.has_value() &&
 12079	            !ModelWasCreatedForAttempt(
 12080	                w,
 12081	                *checkpointModelId,
 12082	                experiment.experimentId,
 12083	                state.attemptStartedEpoch))
 12084	            checkpointModelId.reset();
 12085	        if (checkpointModelId.has_value())
 12086	        {
 12087	            std::cout << "SCHEDULER_ORPHAN_RECOVERED_CHECKPOINT"
 12088	                      << ",experiment_id=" << experiment.experimentId
 12089	                      << ",checkpoint_model_id=" << *checkpointModelId
 12090	                      << std::endl;
 12091	            if (!RecoverTrainOrphanFromModel(w, experiment, *checkpointModelId))
 12092	            {
 12093	                const std::string error = "orphaned_running_train_checkpoint_unusable";
 12094	                MarkExperimentFailed(w, experiment, error + ";checkpoint_model_id=" + std::to_string(*checkpointModelId));
 12095	                std::cout << "SCHEDULER_ORPHAN_FAILED"
 12096	                          << ",experiment_id=" << experiment.experimentId
 12097	                          << ",reason=checkpoint_unusable"
 12098	                          << std::endl;
 12099	            }
 12100	            continue;
 12101	        }
 12102
 12103	        const std::string error = "orphaned_running_train_no_process_no_model";
 12104	        MarkExperimentFailed(w, experiment, error);
 12105	        std::cout << "SCHEDULER_ORPHAN_FAILED"
 12106	                  << ",experiment_id=" << experiment.experimentId
 12107	                  << ",reason=no_recoverable_checkpoint"
 12108	                  << std::endl;
 12109	    }
 12110	    return recoveredOrFailed;
 12111	}
 12112
 12113	[[maybe_unused]] void CompleteInferPhase(pqxx::work& w,
 12114	                               const ExperimentRow& experiment,
 12115	                               int exitCode,
 12116	                               const std::string& commandDisplay)
 12117	{
 12118	    if (!experiment.lastModelId.has_value())
 12119	    {
 12120	        MarkExperimentFailed(w, experiment, "infer_missing_last_model_id", -1);
 12121	        return;
 12122	    }
 12123
 12124	    if (exitCode == 0)
 12125	    {
 12126	        if (HasCompletedAnalysisResult(w, experiment))
 12127	        {
 12128	            std::cout << "SCHEDULER_SKIP_EXISTING_ANALYSIS"
 12129	                      << ",experiment_id=" << experiment.experimentId
 12130	                      << ",model_id=" << *experiment.lastModelId

exec
/bin/bash -lc "nl -ba Database/migrations/024_continuation_sequence_and_worker_attempt_hardening.sql && rg -n 'worker_started_at|current_operation|worker_pid|attempt' Database/migrations/*.sql | sed -n '1,600p'" in /Volumes/Developer SSD/ExpertAdvisor
 succeeded in 0ms:
     1	ALTER TABLE experiment
     2	    ADD COLUMN IF NOT EXISTS worker_started_at timestamptz;
     3
     4	CREATE OR REPLACE FUNCTION continuation_target_sequence_is_valid(sequence integer[])
     5	RETURNS boolean
     6	LANGUAGE sql
     7	IMMUTABLE
     8	STRICT
     9	AS $$
    10	    SELECT cardinality(sequence) > 0
    11	       AND NOT EXISTS (
    12	            SELECT 1
    13	            FROM generate_subscripts(sequence, 1) AS position
    14	            WHERE sequence[position] <= 0
    15	               OR (
    16	                    position > array_lower(sequence, 1)
    17	                    AND sequence[position] <= sequence[position - 1]
    18	               )
    19	       );
    20	$$;
    21
    22	DO $$
    23	BEGIN
    24	    IF EXISTS (
    25	        SELECT 1 FROM pg_constraint
    26	        WHERE conname = 'experiment_continuation_policy_progression_shape_check'
    27	    ) THEN
    28	        ALTER TABLE experiment
    29	            DROP CONSTRAINT experiment_continuation_policy_progression_shape_check;
    30	    END IF;
    31
    32	    ALTER TABLE experiment
    33	        ADD CONSTRAINT experiment_continuation_policy_progression_shape_check
    34	        CHECK (
    35	            CASE COALESCE(
    36	                continuation_policy_progression_mode,
    37	                'fixed_increment'
    38	            )
    39	                WHEN 'fixed_increment' THEN
    40	                    continuation_policy_target_sequence IS NULL
    41	                WHEN 'target_sequence' THEN
    42	                    continuation_policy_target_increment IS NULL
    43	                    AND continuation_policy_target_sequence IS NOT NULL
    44	                    AND continuation_target_sequence_is_valid(
    45	                        continuation_policy_target_sequence
    46	                    )
    47	                    AND (
    48	                        continuation_policy_max_target_epochs IS NULL
    49	                        OR continuation_policy_max_target_epochs =
    50	                           continuation_policy_target_sequence[
    51	                               cardinality(continuation_policy_target_sequence)
    52	                           ]
    53	                    )
    54	                ELSE false
    55	            END
    56	        );
    57
    58	    IF EXISTS (
    59	        SELECT 1 FROM pg_constraint
    60	        WHERE conname = 'experiment_continuation_policy_inherit_config_check'
    61	    ) THEN
    62	        ALTER TABLE experiment
    63	            DROP CONSTRAINT experiment_continuation_policy_inherit_config_check;
    64	    END IF;
    65
    66	    ALTER TABLE experiment
    67	        ADD CONSTRAINT experiment_continuation_policy_inherit_config_check
    68	        CHECK (
    69	            NOT continuation_policy_inherit_to_child
    70	            OR
    71	            CASE COALESCE(
    72	                continuation_policy_progression_mode,
    73	                'fixed_increment'
    74	            )
    75	                WHEN 'fixed_increment' THEN
    76	                    continuation_policy_target_increment IS NOT NULL
    77	                    AND continuation_policy_target_sequence IS NULL
    78	                WHEN 'target_sequence' THEN
    79	                    continuation_policy_target_increment IS NULL
    80	                    AND continuation_policy_target_sequence IS NOT NULL
    81	                    AND continuation_target_sequence_is_valid(
    82	                        continuation_policy_target_sequence
    83	                    )
    84	                ELSE false
    85	            END
    86	        );
    87	END $$;
    88
    89	GRANT EXECUTE ON FUNCTION continuation_target_sequence_is_valid(integer[]) TO pqxx;
    90	GRANT SELECT, INSERT, UPDATE, DELETE ON experiment TO pqxx;
Database/migrations/014_checkpoint_stop_and_eval.sql:31:    worker_pid INTEGER,
Database/migrations/024_continuation_sequence_and_worker_attempt_hardening.sql:2:    ADD COLUMN IF NOT EXISTS worker_started_at timestamptz;
Database/migrations/011_experiment_live_progress.sql:3:    ADD COLUMN IF NOT EXISTS worker_pid integer,
Database/migrations/011_experiment_live_progress.sql:4:    ADD COLUMN IF NOT EXISTS current_operation text;
Database/migrations/011_experiment_live_progress.sql:6:CREATE INDEX IF NOT EXISTS experiment_worker_pid_idx
Database/migrations/011_experiment_live_progress.sql:7:    ON experiment (worker_pid);

exec
/bin/bash -lc "rg -n -i 'multiple schedul|concurrent.*schedul|scheduler.*concurr|claim|SKIP LOCKED|orphan|worker_started_at' Tests Sources | sed -n '1,600p' && nl -ba Tests/SchedulerChildStatusTests.cpp | sed -n '1,260p'" in /Volumes/Developer SSD/ExpertAdvisor
 succeeded in 0ms:
Sources/ExperimentRecommendationCampaignStatusRepository.cpp:62:       e.worker_started_at::text AS worker_started_at,
Sources/ExperimentRecommendationCampaignStatusRepository.cpp:163:        value.workerStartedAt = OptionalText(row, "worker_started_at");
Sources/ExperimentRecommendationCampaignStatusRepository.cpp:381:       ('experiment','current_operation'),('experiment','worker_started_at'),
Sources/ExperimentRecommendationConversionWorkflow.hpp:21:    schedulerClaimedOrRunning,
Sources/ExperimentRecommendationConversionActivationRepository.cpp:133:        "worker_started_at IS NOT NULL AS worker_started,"
Sources/ExperimentRecommendationConversionActivationRepository.cpp:182:        return "activation_requires_null_worker_started_at";
Sources/ExperimentRecommendationConversionActivationRepository.cpp:354:        "duplicate_nonce=0 AND worker_started_at IS NULL AND completed_at "
Sources/ExperimentRecommendationCampaignStatusService.cpp:180:        output << ",worker_started_at=";
Sources/WorkerLifecycleDiagnostics.hpp:34:inline bool ShouldReconcileSchedulerOrphan(bool processExists,
Tests/ExperimentRecommendationConversionActivationRepositoryTests.cpp:206:                "worker_started_at timestamptz,started_at timestamptz,"
Tests/ExperimentRecommendationCampaignActivationRepositoryTests.cpp:571: worker_started_at timestamptz,started_at timestamptz,completed_at timestamptz,
Tests/ExperimentRecommendationCampaignActivationRepositoryTests.cpp:1104:            "worker_started_at IS NOT NULL;") == 0);
Tests/ExperimentRecommendationCampaignExecutionTests.cpp:177:    auto orphanExperiment = Input();
Tests/ExperimentRecommendationCampaignExecutionTests.cpp:178:    orphanExperiment.members[0].experimentId = 999;
Tests/ExperimentRecommendationCampaignExecutionTests.cpp:179:    Throws("campaign_execution_orphan_experiment_invalid", [&] {
Tests/ExperimentRecommendationCampaignExecutionTests.cpp:180:        BuildRecommendationCampaignExecutionPlan(Request(), orphanExperiment);
Tests/ExperimentRecommendationCampaignHandoffRepositoryTests.cpp:392:            pqxx::work orphanActivation{owner};
Tests/ExperimentRecommendationCampaignHandoffRepositoryTests.cpp:393:            SetSearchPath(orphanActivation, schema);
Tests/ExperimentRecommendationCampaignHandoffRepositoryTests.cpp:394:            InsertActivation(orphanActivation, 52, 999, 2, 30, 102);
Tests/ExperimentRecommendationCampaignHandoffRepositoryTests.cpp:395:            orphanActivation.commit();
Tests/ExperimentRecommendationCampaignHandoffRepositoryTests.cpp:406:            pqxx::work removeOrphan{owner};
Tests/ExperimentRecommendationCampaignHandoffRepositoryTests.cpp:407:            SetSearchPath(removeOrphan, schema);
Tests/ExperimentRecommendationCampaignHandoffRepositoryTests.cpp:408:            removeOrphan.exec(
Tests/ExperimentRecommendationCampaignHandoffRepositoryTests.cpp:411:            removeOrphan.commit();
Sources/ExperimentRecommendationConversionWorkflow.cpp:46:        case RecommendationConversionWorkflowState::schedulerClaimedOrRunning:
Sources/ExperimentRecommendationConversionWorkflow.cpp:47:            return "scheduler_claimed_or_running";
Sources/ExperimentRecommendationConversionWorkflow.cpp:71:             RecommendationConversionWorkflowState::schedulerClaimedOrRunning,
Sources/ExperimentRecommendationConversionWorkflow.cpp:253:            RecommendationConversionWorkflowState::schedulerClaimedOrRunning;
Tests/ExperimentRecommendationConversionWorkflowRepositoryTests.cpp:272:            RecommendationConversionWorkflowState::schedulerClaimedOrRunning);
Tests/ExperimentRecommendationConversionWorkflowRepositoryTests.cpp:294:        const auto orphanActivation = FindRecommendationConversionWorkflow(
Tests/ExperimentRecommendationConversionWorkflowRepositoryTests.cpp:296:        assert(orphanActivation->derivation.state ==
Tests/ExperimentRecommendationConversionWorkflowRepositoryTests.cpp:298:        assert(orphanActivation->derivation.diagnosticCodes.front() ==
Tests/ExperimentRecommendationCampaignPlanningTests.cpp:87:        state == RecommendationConversionWorkflowState::schedulerClaimedOrRunning ||
Tests/ExperimentRecommendationCampaignPlanningTests.cpp:100:        state == RecommendationConversionWorkflowState::schedulerClaimedOrRunning ||
Tests/ExperimentRecommendationCampaignPlanningTests.cpp:259:        RecommendationConversionWorkflowState::schedulerClaimedOrRunning,
Sources/ExperimentScheduler.cpp:278:    bool recoverOrphansOnly = false;
Sources/ExperimentScheduler.cpp:1550:        else if (arg == "--recover-orphans-only")
Sources/ExperimentScheduler.cpp:1551:            options.recoverOrphansOnly = true;
Sources/ExperimentScheduler.cpp:3542:    if (options.recoverOrphansOnly && !options.scheduleExperiments)
Sources/ExperimentScheduler.cpp:3543:        throw std::invalid_argument("--recover-orphans-only requires --schedule-experiments");
Sources/ExperimentScheduler.cpp:3850:        std::cout << "SCHEDULER_ORPHAN_MODEL_UNUSABLE"
Sources/ExperimentScheduler.cpp:3938:        !ColumnExists(w, "experiment", "worker_started_at"))
Sources/ExperimentScheduler.cpp:3939:        missing.push_back("experiment.worker_started_at");
Sources/ExperimentScheduler.cpp:4849:        "extract(epoch from COALESCE(worker_started_at, updated_at))::double precision "
Sources/ExperimentScheduler.cpp:4865:int RecoverOrphanedRunningExperiments(pqxx::work& w,
Sources/ExperimentScheduler.cpp:5102:            << ", worker_started_at = NULL"
Sources/ExperimentScheduler.cpp:11544:        ", worker_started_at = now()" +
Sources/ExperimentScheduler.cpp:11736:void RequeueTrainOrphanFromCheckpoint(pqxx::work& w,
Sources/ExperimentScheduler.cpp:11747:    std::cout << "SCHEDULER_ORPHAN_RECOVERED"
Sources/ExperimentScheduler.cpp:11756:bool RecoverTrainOrphanFromModel(pqxx::work& w,
Sources/ExperimentScheduler.cpp:11770:        std::cout << "SCHEDULER_ORPHAN_MODEL_UNUSABLE"
Sources/ExperimentScheduler.cpp:11780:        std::cout << "SCHEDULER_ORPHAN_ADVANCED"
Sources/ExperimentScheduler.cpp:11790:    RequeueTrainOrphanFromCheckpoint(w, experiment, *meta);
Sources/ExperimentScheduler.cpp:11847:        std::cout << "SCHEDULER_ORPHAN_INFER_STILL_RUNNING"
Sources/ExperimentScheduler.cpp:11863:int RecoverOrphanedRunningExperiments(pqxx::work& w,
Sources/ExperimentScheduler.cpp:11933:        if (ShouldReconcileSchedulerOrphan(
Sources/ExperimentScheduler.cpp:11938:            std::cout << "SCHEDULER_ORPHAN_DETECTED"
Sources/ExperimentScheduler.cpp:11954:                    "SCHEDULER_ORPHAN_RECOVERED_INFERENCE_RESULT",
Sources/ExperimentScheduler.cpp:11963:                    "SCHEDULER_ORPHAN_RECOVERED_INFER_LOG",
Sources/ExperimentScheduler.cpp:11974:                    "SCHEDULER_ORPHAN_RECOVERED_INFER_LOG",
Sources/ExperimentScheduler.cpp:11980:                const std::string error = "orphaned_running_infer_no_process_no_result";
Sources/ExperimentScheduler.cpp:11982:                std::cout << "SCHEDULER_ORPHAN_MARKED_FAILED"
Sources/ExperimentScheduler.cpp:11997:                std::cout << "SCHEDULER_ORPHAN_RECOVERED_ANALYSIS"
Sources/ExperimentScheduler.cpp:12005:                const std::string error = "orphaned_running_analyze_no_process_no_result";
Sources/ExperimentScheduler.cpp:12007:                std::cout << "SCHEDULER_ORPHAN_MARKED_FAILED"
Sources/ExperimentScheduler.cpp:12017:            const std::string error = "orphaned_running_invalid_phase";
Sources/ExperimentScheduler.cpp:12019:            std::cout << "SCHEDULER_ORPHAN_MARKED_FAILED"
Sources/ExperimentScheduler.cpp:12033:            std::cout << "SCHEDULER_ORPHAN_RECOVERED_MODEL"
Sources/ExperimentScheduler.cpp:12038:            if (!RecoverTrainOrphanFromModel(w, experiment, *linkedModelId))
Sources/ExperimentScheduler.cpp:12040:                const std::string error = "orphaned_running_train_model_unusable";
Sources/ExperimentScheduler.cpp:12042:                std::cout << "SCHEDULER_ORPHAN_FAILED"
Sources/ExperimentScheduler.cpp:12061:            std::cout << "SCHEDULER_ORPHAN_RECOVERED_MODEL"
Sources/ExperimentScheduler.cpp:12065:            if (!RecoverTrainOrphanFromModel(w, experiment, *finalModelId))
Sources/ExperimentScheduler.cpp:12067:                const std::string error = "orphaned_running_train_model_unusable";
Sources/ExperimentScheduler.cpp:12069:                std::cout << "SCHEDULER_ORPHAN_FAILED"
Sources/ExperimentScheduler.cpp:12087:            std::cout << "SCHEDULER_ORPHAN_RECOVERED_CHECKPOINT"
Sources/ExperimentScheduler.cpp:12091:            if (!RecoverTrainOrphanFromModel(w, experiment, *checkpointModelId))
Sources/ExperimentScheduler.cpp:12093:                const std::string error = "orphaned_running_train_checkpoint_unusable";
Sources/ExperimentScheduler.cpp:12095:                std::cout << "SCHEDULER_ORPHAN_FAILED"
Sources/ExperimentScheduler.cpp:12103:        const std::string error = "orphaned_running_train_no_process_no_model";
Sources/ExperimentScheduler.cpp:12105:        std::cout << "SCHEDULER_ORPHAN_FAILED"
Sources/ExperimentScheduler.cpp:12695:                    "SCHEDULER_ORPHAN_RECOVERED_INFER_LOG",
Sources/ExperimentScheduler.cpp:12710:                    "SCHEDULER_ORPHAN_RECOVERED_INFER_LOG",
Sources/ExperimentScheduler.cpp:13344:            RecoverOrphanedRunningExperiments(w, logState, options.schedulerVerbose);
Sources/ExperimentScheduler.cpp:13376:            recoveryCount = RecoverOrphanedRunningExperiments(w, &logState, options.schedulerVerbose);
Sources/ExperimentScheduler.cpp:13401:              << ",recover_orphans_only=" << (options.recoverOrphansOnly ? "1" : "0")
Sources/ExperimentScheduler.cpp:13405:    if (options.recoverOrphansOnly)
Sources/ExperimentScheduler.cpp:13407:        std::cout << "SCHEDULER_ORPHAN_RECOVERY_DONE"
Sources/ExperimentScheduler.cpp:15802:        warnings.push_back("multiple scheduler processes detected: " + std::to_string(processes.schedulerPids.size()));
Sources/ExperimentScheduler.cpp:16456:        << "[--scheduler-verbose] [--dry-run] [--recover-orphans-only] "
Sources/ExperimentRecommendationCampaignPlanning.cpp:419:            case RecommendationConversionWorkflowState::schedulerClaimedOrRunning:
Tests/ExperimentRecommendationConversionWorkflowTests.cpp:123:        RecommendationConversionWorkflowState::schedulerClaimedOrRunning);
Tests/ExperimentRecommendationConversionWorkflowTests.cpp:142:        RecommendationConversionWorkflowState::schedulerClaimedOrRunning);
Tests/ExperimentRecommendationConversionWorkflowTests.cpp:186:             RecommendationConversionWorkflowState::schedulerClaimedOrRunning,
Sources/ExperimentRecommendationCampaignExecution.cpp:142:                    "campaign_execution_orphan_experiment_invalid");
Tests/SchedulerChildStatusTests.cpp:22:using EA::ExperimentScheduler::ShouldReconcileSchedulerOrphan;
Tests/SchedulerChildStatusTests.cpp:81:    assert(ShouldReconcileSchedulerOrphan(false, false, false));
Tests/SchedulerChildStatusTests.cpp:82:    assert(!ShouldReconcileSchedulerOrphan(true, false, false));
Tests/SchedulerChildStatusTests.cpp:83:    assert(!ShouldReconcileSchedulerOrphan(false, true, false));
Tests/SchedulerChildStatusTests.cpp:84:    assert(!ShouldReconcileSchedulerOrphan(false, false, true));
Tests/ExperimentRecommendationCampaignLaunchRepositoryTests.cpp:660: worker_started_at timestamptz,started_at timestamptz,completed_at timestamptz,
Tests/ExperimentRecommendationCampaignLaunchRepositoryTests.cpp:1394:            "worker_started_at IS NOT NULL;") == 0);
Tests/ExperimentRecommendationCampaignStatusRepositoryTests.cpp:573:            setup.exec("ALTER TABLE experiment ADD COLUMN worker_started_at timestamptz;");
Tests/ExperimentRecommendationCampaignStatusRepositoryTests.cpp:702:                "current_operation='train',worker_started_at=now(),started_at=now(),"
Tests/ExperimentRecommendationCampaignOutcomePolicyTests.cpp:828:    // claim of statistical sufficiency or campaign success.
     1	#include <cassert>
     2	#include <csignal>
     3	#include <fstream>
     4	#include <iostream>
     5	#include <sstream>
     6	#include <string>
     7	#include <unistd.h>
     8
     9	#include "../Sources/SchedulerChildStatus.hpp"
    10	#include "../Sources/WorkerLifecycleDiagnostics.hpp"
    11
    12	using EA::ExperimentScheduler::ChildStatusKind;
    13	using EA::ExperimentScheduler::ObserveChildStatusNonBlocking;
    14	using EA::ExperimentScheduler::WriteSchedulerChildExecFailureDiagnostic;
    15	using EA::ExperimentScheduler::ChildLifecycleDiagnosticState;
    16	using EA::ExperimentScheduler::LogWorkerStarted;
    17	using EA::ExperimentScheduler::IsCurrentWorkerAttemptEvidence;
    18	using EA::ExperimentScheduler::MarkChildExitDiagnosticEmitted;
    19	using EA::ExperimentScheduler::MarkChildTerminationObserved;
    20	using EA::ExperimentScheduler::SchedulerChildLaunchFailureDiagnostic;
    21	using EA::ExperimentScheduler::SchedulerLaunchFailureError;
    22	using EA::ExperimentScheduler::ShouldReconcileSchedulerOrphan;
    23
    24	pid_t ForkSleepingChild()
    25	{
    26	    const pid_t pid = ::fork();
    27	    assert(pid >= 0);
    28	    if (pid == 0)
    29	    {
    30	        for (;;)
    31	            ::pause();
    32	    }
    33	    return pid;
    34	}
    35
    36	void WaitUntilObserved(pid_t pid,
    37	                       ChildStatusKind expectedKind,
    38	                       int expectedValue)
    39	{
    40	    for (int attempt = 0; attempt < 1000; ++attempt)
    41	    {
    42	        const auto observed = ObserveChildStatusNonBlocking(pid);
    43	        if (observed.kind == ChildStatusKind::Running)
    44	        {
    45	            ::usleep(1000);
    46	            continue;
    47	        }
    48	        assert(observed.kind == expectedKind);
    49	        if (expectedKind == ChildStatusKind::Exited)
    50	            assert(observed.exitCode == expectedValue);
    51	        else
    52	        {
    53	            assert(observed.signalNumber == expectedValue);
    54	            assert(observed.exitCode == -expectedValue);
    55	        }
    56	        return;
    57	    }
    58	    assert(false && "child status was not observed");
    59	}
    60
    61	int main()
    62	{
    63	    assert(SchedulerLaunchFailureError(ENOENT) == "launch_failed_errno_2");
    64	    const std::string launchFailure = SchedulerChildLaunchFailureDiagnostic(
    65	        143,
    66	        "analyze",
    67	        ENOENT,
    68	        "/missing/worker --analyze-experiment=143");
    69	    assert(launchFailure.find(
    70	               "SCHEDULER_CHILD_LAUNCH_FAILED,experiment_id=143,phase=analyze,operation=analyze,errno=2") == 0);
    71	    assert(launchFailure.find("error=No such file or directory") != std::string::npos);
    72	    assert(launchFailure.find("command_line=/missing/worker --analyze-experiment=143") !=
    73	           std::string::npos);
    74
    75	    ChildLifecycleDiagnosticState diagnosticState;
    76	    assert(MarkChildTerminationObserved(diagnosticState));
    77	    assert(!MarkChildTerminationObserved(diagnosticState));
    78	    assert(MarkChildExitDiagnosticEmitted(diagnosticState));
    79	    assert(!MarkChildExitDiagnosticEmitted(diagnosticState));
    80
    81	    assert(ShouldReconcileSchedulerOrphan(false, false, false));
    82	    assert(!ShouldReconcileSchedulerOrphan(true, false, false));
    83	    assert(!ShouldReconcileSchedulerOrphan(false, true, false));
    84	    assert(!ShouldReconcileSchedulerOrphan(false, false, true));
    85	    assert(!IsCurrentWorkerAttemptEvidence(99.999, 100.0));
    86	    assert(IsCurrentWorkerAttemptEvidence(100.0, 100.0));
    87	    assert(IsCurrentWorkerAttemptEvidence(100.001, 100.0));
    88
    89	    char startupPath[] = "/tmp/analysis_worker_startup_XXXXXX";
    90	    const int startupFd = ::mkstemp(startupPath);
    91	    assert(startupFd >= 0);
    92	    assert(::close(startupFd) == 0);
    93	    {
    94	        std::ofstream startupLog(startupPath, std::ios::trunc);
    95	        assert(startupLog.is_open());
    96	        std::streambuf* original = std::cout.rdbuf(startupLog.rdbuf());
    97	        LogWorkerStarted(
    98	            "ANALYSIS_WORKER_STARTED",
    99	            "analyze",
   100	            143,
   101	            645);
   102	        std::cout.rdbuf(original);
   103	    }
   104	    std::ifstream startupLog(startupPath);
   105	    const std::string startupText{
   106	        std::istreambuf_iterator<char>(startupLog),
   107	        std::istreambuf_iterator<char>()};
   108	    assert(!startupText.empty());
   109	    assert(startupText.find(
   110	               "ANALYSIS_WORKER_STARTED,experiment_id=143,pid=") == 0);
   111	    assert(startupText.find(",operation=analyze,model_id=645") != std::string::npos);
   112	    assert(::unlink(startupPath) == 0);
   113
   114	    int startedExitPipe[2];
   115	    assert(::pipe(startedExitPipe) == 0);
   116	    pid_t startedExitPid = ::fork();
   117	    assert(startedExitPid >= 0);
   118	    if (startedExitPid == 0)
   119	    {
   120	        ::close(startedExitPipe[0]);
   121	        assert(::dup2(startedExitPipe[1], STDOUT_FILENO) >= 0);
   122	        ::close(startedExitPipe[1]);
   123	        LogWorkerStarted("TRAINING_WORKER_STARTED", "train", 200);
   124	        _exit(0);
   125	    }
   126	    assert(::close(startedExitPipe[1]) == 0);
   127	    WaitUntilObserved(startedExitPid, ChildStatusKind::Exited, 0);
   128	    char startedExitText[256] = {};
   129	    const ssize_t startedExitSize =
   130	        ::read(startedExitPipe[0], startedExitText, sizeof(startedExitText));
   131	    assert(startedExitSize > 0);
   132	    assert(::close(startedExitPipe[0]) == 0);
   133	    assert(std::string(startedExitText, static_cast<size_t>(startedExitSize)).find(
   134	               "TRAINING_WORKER_STARTED,experiment_id=200") == 0);
   135
   136	    int startedSignalPipe[2];
   137	    assert(::pipe(startedSignalPipe) == 0);
   138	    pid_t startedSignalPid = ::fork();
   139	    assert(startedSignalPid >= 0);
   140	    if (startedSignalPid == 0)
   141	    {
   142	        ::close(startedSignalPipe[0]);
   143	        assert(::dup2(startedSignalPipe[1], STDOUT_FILENO) >= 0);
   144	        ::close(startedSignalPipe[1]);
   145	        LogWorkerStarted("INFERENCE_WORKER_STARTED", "infer", 201, 700);
   146	        for (;;)
   147	            ::pause();
   148	    }
   149	    assert(::close(startedSignalPipe[1]) == 0);
   150	    char startedSignalText[256] = {};
   151	    const ssize_t startedSignalSize =
   152	        ::read(startedSignalPipe[0], startedSignalText, sizeof(startedSignalText));
   153	    assert(startedSignalSize > 0);
   154	    assert(std::string(startedSignalText, static_cast<size_t>(startedSignalSize)).find(
   155	               "INFERENCE_WORKER_STARTED,experiment_id=201") == 0);
   156	    assert(::kill(startedSignalPid, SIGTERM) == 0);
   157	    WaitUntilObserved(startedSignalPid, ChildStatusKind::Signaled, SIGTERM);
   158	    assert(::close(startedSignalPipe[0]) == 0);
   159
   160	    int diagnosticPipe[2];
   161	    assert(::pipe(diagnosticPipe) == 0);
   162	    WriteSchedulerChildExecFailureDiagnostic(
   163	        diagnosticPipe[1], 143, "analyze", 7, ENOENT);
   164	    assert(::close(diagnosticPipe[1]) == 0);
   165	    char diagnostic[256] = {};
   166	    const ssize_t diagnosticSize = ::read(
   167	        diagnosticPipe[0], diagnostic, sizeof(diagnostic));
   168	    assert(diagnosticSize > 0);
   169	    assert(::close(diagnosticPipe[0]) == 0);
   170	    assert(std::string(diagnostic, static_cast<size_t>(diagnosticSize)) ==
   171	           "SCHEDULER_CHILD_EXEC_FAILED,experiment_id=143,phase=analyze,errno=2,error=no_such_file_or_directory\n");
   172
   173	    int execPipe[2];
   174	    assert(::pipe(execPipe) == 0);
   175	    pid_t execPid = ::fork();
   176	    assert(execPid >= 0);
   177	    if (execPid == 0)
   178	    {
   179	        ::close(execPipe[0]);
   180	        assert(::dup2(execPipe[1], STDERR_FILENO) >= 0);
   181	        ::close(execPipe[1]);
   182	        char executable[] = "/definitely/not/a/scheduler/worker";
   183	        char* const argv[] = {executable, nullptr};
   184	        ::execv(executable, argv);
   185	        const int execError = errno;
   186	        WriteSchedulerChildExecFailureDiagnostic(
   187	            STDERR_FILENO, 143, "analyze", 7, execError);
   188	        _exit(127);
   189	    }
   190	    assert(::close(execPipe[1]) == 0);
   191	    WaitUntilObserved(execPid, ChildStatusKind::Exited, 127);
   192	    char execDiagnostic[256] = {};
   193	    const ssize_t execDiagnosticSize = ::read(
   194	        execPipe[0], execDiagnostic, sizeof(execDiagnostic));
   195	    assert(execDiagnosticSize > 0);
   196	    assert(::close(execPipe[0]) == 0);
   197	    assert(std::string(execDiagnostic, static_cast<size_t>(execDiagnosticSize)).find(
   198	               "SCHEDULER_CHILD_EXEC_FAILED,experiment_id=143,phase=analyze,errno=2,error=no_such_file_or_directory") !=
   199	           std::string::npos);
   200
   201	    pid_t pid = ::fork();
   202	    assert(pid >= 0);
   203	    if (pid == 0)
   204	        _exit(0);
   205	    WaitUntilObserved(pid, ChildStatusKind::Exited, 0);
   206
   207	    pid = ::fork();
   208	    assert(pid >= 0);
   209	    if (pid == 0)
   210	        _exit(1);
   211	    WaitUntilObserved(pid, ChildStatusKind::Exited, 1);
   212
   213	    pid = ForkSleepingChild();
   214	    assert(ObserveChildStatusNonBlocking(pid).kind == ChildStatusKind::Running);
   215	    assert(::kill(pid, SIGTERM) == 0);
   216	    WaitUntilObserved(pid, ChildStatusKind::Signaled, SIGTERM);
   217
   218	    pid = ForkSleepingChild();
   219	    assert(::kill(pid, SIGKILL) == 0);
   220	    WaitUntilObserved(pid, ChildStatusKind::Signaled, SIGKILL);
   221
   222	    const auto noLongerOwned = ObserveChildStatusNonBlocking(pid);
   223	    assert(noLongerOwned.kind == ChildStatusKind::WaitError);
   224	    assert(noLongerOwned.errorNumber == ECHILD);
   225
   226	    std::cout << "SchedulerChildStatusTests passed\n";
   227	    return 0;
   228	}

The target artifact is incomplete, so the evaluation now separates (a) repository-verifiable direction and constraints from (b) design claims that cannot be reviewed because no recommendation was emitted.
  ✓ Locate governing instructions, the recommendation, and authoritative architecture sources
  ✓ Trace implemented Phase 4–6 authority chains, scheduler, lifecycle, persistence, and tests
  → Test each recommendation claim against repository evidence and identify contradictions or gaps
  • Produce the disposition, required preconditions, readiness assessment, and requested repository status
codex
The direction is now clear enough to disposition: a separate operational layer is compatible with the repository and is plausibly the next major concern, but the submitted artifact cannot be made authoritative. It contains no design, and the repository still lacks an accepted authority mapping from Phase 6D ratification to any operational request. I’m completing the fail-closed conditions and final read-only audit now.
exec
/bin/bash -lc "git status --short && git diff --stat && git diff --check && rg --files Sources Tests Database/migrations docs/architecture/adr | rg -i 'campaign.?operations|operation.?campaign' || true && find Database/migrations -maxdepth 1 -type f -name '*.sql' -print | sort | tail -n 8" in /Volumes/Developer SSD/ExpertAdvisor
 succeeded in 134ms:
Database/migrations/037_experiment_recommendation_conversion_review.sql
Database/migrations/038_experiment_recommendation_conversion_execution.sql
Database/migrations/039_experiment_recommendation_conversion_activation.sql
Database/migrations/040_experiment_recommendation_campaign_approval.sql
Database/migrations/041_experiment_recommendation_campaign_materialization.sql
Database/migrations/042_experiment_recommendation_campaign_follow_up_proposal.sql
Database/migrations/043_experiment_recommendation_campaign_follow_up_proposal_review.sql
Database/migrations/044_experiment_recommendation_campaign_follow_up_proposal_ratification.sql

Repository evidence and the artifact-completeness audit support a fail-closed revise-and-re-review disposition. No repository or database changes were made.
  ✓ Locate governing instructions, the recommendation, and authoritative architecture sources
  ✓ Trace implemented Phase 4–6 authority chains, scheduler, lifecycle, persistence, and tests
  ✓ Test each recommendation claim against repository evidence and identify contradictions or gaps
  ✓ Produce the disposition, required preconditions, readiness assessment, and requested repository status
codex
# Critical Engineering Evaluation

## 1. Executive summary

Recommended disposition: **revise and re-review**.

`CampaignOperations_Architecture_Output.txt` cannot be accepted as an authoritative architectural recommendation because it contains no completed recommendation. It contains the original prompt, an initial work plan, repository file listings, and an abnormally terminated command; no subsystem decomposition, state model, data model, transactions, scheduler contract, budget rules, recovery model, privileges, risks, or implementation sequence was produced. See the [start of the attempted review](</Volumes/Developer SSD/ExpertAdvisor/CampaignOperations_Architecture_Output.txt:117>) and [abnormal termination](</Volumes/Developer SSD/ExpertAdvisor/CampaignOperations_Architecture_Output.txt:639>).

The repository does support two parts of the intended direction:

- A distinct operational coordination layer is compatible with the reserved Research Automation domain.
- Recommendation or research-policy authority must not move into the scheduler.

However, the following are not established:

- Phase 6 is not formally complete as accepted architecture: Phase 6D is implemented, but ADR-0009 remains `Proposed`.
- The repository does not prove that “no meaningful Phase 6E remains.” It proves only that no Phase 6E capability is currently authorized.
- Phase 6D ratification grants no operational authority and does not specify what work should occur next.
- No accepted mapping exists between Phase 6D ratification, existing Phase 4D materialization, Phase 5 launch authority, and a future Campaign Operations subsystem.

Implementation readiness is therefore **not ready**.

## 2. Repository-grounded architectural findings

| Question | Finding |
|---|---|
| Is the submitted recommendation complete? | No. It contains no architecture deliverable. |
| Is Phase 6 implemented through 6D? | Yes: domain, repository, service, migration, documentation, and tests exist at `d78771f`. |
| Is Phase 6D accepted architecture? | No. ADR-0009 and the affected volumes still label it `Proposed`. |
| Is a Phase 6E implementation currently authorized? | No. Phase 6D explicitly grants no Phase 6E or operational capability. |
| Does that prove Phase 6E has no meaningful subject? | No. Phase 6D ratifies advancement into a still-undefined “next separately controlled phase.” |
| Is Campaign Operations plausible? | Yes, as a bounded slice of the reserved Research Automation domain. |
| Is it established as the next authoritative increment? | No. Its input authority, relationship to Phase 6, and operational subject remain undecided. |
| May it own recommendation policy? | No. |
| May the scheduler consume recommendation or ratification rows as policy? | No. |
| Can lifecycle, budgeting, replay, recovery, and privileges be approved? | No corresponding proposal content exists to review. |

The ADR rules say `Proposed` is “not implementation authority” and implementation follows acceptance. ADR-0009 is still listed as proposed in the [ADR index](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/adr/README.md:18>) and [ADR-0009 itself](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:1>). Volume VIII similarly says “proposed Phase 6D recorded,” despite describing the implementation as present ([Volume VIII](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/Volume_VIII_Recommendation_Engine.md:1>)).

Phase 6D’s actual implementation is narrow and coherent: it validates an approved review, enforces a different ratifier, persists immutable ratification, and commits one transaction ([service](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationCampaignFollowUpProposalRatificationService.cpp:49>)). Its migration explicitly says the event grants no Phase 6E or operational authority and applies narrow insert privileges ([migration 044](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/044_experiment_recommendation_campaign_follow_up_proposal_ratification.sql:1>)).

## 3. Strengths

These are strengths of the repository and stated direction—not of the missing recommendation:

- PostgreSQL is already the accepted durable authority, with canonical-first identities, transactions, constraints, and immutable provenance.
- Phase 4D materialization provides an exact, ordered, immutable membership authority ([Phase 4D materialization](</Volumes/Developer SSD/ExpertAdvisor/docs/Phase4DExperimentRecommendationCampaignMaterialization.rst:30>)).
- Phase 5 launch reuses existing conversion and activation authorities with an explicit lock hierarchy and atomic all-member transaction ([Phase 5 launch](</Volumes/Developer SSD/ExpertAdvisor/docs/Phase5ExperimentRecommendationCampaignLaunch.rst:39>)).
- Phase 5 status already demonstrates how to project campaign membership onto lifecycle evidence without duplicating lifecycle authority ([Phase 5 status](</Volumes/Developer SSD/ExpertAdvisor/docs/Phase5ExperimentRecommendationCampaignStatus.rst:7>)).
- ADR-0004 clearly reserves polling, claims, capacity, launch, recovery, and shutdown for the scheduler while excluding recommendation policy ([ADR-0004](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md:21>)).
- Volume X already anticipates campaigns, budgets, reservations, requests, stopping, reconciliation, and operator controls while prohibiting implicit experiment creation and direct worker control ([Volume X](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/Volume_X_Research_Automation.md:13>)).
- Phase 6 provides strong patterns for exact replay, collision-safe identity, append-only evidence, separation of duties, and least privilege.

## 4. Blocking defects

### B1. No architecture was delivered

This is dispositive. There is nothing against which to evaluate lifecycle transitions, ownership, budgeting, requests, concurrency, recovery, privileges, or implementation sequence.

### B2. Phase 6 formal completion is contradicted by authoritative documentation

Phase 6D implementation exists, but its ADR is not accepted. Under the repository’s governance rules, implementation does not silently convert a proposed decision into accepted architecture.

Phase 6 governance can be described as mechanically implemented through 6D, but not formally complete and accepted.

### B3. The post-ratification authority is undefined

Phase 6D is governance evidence only. It does not:

- authorize follow-up;
- identify a new experiment specification;
- identify a new campaign membership;
- authorize conversion or activation;
- allocate a budget;
- create an execution request;
- grant scheduler eligibility.

A Campaign Operations design cannot treat ratification as operational authorization without contradicting Phase 6D.

### B4. “No meaningful Phase 6E remains” is unsupported

The correct repository-grounded conclusion is:

> No Phase 6E implementation is currently authorized because no accepted Phase 6E contract exists.

That is different from concluding that nothing meaningful remains. Phase 6D explicitly advances toward a separate future phase, and its documentation says any later capability requires its own accepted contract and safety rules ([Phase 6D documentation](</Volumes/Developer SSD/ExpertAdvisor/docs/Phase6DRecommendationCampaignFollowUpProposalRatification.rst:121>)).

### B5. Campaign identity and ownership are unresolved

“Campaign” currently refers primarily to the immutable Phase 4D recommendation-campaign materialization. Phase 5 deliberately adds no campaign lifecycle or launch table. A new subsystem must decide whether its operational aggregate:

- is one-to-one with a Phase 4D materialization;
- represents one operational attempt over a materialization;
- permits multiple operational attempts;
- consumes Phase 6D ratification;
- or represents a different future research campaign.

Without that decision, ownership and deterministic identity are ambiguous.

### B6. All fail-closed operational contracts are missing

No reviewable contract exists for budgets, reservations, requests, cancellation, completion, recovery, replay, lock ordering, privileges, or audit. These are architectural decisions, not safe implementation details.

## 5. Required revisions

A replacement architecture document must:

1. Close the Phase 6D status inconsistency by either accepting ADR-0009 and aligning Volumes VIII/XII, or explicitly treating Phase 6D as non-authoritative pending review.
2. State the exact problem Campaign Operations solves independently of the desired solution.
3. Define its exact authoritative input. Phase 6D ratification alone is insufficient operational authority.
4. Define whether Campaign Operations is a bounded implementation of Volume X or a separately governed domain.
5. Distinguish recommendation campaign identity, materialization identity, operational-campaign identity, and operational-attempt identity.
6. Provide the complete lifecycle transition table, including transition owner, preconditions, persisted effects, terminality, retry result, and concurrent loser result.
7. Define budget units, budget grant authority, reservation identity, consumption, release, expiry, overrun, and uncertain-state behavior.
8. Define operational-request identity, uniqueness, payload equivalence, durable downstream binding, and exact replay after experiments have progressed.
9. Define transaction boundaries and a global lock order compatible with the existing Phase 5 proposal → execution → experiment lock hierarchy.
10. Define scheduler interaction without transferring recommendation or research policy.
11. Define bounded restart reconciliation and the authoritative owner of every fact it reconciles.
12. Define cancellation and completion race semantics.
13. Provide a role/privilege matrix, immutable audit model, migration/rollback strategy, and test plan.
14. Provide a genuinely incremental implementation sequence, with later operational phases gated on accepted ADRs.

## 6. Non-blocking risks and deferred decisions

Once the architecture above is resolved, the following can safely be deferred:

- exact table and index names;
- repository method names;
- CLI design;
- presentation formatting;
- archival and retention policy;
- distributed schedulers or remote workers;
- advanced priority/fairness policy;
- adaptive or self-modifying research policy;
- generalized resource-cost estimation beyond the accepted initial budget unit.

An important existing constraint is that the current scheduler implementation does not expose a durable Campaign Operations request interface. It reads ordinary `pending` experiment rows ([pending query](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:4822>)). Its current transition to `running` is also an unconditional update performed before launch ([running transition](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:11532>)), rather than the explicit compare-and-set attempt contract described by ADR-0004. That is a pre-existing implementation/documentation gap, not authority for Campaign Operations to redesign the scheduler. Any scheduler-connected increment must either rely only on the supported single-scheduler behavior or separately harden the claim protocol under ADR-0004.

## 7. Assessment of authority boundaries

The intended separation is correct:

- Recommendation governance owns recommendations, evidence, review, approval, follow-up proposals, and ratification.
- Campaign Operations may own explicitly authorized operational intent, control state, budgets, reservations, request issuance, and operational reconciliation.
- Experiment lifecycle owns experiment creation, identity, lifecycle transitions, and terminal experiment state.
- Scheduler owns eligibility polling, capacity, claims, process launch, completion processing, and orphan recovery.
- Workers own bounded computation and report facts.
- PostgreSQL owns committed durable truth.

The following belong nowhere:

- score- or ratification-driven automatic experiment creation;
- scheduler interpretation of recommendation policy;
- duplicated campaign membership reconstructed from current queries;
- Campaign Operations directly launching workers;
- Campaign Operations rewriting experiment outcomes;
- treating budget availability as execution authorization;
- treating process presence as durable completion evidence.

## 8. Assessment of scheduler interaction

The boundary required by ADR-0004 is sound, but no submitted interaction contract exists.

A valid design must preserve these rules:

- The scheduler must not poll Phase 6 governance or Campaign Operations policy tables.
- Campaign Operations must not assign worker capacity or select runnable phases.
- Any work handed to the existing scheduler must first become valid ordinary experiment lifecycle state through an accepted workflow.
- Campaign Operations may observe durable experiment state but must not create a second scheduler state machine.
- Worker launch must remain outside database transactions.
- A new scheduler work class would require its own accepted ADR covering eligibility, claims, capacity, fairness, retry, recovery, operator control, observability, and shutdown.

## 9. Lifecycle, transactions, concurrency, replay, and recovery

These dimensions are not approvable because the recommendation contains no design.

Before acceptance, the replacement must resolve at least:

- Control state versus derived member state. An operational campaign’s administrative state cannot overwrite member experiment truth.
- Atomic handoff. It must choose whether reservation, request, and downstream lifecycle creation commit together or use a durable compensating protocol.
- Replay after progression. Phase 5 launch reports an ordinary exact retry only while experiments remain `pending/train`; progressed experiments are not rewound ([supported matrix](</Volumes/Developer SSD/ExpertAdvisor/docs/Phase5ExperimentRecommendationCampaignLaunch.rst:64>)). Campaign Operations therefore needs its own durable request-to-result binding.
- Lock ordering. Budget, campaign, reservation, request, proposal, execution, activation, and experiment locks need one global order.
- Recovery bounds. Reconciliation must use bounded batches and must not repeatedly create requests or release reservations merely because a process is absent.
- Cancellation races. The contract must cover cancellation before request creation, after request creation, after lifecycle activation, after scheduler claim, during execution, and concurrent with completion.
- Completion. Operational completion must be distinct from scientific success and must define how terminal failures, cancellations, partial membership, unresolved reservations, and inconsistent evidence affect it.
- Uncertain commits. Retrying after a lost response must look up authoritative request identity rather than repeat side effects blindly.

PostgreSQL should remain the single durable store, but “single source of truth” does not mean one table owns every fact. Each fact must have exactly one domain owner, with reconciliation producing a projection or append-only observation rather than competing truth.

## 10. Budgeting, reservations, and operational-request ownership

No budget or reservation proposal was delivered.

A reviewable design must specify:

- the budgeted resource and unit;
- who grants and amends a budget;
- whether budget is immutable, versioned, or append-only;
- reservation conflict scope;
- when a reservation becomes committed consumption;
- whether failure or cancellation releases anything;
- how uncertain downstream state is handled;
- how overrun is prevented;
- whether unused reservations expire and who may reclaim them;
- the invariant relating grants, reservations, committed consumption, and remaining capacity;
- the exact one-to-one or one-to-many relationship between reservations and requests.

Budget authority must remain separate from scheduler capacity. A campaign may have budget but no scheduler capacity; scheduler capacity must not imply campaign authority.

An operational request must be Campaign Operations-owned durable intent, not a scheduler claim and not an experiment outcome. It must identify the exact upstream authorization and exact intended downstream action. At most one accepted request may consume a reservation, and exact retries must return the same request and downstream binding.

## 11. Privileges, auditability, and testability

Phase 6 provides a good least-privilege precedent: `SELECT`, payload-column `INSERT`, sequence `USAGE`, no update/delete/truncate, and revoked trigger-function execution ([migration 044 privileges](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/044_experiment_recommendation_campaign_follow_up_proposal_ratification.sql:146>)).

No Campaign Operations privilege model was supplied. Required architecture includes:

- separate authority for operator decisions, dispatch, reconciliation, and scheduler reads;
- no scheduler access to recommendation-governance policy unless strictly necessary;
- column- or function-scoped mutations for mutable control state;
- append-only decision, reservation, request, cancellation, reconciliation, and terminal audit evidence;
- actor, reason, expected identity, prior/resulting state, causal request, downstream IDs, and timestamp in audit records;
- protection against using audit rows as a second mutable current-state table.

Required tests include pure state/identity tests, migration and ACL tests, independent-connection concurrency tests, uncertain-retry tests, budget races, request duplication, reservation leakage, cancellation/completion races, bounded restart recovery, scheduler-isolation regressions, and unchanged Phase 4–6 evidence.

## 12. Documentation and terminology findings

- The submitted `.txt` is an ignored, untracked transcript, not part of the authoritative architecture hierarchy.
- Even a complete version would remain a proposal until represented by accepted ADRs and aligned architecture volumes.
- ADR-0009, Volume VIII, and Volume XII must agree on whether Phase 6D is proposed or accepted.
- “Campaign Operations” overlaps both existing “recommendation campaign” terminology and Volume X “Research Automation.”
- The replacement must define, without synonyms:

  - recommendation campaign;
  - campaign materialization;
  - operational campaign;
  - operational attempt;
  - authorization;
  - ratification;
  - execution request;
  - activation;
  - scheduler admission;
  - reservation;
  - cancellation;
  - operational completion;
  - scientific outcome.

## 13. Recommended disposition

**Revise and re-review.**

“Accept with required revisions” is too permissive because the missing material concerns the core authority, lifecycle, concurrency, and safety model. “Reject” would overstate the result: the separate-subsystem direction is plausible and largely consistent with Volume X, but no reviewable architecture was delivered.

## 14. Exact conditions before implementation begins

Implementation must not begin until all of these are satisfied:

1. A complete Campaign Operations architecture document replaces the failed transcript.
2. ADR-0009’s status and affected volumes are made consistent.
3. An accepted ADR establishes whether Campaign Operations—not a Phase 6E recommendation-planning bridge—is the next bounded concern.
4. The ADR defines the exact upstream operational authorization; ratification alone is explicitly insufficient.
5. Domain ownership and identities are complete and collision-safe.
6. The full lifecycle and terminal-state matrix is accepted.
7. Budget, reservation, and request invariants are accepted.
8. Transaction boundaries, uncertain outcomes, and the global lock order are accepted.
9. Scheduler integration is either limited to existing lifecycle state or covered by a separate scheduler-work-class ADR.
10. Restart reconciliation, cancellation, completion, and replay semantics are accepted.
11. Privilege, audit, migration, rollback, compatibility, and test plans are complete.
12. The revised architecture receives another independent fail-closed review.
13. Only then may the smallest accepted implementation increment begin.

## 15. Overall readiness assessment

- Conceptual direction: **plausible but not selected authoritatively**.
- Phase 4–6 implementation foundation: **substantial and reusable**.
- Phase 6 formal architectural closure: **incomplete because ADR-0009 remains proposed**.
- Campaign Operations architecture: **not delivered**.
- Implementation readiness: **not ready**.
- Migration readiness: **not ready**.
- Scheduler integration readiness: **not demonstrated**.

Review audit:

- Files changed: none.
- Behavioral change: none.
- Tests/builds run: none; this was a read-only architecture review.
- Database or production processes accessed: none.
- Current branch/HEAD: `campaign-operations` at `d78771f`.
- `git status --short`: no output.
- `git diff --stat`: no output.
- `git diff --check`: passed with no output.
tokens used
216,225
# Critical Engineering Evaluation

## 1. Executive summary

Recommended disposition: **revise and re-review**.

`CampaignOperations_Architecture_Output.txt` cannot be accepted as an authoritative architectural recommendation because it contains no completed recommendation. It contains the original prompt, an initial work plan, repository file listings, and an abnormally terminated command; no subsystem decomposition, state model, data model, transactions, scheduler contract, budget rules, recovery model, privileges, risks, or implementation sequence was produced. See the [start of the attempted review](</Volumes/Developer SSD/ExpertAdvisor/CampaignOperations_Architecture_Output.txt:117>) and [abnormal termination](</Volumes/Developer SSD/ExpertAdvisor/CampaignOperations_Architecture_Output.txt:639>).

The repository does support two parts of the intended direction:

- A distinct operational coordination layer is compatible with the reserved Research Automation domain.
- Recommendation or research-policy authority must not move into the scheduler.

However, the following are not established:

- Phase 6 is not formally complete as accepted architecture: Phase 6D is implemented, but ADR-0009 remains `Proposed`.
- The repository does not prove that “no meaningful Phase 6E remains.” It proves only that no Phase 6E capability is currently authorized.
- Phase 6D ratification grants no operational authority and does not specify what work should occur next.
- No accepted mapping exists between Phase 6D ratification, existing Phase 4D materialization, Phase 5 launch authority, and a future Campaign Operations subsystem.

Implementation readiness is therefore **not ready**.

## 2. Repository-grounded architectural findings

| Question | Finding |
|---|---|
| Is the submitted recommendation complete? | No. It contains no architecture deliverable. |
| Is Phase 6 implemented through 6D? | Yes: domain, repository, service, migration, documentation, and tests exist at `d78771f`. |
| Is Phase 6D accepted architecture? | No. ADR-0009 and the affected volumes still label it `Proposed`. |
| Is a Phase 6E implementation currently authorized? | No. Phase 6D explicitly grants no Phase 6E or operational capability. |
| Does that prove Phase 6E has no meaningful subject? | No. Phase 6D ratifies advancement into a still-undefined “next separately controlled phase.” |
| Is Campaign Operations plausible? | Yes, as a bounded slice of the reserved Research Automation domain. |
| Is it established as the next authoritative increment? | No. Its input authority, relationship to Phase 6, and operational subject remain undecided. |
| May it own recommendation policy? | No. |
| May the scheduler consume recommendation or ratification rows as policy? | No. |
| Can lifecycle, budgeting, replay, recovery, and privileges be approved? | No corresponding proposal content exists to review. |

The ADR rules say `Proposed` is “not implementation authority” and implementation follows acceptance. ADR-0009 is still listed as proposed in the [ADR index](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/adr/README.md:18>) and [ADR-0009 itself](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:1>). Volume VIII similarly says “proposed Phase 6D recorded,” despite describing the implementation as present ([Volume VIII](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/Volume_VIII_Recommendation_Engine.md:1>)).

Phase 6D’s actual implementation is narrow and coherent: it validates an approved review, enforces a different ratifier, persists immutable ratification, and commits one transaction ([service](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationCampaignFollowUpProposalRatificationService.cpp:49>)). Its migration explicitly says the event grants no Phase 6E or operational authority and applies narrow insert privileges ([migration 044](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/044_experiment_recommendation_campaign_follow_up_proposal_ratification.sql:1>)).

## 3. Strengths

These are strengths of the repository and stated direction—not of the missing recommendation:

- PostgreSQL is already the accepted durable authority, with canonical-first identities, transactions, constraints, and immutable provenance.
- Phase 4D materialization provides an exact, ordered, immutable membership authority ([Phase 4D materialization](</Volumes/Developer SSD/ExpertAdvisor/docs/Phase4DExperimentRecommendationCampaignMaterialization.rst:30>)).
- Phase 5 launch reuses existing conversion and activation authorities with an explicit lock hierarchy and atomic all-member transaction ([Phase 5 launch](</Volumes/Developer SSD/ExpertAdvisor/docs/Phase5ExperimentRecommendationCampaignLaunch.rst:39>)).
- Phase 5 status already demonstrates how to project campaign membership onto lifecycle evidence without duplicating lifecycle authority ([Phase 5 status](</Volumes/Developer SSD/ExpertAdvisor/docs/Phase5ExperimentRecommendationCampaignStatus.rst:7>)).
- ADR-0004 clearly reserves polling, claims, capacity, launch, recovery, and shutdown for the scheduler while excluding recommendation policy ([ADR-0004](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md:21>)).
- Volume X already anticipates campaigns, budgets, reservations, requests, stopping, reconciliation, and operator controls while prohibiting implicit experiment creation and direct worker control ([Volume X](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/Volume_X_Research_Automation.md:13>)).
- Phase 6 provides strong patterns for exact replay, collision-safe identity, append-only evidence, separation of duties, and least privilege.

## 4. Blocking defects

### B1. No architecture was delivered

This is dispositive. There is nothing against which to evaluate lifecycle transitions, ownership, budgeting, requests, concurrency, recovery, privileges, or implementation sequence.

### B2. Phase 6 formal completion is contradicted by authoritative documentation

Phase 6D implementation exists, but its ADR is not accepted. Under the repository’s governance rules, implementation does not silently convert a proposed decision into accepted architecture.

Phase 6 governance can be described as mechanically implemented through 6D, but not formally complete and accepted.

### B3. The post-ratification authority is undefined

Phase 6D is governance evidence only. It does not:

- authorize follow-up;
- identify a new experiment specification;
- identify a new campaign membership;
- authorize conversion or activation;
- allocate a budget;
- create an execution request;
- grant scheduler eligibility.

A Campaign Operations design cannot treat ratification as operational authorization without contradicting Phase 6D.

### B4. “No meaningful Phase 6E remains” is unsupported

The correct repository-grounded conclusion is:

> No Phase 6E implementation is currently authorized because no accepted Phase 6E contract exists.

That is different from concluding that nothing meaningful remains. Phase 6D explicitly advances toward a separate future phase, and its documentation says any later capability requires its own accepted contract and safety rules ([Phase 6D documentation](</Volumes/Developer SSD/ExpertAdvisor/docs/Phase6DRecommendationCampaignFollowUpProposalRatification.rst:121>)).

### B5. Campaign identity and ownership are unresolved

“Campaign” currently refers primarily to the immutable Phase 4D recommendation-campaign materialization. Phase 5 deliberately adds no campaign lifecycle or launch table. A new subsystem must decide whether its operational aggregate:

- is one-to-one with a Phase 4D materialization;
- represents one operational attempt over a materialization;
- permits multiple operational attempts;
- consumes Phase 6D ratification;
- or represents a different future research campaign.

Without that decision, ownership and deterministic identity are ambiguous.

### B6. All fail-closed operational contracts are missing

No reviewable contract exists for budgets, reservations, requests, cancellation, completion, recovery, replay, lock ordering, privileges, or audit. These are architectural decisions, not safe implementation details.

## 5. Required revisions

A replacement architecture document must:

1. Close the Phase 6D status inconsistency by either accepting ADR-0009 and aligning Volumes VIII/XII, or explicitly treating Phase 6D as non-authoritative pending review.
2. State the exact problem Campaign Operations solves independently of the desired solution.
3. Define its exact authoritative input. Phase 6D ratification alone is insufficient operational authority.
4. Define whether Campaign Operations is a bounded implementation of Volume X or a separately governed domain.
5. Distinguish recommendation campaign identity, materialization identity, operational-campaign identity, and operational-attempt identity.
6. Provide the complete lifecycle transition table, including transition owner, preconditions, persisted effects, terminality, retry result, and concurrent loser result.
7. Define budget units, budget grant authority, reservation identity, consumption, release, expiry, overrun, and uncertain-state behavior.
8. Define operational-request identity, uniqueness, payload equivalence, durable downstream binding, and exact replay after experiments have progressed.
9. Define transaction boundaries and a global lock order compatible with the existing Phase 5 proposal → execution → experiment lock hierarchy.
10. Define scheduler interaction without transferring recommendation or research policy.
11. Define bounded restart reconciliation and the authoritative owner of every fact it reconciles.
12. Define cancellation and completion race semantics.
13. Provide a role/privilege matrix, immutable audit model, migration/rollback strategy, and test plan.
14. Provide a genuinely incremental implementation sequence, with later operational phases gated on accepted ADRs.

## 6. Non-blocking risks and deferred decisions

Once the architecture above is resolved, the following can safely be deferred:

- exact table and index names;
- repository method names;
- CLI design;
- presentation formatting;
- archival and retention policy;
- distributed schedulers or remote workers;
- advanced priority/fairness policy;
- adaptive or self-modifying research policy;
- generalized resource-cost estimation beyond the accepted initial budget unit.

An important existing constraint is that the current scheduler implementation does not expose a durable Campaign Operations request interface. It reads ordinary `pending` experiment rows ([pending query](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:4822>)). Its current transition to `running` is also an unconditional update performed before launch ([running transition](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:11532>)), rather than the explicit compare-and-set attempt contract described by ADR-0004. That is a pre-existing implementation/documentation gap, not authority for Campaign Operations to redesign the scheduler. Any scheduler-connected increment must either rely only on the supported single-scheduler behavior or separately harden the claim protocol under ADR-0004.

## 7. Assessment of authority boundaries

The intended separation is correct:

- Recommendation governance owns recommendations, evidence, review, approval, follow-up proposals, and ratification.
- Campaign Operations may own explicitly authorized operational intent, control state, budgets, reservations, request issuance, and operational reconciliation.
- Experiment lifecycle owns experiment creation, identity, lifecycle transitions, and terminal experiment state.
- Scheduler owns eligibility polling, capacity, claims, process launch, completion processing, and orphan recovery.
- Workers own bounded computation and report facts.
- PostgreSQL owns committed durable truth.

The following belong nowhere:

- score- or ratification-driven automatic experiment creation;
- scheduler interpretation of recommendation policy;
- duplicated campaign membership reconstructed from current queries;
- Campaign Operations directly launching workers;
- Campaign Operations rewriting experiment outcomes;
- treating budget availability as execution authorization;
- treating process presence as durable completion evidence.

## 8. Assessment of scheduler interaction

The boundary required by ADR-0004 is sound, but no submitted interaction contract exists.

A valid design must preserve these rules:

- The scheduler must not poll Phase 6 governance or Campaign Operations policy tables.
- Campaign Operations must not assign worker capacity or select runnable phases.
- Any work handed to the existing scheduler must first become valid ordinary experiment lifecycle state through an accepted workflow.
- Campaign Operations may observe durable experiment state but must not create a second scheduler state machine.
- Worker launch must remain outside database transactions.
- A new scheduler work class would require its own accepted ADR covering eligibility, claims, capacity, fairness, retry, recovery, operator control, observability, and shutdown.

## 9. Lifecycle, transactions, concurrency, replay, and recovery

These dimensions are not approvable because the recommendation contains no design.

Before acceptance, the replacement must resolve at least:

- Control state versus derived member state. An operational campaign’s administrative state cannot overwrite member experiment truth.
- Atomic handoff. It must choose whether reservation, request, and downstream lifecycle creation commit together or use a durable compensating protocol.
- Replay after progression. Phase 5 launch reports an ordinary exact retry only while experiments remain `pending/train`; progressed experiments are not rewound ([supported matrix](</Volumes/Developer SSD/ExpertAdvisor/docs/Phase5ExperimentRecommendationCampaignLaunch.rst:64>)). Campaign Operations therefore needs its own durable request-to-result binding.
- Lock ordering. Budget, campaign, reservation, request, proposal, execution, activation, and experiment locks need one global order.
- Recovery bounds. Reconciliation must use bounded batches and must not repeatedly create requests or release reservations merely because a process is absent.
- Cancellation races. The contract must cover cancellation before request creation, after request creation, after lifecycle activation, after scheduler claim, during execution, and concurrent with completion.
- Completion. Operational completion must be distinct from scientific success and must define how terminal failures, cancellations, partial membership, unresolved reservations, and inconsistent evidence affect it.
- Uncertain commits. Retrying after a lost response must look up authoritative request identity rather than repeat side effects blindly.

PostgreSQL should remain the single durable store, but “single source of truth” does not mean one table owns every fact. Each fact must have exactly one domain owner, with reconciliation producing a projection or append-only observation rather than competing truth.

## 10. Budgeting, reservations, and operational-request ownership

No budget or reservation proposal was delivered.

A reviewable design must specify:

- the budgeted resource and unit;
- who grants and amends a budget;
- whether budget is immutable, versioned, or append-only;
- reservation conflict scope;
- when a reservation becomes committed consumption;
- whether failure or cancellation releases anything;
- how uncertain downstream state is handled;
- how overrun is prevented;
- whether unused reservations expire and who may reclaim them;
- the invariant relating grants, reservations, committed consumption, and remaining capacity;
- the exact one-to-one or one-to-many relationship between reservations and requests.

Budget authority must remain separate from scheduler capacity. A campaign may have budget but no scheduler capacity; scheduler capacity must not imply campaign authority.

An operational request must be Campaign Operations-owned durable intent, not a scheduler claim and not an experiment outcome. It must identify the exact upstream authorization and exact intended downstream action. At most one accepted request may consume a reservation, and exact retries must return the same request and downstream binding.

## 11. Privileges, auditability, and testability

Phase 6 provides a good least-privilege precedent: `SELECT`, payload-column `INSERT`, sequence `USAGE`, no update/delete/truncate, and revoked trigger-function execution ([migration 044 privileges](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/044_experiment_recommendation_campaign_follow_up_proposal_ratification.sql:146>)).

No Campaign Operations privilege model was supplied. Required architecture includes:

- separate authority for operator decisions, dispatch, reconciliation, and scheduler reads;
- no scheduler access to recommendation-governance policy unless strictly necessary;
- column- or function-scoped mutations for mutable control state;
- append-only decision, reservation, request, cancellation, reconciliation, and terminal audit evidence;
- actor, reason, expected identity, prior/resulting state, causal request, downstream IDs, and timestamp in audit records;
- protection against using audit rows as a second mutable current-state table.

Required tests include pure state/identity tests, migration and ACL tests, independent-connection concurrency tests, uncertain-retry tests, budget races, request duplication, reservation leakage, cancellation/completion races, bounded restart recovery, scheduler-isolation regressions, and unchanged Phase 4–6 evidence.

## 12. Documentation and terminology findings

- The submitted `.txt` is an ignored, untracked transcript, not part of the authoritative architecture hierarchy.
- Even a complete version would remain a proposal until represented by accepted ADRs and aligned architecture volumes.
- ADR-0009, Volume VIII, and Volume XII must agree on whether Phase 6D is proposed or accepted.
- “Campaign Operations” overlaps both existing “recommendation campaign” terminology and Volume X “Research Automation.”
- The replacement must define, without synonyms:

  - recommendation campaign;
  - campaign materialization;
  - operational campaign;
  - operational attempt;
  - authorization;
  - ratification;
  - execution request;
  - activation;
  - scheduler admission;
  - reservation;
  - cancellation;
  - operational completion;
  - scientific outcome.

## 13. Recommended disposition

**Revise and re-review.**

“Accept with required revisions” is too permissive because the missing material concerns the core authority, lifecycle, concurrency, and safety model. “Reject” would overstate the result: the separate-subsystem direction is plausible and largely consistent with Volume X, but no reviewable architecture was delivered.

## 14. Exact conditions before implementation begins

Implementation must not begin until all of these are satisfied:

1. A complete Campaign Operations architecture document replaces the failed transcript.
2. ADR-0009’s status and affected volumes are made consistent.
3. An accepted ADR establishes whether Campaign Operations—not a Phase 6E recommendation-planning bridge—is the next bounded concern.
4. The ADR defines the exact upstream operational authorization; ratification alone is explicitly insufficient.
5. Domain ownership and identities are complete and collision-safe.
6. The full lifecycle and terminal-state matrix is accepted.
7. Budget, reservation, and request invariants are accepted.
8. Transaction boundaries, uncertain outcomes, and the global lock order are accepted.
9. Scheduler integration is either limited to existing lifecycle state or covered by a separate scheduler-work-class ADR.
10. Restart reconciliation, cancellation, completion, and replay semantics are accepted.
11. Privilege, audit, migration, rollback, compatibility, and test plans are complete.
12. The revised architecture receives another independent fail-closed review.
13. Only then may the smallest accepted implementation increment begin.

## 15. Overall readiness assessment

- Conceptual direction: **plausible but not selected authoritatively**.
- Phase 4–6 implementation foundation: **substantial and reusable**.
- Phase 6 formal architectural closure: **incomplete because ADR-0009 remains proposed**.
- Campaign Operations architecture: **not delivered**.
- Implementation readiness: **not ready**.
- Migration readiness: **not ready**.
- Scheduler integration readiness: **not demonstrated**.

Review audit:

- Files changed: none.
- Behavioral change: none.
- Tests/builds run: none; this was a read-only architecture review.
- Database or production processes accessed: none.
- Current branch/HEAD: `campaign-operations` at `d78771f`.
- `git status --short`: no output.
- `git diff --stat`: no output.
- `git diff --check`: passed with no output.
      438.02 real         8.53 user         5.97 sys
           164102144  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
              382738  page reclaims
               65558  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                 946  messages sent
               14188  messages received
                  51  signals received
                8980  voluntary context switches
              231231  involuntary context switches
         34144935167  instructions retired
         23490808317  cycles elapsed
           111220928  peak memory footprint
