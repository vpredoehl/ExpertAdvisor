#!/usr/bin/env python3

"""Frozen scheduler architecture regression benchmark for RepositoryAgent.

This module contains benchmark-specific topics, navigation guidance, bootstrap
searches, and bootstrap reads. The production controller imports these data but
keeps retrieval, provenance, verification, ledger, and caching mechanics generic.
"""

TOPICS = (
    {
        "id": "training_admission",
        "title": "training admission",
        "question": (
            "Determine how the scheduler decides whether a training worker "
            "may be admitted. Identify capacity checks, policy/admission "
            "checks, candidate selection, and any relevant scheduler state."
        ),
        "hints": (
            "Search for training admission, train capacity, "
            "TrainingWorkerSelection, SchedulerAdmissionService, "
            "FinalExperimentDispatchService, and train worker limits. "
            "Trace role-specific candidate/selection logic AND the capacity or "
            "dispatch gate that decides whether training may start."
        ),
        "required_evidence": (
            "role_specific_candidate_or_selection",
            "capacity_or_dispatch_gate",
        ),
    },
    {
        "id": "training_launch",
        "title": "training launch",
        "question": (
            "Determine how an admitted training worker is actually launched. "
            "Identify command construction, worker-attempt registration or "
            "lifecycle handling, process launch, and experiment state "
            "transitions."
        ),
        "hints": (
            "Search for TrainingWorkerCommand, worker registration, "
            "WorkerProcessController, launch/spawn, and training dispatch. "
            "Do not stop at generic spawn(). Trace the training-specific "
            "command/dispatch path into the process controller."
        ),
        "required_evidence": (
            "worker_specific_path",
            "worker_to_spawn_linkage",
            "process_spawn",
        ),
    },
    {
        "id": "inference_admission",
        "title": "inference admission",
        "question": (
            "Determine how the scheduler decides whether an inference worker "
            "may be admitted. Identify capacity checks, candidate/model "
            "selection, semantic compatibility checks if relevant, and "
            "admission policy."
        ),
        "hints": (
            "Search for InferenceWorkerSelection, infer capacity, semantic "
            "admission, inference dispatch, and SchedulerAdmissionService. "
            "Trace inference-specific candidate/selection logic AND the capacity or "
            "dispatch gate that decides whether inference may start."
        ),
        "required_evidence": (
            "role_specific_candidate_or_selection",
            "capacity_or_dispatch_gate",
        ),
    },
    {
        "id": "inference_launch",
        "title": "inference launch",
        "question": (
            "Determine how an admitted inference worker is actually launched. "
            "Identify command construction, worker-attempt lifecycle, process "
            "launch, and experiment state transitions."
        ),
        "hints": (
            "Search for inference worker launch/spawn, worker registration, "
            "WorkerProcessController, command construction, and infer phase. "
            "Do not treat generic spawn() as proof. Trace an "
            "inference-specific dispatch or command path into spawn(), including the "
            "source-visible bridge/caller-callee handoff."
        ),
        "required_evidence": (
            "worker_specific_path",
            "worker_to_spawn_linkage",
            "process_spawn",
        ),
    },
    {
        "id": "analysis_admission",
        "title": "analysis admission",
        "question": (
            "Determine how the scheduler decides whether an analysis worker "
            "may be admitted. Identify capacity checks, readiness/state "
            "requirements, and any analysis-specific orchestration."
        ),
        "hints": (
            "Search for analysis capacity, analyze phase, "
            "CheckpointAnalysisOrchestrationService, final analysis, "
            "and analysis dispatch. Trace outward from analysis "
            "orchestration to its caller and capacity/readiness gate."
        ),
        "required_evidence": (
            "role_specific_candidate_or_selection",
            "capacity_or_dispatch_gate",
        ),
    },
    {
        "id": "analysis_launch",
        "title": "analysis launch",
        "question": (
            "Determine how admitted analysis work is actually executed. "
            "First establish whether the architecture uses a separate child "
            "process or executes analysis in-process. Then identify the "
            "analysis-specific dispatch/call path, execution mechanism, "
            "worker-attempt lifecycle, and relevant state transitions."
        ),
        "hints": (
            "Search for CheckpointAnalysisOrchestrationService, analyze phase, "
            "analysis dispatch/callers, CheckpointAnalysisOperations, and only "
            "then WorkerProcessController if the source shows a child process. "
            "A valid result may establish in-process execution instead of spawn."
        ),
        "required_evidence": (
            "analysis_specific_path",
            "analysis_execution_mechanism",
        ),
    },
    {
        "id": "priority_preemption",
        "title": "priority/preemption",
        "question": (
            "Determine how scheduler priority ordering and preemption work. "
            "Identify candidate/victim ordering, which phases may be "
            "preempted, strict priority comparison, process stopping, "
            "database/lifecycle transitions, rollback/compensation behavior, "
            "and resume origin."
        ),
        "hints": (
            "Search for preempt, scheduler_priority, CanPreempt, "
            "PriorityRank, selectPreemptionVictim, loadPreemptionVictim, "
            "scheduler_resume_origin, SIGSTOP, and rollback compensation. "
            "Establish both the strict priority/victim-selection rule AND "
            "the process/database state transition."
        ),
        "required_evidence": (
            "priority_or_victim_eligibility",
            "preemption_execution",
        ),
    },
)


TOPIC_NAVIGATION = {
    "training_admission": (
        "Start from TrainingWorkerSelection for role-specific candidate/selection evidence. Do not require a separate eligibility predicate if the source architecture does not have one. "
        "Then inspect callers in ProductionSchedulerDaemon and/or "
        "FinalExperimentDispatchService to find an explicit branch that gates "
        "training dispatch on capacity/readiness. A hasCapacity() helper by "
        "itself is not enough; read the caller that acts on its result."
    ),
    "training_launch": (
        "Start with the final-dispatch operations.prepareReservedLaunch callback in "
        "ProductionSchedulerDaemon. Read one contiguous range showing the "
        "FinalExperimentPhase::Train branch calling BuildTrainCommand, assignment "
        "into preparedCommand, and operations.launchPreparedWorker moving that SAME "
        "preparedCommand into LaunchReservedChildProcess. This is the primary "
        "training worker_specific_path + worker_to_spawn_linkage chain. Then trace "
        "LaunchReservedChildProcess to the concrete process creation operation. "
        "BeginTrainingWorkerCommand/BuildTrainCommand may corroborate command "
        "semantics, but do not substitute isolated command construction for the "
        "prepareReservedLaunch -> preparedCommand -> launchPreparedWorker handoff."
    ),
    "inference_admission": (
        "Start from LoadInferenceWorkerSelection and read its implementation, then "
        "read a caller far enough past the returned selection to show how it is "
        "used to select/resolve inference work. For capacity, do NOT stop at the "
        "ProductionSchedulerDaemon calculation of batch.freeSlots. Trace that field "
        "into FinalExperimentDispatchService and retrieve the branch/loop that uses "
        "freeSlots or maximumCapacity to limit/reserve dispatch. Also trace "
        "reserveWorkerAttempt(candidate, phase, maximumCapacity, ...) far enough to "
        "show the failed-reservation control-flow consequence. The capacity calculation "
        "alone is not a gate; the consumer that suppresses or limits dispatch is."
    ),
    "inference_launch": (
        "Start with the final-dispatch operations.prepareReservedLaunch callback in "
        "ProductionSchedulerDaemon. Read one contiguous range showing the "
        "FinalExperimentPhase::Infer branch calling BuildInferCommand, assignment "
        "into preparedCommand, and operations.launchPreparedWorker moving that SAME "
        "preparedCommand into LaunchReservedChildProcess. This is the primary "
        "inference worker_specific_path + worker_to_spawn_linkage chain. Then trace "
        "LaunchReservedChildProcess to the concrete process creation operation. "
        "Do not use generic scheduler CLI arguments as inference-path evidence."
    ),
    "analysis_admission": (
        "Start inside ClaimCheckpointAnalysis, not at its signature. For "
        "capacity_or_dispatch_gate, retrieve the complete hasCapacity(\"analyze\", "
        "options.maxAnalyzeProcs) branch including the return std::nullopt consequence. "
        "For role_specific_candidate_or_selection, retrieve the contiguous body that "
        "loads pending checkpoint-evaluation rows for status pending/phase analyze, "
        "rejects an empty result, chooses pending.front(), reserves the checkpoint "
        "analysis attempt, and returns the resulting claim. The claim callback binding "
        "to ClaimCheckpointAnalysis may corroborate the path, but a function signature "
        "or callback name alone is not sufficient."
    ),
    "analysis_launch": (
        "First determine whether analysis is launched as a separate child or executed in the current scheduler process. Start from construction of CheckpointAnalysisOperations. Retrieve the execute callback binding/body and the service runOne call site, so the evidence proves the concrete callback binding plus operations_.execute(*claim), not merely the latter endpoint. If the callback directly performs analysis in the scheduler process, prove that call chain; only seek WorkerProcessController/spawn if source shows a separate child."
    ),
    "priority_preemption": (
        "Read SchedulerPolicy::CanPreempt/PriorityRank early, then the concrete "
        "preemption region. For execution, do not stop at PauseWorker or at the "
        "returned signal list: search PauseWorker's definition, then trace the "
        "NativeProcessOperations/CreateNativeProcessOperations callback that actually "
        "invokes the OS/process signal operation. Retrieve that implementation range."
    ),
}


# Controller-owned deterministic navigation seeds. These searches are executed
# before the model begins free navigation. They are clues only: they never count
# as source evidence, and they do not consume the model's repository-tool budget.
TOPIC_BOOTSTRAP_SEARCHES = {
    "training_admission": (
        "TrainingWorkerSelection",
        "maxTrainProcs",
        "hasCapacity",
    ),
    "training_launch": (
        "BeginTrainingWorkerCommand",
        "BuildTrainCommand",
        "prepareReservedLaunch",
        "launchPreparedWorker",
        "preparedCommand",
        "--train",
        "WorkerLaunchRequest",
        "LaunchReservedChildProcess",
        "processes.spawn",
        ".spawn(",
    ),
    "inference_admission": (
        "InferenceWorkerSelection",
        "LoadInferenceWorkerSelection",
        "maxInferProcs",
        "freeSlots",
        "maximumCapacity",
        "reserveWorkerAttempt",
        "AvailableWorkerProcessSlots",
        "launchAllowed",
        "FinalExperimentDispatchService",
        "selection.",
    ),
    "inference_launch": (
        "--infer",
        "BuildInferCommand",
        "prepareReservedLaunch",
        "launchPreparedWorker",
        "preparedCommand",
        "inference dispatch",
        "WorkerLaunchRequest",
        "LaunchReservedChildProcess",
        "processes.spawn",
        ".spawn(",
    ),
    "analysis_admission": (
        "CheckpointAnalysisOperations",
        "CheckpointAnalysisOrchestrationService",
        "ClaimCheckpointAnalysis",
        "maxAnalyzeProcs",
        "hasCapacity(\"analyze\"",
        "runOne(",
    ),
    "analysis_launch": (
        "CheckpointAnalysisOrchestrationService",
        "CheckpointAnalysisOperations",
        "--analyze",
    ),
    "priority_preemption": (
        "CanPreempt",
        "selectPreemptionVictim",
        "PauseWorker",
        "CreateNativeProcessOperations",
        "SIGSTOP",
        ".signal(",
    ),
}


# V21 controller-owned bootstrap reads for launch topics and inference admission. These are
# retrieval/navigation aids only; they do not bypass exact-line provenance or
# semantic verification. The final-dispatch callback is the source-visible
# bridge V16 discovered under a different topic but failed to retrieve inside
# the isolated launch investigations.
TOPIC_BOOTSTRAP_READS = {
    "training_admission": (
        ("Sources/SchedulerCore/FinalExperimentDispatchService.cpp", 180, 270),
    ),
    "training_launch": (
        ("Sources/SchedulerCore/ProductionSchedulerDaemon.cpp", 10327, 10377),
        ("Sources/SchedulerCore/ProductionSchedulerDaemon.cpp", 3388, 3410),
        ("Sources/SchedulerCore/WorkerProcessController.cpp", 120, 160),
    ),
    "inference_admission": (
        ("Sources/SchedulerCore/ProductionSchedulerDaemon.cpp", 2025, 2085),
        ("Sources/SchedulerCore/ProductionSchedulerDaemon.cpp", 10280, 10320),
        ("Sources/SchedulerCore/FinalExperimentDispatchService.cpp", 180, 270),
    ),
    "inference_launch": (
        ("Sources/SchedulerCore/ProductionSchedulerDaemon.cpp", 10327, 10377),
        ("Sources/SchedulerCore/ProductionSchedulerDaemon.cpp", 3388, 3410),
        ("Sources/SchedulerCore/WorkerProcessController.cpp", 120, 160),
    ),
}


# Benchmark-owned semantic criteria and prompt policy. Moved byte-for-byte
# from the frozen M4 controller/verifier so benchmark injection changes no
# scheduler behavior.
CATEGORY_DEFINITIONS = {
    "role_specific_candidate_or_selection": (
        "The excerpt must directly establish that the scheduler is selecting, "
        "claiming, resolving, or otherwise identifying work for the named role. "
        "A distinct eligibility predicate is NOT required. Capacity alone, a "
        "role name in a log, or generic worker infrastructure does not qualify."
    ),
    "capacity_or_dispatch_gate": (
        "The excerpt must directly establish a condition controlling whether "
        "work may be admitted, claimed, dispatched, or started. Capability "
        "computation alone does not qualify. A capacity-class log field does "
        "not qualify."
    ),
    "worker_specific_path": (
        "The excerpt must establish a training-, inference-, or analysis-"
        "specific command, dispatch, or call path leading toward launch. "
        "Generic WorkerLaunchRequest/process infrastructure does not qualify."
    ),
    "worker_to_spawn_linkage": (
        "The excerpt must directly establish a source-visible bridge from the "
        "named training or inference launch path toward the process-launch "
        "operation. A role-specific command and a generic fork/exec shown only "
        "in unrelated excerpts do not qualify. The bridge may be a concrete "
        "caller/callee handoff, launch helper invocation, or equivalent control "
        "flow that connects the role-specific path to the spawn infrastructure."
    ),
    "process_spawn": (
        "The excerpt must directly show the concrete process-creation/execution "
        "operation itself, such as a literal fork(), exec*(), posix_spawn(), or "
        "equivalent OS/process-controller spawn implementation. Merely calling a "
        "helper named Launch*, spawn*, or execute* without showing its concrete "
        "process-creation operation does NOT qualify. Generic concrete process "
        "creation is allowed for this category."
    ),
    "analysis_specific_path": (
        "The excerpt must directly establish an analysis-specific dispatch, "
        "orchestration, callback binding, or call path that leads to execution "
        "of analysis work. A role name in a log is not enough."
    ),
    "analysis_execution_mechanism": (
        "The excerpt must directly establish how analysis work actually runs. "
        "Either a separate process creation/exec path OR a direct in-process "
        "call to the analysis execution operation qualifies. Do not assume "
        "spawn merely because generic process infrastructure exists."
    ),
    "priority_or_victim_eligibility": (
        "The excerpt must directly establish scheduler priority ordering, "
        "strict priority comparison, or victim eligibility/selection."
    ),
    "preemption_execution": (
        "The evidence must directly establish the executable source-code mechanism "
        "used to carry out preemption, or an authoritative lifecycle/state transition "
        "caused by preemption. For architecture analysis, a connected source path "
        "from the preemption operation to the process-control/signal operation is "
        "sufficient; runtime telemetry proving that a historical signal was delivered "
        "is NOT required. Do not claim more than the source path establishes."
    ),
}

INVESTIGATION_SYSTEM = r"""
You are a read-only source-code investigator for the ExpertAdvisor C++ project.

The HOST CONTROLLER assigns exactly one architecture topic at a time.
You must investigate ONLY that assigned topic.

Admission ontology note: do not invent a separate worker eligibility predicate.
For admission topics, role-specific candidate/selection/claim evidence and the
capacity/dispatch gate are separate controller requirements.

You do not decide when the overall architecture investigation is complete.
You are not allowed to produce FINAL:.

You have exactly these repository tools:

1. list_files
   {"tool":"list_files","prefix":"Sources/SchedulerCore"}

2. search
   {"tool":"search","pattern":"some pattern"}

3. read
   {"tool":"read",
    "file":"Sources/SchedulerCore/File.cpp",
    "start":100,
    "end":250}

While investigating:
- Output exactly ONE JSON tool request and nothing else.
- Do not wrap JSON in markdown.
- Prefer search before large reads.
- Follow references when necessary.
- Never invent filenames, symbols, source text, or line numbers.
- A read may contain at most 500 lines.
- Do not investigate unrelated architecture topics.
- Do not propose code or architectural changes.
- Do not use general knowledge as evidence for repository behavior.
"""


EVIDENCE_SYSTEM = r"""
You are extracting evidence from a completed read-only source investigation.

Return exactly ONE JSON object and no markdown.

Schema:

{
  "status": "supported" | "insufficient",
  "summary": "short explanation of what the inspected source establishes",
  "evidence": [
    {
      "file": "Sources/SchedulerCore/File.cpp",
      "start": 100,
      "end": 125,
      "category": "one exact controller-required category",
      "establishes": "what these exact lines establish"
    }
  ],
  "uncertainties": [
    "anything important not established by the inspected source"
  ]
}

Rules:
- Use only source evidence actually present in the investigation transcript.
- Never invent filenames or line numbers.
- Evidence ranges must correspond to source lines that were actually returned.
- Keep ranges as narrow as reasonably possible.
- Do not include a range merely because it was read; it must support the claim.
- A topic is "supported" only when ALL controller-required evidence
  categories for that topic are established.
- Generic infrastructure alone does not prove that a specific worker kind
  uses that infrastructure.
- For training/inference launch topics, generic WorkerProcessController::spawn
  evidence is insufficient unless retrieved evidence also establishes the
  worker-specific path AND a source-visible worker_to_spawn_linkage bridge.
- For analysis launch, do not require spawn if the retrieved source instead
  establishes an analysis-specific path and direct in-process execution.
- For admission topics, worker compatibility/selection alone is insufficient
  unless retrieved evidence also establishes the relevant capacity,
  readiness, admission, or dispatch gate.
- For priority/preemption, process stopping alone is insufficient unless
  retrieved evidence also establishes priority comparison or victim
  eligibility.
- If the evidence is inadequate, use status "insufficient".
- Do not propose changes.
"""


SYNTHESIS_SYSTEM = r"""
You are a source-code architecture analyst.

The host controller has already completed seven separate read-only repository
investigations. You will receive their structured evidence packages.

Your job is synthesis only.

Rules:
- Do not invent repository facts beyond the supplied evidence packages.
- Clearly distinguish confirmed behavior from uncertainty.
- Cover all seven required areas:
  1. training admission
  2. training launch
  3. inference admission
  4. inference launch
  5. analysis admission
  6. analysis launch
  7. priority/preemption
- CONTROLLER STATUS IS AUTHORITATIVE.
- If a package is marked insufficient, say that the requested point was not
  fully established.
- For an insufficient package, describe only its accepted evidence and the
  categories still missing.
- Never state or imply that a missing category was established.
- Never upgrade an insufficient package to a positive conclusion.
- Package summaries, extractor claims, uncertainties, and rejected evidence
  are not evidence.
- Do not propose architectural changes.
- Cite factual claims using exact source references:
  Sources/SchedulerCore/File.cpp:120-145
- Do not repeat the same citation unnecessarily.
- Do not append citation spam.
- Begin exactly with:

FINAL:
"""

BENCHMARK = {
    "name": "scheduler_architecture",
    "topics": TOPICS,
    "topic_navigation": TOPIC_NAVIGATION,
    "topic_bootstrap_searches": TOPIC_BOOTSTRAP_SEARCHES,
    "topic_bootstrap_reads": TOPIC_BOOTSTRAP_READS,
    "category_definitions": CATEGORY_DEFINITIONS,
    "investigation_system": INVESTIGATION_SYSTEM,
    "evidence_system": EVIDENCE_SYSTEM,
    "synthesis_system": SYNTHESIS_SYSTEM,
}
