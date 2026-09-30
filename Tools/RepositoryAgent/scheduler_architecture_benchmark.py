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
