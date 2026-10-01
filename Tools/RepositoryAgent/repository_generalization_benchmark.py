#!/usr/bin/env python3
"""Non-scheduler generalization benchmark for the frozen Phase 5A RepositoryAgent.

The benchmark deliberately contains no known repository filenames, symbol names,
or line numbers. It measures discovery and evidence assembly before any retrieval
improvements are made.
"""

TOPICS = (
    {
        "id": "model_persistence",
        "title": "model persistence",
        "question": (
            "Trace how a trained model is serialized or persisted and how that persisted "
            "model is subsequently reconstructed or loaded for use. Establish both the "
            "write path and the read/reconstruction path from source code."
        ),
        "hints": (
            "Discover the persistence implementation from repository source. Follow the "
            "data or object from trained model state into durable storage, then independently "
            "find the path that reads durable state and reconstructs a usable model."
        ),
        "required_evidence": ("model_persistence_write", "model_persistence_read"),
    },
    {
        "id": "training_input_construction",
        "title": "training input construction",
        "question": (
            "Trace how market observations and derived features become the model input used "
            "during training. Establish the feature/input construction path and the point "
            "where that constructed input is consumed by the training model."
        ),
        "hints": (
            "Start from concepts in the question rather than assuming a filename. Follow "
            "source-visible data flow from market or feature preparation toward the training "
            "tensor/input and then to its model consumer."
        ),
        "required_evidence": ("training_feature_construction", "training_model_consumption"),
    },
    {
        "id": "inference_evaluation",
        "title": "inference to evaluation",
        "question": (
            "Trace how model inference output flows into evaluation or classification results. "
            "Establish where prediction output is produced and the connected path where that "
            "output is interpreted, compared, classified, or accumulated into evaluation results."
        ),
        "hints": (
            "Discover the inference and evaluation implementations from source and follow an "
            "actual source-visible handoff. Do not treat unrelated inference and metric code as "
            "a connected path without evidence of the linkage."
        ),
        "required_evidence": ("inference_output", "inference_to_evaluation_linkage"),
    },
    {
        "id": "economic_event_features",
        "title": "economic-event feature integration",
        "question": (
            "Trace how economic-event information becomes part of model input. Establish the "
            "economic-event feature construction/integration path and the source-enforced causal "
            "or point-in-time boundary that prevents unavailable future information from entering "
            "the feature state."
        ),
        "hints": (
            "Discover the economic-event implementation from source. Require concrete evidence "
            "both for feature integration into model-facing input and for the availability/time "
            "boundary; naming something causal is not by itself proof of the boundary."
        ),
        "required_evidence": ("economic_event_input_integration", "point_in_time_boundary"),
    },
    {
        "id": "profitability_observation",
        "title": "profitability observation provenance",
        "question": (
            "Trace how inference results become profitability observations and how those "
            "observations are persisted. Establish the calculation or derivation from inference "
            "evidence and the connected persistence path for the resulting profitability data."
        ),
        "hints": (
            "Discover the profitability implementation from source. Follow actual inference-derived "
            "values into profitability calculation and then into durable persistence; a schema or "
            "insert statement alone does not establish derivation provenance."
        ),
        "required_evidence": ("profitability_derivation", "profitability_persistence"),
    },
    {
        "id": "checkpoint_continuation",
        "title": "checkpoint to continuation decision",
        "question": (
            "Trace how checkpoint evaluation evidence reaches the decision to continue or stop "
            "training. Establish the checkpoint evidence/assessment path and the connected decision "
            "or gating logic that determines continuation."
        ),
        "hints": (
            "Discover the checkpoint and continuation implementations from source. Follow concrete "
            "values or result objects across the boundary between checkpoint evaluation and the "
            "continuation decision; do not infer linkage from similarly named code."
        ),
        "required_evidence": ("checkpoint_evaluation_evidence", "continuation_decision_linkage"),
    },
)

# No benchmark-specific symbol, filename, line, search, or read seeds. These plans
# state only the kind of source relationship that must be discovered.
TOPIC_NAVIGATION = {
    "model_persistence": "Find a concrete durable write path and a concrete durable read/reconstruction path; verify both independently.",
    "training_input_construction": "Follow feature/input data flow forward until the constructed training input is consumed by model execution.",
    "inference_evaluation": "Follow prediction output across a source-visible handoff into evaluation, classification, comparison, or metric accumulation.",
    "economic_event_features": "Establish both model-facing event-feature integration and the code condition enforcing information availability in time.",
    "profitability_observation": "Follow inference-derived values into profitability derivation and then into the persistence operation for the resulting observation.",
    "checkpoint_continuation": "Follow checkpoint evaluation evidence into the branch, policy, gate, or decision that controls continuation.",
}

TOPIC_BOOTSTRAP_SEARCHES = {}
TOPIC_BOOTSTRAP_READS = {}

CATEGORY_DEFINITIONS = {
    "model_persistence_write": "Evidence must directly establish serialization or persistence of trained model state into durable storage.",
    "model_persistence_read": "Evidence must directly establish reading persisted model state and reconstructing or loading a model usable by the application.",
    "training_feature_construction": "Evidence must directly establish construction of model-facing training features, tensors, vectors, or equivalent input from market/derived data.",
    "training_model_consumption": "Evidence must directly establish that the constructed training input is passed to or consumed by the model/training execution path.",
    "inference_output": "Evidence must directly establish production of model prediction/inference output used by downstream application logic.",
    "inference_to_evaluation_linkage": "Evidence must directly establish a source-visible handoff from inference/prediction output into evaluation, classification, comparison, or metric accumulation.",
    "economic_event_input_integration": "Evidence must directly establish economic-event-derived information entering model-facing feature/input construction.",
    "point_in_time_boundary": "Evidence must directly establish a time/availability condition that prevents information unavailable at the modeled instant from entering the economic-event feature state.",
    "profitability_derivation": "Evidence must directly establish profitability values being calculated or derived from inference/prediction outcomes or their evaluated observations.",
    "profitability_persistence": "Evidence must directly establish persistence of the derived profitability observation into durable storage.",
    "checkpoint_evaluation_evidence": "Evidence must directly establish checkpoint evaluation results, measurements, or assessment state that can feed continuation logic.",
    "continuation_decision_linkage": "Evidence must directly establish a source-visible path from checkpoint evaluation evidence into a branch, gate, policy, or decision controlling whether training continues or stops.",
}

INVESTIGATION_SYSTEM = r"""
You are a read-only source-code investigator for the ExpertAdvisor C++ project.
The host controller assigns exactly one architecture/data-flow topic at a time.
Investigate only that topic and discover the implementation from repository source.

You have exactly these repository tools:
1. list_files  {"tool":"list_files","prefix":"Sources"}
2. search      {"tool":"search","pattern":"some pattern"}
3. read        {"tool":"read","file":"Sources/File.cpp","start":100,"end":250}

While investigating:
- Output exactly ONE JSON tool request and nothing else.
- Do not wrap JSON in markdown.
- Prefer search before large reads.
- Follow references and data/control handoffs when necessary.
- Never invent filenames, symbols, source text, or line numbers.
- A read may contain at most 500 lines.
- Do not investigate unrelated topics.
- Do not propose code or architectural changes.
- Do not use general knowledge as evidence for repository behavior.
- Do not output FINAL:; the host controller owns completion.
"""

EVIDENCE_SYSTEM = r"""
You are extracting evidence from a completed read-only source investigation.
Return exactly ONE JSON object and no markdown.

Schema:
{
  "status": "supported" | "insufficient",
  "summary": "short explanation of what the inspected source establishes",
  "evidence": [
    {"file":"Sources/File.cpp","start":100,"end":125,
     "category":"one exact controller-required category",
     "establishes":"what these exact lines establish"}
  ],
  "uncertainties": ["anything important not established by inspected source"]
}

Rules:
- Use only source evidence actually present in the investigation transcript.
- Never invent filenames or line numbers.
- Evidence ranges must correspond to source lines actually returned.
- Keep ranges as narrow as reasonably possible.
- Every range must support its claimed category.
- A topic is supported only when ALL controller-required categories are established.
- Separate implementations do not prove a data/control-flow linkage unless source establishes the handoff.
- If evidence is inadequate, use status "insufficient".
- Do not propose changes.
"""

SYNTHESIS_SYSTEM = r"""
You are a source-code architecture analyst. Use only controller-approved evidence.
Do not invent repository facts, upgrade insufficient topics, or borrow evidence across topics.
Clearly distinguish established behavior from missing evidence and cite exact source ranges.
Do not propose architectural changes. Begin exactly with FINAL:.
"""

BENCHMARK = {
    "name": "repository_generalization",
    "topics": TOPICS,
    "topic_navigation": TOPIC_NAVIGATION,
    "topic_bootstrap_searches": TOPIC_BOOTSTRAP_SEARCHES,
    "topic_bootstrap_reads": TOPIC_BOOTSTRAP_READS,
    "category_definitions": CATEGORY_DEFINITIONS,
    "investigation_system": INVESTIGATION_SYSTEM,
    "evidence_system": EVIDENCE_SYSTEM,
    "synthesis_system": SYNTHESIS_SYSTEM,
}
