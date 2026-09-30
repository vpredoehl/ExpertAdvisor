#!/usr/bin/env python3
"""Frozen semantic-verification behavior."""
import json
from mlx_lm import generate

def extract_json_object(text):
    """Decode the first complete JSON object without greedy brace matching."""
    text = text.strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj

    return None

def render_prompt(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

def run_generation(model, tokenizer, messages, max_tokens):
    prompt = render_prompt(tokenizer, messages)

    return generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False,
    ).strip()

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

SEMANTIC_VERIFY_SYSTEM = r"""
You are an independent source-evidence verifier.

Judge only whether the exact source excerpt directly establishes the requested
evidence category for the assigned repository-analysis topic.

Rules:
- Use only the supplied excerpt or explicitly supplied evidence bundle.
- Do not use outside knowledge or unstated call chains.
- This is source-code ARCHITECTURE verification, not historical runtime-event
  verification. A concrete executable call/control-flow path establishes the
  architectural mechanism even without telemetry proving a particular past run.
- Do not demand proof that a historical fork, exec, signal, claim, or dispatch
  actually occurred when the category asks how the architecture performs it.
- Do not infer semantics merely from a filename, symbol name, comment, or log.
- Generic process infrastructure cannot establish worker_specific_path.
- worker_to_spawn_linkage requires a source-visible bridge connecting the named
  training/inference path toward spawn. A VERIFIED MULTI-RANGE bundle may establish
  that bridge when its ranges show consecutive caller/callee or argument handoffs;
  merely unrelated role-specific and generic spawn excerpts are not enough.
- For analysis launch, analysis_specific_path requires an analysis-specific
  dispatch/orchestration/call path, and analysis_execution_mechanism may be
  either a proven child-process path or a proven direct in-process execution.
- Capability mapping alone is not capacity_or_dispatch_gate.
- A capacity-class log field is not role_specific_candidate_or_selection.
- Do not require a separate eligibility predicate for admission. Role-specific
  candidate selection, claim, readiness, compatibility, or work resolution may
  establish role_specific_candidate_or_selection when shown directly.
- Calling an unknown callback/function does not prove what that callback
  checks, although an explicit branch on its returned result may establish a
  dispatch/claim gate without establishing the callback's internal policy.
- Evaluate the category, not the extractor's confidence.
- If supported, provide a narrow neutral statement that the excerpt itself
  establishes. Do not preserve an overbroad extractor claim.

Return exactly one JSON object and no markdown:

{
  "supports": true | false,
  "establishes": "precise source-grounded statement, or empty if unsupported",
  "reason": "brief explanation"
}
"""

def verify_evidence_semantics(
    model,
    tokenizer,
    topic,
    category,
    extractor_claim,
    excerpt,
):
    definition = CATEGORY_DEFINITIONS.get(category)

    if definition is None:
        return {
            "supports": False,
            "establishes": "",
            "reason": f"unknown category {category!r}",
        }

    messages = [
        {
            "role": "system",
            "content": SEMANTIC_VERIFY_SYSTEM,
        },
        {
            "role": "user",
            "content":
                f"TOPIC: {topic['title']}\n\n"
                f"CATEGORY: {category}\n\n"
                f"CATEGORY DEFINITION:\n{definition}\n\n"
                f"EXTRACTOR CLAIM (untrusted):\n{extractor_claim}\n\n"
                f"EXACT RETRIEVED SOURCE EXCERPT:\n{excerpt}\n\n"
                "Return the semantic verification JSON now."
        },
    ]

    raw = run_generation(
        model,
        tokenizer,
        messages,
        max_tokens=300,
    )

    obj = extract_json_object(raw)

    # V12: malformed verifier output gets one bounded deterministic retry.
    # The retry sees the same source excerpt/category and no new repository data.
    if not isinstance(obj, dict):
        retry_messages = messages + [
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": (
                    "Your previous response was not a complete valid JSON object. "
                    "Return exactly one JSON object matching the requested schema. "
                    "Do not add markdown or commentary."
                ),
            },
        ]
        raw = run_generation(
            model,
            tokenizer,
            retry_messages,
            max_tokens=300,
        )
        obj = extract_json_object(raw)

    if not isinstance(obj, dict):
        return {
            "supports": False,
            "establishes": "",
            "reason": "semantic verifier did not return valid JSON after one retry",
            "verifier_error": True,
        }

    supports = obj.get("supports") is True
    establishes = str(obj.get("establishes", "")).strip()
    reason = str(obj.get("reason", "")).strip()

    if supports and not establishes:
        return {
            "supports": False,
            "establishes": "",
            "reason": "semantic verifier supplied no grounded statement",
        }

    return {
        "supports": supports,
        "establishes": establishes if supports else "",
        "reason": reason,
    }

def verify_evidence_bundle_semantics(
    model,
    tokenizer,
    topic,
    category,
    items,
):
    """Verify a controller-provenanced multi-range call chain as one category.

    V14 uses this only when no individual range established the category. Each
    range has already passed exact-line provenance. The verifier may connect the
    ranges only when the source text itself exposes a concrete caller/callee,
    argument, callback, or control-flow handoff.
    """
    definition = CATEGORY_DEFINITIONS.get(category)
    if definition is None or len(items) < 2:
        return {"supports": False, "establishes": "", "reason": "bundle unavailable"}

    blocks = []
    for index, item in enumerate(items, start=1):
        blocks.append(
            f"RANGE {index}: {item['file']}:{item['start']}-{item['end']}\n"
            f"{item['excerpt']}"
        )

    messages = [
        {"role": "system", "content": SEMANTIC_VERIFY_SYSTEM},
        {
            "role": "user",
            "content":
                f"TOPIC: {topic['title']}\n\n"
                f"CATEGORY: {category}\n\n"
                f"CATEGORY DEFINITION:\n{definition}\n\n"
                "MULTI-RANGE EVIDENCE BUNDLE:\n"
                + "\n\n".join(blocks)
                + "\n\nJudge the bundle as a connected source-code path. Support it only "
                  "if the visible source establishes the intermediate handoff(s); "
                  "do not bridge unrelated endpoints by assumption. Return the "
                  "semantic verification JSON now."
        },
    ]

    raw = run_generation(model, tokenizer, messages, max_tokens=360)
    obj = extract_json_object(raw)
    if not isinstance(obj, dict):
        retry_messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": "Return exactly one valid JSON object matching the requested schema."},
        ]
        raw = run_generation(model, tokenizer, retry_messages, max_tokens=360)
        obj = extract_json_object(raw)

    if not isinstance(obj, dict):
        return {
            "supports": False,
            "establishes": "",
            "reason": "bundle verifier returned invalid JSON",
            "verifier_error": True,
        }

    supports = obj.get("supports") is True
    establishes = str(obj.get("establishes", "")).strip()
    reason = str(obj.get("reason", "")).strip()
    if supports and not establishes:
        supports = False
        reason = "bundle verifier supplied no grounded statement"

    return {
        "supports": supports,
        "establishes": establishes if supports else "",
        "reason": reason,
    }
