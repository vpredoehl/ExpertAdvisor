#!/usr/bin/env python3

import hashlib
import json
import os
import re
from pathlib import Path

from mlx_lm import load, generate

from expertadvisor_agent import (
    list_files,
    search,
    read_file,
)

from .repository_index import build_repository_index


from .evidence import (
    LEDGER_SCHEMA_VERSION, LEDGER_LEGACY_SCHEMA_VERSION, VERIFIER_SCHEMA_VERSION,
    VERIFIER_IDENTITY, LEDGER_CACHE_NAMESPACE, EVIDENCE_ACCEPTED, EVIDENCE_REJECTED,
    EVIDENCE_INDETERMINATE, EVIDENCE_VERIFIER_ERROR, LEDGER_PATH,
    evidence_source_hash, VerifiedEvidenceLedger,
)
from .verifier import (
    SEMANTIC_VERIFY_SYSTEM, extract_json_object, render_prompt,
    run_generation, verify_evidence_semantics, verify_evidence_bundle_semantics,
    configure_category_definitions,
)
from .retrieval import (
    MAX_READ_LINES,
    GENERIC_INDEX_NAVIGATION_ENABLED,
    get_repository_index, _navigation_symbol_score, topic_index_navigation,
    execute_tool, normalize_call, record_retrieved_lines, retrieved_evidence_excerpt,
    _topic_resolved_symbols, _retrieved_window, generic_relationship_bundle_candidates,
    configure_topic_navigation,
)



REPOSITORY_AGENT_METRICS = {
    "generic_index_navigation_topics": 0,
    "generic_relationship_bundle_attempts": 0,
    "generic_relationship_bundle_accepts": 0,
    "generic_relationship_bundle_cache_hits": 0,
}






MODEL = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"


# Each coverage area gets its own investigation budget.
MAX_SUCCESSFUL_TOOLS_PER_TOPIC = 14
MAX_PROTOCOL_TURNS_PER_TOPIC = 19

# V17 retains the bounded controller-directed recovery pass after the ordinary
# investigation. It does not relax provenance or semantic verification.
MAX_RECOVERY_TOOLS_PER_TOPIC = 6
MAX_RECOVERY_TURNS_PER_TOPIC = 8

# Final synthesis is not a repository investigation. It receives only the
# evidence packages produced by the configured controlled investigations.
SYNTHESIS_MAX_TOKENS = 2200

# Production refactor phase 2C: Phase 1 ledger + generic repository-index navigation.
# The ledger never bypasses provenance. A cached acceptance is reusable only
# after the exact source range has been retrieved again and its content hash
# still matches. Repository source remains read-only.


EVIDENCE_LEDGER = VerifiedEvidenceLedger()


from .benchmark import load_benchmark


def configure_benchmark(benchmark):
    """Bind benchmark-owned policy while leaving production mechanics unchanged."""
    global TOPICS, TOPIC_NAVIGATION, TOPIC_BOOTSTRAP_SEARCHES, TOPIC_BOOTSTRAP_READS
    global CATEGORY_DEFINITIONS, INVESTIGATION_SYSTEM, EVIDENCE_SYSTEM, SYNTHESIS_SYSTEM
    TOPICS = tuple(benchmark["topics"])
    TOPIC_NAVIGATION = dict(benchmark.get("topic_navigation", {}))
    TOPIC_BOOTSTRAP_SEARCHES = dict(benchmark.get("topic_bootstrap_searches", {}))
    TOPIC_BOOTSTRAP_READS = dict(benchmark.get("topic_bootstrap_reads", {}))
    CATEGORY_DEFINITIONS = dict(benchmark.get("category_definitions", {}))
    INVESTIGATION_SYSTEM = str(benchmark["investigation_system"])
    EVIDENCE_SYSTEM = str(benchmark["evidence_system"])
    SYNTHESIS_SYSTEM = str(benchmark["synthesis_system"])
    configure_topic_navigation(TOPIC_NAVIGATION)
    configure_category_definitions(CATEGORY_DEFINITIONS)


configure_benchmark(load_benchmark())

def extract_tool_call(text):
    text = text.strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "tool" in obj:
            return obj
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{.*\}', text, re.DOTALL)

    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict) and "tool" in obj:
                return obj
        except json.JSONDecodeError:
            pass

    return None






























def sanitize_evidence_package(topic, obj, retrieved_lines, model, tokenizer):
    if not isinstance(obj, dict):
        return {
            "topic": topic["title"],
            "status": "insufficient",
            "summary": "The evidence extractor did not return valid JSON.",
            "required_categories": list(topic["required_evidence"]),
            "covered_categories": [],
            "missing_categories": list(topic["required_evidence"]),
            "evidence": [],
            "uncertainties": [
                "Evidence extraction failed."
            ],
        }

    # Qwen may report its own status, but the controller does not trust it.
    # Final status is computed mechanically from provenance-valid evidence
    # coverage of every category required by this topic.
    model_status = obj.get("status")

    evidence = obj.get("evidence", [])

    if not isinstance(evidence, list):
        evidence = []
    else:
        evidence = list(evidence)

    clean_evidence = []
    provenance_valid_by_category = {}

    for item in evidence:
        if not isinstance(item, dict):
            continue

        filename = str(item.get("file", ""))

        try:
            start = int(item.get("start"))
            end = int(item.get("end"))
        except (TypeError, ValueError):
            continue

        category = str(item.get("category", "")).strip()
        establishes = str(item.get("establishes", "")).strip()

        if not filename or start < 1 or end < start or not establishes:
            continue

        if category not in topic["required_evidence"]:
            print(
                "CATEGORY REJECTED: "
                f"{filename}:{start}-{end} "
                f"used unknown category {category!r}."
            )
            continue

        # Exact-line provenance validation. Adjacent/overlapping reads are
        # naturally accepted because provenance is represented by the actual
        # numbered source lines returned, not by requested read intervals.
        excerpt = retrieved_evidence_excerpt(
            retrieved_lines,
            filename,
            start,
            end,
        )

        if excerpt is None:
            print(
                "PROVENANCE REJECTED: "
                f"{filename}:{start}-{end} "
                "was not fully present in exact source lines retrieved "
                "during this topic (or exceeded 120 lines)."
            )
            continue

        provenance_valid_by_category.setdefault(category, []).append({
            "file": filename,
            "start": start,
            "end": end,
            "excerpt": excerpt,
            "extractor_claim": establishes,
        })

        # A ledger hit is valid only after this run has independently
        # re-retrieved the exact range and reproduced the same source hash.
        # Therefore cached semantic acceptance cannot make stale source pass.
        verdict = EVIDENCE_LEDGER.lookup(
            topic["id"],
            category,
            filename,
            start,
            end,
            excerpt,
        )

        if verdict is not None:
            print(
                "LEDGER HIT: "
                f"{filename}:{start}-{end} category={category}"
            )
        else:
            verdict = verify_evidence_semantics(
                model,
                tokenizer,
                topic,
                category,
                establishes,
                excerpt,
            )

        if not verdict["supports"]:
            if not verdict.get("ledger_hit"):
                EVIDENCE_LEDGER.record_decision(
                    topic["id"], category, filename, start, end, excerpt, verdict
                )
            label = "VERIFIER ERROR" if verdict.get("verifier_error") else "SEMANTIC REJECTED"
            print(
                f"{label}: "
                f"{filename}:{start}-{end} "
                f"category={category} "
                f"reason={verdict['reason']}"
            )
            continue

        clean_evidence.append({
            "file": filename,
            "start": start,
            "end": end,
            "category": category,
            # Never pass the extractor's potentially overbroad wording to
            # synthesis. Use the independent verifier's grounded statement.
            "establishes": verdict["establishes"],
        })

        if not verdict.get("ledger_hit"):
            EVIDENCE_LEDGER.accept(
                topic["id"],
                category,
                filename,
                start,
                end,
                excerpt,
                verdict["establishes"],
            )

    individually_covered = {item["category"] for item in clean_evidence}
    # Any still-missing required category may need a relationship proof.
    # Candidate construction is repository-generic and the existing semantic
    # bundle verifier remains authoritative, so no benchmark-specific category
    # allow-list is needed here.
    for category in topic["required_evidence"]:
        if category in individually_covered:
            continue
        candidates = generic_relationship_bundle_candidates(
            topic,
            category,
            retrieved_lines,
            provenance_valid_by_category,
        )
        if len(candidates) < 2:
            continue

        print(
            "GENERIC RELATIONSHIP BUNDLE: "
            f"category={category} "
            + ", ".join(
                f"{x['file']}:{x['start']}-{x['end']}[{x.get('origin', '-')}]"
                for x in candidates
            )
        )

        verdict = EVIDENCE_LEDGER.lookup_bundle(
            topic["id"], category, candidates
        )
        if verdict is not None:
            REPOSITORY_AGENT_METRICS["generic_relationship_bundle_cache_hits"] += 1
            print(
                "BUNDLE LEDGER HIT: "
                f"category={category} ranges={len(candidates)}"
            )
        else:
            REPOSITORY_AGENT_METRICS["generic_relationship_bundle_attempts"] += 1
            verdict = verify_evidence_bundle_semantics(
                model, tokenizer, topic, category, candidates
            )

        if not verdict["supports"]:
            if not verdict.get("bundle_ledger_hit"):
                EVIDENCE_LEDGER.record_bundle_decision(
                    topic["id"], category, candidates, verdict
                )
            label = (
                "BUNDLE VERIFIER ERROR"
                if verdict.get("verifier_error")
                else "BUNDLE SEMANTIC REJECTED"
            )
            print(
                f"{label}: category={category} reason={verdict['reason']}"
            )
            continue

        clean_evidence.append({
            "category": category,
            "establishes": verdict["establishes"],
            "sources": [
                {"file": x["file"], "start": x["start"], "end": x["end"]}
                for x in candidates
            ],
        })
        REPOSITORY_AGENT_METRICS["generic_relationship_bundle_accepts"] += 1
        if not verdict.get("bundle_ledger_hit"):
            EVIDENCE_LEDGER.accept_bundle(
                topic["id"], category, candidates, verdict["establishes"]
            )
            print(
                "BUNDLE SEMANTIC ACCEPTED: "
                f"category={category} ranges={len(candidates)}"
            )

    covered_categories = {
        item["category"]
        for item in clean_evidence
    }

    required_categories = set(topic["required_evidence"])
    missing_categories = sorted(
        required_categories - covered_categories
    )

    # The controller, not Qwen, owns the supported/insufficient decision.
    status = (
        "supported"
        if not missing_categories
        else "insufficient"
    )

    if model_status != status:
        print(
            "STATUS OVERRIDDEN: "
            f"model={model_status!r}, controller={status!r}"
        )

    if missing_categories:
        print(
            "CATEGORY COVERAGE MISSING: "
            + ", ".join(missing_categories)
        )

    uncertainties = obj.get("uncertainties", [])

    if not isinstance(uncertainties, list):
        uncertainties = [str(uncertainties)]

    return {
        "topic": topic["title"],
        "status": status,
        "summary": str(obj.get("summary", "")).strip(),
        "required_categories": list(topic["required_evidence"]),
        "covered_categories": sorted(covered_categories),
        "missing_categories": missing_categories,
        "evidence": clean_evidence,
        "uncertainties": [
            str(x).strip()
            for x in uncertainties
            if str(x).strip()
        ],
    }


def investigation_state_text(
    topic,
    successful_calls,
    executed_calls,
    retrieved_ranges,
):
    """
    Give the investigator compact controller-owned navigation state.

    This does NOT claim that a required evidence category is proven.
    Final category coverage is still decided only by the evidence extractor
    plus sanitize_evidence_package() provenance/category validation.

    During investigation, required categories remain retrieval objectives.
    """
    required = list(topic["required_evidence"])

    searched = []
    read_ranges = []

    for call in successful_calls:
        tool = call.get("tool")

        if tool == "search":
            pattern = str(call.get("pattern", "")).strip()
            if pattern and pattern not in searched:
                searched.append(pattern)

        elif tool == "read":
            try:
                filename = str(call["file"])
                start = int(call.get("start", 1))
                end = int(call.get("end", start + 199))
                read_ranges.append(f"{filename}:{start}-{end}")
            except (KeyError, TypeError, ValueError):
                pass

    lines = [
        "CONTROLLER NAVIGATION STATE:",
        "",
        "Required evidence objectives still to establish:",
    ]

    for category in required:
        lines.append(f"- {category}")

    lines.extend([
        "",
        f"Successful repository tools used: "
        f"{len(successful_calls)}/{MAX_SUCCESSFUL_TOOLS_PER_TOPIC}",
        f"Unique repository requests seen: {len(executed_calls)}",
        "",
        "Searches already completed:",
    ])

    if searched:
        lines.extend(f"- {pattern}" for pattern in searched[-12:])
    else:
        lines.append("- none")

    lines.append("")
    lines.append("Source ranges already read:")

    if read_ranges:
        lines.extend(f"- {item}" for item in read_ranges[-12:])
    else:
        lines.append("- none")

    lines.extend([
        "",
        "TOPIC-SPECIFIC NEXT-PATH GUIDANCE:",
        TOPIC_NAVIGATION.get(topic["id"], ""),
        "",
        "NAVIGATION RULES:",
        "- Do not repeat a previous request. After a duplicate rejection, "
        "change strategy immediately.",
        "- Search results are navigation clues, not final source evidence.",
        "- For training/inference launch, process_spawn requires the concrete "
        "process-creation implementation itself (for example literal fork/exec), "
        "not merely a call to a helper named LaunchReservedChildProcess. Generic "
        "concrete process evidence may establish process_spawn but never "
        "worker_specific_path or worker_to_spawn_linkage. Find the worker-specific "
        "path and retrieve a source-visible bridge/caller-callee handoff into launch "
        "infrastructure.",
        "- For admission topics, semantic capability computation or registry "
        "selection does not establish capacity_or_dispatch_gate. Trace the "
        "scheduler gate that controls whether work may proceed.",
        "- If a service invokes an injected claim/dispatch callback, trace "
        "where that callback is supplied before claiming its internal policy.",
        "- If you found a generic helper, search for its callers and read the "
        "worker-specific call site.",
        "- If you found a worker-specific command/dispatch path, follow its "
        "callee chain until the actual launch/spawn path is read.",
        "- Prefer following symbols discovered in source already read over "
        "starting unrelated broad searches.",
        "- V13 CALL-CHAIN RULE: when two endpoints are known but their bridge is "
        "missing, search the exact helper/callee symbol at the first endpoint, read "
        "its implementation, then search/read its next concrete caller/callee. Do "
        "not spend remaining calls re-reading either endpoint.",
        "- A helper that returns a capacity boolean is not itself proof that "
        "dispatch is gated. Read the caller and the branch that changes control flow.",
        "- For analysis launch, first establish whether the architecture actually "
        "creates a separate process; do not force generic spawn evidence onto an "
        "in-process analysis path.",
        "- Final support requires every required evidence category, so spend "
        "remaining repository calls on the weakest part of the chain.",
    ])

    return "\n".join(lines)


def extract_evidence_object_with_retry(
    model,
    tokenizer,
    evidence_messages,
    max_tokens=800,
):
    """Run category evidence extraction with one bounded JSON-format retry."""
    raw = run_generation(
        model,
        tokenizer,
        evidence_messages,
        max_tokens=max_tokens,
    )
    obj = extract_json_object(raw)

    if isinstance(obj, dict):
        return raw, obj, False

    retry_messages = evidence_messages + [
        {"role": "assistant", "content": raw},
        {
            "role": "user",
            "content": (
                "Your previous evidence response was not a complete valid JSON "
                "object. Return exactly one JSON object matching the requested "
                "evidence schema. Escape all quotes inside JSON strings. Preserve "
                "the same evidence judgment; do not add new repository facts. "
                "Do not add markdown or commentary."
            ),
        },
    ]
    retry_raw = run_generation(
        model,
        tokenizer,
        retry_messages,
        max_tokens=max_tokens,
    )
    retry_obj = extract_json_object(retry_raw)
    return retry_raw, retry_obj, True


def investigate_topic(model, tokenizer, topic):
    print()
    print("=" * 78)
    print(f"INVESTIGATION: {topic['title'].upper()}")
    print("=" * 78)
    print(topic["question"])
    print()

    executed_calls = set()
    successful_calls = []
    transcript_parts = []

    # Requested intervals remain useful for navigation diagnostics.
    retrieved_ranges = []

    # Authoritative provenance is the exact numbered source text actually
    # returned by successful read operations.
    retrieved_lines = {}

    # V15 keeps the deterministic navigation seeds. These searches are controller-owned:
    # they ensure every topic begins with the most useful known symbols instead
    # of spending early model turns rediscovering them. Search results remain
    # navigation clues only and cannot satisfy evidence provenance.
    bootstrap_parts = []
    for pattern in TOPIC_BOOTSTRAP_SEARCHES.get(topic["id"], ()):
        call = {"tool": "search", "pattern": pattern}
        key = normalize_call(call)
        if key in executed_calls:
            continue

        executed_calls.add(key)
        result = execute_tool(call)
        if not result.startswith("TOOL ERROR:"):
            successful_calls.append(call)

        print(f"----- {topic['id']} CONTROLLER SEED SEARCH -----")
        print(json.dumps(call, indent=2))
        print("RESULT:")
        print(result)

        part = (
            "CONTROLLER SEED REQUEST:\n"
            + json.dumps(call, indent=2)
            + "\n\nCONTROLLER SEED RESULT:\n"
            + result
        )
        bootstrap_parts.append(part)
        transcript_parts.append(part)

    # V19: deterministically retrieve recurrent structural bridge regions in
    # the launch topics. These reads enter the same provenance store as model
    # reads and are still subject to the independent semantic verifier.
    for filename, start, end in TOPIC_BOOTSTRAP_READS.get(topic["id"], ()):
        call = {"tool": "read", "file": filename, "start": start, "end": end}
        key = normalize_call(call)
        if key in executed_calls:
            continue
        executed_calls.add(key)
        result = execute_tool(call)
        if not result.startswith("TOOL ERROR:"):
            successful_calls.append(call)
            retrieved_ranges.append((filename, start, end))
            record_retrieved_lines(retrieved_lines, filename, result)

        print(f"----- {topic['id']} REPOSITORY AGENT CONTROLLER SEED READ -----")
        print(json.dumps(call, indent=2))
        print("RESULT:")
        print(result)
        part = (
            "CONTROLLER SEED REQUEST:\n"
            + json.dumps(call, indent=2)
            + "\n\nCONTROLLER SEED RESULT:\n"
            + result
        )
        bootstrap_parts.append(part)
        transcript_parts.append(part)

    bootstrap_text = "\n\n".join(bootstrap_parts)

    # Phase 2C: supplement legacy navigation with controller-owned generic
    # symbol/function/call relationships. These facts are navigation only;
    # final evidence must still be read through the repository tool, pass exact
    # line provenance, and pass the existing semantic verifier/ledger rules.
    if GENERIC_INDEX_NAVIGATION_ENABLED:
        index_navigation = topic_index_navigation(topic)
        REPOSITORY_AGENT_METRICS["generic_index_navigation_topics"] += 1
    else:
        index_navigation = "Generic repository index navigation disabled by EA_GENERIC_INDEX_NAVIGATION."
    print(f"----- {topic['id']} GENERIC INDEX NAVIGATION -----")
    print(index_navigation)

    messages = [
        {
            "role": "system",
            "content": INVESTIGATION_SYSTEM,
        },
        {
            "role": "user",
            "content":
                f"ASSIGNED TOPIC: {topic['title']}\n\n"
                f"QUESTION:\n{topic['question']}\n\n"
                f"SEARCH HINTS:\n{topic['hints']}\n\n"
                f"CONTROLLER NAVIGATION PLAN:\n"
                f"{TOPIC_NAVIGATION.get(topic['id'], '')}\n\n"
                f"GENERIC REPOSITORY INDEX NAVIGATION:\n"
                f"{index_navigation}\n\n"
                "REQUIRED EVIDENCE CATEGORIES:\n- "
                + "\n- ".join(topic["required_evidence"])
                + "\n\nCONTROLLER SEED SEARCH RESULTS:\n"
                + (bootstrap_text or "none")
                + "\n\nUse these seed results to follow concrete symbols and call sites. "
                  "Do not repeat a seed search. You must try to establish ALL "
                  "required categories. Begin with exactly one NEW JSON tool request."
        },
    ]

    for protocol_turn in range(1, MAX_PROTOCOL_TURNS_PER_TOPIC + 1):
        if len(successful_calls) >= MAX_SUCCESSFUL_TOOLS_PER_TOPIC:
            break

        step = len(successful_calls) + 1
        response = run_generation(
            model,
            tokenizer,
            messages,
            max_tokens=420,
        )

        print(f"----- {topic['id']} TOOL STEP {step} -----")
        print(response)

        if response.startswith("FINAL:"):
            notice = (
                "HOST REJECTED: FINAL is not permitted during topic "
                "investigation. Continue with one JSON repository tool "
                "request."
            )
            print(notice)

            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": notice})
            continue

        call = extract_tool_call(response)

        if call is None:
            notice = (
                "HOST REJECTED: output was not one valid JSON tool request. "
                "Continue the assigned topic using exactly one JSON tool "
                "request."
            )
            print(notice)

            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": notice})
            continue

        key = normalize_call(call)

        print("REQUEST:")
        print(json.dumps(call, indent=2))

        if key in executed_calls:
            result = (
                "TOOL NOTICE: this exact request was already executed in "
                "this topic, including controller seed searches. You MUST "
                "change strategy now and request a different search or source range."
            )
        else:
            executed_calls.add(key)
            result = execute_tool(call)

            if not result.startswith("TOOL ERROR:"):
                successful_calls.append(call)

            if call.get("tool") == "read" and not result.startswith("TOOL ERROR:"):
                try:
                    retrieved_file = str(call["file"])
                    retrieved_start = int(call.get("start", 1))
                    retrieved_end = int(call.get("end", retrieved_start + 199))
                    retrieved_end = min(
                        retrieved_end,
                        retrieved_start + MAX_READ_LINES - 1,
                    )
                    retrieved_ranges.append(
                        (retrieved_file, retrieved_start, retrieved_end)
                    )
                    record_retrieved_lines(
                        retrieved_lines,
                        retrieved_file,
                        result,
                    )
                except (KeyError, TypeError, ValueError):
                    pass

        print("RESULT:")
        print(result)

        transcript_parts.append(
            "MODEL TOOL REQUEST:\n"
            + json.dumps(call, indent=2)
            + "\n\nTOOL RESULT:\n"
            + result
        )

        messages.append({"role": "assistant", "content": response})
        messages.append({
            "role": "user",
            "content":
                "TOOL RESULT:\n"
                + result
                + "\n\n"
                + investigation_state_text(
                    topic,
                    successful_calls,
                    executed_calls,
                    retrieved_ranges,
                )
                + "\n\nContinue investigating ONLY the assigned topic. "
                  "Follow concrete symbols/callers from source already found. "
                  "Advance the weakest required evidence objective and request "
                  "exactly one NEW JSON tool request."
        })

    # V15 bounded recovery pass: the ordinary investigator is good at finding
    # endpoints but can stop before retrieving the source-visible bridge.  This
    # second pass reuses the SAME accumulated transcript/provenance store and is
    # explicitly restricted to completing the controller's required categories.
    # It therefore accumulates evidence rather than replacing the first pass.
    recovery_messages = [
        {"role": "system", "content": INVESTIGATION_SYSTEM},
        {
            "role": "user",
            "content":
                f"ASSIGNED TOPIC: {topic['title']}\n\n"
                "This is the REPOSITORY AGENT CONTROLLER RECOVERY PASS. The first navigation "
                "pass is complete. Do not restart broad exploration. Use the "
                "accumulated source/navigation record below to retrieve only the "
                "missing bridge/control-flow ranges needed to make every required "
                "category directly verifiable.\n\n"
                "REQUIRED CATEGORIES:\n- "
                + "\n- ".join(topic["required_evidence"])
                + "\n\nTOPIC-SPECIFIC CALL-CHAIN PLAN:\n"
                + TOPIC_NAVIGATION.get(topic["id"], "")
                + "\n\nACCUMULATED NAVIGATION STATE:\n"
                + investigation_state_text(
                    topic, successful_calls, executed_calls, retrieved_ranges
                )
                + "\n\nFIRST-PASS TRANSCRIPT:\n"
                + "\n\n".join(transcript_parts)
                + "\n\nRecovery rules: follow exact symbols already discovered; prefer "
                  "caller/callee handoff ranges and branches that act on returned "
                  "values. For launch linkage, trace the same argv/request through "
                  "successive helpers toward fork/exec. For admission, retrieve both "
                  "role-specific selection/claim and the branch that gates dispatch. "
                  "For analysis admission, do not stop at ClaimCheckpointAnalysis's "
                  "signature: retrieve the capacity-return branch and the pending-analyze "
                  "load/empty-check/front-selection/reservation/returned-claim body. "
                  "For preemption, retrieve both priority/victim rule and executable "
                  "preemption path. Output exactly one NEW JSON repository request."
        },
    ]

    recovery_successes = 0
    for recovery_turn in range(1, MAX_RECOVERY_TURNS_PER_TOPIC + 1):
        if recovery_successes >= MAX_RECOVERY_TOOLS_PER_TOPIC:
            break

        response = run_generation(
            model, tokenizer, recovery_messages, max_tokens=420
        )
        print(f"----- {topic['id']} REPOSITORY AGENT RECOVERY STEP {recovery_turn} -----")
        print(response)

        call = extract_tool_call(response)
        if call is None:
            notice = (
                "HOST REJECTED: recovery output was not one valid JSON tool "
                "request. Return exactly one NEW JSON repository tool request."
            )
            print(notice)
            recovery_messages.append({"role": "assistant", "content": response})
            recovery_messages.append({"role": "user", "content": notice})
            continue

        key = normalize_call(call)
        print("REQUEST:")
        print(json.dumps(call, indent=2))

        if key in executed_calls:
            result = (
                "TOOL NOTICE: this exact request was already executed. Recovery "
                "must advance the call chain with a different search or read range."
            )
        else:
            executed_calls.add(key)
            result = execute_tool(call)
            if not result.startswith("TOOL ERROR:"):
                successful_calls.append(call)
                recovery_successes += 1

            if call.get("tool") == "read" and not result.startswith("TOOL ERROR:"):
                try:
                    retrieved_file = str(call["file"])
                    retrieved_start = int(call.get("start", 1))
                    retrieved_end = int(call.get("end", retrieved_start + 199))
                    retrieved_end = min(
                        retrieved_end, retrieved_start + MAX_READ_LINES - 1
                    )
                    retrieved_ranges.append(
                        (retrieved_file, retrieved_start, retrieved_end)
                    )
                    record_retrieved_lines(
                        retrieved_lines, retrieved_file, result
                    )
                except (KeyError, TypeError, ValueError):
                    pass

        print("RESULT:")
        print(result)
        part = (
            "V15 RECOVERY TOOL REQUEST:\n"
            + json.dumps(call, indent=2)
            + "\n\nTOOL RESULT:\n"
            + result
        )
        transcript_parts.append(part)

        recovery_messages.append({"role": "assistant", "content": response})
        recovery_messages.append({
            "role": "user",
            "content":
                "TOOL RESULT:\n" + result + "\n\n"
                + investigation_state_text(
                    topic, successful_calls, executed_calls, retrieved_ranges
                )
                + "\n\nContinue the bounded recovery pass. Advance a concrete "
                  "caller/callee, returned-value, dispatch-gate, or signal-control "
                  "chain. Request exactly one NEW JSON repository tool request."
        })

    transcript = "\n\n".join(transcript_parts)

    # V10 extracts each required category independently. This prevents a strong
    # evidence chain for one category from biasing the extractor into declaring
    # a weaker category supported.
    category_evidence = []
    category_summaries = []
    category_uncertainties = []

    for required_category in topic["required_evidence"]:
        evidence_messages = [
            {"role": "system", "content": EVIDENCE_SYSTEM},
            {
                "role": "user",
                "content":
                    f"TOPIC: {topic['title']}\n\n"
                    f"QUESTION:\n{topic['question']}\n\n"
                    f"EXTRACT ONLY THIS REQUIRED CATEGORY: {required_category}\n\n"
                    f"CATEGORY DEFINITION:\n"
                    f"{CATEGORY_DEFINITIONS[required_category]}\n\n"
                    f"TOPIC-SPECIFIC EXTRACTION GUIDANCE:\n"
                    f"{TOPIC_NAVIGATION.get(topic['id'], '')}\n\n"
                    "Return evidence only for this one category. Prefer the narrowest COMPLETE "
                    "evidence. For training/inference worker_specific_path or "
                    "worker_to_spawn_linkage, prefer the contiguous final-dispatch "
                    "prepareReservedLaunch/launchPreparedWorker range when retrieved: "
                    "it can show the role-specific Build*Command branch, the shared "
                    "preparedCommand handoff, and LaunchReservedChildProcess in one "
                    "visible control-flow chain. When a category is a call-chain/linkage objective, "
                    "return MULTIPLE narrow retrieved ranges when necessary to show "
                    "successive caller/callee or argument handoffs; do not force the "
                    "entire chain into one oversized range. For a single-range fact, "
                    "show the relevant branch, returned-value use, caller/callee "
                    "handoff, or control-flow consequence; do not "
                    "reduce evidence to a one-line function name when surrounding "
                    "retrieved lines establish its semantics. If the transcript "
                    "does not directly establish it, return status insufficient "
                    "with an empty evidence array. Do not borrow support from a "
                    "different category.\n\n"
                    + investigation_state_text(
                        topic,
                        successful_calls,
                        executed_calls,
                        retrieved_ranges,
                    )
                    + "\n\nINVESTIGATION TRANSCRIPT:\n"
                    + transcript
                    + "\n\nReturn the structured evidence package now."
            },
        ]

        evidence_response, evidence_object, evidence_retried = (
            extract_evidence_object_with_retry(
                model,
                tokenizer,
                evidence_messages,
                max_tokens=800,
            )
        )

        print()
        print(
            f"----- {topic['id']} EVIDENCE EXTRACTION "
            f"[{required_category}] -----"
        )
        if evidence_retried:
            print("EVIDENCE JSON RETRY: initial extraction was malformed; showing retry.")
        print(evidence_response)

        if not isinstance(evidence_object, dict):
            print(
                "EVIDENCE JSON PARSE FAILED: no complete JSON object decoded "
                "after one retry."
            )
            category_uncertainties.append(
                f"Evidence extraction failed for {required_category}."
            )
            continue

        raw_evidence = evidence_object.get("evidence", [])
        if isinstance(raw_evidence, list):
            for item in raw_evidence:
                if not isinstance(item, dict):
                    continue
                item = dict(item)
                # Category is controller-owned in V9. The extractor cannot
                # relabel evidence into another objective.
                item["category"] = required_category
                category_evidence.append(item)

        summary = str(evidence_object.get("summary", "")).strip()
        if summary:
            category_summaries.append(
                f"{required_category}: {summary}"
            )

        uncertainties = evidence_object.get("uncertainties", [])
        if not isinstance(uncertainties, list):
            uncertainties = [str(uncertainties)]
        category_uncertainties.extend(
            str(x).strip() for x in uncertainties if str(x).strip()
        )

    combined_evidence_object = {
        "status": "supported",  # untrusted; sanitizer recomputes mechanically
        "summary": " | ".join(category_summaries),
        "evidence": category_evidence,
        "uncertainties": category_uncertainties,
    }

    package = sanitize_evidence_package(
        topic,
        combined_evidence_object,
        retrieved_lines,
        model,
        tokenizer,
    )

    print()
    print("----- CONTROLLER EVIDENCE PACKAGE -----")
    print(json.dumps(package, indent=2))

    return package


def clean_final_answer(text):
    lines = text.splitlines()
    cleaned = []

    for line in lines:
        stripped = line.strip()

        if cleaned and stripped and stripped == cleaned[-1].strip():
            continue

        cleaned.append(line)

    return "\n".join(cleaned).strip()


def _source_ref(item):
    sources = item.get("sources")
    if isinstance(sources, list) and sources:
        refs = []
        for source in sources:
            file_name = str(source.get("file", "")).strip()
            start = source.get("start")
            end = source.get("end")
            refs.append(
                f"{file_name}:{start}" if start == end
                else f"{file_name}:{start}-{end}"
            )
        return "; ".join(refs)

    file_name = str(item.get("file", "")).strip()
    start = item.get("start")
    end = item.get("end")
    if start == end:
        return f"{file_name}:{start}"
    return f"{file_name}:{start}-{end}"


def _category_label(category):
    return str(category).replace("_", " ")


def synthesize(model, tokenizer, packages):
    """Render only controller-approved evidence; no model synthesis is allowed.

    V14 retains V11's removal of the final free-form generation step. This makes
    controller status authoritative by construction and prevents evidence from
    one topic from being borrowed to fill another topic's missing categories.
    """
    del model, tokenizer

    print()
    print("=" * 78)
    print("SYNTHESIS")
    print("=" * 78)

    lines = ["FINAL:"]

    for index, package in enumerate(packages, start=1):
        topic = str(package.get("topic", f"topic {index}"))
        status = str(package.get("status", "insufficient"))
        evidence = list(package.get("evidence", []))
        missing = list(package.get("missing_categories", []))

        lines.append("")
        lines.append(f"{index}. {topic.upper()}")
        lines.append(f"   Controller status: {status}")

        if evidence:
            lines.append("   Confirmed evidence:")
            for item in evidence:
                category = _category_label(item.get("category", "evidence"))
                establishes = str(item.get("establishes", "")).strip()
                lines.append(
                    f"   - [{category}] {establishes} "
                    f"({_source_ref(item)})"
                )
        else:
            lines.append("   Confirmed evidence: none")

        if missing:
            lines.append(
                "   Not established: "
                + ", ".join(_category_label(category) for category in missing)
            )

        if status == "insufficient":
            lines.append(
                "   Conclusion: the requested architecture point is not fully "
                "established by controller-approved evidence."
            )

    response = "\n".join(lines)

    print()
    print("===== FINAL SYNTHESIS =====")
    print(response)

    return response

def main():
    print(f"Loading {MODEL} ...")
    model, tokenizer = load(MODEL)

    print("Model loaded.")
    print()
    print(
        "RepositoryAgent: generic index navigation, exact provenance, verified evidence ledger, and generic relationship bundles."
    )
    print(
        "RepositoryAgent flags: "
        f"EA_GENERIC_INDEX_NAVIGATION={int(GENERIC_INDEX_NAVIGATION_ENABLED)}"
    )
    print(
        f"Per-topic repository budget: "
        f"{MAX_SUCCESSFUL_TOOLS_PER_TOPIC} successful repository tools "
        f"(including controller seed searches) within "
        f"{MAX_PROTOCOL_TURNS_PER_TOPIC} model protocol turns."
    )

    packages = []

    for topic in TOPICS:
        package = investigate_topic(
            model,
            tokenizer,
            topic,
        )
        packages.append(package)

    print()
    print("=" * 78)
    print("COVERAGE SUMMARY")
    print("=" * 78)

    for package in packages:
        covered = ",".join(package["covered_categories"]) or "-"
        missing = ",".join(package["missing_categories"]) or "-"

        print(
            f"{package['topic']}: "
            f"{package['status']} "
            f"({len(package['evidence'])} evidence ranges; "
            f"covered={covered}; missing={missing})"
        )

    print()
    print("=" * 78)
    print("REPOSITORY AGENT INSTRUMENTATION")
    print("=" * 78)
    print(json.dumps(REPOSITORY_AGENT_METRICS, sort_keys=True))

    synthesize(
        model,
        tokenizer,
        packages,
    )


if __name__ == "__main__":
    main()
