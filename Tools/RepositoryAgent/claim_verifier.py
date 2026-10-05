#!/usr/bin/env python3
"""Independent claim verifier for Codex-directed RepositoryAgent evidence.

Importing this module does not load model weights.  LazyClaimVerifierRuntime
loads Qwen only on the first verification request and reuses it thereafter.
"""
from __future__ import annotations

import contextlib
import sys

from .verifier import extract_json_object, run_generation

MODEL = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"
MAX_BUNDLE_RANGES = 8

CLAIM_VERIFY_SYSTEM = r"""
You are an independent source-claim verifier.

Judge only whether the exact supplied source excerpt or explicitly supplied
multi-range source bundle directly establishes the proposed claim.

Rules:
- Use only the supplied source. Do not use outside knowledge.
- Do not infer missing call-chain, data-flow, control-flow, aliasing, or runtime links.
- Directly visible syntactic relationships are evidence, not missing-link inference. For example, if the supplied source visibly assigns a call result to a local variable and visibly passes that same local variable to later calls, that data-flow handoff is directly established by the source.
- Do not assume two differently named expressions, aliases, objects, calls, or values are identical unless the supplied source directly establishes that relationship.
- Symbol names, filenames, comments, logs, and structural-index metadata alone do not prove semantics.
- A multi-range bundle may establish a connected path only when the visible source establishes each required handoff.
- Judge the proposed claim as written. If it is broader than the source, reject it rather than silently weakening it.
- If supported, return a narrow neutral statement of what the source establishes.

Return exactly one JSON object and no markdown:
{
  "supports": true | false,
  "establishes": "precise source-grounded statement, or empty if unsupported",
  "reason": "brief explanation"
}
"""

# This is intentionally a separate mode, rather than a relaxation of the
# ordinary bundle verifier.  The controller supplies this only after it has
# selected and reread every direct indexed relationship evidence range.
RELATIONSHIP_CLAIM_VERIFY_SYSTEM = r"""
You are an independent source-claim verifier for an explicitly admitted set of
direct caller-to-callee relationships.

Judge only whether the exact supplied source ranges directly establish the
proposed claim and every listed direct relationship.

Rules:
- Use only the supplied exact source ranges. Structural-index metadata alone is
  not semantic evidence.
- Verify every listed caller->callee call independently from its associated
  exact source range(s).
- A directly syntactically visible call is sufficient to establish that one
  direct structural relationship.
- All listed relationships must be source-visible for the complete chain/set
  claim to be supported. Do not infer missing edges.
- Do not require an additional data-flow or control-flow handoff between
  separate direct call edges merely because they form a call chain.
- Do not infer that an unqualified callee identifies a particular qualified
  implementation unless the supplied relationship identity and source directly
  establish that identity.
- Judge the proposed claim as written. If it is broader than the exact source,
  reject it rather than silently weakening it.

Return exactly one JSON object and no markdown:
{
  "supports": true | false,
  "establishes": "precise source-grounded statement, or empty if unsupported",
  "reason": "brief explanation"
}
"""


def _topic_text(topic) -> str:
    if isinstance(topic, dict):
        return str(topic.get("title") or topic.get("id") or "").strip()
    return str(topic).strip()


def _finish_verdict(obj, invalid_reason: str, *, model_turns: int):
    if not isinstance(obj, dict):
        return {
            "supports": False,
            "establishes": "",
            "reason": invalid_reason,
            "verifier_error": True,
            "model_turns": model_turns,
        }
    supports = obj.get("supports") is True
    establishes = str(obj.get("establishes", "")).strip()
    reason = str(obj.get("reason", "")).strip()
    if supports and not establishes:
        return {
            "supports": False,
            "establishes": "",
            "reason": "claim verifier supplied no grounded statement",
            "model_turns": model_turns,
        }
    return {
        "supports": supports,
        "establishes": establishes if supports else "",
        "reason": reason,
        "model_turns": model_turns,
    }


def verify_source_claim_semantics(model, tokenizer, topic, claim, item):
    messages = [
        {"role": "system", "content": CLAIM_VERIFY_SYSTEM},
        {
            "role": "user",
            "content": (
                f"TOPIC: {_topic_text(topic)}\n\n"
                f"PROPOSED CLAIM:\n{str(claim).strip()}\n\n"
                f"EXACT SOURCE RANGE: {item['file']}:{int(item['start'])}-{int(item['end'])}\n"
                f"{item['excerpt']}\n\n"
                "Return the claim verification JSON now."
            ),
        },
    ]
    model_turns = 1
    raw = run_generation(model, tokenizer, messages, max_tokens=300)
    obj = extract_json_object(raw)
    if not isinstance(obj, dict):
        retry = messages + [
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": "Return exactly one complete valid JSON object matching the requested schema.",
            },
        ]
        model_turns += 1
        raw = run_generation(model, tokenizer, retry, max_tokens=300)
        obj = extract_json_object(raw)
    return _finish_verdict(
        obj,
        "claim verifier did not return valid JSON after one retry",
        model_turns=model_turns,
    )


def verify_source_bundle_claim_semantics(model, tokenizer, topic, claim, items):
    if not 2 <= len(items) <= MAX_BUNDLE_RANGES:
        return {
            "supports": False,
            "establishes": "",
            "reason": f"claim bundle requires 2-{MAX_BUNDLE_RANGES} source ranges",
            "model_turns": 0,
        }
    blocks = []
    for index, item in enumerate(items, start=1):
        blocks.append(
            f"RANGE {index}: {item['file']}:{int(item['start'])}-{int(item['end'])}\n"
            f"{item['excerpt']}"
        )
    messages = [
        {"role": "system", "content": CLAIM_VERIFY_SYSTEM},
        {
            "role": "user",
            "content": (
                f"TOPIC: {_topic_text(topic)}\n\n"
                f"PROPOSED CLAIM:\n{str(claim).strip()}\n\n"
                "EXACT MULTI-RANGE SOURCE BUNDLE:\n"
                + "\n\n".join(blocks)
                + "\n\nJudge this only as a source-visible connected path. Return the claim verification JSON now."
            ),
        },
    ]
    model_turns = 1
    raw = run_generation(model, tokenizer, messages, max_tokens=360)
    obj = extract_json_object(raw)
    if not isinstance(obj, dict):
        retry = messages + [
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": "Return exactly one complete valid JSON object matching the requested schema.",
            },
        ]
        model_turns += 1
        raw = run_generation(model, tokenizer, retry, max_tokens=360)
        obj = extract_json_object(raw)
    return _finish_verdict(
        obj,
        "claim bundle verifier did not return valid JSON after one retry",
        model_turns=model_turns,
    )


def verify_relationship_bundle_claim_semantics(model, tokenizer, topic, claim, relationships):
    """Verify controller-owned direct-edge/evidence associations.

    Unlike an ordinary bundle, one effective reread range may prove more than
    one admitted edge.  The associations are rendered by the server, never
    accepted from an MCP request.
    """
    if not relationships or len(relationships) > 5:
        return {"supports": False, "establishes": "", "reason": "relationship bundle unavailable", "model_turns": 0}
    blocks = []
    for index, relationship in enumerate(relationships, start=1):
        evidence = relationship.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            return {"supports": False, "establishes": "", "reason": "relationship evidence unavailable", "model_turns": 0}
        ranges = []
        for item in evidence:
            ranges.append(
                f"RANGE {item['file']}:{int(item['start'])}-{int(item['end'])}\n{item['excerpt']}"
            )
        blocks.append(
            f"RELATIONSHIP {index}: {relationship['caller']} -> {relationship['callee']}\n"
            "ASSOCIATED EXACT SOURCE RANGE(S):\n" + "\n\n".join(ranges)
        )
    messages = [
        {"role": "system", "content": RELATIONSHIP_CLAIM_VERIFY_SYSTEM},
        {"role": "user", "content": (
            f"TOPIC: {_topic_text(topic)}\n\nPROPOSED CLAIM:\n{str(claim).strip()}\n\n"
            "SERVER-SELECTED RELATIONSHIP EVIDENCE:\n" + "\n\n".join(blocks)
            + "\n\nReturn the claim verification JSON now."
        )},
    ]
    model_turns = 1
    raw = run_generation(model, tokenizer, messages, max_tokens=360)
    obj = extract_json_object(raw)
    if not isinstance(obj, dict):
        retry = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": "Return exactly one complete valid JSON object matching the requested schema."},
        ]
        model_turns += 1
        raw = run_generation(model, tokenizer, retry, max_tokens=360)
        obj = extract_json_object(raw)
    return _finish_verdict(obj, "relationship claim verifier did not return valid JSON after one retry", model_turns=model_turns)


class LazyClaimVerifierRuntime:
    """Lazy, reusable MLX runtime. stdout is never available to corrupt MCP framing."""

    def __init__(self, model_name=MODEL):
        self.model_name = str(model_name)
        self.model = None
        self.tokenizer = None

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def _ensure_loaded(self) -> None:
        if self.loaded:
            return
        # Import lazily so structural MCP operations do not initialize MLX model state.
        from mlx_lm import load
        with contextlib.redirect_stdout(sys.stderr):
            self.model, self.tokenizer = load(self.model_name)

    def verify_claim(self, topic, claim, item):
        self._ensure_loaded()
        # mlx_lm.generate may emit progress/diagnostics; MCP stdout must remain JSON-RPC only.
        with contextlib.redirect_stdout(sys.stderr):
            return verify_source_claim_semantics(
                self.model, self.tokenizer, topic, claim, item
            )

    def verify_bundle_claim(self, topic, claim, items):
        self._ensure_loaded()
        with contextlib.redirect_stdout(sys.stderr):
            return verify_source_bundle_claim_semantics(
                self.model, self.tokenizer, topic, claim, items
            )

    def verify_relationship_bundle_claim(self, topic, claim, relationships):
        self._ensure_loaded()
        with contextlib.redirect_stdout(sys.stderr):
            return verify_relationship_bundle_claim_semantics(
                self.model, self.tokenizer, topic, claim, relationships
            )
