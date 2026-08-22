#!/usr/bin/env python3
"""Static guards for the pre-Phase-2C checkpoint-policy boundary."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def function_body(path: Path, name: str) -> str:
    text = path.read_text(encoding="utf-8")
    start = text.find(f"{name}(")
    if start < 0:
        raise AssertionError(f"missing function {name}")
    brace = text.find("{", start)
    depth = 0
    for index in range(brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[brace : index + 1]
    raise AssertionError(f"unterminated function {name}")


scheduler = ROOT / "Sources" / "ExperimentScheduler.cpp"
policy = ROOT / "Sources" / "CheckpointPolicy.cpp"
scoring = ROOT / "Sources" / "ExperimentRecommendationScoring.cpp"
training = ROOT / "LSTM" / "main.cpp"

for path, names in [
    (scheduler, [
        "DecideCheckpointPolicy",
        "RankCheckpointPolicyEval",
        "LoadCheckpointPolicyRankPopulation",
        "ApplyCheckpointPolicyStopRequest",
    ]),
    (policy, [
        "CheckpointPolicyCanonicalText",
        "CheckpointPolicyEvidenceCanonicalText",
        "RequestedCheckpointPolicyStopEpoch",
        "DecideCheckpointPolicyPure",
    ]),
    (scoring, ["ScoreExperimentRecommendation", "RankRecommendationScores"]),
]:
    for name in names:
        if "profitability" in function_body(path, name).lower():
            raise AssertionError(f"{name} must not inspect profitability")

status = function_body(scheduler, "RunCheckpointPolicyStatusCommand")
if "SetTransactionReadOnly(w);" not in status:
    raise AssertionError("checkpoint policy status is not read-only")
for mutation in [
    "SetTransactionReadWrite",
    "PersistCheckpointPolicyDecision",
    "ApplyCheckpointPolicyStopRequest",
    "EvaluateCheckpointPolicyAfterAnalysis",
]:
    if mutation in status:
        raise AssertionError(f"checkpoint policy status invokes {mutation}")
if "checkpoint_policy_revision" not in status or \
        "CheckpointPolicySemanticHash" not in status:
    raise AssertionError("status does not expose revision/hash")

control = function_body(scheduler, "RunCheckpointControlCommand")
if "CheckpointPolicyCanonicalText" not in control or \
        "checkpoint_policy_revision" not in control:
    raise AssertionError("policy controls do not revision semantic changes")

automatic = function_body(scheduler, "EvaluateCheckpointPolicyAfterAnalysis")
manual = function_body(scheduler, "RunEvaluateCheckpointPolicyCommand")
if "DecideCheckpointPolicyWithIdentity" not in automatic:
    raise AssertionError("automatic path does not use hardened evaluator")
if "EvaluateCheckpointPolicyAfterAnalysis" not in manual:
    raise AssertionError("manual path does not share automatic evaluator")

rank_body = function_body(scheduler, "LoadCheckpointPolicyRankPopulation")
if "a.leader_score DESC NULLS LAST" not in rank_body or \
        "a.infer_accuracy DESC NULLS LAST" not in rank_body or \
        "ce.checkpoint_epoch DESC" not in rank_body or \
        "ce.checkpoint_eval_id ASC" not in rank_body:
    raise AssertionError("legacy checkpoint rank ordering changed")

if "checkpoint_policy_stop_decision_id=$2" not in function_body(
        scheduler, "ApplyCheckpointPolicyStopRequest"):
    raise AssertionError("stop application lacks durable decision attribution")

persist = function_body(scheduler, "PersistCheckpointPolicyDecision")
if "DO NOTHING" not in persist or "DO UPDATE" in persist:
    raise AssertionError("decision persistence is not append-only/idempotent")
for identity_field in [
    "policy_revision", "policy_hash", "evidence_watermark",
    "analysis_id", "inference_eval_result_id",
]:
    if identity_field not in persist:
        raise AssertionError(f"decision persistence omits {identity_field}")
if "checkpoint_epoch = $4 AND checkpoint_eval_id < $1" not in persist:
    raise AssertionError("same-epoch older checkpoint decisions are not superseded")

stop = function_body(scheduler, "ApplyCheckpointPolicyStopRequest")
if "newer.checkpoint_epoch>current_decision.checkpoint_epoch" not in stop or \
        "newer.checkpoint_eval_id>current_decision.checkpoint_eval_id" not in stop:
    raise AssertionError("stop fence does not use epoch/eval checkpoint authority")

status_order = (
    "checkpoint_epoch DESC, \"\n"
    "        \"checkpoint_eval_id DESC, checkpoint_decision_id DESC"
)
if status_order not in status:
    raise AssertionError("status authority does not use epoch/eval ordering")

if "profitability" in policy.read_text(encoding="utf-8").lower():
    raise AssertionError("checkpoint semantic/evidence module mentions profitability")

# Training remains a consumer of stop_after_checkpoint_epoch only; it must not
# know the checkpoint-policy decision schema or profitability policy.
training_text = training.read_text(encoding="utf-8").lower()
if "experiment_checkpoint_decision" in training_text:
    raise AssertionError("training was coupled to checkpoint policy decisions")

print("CheckpointPolicyIsolationTests passed")
