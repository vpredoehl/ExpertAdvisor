#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERIC = ("repository_agent.py", "retrieval.py", "verifier.py")
FORBIDDEN_IMPORT = "scheduler_architecture_benchmark"
FORBIDDEN_TOPIC_IDS = (
    "training_admission", "training_launch", "inference_admission", "inference_launch",
    "analysis_admission", "analysis_launch", "priority_preemption",
)

for name in GENERIC:
    text = (ROOT / name).read_text()
    assert FORBIDDEN_IMPORT not in text, f"{name} directly depends on scheduler benchmark"
    for topic_id in FORBIDDEN_TOPIC_IDS:
        assert topic_id not in text, f"{name} embeds scheduler topic {topic_id}"

scheduler = (ROOT / "scheduler_architecture_benchmark.py").read_text()
for topic_id in FORBIDDEN_TOPIC_IDS:
    assert topic_id in scheduler
for required in (
    '"topics": TOPICS', '"topic_navigation": TOPIC_NAVIGATION',
    '"category_definitions": CATEGORY_DEFINITIONS',
    '"investigation_system": INVESTIGATION_SYSTEM',
    '"evidence_system": EVIDENCE_SYSTEM', '"synthesis_system": SYNTHESIS_SYSTEM',
):
    assert required in scheduler

print("PHASE 5A BENCHMARK INJECTION STRUCTURAL TEST: PASS")
