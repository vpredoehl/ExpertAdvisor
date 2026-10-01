#!/usr/bin/env python3
from Tools.RepositoryAgent.repository_generalization_benchmark import BENCHMARK

assert BENCHMARK["name"] == "repository_generalization"
assert len(BENCHMARK["topics"]) == 6
assert BENCHMARK["topic_bootstrap_searches"] == {}
assert BENCHMARK["topic_bootstrap_reads"] == {}

for topic in BENCHMARK["topics"]:
    assert topic["id"]
    assert topic["question"]
    assert topic["hints"]
    assert len(topic["required_evidence"]) >= 2
    for category in topic["required_evidence"]:
        assert category in BENCHMARK["category_definitions"]

# The baseline must not be seeded with known implementation locations/symbols.
forbidden = (
    "SchedulerCore", "ProductionSchedulerDaemon", "GlobalExperimentControl",
    "LSTM.cpp:", "main.cpp:", "Sources/", "Headers/",
)
policy_text = "\n".join(
    [topic["question"] + "\n" + topic["hints"] for topic in BENCHMARK["topics"]]
    + list(BENCHMARK["topic_navigation"].values())
)
for token in forbidden:
    assert token not in policy_text, f"generalization benchmark leaks implementation seed: {token}"

print("PHASE 5B GENERALIZATION BENCHMARK STRUCTURAL TEST: PASS")
