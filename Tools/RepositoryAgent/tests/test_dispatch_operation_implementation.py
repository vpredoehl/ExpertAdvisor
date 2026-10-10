"""Regression tests for calls inside explicitly assigned operation callbacks."""

from Tools.RepositoryAgent.repository_index import RepositoryIndex


SOURCE_FILE = "Sources/SchedulerCore/ProductionSchedulerDaemon.cpp"
DISPATCH_FILE = "Sources/SchedulerCore/FinalExperimentDispatchService.cpp"

EXPECTED = {
    ("operations.reserveWorkerAttempt", "ReserveExperimentWorkerAttempt"),
    ("operations.prepareReservedLaunch", "BuildTrainCommand"),
    ("operations.prepareReservedLaunch", "BuildInferCommand"),
    ("operations.prepareReservedLaunch", "BuildAnalyzeCommand"),
    ("operations.launchPreparedWorker", "LaunchReservedChildProcess"),
}


def main():
    index = RepositoryIndex().build(prefixes=("Sources/SchedulerCore",))

    implementation_edges = [
        edge
        for edge in index.callees_of("RunFinalExperimentPhase")
        if edge.relationship_kind == "operation_implementation_call"
    ]

    actual = {
        (edge.operation, edge.callee)
        for edge in implementation_edges
    }

    missing = EXPECTED - actual

    assert not missing, (
        f"Missing callback implementation relationships: {sorted(missing)}"
    )

    for edge in implementation_edges:
        assert edge.file == SOURCE_FILE
        assert edge.caller == "RunFinalExperimentPhase"
    assert len({(edge.file, edge.line, edge.callee) for edge in implementation_edges}) == len(implementation_edges)

    invocations = [
        edge
        for edge in index.callees_of(
            "FinalExperimentDispatchService::runPhase"
        )
        if edge.relationship_kind == "operation_invocation"
    ]

    invoked_fields = {edge.operation for edge in invocations}

    for operation in (
        "operations_.reserveWorkerAttempt",
        "operations_.prepareReservedLaunch",
        "operations_.launchPreparedWorker",
    ):
        assert operation in invoked_fields, operation
    assert all(edge.file == DISPATCH_FILE for edge in invocations)

    # Callback implementation calls must not become direct invocations.
    for operation, callee in EXPECTED:
        assert not index.relationship("RunFinalExperimentPhase", callee), (operation, callee)
    trace = index.trace_calls("RunFinalExperimentPhase", max_depth=1, max_nodes=1000)
    assert not ({row["callee"] for row in trace} & {callee for _, callee in EXPECTED}), trace

    print("test_dispatch_operation_implementation: PASS")


if __name__ == "__main__":
    main()
