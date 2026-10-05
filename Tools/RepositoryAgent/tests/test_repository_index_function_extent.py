#!/usr/bin/env python3
"""Generic regression tests for RepositoryIndex function extents and call edges."""

from Tools.RepositoryAgent import repository_index as ri

SOURCE = """\
void previous()
{
    helper_old();
}

std::tuple<int, int>
Demo::next_function(
    int value)
{
    helper(value);
    consumer(value);
    return {value, value};
}

int Demo::recursive(int value)
{
    if (value > 0)
        return recursive(value - 1);
    return value;
}

int RunSchedulerOnce()
{
    return 0;
}

int RunCheckpointEvalAnalyzeJobs()
{
    return 0;
}

int RunScheduler()
{
    SchedulerOperations operations;
    operations.runCycle = [&] { return RunSchedulerOnce(); };
    operations.runCheckpointAnalysis = [&] {
        return RunCheckpointEvalAnalyzeJobs();
    };
    return 0;
}

int SchedulerCycleService::runOnce()
{
    return operations_.runCheckpointAnalysis();
}

int RunProductionSchedulerDaemon()
{
    return RunScheduler();
}
"""

def main():
    old_list_files = ri.list_files
    old_read_file = ri.read_file
    try:
        lines = SOURCE.splitlines()

        def fake_list_files(prefix=""):
            return ["Sources/Synthetic.cpp"]

        def fake_read_file(name, start=1, end=200):
            selected = lines[start - 1:end]
            return "\n".join(
                f"{line_no}: {text}"
                for line_no, text in enumerate(selected, start)
            )

        ri.list_files = fake_list_files
        ri.read_file = fake_read_file
        idx = ri.RepositoryIndex().build(prefixes=("Sources",))

        fn = idx.function_for_line("Sources/Synthetic.cpp", 11)
        assert fn is not None, "interior line must belong to next_function"
        assert fn.name == "Demo::next_function", fn
        assert fn.start_line == 5, fn
        assert fn.end_line == 13, fn

        assert idx.relationship("Demo::next_function", "next_function") == []

        helper_edges = idx.relationship("Demo::next_function", "helper")
        assert [(x.line, x.caller, x.callee) for x in helper_edges] == [
            (10, "Demo::next_function", "helper")
        ], helper_edges

        consumer_edges = idx.relationship("Demo::next_function", "consumer")
        assert [(x.line, x.caller, x.callee) for x in consumer_edges] == [
            (11, "Demo::next_function", "consumer")
        ], consumer_edges

        trace = idx.trace_calls("Demo::next_function", max_depth=1)
        callees = [x["callee"] for x in trace]
        assert "helper" in callees and "consumer" in callees, trace
        assert "Demo::next_function" not in callees and "next_function" not in callees, trace

        # Suppressing a definition signature must not suppress genuine recursion.
        recursive_edges = idx.relationship("Demo::recursive", "recursive")
        assert [(x.line, x.caller, x.callee) for x in recursive_edges] == [
            (18, "Demo::recursive", "recursive")
        ], recursive_edges

        # The real SchedulerCore operation-field pattern is indexed as an
        # operation binding, never as a direct lexical-owner relationship.
        callback_edges = idx.callees_of("RunScheduler")
        assert [(x.callee, x.relationship_kind) for x in callback_edges] == [
            ("RunSchedulerOnce", "operation_binding"),
            ("RunCheckpointEvalAnalyzeJobs", "operation_binding"),
        ], callback_edges
        assert idx.relationship("RunScheduler", "RunSchedulerOnce") == []
        assert idx.relationship("RunScheduler", "RunCheckpointEvalAnalyzeJobs") == []
        operation_invocations = idx.callees_of("SchedulerCycleService::runOnce")
        assert [(x.callee, x.relationship_kind, x.operation) for x in operation_invocations] == [
            ("operations_.runCheckpointAnalysis", "operation_invocation", "operations_.runCheckpointAnalysis")
        ], operation_invocations
        direct_scheduler_edge = idx.relationship(
            "RunProductionSchedulerDaemon", "RunScheduler"
        )
        assert [(x.callee, x.relationship_kind) for x in direct_scheduler_edge] == [
            ("RunScheduler", "direct_invocation")
        ], direct_scheduler_edge

        print("REPOSITORY INDEX FUNCTION EXTENT TEST: PASS")
    finally:
        ri.list_files = old_list_files
        ri.read_file = old_read_file

if __name__ == "__main__":
    main()
