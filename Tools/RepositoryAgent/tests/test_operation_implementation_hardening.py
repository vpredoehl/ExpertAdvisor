"""Focused synthetic regressions for explicit operation implementations."""

from dataclasses import asdict
import unittest
from unittest.mock import patch

from Tools.RepositoryAgent import repository_index as ri
from Tools.RepositoryAgent.codex_interface import CodexRepositoryInterface


FILE = "Sources/SchedulerCore/ImplementationFixture.cpp"
KIND = "operation_implementation_call"


def build(source):
    lines = source.splitlines()
    with patch.object(ri, "list_files", return_value=[FILE]), patch.object(
        ri, "read_file", side_effect=lambda name, start=1, end=200: "\n".join(
            f"{number}: {line}" for number, line in
            enumerate(lines[start - 1:end], start)
        )
    ):
        return ri.RepositoryIndex().build(prefixes=("Sources/SchedulerCore",))


def fixture(body, receiver="operations", field="prepareReservedLaunch"):
    return ("void Owner()\n{\n"
            f"    {receiver}.{field} = [&](int mode) {{\n"
            f"{body}\n    }};\n}}\n")


def implementations(index):
    return [edge for edges in index.calls_by_caller.values() for edge in edges
            if edge.relationship_kind == KIND]


class MetadataInterface(CodexRepositoryInterface):
    # Discovery needs only metadata: no runtime or claim ledger is touched.
    def __init__(self, index):
        self.index = index

    def _idx(self):
        return self.index


class OperationImplementationHardening(unittest.TestCase):
    def assert_visible_only(self, body):
        index = build(fixture(body + "\n    RealOne();\n    RealTwo();"))
        edges = implementations(index)
        self.assertEqual({edge.callee for edge in edges}, {"RealOne", "RealTwo"})
        self.assertEqual(len(edges), 2)
        for edge in edges:
            self.assertEqual(edge.operation, "operations.prepareReservedLaunch")
            self.assertEqual(edge.caller, "Owner")
            self.assertEqual(edge.file, FILE)
        return index

    def test_line_comments(self):
        self.assert_visible_only("    // FakeLine(); } {[&] { Hidden(); };\n"
                                 "    // FakeContinued(); \\\n"
                                 "    FakeContinuation();")

    def test_block_and_multiline_comments(self):
        self.assert_visible_only("    /* FakeBlock(); } {\n"
                                 "       FakeMultiline(); */")

    def test_ordinary_strings_and_characters(self):
        self.assert_visible_only('    const char* text = "FakeString(); } {";\n'
                                 "    char left = '{', right = '}';\n"
                                 "    int multi = 'FakeCharacter()';")

    def test_raw_strings_and_custom_delimiters(self):
        for prefix in ("", "u8", "u", "U", "L"):
            with self.subTest(prefix=prefix):
                self.assert_visible_only(
                    f'    auto text = {prefix}R"custom(FakeRaw(); " }} {{\n'
                    '        FakeMultilineRaw(); )other\" )custom";\n'
                    '    auto simple = R"(FakeSimpleRaw(); } {)";'
                )

    def test_escaped_quotes(self):
        self.assert_visible_only(r'''    auto text = "escaped \" FakeEscaped(); } { \\";
    char quote = '\'';
    char backslash = '\\';''')

    def test_braces_and_fake_assignments_are_masked(self):
        source = fixture('    /* } ; */\n'
                         '    auto text = R"tag(} ; { FakeRaw(); )tag";\n'
                         '    RealOne();\n    RealTwo();')
        source += '\n// operations.fake = [&] { FakeOne(); FakeTwo(); };\n'
        source += '/* operations.fake = [&] { FakeThree(); FakeFour(); }; */\n'
        source += 'auto text = "operations.fake = [&] { FakeFive(); FakeSix(); };";\n'
        source += 'auto raw = R"(operations.fake = [&] { FakeSeven(); FakeEight(); };)";\n'
        edges = implementations(build(source))
        self.assertEqual({edge.callee for edge in edges}, {"RealOne", "RealTwo"})
        masked = ri.RepositoryIndex._mask_operation_source(source)
        self.assertEqual(len(masked), len(source))
        self.assertEqual([n for n, ch in enumerate(masked) if ch == "\n"],
                         [n for n, ch in enumerate(source) if ch == "\n"])

    def test_unbalanced_or_unterminated_assignment(self):
        for source in (
            "void Owner()\n{\noperations.prepareReservedLaunch = [&] {\nOne();\nTwo();\n",
            "void Owner()\n{\noperations.prepareReservedLaunch = [&] {\nOne();\nTwo();\n}\n",
            fixture("    One();\n    Two();").replace("    };", "    }"),
        ):
            with self.subTest(source=source):
                self.assertEqual(implementations(build(source)), [])

    def test_unterminated_or_unsupported_literals_fail_closed(self):
        for literal in ('"unterminated', "'unterminated", 'R"tag(unterminated',
                        'R"invalid delimiter(pretend)invalid delimiter"',
                        'R"abcdefghijklmnopq(too long)abcdefghijklmnopq"',
                        '"user-defined"_suffix', 'R"(user-defined)"_suffix',
                        "1'000; Later(); 2'000;", 'unknownPrefix"literal"',
                        "/* unterminated"):
            with self.subTest(literal=literal):
                # Even calls before the lexical failure are unsafe in this body.
                index = build(fixture("    BeforeOne();\n    BeforeTwo();\n    " +
                                      literal + "\n    AfterOne();\n    AfterTwo();"))
                self.assertEqual(implementations(index), [])
        source = fixture("    SafeOne();\n    SafeTwo();", field="reserveWorkerAttempt")
        source += fixture('    BadOne();\n    BadTwo();\n    "unterminated').replace(
            "void Owner()", "void Broken()")
        self.assertEqual({edge.callee for edge in implementations(build(source))},
                         {"SafeOne", "SafeTwo"})

    def test_arbitrary_lambdas_and_receiver_grammar(self):
        for assignment in ("auto callback", "handler.prepareReservedLaunch",
                           "custom_ops.prepareReservedLaunch", "operations->field",
                           "operations[0].field"):
            with self.subTest(assignment=assignment):
                source = ("void Owner()\n{\n" + assignment +
                          " = [&] {\n    One();\n    Two();\n};\n}\n")
                self.assertEqual(implementations(build(source)), [])

    def test_positional_callbacks_keep_binding_semantics(self):
        fields = ("claim", "afterClaim", "execute", "afterWork", "finalize", "generateReports")
        for first in ("Claim();", "Claim(); Extra();"):
            with self.subTest(first=first):
                source = "void Owner()\n{\nCheckpointAnalysisOperations operations{\n"
                source += "    [&] { " + first + " },\n"
                source += ",\n".join("    [&] { " + field.title() + "(); }"
                                        for field in fields[1:])
                source += "};\n}\n"
                index = build(source)
                self.assertEqual(implementations(index), [])
                bindings = {(edge.operation, edge.callee) for edge in index.callees_of("Owner")
                            if edge.relationship_kind == "operation_binding"}
                expected = {("operations." + field, field.title()) for field in fields[1:]}
                if first == "Claim();":
                    expected.add(("operations.claim", "Claim"))
                self.assertEqual(bindings, expected)
                if first != "Claim();":
                    self.assertEqual({edge.relationship_kind for edge in index.callers_of("Extra")},
                                     {"callback_invocation"})

    def test_operation_invocations_take_precedence(self):
        index = build(fixture("    operations.reserveWorkerAttempt();\n"
                              "    operations_.launchPreparedWorker();\n"
                              "    RealOne();\n    RealTwo();"))
        invocations = [edge for edge in index.callees_of("Owner")
                       if edge.relationship_kind == "operation_invocation"]
        self.assertEqual({edge.operation for edge in invocations},
                         {"operations.reserveWorkerAttempt", "operations_.launchPreparedWorker"})
        self.assertTrue(all(edge.callee == edge.operation for edge in invocations))
        self.assertEqual({edge.callee for edge in implementations(index)}, {"RealOne", "RealTwo"})

    def test_resolved_members_take_precedence(self):
        source = "void Worker::run()\n{\n}\n" + fixture(
            "    Worker worker{};\n    worker.run();\n    Worker().run();\n"
            "    RealOne();\n    RealTwo();")
        index = build(source)
        resolved = [edge for edge in index.callees_of("Owner") if edge.callee == "Worker::run"]
        self.assertEqual(len(resolved), 2)
        self.assertTrue(all(edge.relationship_kind == "direct_invocation" and edge.operation is None
                            for edge in resolved))
        self.assertFalse(any(edge.callee == "run" for edge in implementations(index)))

    def test_direct_relationships_and_traversal_exclude_implementations(self):
        source = "void Leaf()\n{\n}\nvoid Target()\n{\n    Leaf();\n}\n"
        source += fixture("    Target();\n    Other();")
        source += "void DirectOwner()\n{\n    Target();\n}\n"
        index = build(source)
        self.assertEqual(index.relationship("Owner", "Target"), [])
        self.assertEqual(index.trace_calls("Owner"), [])
        self.assertEqual({row["callee"] for row in index.trace_calls("DirectOwner")},
                         {"Target", "Leaf"})
        result = MetadataInterface(index).dispatch({
            "op": "discover_relationship_paths", "scope": "SchedulerCore",
            "from": "Owner", "to": "Target", "max_hops": 3,
        })
        self.assertEqual(result["paths"], [])

    def test_single_call_binding_remains_intact(self):
        for receiver in ("operation", "operations", "operations_", "daemonOperations"):
            with self.subTest(receiver=receiver):
                index = build(fixture("    return Sole();", receiver=receiver))
                edges = index.callees_of("Owner")
                self.assertEqual([(edge.callee, edge.relationship_kind, edge.operation) for edge in edges],
                                 [("Sole", "operation_binding", receiver + ".prepareReservedLaunch")])

    def test_conditional_calls_preserve_fields_and_determinism(self):
        source = fixture("    if (mode == 0) { Train(); }\n"
                         "    else if (mode == 1) { Infer(); }\n"
                         "    else { Analyze(); }")
        first, second = build(source), build(source)
        edges = implementations(first)
        self.assertEqual([(edge.callee, edge.line) for edge in edges],
                         [("Train", 4), ("Infer", 5), ("Analyze", 6)])
        self.assertEqual([asdict(edge) for edge in edges],
                         [asdict(edge) for edge in implementations(second)])
        self.assertEqual(set(asdict(edges[0])),
                         {"file", "line", "caller", "callee", "relationship_kind", "operation"})
        self.assertTrue(all(edge.operation == "operations.prepareReservedLaunch" for edge in edges))

    def test_nested_callbacks_do_not_inherit_outer_operation(self):
        index = build(fixture("    OuterOne();\n"
                              "    auto nested = [&] { NestedOne(); NestedTwo(); };\n"
                              "    operations.launchPreparedWorker = [&] { InnerOne(); InnerTwo(); };\n"
                              "    OuterTwo();"))
        self.assertEqual({(edge.operation, edge.callee) for edge in implementations(index)}, {
            ("operations.prepareReservedLaunch", "OuterOne"),
            ("operations.prepareReservedLaunch", "OuterTwo"),
            ("operations.launchPreparedWorker", "InnerOne"),
            ("operations.launchPreparedWorker", "InnerTwo"),
        })
        self.assertTrue(all(edge.relationship_kind == "callback_invocation"
                            for edge in index.callers_of("NestedOne")))

    def test_unsupported_nested_lambdas_fail_closed(self):
        for capture in ("[&]<typename T>(T t)", "[x = values[0]]()", "[&] [[nodiscard]] ()",
                        "[&] constexpr", "[&] consteval", "[&"):
            with self.subTest(capture=capture):
                index = build(fixture("    OuterOne();\n    auto nested = " + capture +
                                      " { HiddenOne(); HiddenTwo(); };\n    OuterTwo();"))
                self.assertEqual(implementations(index), [])

    def test_comments_between_call_name_and_parenthesis(self):
        index = build(fixture("    First /* comment */ ();\n    Second();\n    Third();"))
        self.assertEqual({edge.callee for edge in implementations(index)}, {"First", "Second", "Third"})
        self.assertEqual(len(index.callers_of("First")), 1)


if __name__ == "__main__":
    unittest.main()
