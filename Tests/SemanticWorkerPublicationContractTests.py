#!/usr/bin/env python3
"""Offline qualification tests: disposable artifacts, no worker execution."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock


REPOSITORY = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "publication_contract_refresh", REPOSITORY / "Scripts/RefreshSemanticWorkerGeneration.py")
refresh = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(refresh)
rollover = refresh.rollover
publisher = refresh.publisher


class SemanticWorkerPublicationContractTests(unittest.TestCase):
    train_commit = "a" * 40
    infer_commit = "b" * 40

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix=".publication-contract-", dir=REPOSITORY)
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.root = self.base / "artifacts"
        products = self.base / "products"
        products.mkdir()
        self.training = products / "lstm-train-worker"
        self.inference = products / "lstm-infer-worker"
        for candidate in (self.training, self.inference):
            candidate.write_bytes(candidate.name.encode())
            candidate.chmod(0o700)
        resources = {}
        for name, _ in publisher.RUNTIME_RESOURCE_SPECS:
            resources[name] = products / name
            resources[name].write_bytes(name.encode())
        self.root.mkdir()
        manifest, identity = publisher.runtime_manifest(resources)
        runtime = publisher.stage_runtime_package(self.root, resources, manifest, identity)
        workers = []
        for role, schema, capabilities in (
            ("train", publisher.LEGACY_WORKER_MANIFEST_SCHEMA_VERSION,
             rollover.TRAINING_CAPABILITIES),
            ("infer", publisher.WORKER_MANIFEST_SCHEMA_VERSION, ["infer"]),
        ):
            # Already-published fixtures have no modern identity interface.
            old = products / ("old-" + role)
            old.write_bytes(("old-" + role).encode())
            old.chmod(0o700)
            relative, manifest, worker = rollover._worker_value(
                13, 171, "c" * 40, publisher.sha256(old), role, schema, capabilities, identity)
            rollover._stage_worker(self.root, old, relative, manifest,
                                   publisher.sha256(old), runtime)
            workers.append(worker)
        registry = {"schema_version": publisher.REGISTRY_SCHEMA_VERSION,
                    "current_layout": 13, "runtimes": [runtime], "workers": workers}
        publisher.validate_existing_registry(self.root, registry)
        publisher.atomic_write_json(self.root / "registry.json", registry)
        publisher.update_current_link_after_registry_commit(
            self.root, Path(workers[1]["executable"]).parent)
        self.configure_contract("refresh")

    def configure_contract(self, operation):
        self.layout, self.width = (14 if operation == "rollover" else 13), 171
        self.records = {}
        for role, path, commit in (("train", self.training, self.train_commit),
                                   ("infer", self.inference, self.infer_commit)):
            self.records[role] = {
                "identity_contract_version": "1", "artifact_role": "lstm-" + role + "-worker",
                "source_commit": commit, "semantic_layout": str(self.layout),
                "model_input_width": str(self.width), "canonical_executable": str(path),
                "executable_sha256": "sha256:" + publisher.sha256(path),
            }
        self.output_override = {}

    def candidate_output(self, command, **kwargs):
        if command[0] == "/usr/bin/strings":
            commit = self.train_commit if Path(command[1]) == self.training else self.infer_commit
            return subprocess.CompletedProcess(command, 0, commit + "\n", "")
        self.assertEqual(command[1:], ["--build-identity"])
        self.assertIn(Path(command[0]), (self.training, self.inference))
        role = "train" if Path(command[0]) == self.training else "infer"
        output = role.upper() + "_WORKER_BUILD_IDENTITY," + ",".join(
            f"{key}={value}" for key, value in self.records[role].items())
        return subprocess.CompletedProcess(command, 0, self.output_override.get(role, output) + "\n", "")

    def publish(self, operation):
        if operation == "infer":
            return publisher.publish(self.root, self.inference, "current", self.layout,
                                     self.width, self.infer_commit, ["infer"])
        arguments = (self.root, self.training, self.inference,
                     self.layout, self.width, self.train_commit)
        if operation == "rollover":
            return rollover.rollover(*arguments, dedicated_training=True,
                                     inference_commit=self.infer_commit)
        return refresh.refresh(*arguments, inference_commit=self.infer_commit)

    def authority_snapshot(self):
        return {str(path.relative_to(self.root)):
                ("link", os.readlink(path)) if path.is_symlink() else
                ("file", path.read_bytes(), path.stat().st_mode)
                for path in self.root.rglob("*")
                if (path.is_symlink() or path.is_file()) and path.name != ".publish.lock"}

    def reject_without_publication(self, operation, message):
        before = self.authority_snapshot()
        with mock.patch.object(publisher.subprocess, "run", side_effect=self.candidate_output), \
                mock.patch.object(publisher, "stage_runtime_package") as stage, \
                mock.patch.object(publisher, "atomic_write_json") as commit, \
                mock.patch.object(publisher, "update_current_link_after_registry_commit") as link:
            with self.assertRaisesRegex(publisher.PublishError, message):
                self.publish(operation)
        stage.assert_not_called()
        commit.assert_not_called()
        link.assert_not_called()
        self.assertEqual(self.authority_snapshot(), before)
        self.assertFalse(list(self.root.rglob("*.stage")))

    def qualified_publication(self, operation):
        self.configure_contract(operation)
        before = publisher.load_registry(self.root / "registry.json")
        old_bytes = {w["executable"]: (self.root / w["executable"]).read_bytes()
                     for w in before["workers"]}
        with mock.patch.object(publisher.subprocess, "run", side_effect=self.candidate_output) as execute:
            self.publish(operation)
        identity_calls = [call.args[0] for call in execute.call_args_list
                          if call.args[0][1:] == ["--build-identity"]]
        self.assertEqual(identity_calls.count([str(self.inference), "--build-identity"]), 1)
        self.assertEqual(identity_calls.count([str(self.training), "--build-identity"]),
                         0 if operation == "infer" else 1)
        after = publisher.load_registry(self.root / "registry.json")
        self.assertEqual(after["current_layout"], self.layout)
        current = {w["worker_role"]: w for w in after["workers"] if w["worker_rule"] == "current"}
        self.assertEqual(current["infer"]["source_commit"], self.infer_commit)
        if operation != "infer":
            self.assertEqual(current["train"]["source_commit"], self.train_commit)
            self.assertTrue(all((w["semantic_layout"], w["model_input_width"]) ==
                                (self.layout, self.width) for w in current.values()))
        for relative, contents in old_bytes.items():
            self.assertEqual((self.root / relative).read_bytes(), contents)

    def test_qualified_infer_publication(self):
        self.qualified_publication("infer")

    def test_infer_publication_preserves_canonical_paths_for_relative_or_aliased_root(self):
        canonical_root = self.root
        alias = self.base / "artifact-alias"
        alias.symlink_to(canonical_root, target_is_directory=True)
        for root in (canonical_root.relative_to(REPOSITORY), alias):
            with self.subTest(root=root), \
                    mock.patch.object(publisher.subprocess, "run", side_effect=self.candidate_output):
                self.root = root
                published = self.publish("infer")
            self.assertTrue(published.is_absolute())
            self.assertEqual(published, published.resolve())
            self.assertTrue(published.is_relative_to(canonical_root))
            publisher.load_registry(canonical_root / "registry.json")
        self.root = canonical_root

    def test_qualified_dedicated_rollover_with_independent_commits(self):
        self.qualified_publication("rollover")

    def test_qualified_refresh_with_independent_commits(self):
        self.qualified_publication("refresh")

    def test_semantic_rejection_is_atomic_for_all_modern_routes(self):
        for operation in ("infer", "rollover", "refresh"):
            for role in (("infer",) if operation == "infer" else ("train", "infer")):
                for key, value in (("semantic_layout", "12"), ("model_input_width", "127"),
                                   ("semantic_layout", None), ("model_input_width", None)):
                    with self.subTest(operation=operation, role=role, field=key, value=value):
                        self.configure_contract(operation)
                        if value is None:
                            del self.records[role][key]
                        else:
                            self.records[role][key] = value
                        self.reject_without_publication(operation, "semantic layout or model input width")

    def test_role_commit_and_hash_rejection_is_atomic(self):
        for operation in ("infer", "rollover", "refresh"):
            for role in (("infer",) if operation == "infer" else ("train", "infer")):
                for key, value in (("artifact_role", "other-worker"), ("source_commit", "e" * 40),
                                   ("executable_sha256", "sha256:" + "f" * 64)):
                    with self.subTest(operation=operation, role=role, field=key):
                        self.configure_contract(operation)
                        self.records[role][key] = value
                        self.reject_without_publication(operation, "build identity mismatch")

    def test_malformed_identity_rejection_is_atomic(self):
        for operation in ("infer", "rollover", "refresh"):
            for role in (("infer",) if operation == "infer" else ("train", "infer")):
                for malformed in ("", "bad-header", "multiple-records", "duplicate-field",
                                  "empty-value", "missing-equals", "missing-version", "wrong-version"):
                    with self.subTest(operation=operation, role=role, malformed=malformed):
                        self.configure_contract(operation)
                        if malformed == "missing-version":
                            del self.records[role]["identity_contract_version"]
                        elif malformed == "wrong-version":
                            self.records[role]["identity_contract_version"] = "99"
                        else:
                            valid = self.candidate_output([str(self.training if role == "train"
                                                              else self.inference), "--build-identity"]).stdout.strip()
                            outputs = {"": "", "bad-header": valid.replace(role.upper(), "UNKNOWN", 1),
                                       "multiple-records": valid + "\n" + valid,
                                       "duplicate-field": valid + ",semantic_layout=" + str(self.layout),
                                       "empty-value": valid.replace("semantic_layout=" + str(self.layout), "semantic_layout="),
                                       "missing-equals": valid + ",malformed"}
                            self.output_override[role] = outputs[malformed]
                        self.reject_without_publication(operation, "build identity")

    def test_current_infer_must_match_training_reference_width(self):
        self.width = 170
        self.records["infer"]["model_input_width"] = "170"
        self.reject_without_publication("infer", "disagrees with training/reference binding")

    def test_failed_qualification_does_not_create_artifact_root(self):
        existing = self.authority_snapshot()
        original = self.root
        self.root = self.base / "unused-artifact-root"
        self.records["infer"].pop("semantic_layout")
        self.reject_without_publication("infer", "semantic layout or model input width")
        self.assertFalse(self.root.exists())
        self.root = original
        self.assertEqual(self.authority_snapshot(), existing)

    def test_published_workers_without_semantic_identity_remain_loadable(self):
        before = self.authority_snapshot()
        with mock.patch.object(publisher.subprocess, "run",
                               side_effect=AssertionError("registry loading must not execute workers")):
            loaded = publisher.load_registry(self.root / "registry.json")
        self.assertEqual(len(loaded["workers"]), 2)
        self.assertEqual(self.authority_snapshot(), before)

    def cli_arguments(self, layout, width, rule="current"):
        return argparse.Namespace(repository_root=REPOSITORY, built_executable=self.inference,
                                  artifact_root=self.root, worker_rule=rule, semantic_layout=layout,
                                  model_input_width=width, source_commit=self.infer_commit,
                                  capabilities=["infer"], worker_role="infer")

    def test_explicit_current_cli_contract_cannot_bypass_source(self):
        for layout, width in ((12, 171), (13, 127), (0, 171), (13, 0),
                              (12, None), (None, 127)):
            with self.subTest(layout=layout, width=width), \
                    mock.patch.object(publisher, "parse_arguments", return_value=self.cli_arguments(layout, width)), \
                    mock.patch.object(publisher, "current_semantic_contract", return_value=(13, 171)) as source, \
                    mock.patch.object(publisher, "publish") as publish:
                with self.assertRaisesRegex(publisher.PublishError, "current contract disagrees with source"):
                    publisher.main()
                source.assert_called_once_with(REPOSITORY)
                publish.assert_not_called()

    def test_matching_current_cli_contract_and_defaults(self):
        for layout, width in ((13, 171), (13, None), (None, 171), (None, None)):
            with self.subTest(layout=layout, width=width), \
                    mock.patch.object(publisher, "parse_arguments", return_value=self.cli_arguments(layout, width)), \
                    mock.patch.object(publisher, "current_semantic_contract", return_value=(13, 171)) as source, \
                    mock.patch.object(publisher, "clean_source_commit", return_value=self.infer_commit), \
                    mock.patch.object(publisher, "publish", return_value=self.inference) as publish, \
                    mock.patch("builtins.print"):
                self.assertEqual(publisher.main(), 0)
                source.assert_called_once_with(REPOSITORY)
                self.assertEqual((publish.call_args.kwargs["layout"], publish.call_args.kwargs["width"]), (13, 171))
                self.assertTrue(publish.call_args.kwargs["check_embedded_commit"])

    def test_historical_cli_keeps_explicit_contract(self):
        with mock.patch.object(publisher, "parse_arguments", return_value=self.cli_arguments(12, 127, "historical")), \
                mock.patch.object(publisher, "current_semantic_contract") as source, \
                mock.patch.object(publisher, "clean_source_commit") as clean, \
                mock.patch.object(publisher, "publish", return_value=self.inference) as publish, \
                mock.patch("builtins.print"):
            self.assertEqual(publisher.main(), 0)
        source.assert_not_called()
        clean.assert_not_called()
        self.assertEqual((publish.call_args.kwargs["layout"], publish.call_args.kwargs["width"]), (12, 127))

    def test_infer_identity_reports_authoritative_compiled_constants(self):
        source = (REPOSITORY / "LSTM/InferWorkerMain.cpp").read_text()
        self.assertIn('#include "ModelInputExpansion.hpp"', source)
        self.assertIn('",semantic_layout=" << EA::kModelInputSemanticLayoutVersion', source)
        self.assertIn('",model_input_width=" << EA::kCurrentModelInputWidth', source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
