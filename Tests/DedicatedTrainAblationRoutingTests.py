#!/usr/bin/env python3
"""Prospective routing with synthetic artifacts only; no worker execution."""
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "refresh", ROOT / "Scripts/RefreshSemanticWorkerGeneration.py")
refresh = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(refresh)
publisher, rollover = refresh.publisher, refresh.rollover

PROBE = r'''
#include "SchedulerCore/TrainingWorkerSelection.hpp"
#include "FeatureAblation.hpp"
#include <iostream>
int main(int argc, const char* argv[]) {
    if (argc != 6) return 1;
    try {
        const auto registry = EA::Scheduler::SemanticWorkerRegistry::Load(
            {argv[1], std::nullopt, 13, 171});
        const int layout = std::stoi(argv[2]);
        const auto mask = EA::FeatureAblationMask::ParseForSemanticLayout(
            argv[4], layout ? layout : 13);
        EA::Scheduler::PersistedWorkerSemanticIdentity identity;
        if (layout) { identity.inputWidth = std::stoul(argv[3]);
                      identity.layoutVersion = layout; }
        identity.modelIdentityExpected = std::string(argv[5]) == "resume";
        const auto selection = EA::Scheduler::SelectTrainingWorker(identity,
            registry, EA::Scheduler::RequiredTrainingWorkerCapabilities(mask.CanonicalText()));
        if (!selection.selected) { std::cout << "REJECT\n"; return 0; }
        if (!registry.validateRuntimeForExecutable(selection.canonicalExecutablePath).ready)
            return 2;
        std::cout << selection.canonicalExecutablePath << '\n';
    } catch (const std::invalid_argument&) { std::cout << "INVALID_MASK\n"; }
}
'''


class DedicatedTrainAblationRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="ea_train_ablation_routing_")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name).resolve()
        cpp, cls.probe = cls.root / "probe.cpp", cls.root / "probe"
        cpp.write_text(PROBE)
        subprocess.run([os.environ.get("CXX", "/usr/bin/clang++"), "-std=c++20",
                        "-Wall", "-Wextra", "-Werror", "-I" + str(ROOT / "Headers"),
                        "-I" + str(ROOT / "Sources"), str(cpp),
                        str(ROOT / "Sources/SchedulerCore/SemanticWorkerRegistry.cpp"),
                        "-o", str(cls.probe)], check=True, capture_output=True, text=True)

    def fixture(self, qualified):
        root = self.root / ("qualified" if qualified else "plain")
        root.mkdir()
        products = root / "synthetic-products"
        products.mkdir()
        resources = {}
        for name, _ in publisher.RUNTIME_RESOURCE_SPECS:
            resources[name] = products / name
            resources[name].write_bytes(("synthetic-" + name).encode())
        manifest, identity = publisher.runtime_manifest(resources)
        runtime = publisher.stage_runtime_package(root, resources, manifest, identity)
        workers, paths = [], {}
        for label, role, layout, width, schema, caps, commit in (
            ("legacy", "train", 13, 171, 1,
             ["train", "infer", "analyze", "train_feature_ablation_v1"], "a" * 40),
            ("dedicated", "train", 13, 171, 2,
             refresh._refreshed_training_capabilities(qualified), "b" * 40),
            ("infer", "infer", 13, 171, 2, ["infer"], "c" * 40),
            ("historical", "train", 9, 103, 1,
             ["train", "train_feature_ablation_v1"], "d" * 40)):
            source = products / label
            source.write_bytes(("synthetic-" + label).encode())
            source.chmod(0o700)
            digest = publisher.sha256(source)
            relative, manifest, worker = rollover._worker_value(
                layout, width, commit, digest, role, schema, caps, identity)
            worker["worker_rule"] = "current" if label in ("dedicated", "infer") else "historical"
            worker["selection_priority"] = 1 if label == "dedicated" else 0
            paths[label] = str(rollover._stage_worker(
                root, source, relative, manifest, digest, runtime))
            workers.append(worker)
        registry = {"schema_version": 5, "current_layout": 13,
                    "runtimes": [runtime], "workers": workers}
        publisher.validate_existing_registry(root, registry)
        publisher.atomic_write_json(root / "registry.json", registry)
        return root / "registry.json", paths

    def test_plain_and_hypothetically_qualified_routing(self):
        for qualified in (False, True):
            registry, paths = self.fixture(qualified)
            before = {p.relative_to(registry.parent): p.read_bytes()
                      for p in registry.parent.rglob("*") if p.is_file()}
            persisted = "dedicated" if qualified else "legacy"
            cases = [
                ("fresh control", 0, 0, "", "fresh", "dedicated"),
                ("persisted control", 13, 171, "", "fresh", persisted),
                ("fresh ablation", 0, 0, "relative_tick_volume", "fresh",
                 "dedicated" if qualified else "REJECT"),
                ("persisted ablation", 13, 171, "relative_tick_volume", "fresh", persisted),
                ("canonical wildcard", 13, 171, "fibonacci.*", "fresh", persisted),
                ("invalid mask", 13, 171, "unknown_feature", "fresh", "INVALID_MASK"),
                ("control continuation", 13, 171, "", "resume", persisted),
                ("masked continuation", 13, 171, "relative_tick_volume", "resume", persisted),
                ("historical control", 9, 103, "", "resume", "historical"),
                ("historical ablation", 9, 103, "fibonacci.*", "resume", "historical"),
                ("unsupported historical mask", 9, 103, "fibonacci_lifecycle.*",
                 "resume", "INVALID_MASK"),
                ("missing continuation identity", 0, 0, "", "resume", "REJECT"),
                ("incompatible width", 13, 103, "", "resume", "REJECT")]
            for label, layout, width, mask, mode, expected in cases:
                with self.subTest(qualified=qualified, case=label):
                    result = subprocess.run([str(self.probe), str(registry), str(layout),
                                             str(width), mask, mode], check=True,
                                            capture_output=True, text=True)
                    self.assertEqual(result.stdout.strip(), paths.get(expected, expected))
            self.assertEqual(before, {p.relative_to(registry.parent): p.read_bytes()
                                     for p in registry.parent.rglob("*") if p.is_file()})


if __name__ == "__main__":
    unittest.main()
