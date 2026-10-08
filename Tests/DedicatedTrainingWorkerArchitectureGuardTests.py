#!/usr/bin/env python3
"""Offline fixtures for the TRAIN guard; no build, worker, or registry access.

Run: python3 Tests/DedicatedTrainingWorkerArchitectureGuardTests.py
"""

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path("ExpertAdvisor.xcodeproj/project.pbxproj")
GUARD = Path("Tests/DedicatedTrainingWorkerArchitectureTests.sh")
TRAIN = "0FA000043A00000100AAA001"
SOURCES = "0FAC00003A00000100AAA001"
PROVENANCE = "0FA000083A00000100AAA001"
RELEASE = "0A10000F2F70000100AAA001"
RELEASE_SOURCES = "0A10000E2F70000100AAA001"
RELEASE_PROVENANCE = "0A10000A2F70000100AAA002"
FRAMEWORKS = "0A10000C2F70000100AAA001"


class DedicatedTrainingWorkerArchitectureGuardTests(unittest.TestCase):
    def setUp(self):
        # Keep every fixture inside the authorized development checkout. Copy
        # only the files read by the guard; never follow the MetaNN symlink.
        temporary = tempfile.TemporaryDirectory(prefix=".train-guard-", dir=ROOT)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for relative in (PROJECT, GUARD, Path("LSTM/main.cpp"),
                         Path("LSTM/TrainWorkerMain.cpp"),
                         Path("Sources/TrainingWorkerApplication.cpp"),
                         Path("Sources/TrainingWorkerApplication.hpp")):
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
        self.project = json.loads(subprocess.check_output(
            ["/usr/bin/plutil", "-convert", "json", "-o", "-", "--",
             str(self.root / PROJECT)]))
        self.objects = self.project["objects"]
        self.phases = self.objects[TRAIN]["buildPhases"]

    def check_guard(self, expected_error=None, *, original_project=False):
        if not original_project:
            (self.root / PROJECT).write_text(json.dumps(self.project))
        result = subprocess.run(["bash", str(self.root / GUARD)],
                                cwd=self.root, capture_output=True, text=True)
        output = result.stdout + result.stderr
        if expected_error is None:
            self.assertEqual(result.returncode, 0, output)
            self.assertIn("DedicatedTrainingWorkerArchitectureTests passed", output)
        else:
            self.assertNotEqual(result.returncode, 0, output)
            self.assertIn("TRAIN architecture: " + expected_error, output)

    def test_current_provenance_first_project_passes(self):
        self.assertLess(self.phases.index(PROVENANCE), self.phases.index(SOURCES))
        self.check_guard(original_project=True)

    def test_unrelated_phase_can_precede_provenance_and_sources(self):
        self.phases[:] = [FRAMEWORKS, PROVENANCE, SOURCES]
        self.check_guard()

    def test_object_definition_order_is_irrelevant(self):
        self.project["objects"] = dict(reversed(list(self.objects.items())))
        self.check_guard()

    def test_missing_train_sources_cannot_match_later_target(self):
        self.phases.remove(SOURCES)
        self.check_guard("TRAIN must own its Sources phase")

    def test_borrowed_release_sources_rejected(self):
        self.phases[self.phases.index(SOURCES)] = RELEASE_SOURCES
        self.check_guard("TRAIN must own its Sources phase")

    def test_additional_source_phase_rejected(self):
        self.phases.append(RELEASE_SOURCES)
        self.check_guard("TRAIN must have only its dedicated Sources phase")

    def test_missing_provenance_rejected(self):
        self.phases.remove(PROVENANCE)
        self.check_guard("TRAIN must own its provenance phase")

    def test_provenance_after_sources_rejected(self):
        self.phases[:] = [SOURCES, PROVENANCE, FRAMEWORKS]
        self.check_guard("TRAIN provenance must precede Sources")

    def test_borrowed_release_provenance_rejected(self):
        self.phases[self.phases.index(PROVENANCE)] = RELEASE_PROVENANCE
        self.check_guard("TRAIN must own its provenance phase")

    def test_train_phases_cannot_be_shared(self):
        for phase in (SOURCES, PROVENANCE):
            with self.subTest(phase=phase):
                release_phases = self.objects[RELEASE]["buildPhases"]
                release_phases.append(phase)
                self.check_guard("TRAIN phases must not be shared with another target")
                release_phases.remove(phase)

    def test_phase_types_are_checked(self):
        for phase, expected in ((SOURCES, "TRAIN Sources must be a source phase"),
                                (PROVENANCE, "TRAIN provenance must be a shell-script phase")):
            with self.subTest(phase=phase):
                original = self.objects[phase]["isa"]
                self.objects[phase]["isa"] = "PBXFrameworksBuildPhase"
                self.check_guard(expected)
                self.objects[phase]["isa"] = original

    def test_required_sources_must_each_appear_once(self):
        for name in ("TrainWorkerMain.cpp", "TrainingWorkerApplication.cpp"):
            files = self.objects[SOURCES]["files"]
            build_file = next(f for f in files if Path(
                self.objects[self.objects[f]["fileRef"]]["path"]).name == name)
            with self.subTest(name=name, mutation="missing"):
                files.remove(build_file)
                self.check_guard("TRAIN must compile exactly one " + name)
                files.append(build_file)
            with self.subTest(name=name, mutation="duplicate"):
                files.append(build_file)
                self.check_guard("TRAIN must compile exactly one " + name)
                files.remove(build_file)

    def test_forbidden_sources_rejected_by_reference(self):
        for name in ("main.cpp", "CheckpointModelPersistence.cpp",
                     "InferenceRuntime.cpp", "ManagedInferenceApplication.cpp"):
            with self.subTest(name=name):
                self.objects["fixture-file"] = {"isa": "PBXFileReference", "path": name}
                self.objects["fixture-build-file"] = {
                    "isa": "PBXBuildFile", "fileRef": "fixture-file"}
                files = self.objects[SOURCES]["files"]
                files.append("fixture-build-file")
                self.check_guard("TRAIN must not compile " + name)
                files.remove("fixture-build-file")

    def test_release_must_keep_its_source_phase(self):
        self.objects[RELEASE]["buildPhases"].remove(RELEASE_SOURCES)
        self.check_guard("Release must retain its Sources phase")


if __name__ == "__main__":
    unittest.main(verbosity=2)
