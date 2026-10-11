#!/usr/bin/env python3
"""Run RepositoryAgent regression modules in isolated Python processes."""

import argparse
import ast
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
TESTS = Path(__file__).resolve().parent

EXCLUDED = {
    "test_ledger.py",
    "test_structural.py",
    "test_verifier.py",
}


def uses_unittest(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        for base in node.bases:
            if isinstance(base, ast.Attribute) and base.attr == "TestCase":
                return True

    return False


def run_scheduler_regressions(integration_binary=None):
    """Run scheduler recovery guards without changing the default suite."""
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["EXPERTADVISOR_REPOSITORY_ROOT"] = str(ROOT)

    commands = [
        (
            "scheduler-static",
            [
                sys.executable,
                str(TESTS / "validate_scheduler_launch_recovery.py"),
            ],
        ),
    ]

    if integration_binary is not None:
        binary = Path(integration_binary).resolve(strict=True)
        allowed_root = ROOT / "DerivedData"

        if not binary.is_relative_to(allowed_root):
            raise ValueError(
                "Integration binary must be inside RepositoryAgent DerivedData"
            )

        if binary.name != "LSTM_Release":
            raise ValueError("Expected LSTM_Release executable")

        commands.append(
            (
                "scheduler-integration",
                [
                    str(
                        ROOT
                        / "Tests"
                        / "SchedulerInterruptedLaunchRecoveryIntegrationTests.sh"
                    ),
                    str(binary),
                ],
            )
        )

    for name, command in commands:
        print(f"RUN {name}", flush=True)

        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            check=False,
        )

        if result.returncode != 0:
            print(f"FAIL: {name} (exit {result.returncode})")
            return 1

        print(f"PASS: {name}", flush=True)

    print("SCHEDULER RECOVERY REGRESSIONS: PASS")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Run isolated RepositoryAgent regressions"
    )

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--scheduler-static",
        action="store_true",
        help="Run scheduler recovery static regression only",
    )

    group.add_argument(
        "--scheduler-integration",
        metavar="ISOLATED_LSTM_RELEASE",
        help="Run scheduler static and disposable-PostgreSQL regressions",
    )

    args = parser.parse_args()

    if args.scheduler_static:
        return run_scheduler_regressions()

    if args.scheduler_integration:
        try:
            return run_scheduler_regressions(args.scheduler_integration)
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["EXPERTADVISOR_REPOSITORY_ROOT"] = str(ROOT)

    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT), existing) if part
    )

    passed = []
    failed = []
    excluded = []

    for path in sorted(TESTS.glob("test_*.py")):
        if path.name in EXCLUDED:
            excluded.append(path.name)
            continue

        module = f"Tools.RepositoryAgent.tests.{path.stem}"

        if uses_unittest(path):
            command = [
                sys.executable,
                "-m",
                "unittest",
                module,
                "-v",
            ]
        else:
            command = [sys.executable, "-m", module]

        print(f"RUN {path.name}", flush=True)

        try:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            failed.append((path.name, "TIMEOUT"))
            print("  FAIL: TIMEOUT", flush=True)
            continue

        if result.returncode:
            failed.append(
                (path.name, result.stdout + result.stderr)
            )
            print("  FAIL", flush=True)
        else:
            passed.append(path.name)
            print("  PASS", flush=True)

    print()
    print("REPOSITORYAGENT ISOLATED REGRESSION SUMMARY")
    print(f"Passed:   {len(passed)}")
    print(f"Failed:   {len(failed)}")
    print(f"Excluded: {len(excluded)}")

    for name, details in failed:
        print(f"\nFAILED: {name}\n{details}")

    for name in excluded:
        print(f"EXCLUDED: {name}")

    if failed:
        return 1

    if len(passed) != 33 or len(excluded) != 3:
        print("Unexpected regression inventory.")
        return 1

    print("\nREPOSITORYAGENT REGRESSION SUITE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
