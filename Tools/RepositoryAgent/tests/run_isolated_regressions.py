#!/usr/bin/env python3
"""Run RepositoryAgent regression modules in isolated Python processes."""

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


def main():
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

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
    print("PHASE 24B ISOLATED REGRESSION SUMMARY")
    print(f"Passed:   {len(passed)}")
    print(f"Failed:   {len(failed)}")
    print(f"Excluded: {len(excluded)}")

    for name, details in failed:
        print(f"\nFAILED: {name}\n{details}")

    for name in excluded:
        print(f"EXCLUDED: {name}")

    if failed:
        return 1

    if len(passed) != 29 or len(excluded) != 3:
        print("Unexpected regression inventory.")
        return 1

    print("\nPHASE 24B REGRESSION SUITE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
