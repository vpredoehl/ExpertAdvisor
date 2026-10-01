#!/usr/bin/env python3
"""Read-only deterministic investigator for the ExpertAdvisor repository."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import GenerateExpertAdvisorReference as reference


AUTHORITY_MAP = {
    "architecture": "docs/architecture/README.md",
    "governing_decisions": "docs/architecture/adr/README.md",
    "model_architecture": "docs/architecture/Volume_IV_Model_Architecture.md",
    "training": "docs/architecture/Volume_V_Training_Engine.md",
    "inference_evaluation": "docs/architecture/Volume_VI_Inference_Evaluation.md",
    "experiment_lifecycle": "docs/architecture/Volume_VII_Experiment_Lifecycle.md",
    "profitability": "docs/architecture/Volume_IX_Trading_Profitability.md",
    "scheduler": "docs/architecture/Volume_XI_Scheduler.md",
    "database": "docs/architecture/Volume_XII_Database.md",
    "physical_schema": "Database/LSTM_schema.sql",
    "model_input_contract": "Headers/ModelInputContract.hpp",
    "semantic_layout_expansion": "Headers/ModelInputExpansion.hpp",
    "tensor_columns": "Headers/FeatureLayout.hpp",
    "market_structures": "Headers/MarketStructureRegistry.hpp",
    "feature_ablation": "Headers/FeatureAblation.hpp",
    "semantic_workers": "Builds/SemanticWorkers/registry.json",
}


class InvestigatorError(RuntimeError):
    pass


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def reference_facts(repo: Path) -> dict:
    value = reference.facts(repo)
    if isinstance(value, dict):
        return value

    # Compatibility with Q1's original tuple API.  This can be removed after
    # GenerateExpertAdvisorReference.py adopts the named dictionary contract.
    layout, returns, width, chain, registry, workers = value
    return {
        "layout": layout,
        "return_width": returns,
        "model_width": width,
        "tensor_width": width - returns,
        "layout_chain": chain,
        "registry": registry,
        "workers": workers,
    }


def authorities(repo: Path) -> dict:
    missing = [path for path in AUTHORITY_MAP.values() if not (repo / path).exists()]
    return {
        "kind": "authorities",
        "authorities": AUTHORITY_MAP,
        "missing": missing,
    }


def semantics(repo: Path) -> dict:
    facts = reference_facts(repo)
    return {
        "kind": "semantics",
        "current_layout": facts["layout"],
        "model_input_width": facts["model_width"],
        "tensor_feature_width": facts["tensor_width"],
        "return_suffix_width": facts["return_width"],
        "layout_chain": [
            {
                "layout": layout,
                "maximum_width_constant": width_name,
                "predecessor": predecessor,
            }
            for layout, width_name, predecessor in facts["layout_chain"]
        ],
    }


def workers(repo: Path) -> dict:
    facts = reference_facts(repo)
    return {
        "kind": "workers",
        "registry_schema": facts["registry"]["schema_version"],
        "current_layout": facts["registry"]["current_layout"],
        "workers": [
            {
                "role": worker["worker_role"],
                "layout": worker["semantic_layout"],
                "model_input_width": worker["model_input_width"],
                "capabilities": worker.get("capabilities", []),
                "source_commit": worker["source_commit"],
                "sha256": worker["sha256"],
                "executable": f"Builds/SemanticWorkers/{worker['executable']}",
            }
            for worker in facts["workers"]
        ],
    }


def repository_status(repo: Path) -> dict:
    try:
        head = git(repo, "rev-parse", "HEAD").stdout.strip()
        branch = git(repo, "branch", "--show-current").stdout.strip() or None
        porcelain = git(repo, "status", "--porcelain").stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise InvestigatorError(f"cannot inspect Git repository: {error}") from error

    try:
        expected = reference.render(repo)
        reference_path = repo / "docs/ai/ExpertAdvisorReference.md"
        reference_current = reference_path.exists() and reference_path.read_text() == expected
    except (OSError, KeyError, json.JSONDecodeError, reference.ReferenceError) as error:
        raise InvestigatorError(f"cannot validate ExpertAdvisor reference: {error}") from error

    return {
        "kind": "status",
        "head": head,
        "branch": branch,
        "clean": not bool(porcelain),
        "reference_current": reference_current,
    }


def text_output(value: dict) -> str:
    kind = value["kind"]
    if kind == "authorities":
        lines = ["ExpertAdvisor authorities:"]
        for name, path in value["authorities"].items():
            lines.append(f"{name}: {path}")
        lines.append(
            "missing: " + (", ".join(value["missing"]) if value["missing"] else "none")
        )
        return "\n".join(lines)

    if kind == "semantics":
        lines = [
            f"current_layout: {value['current_layout']}",
            f"model_input_width: {value['model_input_width']}",
            f"tensor_feature_width: {value['tensor_feature_width']}",
            f"return_suffix_width: {value['return_suffix_width']}",
            "layout_chain:",
        ]
        for item in value["layout_chain"]:
            lines.append(
                f"  {item['layout']}: {item['maximum_width_constant']} "
                f"<- {item['predecessor']}"
            )
        return "\n".join(lines)

    if kind == "workers":
        lines = [
            f"registry_schema: {value['registry_schema']}",
            f"current_layout: {value['current_layout']}",
            "workers:",
        ]
        for worker in value["workers"]:
            lines.append(
                f"  {worker['role']}: layout={worker['layout']} "
                f"width={worker['model_input_width']} "
                f"source_commit={worker['source_commit']} sha256={worker['sha256']}"
            )
            lines.append(f"    executable={worker['executable']}")
            lines.append(
                "    capabilities=" + ",".join(worker["capabilities"])
            )
        return "\n".join(lines)

    if kind == "status":
        return "\n".join([
            f"head: {value['head']}",
            f"branch: {value['branch'] or '(detached)'}",
            f"clean: {'true' if value['clean'] else 'false'}",
            f"reference_current: {'true' if value['reference_current'] else 'false'}",
        ])

    raise InvestigatorError(f"unsupported result kind: {kind}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only deterministic ExpertAdvisor repository investigator"
    )
    parser.add_argument(
        "command",
        choices=("authorities", "semantics", "workers", "status"),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    repo = args.repo_root.resolve()

    try:
        if args.command == "authorities":
            value = authorities(repo)
        elif args.command == "semantics":
            value = semantics(repo)
        elif args.command == "workers":
            value = workers(repo)
        else:
            value = repository_status(repo)
    except (OSError, KeyError, json.JSONDecodeError, reference.ReferenceError,
            InvestigatorError) as error:
        print(f"ExpertAdvisorInvestigator: {error}", file=sys.stderr)
        return 1

    if args.format == "json":
        print(json.dumps(value, indent=2, sort_keys=True))
    else:
        print(text_output(value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
