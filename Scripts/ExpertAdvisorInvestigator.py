#!/usr/bin/env python3
"""Read-only deterministic investigator for the ExpertAdvisor repository."""

from __future__ import annotations

import argparse
import json
import re
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

# Navigation metadata only.  These routes identify where investigation starts;
# they do not restate or supersede architecture.
DOMAIN_ROUTES = {
    "data-pipeline": {
        "volume": "docs/architecture/Volume_II_Data_Pipeline.md",
        "foundation_sections": ("4", "7"),
        "implementation": (
            "Headers/FeatureLayout.hpp",
            "Headers/MarketStructureRegistry.hpp",
        ),
        "adrs": (),
    },
    "labels": {
        "volume": "docs/architecture/Volume_III_Label_Generation.md",
        "foundation_sections": ("3", "4", "5"),
        "implementation": ("Database/LSTM_schema.sql",),
        "adrs": (),
    },
    "model": {
        "volume": "docs/architecture/Volume_IV_Model_Architecture.md",
        "foundation_sections": ("4", "5", "6"),
        "implementation": (
            "Headers/ModelInputContract.hpp",
            "Headers/ModelInputExpansion.hpp",
            "Headers/FeatureLayout.hpp",
            "Headers/MarketStructureRegistry.hpp",
            "Headers/FeatureAblation.hpp",
            "Builds/SemanticWorkers/registry.json",
        ),
        "adrs": (),
    },
    "training": {
        "volume": "docs/architecture/Volume_V_Training_Engine.md",
        "foundation_sections": ("3", "4", "6", "11"),
        "implementation": (
            "Headers/ModelInputContract.hpp",
            "Builds/SemanticWorkers/registry.json",
        ),
        "adrs": (),
    },
    "inference": {
        "volume": "docs/architecture/Volume_VI_Inference_Evaluation.md",
        "foundation_sections": ("3", "4", "6", "11"),
        "implementation": (
            "Headers/ModelInputContract.hpp",
            "Builds/SemanticWorkers/registry.json",
        ),
        "adrs": (),
    },
    "experiment-lifecycle": {
        "volume": "docs/architecture/Volume_VII_Experiment_Lifecycle.md",
        "foundation_sections": ("4", "5", "7", "8", "9"),
        "implementation": ("Database/LSTM_schema.sql",),
        "adrs": (
            "docs/architecture/adr/ADR-0002-deterministic-experiment-identity.md",
        ),
    },
    "recommendations": {
        "volume": "docs/architecture/Volume_VIII_Recommendation_Engine.md",
        "foundation_sections": ("2", "5", "8", "10"),
        "implementation": ("Database/LSTM_schema.sql",),
        "adrs": (
            "docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md",
            "docs/architecture/adr/ADR-0005-manual-recommendation-conversion.md",
            "docs/architecture/adr/ADR-0006-phase-6a-follow-up-proposal.md",
            "docs/architecture/adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md",
            "docs/architecture/adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md",
            "docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md",
        ),
    },
    "profitability": {
        "volume": "docs/architecture/Volume_IX_Trading_Profitability.md",
        "foundation_sections": ("2", "4", "11"),
        "implementation": ("Database/LSTM_schema.sql",),
        "adrs": (),
    },
    "research-automation": {
        "volume": "docs/architecture/Volume_X_Research_Automation.md",
        "foundation_sections": ("2", "6", "8", "9", "10"),
        "implementation": ("Database/LSTM_schema.sql",),
        "adrs": tuple(
            f"docs/architecture/adr/ADR-{n:04d}-{slug}.md"
            for n, slug in (
                (10, "campaign-operations-ownership-and-scope"),
                (11, "campaign-operational-authorization"),
                (12, "campaign-budget-and-reservations"),
                (13, "operational-request-and-handoff"),
                (14, "campaign-lifecycle-controls-completion-and-archival"),
                (15, "cancellation-reconciliation-and-recovery"),
                (16, "scheduler-atomic-claim-hardening"),
                (17, "campaign-privileges-and-audit"),
            )
        ),
    },
    "scheduler": {
        "volume": "docs/architecture/Volume_XI_Scheduler.md",
        "foundation_sections": ("6", "8", "9", "10"),
        "implementation": (
            "Builds/SemanticWorkers/registry.json",
            "Database/LSTM_schema.sql",
        ),
        "adrs": (
            "docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md",
            "docs/architecture/adr/ADR-0016-scheduler-atomic-claim-hardening.md",
            "docs/architecture/adr/ADR-0018-scheduler-generation-52-exact-attempt-authority.md",
            "docs/architecture/adr/ADR-0019-campaign-operations-production-dispatch-admission-and-manager.md",
        ),
    },
    "database": {
        "volume": "docs/architecture/Volume_XII_Database.md",
        "foundation_sections": ("7", "8", "9"),
        "implementation": ("Database/LSTM_schema.sql",),
        "adrs": (
            "docs/architecture/adr/ADR-0001-postgresql-source-of-truth.md",
        ),
    },
}

FOUNDATION = "docs/architecture/Volume_I_Foundation.md"


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


def heading_spans(path: Path) -> list[dict]:
    lines = path.read_text().splitlines()
    headings = []
    pattern = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
    for index, line in enumerate(lines, start=1):
        match = pattern.match(line)
        if match:
            headings.append({
                "level": len(match.group(1)),
                "title": match.group(2),
                "line_start": index,
            })
    for index, heading in enumerate(headings):
        end = len(lines)
        for later in headings[index + 1:]:
            if later["level"] <= heading["level"]:
                end = later["line_start"] - 1
                break
        heading["line_end"] = end
    return headings


def numbered_section(path: Path, number: str) -> dict:
    prefix = f"{number}."
    exact_prefix = f"{number} "
    for heading in heading_spans(path):
        title = heading["title"]
        if title.startswith(prefix) or title.startswith(exact_prefix):
            return {
                "path": str(path),
                "section": number,
                "heading": title,
                "line_start": heading["line_start"],
                "line_end": heading["line_end"],
            }
    raise InvestigatorError(f"section {number} not found in {path}")


def relative_section(repo: Path, relative_path: str, number: str) -> dict:
    item = numbered_section(repo / relative_path, number)
    item["path"] = relative_path
    return item


def domain_list(repo: Path) -> dict:
    missing = []
    for name, route in DOMAIN_ROUTES.items():
        for path in (route["volume"], *route["implementation"], *route["adrs"]):
            if not (repo / path).exists():
                missing.append({"domain": name, "path": path})
    return {
        "kind": "domains",
        "domains": sorted(DOMAIN_ROUTES),
        "missing": missing,
    }


def evidence(repo: Path, domain: str) -> dict:
    route = DOMAIN_ROUTES.get(domain)
    if route is None:
        raise InvestigatorError(f"unknown evidence domain: {domain}")

    required = [FOUNDATION, route["volume"], *route["implementation"], *route["adrs"]]
    missing = [path for path in required if not (repo / path).exists()]
    if missing:
        raise InvestigatorError("missing routed authority: " + ", ".join(missing))

    domain_sections = []
    for number in ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"):
        try:
            domain_sections.append(relative_section(repo, route["volume"], number))
        except InvestigatorError:
            pass

    adr_evidence = []
    for path in route["adrs"]:
        headings = heading_spans(repo / path)
        title = headings[0]["title"] if headings else Path(path).name
        adr_evidence.append({"path": path, "heading": title})

    return {
        "kind": "evidence",
        "domain": domain,
        "constitutional_authority": [
            relative_section(repo, FOUNDATION, number)
            for number in route["foundation_sections"]
        ],
        "domain_authority": {
            "path": route["volume"],
            "sections": domain_sections,
        },
        "accepted_decisions": adr_evidence,
        "implementation_authority": list(route["implementation"]),
        "boundary": (
            "Navigation evidence only; architecture and implementation remain "
            "distinct authorities, and this result does not describe live runtime state."
        ),
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

    if kind == "domains":
        lines = ["evidence_domains:"]
        lines.extend(f"  {name}" for name in value["domains"])
        lines.append(
            "missing: "
            + ("none" if not value["missing"] else json.dumps(value["missing"]))
        )
        return "\n".join(lines)

    if kind == "evidence":
        lines = [f"domain: {value['domain']}", "constitutional_authority:"]
        for item in value["constitutional_authority"]:
            lines.append(
                f"  {item['path']}:{item['line_start']}-{item['line_end']} "
                f"— {item['heading']}"
            )
        lines.append("domain_authority:")
        for item in value["domain_authority"]["sections"]:
            lines.append(
                f"  {item['path']}:{item['line_start']}-{item['line_end']} "
                f"— {item['heading']}"
            )
        lines.append("accepted_decisions:")
        for item in value["accepted_decisions"]:
            lines.append(f"  {item['path']} — {item['heading']}")
        if not value["accepted_decisions"]:
            lines.append("  none routed")
        lines.append("implementation_authority:")
        for path in value["implementation_authority"]:
            lines.append(f"  {path}")
        lines.append(f"boundary: {value['boundary']}")
        return "\n".join(lines)

    raise InvestigatorError(f"unsupported result kind: {kind}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only deterministic ExpertAdvisor repository investigator"
    )
    parser.add_argument(
        "command",
        choices=("authorities", "semantics", "workers", "status", "domains", "evidence"),
    )
    parser.add_argument("domain", nargs="?")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args()
    if args.command == "evidence" and not args.domain:
        parser.error("evidence requires a domain")
    if args.command != "evidence" and args.domain:
        parser.error("domain is only valid with evidence")
    return args


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
        elif args.command == "status":
            value = repository_status(repo)
        elif args.command == "domains":
            value = domain_list(repo)
        else:
            value = evidence(repo, args.domain)
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
