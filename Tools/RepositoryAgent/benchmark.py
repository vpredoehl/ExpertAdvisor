#!/usr/bin/env python3
"""Benchmark selection boundary for RepositoryAgent."""
import importlib
import os

DEFAULT_BENCHMARK_MODULE = "Tools.RepositoryAgent.scheduler_architecture_benchmark"


def load_benchmark(module_name=None):
    """Load a benchmark definition. Scheduler architecture remains the default."""
    name = module_name or os.environ.get("EA_REPOSITORY_BENCHMARK") or DEFAULT_BENCHMARK_MODULE
    module = importlib.import_module(name)
    benchmark = getattr(module, "BENCHMARK", None)
    if not isinstance(benchmark, dict):
        raise RuntimeError(f"benchmark module {name!r} does not export BENCHMARK")
    return benchmark
