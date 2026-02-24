"""
Experiment Runner
=================

Automated runner for all experiment suites with checkpointing,
result aggregation, and reporting.

Usage
-----
    python -m experiments.run_all_experiments --suite all
    python -m experiments.run_all_experiments --suite architecture
    python -m experiments.run_all_experiments --config configs/experiments/depth_scaling.yaml
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class ExperimentResult:
    """Container for a single experiment run's results."""

    name: str
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    duration_seconds: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "completed"
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "config": self.config,
            "metrics": self.metrics,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp,
            "status": self.status,
            "error": self.error,
        }


class ExperimentRunner:
    """
    Orchestrates experiment execution with checkpointing and result storage.

    Parameters
    ----------
    results_dir : str or Path
        Directory to save results.
    seed : int, default=42
        Base random seed.
    """

    def __init__(
        self,
        results_dir: str = "results",
        seed: int = 42,
    ):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self._results: List[ExperimentResult] = []
        self._registry: Dict[str, Callable] = {}

    def register(self, name: str, fn: Callable) -> None:
        """Register an experiment function."""
        self._registry[name] = fn

    def run_single(
        self,
        name: str,
        config: Dict[str, Any],
        fn: Optional[Callable] = None,
    ) -> ExperimentResult:
        """
        Run a single experiment.

        Parameters
        ----------
        name : str
            Experiment name.
        config : dict
            Experiment configuration.
        fn : callable, optional
            Experiment function. Looked up from registry if None.

        Returns
        -------
        ExperimentResult
        """
        if fn is None:
            fn = self._registry.get(name)
            if fn is None:
                raise ValueError(f"Experiment '{name}' not registered.")

        print(f"\n{'='*60}")
        print(f"  Running: {name}")
        print(f"  Config:  {config}")
        print(f"{'='*60}")

        start = time.time()
        try:
            metrics = fn(config, seed=self.seed)
            duration = time.time() - start
            result = ExperimentResult(
                name=name,
                config=config,
                metrics=metrics,
                duration_seconds=duration,
            )
            print(f"  Completed in {duration:.1f}s")
            print(f"  Metrics: {metrics}")
        except Exception as e:
            duration = time.time() - start
            result = ExperimentResult(
                name=name,
                config=config,
                metrics={},
                duration_seconds=duration,
                status="failed",
                error=str(e),
            )
            print(f"  FAILED after {duration:.1f}s: {e}")

        self._results.append(result)
        return result

    def run_sweep(
        self,
        name: str,
        param_name: str,
        param_values: Sequence,
        base_config: Dict[str, Any],
        fn: Optional[Callable] = None,
    ) -> List[ExperimentResult]:
        """
        Run a parameter sweep experiment.

        Parameters
        ----------
        name : str
            Base experiment name.
        param_name : str
            Parameter to sweep.
        param_values : sequence
            Values to sweep over.
        base_config : dict
            Base configuration (param_name will be overridden).
        fn : callable, optional
            Experiment function.

        Returns
        -------
        list of ExperimentResult
        """
        results = []
        for val in param_values:
            config = {**base_config, param_name: val}
            exp_name = f"{name}_{param_name}={val}"
            result = self.run_single(exp_name, config, fn)
            results.append(result)
        return results

    def save_results(self, filename: Optional[str] = None) -> Path:
        """Save all results to JSON."""
        if filename is None:
            filename = f"results_{datetime.now():%Y%m%d_%H%M%S}.json"

        path = self.results_dir / filename
        data = {
            "timestamp": datetime.now().isoformat(),
            "seed": self.seed,
            "n_experiments": len(self._results),
            "results": [r.to_dict() for r in self._results],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        print(f"\nResults saved to {path}")
        return path

    @property
    def results(self) -> List[ExperimentResult]:
        return list(self._results)

    def summary(self) -> str:
        """Print a summary of all results."""
        lines = [
            f"\n{'='*60}",
            "  EXPERIMENT SUMMARY",
            f"{'='*60}",
            f"  Total experiments: {len(self._results)}",
            f"  Completed: {sum(1 for r in self._results if r.status == 'completed')}",
            f"  Failed:    {sum(1 for r in self._results if r.status == 'failed')}",
            "",
        ]
        for r in self._results:
            status = "OK" if r.status == "completed" else "FAIL"
            lines.append(f"  [{status}] {r.name} ({r.duration_seconds:.1f}s)")
            if r.metrics:
                for k, v in r.metrics.items():
                    val = f"{v:.4f}" if isinstance(v, float) else str(v)
                    lines.append(f"         {k}: {val}")
        lines.append(f"{'='*60}")
        text = "\n".join(lines)
        print(text)
        return text


def main():
    """CLI entry point for running experiments."""
    import argparse

    parser = argparse.ArgumentParser(description="Quantum ML Experiment Runner")
    parser.add_argument(
        "--suite",
        choices=["all", "architecture", "advantage", "scalability", "noise"],
        default="architecture",
        help="Experiment suite to run",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    runner = ExperimentRunner(results_dir=args.results_dir, seed=args.seed)

    if args.suite in ("all", "architecture"):
        from experiments.suite_1_architecture.exp_depth_scaling import (
            register_experiments,
        )
        register_experiments(runner)

    # Run all registered experiments
    for name in list(runner._registry.keys()):
        runner.run_single(name, {})

    runner.summary()
    runner.save_results()


if __name__ == "__main__":
    main()
