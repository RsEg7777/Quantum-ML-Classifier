"""
Circuit Resource Estimation
============================

Utilities for counting gates, estimating depth, and profiling
quantum resource requirements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import cirq
import numpy as np


@dataclass
class ResourceProfile:
    """Summary of quantum resources used by a circuit."""

    n_qubits: int = 0
    depth: int = 0
    total_gates: int = 0
    single_qubit_gates: int = 0
    two_qubit_gates: int = 0
    multi_qubit_gates: int = 0
    n_parameters: int = 0
    gate_counts: Dict[str, int] = field(default_factory=dict)
    t_count: int = 0  # relevant for fault-tolerant cost

    def to_dict(self) -> dict:
        """Serialise to dictionary."""
        return {
            "n_qubits": self.n_qubits,
            "depth": self.depth,
            "total_gates": self.total_gates,
            "single_qubit_gates": self.single_qubit_gates,
            "two_qubit_gates": self.two_qubit_gates,
            "multi_qubit_gates": self.multi_qubit_gates,
            "n_parameters": self.n_parameters,
            "gate_counts": self.gate_counts,
            "t_count": self.t_count,
        }

    def __str__(self) -> str:
        lines = [
            "=== Circuit Resource Profile ===",
            f"  Qubits        : {self.n_qubits}",
            f"  Depth         : {self.depth}",
            f"  Total gates   : {self.total_gates}",
            f"  1Q gates      : {self.single_qubit_gates}",
            f"  2Q gates      : {self.two_qubit_gates}",
            f"  Multi-Q gates : {self.multi_qubit_gates}",
            f"  Parameters    : {self.n_parameters}",
            f"  T-count       : {self.t_count}",
        ]
        if self.gate_counts:
            lines.append("  Gate breakdown:")
            for gate, count in sorted(
                self.gate_counts.items(), key=lambda x: -x[1]
            ):
                lines.append(f"    {gate:20s} : {count}")
        return "\n".join(lines)


def estimate_resources(circuit: cirq.Circuit) -> ResourceProfile:
    """
    Analyse a Cirq circuit and return a resource profile.

    Parameters
    ----------
    circuit : cirq.Circuit
        The circuit to analyse.

    Returns
    -------
    ResourceProfile
    """
    profile = ResourceProfile()
    profile.n_qubits = len(circuit.all_qubits())
    profile.depth = len(circuit)

    gate_counts: Dict[str, int] = {}
    n_params = 0

    for moment in circuit:
        for op in moment.operations:
            n_q = len(op.qubits)
            profile.total_gates += 1

            if n_q == 1:
                profile.single_qubit_gates += 1
            elif n_q == 2:
                profile.two_qubit_gates += 1
            else:
                profile.multi_qubit_gates += 1

            gate_name = type(op.gate).__name__ if op.gate else str(op)
            gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1

            # T-count: ZPowGate with exponent 0.25 is a T gate
            if hasattr(op.gate, 'exponent') and hasattr(op.gate, '_num_qubits_'):
                if op.gate._num_qubits_() == 1 and getattr(op.gate, 'exponent', None) == 0.25:
                    profile.t_count += 1

            # Count parameterised gates
            if cirq.is_parameterized(op):
                n_params += len(cirq.parameter_names(op))

    profile.gate_counts = gate_counts
    profile.n_parameters = n_params
    return profile


def compare_circuits(
    circuits: Dict[str, cirq.Circuit],
) -> Dict[str, ResourceProfile]:
    """
    Compare resource profiles of multiple circuits.

    Parameters
    ----------
    circuits : dict
        Mapping from label -> circuit.

    Returns
    -------
    dict
        Mapping from label -> ResourceProfile.
    """
    return {name: estimate_resources(circ) for name, circ in circuits.items()}


def estimate_execution_time(
    profile: ResourceProfile,
    single_gate_time_ns: float = 25.0,
    two_gate_time_ns: float = 100.0,
    measurement_time_ns: float = 500.0,
    n_shots: int = 1024,
) -> float:
    """
    Rough wall-clock estimate in seconds (assumes sequential execution).

    Parameters
    ----------
    profile : ResourceProfile
        Resource profile to estimate from.
    single_gate_time_ns : float
        Time per single-qubit gate in nanoseconds.
    two_gate_time_ns : float
        Time per two-qubit gate in nanoseconds.
    measurement_time_ns : float
        Time per measurement in nanoseconds.
    n_shots : int
        Number of circuit repetitions.

    Returns
    -------
    float
        Estimated time in seconds.
    """
    gate_time = (
        profile.single_qubit_gates * single_gate_time_ns
        + profile.two_qubit_gates * two_gate_time_ns
        + profile.n_qubits * measurement_time_ns  # one meas per qubit
    )
    total_ns = gate_time * n_shots
    return total_ns * 1e-9
