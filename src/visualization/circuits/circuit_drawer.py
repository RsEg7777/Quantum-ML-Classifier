"""
Circuit Drawer
==============

Utilities for rendering quantum circuits to text, SVG, or matplotlib.
Wraps Cirq's built-in drawing with convenience helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import cirq


def circuit_to_text(
    circuit: cirq.Circuit,
    qubit_order: Optional[Any] = None,
) -> str:
    """
    Render circuit as a text diagram.

    Parameters
    ----------
    circuit : cirq.Circuit
        Circuit to render.
    qubit_order : optional
        Qubit ordering for display.

    Returns
    -------
    str
        Text diagram.
    """
    if qubit_order is not None:
        return circuit.to_text_diagram(qubit_order=qubit_order)
    return circuit.to_text_diagram()


def circuit_to_svg(
    circuit: cirq.Circuit,
    save_path: Optional[Union[str, Path]] = None,
) -> str:
    """
    Render circuit as an SVG string.

    Parameters
    ----------
    circuit : cirq.Circuit
        Circuit to render.
    save_path : str or Path, optional
        If provided, write SVG to this file.

    Returns
    -------
    str
        SVG markup.
    """
    svg = cirq.contrib.svg.SVGCircuit(circuit)._repr_svg_()

    if save_path:
        Path(save_path).write_text(svg, encoding="utf-8")

    return svg


def circuit_to_matplotlib(
    circuit: cirq.Circuit,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (12, 4),
) -> Any:
    """
    Render circuit diagram using matplotlib.

    Falls back to text diagram rendered as a monospaced figure.

    Parameters
    ----------
    circuit : cirq.Circuit
        Circuit to render.
    title : str, optional
        Figure title.
    save_path : str or Path, optional
        Save path.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    text_diagram = circuit.to_text_diagram()

    ax.text(
        0.02, 0.98, text_diagram,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
    )
    ax.axis("off")

    if title:
        ax.set_title(title)

    fig.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig


def print_circuit_summary(circuit: cirq.Circuit) -> str:
    """
    Print a compact summary of circuit structure.

    Returns
    -------
    str
        Multi-line summary.
    """
    n_qubits = len(circuit.all_qubits())
    depth = len(circuit)
    n_ops = sum(1 for _ in circuit.all_operations())

    lines = [
        f"Qubits: {n_qubits}",
        f"Depth:  {depth}",
        f"Gates:  {n_ops}",
        "",
        circuit.to_text_diagram(),
    ]
    return "\n".join(lines)
