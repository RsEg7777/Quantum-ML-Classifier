"""
Quantum Kernel
==============

Base quantum kernel with fidelity-based kernel computation.
K(x, x') = |⟨φ(x)|φ(x')⟩|²

References
----------
Havlicek et al., "Supervised learning with quantum-enhanced feature spaces",
Nature 567, 209-212 (2019).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import cirq
import numpy as np
import sympy

from src.quantum.encodings.base import BaseEncoding
from src.quantum.encodings.angle_encoding import AngleEncoding


class QuantumKernel:
    """
    Quantum kernel using state fidelity.

    Computes K(x, x') = |⟨0|U†(x)U(x')|0⟩|² via simulation.

    Parameters
    ----------
    encoding : BaseEncoding
        Encoding scheme to use for feature mapping.
    n_qubits : int, optional
        Number of qubits (inferred from encoding if not given).

    Examples
    --------
    >>> from src.quantum.encodings import AngleEncoding
    >>> encoding = AngleEncoding(n_qubits=4)
    >>> kernel = QuantumKernel(encoding)
    >>> K = kernel.compute_kernel_matrix(X_train)
    """

    def __init__(
        self,
        encoding: Optional[BaseEncoding] = None,
        n_qubits: int = 4,
    ):
        if encoding is not None:
            self.encoding = encoding
            self.n_qubits = encoding.n_qubits
        else:
            self.encoding = AngleEncoding(n_qubits=n_qubits)
            self.n_qubits = n_qubits

        self.qubits = self.encoding.qubits
        self._simulator = cirq.Simulator()

    def _get_state_vector(self, data: np.ndarray) -> np.ndarray:
        """Get the state vector for a data point."""
        circuit = self.encoding.encode(data=data)
        result = self._simulator.simulate(circuit)
        return result.final_state_vector

    def compute_entry(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compute a single kernel entry K(x1, x2).

        Returns
        -------
        float
            |⟨φ(x1)|φ(x2)⟩|²
        """
        sv1 = self._get_state_vector(x1)
        sv2 = self._get_state_vector(x2)
        fidelity = np.abs(np.vdot(sv1, sv2)) ** 2
        return float(fidelity)

    def compute_kernel_matrix(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute the kernel (Gram) matrix.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            First dataset.
        Y : np.ndarray, shape (m, d), optional
            Second dataset. If None, computes K(X, X).

        Returns
        -------
        np.ndarray, shape (n, m)
            Kernel matrix.
        """
        if Y is None:
            Y = X

        n = len(X)
        m = len(Y)
        K = np.zeros((n, m))

        # Precompute state vectors
        sv_x = [self._get_state_vector(x) for x in X]
        sv_y = [self._get_state_vector(y) for y in Y] if Y is not X else sv_x

        for i in range(n):
            for j in range(m):
                K[i, j] = np.abs(np.vdot(sv_x[i], sv_y[j])) ** 2

        return K

    def as_sklearn_kernel(self):
        """
        Return a callable compatible with sklearn's SVC(kernel=...).

        Returns
        -------
        callable
            Function (X, Y) -> kernel matrix.
        """
        def kernel_fn(X, Y):
            return self.compute_kernel_matrix(X, Y)
        return kernel_fn
