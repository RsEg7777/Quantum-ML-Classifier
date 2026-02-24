"""
Measurement Error Mitigation
=============================

Corrects measurement errors using calibration-based confusion
matrix inversion.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import cirq
import numpy as np


class MeasurementErrorMitigator:
    """
    Mitigates readout errors using an inverse confusion matrix.

    The calibration procedure prepares each computational basis state
    and measures to estimate the confusion matrix A, where A[i][j] =
    P(measure i | prepared j). Corrected counts are obtained via
    A^{-1} @ noisy_counts.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    calibration_shots : int, default=8192
        Shots per calibration circuit.
    method : str, default='inverse'
        Correction method: 'inverse' (matrix inversion) or 'least_squares'
        (constrained optimisation ensuring non-negative probabilities).
    """

    def __init__(
        self,
        n_qubits: int,
        calibration_shots: int = 8192,
        method: str = "inverse",
    ):
        self.n_qubits = n_qubits
        self.calibration_shots = calibration_shots
        self.method = method
        self._confusion_matrix: Optional[np.ndarray] = None
        self._inverse_matrix: Optional[np.ndarray] = None

    @property
    def confusion_matrix(self) -> Optional[np.ndarray]:
        """Return the calibrated confusion matrix (or None if uncalibrated)."""
        return self._confusion_matrix

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def generate_calibration_circuits(
        self, qubits: Sequence[cirq.Qid]
    ) -> List[cirq.Circuit]:
        """
        Generate 2^n calibration circuits, one per basis state.

        Parameters
        ----------
        qubits : sequence of cirq.Qid
            Qubits to calibrate.

        Returns
        -------
        list of cirq.Circuit
        """
        n = len(qubits)
        circuits = []
        for state_idx in range(2 ** n):
            circuit = cirq.Circuit()
            bits = format(state_idx, f"0{n}b")
            for bit, qubit in zip(bits, qubits):
                if bit == "1":
                    circuit.append(cirq.X(qubit))
            circuit.append(cirq.measure(*qubits, key="result"))
            circuits.append(circuit)
        return circuits

    def calibrate_from_counts(
        self,
        calibration_counts: List[Dict[str, int]],
    ) -> np.ndarray:
        """
        Build confusion matrix from calibration measurement counts.

        Parameters
        ----------
        calibration_counts : list of dict
            Each element is {bitstring: count} for the corresponding
            basis state preparation.

        Returns
        -------
        np.ndarray
            Confusion matrix of shape (2^n, 2^n).
        """
        n_states = 2 ** self.n_qubits
        A = np.zeros((n_states, n_states))

        for prep_idx, counts in enumerate(calibration_counts):
            total = sum(counts.values())
            for bitstring, count in counts.items():
                meas_idx = int(bitstring, 2)
                A[meas_idx, prep_idx] = count / total

        self._confusion_matrix = A
        self._compute_inverse()
        return A

    def calibrate(
        self,
        qubits: Sequence[cirq.Qid],
        sampler: cirq.Sampler,
    ) -> np.ndarray:
        """
        Run calibration circuits and build the confusion matrix.

        Parameters
        ----------
        qubits : sequence of cirq.Qid
            Qubits to calibrate.
        sampler : cirq.Sampler
            Simulator or hardware sampler.

        Returns
        -------
        np.ndarray
            Calibrated confusion matrix.
        """
        circuits = self.generate_calibration_circuits(qubits)
        calibration_counts = []

        for circ in circuits:
            result = sampler.run(circ, repetitions=self.calibration_shots)
            bits_array = result.measurements["result"]
            counts: Dict[str, int] = {}
            for row in bits_array:
                key = "".join(str(b) for b in row)
                counts[key] = counts.get(key, 0) + 1
            calibration_counts.append(counts)

        return self.calibrate_from_counts(calibration_counts)

    # ------------------------------------------------------------------
    # Correction
    # ------------------------------------------------------------------

    def _compute_inverse(self) -> None:
        """Compute the inverse (or pseudo-inverse) of the confusion matrix."""
        if self._confusion_matrix is None:
            raise RuntimeError("Must calibrate before correcting.")
        try:
            self._inverse_matrix = np.linalg.inv(self._confusion_matrix)
        except np.linalg.LinAlgError:
            self._inverse_matrix = np.linalg.pinv(self._confusion_matrix)

    def correct_counts(
        self,
        counts: Dict[str, int],
    ) -> Dict[str, float]:
        """
        Apply error mitigation to raw measurement counts.

        Parameters
        ----------
        counts : dict
            Raw {bitstring: count} dictionary.

        Returns
        -------
        dict
            Corrected {bitstring: count} dictionary.
        """
        if self._inverse_matrix is None:
            raise RuntimeError("Must calibrate before correcting.")

        n_states = 2 ** self.n_qubits
        total = sum(counts.values())

        # Build probability vector from counts
        prob_vec = np.zeros(n_states)
        for bitstring, count in counts.items():
            idx = int(bitstring, 2)
            prob_vec[idx] = count / total

        if self.method == "inverse":
            corrected = self._inverse_matrix @ prob_vec
        elif self.method == "least_squares":
            corrected = self._constrained_correction(prob_vec)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Clip and re-normalise
        corrected = np.clip(corrected, 0, None)
        corrected_sum = corrected.sum()
        if corrected_sum > 0:
            corrected /= corrected_sum

        # Convert back to counts
        result = {}
        for idx in range(n_states):
            if corrected[idx] > 1e-10:
                bitstring = format(idx, f"0{self.n_qubits}b")
                result[bitstring] = corrected[idx] * total
        return result

    def correct_probabilities(
        self,
        prob_vec: np.ndarray,
    ) -> np.ndarray:
        """
        Correct a probability vector directly.

        Parameters
        ----------
        prob_vec : np.ndarray
            Raw probability vector of length 2^n.

        Returns
        -------
        np.ndarray
            Corrected probability vector.
        """
        if self._inverse_matrix is None:
            raise RuntimeError("Must calibrate before correcting.")

        if self.method == "least_squares":
            corrected = self._constrained_correction(prob_vec)
        else:
            corrected = self._inverse_matrix @ prob_vec

        corrected = np.clip(corrected, 0, None)
        s = corrected.sum()
        if s > 0:
            corrected /= s
        return corrected

    def _constrained_correction(self, prob_vec: np.ndarray) -> np.ndarray:
        """Non-negative least-squares correction."""
        from scipy.optimize import nnls

        corrected, _ = nnls(self._confusion_matrix, prob_vec)
        return corrected
