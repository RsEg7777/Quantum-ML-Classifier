"""
Quantum Encoding Schemes
========================

Data encoding methods for mapping classical data to quantum states.

Available Encodings
-------------------
AngleEncoding
    Rotation-based encoding using RX, RY, RZ gates.
AmplitudeEncoding
    Encode data in quantum state amplitudes (logarithmic compression).
IQPEncoding
    Instantaneous Quantum Polynomial circuits.
BasisEncoding
    Computational basis state preparation.
HamiltonianEncoding
    Time-evolution based encoding.
"""

from src.quantum.encodings.amplitude_encoding import (
    AmplitudeEncoding,
    ApproximateAmplitudeEncoding,
    MottonenAmplitudeEncoding,
)
from src.quantum.encodings.angle_encoding import AngleEncoding, DenseAngleEncoding, HadamardAngleEncoding
from src.quantum.encodings.base import BaseEncoding, CompositeEncoding, RepeatedEncoding
from src.quantum.encodings.basis_encoding import BasisEncoding
from src.quantum.encodings.hamiltonian_encoding import HamiltonianEncoding
from src.quantum.encodings.iqp_encoding import IQPEncoding, PauliFeatureMapEncoding, ZFeatureMapEncoding

__all__ = [
    "BaseEncoding",
    "RepeatedEncoding",
    "CompositeEncoding",
    "AngleEncoding",
    "DenseAngleEncoding",
    "HadamardAngleEncoding",
    "AmplitudeEncoding",
    "MottonenAmplitudeEncoding",
    "ApproximateAmplitudeEncoding",
    "BasisEncoding",
    "IQPEncoding",
    "ZFeatureMapEncoding",
    "PauliFeatureMapEncoding",
    "HamiltonianEncoding",
]
