"""
Quantum Circuit Architectures
=============================

Variational quantum circuits for hybrid quantum-classical ML.

Available Circuits
------------------
VQC
    Variational Quantum Classifier - configurable ansatz.
QCNN
    Quantum Convolutional Neural Network.
QAOA
    QAOA-inspired circuits for feature learning.
"""

from src.quantum.circuits.qaoa import QAOA, DataReuploadingQAOA, QAOALayer
from src.quantum.circuits.qcnn import QCNN, ConvolutionalLayer, FullyConnectedLayer, PoolingLayer
from src.quantum.circuits.qgnn import QGNN, QGNNLayer
from src.quantum.circuits.qrnn import QRNN, QRNNCell
from src.quantum.circuits.tensor_networks import MPSAnsatz, TTNAnsatz
from src.quantum.circuits.vqc import VQC, HardwareEfficientAnsatz, StronglyEntanglingAnsatz

__all__ = [
    # VQC
    "VQC",
    "HardwareEfficientAnsatz",
    "StronglyEntanglingAnsatz",
    # QCNN
    "QCNN",
    "ConvolutionalLayer",
    "PoolingLayer",
    "FullyConnectedLayer",
    # QAOA
    "QAOA",
    "QAOALayer",
    "DataReuploadingQAOA",
    # Tensor Networks
    "MPSAnsatz",
    "TTNAnsatz",
    # QRNN/QGNN
    "QRNN",
    "QRNNCell",
    "QGNN",
    "QGNNLayer",
]
