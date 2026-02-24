"""
Configurable MLP
================

Multi-layer perceptron for use as classical encoder/decoder
in hybrid quantum-classical models.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _ensure_tf():
    return __import__("tensorflow")


class ClassicalMLP:
    """
    Configurable Multi-Layer Perceptron.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    hidden_layers : list of int
        Hidden layer sizes.
    output_dim : int
        Output dimension.
    activation : str, default='relu'
        Activation function for hidden layers.
    output_activation : str or None, default=None
        Activation for output layer (None, 'softmax', 'sigmoid').
    dropout_rate : float, default=0.0
        Dropout rate between layers.
    batch_norm : bool, default=False
        Whether to use batch normalization.

    Examples
    --------
    >>> mlp = ClassicalMLP(input_dim=4, hidden_layers=[64, 32], output_dim=3)
    >>> model = mlp.build()
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        output_dim: int,
        activation: str = "relu",
        output_activation: Optional[str] = None,
        dropout_rate: float = 0.0,
        batch_norm: bool = False,
    ):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.activation = activation
        self.output_activation = output_activation
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm

    def build(self):
        """
        Build and return the Keras Sequential model.

        Returns
        -------
        tf.keras.Model
            Compiled MLP model.
        """
        tf = _ensure_tf()

        layers = [tf.keras.layers.Input(shape=(self.input_dim,))]

        for i, units in enumerate(self.hidden_layers):
            layers.append(
                tf.keras.layers.Dense(units, name=f"hidden_{i}")
            )
            if self.batch_norm:
                layers.append(tf.keras.layers.BatchNormalization())
            layers.append(tf.keras.layers.Activation(self.activation))
            if self.dropout_rate > 0:
                layers.append(tf.keras.layers.Dropout(self.dropout_rate))

        layers.append(
            tf.keras.layers.Dense(
                self.output_dim,
                activation=self.output_activation,
                name="output",
            )
        )

        model = tf.keras.Sequential(layers)
        return model

    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary."""
        return {
            "input_dim": self.input_dim,
            "hidden_layers": self.hidden_layers,
            "output_dim": self.output_dim,
            "activation": self.activation,
            "output_activation": self.output_activation,
            "dropout_rate": self.dropout_rate,
            "batch_norm": self.batch_norm,
        }
