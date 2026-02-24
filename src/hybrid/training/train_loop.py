"""
Training Loop
=============

Main training orchestrator for hybrid quantum-classical models.
Supports TF/Keras training with early stopping, LR scheduling,
and custom quantum-aware callbacks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _ensure_tf():
    return __import__("tensorflow")


class TrainingConfig:
    """
    Training configuration.

    Parameters
    ----------
    epochs : int, default=100
        Number of training epochs.
    batch_size : int, default=32
        Batch size.
    learning_rate : float, default=0.01
        Initial learning rate.
    optimizer : str, default='adam'
        Optimizer name.
    early_stopping_patience : int, default=10
        Early stopping patience (0 to disable).
    min_delta : float, default=1e-4
        Minimum improvement for early stopping.
    """

    def __init__(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        optimizer: str = "adam",
        early_stopping_patience: int = 10,
        min_delta: float = 1e-4,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.early_stopping_patience = early_stopping_patience
        self.min_delta = min_delta


def get_callbacks(
    config: TrainingConfig,
    checkpoint_path: Optional[str] = None,
) -> List:
    """
    Create standard Keras callbacks.

    Parameters
    ----------
    config : TrainingConfig
        Training configuration.
    checkpoint_path : str, optional
        Path for model checkpoints.

    Returns
    -------
    list
        Keras callbacks.
    """
    tf = _ensure_tf()
    callbacks = []

    # Early stopping
    if config.early_stopping_patience > 0:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.early_stopping_patience,
                min_delta=config.min_delta,
                restore_best_weights=True,
                verbose=1,
            )
        )

    # LR reduction on plateau
    callbacks.append(
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=config.early_stopping_patience // 2 or 5,
            min_lr=1e-6,
            verbose=1,
        )
    )

    # Model checkpoint
    if checkpoint_path:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
                verbose=0,
            )
        )

    return callbacks


def train_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    config: Optional[TrainingConfig] = None,
    extra_callbacks: Optional[List] = None,
) -> Dict[str, Any]:
    """
    Train a Keras model with standard configuration.

    Parameters
    ----------
    model : tf.keras.Model
        Compiled model to train.
    X_train, y_train : np.ndarray
        Training data.
    X_val, y_val : np.ndarray, optional
        Validation data.
    config : TrainingConfig, optional
        Training configuration. Uses defaults if None.
    extra_callbacks : list, optional
        Additional Keras callbacks.

    Returns
    -------
    dict
        Training history and final metrics.
    """
    if config is None:
        config = TrainingConfig()

    validation_data = None
    if X_val is not None and y_val is not None:
        validation_data = (X_val, y_val)

    callbacks = get_callbacks(config)
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    history = model.fit(
        X_train,
        y_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=1,
    )

    result = {
        "history": history.history,
        "epochs_trained": len(history.history["loss"]),
        "final_train_loss": history.history["loss"][-1],
    }

    if "val_loss" in history.history:
        result["final_val_loss"] = history.history["val_loss"][-1]
        result["best_val_loss"] = min(history.history["val_loss"])

    if "accuracy" in history.history:
        result["final_train_acc"] = history.history["accuracy"][-1]
    if "val_accuracy" in history.history:
        result["best_val_acc"] = max(history.history["val_accuracy"])

    return result
