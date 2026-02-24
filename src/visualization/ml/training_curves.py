"""
Training Curves Visualization
==============================

Plotting utilities for training/validation loss, accuracy,
and other metrics over epochs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np


def _get_plt():
    """Lazy import matplotlib to avoid import-time cost."""
    import matplotlib.pyplot as plt
    return plt


def plot_loss_curves(
    train_losses: Sequence[float],
    val_losses: Optional[Sequence[float]] = None,
    title: str = "Loss Curves",
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (8, 5),
    **kwargs,
) -> Any:
    """
    Plot training (and optionally validation) loss curves.

    Parameters
    ----------
    train_losses : sequence of float
        Training loss per epoch.
    val_losses : sequence of float, optional
        Validation loss per epoch.
    title : str
        Plot title.
    save_path : str or Path, optional
        If provided, save figure to this path.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=figsize)
    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, label="Train", linewidth=2)
    if val_losses is not None:
        ax.plot(epochs, val_losses, label="Validation", linewidth=2, linestyle="--")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig


def plot_accuracy_curves(
    train_acc: Sequence[float],
    val_acc: Optional[Sequence[float]] = None,
    title: str = "Accuracy Curves",
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (8, 5),
) -> Any:
    """Plot training (and optionally validation) accuracy curves."""
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=figsize)
    epochs = range(1, len(train_acc) + 1)

    ax.plot(epochs, train_acc, label="Train", linewidth=2)
    if val_acc is not None:
        ax.plot(epochs, val_acc, label="Validation", linewidth=2, linestyle="--")

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig


def plot_metric_comparison(
    metrics: Dict[str, Sequence[float]],
    title: str = "Model Comparison",
    ylabel: str = "Metric",
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (8, 5),
) -> Any:
    """
    Plot multiple named metric curves on the same axes.

    Parameters
    ----------
    metrics : dict
        Mapping from label -> sequence of values.
    """
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=figsize)

    for label, values in metrics.items():
        ax.plot(range(1, len(values) + 1), values, label=label, linewidth=2)

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (6, 5),
    cmap: str = "Blues",
) -> Any:
    """
    Plot a confusion matrix as a heatmap.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix of shape (n_classes, n_classes).
    class_names : list of str, optional
        Class labels.
    """
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    fig.colorbar(im, ax=ax)

    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(
                j, i, f"{cm[i, j]:.0f}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig
