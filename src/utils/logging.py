"""
Logging Utilities
=================

Structured logging with Loguru and Rich formatting.

Usage
-----
>>> from src.utils.logging import get_logger, setup_logging
>>> setup_logging(level="DEBUG")
>>> logger = get_logger(__name__)
>>> logger.info("Training started", epoch=1, loss=0.5)
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from loguru import logger

# Remove default handler
logger.remove()

# Default format
DEFAULT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

# Simple format for console
SIMPLE_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<level>{message}</level>"
)

# File format (no colors)
FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss} | "
    "{level: <8} | "
    "{name}:{function}:{line} | "
    "{message}"
)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    rotation: str = "10 MB",
    retention: str = "1 week",
    format: Optional[str] = None,
    colorize: bool = True,
    serialize: bool = False,
) -> None:
    """
    Setup logging configuration.

    Parameters
    ----------
    level : str, default="INFO"
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    log_file : str or Path, optional
        Path to log file. If None, logs only to console.
    rotation : str, default="10 MB"
        When to rotate the log file.
    retention : str, default="1 week"
        How long to keep old log files.
    format : str, optional
        Custom log format. Uses default if None.
    colorize : bool, default=True
        Whether to colorize console output.
    serialize : bool, default=False
        Whether to serialize logs as JSON.

    Examples
    --------
    >>> setup_logging(level="DEBUG")
    >>> setup_logging(level="INFO", log_file="logs/training.log")
    """
    # Remove existing handlers
    logger.remove()

    # Console handler
    console_format = format or (SIMPLE_FORMAT if colorize else FILE_FORMAT)
    logger.add(
        sys.stderr,
        format=console_format,
        level=level,
        colorize=colorize,
        serialize=serialize,
    )

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            str(log_path),
            format=FILE_FORMAT,
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip",
            serialize=serialize,
        )


def get_logger(name: Optional[str] = None) -> "logger":
    """
    Get a logger instance.

    Parameters
    ----------
    name : str, optional
        Logger name (typically __name__).

    Returns
    -------
    logger
        Loguru logger instance.

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.info("Hello, world!")
    """
    if name:
        return logger.bind(name=name)
    return logger


class TrainingLogger:
    """
    Specialized logger for training loops.

    Provides structured logging for epochs, batches, and metrics.

    Parameters
    ----------
    name : str
        Experiment or model name.
    log_dir : str or Path, optional
        Directory for log files.

    Examples
    --------
    >>> tlogger = TrainingLogger("vqc_iris")
    >>> tlogger.log_epoch(epoch=1, train_loss=0.5, val_loss=0.6, train_acc=0.8)
    >>> tlogger.log_batch(batch=10, loss=0.45)
    """

    def __init__(
        self,
        name: str,
        log_dir: Optional[Union[str, Path]] = None,
    ):
        self.name = name
        self.start_time = datetime.now()
        self._logger = get_logger(name)

        if log_dir:
            log_path = Path(log_dir) / f"{name}_{self.start_time:%Y%m%d_%H%M%S}.log"
            setup_logging(log_file=log_path)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        train_acc: Optional[float] = None,
        val_acc: Optional[float] = None,
        lr: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Log epoch metrics."""
        metrics = {
            "epoch": epoch,
            "train_loss": f"{train_loss:.4f}",
        }
        if val_loss is not None:
            metrics["val_loss"] = f"{val_loss:.4f}"
        if train_acc is not None:
            metrics["train_acc"] = f"{train_acc:.4f}"
        if val_acc is not None:
            metrics["val_acc"] = f"{val_acc:.4f}"
        if lr is not None:
            metrics["lr"] = f"{lr:.2e}"
        metrics.update(kwargs)

        msg = " | ".join(f"{k}={v}" for k, v in metrics.items())
        self._logger.info(f"Epoch {epoch}: {msg}")

    def log_batch(
        self,
        batch: int,
        loss: float,
        **kwargs: Any,
    ) -> None:
        """Log batch metrics (debug level)."""
        metrics = {"batch": batch, "loss": f"{loss:.4f}"}
        metrics.update(kwargs)

        msg = " | ".join(f"{k}={v}" for k, v in metrics.items())
        self._logger.debug(f"Batch: {msg}")

    def log_metrics(self, metrics: Dict[str, Any], prefix: str = "") -> None:
        """Log arbitrary metrics."""
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        formatted = {
            k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics.items()
        }
        msg = " | ".join(f"{k}={v}" for k, v in formatted.items())
        self._logger.info(msg)

    def log_config(self, config: Dict[str, Any]) -> None:
        """Log configuration."""
        self._logger.info(f"Configuration: {config}")

    def log_model_summary(self, model: Any) -> None:
        """Log model summary."""
        if hasattr(model, "summary"):
            self._logger.info("Model Summary:")
            # Capture summary to string
            summary_lines = []
            model.summary(print_fn=lambda x: summary_lines.append(x))
            for line in summary_lines:
                self._logger.info(line)

    def success(self, message: str) -> None:
        """Log success message."""
        self._logger.success(message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self._logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message."""
        self._logger.error(message)


# Initialize with default settings
setup_logging()
