"""
Configuration Management
========================

YAML-based configuration with OmegaConf for type-safe, hierarchical configs.

Usage
-----
>>> from src.utils.config import load_config, get_config
>>> cfg = load_config("configs/models/vqc_config.yaml")
>>> print(cfg.model.n_qubits)
4
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from omegaconf import DictConfig, OmegaConf

# Global config instance
_CONFIG: Optional[DictConfig] = None

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()


def get_project_root() -> Path:
    """Get the project root directory."""
    return PROJECT_ROOT


def load_config(
    config_path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None,
    resolve: bool = True,
) -> DictConfig:
    """
    Load configuration from a YAML file.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML configuration file.
    overrides : dict, optional
        Dictionary of overrides to apply to the config.
    resolve : bool, default=True
        Whether to resolve interpolations in the config.

    Returns
    -------
    DictConfig
        Loaded configuration object.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.

    Examples
    --------
    >>> cfg = load_config("configs/models/vqc_config.yaml")
    >>> cfg = load_config("config.yaml", overrides={"model.n_qubits": 8})
    """
    config_path = Path(config_path)

    # Handle relative paths
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load base config
    cfg = OmegaConf.load(config_path)

    # Apply overrides
    if overrides:
        override_cfg = OmegaConf.create(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    # Resolve interpolations
    if resolve:
        OmegaConf.resolve(cfg)

    return cfg


def get_config() -> Optional[DictConfig]:
    """
    Get the global configuration instance.

    Returns
    -------
    DictConfig or None
        The global configuration, or None if not set.
    """
    global _CONFIG
    return _CONFIG


def set_config(cfg: DictConfig) -> None:
    """
    Set the global configuration instance.

    Parameters
    ----------
    cfg : DictConfig
        Configuration to set as global.
    """
    global _CONFIG
    _CONFIG = cfg


def create_config(config_dict: Dict[str, Any]) -> DictConfig:
    """
    Create a configuration from a dictionary.

    Parameters
    ----------
    config_dict : dict
        Dictionary to convert to config.

    Returns
    -------
    DictConfig
        Configuration object.
    """
    return OmegaConf.create(config_dict)


def merge_configs(*configs: DictConfig) -> DictConfig:
    """
    Merge multiple configurations.

    Later configs override earlier ones.

    Parameters
    ----------
    *configs : DictConfig
        Configurations to merge.

    Returns
    -------
    DictConfig
        Merged configuration.
    """
    return OmegaConf.merge(*configs)


def to_dict(cfg: DictConfig) -> Dict[str, Any]:
    """
    Convert configuration to a plain dictionary.

    Parameters
    ----------
    cfg : DictConfig
        Configuration to convert.

    Returns
    -------
    dict
        Plain dictionary representation.
    """
    return OmegaConf.to_container(cfg, resolve=True)


def save_config(cfg: DictConfig, path: Union[str, Path]) -> None:
    """
    Save configuration to a YAML file.

    Parameters
    ----------
    cfg : DictConfig
        Configuration to save.
    path : str or Path
        Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)


# Default configuration schema
DEFAULT_CONFIG = {
    "model": {
        "n_qubits": 4,
        "n_layers": 3,
        "entanglement": "full",
        "encoding": "angle",
        "measurement": "z",
    },
    "training": {
        "epochs": 100,
        "batch_size": 32,
        "optimizer": "adam",
        "learning_rate": 0.01,
        "early_stopping": {
            "enabled": True,
            "patience": 10,
            "min_delta": 1e-4,
        },
    },
    "data": {
        "dataset": "iris",
        "test_size": 0.2,
        "val_size": 0.1,
        "normalize": True,
        "seed": 42,
    },
    "noise": {
        "enabled": False,
        "type": "depolarizing",
        "probability": 0.01,
    },
    "logging": {
        "level": "INFO",
        "wandb": False,
        "tensorboard": True,
    },
    "paths": {
        "data": "${project_root}/data",
        "checkpoints": "${project_root}/checkpoints",
        "logs": "${project_root}/logs",
        "results": "${project_root}/results",
    },
}


def get_default_config() -> DictConfig:
    """
    Get the default configuration.

    Returns
    -------
    DictConfig
        Default configuration with all standard settings.
    """
    cfg = OmegaConf.create(DEFAULT_CONFIG)
    # Add project_root for interpolation
    OmegaConf.update(cfg, "project_root", str(PROJECT_ROOT))
    return cfg
