"""
Utilities Package
=================

Configuration, logging, reproducibility, and helper utilities.
"""

from src.utils.config import (
    create_config,
    get_config,
    get_default_config,
    get_project_root,
    load_config,
    merge_configs,
    save_config,
    set_config,
    to_dict,
)
from src.utils.logging import TrainingLogger, get_logger, setup_logging
from src.utils.reproducibility import (
    ReproducibilityContext,
    generate_seed_from_string,
    get_random_state,
    get_reproducibility_info,
    hash_array,
    save_reproducibility_info,
    set_seed,
)

__all__ = [
    # Config
    "load_config",
    "get_config",
    "set_config",
    "create_config",
    "merge_configs",
    "to_dict",
    "save_config",
    "get_default_config",
    "get_project_root",
    # Logging
    "setup_logging",
    "get_logger",
    "TrainingLogger",
    # Reproducibility
    "set_seed",
    "get_random_state",
    "generate_seed_from_string",
    "get_reproducibility_info",
    "save_reproducibility_info",
    "ReproducibilityContext",
    "hash_array",
]
