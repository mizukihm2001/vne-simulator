# utils/config_loader.py

import yaml
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file into a dictionary.

    Args:
        path: Path to the YAML file

    Returns:
        Parsed dictionary
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)
