from typing import Any

import yaml


def load_config(config_path: str) -> dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Configuration as a dictionary.
    """
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config
