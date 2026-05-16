"""
Configuration management.

Loads YAML config files into a dot-accessible namespace.
All scripts accept --config to specify an alternate config file.
"""

import yaml
from pathlib import Path
from typing import Any


class Config:
    """Dot-accessible configuration namespace loaded from YAML."""

    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __repr__(self) -> str:
        return f"Config({vars(self)})"


def load_config(path: str = "configs/default.yaml") -> Config:
    """
    Load configuration from a YAML file.

    Args:
        path: Path to the YAML config file (relative to project root).

    Returns:
        Config object with dot-notation access to all settings.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            "Run scripts from the project root directory."
        )

    with open(config_path) as f:
        data = yaml.safe_load(f)

    return Config(data)
