import logging
import logging.config
from pathlib import Path

import yaml


def setup_logger(name: str, config_file: str) -> logging.Logger:
    """
    Set up a logger based on a YAML configuration file.

    Args:
        name (str): Name of the logger.
        config_file (str): Path to the YAML configuration file.

    Returns:
        logging.Logger: Configured logger instance.

    """
    with open(config_file) as f:
        config = yaml.safe_load(f)

    # Ensure log directory exists
    log_file = Path(config["handlers"]["file"]["filename"])
    if not log_file.is_absolute():
        # Make it relative to the project root
        project_root = Path(__file__).resolve().parents[2]
        log_file = project_root / log_file
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Update the config with the absolute path
    config["handlers"]["file"]["filename"] = str(log_file)

    # Apply the logging configuration
    logging.config.dictConfig(config)
    return logging.getLogger(name)
