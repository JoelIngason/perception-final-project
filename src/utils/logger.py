import logging
import logging.config
from pathlib import Path

import yaml


def setup_logger(name: str, config_file: str) -> logging.Logger:
    with open(config_file) as f:
        config = yaml.safe_load(f)

    # Extract the log file path from the configuration
    log_file = config.get("handlers", {}).get("file", {}).get("filename", "")

    if log_file:
        log_dir = Path(log_file).parent

        # If the directory is relative, make it relative to the current script
        if not Path(log_dir).is_absolute():
            script_dir = Path(__file__).resolve().parent
            log_dir = script_dir / log_dir

        # Create the directory if it doesn't exist
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        # Update the config with the absolute path
        config["handlers"]["file"]["filename"] = str(log_dir / "app.log")

    # Apply the logging configuration
    logging.config.dictConfig(config)
    return logging.getLogger(name)
