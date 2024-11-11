import logging
import logging.config
import yaml

def setup_logger(name: str, config_file: str) -> logging.Logger:
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    logging.config.dictConfig(config)
    logger = logging.getLogger(name)
    return logger
