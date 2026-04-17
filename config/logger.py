# Imports
import logging
from datetime import datetime, timezone
from typing import Callable
import os

def create_logger(name: str, 
                  log_path: str = "logs", 
                  file_name: str | Callable = lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d")
                ) -> logging.Logger:
    """
    Creates a logger that logs to both a file and the console.

    Args:
        name (str): The name of the logger.
        log_path (str, optional): The path to the directory where log files will be stored. Defaults to "logs".
        file_name (str | callable, optional): The name of the log file or a callable that returns the name. Defaults to a function that returns the current date in UTC.
    Returns:
        logging.Logger: The created logger.
    """
    
    root_logger = logging.getLogger(name)
    root_logger.setLevel(logging.DEBUG)

    log_file = f'{log_path}/{file_name() if callable(file_name) else file_name}.log'
    # Create file if it doesn't exist
    if not os.path.exists(log_file):
        os.makedirs(log_path, exist_ok = True)
        with open(file = log_file, mode = 'w'):
            pass

    file_handler = logging.FileHandler(log_file)
    log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    return root_logger