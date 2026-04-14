# Imports
from dotenv import load_dotenv
import os

from config.logger import create_logger

load_dotenv()
logger = create_logger(__name__)

# Create config class
class BaseConfig:
    # Data preprocessing
    INPUT_DATA_PATH: str = 'data/input'
    PROCESSED_DATA_PATH: str = 'data/processed'
    INPUT_DATA_FILENAME: str = 'Final_Energy_Dataset_with_weather.csv'
    NUMBER_CLIENTS: int = 20

    # Data parameters
    LOOKBACK: int = 48
    NUM_FEATURES: int = 8

def _get_config() -> BaseConfig:
    """
    Get config for the current environment.
    The environment is determined by the ENV environment variable. If ENV is not set, it defaults to 'test'.
    
    Returns:
        BaseConfig: The config for the current environment.
    """
    _config_map: dict[str, BaseConfig] = {
        'test': BaseConfig(),
    }

    env: str = os.getenv('ENV', 'test')
    _config: BaseConfig = _config_map.get(env, BaseConfig())
    logger.info(f"Using config for environment: {env}")
    return _config

config: BaseConfig = _get_config()