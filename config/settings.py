# Imports
from dotenv import load_dotenv
import os
import torch

from config.logger import create_logger

load_dotenv()
logger = create_logger(__name__)

# Create config class
class BaseConfig:
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Using {DEVICE} as the device')

    # Data preprocessing
    INPUT_DATA_PATH: str = 'data/input'
    PROCESSED_DATA_PATH: str = 'data/processed'
    INPUT_DATA_FILENAME: str = 'Final_Energy_Dataset_with_weather.csv'
    NUMBER_CLIENTS: int = 20
    PICK_RANDOM_CLIENTS: bool = True

    # Data saving
    SAVE_DATA_PATH: str = 'save'

    # Data parameters
    LOOKBACK: int = 48
    NUM_FEATURES: int = 7 # weekday, tod_sin, tod_cos, temp, rhum, wspd, wdir

    # Simulation parameters
    SIM_MAX_ROUNDS: int = 20
    SIM_TOTAL_CLIENTS: int = 20
    SIM_BATCH_SIZE: int = 64
    SIM_LEARNING_RATE: float = 1e-3
    SIM_LOCAL_EPOCHS: int = 50
    SIM_CLIENTS_FRACTION: float = 0.5
    SIM_THREADED: bool = True

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