# Imports
import sys

from config import create_logger

logger = create_logger(__name__)

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python run.py <command>")
        sys.exit(1)

    match sys.argv[1]:
        case 'preprocess':
            logger.info("Running data preprocessing...")
            from data import run_preprocessing
            run_preprocessing()
            
        case 'check':
            logger.info("Running checks...")
            from app.models import check_dataset, check_models
            check_models()
            check_dataset()

        case 'run-simulation':
            logger.info("Running simulation...")

        case _:
            print("Available commands : [preprocess, check-models]")