# Imports
import sys

from config import create_logger

logger = create_logger(__name__)

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python run.py <command>")
        sys.exit(1)

    match sys.argv[1]:
        case 'preprocess':
            logger.info("Running data preprocessing...")
            from data import run_preprocessing
            run_preprocessing()
            
        case 'check':
            logger.info("Running checks...")
            from app.models import check_dataset, check_models, check_client, check_server
            from app.attacking_models import check_malicious_client
            if 'models' in sys.argv:
                check_models()
            if 'dataset' in sys.argv:
                check_dataset()
            if 'client' in sys.argv:
                check_client()
            if 'malicious-client' in sys.argv:
                check_malicious_client()
            if 'server' in sys.argv:
                check_server()

        case 'run-simulation':
            logger.info("Running simulation...")
            if len(sys.argv) % 2 != 0:
                raise Exception('Each option should be named')

            options: dict = {
                sys.argv[2*_].replace('--', '') : sys.argv[2*_+1]
                for _ in range(1, len(sys.argv)//2)
            }

            from app import multi_run
            multi_run(**options)

        case 'test':
            logger.info('Running test simulation...')
            from app import simulate_clean, simulate_malicious_clients, simulate_attacked_server, simulate_attacked_and_malicious
            if 'clean' in sys.argv:
                simulate_clean()
            if 'malicious-client' in sys.argv:
                simulate_malicious_clients()
            if 'attacked-server' in sys.argv:
                simulate_attacked_server()
            if 'both' in sys.argv:
                simulate_attacked_and_malicious()

        case 'group-data':
            logger.info('Running data grouping...')
            if len(sys.argv) % 2 != 0:
                raise Exception('Each option should be named')

            options: dict = {
                sys.argv[2*_].replace('--', '') : sys.argv[2*_+1]
                for _ in range(1, len(sys.argv)//2)
            }

            from app import data_grouping
            data_grouping(**options)

        case 'show-results':
            logger.info('Show multirun results...')
            from app import show_simulation_results
            show_simulation_results('clean_run_grouped')

        case _:
            print("Available commands : [preprocess, check, run-simulation, test, group-data, show-results]")