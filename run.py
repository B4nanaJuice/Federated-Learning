# Imports
import sys

from config import create_logger

logger = create_logger(__name__)

COMMANDS = ['preprocess', 'check', 'run-simulation', 'test', 'group-data', 'show-results']


def parse_named_options(args: list[str]) -> dict:
    """Parse --key value pairs from argument list."""
    if len(args) % 2 != 0:
        raise ValueError("Each option must be a --key value pair.")
    return {
        args[2 * i].lstrip('-'): args[2 * i + 1]
        for i in range(len(args) // 2)
    }


def cmd_preprocess():
    logger.info("Running data preprocessing...")
    from data import run_preprocessing
    run_preprocessing()


def cmd_check(args: list[str]):
    logger.info("Running checks...")
    from app.models import check_dataset, check_models, check_client, check_server
    from app.attacking_models import check_malicious_client
    from app.scoring import check_scoring_entity

    checks = {
        'models':          check_models,
        'dataset':         check_dataset,
        'client':          check_client,
        'malicious-client': check_malicious_client,
        'server':          check_server,
        'scoring':         check_scoring_entity
    }
    for flag, fn in checks.items():
        if flag in args:
            fn()


def cmd_run_simulation(args: list[str]):
    logger.info("Running simulation...")
    options = parse_named_options(args)
    from app import multi_run
    multi_run(**options)


def cmd_test(args: list[str]):
    logger.info("Running test simulation...")
    from app import simulate_clean, simulate_malicious_clients, simulate_attacked_server, simulate_attacked_and_malicious

    tests = {
        'clean':            simulate_clean,
        'malicious-client': simulate_malicious_clients,
        'attacked-server':  simulate_attacked_server,
        'both':             simulate_attacked_and_malicious,
    }
    for flag, fn in tests.items():
        if flag in args:
            fn()


def cmd_group_data(args: list[str]):
    logger.info("Running data grouping...")
    options = parse_named_options(args)
    from app import data_grouping
    data_grouping(**options)


def cmd_show_results():
    logger.info("Showing multirun results...")
    from app.plots import compare_loss, compare_MSE
    compare_MSE([
        'clean_run_grouped',
        '5%_clients_model_grouped',
        '20%_clients_model_grouped',
        'clients_gradient_inversion_grouped',
        'clients_gradient_amplification_grouped'
    ])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python run.py <command>\nAvailable commands: {', '.join(COMMANDS)}")
        sys.exit(1)

    command, *extra_args = sys.argv[1:]

    match command:
        case 'preprocess':      cmd_preprocess()
        case 'check':           cmd_check(extra_args)
        case 'run-simulation':  cmd_run_simulation(extra_args)
        case 'test':            cmd_test(extra_args)
        case 'group-data':      cmd_group_data(extra_args)
        case 'show-results':    cmd_show_results()
        case _:
            print(f"Unknown command: '{command}'\nAvailable commands: {', '.join(COMMANDS)}")
            sys.exit(1)