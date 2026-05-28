# Imports
import sys

from config import create_logger

logger = create_logger(__name__)

COMMANDS = ['preprocess', 'check', 'run-simulation', 'test', 'group-data', 'show-results']


def parse_named_options(args: list[str]) -> dict:
    """Parse --key value pairs from argument list."""
    if len(args) % 2 != 0:
        raise ValueError('Each option must be a --key value pair.')
    return {
        args[2 * i].lstrip('-'): args[2 * i + 1]
        for i in range(len(args) // 2)
    }


def cmd_preprocess():
    logger.info('Running data preprocessing...')
    from data import run_preprocessing
    iid: bool = sys.argv[-1] == 'iid'
    run_preprocessing(iid = iid)


def cmd_check(args: list[str]):
    logger.info('Running checks...')
    from app.models import check_dataset, check_models, check_client, check_server
    from app.attacking_models import check_malicious_client
    from app.scoring import check_scoring_entity, check_scoring_server, check_scoring_client
    from app.degraded import check_network, check_multiline_client

    checks = {
        'models':           check_models,
        'dataset':          check_dataset,
        'client':           check_client,
        'malicious-client': check_malicious_client,
        'server':           check_server,
        'scoring':          check_scoring_entity,
        'scoring-server':   check_scoring_server,
        'scoring-client':   check_scoring_client,
        'network':          check_network,
        'ml_client':        check_multiline_client
    }
    for flag, fn in checks.items():
        if flag in args:
            fn()


def cmd_run_simulation(args: list[str]):
    logger.info('Running simulation...')
    options = parse_named_options(args)
    from app import multi_run
    multi_run(**options)

def cmd_run_scoring_simulation(args: list[str]):
    logger.info('Running scoring simulation...')
    options = parse_named_options(args)
    from app.services import SimulationService
    SimulationService.sigma_measurment(**options)

def cmd_run_defenses_simulation(args: list[str]):
    logger.info('Running scoring simulation...')
    options = parse_named_options(args)
    from app import simulate_defenses
    simulate_defenses(**options)

def cmd_test(args: list[str]):
    logger.info('Running test simulation...')
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
    logger.info('Running data grouping...')
    options = parse_named_options(args)
    from app import data_grouping
    data_grouping(**options)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python run.py <command>\nAvailable commands: {', '.join(COMMANDS)}")
        sys.exit(1)

    command, *extra_args = sys.argv[1:]

    match command:
        case 'preprocess':      cmd_preprocess()
        case 'check':           cmd_check(extra_args)
        case 'run-simulation':  cmd_run_simulation(extra_args)
        case 'run-scoring':     cmd_run_scoring_simulation(extra_args)
        case 'run-defenses':    cmd_run_defenses_simulation(extra_args)
        case 'test':            cmd_test(extra_args)
        case 'group-data':      cmd_group_data(extra_args)
        case _:
            print(f"Unknown command: '{command}'\nAvailable commands: {', '.join(COMMANDS)}")
            sys.exit(1)