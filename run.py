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

def cmd_run_simulation(args: list[str]):
    logger.info('Running simulation...')
    options = parse_named_options(args)
    from app import multi_run
    multi_run(**options)

def cmd_run_server_scoring(args: list[str]):
    logger.info('Running server scoring simulation')
    options = parse_named_options(args)
    from app.services import SimulationService
    SimulationService.simulate_server_scoring(**options)

def cmd_run_client_scoring(args: list[str]) -> None:
    logger.info('Running client scoring simulation')
    options = parse_named_options(args)
    from app.services import SimulationService
    SimulationService.simulate_client_scoring(**options)

def cmd_run_decay(args: list[str]) -> None:
    logger.info('Running decay measurment')
    options = parse_named_options(args)
    from app.services import SimulationService
    SimulationService.sigma_decay_measurment(**options)

def cmd_run_defenses_simulation(args: list[str]):
    logger.info('Running scoring simulation...')
    options = parse_named_options(args)
    from app import simulate_defenses
    simulate_defenses(**options)

def cmd_group_data(args: list[str]):
    logger.info('Running data grouping...')
    options = parse_named_options(args)
    from app.services import SimulationService
    SimulationService.group_data(**options)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python run.py <command>\nAvailable commands: {', '.join(COMMANDS)}")
        sys.exit(1)

    command, *extra_args = sys.argv[1:]

    match command:
        case 'preprocess':      cmd_preprocess()
        case 'run-simulation':  cmd_run_simulation(extra_args)
        case 'client-scoring':  cmd_run_client_scoring(extra_args)
        case 'server-scoring':  cmd_run_server_scoring(extra_args)
        case 'run-decay':       cmd_run_decay(extra_args)
        case 'run-defenses':    cmd_run_defenses_simulation(extra_args)
        case 'group-data':      cmd_group_data(extra_args)
        case _:
            print(f"Unknown command: '{command}'\nAvailable commands: {', '.join(COMMANDS)}")
            sys.exit(1)