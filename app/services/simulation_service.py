# Imports
import os
import json
import numpy as np
from glob import glob
from typing import List, Dict

from config import create_logger, config
from app.models import NormalMLP, Client
from app.scoring import ScoringMetric, ScoringServer, ScoringClient
from app.attacking_models import AttackedServer, MaliciousClient

logger = create_logger(__name__)

# Static class with methods
class SimulationService:

    # Scoring-based simulation
    @staticmethod
    def sigma_measurment(*args, **options) -> None:
        
        # Simulation parameters from options
        ## Overall parameters
        save_filename: str = options.get('save-filename', 'scoring')

        ## Server parameters
        server_max_rounds: int = 10

        ## Scoring parameters
        scoring_metric: ScoringMetric = {
            'dataset': ScoringMetric.DATASET,
            'distance': ScoringMetric.DISTANCE,
            'distribution': ScoringMetric.DISTRIBUTION,
            'similarity': ScoringMetric.SIMILARITY
        }.get(options.get('metric', 'distance'))
        scoring_threshold: float = .4
        scoring_sigmas: List[float] = np.linspace(0.01, 1, 21).tolist()

        ## Client parameters
        client_count: int = 10
        client_epochs: int = 15
        client_batch_size: int = 128
        client_lr: float = 1e-3
        client_fraction: float = .5

        # Simulation results
        rejected: List[int] = []

        # Simulation
        for sigma in scoring_sigmas:

            logger.info(f'Starting simulation for sigma = {sigma}')
                
            server: ScoringServer = ScoringServer(
                global_model = NormalMLP(),
                max_rounds = server_max_rounds,
                metric = scoring_metric,
                threshold = scoring_threshold,
                metric_parameters = {'sigma': sigma}
            )

            for _ in range(client_count):
                server.register_client(
                    Client(
                        client_id = _+1,
                        model = NormalMLP(),
                        local_epochs = client_epochs,
                        batch_size = client_batch_size,
                        learning_rate = client_lr
                    )
                )
            
            server.run(client_fraction = client_fraction)

            rejected.append(server.rejected_models)
            logger.info(f'Rejected models for sigma : {sigma} = {server.rejected_models}')

        logger.info(f'rejected models: {rejected}')
        logger.info('Saving data to file...')

        # Save data to file
        with open(f'{config.SAVE_DATA_PATH}/sigma_testing/{save_filename}.json', mode = 'w', encoding = 'utf-8') as f:
            f.write(json.dumps(
                {
                    'parameters': scoring_sigmas,
                    'rejected': rejected
                },
                indent = 4
            ))
        return

    # Simulate client scoring
    @staticmethod
    def simulate_client_scoring(*args, **options) -> None:
        
        # Simulation parameters from options
        ## Overall parameters
        run_count: int = 10
        save_filename: str = options.get('save-filename', 'scoring')

        ## Attack parameters
        attack_partial: bool = options.get('partial', 'false').lower() == 'true'

        ## Scoring parameters
        scoring_metric: ScoringMetric = ScoringMetric[options.get('metric', 'distance').upper()]
        scoring_threshold: float = .4
        scoring_parameters: Dict[str, float] | None = {
            ScoringMetric.DISTANCE: {'sigma': 6.8},
            ScoringMetric.DATASET: {'sigma': .3}
        }.get(scoring_metric)

        ## Server parameters
        server_max_rounds: int = 20

        ## Client parameters
        client_count: int = 20
        client_epochs: int = 15
        client_batch_size: int = 128
        client_lr: float = 1e-3
        client_fraction: float = .5

        for run in range(run_count):
            
            # Create Attacked server
            server: AttackedServer = AttackedServer(
                global_model = NormalMLP(),
                max_rounds = server_max_rounds,
                partial_attack = attack_partial,
                attack_rate = lambda x: x in [5, 6, 7, 17, 18, 19] if attack_partial else x == 19
            )

            for _ in range(client_count):
                server.register_client(
                    ScoringClient(
                        client_id = _+1,
                        model = NormalMLP(),
                        metric = scoring_metric,
                        threshold = scoring_threshold,
                        metric_parameters = scoring_parameters,
                        local_epochs = client_epochs,
                        batch_size = client_batch_size,
                        learning_rate = client_lr
                    )
                )

            server.run(client_fraction = client_fraction)

            server.run_test(dataset_index = 5, days_count = 5)
            server.save_metrics(f'{save_filename}_{run}')
        return
    
    # Simulate server scoring
    @staticmethod
    def simulate_server_scoring(*args, **options) -> None:

        # Simulation parameters from options
        ## Overall parameters
        run_count: int = 10
        save_filename: str = options.get('save-filename', 'scoring')

        ## Attack parameters
        malicious_percentage: float = float(options.get('malicious', 0))
        assert 0 <= malicious_percentage <= 100

        ## Scoring parameters
        scoring_metric: ScoringMetric = ScoringMetric[options.get('metric', 'distance').upper()]
        scoring_threshold: float = .4
        scoring_parameters: Dict[str, float] | None = {
            ScoringMetric.DISTANCE: {'sigma': 6.8},
            ScoringMetric.DATASET: {'sigma': .3}
        }.get(scoring_metric)

        ## Server parameters
        server_max_rounds: int = 20

        ## Client parameters
        client_count: int = 20
        client_epochs: int = 15
        client_batch_size: int = 128
        client_lr: float = 1e-3
        client_fraction: float = .5

        for run in range(run_count):
            
            # Create Scoring server
            server: ScoringServer = ScoringServer(
                global_model = NormalMLP(),
                max_rounds = server_max_rounds,
                metric = scoring_metric,
                threshold = scoring_threshold,
                metric_parameters = scoring_parameters,
            )

            # Add Malicious clients
            _ = 1
            while _ <= int(client_count * malicious_percentage / 100):
                server.register_client(
                    MaliciousClient(
                        client_id = _,
                        model = NormalMLP(),
                        local_epochs = client_epochs,
                        batch_size = client_batch_size,
                        learning_rate = client_lr,
                        attack_rate = 1
                    )
                )
                _ += 1

            while _ <= client_count:
                server.register_client(
                    Client(
                        client_id = _,
                        model = NormalMLP(),
                        local_epochs = client_epochs,
                        batch_size = client_batch_size,
                        learning_rate = client_lr,
                    )
                )
                _ += 1

            server.run(client_fraction = client_fraction)

            server.run_test(dataset_index = 5, days_count = 5)
            server.save_metrics(f'{save_filename}_{run}')
        return

    # Simulate defenses
    @staticmethod
    def simulate_defense(*args, **options) -> None:
        pass

    # Method for grouping data
    @staticmethod
    def group_data(**options) -> Dict[str, any]:

        save_filename: str = options.get('save-filename', 'run').replace(' ', '_')
        logger.info(f'Beginning data grouping for files {save_filename}.')

        run_count: int = len(glob(f'{config.SAVE_DATA_PATH}/{save_filename}*'))
        logger.info(f'Found {run_count} files.')
        
        metrics_name: List[str] = ['MAE', 'MSE', 'RMSE']
        columns: List[str] = ['load', 'pv', 'net']

        output_data: Dict[str, any] = {
            'training_loss': None,
            'MAE': {k: [] for k in ['load', 'pv', 'net']},
            'MSE': {k: [] for k in ['load', 'pv', 'net']},
            'RMSE': {k: [] for k in ['load', 'pv', 'net']}
        }

        for _ in range(run_count):
            with open(f'{config.SAVE_DATA_PATH}/{save_filename}_{_}.json', mode = 'r', encoding = 'utf-8') as f:
                data: Dict = json.load(fp = f)

            # Append training loss
            # Output list of list is a matrix with run_count rows and rounds columns
            if not output_data['training_loss']:
                output_data['training_loss'] = [np.array(data['training_loss']).mean(axis = 1)]
            else:
                output_data['training_loss'] = np.append(output_data['training_loss'], [np.array(data['training_loss']).mean(axis = 1)], axis = 0).tolist()

            # Append each metric for each column
            for _m in metrics_name:
                for _c in columns:
                    output_data[_m][_c].append(data[_m][_c])

        # Check if the output data can be written in the file (create a new directory if needed)
        if not os.path.exists(f'{config.SAVE_DATA_PATH}/grouping'):
            logger.info(f'Grouping directory not found. Creating one.')
            os.makedirs(f'{config.SAVE_DATA_PATH}/grouping')
        with open(f'{config.SAVE_DATA_PATH}/grouping/{save_filename}.json', mode = 'w', encoding = 'utf-8') as f:
            f.write(json.dumps(output_data, indent = 4))

        logger.info(f'Data grouping for {save_filename} ended successfully !')
        return output_data