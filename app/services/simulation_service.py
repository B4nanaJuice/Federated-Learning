# Imports
import json
import numpy as np
from typing import List, Dict

from config import create_logger, config
from app.models import NormalMLP, Client
from app.scoring import ScoringMetric, ScoringServer

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
        scoring_metric: ScoringMetric = ScoringMetric[options.get('metric', 'distance').upper()]
        scoring_threshold: float = .4
        scoring_sigmas: List[float] = np.linspace(1, 10, 21).tolist() # 1, 1.5, 2, 2.5, ...

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