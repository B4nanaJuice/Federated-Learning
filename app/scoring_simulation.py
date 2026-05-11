# Imports
import json
import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt

from config import create_logger, config
from app.models import Client, NormalMLP, Server
from app.attacking_models import MaliciousClient
from app.scoring import ScoringServer, ScoringMetric, KrumServer, MKrumServer, NormAggServer, CBAAFedAvgServer

logger = create_logger(__name__)

def simulate_scoring(**options):

    # Get parameters from options
    ## Overall parameters
    model = NormalMLP
    run_count: int = int(options.get('run-count', 10))
    save_filename: str = options.get('save-filename', 'scoring')

    ## Server parameters
    server_max_rounds: int = int(options.get('rounds', 20))
    server_scoring: bool = options.get('server-scoring', 'true') == 'true'

    ## Scoring parameters
    scoring_metric: ScoringMetric = ScoringMetric[options.get('metric', 'distance').upper()]
    scoring_threshold: float = float(options.get('threshold', .4))
    scoring_sigmas: List[float] = [float(_) for _ in options.get('sigma', '1').split(',')]
    scoring_bins: List[int] = [int(_) for _ in options.get('bins', '100').split(',')]

    ## Client parameters
    client_count: int = int(options.get('client-count', 10))
    client_epochs: int = int(options.get('epochs', 10))
    client_batch_size: int = int(options.get('batch', 128))
    client_lr: float = float(options.get('lr', 1e-3))
    client_fraction: float = float(options.get('fraction', .5))

    # Generate result variables
    results: Dict[str, any] = {
        # Rejected percentage of models -> Size = size sigmas
        'rejected': np.zeros(max(len(scoring_sigmas), len(scoring_bins))),
        # RMSE of training phase over rounds -> Size = rounds x parameters (sigmas or bins)
        'RMSE': np.zeros((max(len(scoring_sigmas), len(scoring_bins)), server_max_rounds)) 
    }

    for idx in range(max(len(scoring_sigmas), len(scoring_bins))):
        sigma: float = scoring_sigmas[min(idx, len(scoring_sigmas)-1)]
        bins: int = scoring_bins[min(idx, len(scoring_bins)-1)]
        logger.info(f'Simulation with sigma = {sigma}')

        for run in range(run_count):
            
            # Server creation
            if server_scoring:
                server: ScoringServer = ScoringServer(
                    global_model = model(),
                    max_rounds = server_max_rounds,
                    metric = scoring_metric,
                    threshold = scoring_threshold,
                    metric_parameters = {
                        'sigma': sigma,
                        'bins': bins
                    }
                )
            else:
                server: Server = Server(
                    global_model = model(),
                    max_rounds = server_max_rounds
                )

            # Register clients
            for _ in range(client_count):
                server.register_client(Client(
                    client_id = _+1,
                    model = model(),
                    local_epochs = client_epochs,
                    batch_size = client_batch_size,
                    learning_rate = client_lr
                ))

            # Run server
            server.run(client_fraction = client_fraction)

            # Append results
            results['rejected'][idx] += server.rejected_models
            average_mse = np.array([sum(_)/len(_) for _ in server.training_loss])
            average_rmse = np.sqrt(average_mse)
            results['RMSE'][idx] += average_rmse

    # Normalize data
    results['rejected'] /= run_count
    results['rejected'] = results['rejected'] * 100 / (client_count * client_fraction * server_max_rounds)

    results['RMSE'] /= run_count

    # Save data to file
    with open(f'{config.SAVE_DATA_PATH}/{save_filename}.json', mode = 'w', encoding = 'utf-8') as f:
        f.write(json.dumps(
            {
                'parameters': scoring_sigmas if len(scoring_sigmas) > len(scoring_bins) else scoring_bins,
                'rejected': results['rejected'].tolist(),
                'RMSE': results['RMSE'].tolist()
            },
            indent = 4
        ))

# Simulate all defenses
def simulate_defenses(**options):

    # Get variable parameters
    defense: str = options.get('defense')
    malicious_percentage: int = int(options.get('malicious'))

    server_options: Dict[str, any] = {
        'global_model': NormalMLP(),
        'max_rounds': 10
    }

    server: Server = {
        'fedavg': Server(**server_options),
        'krum': KrumServer(**server_options),
        'mkrum': MKrumServer(*server_options),
        'norm': NormAggServer(**server_options),
        'cbaa': CBAAFedAvgServer(**server_options),
        'distance': ScoringServer(**server_options, metric = ScoringMetric.DISTANCE, metric_parameters = {'sigma': 0.8}),
        'distribution': ScoringServer(**server_options, metric = ScoringMetric.DISTRIBUTION)
    }.get('defense')

    clients: List[Client] = []
    client_id: int = 1

    for _ in range(int(20 * malicious_percentage / 100)):
        clients.append(MaliciousClient(
            client_id = client_id,
            batch_size = 128,
            attack_rate = lambda x: True,
        ))
        client_id += 1

    while client_id <= 20:
        clients.append(Client(
            client_id = client_id,
            batch_size = 128
        ))
        client_id += 1

    server.register_clients(clients = clients)
    server.run(.5)
    server.run_test(dataset_index = 5, days_count = 5)
    server.save_metrics(f'defenses/{defense}_{malicious_percentage}')