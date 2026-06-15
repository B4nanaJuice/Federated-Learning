# Imports
import os
import json
import numpy as np
from glob import glob
from typing import List, Dict

from config import create_logger, config
from app.models import NormalMLP, Client, Server
from app.scoring import ScoringMetric, ScoringServer, ScoringClient
from app.attacking_models import AttackedServer, MaliciousClient

logger = create_logger(__name__)

# Static class with methods
class SimulationService:

    # Baseline simulations
    @staticmethod
    def simulate_baseline(*args, **options) -> None:
        
        # Simulation parameters from options
        ## Overall parameters
        run_count: int = 10

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
            server: Server = Server(
                global_model = NormalMLP(),
                max_rounds = server_max_rounds
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

            server.run_test(dataset_index = 5, days_count = 5)
            server.save_metrics(f'clean_{run}')
        return
    
    # Baseline simulations
    @staticmethod
    def simulate_data_poisoning(*args, **options) -> None:
        
        # Simulation parameters from options
        ## Overall parameters
        run_count: int = 10

        ## Attack parameters
        malicious_percentage: float = float(options.get('malicious', 0))
        assert 0 <= malicious_percentage <= 100

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
            server: Server = Server(
                global_model = NormalMLP(),
                max_rounds = server_max_rounds
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
                        attack_rate = 1,
                        attack_target = 'data'
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
            server.save_metrics(f'{malicious_percentage}_data_{run}')
        return
    
    # Baseline simulations
    @staticmethod
    def simulate_model_poisoning(*args, **options) -> None:
        
        # Simulation parameters from options
        ## Overall parameters
        run_count: int = 10

        ## Attack parameters
        malicious_percentage: float = float(options.get('malicious', 0))
        assert 0 <= malicious_percentage <= 100
        attack_method: str = options.get('attack', 'gaussian_noise')
        assert attack_method in ['gaussian_noise', 'gaussian_weights', 'uniform_noise', 'uniform_weights', 'gradient_inversion', 'gradient_amplification']

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
            server: Server = Server(
                global_model = NormalMLP(),
                max_rounds = server_max_rounds
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
                        attack_rate = 1,
                        attack_method = attack_method,
                        attack_target = 'model'
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
            server.save_metrics(f'{malicious_percentage}_{attack_method}_{run}')
        return
    
    # Baseline simulations
    @staticmethod
    def simulate_server_attack(*args, **options) -> None:
        
        # Simulation parameters from options
        ## Overall parameters
        run_count: int = 10

        ## Attack parameters
        attack_partial: bool = options.get('partial', 'true').lower() == 'true'
        assert attack_partial in [True, False]

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
            server: Server = AttackedServer(
                global_model = NormalMLP(),
                max_rounds = server_max_rounds,
                attack_rate = lambda x: x in list(range(13, 21)) if attack_partial else x == 19,
                partial_attack = attack_partial
            )

            # Add Malicious clients
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
            server.save_metrics(f'{"partial" if attack_partial else "total"}_{run}')
        return

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

            rejected.append(sum(server.rejected_models))
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

    # Scoring simulations for computing best sigma decay factor
    @staticmethod
    def sigma_decay_measurment(*args, **options) -> None:
        
        # Simulation parameters from options
        ## Overall parameters
        save_filename: str = options.get('save-filename', 'scoring')

        ## Server parameters
        server_max_rounds: int = 20

        ## Scoring parameters
        scoring_metric: ScoringMetric = {
            'dataset': ScoringMetric.DATASET,
            'distance': ScoringMetric.DISTANCE
        }.get(options.get('metric', 'distance'))
        scoring_threshold: float = .4
        scoring_sigma: float = {
            'distance': 7,
            'dataset': .3
        }.get(options.get('metric'))
        scoring_decays: List[float] = list(range(2, 27))
        scoring_decay_type: str = options.get('decay', 'root') # log or root

        ## Client parameters
        client_count: int = 10
        client_epochs: int = 15
        client_batch_size: int = 128
        client_lr: float = 1e-3
        client_fraction: float = .5

        # Simulation results
        rejected: List[List[int]] = []

        # Simulation
        for decay_factor in scoring_decays:

            logger.info(f'Starting simulation for decay = {decay_factor}')
                
            server: ScoringServer = ScoringServer(
                global_model = NormalMLP(),
                max_rounds = server_max_rounds,
                metric = scoring_metric,
                threshold = scoring_threshold,
                metric_parameters = {'sigma': scoring_sigma, 'decay': decay_factor, 'decay_type': scoring_decay_type}
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
            logger.info(f'Rejected models for decay : {decay_factor} = {sum(server.rejected_models)}')

        logger.info(f'rejected models: {rejected}')
        logger.info('Saving data to file...')

        # Save data to file
        with open(f'{config.SAVE_DATA_PATH}/sigma_testing/decay_{scoring_decay_type}_{save_filename}.json', mode = 'w', encoding = 'utf-8') as f:
            f.write(json.dumps(
                {
                    'parameters': scoring_decays,
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
                attack_rate = lambda x: x in list(range(13, 21)) if attack_partial else x == 19
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
            server.save_metrics(f'{save_filename} {"partial" if attack_partial else "total"}_{run}')
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

    # Simulate decay for client scoring
    @staticmethod
    def simulate_client_decay_scoring(*args, **options) -> None:
        
        # Simulation parameters from options
        ## Overall parameters
        run_count: int = 10
        save_filename: str = options.get('save-filename', 'scoring')

        ## Attack parameters
        attack_partial: bool = options.get('partial', 'false').lower() == 'true'

        ## Scoring parameters
        scoring_metric: ScoringMetric = ScoringMetric[options.get('metric', 'distance').upper()]
        scoring_threshold: float = .4
        scoring_sigma: float = {
            ScoringMetric.DISTANCE: 6.8,
            ScoringMetric.DATASET: .3
        }.get(scoring_metric)
        scoring_decay_type: str = options.get('decay', 'root').lower()
        scoring_decay: float = {
            'distance': {
                'log': 22
            }, 'dataset': {
                'log': 4,
                'root': 4
            }
        }.get(options.get('metric', 'distance')).get(scoring_decay_type)
        if scoring_decay is None:
            return
        scoring_parameters: Dict[str, any] = {
            'sigma': scoring_sigma,
            'decay': scoring_decay,
            'decay_type': scoring_decay_type
        }

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
                attack_rate = lambda x: x in list(range(13, 21)) if attack_partial else x == 19
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
            server.save_metrics(f'{scoring_decay_type}_{save_filename} {"partial" if attack_partial else "total"}_{run}')
        return

    # Simulate decay for server scoring
    @staticmethod
    def simulate_server_decay_scoring(*args, **options) -> None:
        
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
        scoring_sigma: float = {
            ScoringMetric.DISTANCE: 6.8,
            ScoringMetric.DATASET: .3
        }.get(scoring_metric)
        scoring_decay_type: str = options.get('decay', 'root').lower()
        scoring_decay: float = {
            'distance': {
                'log': 22
            }, 'dataset': {
                'log': 4,
                'root': 4
            }
        }.get(options.get('metric', 'distance')).get(scoring_decay_type)
        if scoring_decay is None:
            return
        scoring_parameters: Dict[str, any] = {
            'sigma': scoring_sigma,
            'decay': scoring_decay,
            'decay_type': scoring_decay_type
        }

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
            server.save_metrics(f'{scoring_decay_type}_{save_filename}_{run}')
        return

    # Simulate defense for server's attack (attacked server)
    @staticmethod
    def simulate_client_defense(*args, **options) -> None:

        from app.scoring import AttackedCBAAFedAvgServer, AttackedCLRAServer, AttackedFLTrustServer, AttackedKrumServer, AttackedMKrumServer, AttackedNormAggServer, AttackedRFAServer, AttackedTMeanServer, AttackedWeightedFedAvgServer
        
        # Simulation parameters from options
        ## Overall parameters
        run_count: int = 10

        ## Attack parameters
        attack_partial: bool = options.get('partial', 'false').lower() == 'true'

        ## Defense parameters
        defense: str = options.get('defense', 'fedavg').lower()

        ## Server parameters
        server_max_rounds: int = 20

        ## Client parameters
        client_count: int = 20
        client_epochs: int = 15
        client_batch_size: int = 128
        client_lr: float = 1e-3
        client_fraction: float = .5

        server_options: Dict[str, any] = {
            'global_model': NormalMLP(),
            'max_rounds': server_max_rounds, # 20
            'partial_attack': attack_partial,
            'attack_rate': lambda x: x in list(range(13, 21)) if attack_partial else x == 19
        }

        for run in range(run_count):

            server: Server = {
                'fedavg': AttackedServer(**server_options),
                'krum': AttackedKrumServer(**server_options),
                'mkrum': AttackedMKrumServer(**server_options),
                'norm': AttackedNormAggServer(**server_options),
                'cbaa': AttackedCBAAFedAvgServer(**server_options),
                'tmean': AttackedTMeanServer(**server_options),
                'rfa': AttackedRFAServer(**server_options),
                'fltrust': AttackedFLTrustServer(**server_options),
                'clra': AttackedCLRAServer(**server_options)
            }.get(defense)

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

            server.run_test(dataset_index = 5, days_count = 5)
            server.save_metrics(f'{defense} {"partial" if attack_partial else "total"}_{run}')
        return

    # Simulate defense for clients attack
    @staticmethod
    def simulate_server_defense(*args, **options) -> None:
        
        from app.scoring import KrumServer, MKrumServer, NormAggServer, CBAAFedAvgServer, TMeanServer, RFAServer, FLTrustServer, CLRAServer

        # Simulation parameters from options
        ## Overall parameters
        run_count: int = 10

        ## Attack parameters
        malicious_percentage: float = float(options.get('malicious', 0))
        assert 0 <= malicious_percentage <= 100

        ## Defense parameters
        defense: str = options.get('defense', 'fedavg').lower()

        ## Server parameters
        server_max_rounds: int = 20

        ## Client parameters
        client_count: int = 20
        client_epochs: int = 15
        client_batch_size: int = 128
        client_lr: float = 1e-3
        client_fraction: float = .5

        server_options: Dict[str, any] = {
            'global_model': NormalMLP(),
            'max_rounds': server_max_rounds
        }

        for run in range(run_count):

            server: Server = {
                'fedavg': Server(**server_options),
                'krum': KrumServer(**server_options),
                'mkrum': MKrumServer(**server_options),
                'norm': NormAggServer(**server_options),
                'cbaa': CBAAFedAvgServer(**server_options),
                'tmean': TMeanServer(**server_options),
                'rfa': RFAServer(**server_options),
                'fltrust': FLTrustServer(**server_options),
                'clra': CLRAServer(**server_options)
            }.get(defense)

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
            server.save_metrics(f'{defense} {malicious_percentage}_{run}')
        return
    
    # Offline training simulation
    @staticmethod
    def simulate_offline_training(*args, **options) -> None:
        from app.degraded.network import Network
        from app.degraded.offline_client import OfflineClient
        from app.degraded.offline_server import OfflineServer
    
        # Simulation parameters from options
        ## Overall parameters
        run_count: int = 10

        ## Attack parameters
        attack_partial: bool = True

        ## Scoring parameters
        scoring_metric_str: str = options.get('metric', 'distance')
        scoring_metric: ScoringMetric = ScoringMetric[scoring_metric_str.upper()]
        scoring_threshold: float = .4
        scoring_sigma: float = {
            ScoringMetric.DISTANCE: 7,
            ScoringMetric.DATASET: .3
        }.get(scoring_metric)

        scoring_parameters: Dict[str, any] = {'sigma': scoring_sigma}

        ## Server parameters
        server_max_rounds: int = 20

        ## Client parameters
        client_count: int = 20
        client_epochs: int = 15
        client_batch_size: int = 128
        client_lr: float = 1e-3
        client_fraction: float = .5

        for run in range(run_count):

            # create network and server
            network: Network = Network()
            server: OfflineServer = OfflineServer(
                global_model = NormalMLP(), 
                max_rounds = server_max_rounds, 
                partial_attack = attack_partial, 
                attack_rate = lambda x: x in list(range(13, 21))
            )
            server.register_network(network)

            clients: List[OfflineClient] = []
            for _ in range(1, client_count + 1):
                clients.append(
                    OfflineClient(
                        client_id = _,
                        model = NormalMLP(),
                        metric = scoring_metric,
                        threshold = scoring_threshold,
                        metric_parameters = scoring_parameters,
                        local_epochs = client_epochs,
                        batch_size = client_batch_size,
                        learning_rate = client_lr
                    )
                )
            network.register_clients(clients)
            server.register_clients(clients)

            server.run(client_fraction = client_fraction)

            server.run_test(dataset_index = 5, days_count = 5)
            server.save_metrics(f'offline {scoring_metric_str}_{run}')
        return

    # Method for grouping data
    @staticmethod
    def group_data(**options) -> Dict[str, any]:

        save_filename: str = options.get('save-filename', 'run')
        logger.info(f'Beginning data grouping for files {save_filename}.')

        run_count: int = len(glob(f'{config.SAVE_DATA_PATH}/{save_filename}_*'))
        logger.info(f'Found {run_count} files.')
        if run_count == 0:
            logger.warning('0 files are found. Aborting.')
            return
        
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