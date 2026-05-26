# Imports
import os
import json
import numpy as np
from glob import glob
from tqdm import tqdm
from typing import List, Callable, Dict

from config import create_logger, config
from app.models import Client, Server, NormalMLP, SoftGatedMoE
from app.attacking_models import MaliciousClient, AttackedServer

logger = create_logger(__name__)

def simulate_clean():
    logger.info('Starting clean simulation')

    # Create server
    server: Server = Server(
        global_model = NormalMLP(),
        max_rounds = 50
    )

    # Add clients
    clients: List[Client] = [
        Client(client_id = _, model = NormalMLP(), local_epochs = 10, batch_size = 256)
        for _ in range(1, 11)
    ]
    server.register_clients(clients = clients)

    # Run
    server.run(.5)

    server.run_test()
    server.plot()

    logger.info('End of clean simulation')
    return

def simulate_malicious_clients():
    logger.info('Starting malicious clients simulation')

    # Create server
    server: Server = Server(
        global_model = NormalMLP(),
        max_rounds = 50
    )

    # Add clients
    clients: List[Client] = [
        Client(client_id = _, model = NormalMLP(), local_epochs = 20, batch_size = 256)
        for _ in range(1, 7)
    ]
    malicious_clients: List[Client] = [
        MaliciousClient(client_id = _, model = NormalMLP(), local_epochs = 20, batch_size = 256, attack_rate = lambda x: x in [10, 25])
        for _ in range(8, 11)
    ]
    server.register_clients(clients = clients)
    server.register_clients(clients = malicious_clients)

    # Run
    server.run(.5)

    server.run_test()
    server.plot()

    logger.info('End of malicious clients simulation')
    return

def simulate_attacked_server():
    logger.info('Starting attacked server simulation')

    # Create server
    server: AttackedServer = AttackedServer(
        global_model = NormalMLP(),
        max_rounds = 50,
        attack_rate = lambda x: x == 10
    )

    # Add clients
    clients: List[Client] = [
        Client(client_id = _, model = NormalMLP(), local_epochs = 10, batch_size = 256)
        for _ in range(1, 11)
    ]
    server.register_clients(clients = clients)

    # Run
    server.run(.5)

    server.run_test()
    server.plot()

    logger.info('End of attacked server simulation')
    return

def simulate_attacked_and_malicious():
    logger.info('Starting attacked server and malicious clients simulation')

    # Create server
    server: AttackedServer = AttackedServer(
        global_model = NormalMLP(),
        max_rounds = 50,
        attack_rate = lambda x: x == 10
    )

    # Add clients
    clients: List[Client] = [
        Client(client_id = _, model = NormalMLP(), local_epochs = 20, batch_size = 256)
        for _ in range(1, 7)
    ]
    malicious_clients: List[Client] = [
        MaliciousClient(client_id = _, model = NormalMLP(), local_epochs = 20, batch_size = 256, attack_rate = lambda x: x == 15)
        for _ in range(8, 11)
    ]
    server.register_clients(clients = clients)
    server.register_clients(clients = malicious_clients)

    # Run
    server.run(.5)

    server.run_test()
    server.plot()

    logger.info('End of attacked server and malicious clients simulation')

def multi_run(**options):

    # Get parsed cli options
    attacked_server: bool = bool(eval(options.get('attacked-server', 'False')))
    model = NormalMLP if options.get('model', 'normalmlp').lower() == 'normalmlp' else SoftGatedMoE
    max_rounds: int = int(options.get('max-rounds', 10))
    min_clients: int = int(options.get('min-clients', 10))
    server_attack_rate: float | Callable = eval(options.get('server-attack-rate', '.2'))
    server_attack_method: str = options.get('server-attack-method', 'uniform_noise')
    partial_attack: bool = options.get('partial-attack', 'False') == 'True'

    total_clients: int = int(options.get('total-clients', 10))
    malicious_client_count: int = int(options.get('malicious-client-count', 0))
    epochs: int = int(options.get('epochs', 10))
    batch_size: int = int(options.get('batch-size', 128))
    learning_rate: float = float(options.get('lr', 1e-3))
    client_attack_rate: float | Callable = eval(options.get('client-attack-rate', '.2'))
    client_attack_method: str = options.get('client-attack-method', 'uniform_noise')
    client_attack_target: str = options.get('client-attack-target', 'model')
    client_fraction: float = float(options.get('client-fraction', .5))

    save_filename: str = options.get('save-filename', 'multi_run')
    run_count: int = int(options.get('run-count', 5))

    logger.info(f'Starting simulation with {run_count} runs. {total_clients} total clients with {malicious_client_count} malicious clients.')
    
    # Making run and save all metrics
    for run in tqdm(range(run_count), desc = 'Run count'):

        # Server creation
        if attacked_server:
            server: Server = AttackedServer(
                global_model = model(),
                max_rounds = max_rounds,
                min_clients = min_clients,
                attack_rate = server_attack_rate,
                attack_method = server_attack_method,
                partial_attack = partial_attack
            )
        else:
            server: Server = Server(
                global_model = model(),
                max_rounds = max_rounds,
                min_clients = min_clients
            )

        # Clients
        clients: List[Client] = []
        client_count: int = 0

        # Add honnest clients
        for _ in range(total_clients - malicious_client_count):
            client_count += 1
            clients.append(Client(
                client_id = client_count, 
                model = model(), 
                local_epochs = epochs, 
                batch_size = batch_size,
                learning_rate = learning_rate
            ))

        for _ in range(malicious_client_count):
            client_count += 1
            clients.append(MaliciousClient(
                client_id = client_count, 
                model = model(), 
                local_epochs = epochs, 
                batch_size = batch_size,
                learning_rate = learning_rate,
                attack_rate = client_attack_rate,
                attack_method = client_attack_method,
                partial_attack = partial_attack,
                attack_target = client_attack_target
            ))

        server.register_clients(clients = clients)

        server.run(client_fraction = client_fraction)
        server.run_test()
        server.save_metrics(f'{save_filename}_{run}')

def data_grouping(**options) -> Dict[str, any]:

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
        if not output_data['training_loss']:
            output_data['training_loss'] = [data['training_loss']]
        else:
            output_data['training_loss'] = np.append(output_data['training_loss'], [data['training_loss']], axis = 0).tolist()

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
    