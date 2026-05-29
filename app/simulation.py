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

def multi_run(**options):

    # Get parsed cli options
    attacked_server: bool = bool(eval(options.get('attacked-server', 'False')))
    model = NormalMLP
    max_rounds: int = int(options.get('max-rounds', 20))
    server_attack_rate: float | Callable = eval(options.get('server-attack-rate', '.2'))
    server_attack_method: str = options.get('server-attack-method', 'uniform_noise')
    partial_attack: bool = options.get('partial-attack', 'False') == 'True'

    total_clients: int = int(options.get('total-clients', 20))
    malicious_client_count: int = int(options.get('malicious-client-count', 0))
    epochs: int = int(options.get('epochs', 15))
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
                attack_rate = server_attack_rate,
                attack_method = server_attack_method,
                partial_attack = partial_attack
            )
        else:
            server: Server = Server(
                global_model = model(),
                max_rounds = max_rounds
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
    