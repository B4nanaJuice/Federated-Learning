# Imports
import json
import numpy as np
from tqdm import tqdm
from typing import List, Dict
import matplotlib.pyplot as plt

from config import create_logger, config
from app.models import Client, NormalMLP, Server
from app.attacking_models import AttackedServer, MaliciousClient
from app.scoring import ScoringServer, ScoringMetric, ScoringClient, KrumServer, MKrumServer, NormAggServer, CBAAFedAvgServer, TMeanServer, RFAServer, FLTrustServer
from app.scoring import AttackedKrumServer, AttackedMKrumServer, AttackedNormAggServer, AttackedCBAAFedAvgServer, AttackedTMeanServer, AttackedRFAServer, AttackedFLTrustServer

logger = create_logger(__name__)

# Simulate all defenses
def simulate_defenses(**options):

    # Get variable parameters
    defense: str = options.get('defense')
    partial: bool = options.get('partial', 'false').lower() == 'true'

    server_options: Dict[str, any] = {
        'global_model': NormalMLP(),
        'max_rounds': 20, # 20
        'partial_attack': partial,
        'attack_rate': lambda x: x in [5, 6, 7, 17, 18, 19] if partial else x == 19
    }

    for run in tqdm(range(10)): # 10

        server: Server = {
            'tmean': AttackedTMeanServer(**server_options),
            'rfa': AttackedRFAServer(**server_options),
            'fltrust': AttackedFLTrustServer(**server_options)
        }.get(defense)

        clients: List[Client] = []
        client_id: int = 1
        client_count: int = 20 # 20

        while client_id <= client_count:
            clients.append(Client(
                client_id = client_id,
                model = NormalMLP(),
                local_epochs = 15, # 15
                batch_size = 128
            ))
            client_id += 1

        server.register_clients(clients = clients)
        server.run(.5)
        server.run_test(dataset_index = 5, days_count = 5)
        server.save_metrics(f'{defense}_{"partial" if partial else "total"}_{run}')
        # server.save_metrics(f'defenses/{defense}_{"partial" if partial else "total"}')