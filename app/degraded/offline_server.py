# Imports
import torch.nn as nn
from app.models.server import Server
from app.attacking_models.attacked_server import AttackedServer
from app.degraded.network_interface import NetworkInterface
from tqdm import tqdm
from typing import Dict, List
import torch
import copy
import threading

class OfflineServer(AttackedServer):
    def __init__(self, global_model: nn.Module, **kwargs):
        super().__init__(global_model = global_model, **kwargs)
        self.network: NetworkInterface = None

    def register_network(self, network: NetworkInterface) -> None:
        self.network = network
        return
    
    def run(self, client_fraction: float = 1.0) -> None:
        for round in tqdm(range(1, self.max_rounds + 1), desc = 'Round'):
            self.select_clients(client_fraction)
            print('Client selection finished, calling broadcast')
            threads = self.broadcast(round = round)
            print('Broadcast finished, calling end_model_broadcast')
            self.network.end_model_broadcast(self.collect_updates) # Start model evaluation phase
            # Clients train their local model
            [t.join() for t in threads]
            print('end_model_broadcast finished, calling aggregate')
            self.aggregate()

        return
    
    def broadcast(self, round: int, threaded: bool = True) -> List[threading.Thread]:

        if self.can_attack():
            print('Server is attacking')
            self.global_model.load_state_dict(self.poison_model(self.global_model, self.attack_method, self.partial_attack))
            self.attacked_rounds.append(self.current_round)

        self.broadcast_model = copy.deepcopy(self.global_model.state_dict())

        threads: List[threading.Thread] = []
        for client in self.selected_clients:
            threads.append(threading.Thread(target = client.receive_global_model, args = (self.broadcast_model, round)))

        [t.start() for t in threads]

        return threads