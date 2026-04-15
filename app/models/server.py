# Imports
import copy
import torch
import threading
import torch.nn as nn
from typing import Optional, Callable, Dict, List

from app.models.client import Client
from config import create_logger

logger = create_logger(__name__)


class Server:
    def __init__(self,
                 global_model: nn.Module,
                 max_rounds: int = 50,
                 min_clients: int = 2,
                 aggregation_function: Optional[Callable] = None
                 ):
        
        # Coordination
        self.current_round: int = 0
        self.max_rounds: int = max_rounds
        self.min_clients: int = min_clients

        # Clients registry
        self.client_registry: Dict[str, Client] = {}
        self.selected_clients: List[Client] = []
        self.client_weights: Dict[str, float] = {} # For the scoring and the weighted aggregation

        # Global model
        self.global_model: nn.Module = global_model
        self.aggregation_function: Callable = aggregation_function or self._fedavg
        self.model_checkpoint = copy.deepcopy(global_model.state_dict())

        # Model exchange
        self.received_updates: List[Dict] = []
        self.broadcast_model: Optional[Dict[str, torch.Tensor]] = None

        # Global metrics
        self.global_val_loss: float = float('inf')
        self.participation_rate: float = 0.0

    def register_client(self, client: Client) -> None:
        self.client_registry[client.client_id] = client
        return

    def select_clients(self, fraction: float = 1.0) -> List[Client]:
        import random as rd
        k = max(self.min_clients, int(len(self.client_registry) * fraction))
        k = min(k, len(self.client_registry))
        self.selected_clients = rd.sample(list(self.client_registry.values()), k)
        self.participation_rate = len(self.selected_clients) / len(self.client_registry)
        return self.selected_clients
    
    def broadcast(self) -> Dict[str, torch.Tensor]:
        self.broadcast_model = copy.deepcopy(self.global_model.state_dict())
        for client in self.selected_clients:
            client.receive_global_model(self.broadcast_model)
        return self.broadcast_model
    
    def collect_updates(self) -> None:
        self.received_updates = []
        threads: List[threading.Thread] = []
        for client in self.selected_clients:
            threads.append(threading.Thread(target = client.train_local))

        [t.start() for t in threads]
        [t.join() for t in threads]

        for client in self.selected_clients:
            update = client.send_update()
            self.received_updates.append(update)
            self.client_weights[client.client_id] = 1/len(self.selected_clients)

    def aggregate(self) -> None:
        if len(self.received_updates) < self.min_clients:
            raise Exception('Number of minimum models not reached')
        
        # weights_before = copy.deepcopy(self.global_model.state_dict())
        new_state = self.aggregation_function(self.received_updates, self.client_weights)
        self.global_model.load_state_dict(new_state)

        self.current_round += 1
        return
    
    def save_checkpoint(self) -> None:
        self.model_checkpoint = copy.deepcopy(self.global_model.state_dict())
        return
    
    def load_checkpoint(self) -> None:
        self.global_model.load_state_dict(self.model_checkpoint)
        return
    
    @staticmethod
    def _fedavg(updates: List[Dict], weights: Dict[str, float]) -> Dict[str, torch.Tensor]:
        aggregated_delta: Dict[str, torch.Tensor] = {}

        for update in updates:
            client_id = update.get('client_id')
            weight = weights.get(client_id)

            for k, delta in update.get('delta_weights').items():
                if k not in aggregated_delta:
                    aggregated_delta[k] = torch.zeros_like(delta)
                aggregated_delta[k] = weight * delta

        return aggregated_delta

    def run(self, client_fraction: float = 1.0) -> None:
        for round in range(1, self.max_rounds + 1):
            logger.info(f'Round {round}/{self.max_rounds}')
            self.select_clients(client_fraction)
            self.broadcast()
            self.collect_updates()
            self.aggregate()

            logger.info(f'Active clients: {len(self.selected_clients)}')
            logger.info(f'Participation rate: {self.participation_rate:.0%}')

            if round % 10 == 0:
                self.save_checkpoint()
                logger.info(f'Saved checkpoing model (round {round})')
        
        return

def check_server():
    logger.info('Starting server check')

    from app.models.model import NormalMLP
    server: Server = Server(global_model = NormalMLP(), max_rounds = 3)
    for i in range(1, 4):
        server.register_client(Client(client_id = i, model = NormalMLP(), batch_size = 64, local_epochs = 3))
    server.run()
    
    logger.info('Server check ended successfully')