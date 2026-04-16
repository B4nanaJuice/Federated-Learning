# Imports
import copy
import torch
import threading
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from typing import Optional, Callable, Dict, List
import matplotlib.pyplot as plt

from app.models.client import Client
from app.models.dataloader import EnergyDataset
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
        self.model_checkpoint: Dict[str, torch.Tensor] = copy.deepcopy(global_model.state_dict())

        # Model exchange
        self.received_updates: List[Dict] = []
        self.broadcast_model: Optional[Dict[str, torch.Tensor]] = None

        # Global metrics
        self.global_val_loss: float = float('inf')
        self.participation_rate: float = 0.0
        self.training_loss: List[List[float]] = []

        # Validation phase
        self.validation_predictions: Dict = {}

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

        training_loss: List[float] = []
        for client in self.selected_clients:
            update = client.send_update()
            training_loss.append(update.get('train_loss'))
            self.received_updates.append(update)
            self.client_weights[client.client_id] = 1/len(self.selected_clients)
        self.training_loss.append(training_loss)

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
        self.global_model.load_state_dict(copy.deepcopy(self.model_checkpoint))
        return
    
    @staticmethod
    def _fedavg(updates: List[Dict], weights: Dict[str, float]) -> Dict[str, torch.Tensor]:
        aggregated_delta: Dict[str, torch.Tensor] = {}

        for update in updates:
            client_id = update.get('client_id')
            weight = weights.get(client_id)

            for k, delta in update.get('weights').items():
                if k not in aggregated:
                    aggregated[k] = torch.zeros_like(delta)
                aggregated[k] += weight * delta

        return aggregated

    def run(self, client_fraction: float = 1.0) -> None:
        for round in tqdm(range(1, self.max_rounds + 1), desc = 'Round'):
            # logger.info(f'Round {round}/{self.max_rounds}')
            self.select_clients(client_fraction)
            self.broadcast()
            self.collect_updates()
            self.aggregate()

            # logger.info(f'Active clients: {len(self.selected_clients)}')
            # logger.info(f'Participation rate: {self.participation_rate:.0%}')

            if round % 10 == 0:
                self.save_checkpoint()
                # logger.info(f'Saved checkpoing model (round {round})')

        return
    
    def validate(self, validation_dataset_index: int = 1) -> None:
        
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.global_model = self.global_model.to(device = device)
        self.global_model.eval()

        # Get data
        _tensor: torch.Tensor = torch.load(f'data/processed/val/building_{validation_dataset_index}.pt')
        features: torch.Tensor = _tensor[:, :-3]
        targets: torch.Tensor = _tensor[:, -3:]
        dataset: EnergyDataset = EnergyDataset(features, targets)

        with torch.no_grad():
            features, targets = dataset[:]
            features = features.to(device = device)
            targets = targets.to(device = device)

            predictions: torch.Tensor = self.global_model(features)

            self.validation_predictions = {
                'load': predictions[:, 0],
                'pv': predictions[:, 1],
                'net': predictions[:, 2]
            }

        mse: Dict = {
            'load': np.square(np.subtract(self.validation_predictions['load'], targets[:, 0])).mean(),
            'pv': np.square(np.subtract(self.validation_predictions['pv'], targets[:, 1])).mean(),
            'net': np.square(np.subtract(self.validation_predictions['net'], targets[:, 2])).mean()
        }

        logger.debug(f'MSE load: {mse['load']:.4f}, pv: {mse['pv']:.4f}, net: {mse['net']:.4f}')

        # print(server.validation_predictions)
        _len = len(dataset)
        x = np.linspace(1, _len, _len)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

        ax1.plot(x, targets[:, 0], label = 'load_true')
        ax2.plot(x, targets[:, 1], label = 'pv_true')
        ax3.plot(x, targets[:, 2], label = 'net_true')

        ax1.plot(x, self.validation_predictions['load'], label = 'load')
        ax2.plot(x, self.validation_predictions['pv'], label = 'pv')
        ax3.plot(x, self.validation_predictions['net'], label = 'net')

        ax1.legend()
        ax2.legend()
        ax3.legend()
        plt.show()

        return
    
    def plot_data(self) -> None:
        # Plot loss
        x = np.linspace(1, self.max_rounds, self.max_rounds)
        min_loss = [min(_) for _ in self.training_loss]
        max_loss = [max(_) for _ in self.training_loss]
        mean_loss = [sum(_)/len(_) for _ in self.training_loss]

        plt.fill_between(x, min_loss, max_loss, color = '#89abcd')
        plt.plot(x, mean_loss)

        plt.show()

        return

def check_server():
    logger.info('Starting server check')

    from app.models.model import NormalMLP, SoftGatedMoE
    server: Server = Server(global_model = NormalMLP(), max_rounds = 1)
    for i in range(1, 4):
        server.register_client(Client(client_id = i, model = NormalMLP(), batch_size = 64, local_epochs = 30))
    server.run()

    server.validate(validation_dataset_index = 18)
    # server.plot_data()
    
    logger.info('Server check ended successfully')