# Imports
import copy
import torch
import threading
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from typing import Optional, Callable, Dict, List
import matplotlib.pyplot as plt
import matplotlib as mpl

from app.models.client import Client
from app.models.dataloader import EnergyDataset
from config import create_logger, config

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
        self.participation_rate: float = 0.0
        self.training_loss: List[List[float]] = []

        # Validation phase
        self.validation_predictions: Dict = {}
        self.validation_MSE: Dict = {}

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
    
    def collect_updates(self, threaded: bool = config.SIM_THREADED) -> None:
        self.received_updates = []

        if threaded:
            threads: List[threading.Thread] = []
            for client in self.selected_clients:
                threads.append(threading.Thread(target = client.train_local))

            [t.start() for t in threads]
            [t.join() for t in threads]
        else:
            for client in self.selected_clients:
                client.train_local()

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
        aggregated: Dict[str, torch.Tensor] = {}

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
            self.select_clients(client_fraction)
            self.broadcast()
            self.collect_updates()
            self.aggregate()

            if round % 10 == 0:
                self.save_checkpoint()

        return
    
    def run_validation(self, dataset_index: int = 1, days_count: int = 10) -> None:

        self.global_model = self.global_model.to(device = config.DEVICE)
        self.global_model.eval()

        # Get data
        _tensor: torch.Tensor = torch.load(f'data/processed/val/building_{dataset_index}.pt')
        features: torch.Tensor = _tensor[:, :-3]
        targets: torch.Tensor = _tensor[:, -3:]
        dataset: EnergyDataset = EnergyDataset(features, targets)

        with torch.no_grad():
            features, targets = dataset[:days_count*48]
            features = features.to(device = config.DEVICE)
            targets = targets.to(device = config.DEVICE)

            predictions: torch.Tensor = self.global_model(features)

            # Save load, pv and net predictions
            self.validation_predictions = {
                'load': predictions[:, 0],
                'pv': predictions[:, 1],
                'net': predictions[:, 2],
                'load_true': targets[:, 0],
                'pv_true': targets[:, 1],
                'net_true': targets[:, 2],
            }

        # Compute MSE for load, pv and net consumption
        self.validation_MSE = {
            'load': np.square(np.subtract(predictions[:, 0].cpu(), targets[:, 0].cpu())).mean(),
            'pv': np.square(np.subtract(predictions[:, 1].cpu(), targets[:, 1].cpu())).mean(),
            'net': np.square(np.subtract(predictions[:, 2].cpu(), targets[:, 2].cpu())).mean()
        }

        return
    
    def plot(self, show_loss: bool = True, show_validation: bool = True) -> None:
        # Generate layout from show_* options
        fig = plt.figure()
        gs = mpl.gridspec.GridSpec(3, 2, wspace = 0.25, hspace = 0.25)

        if show_loss:
            loss_plot = fig.add_subplot(gs[:, 1])
            if not show_validation:
                loss_plot = fig.add_subplot(gs[:, :])

            x = np.linspace(1, self.max_rounds, self.max_rounds)
            min_loss = [min(_) for _ in self.training_loss]
            max_loss = [max(_) for _ in self.training_loss]
            mean_loss = [sum(_)/len(_) for _ in self.training_loss]

            loss_plot.fill_between(x, min_loss, max_loss, color = '#89abcd')
            loss_plot.plot(x, min_loss, '--', label = 'Minimum loss')
            loss_plot.plot(x, max_loss, '--', label = 'Maximum loss')
            loss_plot.plot(x, mean_loss, label = 'Average loss')
            loss_plot.legend()

        if show_validation:
            load_plot = fig.add_subplot(gs[0, 0])
            pv_plot = fig.add_subplot(gs[1, 0])
            net_plot = fig.add_subplot(gs[2, 0])
            if not show_loss:
                load_plot = fig.add_subplot(gs[0, :])
                pv_plot = fig.add_subplot(gs[1, :])
                net_plot = fig.add_subplot(gs[2, :])
            
            _len: int = len(self.validation_predictions['load'])
            x = np.linspace(1, _len, _len)

            load_plot.plot(x, self.validation_predictions['load_true'].cpu(), label = 'load truth')
            pv_plot.plot(x, self.validation_predictions['pv_true'].cpu(), label = 'pv truth')
            net_plot.plot(x, self.validation_predictions['net_true'].cpu(), label = 'net truth')

            load_plot.plot(x, self.validation_predictions['load'].cpu(), label = 'load prediction')
            pv_plot.plot(x, self.validation_predictions['pv'].cpu(), label = 'pv prediction')
            net_plot.plot(x, self.validation_predictions['net'].cpu(), label = 'net prediction')

            load_plot.legend()
            pv_plot.legend()
            net_plot.legend()

        plt.show()

def check_server():
    logger.info('Starting server check')

    from app.models.model import NormalMLP
    server: Server = Server(global_model = NormalMLP(), max_rounds = 2)
    for i in range(1, 4):
        server.register_client(Client(client_id = i, model = NormalMLP(), batch_size = 128, local_epochs = 3))
    server.run()

    logger.info(f'Starting validation phase')
    server.run_validation(dataset_index = 5, days_count = 5)
    server.plot()
    server.plot(show_loss = False)
    server.plot(show_validation = False)
    
    logger.info('Server check ended successfully')