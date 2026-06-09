# Imports
import copy
import json
import torch
import threading
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error
from typing import Optional, Dict, List

from app.models.client import Client
from config import create_logger, config
from app.services.aggregation_service import AggregationService
from app.models.dataloader import EnergyDataset

logger = create_logger(__name__)


class Server:
    def __init__(self,
                 global_model: nn.Module,
                 max_rounds: int = 50,
                 **kwargs
                 ):
        
        # Coordination
        self.current_round: int = 0
        self.max_rounds: int = max_rounds

        # Clients registry
        self.client_registry: Dict[str, Client] = {}
        self.selected_clients: List[Client] = []

        # Global model
        self.global_model: nn.Module = global_model

        # Model exchange
        self.received_updates: List[Dict] = []
        self.broadcast_model: Optional[Dict[str, torch.Tensor]] = None

        # Metrics
        self.training_loss: List[List[float]] = []
        self.test_predictions: Dict[str, List[float]] = {}

    def register_client(self, client: Client) -> None:
        """
        Add a client to the server's client registry.

        Args:
            client (Client): The client to add to the registry.
        """
        self.client_registry[client.client_id] = client
        return
    
    def register_clients(self, clients: List[Client]) -> None:
        """
        Add multiple clients to the server's client registry.

        Args:
            clients (List[Client]): A list containing the clients that will be added to the registry.
        """
        for _client in clients:
            self.register_client(_client)
        return

    def select_clients(self, fraction: float = 1.0) -> List[Client]:
        """
        Select clients that will train their local model on the current round.

        Args:
            fraction (float): The percentage of clients that will participate in the current round.

        Returns:
            List[Client]: A list containing the selected clients.
        """
        import random as rd
        k = int(len(self.client_registry) * fraction)
        self.selected_clients = rd.sample(list(self.client_registry.values()), k)
        return self.selected_clients
    
    def broadcast(self, round: int, threaded: bool = config.SIM_THREADED) -> Dict[str, torch.Tensor]:
        """
        Broadcast the model to selected clients.

        Args:
            round (int): The current round.
            threaded (bool = config.SIM_THREADED): Broadcast the model in parallel or on a signle thread.

        Returns:
            Dict[str, torch.Tensor]: The broadcasted model's weights.
        """

        self.broadcast_model = copy.deepcopy(self.global_model.state_dict())
        if threaded:
            threads: List[threading.Thread] = []
            for client in self.selected_clients:
                threads.append(threading.Thread(target = client.receive_global_model, args = (self.broadcast_model, round)))

            [t.start() for t in threads]
            [t.join() for t in threads]
        else:
            for client in self.selected_clients:
                client.receive_global_model(self.broadcast_model, round)
            pass
        return self.broadcast_model
    
    def collect_updates(self, threaded: bool = config.SIM_THREADED) -> None:
        """
        Tell clients to train their local model and collect update of each client 
        participating in the current round.

        Args:
            threaded (bool = config.SIM_THREADED): Train the clients on different threads or on a single one.
        """
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

        self.training_loss.append(training_loss)

    def aggregate(self) -> None:
        """
        Aggregate all the received updates into one model. The default aggregation method is FedAvg.
        """

        new_state = AggregationService.fed_avg(self.received_updates)
        self.global_model.load_state_dict(new_state)

        self.current_round += 1
        return

    def run(self, client_fraction: float = 1.0) -> None:
        """
        Run the simulation. For each round, broadcast the model, collect updates and aggregate.

        Args:
            client_fraction (float = 1.0): The fraction of clients that will participate in each round.
        """
        for round in tqdm(range(1, self.max_rounds + 1), desc = 'Round'):
            self.select_clients(client_fraction)
            self.broadcast(round = round)
            self.collect_updates()
            self.aggregate()

        return
    
    def run_test(self, dataset_index: int = 1, days_count: int = 10) -> None:
        """
        Evaluate the model with the test dataset.

        Args:
            dataset_index (int = 1): Index of the used dataset for testing the global model.
            days_count (int = 10): Number of days the model will predict the electric consumption.
        """

        self.global_model = self.global_model.to(device = config.DEVICE)
        self.global_model.eval()

        # Get data
        _tensor: torch.Tensor = torch.load(f'data/processed/test/iid/building_{dataset_index}.pt')
        features: torch.Tensor = _tensor[:, :-3]
        targets: torch.Tensor = _tensor[:, -3:]
        dataset: EnergyDataset = EnergyDataset(features, targets)

        with torch.no_grad():
            features, targets = dataset[:days_count*48]
            features = features.to(device = config.DEVICE)
            targets = targets.to(device = config.DEVICE)

            predictions: torch.Tensor = self.global_model(features)

            # Save load, pv and net predictions
            self.test_predictions = {
                'load': predictions[:, 0].tolist(),
                'pv': predictions[:, 1].tolist(),
                'net': predictions[:, 2].tolist(),
                'load_true': targets[:, 0].tolist(),
                'pv_true': targets[:, 1].tolist(),
                'net_true': targets[:, 2].tolist(),
            }

        return

    def save_metrics(self, filename: str) -> Dict[str, any]:
        """
        Save the metrics to a given filename, in JSON format.
        
        Args:
            filename (str): The name of the file the metrics will be saved under.
        """
        # Convert attributes to dict
        metrics: Dict[str, any] = {
            'training_loss': self.training_loss,
            'MAE': {
                k: mean_absolute_error(self.test_predictions[f'{k}_true'], self.test_predictions[k])
                for k in ['load', 'pv', 'net']
            }, 'MSE': {
                k: mean_squared_error(self.test_predictions[f'{k}_true'], self.test_predictions[k])
                for k in ['load', 'pv', 'net']
            }, 'RMSE': {
                k: root_mean_squared_error(self.test_predictions[f'{k}_true'], self.test_predictions[k])
                for k in ['load', 'pv', 'net']
            }
        }

        # Save dict to file
        with open(f'{config.SAVE_DATA_PATH}/{filename}.json', mode = 'w', encoding = 'utf-8') as f:
            f.write(json.dumps(metrics, indent = 4))
        return metrics # Returning metrics in case we need directly after computing

def check_server():
    logger.info('Starting server check')

    from app.models.model import NormalMLP
    server: Server = Server(global_model = NormalMLP(), max_rounds = 5)
    
    # Register clients
    server.register_client(Client(client_id = 1, model = NormalMLP(), batch_size = 128, local_epochs = 4))
    server.register_client(Client(client_id = 2, model = NormalMLP(), batch_size = 128, local_epochs = 4))
        
    server.run(client_fraction = 1)

    logger.info(f'Starting test phase')
    server.run_test(dataset_index = 1, days_count = 10)

    server.save_model('check_server')
    server.save_metrics('check_server')

    _s: Server = Server(NormalMLP())
    _s.load_metrics('check_server')

    assert _s.training_loss is not None
    assert len(_s.test_predictions) == len(server.test_predictions)

    logger.info('Server check ended successfully')