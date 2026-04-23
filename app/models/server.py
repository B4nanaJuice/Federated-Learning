# Imports
import copy
import json
import torch
import threading
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Optional, Callable, Dict, List
from sklearn.metrics import mean_squared_error

from app.models.client import Client
from app.models.malicious_client import MaliciousClient
from app.models.model import NormalMLP, SoftGatedMoE
from app.models.dataloader import EnergyDataset
from config import create_logger, config

logger = create_logger(__name__)


class Server:
    def __init__(self,
                 global_model: nn.Module,
                 max_rounds: int = 50,
                 min_clients: int = 2,
                 aggregation_function: Optional[Callable] = None,
                 **kwargs
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
        self.MAE: Dict[str, List[float]] = {}
        self.RMSE: Dict[str, List[float]] = {}

        # Test phase
        self.test_predictions: Dict[str, List[float]] = {}
        self.test_MSE: Dict[str, float] = {}

    def register_client(self, client: Client) -> None:
        self.client_registry[client.client_id] = client
        return
    
    def register_clients(self, clients: List[Client]) -> None:
        for _client in clients:
            self.register_client(_client)
        return

    def select_clients(self, fraction: float = 1.0) -> List[Client]:
        import random as rd
        k = max(self.min_clients, int(len(self.client_registry) * fraction))
        k = min(k, len(self.client_registry))
        self.selected_clients = rd.sample(list(self.client_registry.values()), k)
        self.participation_rate = len(self.selected_clients) / len(self.client_registry)
        return self.selected_clients
    
    def broadcast(self, round: int) -> Dict[str, torch.Tensor]:
        self.broadcast_model = copy.deepcopy(self.global_model.state_dict())
        for client in self.selected_clients:
            client.receive_global_model(self.broadcast_model, round)
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
            self.MAE[client.client_id] = self.MAE.get(client.client_id, []) + update.get('MAE', [])
            self.RMSE[client.client_id] = self.RMSE.get(client.client_id, []) + update.get('RMSE', [])

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
            self.broadcast(round = round)
            self.collect_updates()
            self.aggregate()

            if round % 10 == 0:
                self.save_checkpoint()

        return
    
    def run_test(self, dataset_index: int = 1, days_count: int = 10) -> None:

        self.global_model = self.global_model.to(device = config.DEVICE)
        self.global_model.eval()

        # Get data
        _tensor: torch.Tensor = torch.load(f'data/processed/test/building_{dataset_index}.pt')
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

        # Compute MSE for load, pv and net consumption
        self.test_MSE = {
            'load': mean_squared_error(self.test_predictions['load_true'], self.test_predictions['load']),
            'pv': mean_squared_error(self.test_predictions['pv_true'], self.test_predictions['pv']),
            'net': mean_squared_error(self.test_predictions['net_true'], self.test_predictions['net'])
        }

        return
    
    def plot_test_loss(self, plot: plt.Axes) -> None:
        x = np.linspace(1, len(self.training_loss), len(self.training_loss))
        min_loss = [min(_) for _ in self.training_loss]
        max_loss = [max(_) for _ in self.training_loss]
        mean_loss = [sum(_)/len(_) for _ in self.training_loss]          

        plot.fill_between(x, min_loss, max_loss, color = "#bcbcbc", label = 'MSE Range')
        plot.plot(x, mean_loss, label = 'Average MSE Loss', color = '#133E71')

        # If some registered clients are malicious, get their attacked rounds
        client_attacked_rounds: List[int] = []
        for client in self.client_registry.values():
            if type(client) == MaliciousClient:
                logger.info(f'Client {client.client_id} attacked rounds: {client.send_attacked_rounds()}')
                if len(client.send_attacked_rounds()) > 0:
                    client_attacked_rounds += client.send_attacked_rounds()

        if len(client_attacked_rounds) > 0:
            plot.vlines(client_attacked_rounds, min(min_loss), max(max_loss), linestyles = 'dashed', label = 'Client attack', linewidths = 1, color = '#8AB425')

        # If aggregation server is attacked, get its attacked rounds
        server_attacked_rounds: List[int] | None = getattr(self, 'attacked_rounds', None)
        if server_attacked_rounds:
            plot.vlines(server_attacked_rounds, min(min_loss), max(max_loss), linestyles = 'dashed', label = 'Server attack', linewidth = 1, color = '#F59A00')

        if server_attacked_rounds or len(client_attacked_rounds) > 0:
            first_attack_round: int = min(min(server_attacked_rounds or [float('inf')]), min(client_attacked_rounds + [float('inf')]))
            average_before_attack: list[float] = mean_loss[:first_attack_round]
            average_before_attack: float = sum(average_before_attack)/len(average_before_attack)

            plot.hlines(average_before_attack, 0, self.current_round, linestyles = 'dashed', label = 'Average loss before attack', linewidth = 1, color = '#000000')


        # Hide axes
        plot.spines['top'].set_visible(False)
        plot.spines['right'].set_visible(False)
        plot.legend()
        plot.set_xlabel('Round')
        plot.set_ylabel('Mean Squared Error (MSE) loss')
        plot.set_title('Training MSE loss over rounds')
        plot.set_yscale('log')
        return

    def plot_MAE(self, plot: plt.Axes) -> None:
        plot.boxplot(self.MAE.values(), medianprops = {'color': '#F59A00'})

        plot.spines['top'].set_visible(False)
        plot.spines['right'].set_visible(False)
        plot.set_xlabel('Client ID')
        plot.set_ylabel('Mean Absolute Error (MAE)')
        plot.set_title('Mean Absolute Error for server\'s clients')
        plot.grid(axis = 'y')
        plot.set_xticklabels([str(_) for _ in self.RMSE.keys()])
        plot.set_yscale('log')
        return
    
    def plot_RMSE(self, plot: plt.Axes) -> None:
        plot.boxplot(self.RMSE.values(), medianprops = {'color': '#F59A00'})

        plot.spines['top'].set_visible(False)
        plot.spines['right'].set_visible(False)
        plot.set_xlabel('Client ID')
        plot.set_ylabel('Root Mean Squared Error (RMSE)')
        plot.set_title('Root Mean Squared Error for server\'s clients')
        plot.grid(axis = 'y')
        plot.set_xticklabels([str(_) for _ in self.RMSE.keys()])
        plot.set_yscale('log')
        return
    
    def plot_pred(self, plot: plt.Axes, column: str) -> None:
        _len: int = len(self.test_predictions[column])
        x = np.linspace(1, _len, _len)

        plot.plot(x, self.test_predictions[f'{column}_true'], label = f'{column} truth', color = '#133E71')
        plot.plot(x, self.test_predictions[column], label = f'{column} prediction', color = '#009FE3')

        plot.spines['top'].set_visible(False)
        plot.spines['right'].set_visible(False)
        plot.legend()
        plot.set_xlabel('$\\Delta t$ (30 minutes)')
        plot.set_ylabel('Normalized value')
        plot.set_title(f'{column.capitalize()} prediction vs. ground truth (MSE: {self.test_MSE[column]:.4f})')
        return

    def plot(self) -> None:
        # Generate layout from plot_* methods
        fig = plt.figure()
        gs = mpl.gridspec.GridSpec(3, 2, wspace = .25, hspace = .5)

        # === PLOT LOSS ===
        loss_plot = fig.add_subplot(gs[0, 1])
        self.plot_test_loss(loss_plot)

        # === PLOT MAE ===
        mae_plot = fig.add_subplot(gs[1, 1])
        self.plot_MAE(mae_plot)

        # === PLOT RMSE ===
        rmse_plot = fig.add_subplot(gs[2, 1])
        self.plot_RMSE(rmse_plot)

        # === PLOT PREDICTIONS ===
        load_plot = fig.add_subplot(gs[0, 0])
        self.plot_pred(load_plot, 'load')
        pv_plot = fig.add_subplot(gs[1, 0])
        self.plot_pred(pv_plot, 'pv')
        net_plot = fig.add_subplot(gs[2, 0])
        self.plot_pred(net_plot, 'net')

        plt.show()
        return
    
    def save_model(self, filename: str) -> None:
        torch.save(self.global_model.state_dict(), f'{config.SAVE_DATA_PATH}/{filename}.pt')
        return

    def save_metrics(self, filename: str) -> None:
        # Convert attributes to dict
        metrics: Dict[str, any] = {
            'predictions': self.test_predictions,
            'test_MSE': self.test_MSE,
            'training_loss': self.training_loss,
            'MAE': self.MAE,
            'RMSE': self.RMSE
        }

        # Save dict to file
        with open(f'{config.SAVE_DATA_PATH}/{filename}.json', mode = 'w', encoding = 'utf-8') as f:
            f.write(json.dumps(metrics))
        return

    def load_model(self, filename: str) -> None:
        self.global_model.load_state_dict(torch.load(f'{config.SAVE_DATA_PATH}/{filename}.pt', weights_only = True))
        return

    def load_metrics(self, filename: str) -> None:
        # Load data from file
        with open(f'{config.SAVE_DATA_PATH}/{filename}.json', mode = 'r', encoding = 'utf-8') as f:
            metrics: Dict[str, any] = json.loads(f.read())

        # Load metrics from dict
        self.test_predictions = metrics.get('predictions', {})
        self.test_MSE = metrics.get('test_MSE', {})
        self.training_loss = metrics.get('training_loss', [])
        self.MAE = metrics.get('MAE', {})
        self.RMSE = metrics.get('RMSE', {})
        return

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
    # server.plot()

    server.save_model('check_server')
    server.save_metrics('check_server')

    _s: Server = Server(NormalMLP())
    _s.load_metrics('check_server')
    # _s.plot()

    assert _s.training_loss is not None
    assert _s.test_MSE is not None
    assert len(_s.test_predictions) == len(server.test_predictions)
    assert len(_s.MAE) == len(server.MAE), f'Size of _s.MAE should be the same as origin server\'s MAE ({len(_s.MAE)} =/= {len(server.MAE)})'

    logger.info('Server check ended successfully')