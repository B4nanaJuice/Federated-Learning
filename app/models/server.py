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
        self.test_predictions: Dict = {}
        self.test_MSE: Dict = {}

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
            'load': np.square(np.subtract(predictions[:, 0].cpu(), targets[:, 0].cpu())).mean().item(),
            'pv': np.square(np.subtract(predictions[:, 1].cpu(), targets[:, 1].cpu())).mean().item(),
            'net': np.square(np.subtract(predictions[:, 2].cpu(), targets[:, 2].cpu())).mean().item()
        }

        return
    
    def plot_test_loss(self, plot: plt.Axes) -> None:
        x = np.linspace(1, self.max_rounds, self.max_rounds)
        min_loss = [min(_) for _ in self.training_loss]
        max_loss = [max(_) for _ in self.training_loss]
        mean_loss = [sum(_)/len(_) for _ in self.training_loss]

        # If some registered clients are malicious, get their attacked rounds
        attacked_rounds: Dict = {}
        for client in self.client_registry.values():
            if type(client) == MaliciousClient:
                logger.info(f'Client {client.client_id} attacked rounds: {client.send_attacked_rounds()}')
                if len(client.send_attacked_rounds()) > 0:
                    attacked_rounds[client.client_id] = client.send_attacked_rounds()
                

        plot.fill_between(x, min_loss, max_loss, color = "#bcbcbc")
        plot.plot(x, mean_loss, label = 'Average MSE Loss', color = '#133E71')

        for k, v in attacked_rounds.items():
            plot.vlines(v, min(min_loss), max(max_loss), linestyles = 'dashed', label = f'Attack', linewidths = 1, color = '#8AB425')

        # Hide axes
        plot.spines['top'].set_visible(False)
        plot.spines['right'].set_visible(False)
        plot.legend()
        plot.set_xlabel('Round')
        plot.set_ylabel('Mean Squared Error (MSE) loss')
        plot.set_title('Training MSE loss over rounds')
        return

    def plot_MAE(self, plot: plt.Axes) -> None:
        plot.boxplot(self.MAE.values())

        plot.spines['top'].set_visible(False)
        plot.spines['right'].set_visible(False)
        plot.set_xlabel('Client ID')
        plot.set_ylabel('Mean Absolute Error (MAE)')
        plot.set_title('Mean Absolute Error for server\'s clients')
        plot.grid(axis = 'y')
        plot.set_xticklabels([str(_) for _ in self.RMSE.keys()])
        return
    
    def plot_RMSE(self, plot: plt.Axes) -> None:
        plot.boxplot(self.RMSE.values())

        plot.spines['top'].set_visible(False)
        plot.spines['right'].set_visible(False)
        plot.set_xlabel('Client ID')
        plot.set_ylabel('Root Mean Squared Error (RMSE)')
        plot.set_title('Root Mean Squared Error for server\'s clients')
        plot.grid(axis = 'y')
        plot.set_xticklabels([str(_) for _ in self.RMSE.keys()])
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
        plot.set_title(f'{column.capitalize()} prediction vs. ground truth')
        return

    def plot(self) -> None:
        # Generate layout from show_* options
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

    def to_dict(self) -> Dict:
        return {
            'current_round': self.current_round,
            'max_rounds': self.max_rounds,
            'min_clients': self.min_clients,

            'client_registry': [_ for _ in self.client_registry],

            'global_model': str(type(self.global_model)),

            'training_loss': self.training_loss,
            'test_predictions': self.test_predictions,
            'test_MSE': self.test_MSE
        }
    
    @staticmethod
    def save_state(server: Server, filename: str = 'server_save') -> None:
        if filename.endswith('.json'):
            filename = filename.replace('.json', '')

        # Save the model
        torch.save(server.global_model.state_dict(), f'{config.SAVE_DATA_PATH}/{filename}_model.pt')

        # Save the server state
        with open(f'{config.SAVE_DATA_PATH}/{filename}_state.json', mode = 'w', encoding = 'utf-8') as f:
            f.write(json.dumps(server.to_dict()))
        return
    
    def save_model(self, filename: str) -> None:
        torch.save(self.global_model.state_dict(), f'{config.SAVE_DATA_PATH}/{filename}.pt')
        return
    
    def save_metrics(self, filename: str) -> None:
        return
    
    @staticmethod
    def load_state(filename: str = 'server_save') -> Server:
        if filename.endswith('.json'):
            filename = filename.replace('.json', '')

        # Load the server
        with open(f'{config.SAVE_DATA_PATH}/{filename}_state.json', mode = 'r', encoding = 'utf-8') as f:
            server_data: Dict = json.load(f)
        server: Server = Server(
            global_model = NormalMLP() if 'NormalMLP' in server_data.get('global_model') else SoftGatedMoE(),
            max_rounds = server_data.get('max_rounds', 20),
            min_clients = server_data.get('min_clients', 2)
        )

        # Load metrics
        server.training_loss = server_data.get('training_loss', [])
        server.test_predictions = server_data.get('test_predictions', {})
        server.test_MSE = server_data.get('test_MSE', {})

        # Load model
        server.global_model.load_state_dict(torch.load(f'{config.SAVE_DATA_PATH}/{filename}_model.pt', weights_only = True))

        return server

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
    server.plot()
    
    logger.info('Server check ended successfully')