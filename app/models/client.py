# Imports
import copy
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict

from app.models.dataloader import EnergyDataset
from app.models.model import NormalMLP, SoftGatedMoE
from config import create_logger, config

logger = create_logger(__name__)

class Client:
    def __init__(self, 
                 client_id: int, 
                 model: NormalMLP | SoftGatedMoE = NormalMLP(), 
                 local_epochs: int = 5, 
                 batch_size: int = 32,
                 learning_rate: float = 0.001
                 ):
        
        # Identity
        self.client_id: int = client_id
        self.round_id: int = 0

        # Local data
        self._tensor: torch.Tensor = torch.load(f'data/processed/train/building_{client_id}.pt')
        self._features: torch.Tensor = self._tensor[:, :-3]
        self._targets: torch.Tensor = self._tensor[:, -3:]
        self.dataset: EnergyDataset = EnergyDataset(self._features, self._targets)
        self.num_samples: int = len(self.dataset)

        # Local model
        self.model: NormalMLP | SoftGatedMoE = copy.deepcopy(model)
        self.loss_function: nn.MSELoss = nn.MSELoss()
        self.local_epochs: int = local_epochs
        self.batch_size: int = batch_size
        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate)

        # Local metrics
        self.train_loss: float = float('inf')
        self.compute_time: float = 0.0

    def receive_global_model(self, global_weights: Dict[int, torch.Tensor], round_id: int) -> None:
        """
        Receive global model weights from the server and update the local model.
        
        Args:
            global_weights (Dict[int, torch.Tensor]): The global model weights received from the server.
        """
        self.model.load_state_dict(copy.deepcopy(global_weights))
        self.round_id = round_id
        return
    
    def train_local(self) -> None:
        t0: float = time.time()

        self.model = self.model.to(device = config.DEVICE)
        self.model.train()

        for _ in tqdm(range(self.local_epochs), desc = f'Client {self.client_id:2d}'):
            epoch_loss: float = 0.0

            for batch in range(len(self.dataset) // self.batch_size + 1):
                # Get batch data
                x_batch, y_batch = self.get_batch(batch)
                # Move batches to device (GPU if available)
                x_batch, y_batch = x_batch.to(device = config.DEVICE), y_batch.to(device = config.DEVICE)

                # Forward pass
                self.optimizer.zero_grad()
                predictions: torch.Tensor = self.model(x_batch)
                loss: torch.Tensor = self.loss_function(predictions, y_batch)
                epoch_loss += loss.item() * len(x_batch)

                # Backward pass
                loss.backward()
                self.optimizer.step()

            self.train_loss = epoch_loss / self.num_samples

        # Add differntial privacy noise to delta weights
        # Add compression to delta weights

        self.compute_time = time.time() - t0

        return

    def get_batch(self, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        start_idx: int = batch * self.batch_size
        end_idx: int = min((batch + 1) * self.batch_size, len(self.dataset))
        x_batch, y_batch = self.dataset[start_idx:end_idx]
        return x_batch, y_batch

    def send_update(self) -> Dict:
        return {
            'client_id': self.client_id,
            'round_id': self.round_id,
            'num_samples': self.num_samples,
            'weights': copy.deepcopy(self.model.state_dict()),
            'train_loss': self.train_loss
        }

# Method to check if client's training works
def check_client():
    logger.info('Starting client check')

    client: Client = Client(client_id = 1, batch_size = 64)
    client.train_local()

    logger.info(f'Model training time: {client.compute_time:.1f}s')
    logger.info('Client check ended successfully')
