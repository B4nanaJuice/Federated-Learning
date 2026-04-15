# Imports
import copy
import time
import torch
import torch.nn as nn
from typing import Optional, Dict

from app.models.dataloader import EnergyDataset
from app.models.model import NormalMLP, SoftGatedMoE
from config import create_logger

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
        self.delta_weights: Optional[Dict[int, torch.Tensor]] = None

        # Local metrics
        self.train_loss: float = float('inf')
        self.compute_time: float = 0.0

    def receive_global_model(self, global_weights: Dict[int, torch.Tensor]) -> None:
        """
        Receive global model weights from the server and update the local model.
        
        Args:
            global_weights (Dict[int, torch.Tensor]): The global model weights received from the server.
        """
        self.model.load_state_dict(copy.deepcopy(global_weights))
        self.round_id += 1
        return
    
    def train_local(self) -> None:
        t0: float = time.time()
        weights_before: Dict[int, torch.Tensor] = copy.deepcopy(self.model.state_dict())
        
        self.model = SoftGatedMoE()

        self.model.train()
        for _ in range(self.local_epochs):
            logger.info(f'Client {self.client_id} - Epoch {_+1}/{self.local_epochs}')
            epoch_loss: float = 0.0
            for batch in range(len(self.dataset) // self.batch_size + 1):
                # Get batch data
                start_idx: int = batch * self.batch_size
                end_idx: int = min((batch + 1) * self.batch_size, len(self.dataset))
                x_batch, y_batch = self.dataset[start_idx:end_idx]

                # Forward pass
                self.optimizer.zero_grad()
                predictions: torch.Tensor = self.model(x_batch)
                assert not torch.isnan(predictions).any().item(), 'Predictions contain NaN'
                loss: torch.Tensor = self.loss_function(predictions, y_batch)
                epoch_loss += loss.item() * len(x_batch)
                logger.debug(f'loss: {loss}')
                logger.debug(f'Loss item: {loss.item()}, len(x_batch): {len(x_batch)}, epoch_loss: {epoch_loss}')

                # Backward pass
                loss.backward()
                self.optimizer.step()

            self.train_loss = epoch_loss / self.num_samples
            logger.info(f'Epoch training loss: {self.train_loss}')

        weights_after: Dict[int, torch.Tensor] = copy.deepcopy(self.model.state_dict())
        self.delta_weights = {
            key: weights_after[key] - weights_before[key] 
            for key in weights_before
        }

        # Add differntial privacy noise to delta weights
        # Add compression to delta weights

        self.compute_time = time.time() - t0
        return
    
    def send_update(self) -> Dict:
        return {
            'client_id': self.client_id,
            'round_id': self.round_id,
            'num_samples': self.num_samples,
            'delta_weights': self.delta_weights,
            'train_loss': self.train_loss
        }

# Method to check if client's training works
def check_client():
    client: Client = Client(client_id = 1)
    client.train_local()

    logger.info(f'Model training time: {client.compute_time:.1f}s')
