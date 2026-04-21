# Imports
import copy
import time
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, List
import matplotlib.pyplot as plt

from app.models.dataloader import EnergyDataset
from app.models.model import NormalMLP, SoftGatedMoE
from app.models.utils import EarlyStopper
from config import create_logger, config

logger = create_logger(__name__)

class Client:
    def __init__(self, 
                 client_id: int, 
                 model: NormalMLP | SoftGatedMoE = NormalMLP(), 
                 local_epochs: int = 5, 
                 batch_size: int = 32,
                 learning_rate: float = 0.001,
                 **kwargs
                 ):
        
        # Identity
        self.client_id: int = client_id
        self.round_id: int = 0

        # Local data
        self._train_tensor: torch.Tensor = torch.load(f'data/processed/train/building_{client_id}.pt')
        self._train_features: torch.Tensor = self._train_tensor[:, :-3]
        self._train_targets: torch.Tensor = self._train_tensor[:, -3:]
        self.train_dataset: EnergyDataset = EnergyDataset(self._train_features, self._train_targets)

        self._validation_tensor: torch.Tensor = torch.load(f'data/processed/val/building_{client_id}.pt')
        self._validation_features: torch.Tensor = self._validation_tensor[:, :-3]
        self._validation_targets: torch.Tensor = self._validation_tensor[:, -3:]
        self.validation_dataset: EnergyDataset = EnergyDataset(self._validation_features, self._validation_targets)

        self.num_samples: int = len(self.train_dataset)

        # Local model
        self.model: NormalMLP | SoftGatedMoE = copy.deepcopy(model)
        self.loss_function: nn.MSELoss = nn.MSELoss()
        self.local_epochs: int = local_epochs
        self.batch_size: int = batch_size
        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate)

        # Local metrics
        self.train_loss: float = float('inf')
        self.compute_time: float = 0.0
        self.hist_train_loss: List[float] = []
        self.hist_validation_loss: List[float] = []

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

        early_stopper: EarlyStopper = EarlyStopper(patience = 5, min_delta = 1e-3)
        self.model = self.model.to(device = config.DEVICE)
        self.model.train()

        for _ in range(self.local_epochs):
            epoch_loss: float = 0.0

            for batch in range(len(self.train_dataset) // self.batch_size + 1):
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
            self.hist_train_loss.append(self.train_loss)

            # Validation
            with torch.no_grad():
                x_val, y_val = self.validation_dataset[:]
                x_val, y_val = x_val.to(device = config.DEVICE), y_val.to(device = config.DEVICE)

                predictions = self.model(x_val)
                loss = self.loss_function(predictions, y_val)
                val_loss = loss.item()
                self.hist_validation_loss.append(val_loss)

                # Test if stop early
                if early_stopper.early_stop(val_loss):
                    logger.info(f'Client {self.client_id} stopped early at epoch {_+1}.')
                    break

        # Add differntial privacy noise to delta weights
        # Add compression to delta weights

        self.compute_time = time.time() - t0

        return

    def get_batch(self, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        start_idx: int = batch * self.batch_size
        end_idx: int = min((batch + 1) * self.batch_size, len(self.train_dataset))
        x_batch, y_batch = self.train_dataset[start_idx:end_idx]
        return x_batch, y_batch

    def send_update(self) -> Dict:
        return {
            'client_id': self.client_id,
            'round_id': self.round_id,
            'num_samples': self.num_samples,
            'weights': copy.deepcopy(self.model.state_dict()),
            'train_loss': self.train_loss
        }
    
    def plot(self) -> None:
        
        x: List[int] = list(range(1, len(self.hist_train_loss) + 1))
        plt.plot(x, self.hist_train_loss, label = 'Train Loss', color = '#133E71')
        plt.plot(x, self.hist_validation_loss, label = 'Validation Loss', color = '#009FE3')

        plt.legend()
        plt.grid(visible = True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

        return

# Method to check if client's training works
def check_client():
    logger.info('Starting client check')

    client: Client = Client(client_id = 1, batch_size = 64, local_epochs = 300)
    client.train_local()

    logger.info(f'Model training time: {client.compute_time:.1f}s')
    logger.info(f'Model MSE loss: {client.train_loss:.4f}')

    client.plot()

    logger.info('Client check ended successfully')
