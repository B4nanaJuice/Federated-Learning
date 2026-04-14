# Imports
import torch
from torch.utils.data import Dataset

from config import config

# Create EnergyDataset class from Dataset
class EnergyDataset(Dataset):
    def __init__(self, 
                 x: torch.Tensor, 
                 y: torch.Tensor, 
                 lookback: int = config.LOOKBACK):
        
        self.x: torch.Tensor = x
        self.y: torch.Tensor = y
        self.lookback: int = lookback

    def __len__(self) -> int:
        return len(self.x) - self.lookback + 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.x[idx:idx + self.lookback]
        y = self.y[idx + self.lookback]
        return x, y