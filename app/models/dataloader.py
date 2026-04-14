# Imports
import torch
from torch.utils.data import Dataset

from config import config, create_logger

logger = create_logger(__name__)

class EnergyDataset(Dataset):
    def __init__(self, features: torch.Tensor, targets: torch.Tensor, lookback: int = config.LOOKBACK):
        self.x = features
        self.y = targets
        self.lookback = lookback

    def __len__(self) -> int:
        return len(self.x) - self.lookback + 1

    def __getitem__(self, idx: int | slice) -> tuple[torch.Tensor, torch.Tensor]:
        if type(idx) == int:
            x = self.x[idx:idx + self.lookback]
            y = self.y[idx:idx + self.lookback]
        elif type(idx) == slice:
            start, stop, step = idx.indices(len(self))
            x = torch.stack([self.x[i:i + self.lookback] for i in range(start, stop, step)])
            y = torch.stack([self.y[i:i + self.lookback] for i in range(start, stop, step)])
        return x, y
    
# Method for testing the dataset
def check_dataset():
    tensor: torch.Tensor = torch.load('data/processed/train/building_1.pt')
    features: torch.Tensor = tensor[:, :-3]
    targets: torch.Tensor = tensor[:, -3:]
    dataset = EnergyDataset(features, targets)

    assert len(dataset) == features.shape[0] - config.LOOKBACK + 1, f"Expected dataset length {features.shape[0] - config.LOOKBACK + 1}, got {len(dataset)}"
    # Test if can get multiple items without error
    x, y = dataset[:3]
    assert x.shape == (3, config.LOOKBACK, config.NUM_FEATURES), f"Expected x shape (3, LOOKBACK, NUM_FEATURES), got {x.shape}"
    assert y.shape == (3, config.LOOKBACK, 3), f"Expected y shape (3, LOOKBACK, 3), got {y.shape}"

    logger.info(f'Dataset: {len(dataset)} samples')
    logger.info(f'dataset[0] -> x shape: {dataset[0][0].shape}, y shape: {dataset[0][1].shape}')
    logger.info(f'dataset[:3] -> x shape: {x.shape}, y shape: {y.shape}')