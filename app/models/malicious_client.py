# Imports
import random as rd
from typing import Dict, List
import torch

from app.models.client import Client
from config import create_logger

logger = create_logger(__name__)

class MaliciousClient(Client):
    def __init__(self, 
                 client_id: int, 
                 attack_rate: float = 0.01, 
                 attack_target: str = 'data', 
                 attack_method: str = 'random_noise', 
                 **client_args
                 ):
        
        super().__init__(client_id, **client_args)
        self.attack_rate: float = attack_rate
        self.attack_target: str = attack_target
        self.attack_method: str = attack_method
        self.attacked_rounds: List[int] = []
    
    @staticmethod
    def poison_data(x: torch.Tensor, y: torch.Tensor, attack_method: str) -> tuple[torch.Tensor, torch.Tensor]:

        match attack_method:
            case 'random_noise':
                x += torch.randn_like(x)
                y += torch.randn_like(y)
                # Clip values (only y because x has normal distributed values)
                y[y > 1] = 1
                y[y < 0] = 0

                return x, y
        
            case _:
                logger.warning(f'Unknown attack method {attack_method}.')
                return x, y

    @staticmethod
    def poison_model(model: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return model

    def can_attack(self) -> bool:
        rd.seed(self.round_id * self.client_id * self.attack_rate)
        return rd.random() < self.attack_rate
    
    def get_batch(self, batch: int) -> tuple[torch.Tensor, torch.Tensor]:

        x_batch, y_batch = super().get_batch(batch)

        if self.attack_target == 'data' and self.can_attack():
            x_batch, y_batch = self.poison_data(x_batch, y_batch, self.attack_method)
            self.attacked_rounds.append(self.round_id) if self.round_id not in self.attacked_rounds else None

        return x_batch, y_batch
    
    def send_update(self) -> Dict:
        
        if self.attack_target == 'model' and self.can_attack():
            self.model = self.poison_model(self.model)
            self.attacked_rounds.append(self.round_id)
        
        return super().send_update()
    
    def send_attacked_rounds(self) -> List[int]:
        return self.attacked_rounds