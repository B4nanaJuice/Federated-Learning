# Imports
import torch
import random as rd
import torch.nn as nn
from typing import List, Dict, Callable

from config import create_logger

logger = create_logger(__name__)

class MaliciousEntity:
    def __init__(self, 
                 attack_rate: float | Callable = .2, 
                 attack_method: str = 'uniform_weights',
                 **kwargs
                ):
        self.attack_rate: float | Callable = attack_rate
        self.attack_method: str = attack_method
        self.attacked_rounds: List[int] = []

    @staticmethod
    def poison_model(model: nn.Module, attack_method: str) -> Dict[str, torch.Tensor]:
        
        model: Dict[str, torch.Tensor] = model.state_dict()

        match attack_method:
            case 'gaussian_noise':
                for k, layer in model.items():
                    model[k] += torch.randn_like(layer)

                return model

            case 'gaussian_weights':
                for k, layer in model.items():
                    model[k] = torch.randn_like(layer)

                return model
            
            case 'uniform_noise':
                for k, layer in model.items():
                    model[k] += torch.rand_like(layer)

                return model

            case 'uniform_weights':
                for k, layer in model.items():
                    model[k] = torch.rand_like(layer)

                return model

            case _:
                logger.warning(f'Unknown attack method {attack_method}.')
                return model
            
    def can_attack(self) -> bool:
        round_value = getattr(self, 'round_id', getattr(self, 'current_round', None))

        if callable(self.attack_rate) and round_value:
            return self.attack_rate(round_value)
        return rd.random() < self.attack_rate
    
    def send_attacked_rounds(self) -> List[int]:
        return self.attacked_rounds