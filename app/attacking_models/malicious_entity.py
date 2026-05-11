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
                 attack_method: str = 'gaussian_weights',
                 partial_attack: bool = False,
                 **kwargs
                ):

        self.attack_rate: float | Callable = attack_rate
        self.attack_method: str = attack_method
        self.attacked_rounds: List[int] = []
        self.partial_attack: bool = partial_attack

    @staticmethod
    def poison_model(model: nn.Module, attack_method: str, partial: bool, coef: float = 1) -> Dict[str, torch.Tensor]:
        
        model: Dict[str, torch.Tensor] = model.state_dict()
        keys: List = list(model.keys())
        target_keys: List = keys[-2:] if partial else keys

        match attack_method:
            case 'gaussian_noise':
                fn = lambda layer: model[layer] + coef * torch.randn_like(model[layer])
            case 'gaussian_weights':
                fn = lambda layer: coef * torch.randn_like(model[layer])
            case 'uniform_noise':
                fn = lambda layer: model[layer] + coef * torch.rand_like(model[layer])
            case 'uniform_weights':
                fn = lambda layer: coef * torch.rand_like(model[layer])
            case 'gradient_inversion':
                fn = lambda layer: model[layer] * -coef
            case 'gradient_amplification':
                fn = lambda layer: model[layer] * coef
            case _:
                logger.warning(f'Unknown attack method {attack_method}.')
                return model
            
        for k in target_keys:
            model[k] = fn(k)

        return model
            
    def can_attack(self) -> bool:
        round_value = getattr(self, 'round_id', getattr(self, 'current_round', None))

        if callable(self.attack_rate) and round_value is not None:
            return self.attack_rate(round_value)
        return rd.random() < self.attack_rate
    
    def send_attacked_rounds(self) -> List[int]:
        return self.attacked_rounds