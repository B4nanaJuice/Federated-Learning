# Imports
import torch
import random as rd
import torch.nn as nn
from typing import List, Dict, Callable

from config import create_logger

logger = create_logger(__name__)

class MaliciousEntity:
    """
    Base class for malicious or attacked entities like malicious Client or Attacked server. Each entity
    will poison their data or model.

    Attributes:
        attack_rate (float | Callable = .2): How often the attack will happen. If `attack_rate` is set to a float between 0 and 1, then each round, a random number will be generated. If that number is below the `attack_rate`, the entity will poison the target. If the `attack_rate` is set to a Callable, then the poison will happen depending on the output of the method.
        attack_method (str = 'gaussian_weights'): The method used for the poisoning. Available values are : 'gaussian noise' to add a gaussian noise to the values, 'gaussian_weights' to replaces the values by a random value following a gaussian distribution, 'uniform_noise' to add a uniform noise to the values, 'uniform_weights' to replace the values by a randopm value following a uniform distribution, 'gradient_inversion' to invert the sign of the values or 'gradient_amplification' to multiply the values by a certain coefficient.
        attacked_rounds (List[int]): A list of the rounds when the entity poisoned the data.
        partial_attack (bool = False): Choose whether the entity poisons the entire model or only the last layers.

    Examples:
        `MaliciousEntity(attack_rate = lambda x: x == 5)`: The entity will poison its data only at round 5
        `MaliciousEntity(attack_rate = lambda x: x in range(5, 9))`: The entity will poison its data only at rounds 5, 6, 7, and 8
    """
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
        """
        Poison the model

        Args:
            model (nn.Module): The model to poison
            attack_method (str): The method used for poisoning the model
            partial (bool): Choose to poison only the last layers of the model or not
            coef (float = 1): The coefficient used to multiply the values
        
        Returns:
            Dict[str, torch.Tensor]: The poisonned model
        """
        
        model: Dict[str, torch.Tensor] = model.state_dict()
        keys: List = list(model.keys())
        target_keys: List = keys[-4:] if partial else keys

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
        """
        Decide if the entity can poison its data or model during the current round.

        Returns:
            bool: If the entity can attack
        """
        round_value = getattr(self, 'current_round', None)

        if callable(self.attack_rate) and round_value is not None:
            return self.attack_rate(round_value)
        return rd.random() < self.attack_rate
    
    def send_attacked_rounds(self) -> List[int]:
        """
        Send the list containing the rounds when the entity poisoned its data or model.

        Returns:
            List[int]: The list with attacked rounds
        """
        return self.attacked_rounds