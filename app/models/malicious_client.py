# Imports
import random as rd
from typing import Dict, List
import torch
import torch.nn as nn

from app.models.client import Client
from config import create_logger

logger = create_logger(__name__)

class MaliciousClient(Client):
    def __init__(self, 
                 client_id: int, 
                 attack_rate: float = 0.01, 
                 attack_method: str = 'uniform_weights', 
                 **client_args
                 ):
        
        super().__init__(client_id, **client_args)
        self.attack_rate: float = attack_rate
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
        return rd.random() < self.attack_rate
    
    def send_update(self) -> Dict:
        
        if self.can_attack():
            self.model.load_state_dict(self.poison_model(self.model, self.attack_method))
            self.attacked_rounds.append(self.round_id)
        
        return super().send_update()
    
    def send_attacked_rounds(self) -> List[int]:
        return self.attacked_rounds
    
def check_malicious_client():
    logger.info('Starting malicious client check')

    client: MaliciousClient = MaliciousClient(client_id = 1, batch_size = 128, attack_rate = .5)
    client.train_local()

    logger.info(f'Model training time: {client.compute_time:.1f}s')
    logger.info(f'Model MSE loss: {client.train_loss:.4f}')
    logger.info('Malicious client check ended successfully')