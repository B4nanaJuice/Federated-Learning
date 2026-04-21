# Imports
import random as rd
from typing import Dict, List
import torch
import torch.nn as nn

from app.models.client import Client
from app.models.malicious_entity import MaliciousEntity
from config import create_logger

logger = create_logger(__name__)

class MaliciousClient(Client, MaliciousEntity):
    def __init__(self, client_id: int, **kwargs):

        Client.__init__(self, client_id, **kwargs)
        MaliciousClient.__init__(self, **kwargs)
    
    def send_update(self) -> Dict:
        
        if self.can_attack():
            self.model.load_state_dict(self.poison_model(self.model, self.attack_method))
            self.attacked_rounds.append(self.round_id)
        
        # return super(Client, self).send_update()
        return super().send_update()
    
def check_malicious_client():
    logger.info('Starting malicious client check')

    client: MaliciousClient = MaliciousClient(client_id = 1, batch_size = 128, attack_rate = .5)
    client.train_local()

    logger.info(f'Model training time: {client.compute_time:.1f}s')
    logger.info(f'Model MSE loss: {client.train_loss:.4f}')
    logger.info('Malicious client check ended successfully')