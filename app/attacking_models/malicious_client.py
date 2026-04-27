# Imports
from typing import Dict
import torch

from app.models.client import Client
from app.attacking_models.malicious_entity import MaliciousEntity
from config import create_logger

logger = create_logger(__name__)

class MaliciousClient(Client, MaliciousEntity):
    def __init__(self, client_id: int | str, attack_target: str = 'model', **kwargs):

        Client.__init__(self, client_id, **kwargs)
        MaliciousEntity.__init__(self, **kwargs)
        self.attack_target: str = self.attack_target
    
    def send_update(self) -> Dict:
        
        if self.attack_target == 'model' and self.can_attack():
            self.model.load_state_dict(self.poison_model(self.model, self.attack_method, self.partial_attack))
            self.attacked_rounds.append(self.round_id)
        
        # return super(Client, self).send_update()
        return super().send_update()
    
    def get_batch(self, batch: int) -> tuple[torch.Tensor, torch.Tensor]:

        x_batch, y_batch = super().get_batch(batch = batch)

        if self.attack_method == 'data' and self.can_attack():
            match self.attack_method:
                case 'gaussian_noise':
                    x_batch += torch.randn_like(x_batch)
                case 'gaussian_weights':
                    x_batch = torch.randn_like(x_batch)
                case 'uniform_noise':
                    x_batch += torch.rand_like(x_batch)
                case 'uniform_weights':
                    x_batch = torch.rand_like(x_batch)
                case _:
                    logger.warning(f'Unknown attack method {self.attack_method}.')

        return x_batch, y_batch
    
def check_malicious_client():
    logger.info('Starting malicious client check')

    client: MaliciousClient = MaliciousClient(client_id = 1, batch_size = 128, attack_rate = .5, local_epochs = 30)
    client.train_local()

    compute_time = client.compute_time
    mse = client.train_loss
    mae = sum(client.MAE)/len(client.MAE)
    rmse = sum(client.RMSE)/len(client.RMSE)

    logger.info(f'Compute time : {compute_time:.8f}')
    logger.info(f'Train loss (MSE) : {mse:.8f}')
    logger.info(f'MAE : {mae:.8f}')
    logger.info(f'RMSE : {rmse:.8f}')

    client.plot()
    
    logger.info('Malicious client check ended successfully')