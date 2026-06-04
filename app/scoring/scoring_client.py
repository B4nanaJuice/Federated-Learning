# Imports
import copy
import torch
import torch.nn as nn
from typing import Dict, List

from config import create_logger
from app.models import NormalMLP, Client, Server
from app.scoring.scoring_entity import ScoringEntity, ScoringMetric

logger = create_logger(__name__)

class ScoringClient(Client, ScoringEntity):
    def __init__(self, client_id: str, model: nn.Module = NormalMLP(), **kwargs):

        Client.__init__(self, client_id = client_id, model = model, **kwargs)
        ScoringEntity.__init__(self, **kwargs)

    def receive_global_model(self, global_weights: Dict[int, torch.Tensor], round_id: int) -> None:

        self.compute_score('server', global_weights)
        logger.info(f'Score of server: {self.scores.get("server")}')
        if self.scores.get('server') < self.threshold:
            # Load saved model
            self.model.load_state_dict(self.saved_model)
            self.rejected_models += 1
            logger.info(f'Received model from server has a score too low ({self.scores.get("server")} < {self.threshold})')
        else:
            self.model.load_state_dict(copy.deepcopy(global_weights))

        self.current_round = round_id
        return
    
    def send_update(self) -> Dict:
        # Save model
        self.saved_model = copy.deepcopy(self.model.state_dict())
        return super().send_update()
    
def check_scoring_client():
    logger.info('Starting scoring client check...')

    server: Server = Server(
        global_model = NormalMLP(),
        max_rounds = 5
    )

    server.register_client(ScoringClient(1, NormalMLP(), local_epochs = 2, metric = ScoringMetric.DISTANCE, metric_parameters = {'sigma': 8}))
    server.register_client(ScoringClient(2, NormalMLP(), local_epochs = 2, metric = ScoringMetric.DISTRIBUTION))
    server.register_client(ScoringClient(3, NormalMLP(), local_epochs = 2, metric = ScoringMetric.SIMILARITY))

    server.run()

    for id, client in server.client_registry.items():
        assert client.rejected_models >= 0
        logger.info(f'Client {id} rejected updates: {client.rejected_models}')

    logger.info('Scoring client check ended successfully')