# Imports
import torch.nn as nn
import torch
from typing import Dict, List
import copy
import matplotlib.pyplot as plt
import numpy as np

from app.scoring.scoring_entity import ScoringEntity, ScoringMetric
from app.models import NormalMLP, Server, Client
from config import create_logger, config

logger = create_logger(__name__)

class ScoringServer(Server, ScoringEntity):
    def __init__(self, global_model: nn.Module, **kwargs):

        Server.__init__(self, global_model, **kwargs)
        ScoringEntity.__init__(self, **kwargs)

    def broadcast(self, round: int) -> Dict[str, torch.Tensor]:

        broadcasted_model: Dict[str, torch.Tensor] = super().broadcast(round)
        self.saved_model = copy.deepcopy(broadcasted_model)
        return self.broadcast_model

    def collect_updates(self, threaded: bool = config.SIM_THREADED) -> None:

        super().collect_updates(threaded = threaded)

        kept_updates: List[Dict] = []

        for update in self.received_updates:
            client_id = update.get('client_id')

            self.compute_score(client_id, update.get('weights'))
            logger.info(f'Score of client {client_id}: {self.scores.get(client_id)}')

            if self.scores.get(client_id) < self.threshold:
                logger.info(f'Client {client_id} has a score too low ({self.scores.get(client_id)} < {self.threshold})')
                self.rejected_models += 1
            else:
                kept_updates.append(update)

        self.received_updates = copy.deepcopy(kept_updates)
        logger.info(f'Scores: {self.scores}')
        return
    
    def aggregate(self) -> None:
        if len(self.received_updates) < self.min_clients:
            raise Exception('Number of minimum models not reached')
        
        self.current_round += 1

        logger.info(f'Received updates count: {len(self.received_updates)}')
        logger.info(f'Kept updates: {[_.get("client_id") for _ in self.received_updates]}')
        
        if len(self.received_updates) == 0:
            logger.info('No update is trusted. Taking the saved model instead.')
            self.global_model.load_state_dict(self.saved_model)
            return
        
        new_state = self.aggregation_function(self.received_updates, self.scores)
        self.global_model.load_state_dict(new_state)
        return
    
def check_scoring_server():
    logger.info('Starting scoring server check...')

    server: ScoringServer = ScoringServer(
        global_model = NormalMLP(),
        max_rounds = 5,
        metric = ScoringMetric.DISTANCE,
        min_clients = 0,
        metric_parameters = {'sigma': 2.0}
    )

    server.register_client(Client(1, NormalMLP(), local_epochs = 2))
    server.register_client(Client(2, NormalMLP(), local_epochs = 2))

    server.run()

    assert server.rejected_models >= 0
    logger.info(f'Server rejected updates: {server.rejected_models}')

    logger.info('Scoring server check ended successfully')