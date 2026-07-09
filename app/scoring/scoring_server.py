# Imports
import copy
import torch
import torch.nn as nn
from typing import Dict, List

from config import create_logger, config
from app.services.aggregation_service import AggregationService
from app.models import NormalMLP, Server, Client
from app.scoring.scoring_entity import ScoringEntity, ScoringMetric

logger = create_logger(__name__)

class ScoringServer(Server, ScoringEntity):
    def __init__(self, global_model: nn.Module, **kwargs):

        Server.__init__(self, global_model, **kwargs)
        ScoringEntity.__init__(self, **kwargs)

    def broadcast(self, round_id: int, threaded: bool = config.SIM_THREADED) -> Dict[str, torch.Tensor]:
        """
        Broadcast the model to selected clients. Make a save of the gloabl model before broadcasting it to the clients.

        Args:
            round_id (int): The current round.
            threaded (bool = config.SIM_THREADED): Broadcast the model in parallel or on a signle thread.

        Returns:
            Dict[str, torch.Tensor]: The broadcasted model's weights.
        """

        broadcasted_model: Dict[str, torch.Tensor] = super().broadcast(round_id, threaded)
        self.saved_model = copy.deepcopy(broadcasted_model)
        return self.broadcast_model

    def collect_updates(self, threaded: bool = config.SIM_THREADED) -> None:
        """
        Tell clients to train their local model and collect update of each client participating in the current round. Compute a score for each received model in order to keep it or not for the aggregation phase.

        Args:
            threaded (bool = config.SIM_THREADED): Train the clients on different threads or on a single one.
        """

        super().collect_updates(threaded = threaded)

        kept_updates: List[Dict] = []
        rejected: int = 0

        for update in self.received_updates:
            client_id = update.get('client_id')

            self.compute_score(client_id, update.get('weights'))
            logger.info(f'Score of client {client_id}: {self.scores.get(client_id)}')

            if self.scores.get(client_id) < self.threshold:
                logger.info(f'Client {client_id} has a score too low ({self.scores.get(client_id)} < {self.threshold})')
                rejected += 1
            else:
                kept_updates.append(update)
        
        self.rejected_models.append(rejected)
        self.received_updates = copy.deepcopy(kept_updates)
        logger.info(f'Scores: {self.scores}')
        return
    
    def aggregate(self) -> None:
        """
        Aggregate all the received updates into one model. The default aggregation method is FedAvg. If no model is kept (all received models have a trust score too low), then the previous global model is taken.
        """

        self.current_round += 1

        logger.info(f'Received updates count: {len(self.received_updates)}')
        logger.info(f'Kept updates: {[_.get("client_id") for _ in self.received_updates]}')
        
        if len(self.received_updates) == 0:
            logger.info('No update is trusted. Taking the saved model instead.')
            self.global_model.load_state_dict(self.saved_model)
            return
        
        new_state = AggregationService.weighted_fed_avg(self.received_updates, self.scores)
        self.global_model.load_state_dict(new_state)
        return
    
def check_scoring_server():
    logger.info('Starting scoring server check...')

    server: ScoringServer = ScoringServer(
        global_model = NormalMLP(),
        max_rounds = 5,
        metric = ScoringMetric.DISTANCE,
        metric_parameters = {'sigma': 2.0}
    )

    server.register_client(Client(1, NormalMLP(), local_epochs = 2))
    server.register_client(Client(2, NormalMLP(), local_epochs = 2))

    server.run()

    assert server.rejected_models >= 0
    logger.info(f'Server rejected updates: {server.rejected_models}')

    logger.info('Scoring server check ended successfully')