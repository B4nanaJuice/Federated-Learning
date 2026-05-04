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

        for update in self.received_updates:
            self.compute_score(update.get('client_id'), update.get('weights'))

        logger.info(f'Scores: {self.scores}')
        return
    
    def aggregate(self) -> None:
        if len(self.received_updates) < self.min_clients:
            raise Exception('Number of minimum models not reached')
        
        new_state = self._filtered_fedavg(self.received_updates, self.scores, self.threshold)
        if not new_state:
            self.global_model.load_state_dict(self.saved_model)
        else:
            self.global_model.load_state_dict(new_state)

        self.current_round += 1
        return
    
    @staticmethod
    def _filtered_fedavg(updates: List[Dict], weights: Dict[str, float], threshold: float) -> Dict[str, torch.Tensor]:
        aggregated: Dict[str, torch.Tensor] = {}
        taken_scores: List[float] = [_ for _ in weights.values() if _ > threshold]

        if len(taken_scores) == 0:
            return None

        sum_score: float = sum(taken_scores)

        for update in updates:
            client_id = update.get('client_id')
            score = weights.get(client_id)

            if score < threshold:
                logger.info(f'Client {client_id} has a score too low ({score} < {threshold})')
                pass

            for k, delta in update.get('weights').items():
                if k not in aggregated:
                    aggregated[k] = torch.zeros_like(delta)
                aggregated[k] += score * delta / sum_score

        return aggregated
    
def check_scoring_server():
    logger.info('Starting scoring server check...')

    sigmas: List[float] = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
    results: Dict[str, List[float]] = {}
    run_count: int = 10

    for sigma in sigmas:
        logger.info(f'Test with sigma = {sigma}')

        for _ in range(run_count):

            server: ScoringServer = ScoringServer(
                global_model = NormalMLP(),
                metric = ScoringMetric.DISTANCE,
                max_rounds = 15,
                metric_parameters = {'sigma': sigma}
            )

            for _ in range(5):
                server.register_client(Client(_+1, local_epochs = 10))

            server.run(client_fraction = 1)
            average_loss = [sum(_)/len(_) for _ in server.training_loss]
            if not f'{sigma}' in results:
                results[f'{sigma}'] = np.array(average_loss) / run_count
            else:
                results[f'{sigma}'] += np.array(average_loss) / run_count

    torch.save(results, f'{config.SAVE_DATA_PATH}/distance_scoring.pt')

    logger.info('Scoring server check ended successfully')