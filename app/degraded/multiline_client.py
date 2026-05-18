# Imports
import copy
import time
import torch
from typing import Dict

from config import create_logger
from app.scoring import ScoringClient
from app.degraded.network_interface import NetworkInterface

logger = create_logger(__name__)

class MultilineClient(ScoringClient):

    def __init__(self, client_id: int, **kwargs):
        super().__init__(client_id, **kwargs)
        self.online_mode: bool = True
    
    # Register network
    def register_network(self, network) -> None:
        logger.debug(f'Registering network (type {type(network)}) for client {self.client_id}')
        self.network: NetworkInterface = network

    # Send vote to network
    def send_vote(self, vote: bool) -> bool:
        return self.network.add_vote(self.client_id, vote)
    
    # Get majority vote
    def get_majority_vote(self) -> bool:
        return self.network.get_majority_vote()
    
    def receive_global_model(self, global_weights: Dict[int, torch.Tensor], round_id: int) -> None:

        self.compute_score('server', global_weights)
        logger.info(f'Score of server: {self.scores.get("server")}')

        # Compute vote and send it to the network
        _vote: bool = self.scores.get('server') > self.threshold
        self.send_vote(_vote)

        logger.debug(f'Client {self.client_id} has a vote {_vote} for the server\'s model')

        # Wait for the majority vote
        self.online_mode = -1
        while self.online_mode == -1:
            try:
                self.online_mode = self.get_majority_vote()
            except:
                time.sleep(1)
                pass

        logger.debug(f'The majority vote is : {self.online_mode}')

        # Do according to the online/offline mode
        if self.online_mode:
            # Continue online mode
            self.model.load_state_dict(copy.deepcopy(global_weights))
        else:
            # Switch to offline mode
            self.model.load_state_dict(self.saved_model)

        self.round_id = round_id
        return
    
def check_multiline_client():
    pass