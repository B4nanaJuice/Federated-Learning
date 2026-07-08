# Imports
from typing import List, Dict
import torch
import copy
from config import create_logger
from app.scoring import ScoringClient
from app.degraded.network_interface import NetworkInterface
import time

logger = create_logger(__name__)

class OfflineClient(ScoringClient):
    """
    Class representing a client (node) in a degraded training mode.

    Attributes:
        vote (bool): The vote of the client used in the majority vote.
        network (NetworkInterface): The network in which the client is.
        offline_training (bool): Information about the current training mode. It can only be changed with a majority vote.
        temp_model (Dict[str, torch.tensor]): A temporary save of the received model which will be taken or discarded depending on the computed score and the majority vote.
    """
    def __init__(self, client_id: int | str, **kwargs):
        super().__init__(client_id, **kwargs)
        self.vote: bool = None
        self.network: NetworkInterface = None
        self.offline_training: bool = False
        self.temp_model: Dict[str, torch.Tensor] = None

    def register_network(self, network: NetworkInterface) -> None:
        """
        Register the network the client is in.

        Args:
            network (NetworkInterface): The network the aggregation server is currently in.
        """
        self.network = network
        return
    
    def receive_global_model(self, global_weights: Dict[int, torch.Tensor], round_id: int) -> None:
        """
        Receive global model weights from the server and update the local model. The received model is temporarily saved unbtil all the clients decided to keep the global model or not for the current training round.
        
        Args:
            global_weights (Dict[int, torch.Tensor]): The global model weights received from the server.
            round_id (int): The id of the began round.
        """

        logger.debug(f'Client {self.client_id} received global model for round {round_id}')
        self.vote = None
        self.current_round = round_id
        self.temp_model = copy.deepcopy(global_weights)

        # Compute score
        self.compute_score('server', global_weights)
        logger.info(f'Score of server: {self.scores.get("server")}')
        self.vote = self.scores.get('server') < self.threshold
        logger.info(f'Client {self.client_id} has their vote ({self.vote}) ready to be sent.')

        # Wait for the right phase (Model evaluation phase)
        while self.network.get_phase() != 0:
            logger.debug(f'Client {self.client_id} is waiting for model evaluation phase (current: {self.network.get_phase()})')
            time.sleep(1)

        # Send vote
        if self.vote != self.offline_training:
            self.network.receive_vote(self.client_id, self.vote)

        return

    def send_vote(self) -> bool:
        """
        Send the computed score as a vote to the network.

        Returns:
            bool: The vote of the client for the current round.
        """

        while self.vote is None: # Wait until the client computed the score
            time.sleep(1)

        self.network.receive_vote(self.client_id, self.vote)
        return self.vote

    def refresh_offline_training(self) -> None:
        """
        Wait the majority vote result and check if the round needs to be run in offline mode.
        """

        # Wait for the right phase (Training mode decision phase)
        while self.network.get_phase() != 1:
            time.sleep(1)

        logger.debug(f'Checking if offline mode needs to be switched.')
        self.offline_training = self.network.compute_majority_vote()
        logger.debug(f'Set client {self.client_id} offline mode to {self.offline_training}.')
        return
    
    def train_local(self) -> None:
        """
        Train the client's local model based on the local dataset.
        """

        # Wait for the phase to be local training
        while self.network.get_phase() != 2:
            logger.debug(f'Client {self.client_id} is waiting for Local training phase (current: {self.network.get_phase()})')
            time.sleep(1)

        # If offline training is set to true, then exchange model with neightbour
        if self.offline_training:
            logger.debug(f'Client {self.client_id} is fetching neighbour\'s model.')
            self.model.load_state_dict(copy.deepcopy(self.network.exchange_model(self.client_id, self.saved_model)))
        # Else, set the model to the temp model
        else:
            logger.debug(f'Normal mode, taking broadcasted model.')
            self.model.load_state_dict(copy.deepcopy(self.temp_model))

        logger.debug(f'Starting training for client {self.client_id}')
        return super().train_local()
    
    def send_saved_model(self) -> Dict[str, torch.Tensor]:
        """
        Send the saved model to the network so the neighbour can use it and train it during offline training mode.

        Returns:
            Dict[str, torch.Tensor]: The client's saved model.
        """
        return self.saved_model
    
    def __repr__(self) -> str:
        return f'<OfflineClient {self.client_id}>'
        