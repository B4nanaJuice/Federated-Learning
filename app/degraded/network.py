# Imports
from config import create_logger
from typing import List, Dict, Callable
from app.models import NormalMLP
from app.degraded.offline_client import OfflineClient
from app.degraded.network_interface import NetworkInterface
import threading
from enum import Enum
import time
import copy
import torch

logger = create_logger(__name__)

class NetworkPhase(Enum):
    MODEL_EVALUATION = 0
    TRAINING_MODE_DECISION = 1
    LOCAL_TRAINING = 2

class NetworkException(Exception):
    def __init__(self, message: str, code: int = 400):
        self.message: str = message
        self.code: int = code

class Network(NetworkInterface):

    def __init__(self):
        self.clients: Dict[str, OfflineClient] = {}
        self.votes: Dict[str, bool] = {}
        self.majority_vote: bool = None
        self.phase: NetworkPhase = NetworkPhase.LOCAL_TRAINING
        self.clients_model: Dict[str, Dict] = {}
    
    # Method for registering clients (bool is for if the client has been added)
    def register_client(self, client: OfflineClient) -> bool: 
        
        if client.client_id in self.clients:
            logger.info(f'Client {client.client_id} is already registered.')
            return False
        
        client.register_network(self)
        self.clients[client.client_id] = client
        logger.debug(f'Successfully registered client {client.client_id}.')
        return True

    # Method for registering multiple clients (int is for the number of clients added)
    def register_clients(self, clients: List[OfflineClient]) -> int: 
        
        resp: List[bool] = [self.register_client(client) for client in clients]
        return len([_ for _ in resp if _])

    # Method for receiving a vote from a client (bool is for the client's vote, can be different is the lcient has already voted for this round)
    def receive_vote(self, client_id: int | str, vote: bool) -> bool:

        logger.debug(f'Received vote {vote} for client {client_id}.') 
        
        if client_id in self.votes:
            logger.info(f'Client {client_id} has already voted for this round.')
            return self.votes[client_id]
        
        self.votes[client_id] = vote
        if len(self.votes.keys()) == 1:
            logger.debug(f'Client {client_id} voted first, need to ask other clients.')
            self.ask_vote()

        return vote

    # Method for asking every client to compute their vote (int is for the number of clients that have been asked)
    def ask_vote(self) -> int: 
        
        clients: List[OfflineClient] = [_ for _ in self.clients.values() if _.client_id not in self.votes]
        logger.debug(f'Asking vote for clients {clients}.')

        if len(clients) > 0:
            threads: List[threading.Thread] = []

            for client in clients:
                threads.append(threading.Thread(target = client.send_vote))
            
            [t.start() for t in threads]
            threads[-1].join(5) # Wait 5 seconds after the last thread has been launched
            # Try to kill the remaining threads
        return len(clients)

    # Method for computing the majority vote
    def compute_majority_vote(self) -> bool: 
        
        true_votes: int = len([_ for _ in self.votes.values() if _])
        return true_votes / len(self.votes.values()) > 0.5

    # Method for reseting votes (new round) (int is the number of cleared votes)
    def reset_votes(self) -> int: 
        
        votes_count: int = len(self.votes.values())
        self.votes = {}
        return votes_count
    
    # Method for getting the actual phase (int is for the phase id (see NetworkPhase enum class))
    def get_phase(self) -> int: 
        
        return self.phase.value
    
    # Method for the server to tell the model has been broadcasted
    def end_model_broadcast(self, callback: Callable) -> None: 

        logger.info('Starting model evaluation phase for all the clients')
        self.votes = {}                             # Reset votes
        self.majority_vote = None                   # Reset majority vote
        self.phase = NetworkPhase.MODEL_EVALUATION  # Set new phase
        self.clients_model = {}                     # Reset exchanged models

        time.sleep(6) # Wait 6 seconds to maybe get a majority vote ask

        if len(self.votes.keys()) == 0:
            self.phase = NetworkPhase.LOCAL_TRAINING
            # Call training phase for clients
        else:
            # Decide the training mode if there are votes (and then refresh training mode for clients)
            self.phase = NetworkPhase.TRAINING_MODE_DECISION
            [_.refresh_offline_training() for _ in self.clients.values()]
            self.phase = NetworkPhase.LOCAL_TRAINING 
        return callback()
    
    def __get_neighbour(self, client_id: int) -> int:
        return client_id + (client_id % 2) * 2 - 1
    
    # Method to exchange models with client's neighbour (dict is mneighbour's model weights)
    def exchange_model(self, source_client_id: int | str, source_client_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]: 
        
        self.clients_model[source_client_id] = copy.deepcopy(source_client_weights)
        neighbour_id: int = self.__get_neighbour(source_client_id)
        logger.debug(f'Client {source_client_id} neighbour : {neighbour_id}')

        # If neighbour hasn't sent their model yet, get it, and save it
        if neighbour_id not in self.clients_model:
            logger.debug(f'Client {neighbour_id}\'s model not saved in network, fetching it.')
            self.clients_model[neighbour_id] = copy.deepcopy(self.clients[neighbour_id].send_saved_model())

        return self.clients_model[neighbour_id]
    
def check_network():
    
    network: Network = Network()

    clients: List[OfflineClient] = []
    for _ in range(1, 8):
        clients.append(OfflineClient(_, model = NormalMLP()))

    network.register_clients(clients)

    clients[3].vote = False
    clients[3].send_vote()

    print(network.votes)
    print('Majority vote:', network.compute_majority_vote())
    for client in network.clients.values():
        client.refresh_offline_training()
        logger.info(f'Client {client.client_id}\'s offline training set to {client.offline_training}')