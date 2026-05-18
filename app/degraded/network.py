# Imports
import copy
import numpy as np
import random as rd
from typing import List, Dict

from app.degraded.multiline_client import MultilineClient
from config import create_logger
from app.degraded.network_interface import NetworkInterface

logger = create_logger(__name__)

class NetworkException(Exception):
    def __init__(self, message: str, code: int = 400):
        self.message: str = message
        self.code: int = code

# Network that will handle clients' communication
class Network(NetworkInterface):
    def __init__(self):

        self.client_count: int = 0
        self.clients: List[MultilineClient] = []
        self.adjacent_matrix: np.ndarray = []
        self.votes: Dict[str, bool] = {}

    # Method for registering clients
    def register_clients(self, clients: List[MultilineClient]) -> None:

        self.clients = clients
        for client in clients:
            client.register_network(self)

        self.client_count = len(self.clients)
        return

    # Method for choosing each client's neighbour
    def generate_adjacent_matrix(self, k: int = 1) -> np.ndarray:
        
        n: int = self.client_count
        if k >= n or k*n % 2 != 0:
            raise NetworkException('Unable to generate a graph with these parameters.')
        
        self.adjacent_matrix = np.zeros((n, n), dtype = int)

        for i in range(n):
            for j in range(1, k // 2 + 1):
                neighbour = (i + j) % n
                self.adjacent_matrix[i, neighbour] = 1
                self.adjacent_matrix[neighbour, i] = 1
            
            if k % 2 == 1:
                neighbour = (i + n // 2) % n
                if self.adjacent_matrix[i, neighbour] == 0:
                    self.adjacent_matrix[i, neighbour] = 1
                    self.adjacent_matrix[neighbour, i] = 1

        return self.adjacent_matrix
    
    # Method for clearing votes
    def clear_votes(self) -> int:
        
        cleared_votes: int = len(self.votes.keys())
        self.votes = copy.deepcopy({})
        return cleared_votes
    
    # Method for adding a vote from a client
    def add_vote(self, client_id: int | str, vote: bool) -> bool:
        # return bool: if the vote was added: return true
        # for any reason the vote wasn't added: return false
        if client_id in self.votes:
            logger.info(f'Client {client_id} has already voted for this round. Ingnoring the new vote.')
            return False
        
        self.votes[client_id] = vote
        return True
    
    # Method for getting the majority vote
    def get_majority_vote(self) -> bool:

        if len(self.votes.keys()) == 0:
            raise NetworkException('Unable to get the majority vote as no one voted yet.')
        if len(self.votes.keys()) != self.client_count:
            raise NetworkException('Not all clients have voted yet.')
        
        return bool(round(sum(self.votes.values())/self.client_count))
    
    @property
    def majority_vote(self):
        return self.get_majority_vote()
    
def check_network():
    logger.info('Starting network check')
    
    net: Network = Network()
    clients: List[MultilineClient] = [
        MultilineClient(1),
        MultilineClient(2),
        MultilineClient(3),
        MultilineClient(4)
    ]
    net.register_clients(clients)

    assert net.client_count == len(clients)

    for client in clients:
        client.send_vote(rd.random() > .5)
        assert client.client_id in net.votes

    logger.debug(f'Votes : {net.votes}')

    assert net.majority_vote in [True, False]
    logger.debug(f'Majority vote : {net.majority_vote}')

    net.clear_votes()
    assert len(net.votes.keys()) == 0

    logger.debug(f'Cleared votes : {net.votes}')

    net.generate_adjacent_matrix()
    assert type(net.adjacent_matrix) == np.ndarray
    assert len(net.adjacent_matrix) == net.client_count

    logger.debug(f'Adjacent matrix :')
    for line in net.adjacent_matrix:
        logger.debug(line)

    logger.info('Network check ended successfully')