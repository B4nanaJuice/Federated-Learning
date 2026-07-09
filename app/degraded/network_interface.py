# Imports
from typing import List, Dict, Callable
from app.models import Client
import torch

class NetworkInterface:
    """
    Interface of the Network class.
    """

    # Method for registering clients (bool is for if the client has been added)
    def register_client(self, client: Client) -> bool: pass
    # Method for registering multiple clients (int is for the number of clients added)
    def register_clients(self, clients: List[Client]) -> int: pass
    # Method for receiving a vote from a client (bool is for if the vote has been added or not)
    def receive_vote(self, client_id: int | str, vote: bool) -> bool: pass
    # Method for asking every client to compute their vote (int is for the number of clients that have been asked)
    def ask_vote(self) -> int: pass
    # Method for computing the majority vote (bool is the majority vote)
    def compute_majority_vote(self) -> bool: pass
    # Method for reseting votes (new round) (int is the number of cleared votes)
    def reset_votes(self) -> int: pass
    # Method for getting the actual phase (int is for the phase id (see NetworkPhase enum class))
    def get_phase(self) -> int: pass
    # Method for the server to tell the model has been broadcasted
    def end_model_broadcast(self, callback: Callable) -> None: pass
    # Method to exchange models with client's neighbour (dict is mneighbour's model weights)
    def exchange_model(self, source_client_id: int | str, source_client_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]: pass