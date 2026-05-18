# Imports
from typing import List

from app.models import Client
from app.degraded.network_interface import NetworkInterface
from app.scoring import ScoringServer

class MultilineServer(ScoringServer):
    
    def register_network(self, network: NetworkInterface) -> None:
        self.network: NetworkInterface = network
        return
    
    def select_clients(self, fraction: float = 1.0) -> List[Client]:
        self.network.clear_votes()
        return super().select_clients(fraction)