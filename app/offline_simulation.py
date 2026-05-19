# Imports
from typing import List

from config import create_logger
from app.models import NormalMLP
from app.scoring import ScoringServer
from app.degraded import Network, MultilineClient, MultilineServer

logger = create_logger(__name__)

def simulate_offline_training():
    
    # Create network
    net: Network = Network()

    # Create clients
    clients: List[MultilineClient] = [
        MultilineClient(1, local_epochs = 3, batch_size = 128, metric_parameters = {'sigma': 8}),
        MultilineClient(2, local_epochs = 3, batch_size = 128, metric_parameters = {'sigma': 8}),
        MultilineClient(3, local_epochs = 3, batch_size = 128, metric_parameters = {'sigma': 1}),
        MultilineClient(4, local_epochs = 3, batch_size = 128, metric_parameters = {'sigma': 8})
    ]
    net.register_clients(clients)

    # Create server
    server: MultilineServer = MultilineServer(NormalMLP(), max_rounds = 3, metric_parameters = {'sigma': 8})
    server.register_clients(clients)
    server.register_network(net)

    server.run()
