# Imports
from typing import List
import random as rd

from app.models import Client, Server, NormalMLP, MaliciousClient, AttackedServer
from config import config, create_logger

logger = create_logger(__name__)

# Simulation method
def simulate():
    # Create server
    aggregation_server: Server = Server(
        global_model = NormalMLP(),
        max_rounds = config.SIM_MAX_ROUNDS
    )

    # Add 20 clients on the server
    clients: List[Client] = [
        Client(1, NormalMLP(), local_epochs = rd.randint(10, 20), batch_size = 2**rd.randint(5, 7)),
        Client(2, NormalMLP(), local_epochs = rd.randint(10, 20), batch_size = 2**rd.randint(5, 7)),
        Client(3, NormalMLP(), local_epochs = rd.randint(10, 20), batch_size = 2**rd.randint(5, 7)),
        Client(4, NormalMLP(), local_epochs = rd.randint(10, 20), batch_size = 2**rd.randint(5, 7)),
        Client(5, NormalMLP(), local_epochs = rd.randint(10, 20), batch_size = 2**rd.randint(5, 7)),
        Client(6, NormalMLP(), local_epochs = rd.randint(10, 20), batch_size = 2**rd.randint(5, 7)),
        Client(7, NormalMLP(), local_epochs = rd.randint(10, 20), batch_size = 2**rd.randint(5, 7)),
        Client(8, NormalMLP(), local_epochs = rd.randint(10, 20), batch_size = 2**rd.randint(5, 7)),
        Client(9, NormalMLP(), local_epochs = rd.randint(10, 20), batch_size = 2**rd.randint(5, 7)),
        Client(10, NormalMLP(), local_epochs = rd.randint(10, 20), batch_size = 2**rd.randint(5, 7)),
        Client(11, NormalMLP(), local_epochs = rd.randint(10, 20), batch_size = 2**rd.randint(5, 7)),
        Client(12, NormalMLP(), local_epochs = rd.randint(10, 20), batch_size = 2**rd.randint(5, 7)),
        Client(13, NormalMLP(), local_epochs = rd.randint(10, 20), batch_size = 2**rd.randint(5, 7)),
        Client(14, NormalMLP(), local_epochs = rd.randint(10, 20), batch_size = 2**rd.randint(5, 7)),
        Client(15, NormalMLP(), local_epochs = rd.randint(10, 20), batch_size = 2**rd.randint(5, 7)),
        MaliciousClient(16, attack_rate = .1, attack_method = 'random', local_epochs = rd.randint(10, 20), batch_size = 2**rd.randint(5, 7)),
        MaliciousClient(17, attack_rate = .1, attack_method = 'random', local_epochs = rd.randint(10, 20), batch_size = 2**rd.randint(5, 7)),
        MaliciousClient(18, attack_rate = .1, attack_method = 'random', local_epochs = rd.randint(10, 20), batch_size = 2**rd.randint(5, 7)),
        MaliciousClient(19, attack_rate = .1, attack_method = 'random', local_epochs = rd.randint(10, 20), batch_size = 2**rd.randint(5, 7)),
        MaliciousClient(20, attack_rate = .1, attack_method = 'random', local_epochs = rd.randint(10, 20), batch_size = 2**rd.randint(5, 7))
    ]
    
    aggregation_server.register_clients(clients)
    
    # Run the server
    aggregation_server.run(client_fraction = config.SIM_CLIENTS_FRACTION)

    # Plot data
    aggregation_server.run_validation()
    aggregation_server.plot()
    return

def fast_simulate():
    # Create server
    aggregation_server: Server = Server(
        global_model = NormalMLP(),
        max_rounds = 20,
        # attack_rate = .05
    )

    # Add clients
    clients: List[Client] = [
        Client(client_id = 1, local_epochs = 10, batch_size = 256, model = NormalMLP()),
        Client(client_id = 2, local_epochs = 10, batch_size = 256, model = NormalMLP()),
        Client(client_id = 3, local_epochs = 10, batch_size = 256, model = NormalMLP()),
        Client(client_id = 4, local_epochs = 10, batch_size = 256, model = NormalMLP()),
        Client(client_id = 5, local_epochs = 10, batch_size = 256, model = NormalMLP()),
        Client(client_id = 6, local_epochs = 10, batch_size = 256, model = NormalMLP()),
    ]

    aggregation_server.register_clients(clients)

    # Run the server
    aggregation_server.run(2/3)

    # Plot data
    aggregation_server.run_validation()
    aggregation_server.plot()
    return
