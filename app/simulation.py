# Imports
from typing import List
import random as rd

from app.models import Client, Server, NormalMLP, MaliciousClient, AttackedServer
from config import config, create_logger

logger = create_logger(__name__)

def simulate_clean():
    # Create server
    server: Server = Server(
        global_model = NormalMLP(),
        max_rounds = 50
    )

    # Add clients
    clients: List[Client] = [
        Client(client_id = _, model = NormalMLP(), local_epochs = 10, batch_size = 256)
        for _ in range(1, 21)
    ]
    server.register_clients(clients = clients)

    # Run
    server.run(.5)

    server.run_validation()
    server.plot()
    Server.save_state(server, 'clean_run')
    return

def simulate_malicious_clients():
    pass

def simulate_attacked_server():
    pass