# Imports
from typing import List

from app.models import Client, Server, NormalMLP
from app.attacking_models import MaliciousClient, AttackedServer
from config import create_logger

logger = create_logger(__name__)

def simulate_clean():
    logger.info('Starting clean simulation')

    # Create server
    server: Server = Server(
        global_model = NormalMLP(),
        max_rounds = 50
    )

    # Add clients
    clients: List[Client] = [
        Client(client_id = _, model = NormalMLP(), local_epochs = 10, batch_size = 256)
        for _ in range(1, 11)
    ]
    server.register_clients(clients = clients)

    # Run
    server.run(.5)

    server.run_test()
    server.plot()

    logger.info('End of clean simulation')
    return

def simulate_malicious_clients():
    logger.info('Starting malicious clients simulation')

    # Create server
    server: Server = Server(
        global_model = NormalMLP(),
        max_rounds = 50
    )

    # Add clients
    clients: List[Client] = [
        Client(client_id = _, model = NormalMLP(), local_epochs = 20, batch_size = 256)
        for _ in range(1, 7)
    ]
    malicious_clients: List[Client] = [
        MaliciousClient(client_id = _, model = NormalMLP(), local_epochs = 20, batch_size = 256, attack_rate = lambda x: x in [10, 25])
        for _ in range(8, 11)
    ]
    server.register_clients(clients = clients)
    server.register_clients(clients = malicious_clients)

    # Run
    server.run(.5)

    server.run_test()
    server.plot()

    logger.info('End of malicious clients simulation')
    return

def simulate_attacked_server():
    logger.info('Starting attacked server simulation')

    # Create server
    server: AttackedServer = AttackedServer(
        global_model = NormalMLP(),
        max_rounds = 50,
        attack_rate = lambda x: x == 10
    )

    # Add clients
    clients: List[Client] = [
        Client(client_id = _, model = NormalMLP(), local_epochs = 10, batch_size = 256)
        for _ in range(1, 11)
    ]
    server.register_clients(clients = clients)

    # Run
    server.run(.5)

    server.run_test()
    server.plot()

    logger.info('End of attacked server simulation')
    return

def simulate_attacked_and_malicious():
    logger.info('Starting attacked server and malicious clients simulation')

    # Create server
    server: AttackedServer = AttackedServer(
        global_model = NormalMLP(),
        max_rounds = 50,
        attack_rate = lambda x: x == 10
    )

    # Add clients
    clients: List[Client] = [
        Client(client_id = _, model = NormalMLP(), local_epochs = 20, batch_size = 256)
        for _ in range(1, 7)
    ]
    malicious_clients: List[Client] = [
        MaliciousClient(client_id = _, model = NormalMLP(), local_epochs = 20, batch_size = 256, attack_rate = lambda x: x == 15)
        for _ in range(8, 11)
    ]
    server.register_clients(clients = clients)
    server.register_clients(clients = malicious_clients)

    # Run
    server.run(.5)

    server.run_test()
    server.plot()

    logger.info('End of attacked server and malicious clients simulation')