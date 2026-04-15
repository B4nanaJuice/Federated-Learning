# Imports
from app.models import Client, Server, NormalMLP
from config import config, create_logger

logger = create_logger(__name__)

# Simulation method
def simulate():
    # Create server
    aggregation_server: Server = Server(
        global_model = NormalMLP(),
        max_rounds = config.SIM_MAX_ROUNDS,
        min_clients = config.SIM_MIN_CLIENTS
    )

    # Add 20 clients on the server
    for _ in range(1, 21):
        _client: Client = Client(
            client_id = _,
            model = NormalMLP(),
            batch_size = config.SIM_BATCH_SIZE,
            learning_rate = config.SIM_LEARNING_RATE,
            local_epochs = config.SIM_LOCAL_EPOCHS
        )

        aggregation_server.register_client(_client)
    
    # Run the server
    aggregation_server.run(client_fraction = config.SIM_CLIENTS_FRACTION)

