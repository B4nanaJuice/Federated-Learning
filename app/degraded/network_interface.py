# Imports
import numpy as np

# Create interface
class NetworkInterface:
    def register_clients(self, clients) -> None:
        pass
    def generate_adjacent_matrix(self, k: int = 1) -> np.ndarray:
        pass
    def clear_votes(self) -> int:
        pass
    def add_vote(self, client_id: int | str, vote: bool) -> bool:
        pass
    def get_majority_vote(self) -> bool:
        pass