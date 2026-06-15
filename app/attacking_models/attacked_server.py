# Imports
import torch
import torch.nn as nn
from typing import Dict

from app.models.server import Server
from app.attacking_models.malicious_entity import MaliciousEntity

class AttackedServer(Server, MaliciousEntity):
    def __init__(self, global_model: nn.Module, **kwargs):

        Server.__init__(self, global_model, **kwargs)
        MaliciousEntity.__init__(self, **kwargs)

    def broadcast(self, round_id: int, threaded: bool = True) -> Dict[str, torch.Tensor]:
        """
        Poisons the model before broadcasting it to the clients.

        Args:
            round (int): he id of the current round.

        Returns:
            Dict[str, torch.Tensor]: The poisonned model.
        """

        if self.can_attack():
            self.global_model.load_state_dict(self.poison_model(self.global_model, self.attack_method, self.partial_attack))
            self.attacked_rounds.append(self.current_round)

        return super().broadcast(round)