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

    def can_attack(self) -> bool:
        return self.current_round == 10

    def broadcast(self, round: int) -> Dict[str, torch.Tensor]:

        if self.can_attack():
            self.global_model.load_state_dict(self.poison_model(self.global_model, self.attack_method, self.partial_attack))
            self.attacked_rounds.append(self.current_round)

        # return super(Server, self).broadcast(round = round)
        return super().broadcast(round)