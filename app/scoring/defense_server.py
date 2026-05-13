# Import
import torch.nn as nn
from typing import Dict

from app.models import Server
from app.attacking_models import AttackedServer
from app.services import AggregationService
    
## CLEAN SERVERS
# Weighted_FedAvg
class WeightedFedAvgServer(Server):
    def aggregate(self) -> None:

        weights: Dict[int|str, float] = {_.get('client_id'): 1/len(self.received_updates) for _ in self.received_updates}
        new_state = AggregationService.weighted_fed_avg(self.received_updates, weights)
        self.global_model.load_state_dict(new_state)

        self.current_round += 1
        return
    
# Krum
class KrumServer(Server):
    def aggregate(self) -> None:
        
        new_state = AggregationService.m_krum(self.received_updates, 1, m = 1)
        self.global_model.load_state_dict(new_state)

        self.current_round += 1
        return

# Multi krum
class MKrumServer(Server):
    def aggregate(self) -> None:
        
        new_state = AggregationService.m_krum(self.received_updates, 1, m = 2)
        self.global_model.load_state_dict(new_state)

        self.current_round += 1
        return
    
# Norm based server
class NormAggServer(Server):
    def aggregate(self) -> None:

        new_state = AggregationService.norm_based_aggregation(self.received_updates, 1)
        self.global_model.load_state_dict(new_state)

        self.current_round += 1
        return
    
# CBAA-FedAvg server
class CBAAFedAvgServer(Server):
    def aggregate(self) -> None:

        new_state = AggregationService.cbaa_fed_avg(self.received_updates)
        self.global_model.load_state_dict(new_state)

        self.current_round += 1
        return
    
## ATTACKED SERVERS
# Weighted_FedAvg
class AttackedWeightedFedAvgServer(AttackedServer):
    def aggregate(self) -> None:

        weights: Dict[int|str, float] = {_.get('client_id'): 1/len(self.received_updates) for _ in self.received_updates}
        new_state = AggregationService.weighted_fed_avg(self.received_updates, weights)
        self.global_model.load_state_dict(new_state)

        self.current_round += 1
        return
    
# Krum
class AttackedKrumServer(AttackedServer):
    def aggregate(self) -> None:
        
        new_state = AggregationService.m_krum(self.received_updates, 1, m = 1)
        self.global_model.load_state_dict(new_state)

        self.current_round += 1
        return

# Multi krum
class AttackedMKrumServer(AttackedServer):
    def aggregate(self) -> None:
        
        new_state = AggregationService.m_krum(self.received_updates, 1, m = 2)
        self.global_model.load_state_dict(new_state)

        self.current_round += 1
        return
    
# Norm based server
class AttackedNormAggServer(AttackedServer):
    def aggregate(self) -> None:

        new_state = AggregationService.norm_based_aggregation(self.received_updates, 1)
        self.global_model.load_state_dict(new_state)

        self.current_round += 1
        return
    
# CBAA-FedAvg server
class AttackedCBAAFedAvgServer(AttackedServer):
    def aggregate(self) -> None:

        new_state = AggregationService.cbaa_fed_avg(self.received_updates)
        self.global_model.load_state_dict(new_state)

        self.current_round += 1
        return