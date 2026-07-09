# Import
import torch.nn as nn
from typing import Dict

from app.models import Server
from app.attacking_models import AttackedServer
from app.services.aggregation_service import AggregationService
    
## CLEAN SERVERS
# Weighted_FedAvg
class WeightedFedAvgServer(Server):
    """
    Central server using Weighted Federated Averaging aggregation method.
    """
    def aggregate(self) -> None:
        """
        Aggregate all the received updates using the Weighted Federated Averaging method.
        """

        weights: Dict[int|str, float] = {_.get('client_id'): 1/len(self.received_updates) for _ in self.received_updates}
        new_state = AggregationService.weighted_fed_avg(self.received_updates, weights)
        self.global_model.load_state_dict(new_state)

        self.current_round += 1
        return
    
# Krum
class KrumServer(Server):
    """
    Central server using Krum aggregation method.
    """
    def aggregate(self) -> None:
        """
        Aggregate all the received updates using the Krum method.
        """
        
        new_state = AggregationService.m_krum(self.received_updates, 1, m = 1)
        self.global_model.load_state_dict(new_state)

        self.current_round += 1
        return

# Multi krum
class MKrumServer(Server):
    """
    Central server using Multi-Krum aggregation method.
    """
    def aggregate(self) -> None:
        """
        Aggregate all the received updates using the Multi-Krum method.
        """
        
        new_state = AggregationService.m_krum(self.received_updates, 1, m = 2)
        self.global_model.load_state_dict(new_state)

        self.current_round += 1
        return
    
# Norm based server
class NormAggServer(Server):
    """
    Central server using norm aggregation method.
    """
    def aggregate(self) -> None:
        """
        Aggregate all the received updates using the norm aggregation method.
        """

        new_state = AggregationService.norm_based_aggregation(self.received_updates, 1)
        self.global_model.load_state_dict(new_state)

        self.current_round += 1
        return
    
# CBAA-FedAvg server
class CBAAFedAvgServer(Server):
    """
    Central server using Centroid-Based Anomaly-Aware Federated Averaging aggregation method.
    """
    def aggregate(self) -> None:
        """
        Aggregate all the received updates using the Centroid-Based Anomaly-Aware Federated Averaging method.
        """

        new_state = AggregationService.cbaa_fed_avg(self.received_updates)
        self.global_model.load_state_dict(new_state)

        self.current_round += 1
        return

# Trimmed Mean
class TMeanServer(Server):
    """
    Central server using Trimmed Mean aggregation method.
    """
    def aggregate(self) -> None:
        """
        Aggregate all the received updates using the Trimmed-Mean method.
        """

        new_state = AggregationService.t_mean(self.received_updates)
        self.global_model.load_state_dict(new_state)

        self.current_round += 1
        return

# RFA
class RFAServer(Server):
    """
    Central server using Robust Federated Aggregation method.
    """
    def aggregate(self) -> None:
        """
        Aggregate all the received updates using the Robust Federated Aggregation method.
        """

        new_state = AggregationService.rfa(self.received_updates)
        self.global_model.load_state_dict(new_state)

        self.current_round += 1
        return

# FLTrust
class FLTrustServer(Server):
    """
    Central server using FLTrust aggregation method.
    """
    def aggregate(self) -> None:
        """
        Aggregate all the received updates using the FLTrust method.
        """

        new_state = AggregationService.fl_trust(self.received_updates, self.broadcast_model)
        self.global_model.load_state_dict(new_state)

        self.current_round += 1
        return
    
# CLRA
class CLRAServer(Server):
    """
    Central server using CLRA aggregation method.
    """
    def aggregate(self) -> None:
        """
        Aggregate all the received updates using the CLRA method.
        """

        new_state = AggregationService.clra(self.received_updates, self.broadcast_model)
        self.global_model.load_state_dict(new_state)

        self.current_round += 1
        return
    
## ATTACKED SERVERS
# Weighted_FedAvg
class AttackedWeightedFedAvgServer(AttackedServer):
    """
    Attacked central server using Weighted Federated Averaging aggregation method.
    """
    def aggregate(self) -> None:
        """
        Aggregate all the received updates using the Weighted Federated Averaging method.
        """

        weights: Dict[int|str, float] = {_.get('client_id'): 1/len(self.received_updates) for _ in self.received_updates}
        new_state = AggregationService.weighted_fed_avg(self.received_updates, weights)
        self.global_model.load_state_dict(new_state)

        self.current_round += 1
        return
    
# Krum
class AttackedKrumServer(AttackedServer):
    """
    Attacked central server using Krum aggregation method.
    """
    def aggregate(self) -> None:
        """
        Aggregate all the received updates using the Krum method.
        """
        
        new_state = AggregationService.m_krum(self.received_updates, 1, m = 1)
        self.global_model.load_state_dict(new_state)

        self.current_round += 1
        return

# Multi krum
class AttackedMKrumServer(AttackedServer):
    """
    Attacked central server using Multi-Krum aggregation method.
    """
    def aggregate(self) -> None:
        """
        Aggregate all the received updates using the Multi-Krum method.
        """
        
        new_state = AggregationService.m_krum(self.received_updates, 1, m = 2)
        self.global_model.load_state_dict(new_state)

        self.current_round += 1
        return
    
# Norm based server
class AttackedNormAggServer(AttackedServer):
    """
    Attacked central server using Norm aggregation method.
    """
    def aggregate(self) -> None:
        """
        Aggregate all the received updates using the Norm aggregation method.
        """

        new_state = AggregationService.norm_based_aggregation(self.received_updates, 1)
        self.global_model.load_state_dict(new_state)

        self.current_round += 1
        return
    
# CBAA-FedAvg server
class AttackedCBAAFedAvgServer(AttackedServer):
    """
    Attacked central server using Centroid-Based Anomaly-Aware Federated Averaging aggregation method.
    """
    def aggregate(self) -> None:
        """
        Aggregate all the received updates using the Centroid-Based Anomaly-Aware Federated Averaging method.
        """

        new_state = AggregationService.cbaa_fed_avg(self.received_updates)
        self.global_model.load_state_dict(new_state)

        self.current_round += 1
        return
    
# Trimmed Mean
class AttackedTMeanServer(AttackedServer):
    """
    Attacked central server using Trimmed Mean aggregation method.
    """
    def aggregate(self) -> None:
        """
        Aggregate all the received updates using the Trimmed-Mean method.
        """

        new_state = AggregationService.t_mean(self.received_updates)
        self.global_model.load_state_dict(new_state)

        self.current_round += 1
        return

# RFA
class AttackedRFAServer(AttackedServer):
    """
    Attacked central server using Robust Federated Aggregation method.
    """
    def aggregate(self) -> None:
        """
        Aggregate all the received updates using the Robust Federated Aggregation method.
        """

        new_state = AggregationService.rfa(self.received_updates)
        self.global_model.load_state_dict(new_state)

        self.current_round += 1
        return

# FLTrust
class AttackedFLTrustServer(AttackedServer):
    """
    Attacked central server using FLTrust aggregation method.
    """
    def aggregate(self) -> None:
        """
        Aggregate all the received updates using the FLTrust method.
        """

        new_state = AggregationService.fl_trust(self.received_updates, self.broadcast_model)
        self.global_model.load_state_dict(new_state)

        self.current_round += 1
        return
    
# CLRA
class AttackedCLRAServer(AttackedServer):
    """
    Attacked central server using CLRA aggregation method.
    """
    def aggregate(self) -> None:
        """
        Aggregate all the received updates using the CLRA method.
        """

        new_state = AggregationService.clra(self.received_updates, self.broadcast_model)
        self.global_model.load_state_dict(new_state)

        self.current_round += 1
        return