# Imports
from typing import Dict, Tuple, Callable, List
import torch.nn as nn
from enum import Enum
import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats
import copy
import pandas as pd

from app.models import EnergyDataset, NormalMLP
from app.attacking_models import MaliciousEntity
from config import config, create_logger

logger = create_logger(__name__)

class ScoringMetric(Enum):
    DISTANCE = 0
    DISTRIBUTION = 1
    SIMILARITY = 2
    DATASET = 3

class ScoringEntity:
    def __init__(self,
                 metric: ScoringMetric = ScoringMetric.DISTANCE,
                 threshold: float = .4
                ):
        
        self.scores: Dict[str: float] = {}
        self.metric: ScoringMetric = metric
        self.threshold: float = threshold
        self.saved_model: nn.Module = None

    def compute_score(self, entity_name: str, model: nn.Module) -> None:

        if not self.saved_model:
            self.scores[entity_name] = 1
            return
        
        metrics: Dict[ScoringMetric, Callable] = {
            ScoringMetric.DISTANCE: self.get_distance,
            ScoringMetric.DISTRIBUTION: self.get_distribution,
            ScoringMetric.SIMILARITY: self.get_similarity,
            ScoringMetric.DATASET: self.get_validation
        }
        
        self.scores[entity_name] = metrics.get(self.metric, lambda _: 0.0)(model)
        return

    def get_distribution(self, model: nn.Module, bins = 100) -> float:
        """
        Distribution based on Jensen-Shannon divergence score
        """
        
        w_a: torch.Tensor = torch.cat([p.data.flatten() for p in model.parameters()])
        w_b: torch.Tensor = torch.cat([p.data.flatten() for p in self.saved_model.parameters()])

        _range: Tuple[float, float] = (
            min(w_a.min().item(), w_b.min().item()),
            max(w_a.max().item(), w_b.max().item())
        )

        pa, _ = np.histogram(w_a, bins = bins, range = _range, density = True)
        pb, _ = np.histogram(w_b, bins = bins, range = _range, density = True)

        pa = (pa + 1e-10) / pa.sum()
        pb = (pb + 1e-10) / pb.sum()

        m = (pa + pb) / 2
        js = (stats.entropy(pa, m, base = 2) + stats.entropy(pb, m, base = 2)) / 2
        return float(1 - js)
        

    def get_distance(self, model: nn.Module, sigma: float = 1.0) -> float:
        
        dist: torch.Tensor = torch.Tensor([0])
        for p_a, p_b in zip(model.parameters(), self.saved_model.parameters()):
            dist += (p_a.data - p_b.data).pow(2).sum()
        dist: float = dist.sqrt().item()

        return torch.exp(torch.tensor(-dist / sigma)).item()

    def get_similarity(self, model: nn.Module) -> float:
        
        w_a: torch.Tensor = torch.cat([p.data.flatten() for p in model.parameters()])
        w_b: torch.Tensor = torch.cat([p.data.flatten() for p in self.saved_model.parameters()])

        _cos: float = F.cosine_similarity(w_a.unsqueeze(0), w_b.unsqueeze(0)).item()
        cosine: float = min(1, max(0, (_cos + 1) / 2))
        sign: float = (torch.sign(w_a) == torch.sign(w_b)).float().mean().item()
        _pearson: float = torch.corrcoef(torch.stack([w_a, w_b]))[0, 1].item()
        magnitude: float = (_pearson + 1) / 2

        return (cosine + sign + magnitude) / 3
    
    def get_validation(self, model: nn.Module, dataset: EnergyDataset) -> float:
        return .0

def check_scoring_entity():
    logger.info('Starting scoring entity check')
    
    # Load model
    base_model: nn.Module = NormalMLP()
    base_model.load_state_dict(torch.load(f'{config.SAVE_DATA_PATH}/test_model.pt'))

    # Create scoring entity
    se: ScoringEntity = ScoringEntity(metric = ScoringMetric.DISTANCE)
    se.saved_model = NormalMLP()
    se.saved_model.load_state_dict(torch.load(f'{config.SAVE_DATA_PATH}/test_model.pt'))

    se.compute_score('entity1', base_model)
    assert se.scores.get('entity1') is not None
    assert se.scores.get('entity1') == 1

    malicious_model = NormalMLP()
    weights = torch.load(f'{config.SAVE_DATA_PATH}/test_model.pt')
    for _, layer in weights.items():
        weights[_] = layer * -1
    malicious_model.load_state_dict(weights)

    se.compute_score('entity2', malicious_model)
    assert se.scores.get('entity2') is not None

    logger.info(f'Scores of entity: {se.scores}')

    logger.info('Scoring entity check ended successfully')
