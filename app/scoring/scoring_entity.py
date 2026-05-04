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
from sklearn.metrics import mean_absolute_error
from time import time
import matplotlib as mpl
import matplotlib.pyplot as plt

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
                 threshold: float = .4,
                 metric_parameters: Dict[str, any] = {},
                 **kwargs
                ):
        
        self.scores: Dict[str, float] = {}
        self.metric: ScoringMetric = metric
        self.threshold: float = threshold
        self.saved_model: Dict[str, torch.Tensor] = None
        self.metric_parameters: Dict[str, any] = metric_parameters
        self.rejected_models: int = 0
        
        # Validation dataset
        self._tensor: torch.Tensor = torch.load(f'app/scoring/validation_dataset.pt')
        self._features: torch.Tensor = self._tensor[:, :-3]
        self._targets: torch.Tensor = self._tensor[:, -3:] # Take only the pv
        self.dataset: EnergyDataset = EnergyDataset(self._features, self._targets)

    def compute_score(self, entity_name: str, model: Dict[str, torch.Tensor]) -> float:

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
        return self.scores[entity_name]

    def get_distribution(self, model: Dict[str, torch.Tensor], bins = 100) -> float:
        """
        Distribution based on Jensen-Shannon divergence score
        """

        if 'bins' in self.metric_parameters:
            logger.info(f'Taking bins from parameters: {self.metric_parameters["bins"]}')
            bins = self.metric_parameters['bins']
        
        w_a: torch.Tensor = torch.cat([p.data.flatten() for p in model.values()])
        w_b: torch.Tensor = torch.cat([p.data.flatten() for p in self.saved_model.values()])

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
        

    def get_distance(self, model: Dict[str, torch.Tensor], sigma: float = 1.0) -> float:

        if 'sigma' in self.metric_parameters:
            logger.info(f'Taking sigma from parameters: {self.metric_parameters["sigma"]}')
            sigma = self.metric_parameters['sigma']
        
        dist: torch.Tensor = torch.Tensor([0])
        for p_a, p_b in zip(model.values(), self.saved_model.values()):
            dist += (p_a.data - p_b.data).pow(2).sum()
        dist: float = dist.sqrt().item()

        return torch.exp(torch.tensor(-dist / sigma)).item()

    def get_similarity(self, model: Dict[str, torch.Tensor]) -> float:
        
        w_a: torch.Tensor = torch.cat([p.data.flatten() for p in model.values()])
        w_b: torch.Tensor = torch.cat([p.data.flatten() for p in self.saved_model.values()])

        _cos: float = F.cosine_similarity(w_a.unsqueeze(0), w_b.unsqueeze(0)).item()
        cosine: float = min(1, max(0, (_cos + 1) / 2))
        sign: float = (torch.sign(w_a) == torch.sign(w_b)).float().mean().item()
        _pearson: float = torch.corrcoef(torch.stack([w_a, w_b]))[0, 1].item()
        magnitude: float = (_pearson + 1) / 2

        return (cosine + sign + magnitude) / 3
    
    def get_validation(self, model: Dict[str, torch.Tensor], sigma: float = 1.0) -> float:

        if 'sigma' in self.metric_parameters:
            logger.info(f'Taking sigma from parameters: {self.metric_parameters["sigma"]}')
            sigma = self.metric_parameters['sigma']

        _model = NormalMLP()
        _model.load_state_dict(model)
        
        with torch.no_grad():
            x_val, y_val = self.dataset[:]
            x_val = x_val.to(device = config.DEVICE)

            predictions = _model(x_val)
            mae: float = mean_absolute_error(y_val[:, 1], predictions[:, 1])

            return np.exp(-mae / sigma)

def evaluate_poisonous_model_scoring():
    # Parameters
    run_count: int = int(1e3)
    attack_coef: List[float] = [.1, .2, .3, .5, .8, 1, 2, 3, 5, 8, 10]
    attack_methods: List[str] = ['gaussian_noise', 'gaussian_weights', 'gradient_inversion', 'gradient_amplification']

    # Instanciate entities
    distance_score: ScoringEntity = ScoringEntity(metric = ScoringMetric.DISTANCE)
    distribution_score: ScoringEntity = ScoringEntity(metric = ScoringMetric.DISTRIBUTION)
    similarity_score: ScoringEntity = ScoringEntity(metric = ScoringMetric.SIMILARITY)
    dataset_score: ScoringEntity = ScoringEntity(metric = ScoringMetric.DATASET)

    # Instanciate clean model
    model: NormalMLP = NormalMLP()
    poison_model: NormalMLP = NormalMLP()
    distance_score.saved_model = copy.deepcopy(model)
    distribution_score.saved_model = copy.deepcopy(model)
    similarity_score.saved_model = copy.deepcopy(model)
    dataset_score.saved_model = copy.deepcopy(model)
    scoring_entities: List[ScoringEntity] = [distance_score, distribution_score, similarity_score, dataset_score]

    # Create plotting layout
    fig = plt.figure()
    gs = mpl.gridspec.GridSpec(len(scoring_entities), len(attack_methods), wspace = .25, hspace = .5)

    for _entity_idx in range(len(scoring_entities)):
        entity = scoring_entities[_entity_idx]

        for _attack_idx in range(len(attack_methods)):
            attack = attack_methods[_attack_idx]
            logger.info(f'Testing {entity.metric.name} with {attack}')
            
            # Create plot
            _plot = fig.add_subplot(gs[_attack_idx, _entity_idx])

            # Run simulations
            results: List[List[float]] = []

            _t = time()
        
            for coef in attack_coef:
                results_for_coef: List[float] = []

                for _ in range(run_count):
                    poison_model.load_state_dict(MaliciousEntity.poison_model(poison_model, attack, coef))
                    _score: float = entity.compute_score('', model = poison_model)
                    results_for_coef.append(_score)

                results.append(results_for_coef)

            _t = time() - _t

            # Plot simulations with threshold
            _plot.boxplot(results, medianprops = { 'color': '#F59A00' }, label = 'Score')
            _plot.hlines(entity.threshold, 0, len(attack_coef) - 1, colors = '#bcbcbc', linestyles = 'dashed', label = f'Threshold ({entity.threshold})')
            _plot.spines['top'].set_visible(False)
            _plot.spines['right'].set_visible(False)
            _plot.set_xticklabels([str(_) for _ in attack_coef])
            _plot.set_title(f'{entity.metric.name} (avg: {round(_t/(run_count * len(attack_coef)), 4)}s)')
            _plot.set_ylabel(attack)
            # _plot.legend()

    plt.show()
    return

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
