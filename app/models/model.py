# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config, create_logger

# create Logger
logger = create_logger(__name__)

# Create model class
class NormalMLP(nn.Module):
    def __init__(self, 
                 sequence_len: int = config.LOOKBACK, 
                 num_features: int = config.NUM_FEATURES,
                 hidden: int = 32,
                 dropout: float = .2,
                 out_size: int = 3):       # 3 outputs for load, pv and net
        
        super(NormalMLP, self).__init__()
        input_size: int = sequence_len * num_features
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden), # 48 * 7 = 336 > 32
            nn.ReLU(),
            nn.Linear(hidden, hidden),     # 32 > 32
            nn.ReLU(),
            nn.Dropout(p = dropout),
            nn.Linear(hidden, out_size)    # 32 > 3
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class SoftGatedMoE(nn.Module):
    def __init__(self,
                 sequence_len: int = config.LOOKBACK, 
                 num_features: int = config.NUM_FEATURES,
                 num_experts: int = 4,
                 expert_units: int = 8,
                 shared_units: int = 16,
                 dropout: float = .2,
                 out_size: int = 3):
        
        super(SoftGatedMoE, self).__init__()
        self.num_experts = num_experts
        self.expert_units = expert_units
        input_size: int = sequence_len * num_features
        self.flatten = nn.Flatten()

        self.experts: list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, expert_units),
                nn.ReLU()
            ) for _ in range(num_experts)
        ])

        self.gate = nn.Linear(input_size, num_experts)

        self.shared = nn.Sequential(
            nn.Linear(expert_units, shared_units),
            nn.ReLU(),
            nn.Linear(shared_units, shared_units),
            nn.ReLU(),
            nn.Dropout(p = dropout),
            nn.Linear(shared_units, out_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = self.flatten(x)
        expert_outputs = torch.stack([expert(x_flat) for expert in self.experts], dim = 1)
        gate_weights = F.softmax(self.gate(x_flat), dim = -1)
        mixed = (gate_weights.unsqueeze(-1) * expert_outputs).sum(dim = 1)
        return self.shared(mixed)
    
# Method for fast check
def check_models():
    N = 16
    x = torch.randn(N, config.LOOKBACK, config.NUM_FEATURES)
    mlp = NormalMLP()
    moe = SoftGatedMoE()

    mlp_out = mlp(x)
    moe_out = moe(x)

    assert mlp_out.shape == (N, 3), f"Expected MLP output shape (N, 1), got {mlp_out.shape}"
    assert moe_out.shape == (N, 3), f"Expected MoE output shape (N, 1), got {moe_out.shape}"

    logger.info(f'Input: {tuple(x.shape)}')
    logger.info(f'MLP -> Output: {tuple(mlp_out.shape)}')
    logger.info(f'MoE -> Output: {tuple(moe_out.shape)}')

    n_mlp = sum(p.numel() for p in mlp.parameters())
    n_moe = sum(p.numel() for p in moe.parameters())
    logger.info(f'parameters: MLP: {n_mlp}, MoE: {n_moe}')