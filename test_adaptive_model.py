import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

import torch

from research.adaptive_moe.model import AdaptiveMoE

model = AdaptiveMoE(
    input_dim=10,
    n_experts=3,
)

features = torch.randn(
    16,
    10,
)

expert_predictions = torch.rand(
    16,
    3,
)

probabilities, weights = model(
    features,
    expert_predictions,
)

print("Probabilities:", probabilities.shape)
print("Weights:", weights.shape)