import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

from research.configs.trainer_config import TrainerConfig
from research.adaptive_moe.trainer import AdaptiveMoETrainer

config = TrainerConfig(
    epochs=10
)

trainer = AdaptiveMoETrainer(config)

trainer.fit()
trainer.inspect_gating()