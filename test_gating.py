import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

from research.gating.gating_trainer import GatingTrainer

trainer = GatingTrainer()

trainer.summary()