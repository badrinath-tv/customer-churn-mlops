import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

from research.data.validation import DatasetValidator

validator = DatasetValidator(
    dataset_path=Path("data/raw/Churn_Modelling.csv"),
    target_column="Exited",
)

df = validator.validate()