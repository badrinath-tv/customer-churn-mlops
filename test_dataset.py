import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

from research.data.dataset import load_dataset

data = load_dataset("dataset_1")

print("\nReturned Keys")
print("-" * 40)

for key in data:
    print(key)

print("\nTraining Shape")
print(data["X_train"].shape)

print("\nTesting Shape")
print(data["X_test"].shape)

print("\nFeatures")

for feature in data["feature_names"]:
    print(feature)