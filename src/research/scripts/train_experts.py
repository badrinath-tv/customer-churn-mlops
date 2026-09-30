"""
===========================================================================
Adaptive Mixture of Experts (A-MoE)

Train Expert Models

Author : Badrinath & Abdul Azim
=========================================================================== 
"""

import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Add project src to Python path
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT / "src"))

# --------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------

from research.data.dataset import load_dataset

from research.experts.catboost_expert import CatBoostExpert
from research.experts.lightgbm_expert import LightGBMExpert
from research.experts.xgboost_expert import XGBoostExpert
from research.experts.expert_manager import ExpertManager

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DATASET_NAME = "dataset_1"

MODEL_SAVE_PATH = Path(
    "models/experts/dataset_1"
)

MODEL_SAVE_PATH.mkdir(
    parents=True,
    exist_ok=True,
)

# --------------------------------------------------------------------------
# Load Dataset
# --------------------------------------------------------------------------

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

data = load_dataset(DATASET_NAME)

X_train = data["X_train"]
X_test = data["X_test"]

y_train = data["y_train"]
y_test = data["y_test"]

feature_names = data["feature_names"]

print("\nDataset Loaded Successfully")

print(f"Training Samples : {X_train.shape}")
print(f"Testing Samples  : {X_test.shape}")

print("\nFeatures")

for feature in feature_names:
    print(f"• {feature}")

# --------------------------------------------------------------------------
# Create Expert Manager
# --------------------------------------------------------------------------

manager = ExpertManager()

manager.register(
    CatBoostExpert()
)

manager.register(
    LightGBMExpert()
)

manager.register(
    XGBoostExpert()
)

manager.summary()

# --------------------------------------------------------------------------
# Train Experts
# --------------------------------------------------------------------------

print()
print("=" * 70)
print("TRAINING EXPERTS")
print("=" * 70)

manager.fit(
    X_train,
    y_train,
)

# --------------------------------------------------------------------------
# Save Expert Models
# --------------------------------------------------------------------------

manager.save_all(
    MODEL_SAVE_PATH
)

print("\nExperts saved successfully.")

print(MODEL_SAVE_PATH)

# --------------------------------------------------------------------------
# Generate Prediction Matrix
# --------------------------------------------------------------------------

prediction_matrix = manager.predict_proba(
    X_test
)

print("\nPrediction Matrix Shape")
print(prediction_matrix.shape)

print("\nSample Predictions")
print(prediction_matrix[:5])

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

print()
print("=" * 70)
print("EXPERT LAYER COMPLETED SUCCESSFULLY")
print("=" * 70)

print(f"\nNumber of Experts : {prediction_matrix.shape[1]}")
print(f"Test Samples      : {prediction_matrix.shape[0]}")
print(f"Feature Count     : {X_train.shape[1]}")

print("\nExpert models are ready for the Adaptive Gating Network.")