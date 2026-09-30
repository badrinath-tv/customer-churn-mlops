"""
===========================================================================
Adaptive Mixture of Experts (A-MoE)

Gating Network Trainer

Author : Badrinath & Abdul Azim
===========================================================================
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from research.data.dataset import load_dataset
from research.experts.expert_manager import ExpertManager
from research.experts.catboost_expert import CatBoostExpert
from research.experts.lightgbm_expert import LightGBMExpert
from research.experts.xgboost_expert import XGBoostExpert
from research.fusion.weighted_fusion import WeightedFusion
from research.gating.gating_network import GatingNetwork


class GatingTrainer:

    def __init__(
        self,
        dataset_name="dataset_1",
        learning_rate=1e-3,
        epochs=50,
        device=None,
    ):

        self.dataset_name = dataset_name
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.device = (
            device
            if device
            else (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        print("=" * 70)
        print("INITIALIZING GATING TRAINER")
        print("=" * 70)

        print(f"Device : {self.device}")

        # ==============================================================
        # Load Dataset
        # ==============================================================

        self.data = load_dataset(
            self.dataset_name
        )

        self.X_train = self.data["X_train"]
        self.X_test = self.data["X_test"]

        self.y_train = self.data["y_train"]
        self.y_test = self.data["y_test"]

        self.feature_names = self.data["feature_names"]

        # ==============================================================
        # Expert Manager
        # ==============================================================

        self.manager = ExpertManager()

        self.manager.register(
            CatBoostExpert()
        )

        self.manager.register(
            LightGBMExpert()
        )

        self.manager.register(
            XGBoostExpert()
        )

        # ==============================================================
        # Load Expert Models
        # ==============================================================

        self.manager.load_all(
            Path("models/experts/dataset_1")
        )

        print("\nExpert models loaded successfully.")

        # ==============================================================
        # Generate Expert Prediction Matrix (Only Once)
        # ==============================================================

        print("\nGenerating Expert Prediction Matrix...")

        self.train_prediction_matrix = (
            self.manager.predict_proba(
                self.X_train
            )
        )

        self.test_prediction_matrix = (
            self.manager.predict_proba(
                self.X_test
            )
        )

        print("Done.")

        print(
            f"Training Prediction Matrix : "
            f"{self.train_prediction_matrix.shape}"
        )

        print(
            f"Testing Prediction Matrix  : "
            f"{self.test_prediction_matrix.shape}"
        )

        # ==============================================================
        # Convert Everything to Torch Tensors
        # ==============================================================

        print("\nConverting Data to PyTorch Tensors...")

        self.X_train_tensor = torch.tensor(
            self.X_train.values,
            dtype=torch.float32,
            device=self.device,
        )

        self.X_test_tensor = torch.tensor(
            self.X_test.values,
            dtype=torch.float32,
            device=self.device,
        )

        self.train_prediction_tensor = torch.tensor(
            self.train_prediction_matrix,
            dtype=torch.float32,
            device=self.device,
        )

        self.test_prediction_tensor = torch.tensor(
            self.test_prediction_matrix,
            dtype=torch.float32,
            device=self.device,
        )

        self.y_train_tensor = torch.tensor(
            self.y_train.values,
            dtype=torch.float32,
            device=self.device,
        )

        self.y_test_tensor = torch.tensor(
            self.y_test.values,
            dtype=torch.float32,
            device=self.device,
        )

        print("Tensor conversion completed.")

        # ==============================================================
        # Gating Network
        # ==============================================================

        self.gating_network = GatingNetwork(

            input_dim=self.X_train.shape[1],

            n_experts=3,

        ).to(self.device)

        # ==============================================================
        # Fusion Layer
        # ==============================================================

        self.fusion = WeightedFusion()

        # ==============================================================
        # Optimizer
        # ==============================================================

        self.optimizer = optim.Adam(

            self.gating_network.parameters(),

            lr=self.learning_rate,

        )

        # ==============================================================
        # Loss Function
        # ==============================================================

        self.criterion = nn.BCELoss()

        print("\nGating Network")

        print(self.gating_network)

        print("\nInitialization Complete.")

    # ------------------------------------------------------------------

    def summary(self):

        print()

        print("=" * 70)
        print("TRAINER SUMMARY")
        print("=" * 70)

        print(f"Training Samples : {self.X_train.shape}")
        print(f"Testing Samples  : {self.X_test.shape}")

        print(
            f"Training Prediction Matrix : "
            f"{self.train_prediction_matrix.shape}"
        )

        print(
            f"Testing Prediction Matrix  : "
            f"{self.test_prediction_matrix.shape}"
        )

        print(f"Features         : {len(self.feature_names)}")
        print("Experts          : 3")
        print(f"Epochs           : {self.epochs}")
        print(f"Learning Rate    : {self.learning_rate}")
        print(f"Device           : {self.device}")