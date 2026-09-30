"""
===========================================================================
Adaptive Mixture of Experts (A-MoE)

Adaptive MoE Trainer

Author : Badrinath & Abdul Azim
===========================================================================
"""

from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from research.configs.trainer_config import TrainerConfig
from research.data.dataset import load_dataset
from research.adaptive_moe.model import AdaptiveMoE

from research.experts.expert_manager import ExpertManager
from research.experts.catboost_expert import CatBoostExpert
from research.experts.lightgbm_expert import LightGBMExpert
from research.experts.xgboost_expert import XGBoostExpert

from research.gating.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)

from research.gating.history import TrainingHistory
from research.gating.losses import binary_cross_entropy


class AdaptiveMoETrainer:

    def __init__(
        self,
        config: TrainerConfig,
    ):

        self.config = config
        self.device = config.device

        print("=" * 70)
        print("INITIALIZING ADAPTIVE MoE TRAINER")
        print("=" * 70)

        # ----------------------------------------------------------
        # Dataset
        # ----------------------------------------------------------

        data = load_dataset(config.dataset_name)

        self.X_train = data["X_train"]
        self.X_test = data["X_test"]

        self.y_train = data["y_train"]
        self.y_test = data["y_test"]

        self.feature_names = data["feature_names"]

        # ----------------------------------------------------------
        # Experts
        # ----------------------------------------------------------

        self.manager = ExpertManager()

        self.manager.register(CatBoostExpert())
        self.manager.register(LightGBMExpert())
        self.manager.register(XGBoostExpert())

        self.manager.load_all(
            Path("models/experts/dataset_1")
        )

        print("Experts Loaded.")

        # ----------------------------------------------------------
        # Cache Expert Predictions
        # ----------------------------------------------------------

        print("Caching Expert Predictions...")

        train_predictions = self.manager.predict_proba(
            self.X_train
        )

        test_predictions = self.manager.predict_proba(
            self.X_test
        )

        print("Done.")

        # ----------------------------------------------------------
        # TensorDataset
        # ----------------------------------------------------------

        self.train_dataset = TensorDataset(

            torch.tensor(
                self.X_train.values,
                dtype=torch.float32,
            ),

            torch.tensor(
                train_predictions,
                dtype=torch.float32,
            ),

            torch.tensor(
                self.y_train.values,
                dtype=torch.float32,
            ),

        )

        self.test_dataset = TensorDataset(

            torch.tensor(
                self.X_test.values,
                dtype=torch.float32,
            ),

            torch.tensor(
                test_predictions,
                dtype=torch.float32,
            ),

            torch.tensor(
                self.y_test.values,
                dtype=torch.float32,
            ),

        )

        # ----------------------------------------------------------
        # DataLoaders
        # ----------------------------------------------------------

        self.train_loader = DataLoader(

            self.train_dataset,

            batch_size=config.batch_size,

            shuffle=True,

        )

        self.test_loader = DataLoader(

            self.test_dataset,

            batch_size=config.batch_size,

            shuffle=False,

        )

        # ----------------------------------------------------------
        # Model
        # ----------------------------------------------------------

        self.model = AdaptiveMoE(

            input_dim=len(self.feature_names),

            n_experts=3,

        ).to(self.device)

        # ----------------------------------------------------------
        # Optimizer
        # ----------------------------------------------------------

        self.optimizer = optim.Adam(

            self.model.parameters(),

            lr=config.learning_rate,

        )

        # ----------------------------------------------------------
        # Scheduler
        # ----------------------------------------------------------

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(

            self.optimizer,

            mode="min",

            factor=0.5,

            patience=3,

        )

        # ----------------------------------------------------------
        # Loss
        # ----------------------------------------------------------

        self.criterion = binary_cross_entropy()

        # ----------------------------------------------------------
        # Callbacks
        # ----------------------------------------------------------

        self.early_stopping = EarlyStopping(

            patience=config.early_stopping_patience,

        )

        self.checkpoint = ModelCheckpoint(

            config.model_path,

        )

        # ----------------------------------------------------------
        # History
        # ----------------------------------------------------------

        self.history = TrainingHistory()

        print("\nInitialization Completed Successfully.")

    # ==============================================================
    # Train One Epoch
    # ==============================================================

    def train_one_epoch(self):

        self.model.train()

        running_loss = 0.0

        for (

            features,

            expert_predictions,

            labels,

        ) in self.train_loader:

            features = features.to(self.device)

            expert_predictions = expert_predictions.to(self.device)

            labels = labels.to(self.device)

            # -------------------- Forward --------------------

            outputs, weights = self.model(

                features,

                expert_predictions,

            )

            loss = self.criterion(

                outputs,

                labels,

            )

            # -------------------- Backward --------------------

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

        # ==============================================================
    # Validation
    # ==============================================================

    def validate(self):

        self.model.eval()

        running_loss = 0.0

        correct = 0
        total = 0

        with torch.no_grad():

            for (

                features,

                expert_predictions,

                labels,

            ) in self.test_loader:

                features = features.to(self.device)

                expert_predictions = expert_predictions.to(self.device)

                labels = labels.to(self.device)

                outputs, _ = self.model(

                    features,

                    expert_predictions,

                )

                loss = self.criterion(

                    outputs,

                    labels,

                )

                running_loss += loss.item()

                predictions = (

                    outputs >= 0.5

                ).float()

                correct += (

                    predictions == labels

                ).sum().item()

                total += labels.size(0)

        validation_loss = (

            running_loss /

            len(self.test_loader)

        )

        accuracy = correct / total

        return validation_loss, accuracy
    # ==============================================================
    # Inspect Gating Network
    # ==============================================================

    def inspect_gating(self, samples=5):

        self.model.eval()

        print("\n" + "=" * 70)
        print("GATING WEIGHTS")
        print("=" * 70)

        with torch.no_grad():

            features, expert_predictions, _ = next(iter(self.test_loader))

            features = features.to(self.device)
            expert_predictions = expert_predictions.to(self.device)

            _, weights = self.model(
                features,
                expert_predictions,
            )

            for i in range(min(samples, len(weights))):

                print(
                    f"Sample {i+1}: "
                    f"{weights[i].cpu().numpy()}"
                )

    

    # ==============================================================
    # Fit
    # ==============================================================

    def fit(self):

        print()

        print("=" * 70)
        print("TRAINING ADAPTIVE MoE")
        print("=" * 70)

        for epoch in range(self.config.epochs):

            train_loss = self.train_one_epoch()

            validation_loss, accuracy = self.validate()

            self.scheduler.step(
                validation_loss
            )

            self.history.update(

                train_loss,

                validation_loss,

                accuracy,

            )

            self.checkpoint.step(

                validation_loss,

                self.model,

            )

            if self.early_stopping.step(

                validation_loss

            ):

                print("\nEarly Stopping Triggered.")

                break

            print(

                f"Epoch {epoch+1:03d}/{self.config.epochs}"

                f" | Train Loss : {train_loss:.6f}"

                f" | Val Loss : {validation_loss:.6f}"

                f" | Accuracy : {accuracy:.4f}"

            )

        print("\nTraining Completed Successfully.")