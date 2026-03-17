"""Trainer for HandsomeNet model-to-first-train workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from handsomenet.constants import INPUT_HEIGHT, INPUT_WIDTH
from handsomenet.models.factory import build_model
from handsomenet.training.losses import CoordinateMSELoss
from handsomenet.training.metrics import mean_2d_pixel_error
from handsomenet.types import TensorBatch
from handsomenet.utils.visualization import save_prediction_overlay


@dataclass(frozen=True)
class TrainerConfig:
    """Configuration for baseline or HandsomeNet training."""

    model_name: str
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    device: str = "cpu"
    output_dir: Path = Path("artifacts/runs/default")
    checkpoint_dir: Path = Path("artifacts/checkpoints")
    max_train_batches: int | None = None
    max_val_batches: int | None = None


@dataclass(frozen=True)
class EpochMetrics:
    """Aggregated metrics for one training or validation epoch."""

    loss: float
    pixel_error: float


class Trainer:
    """Shared trainer for the baseline model and HandsomeNet."""

    def __init__(self, config: TrainerConfig, model: nn.Module | None = None) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.model = model if model is not None else build_model(config.model_name)
        self.model.to(self.device)
        self.loss_fn = CoordinateMSELoss()
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def fit(
        self,
        train_loader: DataLoader[TensorBatch],
        val_loader: DataLoader[TensorBatch],
        num_epochs: int,
    ) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        for epoch in range(1, num_epochs + 1):
            train_metrics = self.run_epoch(train_loader, training=True)
            val_metrics = self.run_epoch(val_loader, training=False)
            self.save_checkpoint(epoch)
            self.visualize_predictions(val_loader, epoch)
            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": train_metrics.loss,
                    "train_pixel_error": train_metrics.pixel_error,
                    "val_loss": val_metrics.loss,
                    "val_pixel_error": val_metrics.pixel_error,
                }
            )
        return history

    def run_epoch(self, loader: DataLoader[TensorBatch], training: bool) -> EpochMetrics:
        self.model.train(training)
        total_loss = 0.0
        total_pixel_error = 0.0
        num_batches = 0

        for batch_index, batch in enumerate(loader):
            if training and self.config.max_train_batches is not None:
                if batch_index >= self.config.max_train_batches:
                    break
            if not training and self.config.max_val_batches is not None:
                if batch_index >= self.config.max_val_batches:
                    break

            images = batch.images.to(self.device)
            targets = batch.targets.to(self.device)

            context = torch.enable_grad() if training else torch.no_grad()
            with context:
                predictions = self.model(images)
                loss = self.loss_fn(predictions, targets)
                pixel_error = mean_2d_pixel_error(
                    predictions, targets, width=INPUT_WIDTH, height=INPUT_HEIGHT
                )

            if training:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

            total_loss += float(loss.detach().cpu())
            total_pixel_error += float(pixel_error.detach().cpu())
            num_batches += 1

        if num_batches == 0:
            raise ValueError("No batches were processed in run_epoch.")

        return EpochMetrics(
            loss=total_loss / num_batches,
            pixel_error=total_pixel_error / num_batches,
        )

    def save_checkpoint(self, epoch: int) -> Path:
        checkpoint_path = self.config.checkpoint_dir / f"{self.config.model_name}_epoch_{epoch}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_name": self.config.model_name,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": {
                    "learning_rate": self.config.learning_rate,
                    "weight_decay": self.config.weight_decay,
                    "max_grad_norm": self.config.max_grad_norm,
                },
            },
            checkpoint_path,
        )
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: Path) -> int:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return int(checkpoint["epoch"])

    def visualize_predictions(self, loader: DataLoader[TensorBatch], epoch: int) -> None:
        self.model.eval()
        batch = next(iter(loader))
        images = batch.images.to(self.device)
        with torch.no_grad():
            predictions = self.model(images).cpu().numpy()

        output_dir = self.config.output_dir / "visualizations"
        targets = batch.targets.numpy()
        images_np = batch.images.numpy()

        for index in range(min(4, images_np.shape[0])):
            target_pixels = _normalized_to_input_pixels(targets[index])
            prediction_pixels = _normalized_to_input_pixels(predictions[index])
            save_prediction_overlay(
                image_chw=images_np[index],
                target_points_xy=target_pixels,
                prediction_points_xy=prediction_pixels,
                output_path=output_dir / f"epoch_{epoch}_sample_{index}.jpg",
            )


def _normalized_to_input_pixels(points_xy: np.ndarray) -> np.ndarray:
    pixel_scale = np.array([INPUT_WIDTH, INPUT_HEIGHT], dtype=np.float32)
    return points_xy * pixel_scale
