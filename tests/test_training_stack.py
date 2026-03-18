import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from handsomenet.data.tensor_dataset import collate_tensor_samples
from handsomenet.training.metrics import mean_2d_pixel_error
from handsomenet.training.runtime import resolve_device
from handsomenet.training.trainer import Trainer, TrainerConfig
from handsomenet.types import GeometryMetadata, TensorBatch, TensorSample


class DummyTensorDataset(Dataset[TensorSample]):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> TensorSample:
        image = torch.full((3, 224, 224), 0.5, dtype=torch.float32)
        targets = torch.full((21, 2), 0.25, dtype=torch.float32)
        geometry = GeometryMetadata(
            original_width=224,
            original_height=224,
            resize_scale=1.0,
            pad_x=0.0,
            pad_y=0.0,
            final_width=224,
            final_height=224,
        )
        return TensorSample(
            image=image,
            targets=targets,
            geometry=geometry,
            sample_id=str(index),
            annotation_index=index % 2,
            image_variant_index=index // 2,
            image_path=Path(f"dummy_{index}.jpg"),
        )


def test_mean_2d_pixel_error_returns_finite_value() -> None:
    predictions = torch.zeros(2, 21, 2)
    targets = torch.ones(2, 21, 2) * 0.5

    error = mean_2d_pixel_error(predictions, targets, width=224, height=224)

    assert torch.isfinite(error)
    assert error.item() > 0.0


def test_collate_tensor_samples_returns_expected_batch() -> None:
    dataset = DummyTensorDataset()
    samples = [dataset[0], dataset[1]]

    batch = collate_tensor_samples(samples)

    assert isinstance(batch, TensorBatch)
    assert batch.images.shape == (2, 3, 224, 224)
    assert batch.targets.shape == (2, 21, 2)
    assert batch.annotation_indices.shape == (2,)
    assert batch.image_variant_indices.shape == (2,)


def test_trainer_checkpoint_round_trip_and_fit(tmp_path: Path) -> None:
    dataset = DummyTensorDataset()
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_tensor_samples)
    config = TrainerConfig(
        model_name="baseline",
        device="cpu",
        output_dir=tmp_path / "runs",
        checkpoint_dir=tmp_path / "checkpoints",
        max_train_batches=1,
        max_val_batches=1,
    )
    trainer = Trainer(config)

    history = trainer.fit(train_loader=loader, val_loader=loader, num_epochs=1)

    assert len(history) == 1
    assert history[0]["train_loss"] >= 0.0
    checkpoint_path = config.checkpoint_dir / "baseline_epoch_1.pt"
    assert checkpoint_path.exists()

    loaded_epoch = trainer.load_checkpoint(checkpoint_path)
    assert loaded_epoch == 1

    history_path = config.output_dir / "history.json"
    latest_checkpoint_path = config.output_dir / "latest_checkpoint.txt"
    assert history_path.exists()
    assert latest_checkpoint_path.exists()

    persisted_history = json.loads(history_path.read_text())
    assert len(persisted_history) == 1
    assert persisted_history[0]["epoch"] == 1.0
    assert latest_checkpoint_path.read_text().strip() == str(checkpoint_path)

    resumed_history = trainer.fit(
        train_loader=loader,
        val_loader=loader,
        num_epochs=1,
        start_epoch=loaded_epoch,
    )
    assert len(resumed_history) == 2
    assert resumed_history[-1]["epoch"] == 2.0


def test_trainer_evaluate_writes_metrics_and_visualizations(tmp_path: Path) -> None:
    dataset = DummyTensorDataset()
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_tensor_samples)
    config = TrainerConfig(
        model_name="baseline",
        device="cpu",
        output_dir=tmp_path / "eval_runs",
        checkpoint_dir=tmp_path / "checkpoints",
        max_train_batches=1,
        max_val_batches=1,
    )
    trainer = Trainer(config)
    trainer.fit(train_loader=loader, val_loader=loader, num_epochs=1)

    checkpoint_path = config.checkpoint_dir / "baseline_epoch_1.pt"
    eval_output_dir = tmp_path / "eval_only"
    eval_trainer = Trainer(
        TrainerConfig(
            model_name="baseline",
            device="cpu",
            output_dir=eval_output_dir,
            checkpoint_dir=tmp_path / "unused_checkpoints",
            max_val_batches=1,
        )
    )
    loaded_epoch = eval_trainer.load_checkpoint(checkpoint_path)
    assert loaded_epoch == 1

    metrics = eval_trainer.evaluate(loader, split_name="val")

    assert metrics["loss"] >= 0.0
    assert metrics["pixel_error"] >= 0.0
    metrics_path = eval_output_dir / "val_metrics.json"
    assert metrics_path.exists()
    persisted_metrics = json.loads(metrics_path.read_text())
    assert persisted_metrics == metrics
    visualization_path = eval_output_dir / "visualizations" / "epoch_val_sample_0.jpg"
    assert visualization_path.exists()


def test_resolve_device_prefers_available_backends() -> None:
    resolved = resolve_device("auto")

    assert resolved in {"mps", "cuda", "cpu"}
    assert resolve_device("cpu") == "cpu"
