from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from handsomenet.data.tensor_dataset import collate_tensor_samples
from handsomenet.training.metrics import mean_2d_pixel_error
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


def test_trainer_checkpoint_round_trip_and_fit() -> None:
    dataset = DummyTensorDataset()
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_tensor_samples)
    config = TrainerConfig(
        model_name="baseline",
        device="cpu",
        output_dir=Path("artifacts/runs/test_trainer"),
        checkpoint_dir=Path("artifacts/checkpoints/test_trainer"),
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
