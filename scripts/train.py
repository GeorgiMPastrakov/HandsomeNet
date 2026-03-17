"""Train the baseline model or HandsomeNet on FreiHAND."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from handsomenet.data.splits import build_freihand_split
from handsomenet.data.tensor_dataset import FreiHANDTensorDataset, collate_tensor_samples
from handsomenet.training.trainer import Trainer, TrainerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["baseline", "handsomenet"], default="baseline")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/freihand"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overfit-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = FreiHANDTensorDataset(args.data_root)
    split = build_freihand_split(dataset.raw_dataset.num_unique_samples, seed=args.seed)

    train_dataset = Subset(dataset, split.train_indices)
    val_dataset = Subset(dataset, split.val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_tensor_samples,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_tensor_samples,
    )

    config = TrainerConfig(
        model_name=args.model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_dir=Path("artifacts/runs") / args.model,
        checkpoint_dir=Path("artifacts/checkpoints"),
        max_train_batches=args.overfit_batches,
        max_val_batches=args.overfit_batches,
    )
    trainer = Trainer(config)
    history = trainer.fit(train_loader=train_loader, val_loader=val_loader, num_epochs=args.epochs)
    print(history)


if __name__ == "__main__":
    main()
