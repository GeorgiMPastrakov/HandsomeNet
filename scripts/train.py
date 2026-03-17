"""Train the baseline model or HandsomeNet on FreiHAND."""

from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader, Subset

from handsomenet.data.splits import build_freihand_split
from handsomenet.data.tensor_dataset import FreiHANDTensorDataset, collate_tensor_samples
from handsomenet.training.runtime import resolve_device, resolve_run_name
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
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--limit-train-unique", type=int, default=None)
    parser.add_argument("--limit-val-unique", type=int, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--stop-train-pixel-error", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = FreiHANDTensorDataset(args.data_root)
    split = build_freihand_split(
        dataset.raw_dataset.num_unique_samples,
        val_fraction=args.val_fraction,
        seed=args.seed,
        limit_train_unique=args.limit_train_unique,
        limit_val_unique=args.limit_val_unique,
    )

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

    device = resolve_device(args.device)
    run_name = args.run_name or resolve_run_name(
        model_name=args.model,
        limit_train_unique=args.limit_train_unique,
        limit_val_unique=args.limit_val_unique,
        resume_checkpoint=args.resume_checkpoint,
    )
    output_dir = Path("artifacts/runs") / run_name
    checkpoint_dir = Path("artifacts/checkpoints") / run_name
    config = TrainerConfig(
        model_name=args.model,
        device=device,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        max_train_batches=args.overfit_batches,
        max_val_batches=args.overfit_batches,
        stop_train_pixel_error=args.stop_train_pixel_error,
    )
    trainer = Trainer(config)
    start_epoch = 0
    if args.resume_checkpoint is not None:
        start_epoch = trainer.load_checkpoint(args.resume_checkpoint)
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        start_epoch=start_epoch,
    )
    print(history)


if __name__ == "__main__":
    main()
