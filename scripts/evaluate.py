"""Evaluate a saved baseline or HandsomeNet checkpoint on the FreiHAND validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader, Subset

from handsomenet.data.splits import build_freihand_split
from handsomenet.data.tensor_dataset import FreiHANDTensorDataset, collate_tensor_samples
from handsomenet.training.runtime import resolve_device
from handsomenet.training.trainer import Trainer, TrainerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["baseline", "handsomenet"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/freihand"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--limit-val-unique", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = FreiHANDTensorDataset(args.data_root)
    split = build_freihand_split(
        dataset.raw_dataset.num_unique_samples,
        val_fraction=args.val_fraction,
        seed=args.seed,
        limit_val_unique=args.limit_val_unique,
    )
    val_dataset = Subset(dataset, split.val_indices)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_tensor_samples,
    )

    output_dir = args.output_dir or Path("artifacts/evals") / args.checkpoint.stem
    trainer = Trainer(
        TrainerConfig(
            model_name=args.model,
            device=resolve_device(args.device),
            output_dir=output_dir,
            checkpoint_dir=output_dir / "checkpoints_unused",
        )
    )
    trainer.load_checkpoint(args.checkpoint)
    metrics = trainer.evaluate(val_loader, split_name="val")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
