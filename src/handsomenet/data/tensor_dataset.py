"""Training-facing tensor dataset wrappers."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from handsomenet.data.freihand import FreiHANDDataset
from handsomenet.data.splits import image_index_to_annotation_index, image_index_to_variant_index
from handsomenet.types import TensorBatch, TensorSample


class FreiHANDTensorDataset(Dataset[TensorSample]):
    """Torch-friendly wrapper over the raw FreiHAND dataset contract."""

    def __init__(self, root: Path) -> None:
        self.raw_dataset = FreiHANDDataset(root)

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, index: int) -> TensorSample:
        raw_sample = self.raw_dataset[index]
        image_tensor = torch.from_numpy(raw_sample.image.copy()).permute(2, 0, 1).float() / 255.0
        targets_tensor = torch.from_numpy(raw_sample.targets.normalized_2d).float()
        annotation_index = image_index_to_annotation_index(
            index, self.raw_dataset.num_unique_samples
        )
        image_variant_index = image_index_to_variant_index(
            index, self.raw_dataset.num_unique_samples
        )

        return TensorSample(
            image=image_tensor,
            targets=targets_tensor,
            geometry=raw_sample.geometry,
            sample_id=raw_sample.sample_id,
            annotation_index=annotation_index,
            image_variant_index=image_variant_index,
            image_path=raw_sample.image_path,
        )


def collate_tensor_samples(samples: list[TensorSample]) -> TensorBatch:
    """Collate tensor samples into a batch with preserved metadata."""

    return TensorBatch(
        images=torch.stack([sample.image for sample in samples], dim=0),
        targets=torch.stack([sample.targets for sample in samples], dim=0),
        geometries=tuple(sample.geometry for sample in samples),
        sample_ids=tuple(sample.sample_id for sample in samples),
        annotation_indices=torch.tensor(
            [sample.annotation_index for sample in samples], dtype=torch.long
        ),
        image_variant_indices=torch.tensor(
            [sample.image_variant_index for sample in samples], dtype=torch.long
        ),
        image_paths=tuple(sample.image_path for sample in samples),
    )
