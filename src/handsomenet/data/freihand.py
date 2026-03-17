"""FreiHAND dataset integration entrypoints.

Implementation is intentionally deferred until the first project checkpoint:
verifying the image/annotation/projection/geometry contract.
"""

from __future__ import annotations

from pathlib import Path

from handsomenet.types import DatasetSample


class FreiHANDDataset:
    """Placeholder dataset interface for HandsomeNet v1."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def __len__(self) -> int:
        raise NotImplementedError("FreiHAND dataset loading is not implemented yet.")

    def __getitem__(self, index: int) -> DatasetSample:
        raise NotImplementedError("FreiHAND dataset loading is not implemented yet.")

