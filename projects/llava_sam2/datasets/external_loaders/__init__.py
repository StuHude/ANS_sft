"""
External dataset loaders for SAV, SA1B, and OpenImage datasets.

These are copied into the project to make the codebase self-contained.
Only data files remain external.
"""

from .npz_dataloader import MaskletNPZDataset
from .sa1b_loader import SA1BDataset
from .openimage_loader import SegmentationDataset

__all__ = [
    'MaskletNPZDataset',
    'SA1BDataset',
    'SegmentationDataset',
]
