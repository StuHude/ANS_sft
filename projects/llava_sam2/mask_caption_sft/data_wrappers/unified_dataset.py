"""
Unified Dataset for Mask Captioning + Segmentation SFT Training

This dataset combines:
1. SAV dataset (video masklets with frame pairs)
2. SA-1B dataset (single-frame segmentation)
3. OpenImage dataset (single-frame segmentation)
4. RefCOCO dataset (referring expression segmentation)

Training flow:
- SAV/SA-1B/OpenImage: image+mask → caption → image2+caption → mask2' (loop)
- RefCOCO: image+caption → mask' (direct segmentation, no captioning)
"""

import torch
import random
from torch.utils.data import Dataset, ConcatDataset


class MaskCaptionSFTDataset(ConcatDataset):
    """
    Unified dataset that concatenates all wrapper datasets.

    Args:
        datasets: List of dataset wrappers (SAV, SA-1B, OpenImage, RefCOCO)
        weights: Optional sampling weights for each dataset
    """

    def __init__(self, datasets: list, weights: list = None):
        super().__init__(datasets)
        self.weights = weights

        # Print dataset sizes
        print("\n" + "=" * 60)
        print("Unified Mask Captioning SFT Dataset")
        print("=" * 60)
        for i, ds in enumerate(datasets):
            dataset_type = getattr(ds, '__class__').__name__
            print(f"  [{i}] {dataset_type}: {len(ds)} samples")
        print(f"\n  Total samples: {len(self)}")
        print("=" * 60 + "\n")


class MaskCaptionCollator:
    """
    Custom collator for mask captioning SFT training.

    Handles different data formats from different datasets:
    - SAV/SA-1B/OpenImage: Raw images and masks (need preprocessing)
    - RefCOCO: Already preprocessed data_dict

    Returns batches grouped by dataset type for efficient processing.
    """

    def __init__(self, preprocessor=None):
        """
        Args:
            preprocessor: Sa2VADataPreprocessor for image/mask preprocessing
        """
        self.preprocessor = preprocessor

    def __call__(self, batch):
        """
        Collate a batch of samples.

        Args:
            batch: List of dicts from dataset __getitem__

        Returns:
            dict with keys:
            - batch_data: List of samples grouped by type
            - batch_size: Number of samples
        """
        # Group samples by dataset type
        sav_samples = []
        sa1b_samples = []
        openimage_samples = []
        refcoco_samples = []

        for sample in batch:
            if sample is None:
                continue

            dataset_type = sample.get('dataset_type', 'unknown')

            if dataset_type == 'sav':
                sav_samples.append(sample)
            elif dataset_type == 'sa1b':
                sa1b_samples.append(sample)
            elif dataset_type == 'openimage':
                openimage_samples.append(sample)
            elif dataset_type == 'refcoco':
                refcoco_samples.append(sample)

        return {
            'sav_samples': sav_samples,
            'sa1b_samples': sa1b_samples,
            'openimage_samples': openimage_samples,
            'refcoco_samples': refcoco_samples,
            'batch_size': len(batch),
        }
