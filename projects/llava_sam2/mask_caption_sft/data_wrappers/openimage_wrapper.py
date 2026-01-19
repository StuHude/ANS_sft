"""
OpenImage Dataset Wrapper for Mask Captioning SFT Training

OpenImage dataset returns: (image, mask, label_name)
Training flow:
1. Input (image, mask) → model generates caption
2. Input (image, caption) → model segments → mask'
3. Compute IoU(mask', mask) as loss
"""

import sys
import torch
from torch.utils.data import Dataset
sys.path.insert(0, '/data/xyc/openv7/data')
from openiamgev7 import SegmentationDataset


class OpenImageDatasetWrapper(Dataset):
    """
    Wrapper for OpenImage dataset to unify interface.

    Args:
        annotation_csv: Path to annotation CSV file
        label_csv: Path to label mapping CSV file
        image_dir: Directory containing images
        mask_dir: Directory containing masks
    """

    def __init__(
        self,
        annotation_csv: str,
        label_csv: str,
        image_dir: str,
        mask_dir: str
    ):
        self.dataset = SegmentationDataset(
            annotation_csv=annotation_csv,
            label_csv=label_csv,
            image_dir=image_dir,
            mask_dir=mask_dir,
            transform=None  # We'll handle transforms in training loop
        )
        print(f"✓ OpenImage dataset initialized: {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
            - dataset_type: 'openimage'
            - image1: Image tensor [C, H, W]
            - mask1: Mask tensor [H, W]
            - image2: Same as image1
            - mask2: Same as mask1
            - caption: None (no ground truth caption, but we have label_name)
            - label_name: Object label name (str)
        """
        try:
            image, mask, label_name = self.dataset[idx]

            # Ensure tensors are in correct format
            if mask.dim() == 3:  # [C, H, W]
                if mask.shape[0] == 1:
                    mask = mask.squeeze(0)

            return {
                'dataset_type': 'openimage',
                'image1': image,
                'mask1': mask,
                'image2': image,  # Same image for single-frame dataset
                'mask2': mask,    # Same mask
                'caption': None,  # No GT caption (only label_name)
                'label_name': label_name,  # Keep label for potential use
            }
        except Exception as e:
            print(f"Error loading OpenImage sample {idx}: {e}")
            # Return a dummy sample
            return {
                'dataset_type': 'openimage',
                'image1': torch.zeros(3, 224, 224),
                'mask1': torch.zeros(224, 224),
                'image2': torch.zeros(3, 224, 224),
                'mask2': torch.zeros(224, 224),
                'caption': None,
                'label_name': 'Unknown',
            }
