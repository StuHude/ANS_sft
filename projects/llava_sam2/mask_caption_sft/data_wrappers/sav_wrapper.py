"""
SAV Dataset Wrapper for Mask Captioning SFT Training

SAV dataset returns: (image1, mask1) + (image2, mask2)
Training flow:
1. Input (image1, mask1) → model generates caption
2. Input (image2, caption) → model segments → mask2'
3. Compute IoU(mask2', mask2) as loss
"""

import sys
import torch
from torch.utils.data import Dataset
sys.path.insert(0, '/data/xyc/stage3/AlphaCLIP_mhx/sav_dataset')
from npz_dataloader import MaskletNPZDataset


class SAVDatasetWrapper(Dataset):
    """
    Wrapper for SAV dataset to unify interface.

    Args:
        npz_dir: Directory containing SAV NPZ files
        prefix: Prefix for NPZ files (default: "masklet_data")
        shuffle: Whether to shuffle files
    """

    def __init__(self, npz_dir: str, prefix: str = "masklet_data", shuffle: bool = False):
        self.dataset = MaskletNPZDataset(
            npz_dir=npz_dir,
            prefix=prefix,
            transform=None,  # We'll handle transforms in training loop
            shuffle=shuffle
        )
        print(f"✓ SAV dataset initialized: {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
            - dataset_type: 'sav'
            - image1: First frame image tensor [C, H, W]
            - mask1: First frame mask tensor [H, W]
            - image2: Second frame image tensor [C, H, W]
            - mask2: Second frame mask tensor [H, W]
            - caption: None (no ground truth caption for SAV)
        """
        try:
            (frame1, mask1), (frame2, mask2) = self.dataset[idx]

            # Convert frames from HWC to CHW format if needed
            # NPZ files store images as (H, W, C), PyTorch expects (C, H, W)
            if frame1.dim() == 3 and frame1.shape[-1] in [1, 3, 4]:
                # Last dimension is channels, permute to (C, H, W)
                frame1 = frame1.permute(2, 0, 1)
            if frame2.dim() == 3 and frame2.shape[-1] in [1, 3, 4]:
                frame2 = frame2.permute(2, 0, 1)

            # Ensure tensors are in correct format
            if frame1.dim() == 4:  # [B, C, H, W]
                frame1 = frame1.squeeze(0)
            if frame2.dim() == 4:
                frame2 = frame2.squeeze(0)
            if mask1.dim() == 3:  # [B, H, W] or [C, H, W]
                if mask1.shape[0] == 1:
                    mask1 = mask1.squeeze(0)
            if mask2.dim() == 3:
                if mask2.shape[0] == 1:
                    mask2 = mask2.squeeze(0)

            return {
                'dataset_type': 'sav',
                'image1': frame1,
                'mask1': mask1,
                'image2': frame2,
                'mask2': mask2,
                'caption': None,  # No GT caption
            }
        except Exception as e:
            print(f"Error loading SAV sample {idx}: {e}")
            # Return a dummy sample
            return {
                'dataset_type': 'sav',
                'image1': torch.zeros(3, 224, 224),
                'mask1': torch.zeros(224, 224),
                'image2': torch.zeros(3, 224, 224),
                'mask2': torch.zeros(224, 224),
                'caption': None,
            }
