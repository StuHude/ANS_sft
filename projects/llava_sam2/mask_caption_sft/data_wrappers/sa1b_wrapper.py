"""
SA-1B Dataset Wrapper for Mask Captioning SFT Training

SA-1B dataset returns: (image, mask)
Training flow:
1. Input (image, mask) → model generates caption
2. Input (image, caption) → model segments → mask'
3. Compute IoU(mask', mask) as loss
"""

import sys
import torch
from torch.utils.data import Dataset
sys.path.insert(0, '/data/xyc/mhx/SA1b/OpenDataLab___SA-1B/dataset')
from SA1t import SA1BDataset


class SA1BDatasetWrapper(Dataset):
    """
    Wrapper for SA-1B dataset to unify interface.

    Args:
        dataset_dir: Root directory of SA-1B dataset
        annotation_dir: Annotation directory name (default: 'js')
        image_dir: Image directory name (default: 'img')
        min_object: Minimum object area (default: 0)
        target_size: Target image size (H, W) (default: (1024, 1024))
        max_samples: Maximum number of samples to load (default: None)
    """

    def __init__(
        self,
        dataset_dir: str,
        annotation_dir: str = 'js',
        image_dir: str = 'img',
        min_object: int = 0,
        target_size: tuple = (1024, 1024),
        max_samples: int = None
    ):
        self.dataset = SA1BDataset(
            dataset_dir=dataset_dir,
            ids=None,
            annotation_dir=annotation_dir,
            image_dir=image_dir,
            min_object=min_object,
            target_size=target_size,
            transform=None,  # We'll handle transforms in training loop
            max_samples=max_samples
        )
        print(f"✓ SA-1B dataset initialized: {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
            - dataset_type: 'sa1b'
            - image1: Image tensor [C, H, W]
            - mask1: Mask tensor [H, W]
            - image2: Same as image1
            - mask2: Same as mask1
            - caption: None (no ground truth caption)
        """
        try:
            image, mask, class_id = self.dataset[idx]

            # Ensure tensors are in correct format
            if mask.dim() == 3:  # [C, H, W]
                if mask.shape[0] == 1:
                    mask = mask.squeeze(0)

            return {
                'dataset_type': 'sa1b',
                'image1': image,
                'mask1': mask,
                'image2': image,  # Same image for single-frame dataset
                'mask2': mask,    # Same mask
                'caption': None,  # No GT caption
                'class_id': class_id,  # Keep class_id for potential use
            }
        except Exception as e:
            print(f"Error loading SA-1B sample {idx}: {e}")
            # Return a dummy sample
            return {
                'dataset_type': 'sa1b',
                'image1': torch.zeros(3, 1024, 1024),
                'mask1': torch.zeros(1024, 1024),
                'image2': torch.zeros(3, 1024, 1024),
                'mask2': torch.zeros(1024, 1024),
                'caption': None,
                'class_id': torch.tensor(0, dtype=torch.long),
            }
