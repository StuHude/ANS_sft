"""
RefCOCO Dataset Wrapper for Mask Captioning SFT Training

RefCOCO dataset returns: (image, mask, caption)
Training flow:
1. SKIP mask captioning (already has ground truth caption)
2. Input (image, caption) → model segments → mask'
3. Compute IoU(mask', mask) + classification loss

NOTE: RefCOCO dataset needs to be downloaded first!
Expected structure:
./data/ref_seg/refcoco/
  ├── coco2014/train2014/  (images)
  ├── instances.json       (COCO annotations)
  └── refs(unc).p          (referring expressions)

Download instructions:
1. Download COCO 2014 train images
2. Download RefCOCO annotations from:
   https://github.com/lichengunc/refer
"""

import sys
import torch
from torch.utils.data import Dataset

# Add paths for project dependencies
sys.path.insert(0, '/data/xyc/ANS')
# Import directly from the module to avoid circular import through __init__.py
from projects.llava_sam2.datasets.RefCOCO_Dataset import ReferSegmDataset


class RefCOCODatasetWrapper(Dataset):
    """
    Wrapper for RefCOCO dataset to unify interface.

    Args:
        data_root: Root directory for RefCOCO dataset
        tokenizer: Tokenizer dict config
        special_tokens: List of special tokens
        extra_image_processor: Extra image processor config
        prompt_template: Prompt template
        split: Which RefCOCO split ('refcoco', 'refcoco+', 'refcocog')
        max_length: Maximum sequence length
    """

    def __init__(
        self,
        data_root: str,
        tokenizer: dict,
        special_tokens: list,
        extra_image_processor: dict,
        prompt_template: str,
        split: str = 'refcoco',
        max_length: int = 8192
    ):
        # Determine split-specific parameters
        if split == 'refcoco':
            split_file = 'refs(unc).p'
        elif split == 'refcoco+':
            split_file = 'refs(unc).p'
        elif split == 'refcocog':
            split_file = 'refs(umd).p'
        else:
            raise ValueError(f"Unknown RefCOCO split: {split}")

        self.dataset = ReferSegmDataset(
            data_root=data_root,
            tokenizer=tokenizer,
            special_tokens=special_tokens,
            extra_image_processor=extra_image_processor,
            data_prefix=dict(img_path='coco2014/train2014/'),
            ann_file='instances.json',
            split_file=split_file,
            prompt_template=prompt_template,
            num_classes_per_sample=5,
            max_length=max_length,
        )
        print(f"✓ RefCOCO ({split}) dataset initialized: {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
            - dataset_type: 'refcoco'
            - data_dict: Original data from ReferSegmDataset (preprocessed)
                         Contains input_ids, pixel_values, labels, etc.
        """
        try:
            # ReferSegmDataset returns a preprocessed data_dict
            # with tokenized text, image tensors, masks, etc.
            data_dict = self.dataset[idx]

            return {
                'dataset_type': 'refcoco',
                'data_dict': data_dict,  # Already preprocessed by ReferSegmDataset
            }
        except Exception as e:
            print(f"Error loading RefCOCO sample {idx}: {e}")
            # Return a dummy sample
            return {
                'dataset_type': 'refcoco',
                'data_dict': None,
            }
