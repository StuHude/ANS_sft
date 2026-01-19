"""
Data Preprocessor for Mask Captioning SFT Training

Handles preprocessing for different data types:
1. Mask → Caption task: Prepare visual prompt with mask
2. Caption → Mask task: Prepare text prompt with referring expression
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms


class MaskCaptionPreprocessor:
    """
    Preprocessor for mask captioning SFT training.

    Prepares data for two-stage training:
    Stage 1: image + mask → caption (mask captioning)
    Stage 2: image + caption → mask (referring segmentation)
    """

    def __init__(
        self,
        image_size: int = 1024,
        mask_threshold: float = 0.5,
        image_processor=None,
    ):
        """
        Args:
            image_size: Target image size (default: 1024 for SAM2)
            mask_threshold: Threshold for binary masks
            image_processor: Optional image processor (e.g., InternVL processor)
        """
        self.image_size = image_size
        self.mask_threshold = mask_threshold
        self.image_processor = image_processor

        # Basic image transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def prepare_mask_to_caption(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        instruction: str = "Please describe what you see in the highlighted region.",
    ):
        """
        Prepare data for mask → caption task.

        Args:
            image: Image tensor [C, H, W] or PIL Image
            mask: Mask tensor [H, W] or [1, H, W]
            instruction: Task instruction

        Returns:
            dict with:
            - pixel_values: Processed image tensor
            - prompt_masks: List of mask tensors for visual prompting
            - vp_overall_mask: Binary mask indicating visual prompt region
            - prompt_text: Text prompt with placeholders
        """
        # Convert PIL to tensor if needed
        if isinstance(image, Image.Image):
            image = self.image_transform(image)

        # Ensure image is [C, H, W]
        if image.dim() == 4:
            image = image.squeeze(0)

        # Ensure mask is [H, W]
        if mask.dim() == 3:
            mask = mask.squeeze(0)

        # Resize mask to match image size
        if mask.shape[0] != self.image_size or mask.shape[1] != self.image_size:
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=(self.image_size, self.image_size),
                mode='nearest'
            ).squeeze()

        # Binarize mask
        mask = (mask > self.mask_threshold).float()

        # Create prompt_masks (list of masks for SAM2 visual prompting)
        prompt_masks = [mask.unsqueeze(0)]  # [1, H, W]

        # Create vp_overall_mask (indicates which regions have visual prompts)
        vp_overall_mask = (mask > 0).float().unsqueeze(0)  # [1, H, W]

        # Create prompt text with visual prompt markers
        prompt_text = f"<image>\n<vp>{instruction}</vp>"

        return {
            'pixel_values': image,
            'prompt_masks': prompt_masks,
            'vp_overall_mask': vp_overall_mask,
            'prompt_text': prompt_text,
        }

    def prepare_caption_to_mask(
        self,
        image: torch.Tensor,
        caption: str,
        instruction: str = None,
    ):
        """
        Prepare data for caption → mask task.

        Args:
            image: Image tensor [C, H, W] or PIL Image
            caption: Referring expression / caption
            instruction: Optional task instruction (default: use caption directly)

        Returns:
            dict with:
            - pixel_values: Processed image tensor
            - prompt_masks: Empty list (no visual prompt for this task)
            - vp_overall_mask: Zero tensor (no visual prompt)
            - prompt_text: Text prompt with caption
        """
        # Convert PIL to tensor if needed
        if isinstance(image, Image.Image):
            image = self.image_transform(image)

        # Ensure image is [C, H, W]
        if image.dim() == 4:
            image = image.squeeze(0)

        # No visual prompts for caption→mask task
        prompt_masks = []
        vp_overall_mask = torch.zeros(1, self.image_size, self.image_size)

        # Create prompt text
        if instruction is None:
            instruction = f"Please segment {caption} in this image. [SEG]"
        prompt_text = f"<image>\n{instruction}"

        return {
            'pixel_values': image,
            'prompt_masks': prompt_masks,
            'vp_overall_mask': vp_overall_mask,
            'prompt_text': prompt_text,
            'caption': caption,
        }

    def batch_prepare(self, samples: list, task: str):
        """
        Batch prepare multiple samples.

        Args:
            samples: List of sample dicts
            task: 'mask_to_caption' or 'caption_to_mask'

        Returns:
            Batched dict
        """
        if task == 'mask_to_caption':
            prepared = [
                self.prepare_mask_to_caption(s['image'], s['mask'])
                for s in samples
            ]
        elif task == 'caption_to_mask':
            prepared = [
                self.prepare_caption_to_mask(s['image'], s['caption'])
                for s in samples
            ]
        else:
            raise ValueError(f"Unknown task: {task}")

        # Stack tensors
        batch = {
            'pixel_values': torch.stack([p['pixel_values'] for p in prepared]),
            'prompt_masks': [p['prompt_masks'] for p in prepared],
            'vp_overall_mask': torch.stack([p['vp_overall_mask'] for p in prepared]),
            'prompt_text': [p['prompt_text'] for p in prepared],
        }

        if task == 'caption_to_mask':
            batch['caption'] = [p['caption'] for p in prepared]

        return batch
