"""
Mask Captioning + Referring Segmentation SFT Training

This module implements a new SFT training loop that:
1. Takes image + mask as input
2. Generates caption using the main model
3. Uses EMA model to generate mask from the caption
4. Computes IoU loss between predicted and ground truth masks

Supports 4 datasets:
- SAV: Frame-pair (image1, mask1) -> (image2, mask2)
- SA-1B: Single image + mask
- OpenImage: Single image + mask
- RefCOCO: image + mask + caption (only for referring segmentation part)
"""

from .data_wrappers import *
from .trainer import *
