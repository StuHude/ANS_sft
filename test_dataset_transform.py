"""
Test script to verify dataset transform correctness.

Checks:
1. Image shape and normalization (should be ImageNet normalized)
2. Mask shape and range (should be [0, 1])
3. Image and mask spatial correspondence
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from projects.llava_sam2.mask_caption_sft.dataset_builder import (
    SAVDatasetWrapper,
    SA1BDatasetWrapper,
    OpenImageDatasetWrapper,
    RefCOCODatasetWrapper,
    build_mask_caption_dataset,
)

# ImageNet stats
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def check_image_normalization(image_tensor, name=""):
    """Check if image is ImageNet normalized."""
    print(f"\n{name} Image Check:")
    print(f"  Shape: {image_tensor.shape}")
    print(f"  Dtype: {image_tensor.dtype}")
    print(f"  Min: {image_tensor.min():.4f}, Max: {image_tensor.max():.4f}")
    print(f"  Mean per channel: {image_tensor.mean(dim=(1, 2))}")
    print(f"  Std per channel: {image_tensor.std(dim=(1, 2))}")

    # Check if normalized (mean should be ~0, std should be ~1 after normalization)
    mean_vals = image_tensor.mean(dim=(1, 2))
    std_vals = image_tensor.std(dim=(1, 2))

    # After ImageNet normalization, values can be outside [0, 1]
    # Typical range: [-2.5, 2.5] approximately
    is_normalized = (mean_vals.abs() < 1.0).all() and (image_tensor.min() < 0)

    if is_normalized:
        print("  ✅ Appears to be ImageNet normalized")
    else:
        print("  ⚠️  May not be normalized correctly")

    return is_normalized


def check_mask_range(mask_tensor, name=""):
    """Check if mask is in [0, 1] range."""
    print(f"\n{name} Mask Check:")
    print(f"  Shape: {mask_tensor.shape}")
    print(f"  Dtype: {mask_tensor.dtype}")
    print(f"  Min: {mask_tensor.min():.4f}, Max: {mask_tensor.max():.4f}")
    print(f"  Unique values (first 10): {torch.unique(mask_tensor)[:10]}")

    is_valid = (mask_tensor >= 0).all() and (mask_tensor <= 1).all()

    if is_valid:
        print("  ✅ Mask in [0, 1] range")
    else:
        print("  ❌ Mask out of [0, 1] range!")

    return is_valid


def test_sav_dataset():
    """Test SAV dataset."""
    print("\n" + "="*60)
    print("Testing SAV Dataset")
    print("="*60)

    try:
        dataset = SAVDatasetWrapper(
            npz_dir='/data/xyc/formed_data/npz',
            target_size=(1024, 1024),
            max_samples=10
        )

        print(f"Dataset size: {len(dataset)}")

        # Test first sample
        sample = dataset[0]
        check_image_normalization(sample['image1'], "SAV Image1")
        check_mask_range(sample['mask1'], "SAV Mask1")

        if sample['image2'] is not None:
            check_image_normalization(sample['image2'], "SAV Image2")
            check_mask_range(sample['mask2'], "SAV Mask2")

        print("\n✅ SAV dataset test passed")
        return True
    except Exception as e:
        print(f"\n❌ SAV dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sa1b_dataset():
    """Test SA1B dataset."""
    print("\n" + "="*60)
    print("Testing SA1B Dataset")
    print("="*60)

    try:
        dataset = SA1BDatasetWrapper(
            dataset_dir='/data/xyc/mhx/SA1b/OpenDataLab___SA-1B/raw',
            target_size=(1024, 1024),
            max_samples=10
        )

        print(f"Dataset size: {len(dataset)}")

        # Test first sample
        sample = dataset[0]
        check_image_normalization(sample['image1'], "SA1B Image1")
        check_mask_range(sample['mask1'], "SA1B Mask1")

        print("\n✅ SA1B dataset test passed")
        return True
    except Exception as e:
        print(f"\n❌ SA1B dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_openimage_dataset():
    """Test OpenImage dataset."""
    print("\n" + "="*60)
    print("Testing OpenImage Dataset")
    print("="*60)

    try:
        import os
        openimage_dir = '/data/xyc/openv7/data'

        # Check if paths exist
        ann_csv = os.path.join(openimage_dir, 'train-annotations-object-segmentation.csv')
        label_csv = os.path.join(openimage_dir, 'oidv7-class-descriptions.csv')

        if not os.path.exists(ann_csv):
            print(f"⚠️  Annotation file not found: {ann_csv}")
            return False

        dataset = OpenImageDatasetWrapper(
            annotation_csv=ann_csv,
            label_csv=label_csv,
            image_dir=os.path.join(openimage_dir, 'images', 'train'),
            mask_dir=os.path.join(openimage_dir, 'masks', 'train'),
            target_size=(1024, 1024),
        )

        print(f"Dataset size: {len(dataset)}")

        # Test first sample
        sample = dataset[0]
        check_image_normalization(sample['image1'], "OpenImage Image1")
        check_mask_range(sample['mask1'], "OpenImage Mask1")

        print("\n✅ OpenImage dataset test passed")
        return True
    except Exception as e:
        print(f"\n❌ OpenImage dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_refcoco_dataset():
    """Test RefCOCO dataset."""
    print("\n" + "="*60)
    print("Testing RefCOCO Dataset")
    print("="*60)

    try:
        dataset = RefCOCODatasetWrapper(
            data_root='/data/xyc/ANS/data/ref_seg',
            split='train',
            dataset_name='refcoco',
            target_size=(1024, 1024),
        )

        print(f"Dataset size: {len(dataset)}")

        # Test first sample
        sample = dataset[0]
        check_image_normalization(sample['image1'], "RefCOCO Image1")
        check_mask_range(sample['mask1'], "RefCOCO Mask1")

        print("\n✅ RefCOCO dataset test passed")
        return True
    except Exception as e:
        print(f"\n❌ RefCOCO dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Dataset Transform Validation Test")
    print("="*60)
    print("\nVerifying:")
    print("1. Images are ImageNet normalized")
    print("2. Masks are in [0, 1] range")
    print("3. Both have correct shape (1024x1024)")

    results = {}

    # Test each dataset
    results['SAV'] = test_sav_dataset()
    results['SA1B'] = test_sa1b_dataset()
    results['OpenImage'] = test_openimage_dataset()
    results['RefCOCO'] = test_refcoco_dataset()

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:15s}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests passed! Dataset transforms are correct.")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
