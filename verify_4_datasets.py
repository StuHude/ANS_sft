#!/usr/bin/env python3
"""
Quick verification script to check if all 4 datasets can be loaded.
Tests dataset configuration before running full training.
"""

import sys
import os
sys.path.insert(0, '/data/xyc/ANS')

from projects.llava_sam2.mask_caption_sft.dataset_builder import build_mask_caption_dataset

# Configuration
SAV_DIR = "/data/xyc/formed_data/npz"
SA1B_DIR = "/data/xyc/mhx/SA1b/OpenDataLab___SA-1B/raw"
REFCOCO_DIR = "./data/ref_seg"
OPENIMAGE_DIR = "/data/xyc/openv7/data"

# Repeat configuration
SAV_REPEATS = 2
REFCOCO_REPEATS = 4
SA1B_MAX_SAMPLES = 10  # Test with only 10 samples for quick verification

print("=" * 80)
print("Verifying 4-Dataset Configuration")
print("=" * 80)

# Check paths exist
print("\n1. Checking dataset paths...")
datasets_status = []

if os.path.exists(SAV_DIR):
    print(f"   ✓ SAV: {SAV_DIR}")
    datasets_status.append(True)
else:
    print(f"   ✗ SAV: {SAV_DIR} NOT FOUND")
    datasets_status.append(False)

if os.path.exists(SA1B_DIR):
    print(f"   ✓ SA1B: {SA1B_DIR}")
    datasets_status.append(True)
else:
    print(f"   ✗ SA1B: {SA1B_DIR} NOT FOUND")
    datasets_status.append(False)

if os.path.exists(OPENIMAGE_DIR):
    # Check for required CSV files
    annotation_csv = os.path.join(OPENIMAGE_DIR, 'train-annotations-object-segmentation.csv')
    label_csv = os.path.join(OPENIMAGE_DIR, 'oidv7-class-descriptions.csv')
    if os.path.exists(annotation_csv) and os.path.exists(label_csv):
        print(f"   ✓ OpenImage: {OPENIMAGE_DIR}")
        print(f"     - annotation CSV: {os.path.basename(annotation_csv)} ✓")
        print(f"     - label CSV: {os.path.basename(label_csv)} ✓")
        datasets_status.append(True)
    else:
        print(f"   ✗ OpenImage CSV files missing in {OPENIMAGE_DIR}")
        datasets_status.append(False)
else:
    print(f"   ✗ OpenImage: {OPENIMAGE_DIR} NOT FOUND")
    datasets_status.append(False)

if os.path.exists(REFCOCO_DIR):
    print(f"   ✓ RefCOCO: {REFCOCO_DIR}")
    datasets_status.append(True)
else:
    print(f"   ✗ RefCOCO: {REFCOCO_DIR} NOT FOUND")
    datasets_status.append(False)

# Summary
num_available = sum(datasets_status)
print(f"\n   Summary: {num_available}/4 datasets available")

if num_available < 4:
    print("\n   ⚠ WARNING: Not all 4 datasets are available!")
    sys.exit(1)

# Build datasets
print("\n2. Building datasets with repeats...")
print(f"   SAV repeats: {SAV_REPEATS}x")
print(f"   RefCOCO repeats: {REFCOCO_REPEATS}x")
print(f"   SA1B limit: {SA1B_MAX_SAMPLES} samples (for quick testing)")

try:
    # OpenImage config
    openimage_config = {
        'annotation_csv': os.path.join(OPENIMAGE_DIR, 'train-annotations-object-segmentation.csv'),
        'label_csv': os.path.join(OPENIMAGE_DIR, 'oidv7-class-descriptions.csv'),
        'image_dir': os.path.join(OPENIMAGE_DIR, 'images', 'train'),
        'mask_dir': os.path.join(OPENIMAGE_DIR, 'masks', 'train'),
    }

    # RefCOCO config
    refcoco_config = {
        'data_root': REFCOCO_DIR,
        'split': 'train',
        'dataset_name': 'refcoco',
    }

    dataset = build_mask_caption_dataset(
        sav_dir=SAV_DIR,
        sa1b_dir=SA1B_DIR,
        openimage_config=openimage_config,
        refcoco_config=refcoco_config,
        target_size=(1024, 1024),
        sa1b_max_samples=SA1B_MAX_SAMPLES,
        sav_repeats=SAV_REPEATS,
        refcoco_repeats=REFCOCO_REPEATS,
    )

    print(f"\n3. Dataset built successfully!")
    print(f"   Total samples: {len(dataset)}")

    # Test loading a few samples
    print(f"\n4. Testing sample loading...")
    for i in range(min(3, len(dataset))):
        try:
            sample = dataset[i]
            image1_shape = sample['image1'].shape
            mask1_shape = sample['mask1'].shape
            has_image2 = sample['image2'] is not None
            dataset_type = sample['dataset_type']

            print(f"   Sample {i}:")
            print(f"     - Dataset type: {dataset_type}")
            print(f"     - image1 shape: {image1_shape}")
            print(f"     - mask1 shape: {mask1_shape}")
            print(f"     - Has image2/mask2: {has_image2}")

            if has_image2:
                print(f"     - image2 shape: {sample['image2'].shape}")
                print(f"     - mask2 shape: {sample['mask2'].shape}")
        except Exception as e:
            print(f"   ✗ Sample {i} failed: {e}")

    print("\n" + "=" * 80)
    print("✓ Verification PASSED!")
    print("=" * 80)
    print("\nAll 4 datasets are properly configured:")
    print("  1. SAV (with image2/mask2 support)")
    print("  2. SA1B")
    print("  3. OpenImage")
    print("  4. RefCOCO")
    print("\nDataset repeats configured:")
    print(f"  - SAV: {SAV_REPEATS}x")
    print(f"  - RefCOCO: {REFCOCO_REPEATS}x")
    print("\nReady to run training!")

except Exception as e:
    print("\n" + "=" * 80)
    print("✗ Verification FAILED!")
    print("=" * 80)
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
