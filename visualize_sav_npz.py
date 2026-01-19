"""
可视化SAV数据集的NPZ文件
检查数据是否正确，图像和mask是否对应
"""

import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import glob
from pathlib import Path

def visualize_sav_sample(npz_path, output_dir):
    """
    可视化一个SAV npz文件

    Args:
        npz_path: npz文件路径
        output_dir: 输出目录
    """
    # 加载npz
    data = np.load(npz_path)

    print(f"\n处理文件: {os.path.basename(npz_path)}")
    print(f"NPZ keys: {list(data.keys())}")

    # 读取数据
    frame1 = data['frame1']
    mask1 = data['mask1']
    frame2 = data['frame2']
    mask2 = data['mask2']

    print(f"Frame1 shape: {frame1.shape}, dtype: {frame1.dtype}, range: [{frame1.min()}, {frame1.max()}]")
    print(f"Mask1 shape: {mask1.shape}, dtype: {mask1.dtype}, range: [{mask1.min()}, {mask1.max()}]")
    print(f"Frame2 shape: {frame2.shape}, dtype: {frame2.dtype}, range: [{frame2.min()}, {frame2.max()}]")
    print(f"Mask2 shape: {mask2.shape}, dtype: {mask2.dtype}, range: [{mask2.min()}, {mask2.max()}]")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存原始图像
    if frame1.max() > 1.0:
        frame1_img = frame1.astype(np.uint8)
        frame2_img = frame2.astype(np.uint8)
    else:
        frame1_img = (frame1 * 255).astype(np.uint8)
        frame2_img = (frame2 * 255).astype(np.uint8)

    # 保存原始mask
    if mask1.max() > 1.0:
        mask1_img = mask1.astype(np.uint8)
        mask2_img = mask2.astype(np.uint8)
    else:
        mask1_img = (mask1 * 255).astype(np.uint8)
        mask2_img = (mask2 * 255).astype(np.uint8)

    basename = os.path.basename(npz_path).replace('.npz', '')

    # 如果是(H, W, C)格式
    if len(frame1.shape) == 3 and frame1.shape[-1] in [1, 3, 4]:
        Image.fromarray(frame1_img).save(f"{output_dir}/{basename}_frame1.png")
        Image.fromarray(frame2_img).save(f"{output_dir}/{basename}_frame2.png")
    # 如果是(C, H, W)格式
    elif len(frame1.shape) == 3:
        frame1_img = frame1_img.transpose(1, 2, 0)
        frame2_img = frame2_img.transpose(1, 2, 0)
        Image.fromarray(frame1_img).save(f"{output_dir}/{basename}_frame1.png")
        Image.fromarray(frame2_img).save(f"{output_dir}/{basename}_frame2.png")

    # Mask (H, W)
    if len(mask1.shape) == 2:
        Image.fromarray(mask1_img, mode='L').save(f"{output_dir}/{basename}_mask1.png")
        Image.fromarray(mask2_img, mode='L').save(f"{output_dir}/{basename}_mask2.png")
    elif len(mask1.shape) == 3:
        # (1, H, W) or (H, W, 1)
        if mask1.shape[0] == 1:
            mask1_img = mask1_img[0]
            mask2_img = mask2_img[0]
        elif mask1.shape[-1] == 1:
            mask1_img = mask1_img[..., 0]
            mask2_img = mask2_img[..., 0]
        Image.fromarray(mask1_img, mode='L').save(f"{output_dir}/{basename}_mask1.png")
        Image.fromarray(mask2_img, mode='L').save(f"{output_dir}/{basename}_mask2.png")

    # 创建组合可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Frame1
    if len(frame1.shape) == 3:
        if frame1.shape[0] in [1, 3]:  # (C, H, W)
            frame1_vis = frame1.transpose(1, 2, 0) if frame1.max() <= 1.0 else (frame1 / 255.0).transpose(1, 2, 0)
        else:  # (H, W, C)
            frame1_vis = frame1 if frame1.max() <= 1.0 else frame1 / 255.0

        # 如果是单通道，转为RGB
        if frame1_vis.shape[-1] == 1:
            frame1_vis = np.repeat(frame1_vis, 3, axis=-1)

        axes[0, 0].imshow(frame1_vis)
    else:
        axes[0, 0].imshow(frame1, cmap='gray')
    axes[0, 0].set_title('Frame 1')
    axes[0, 0].axis('off')

    # Mask1
    if len(mask1.shape) == 2:
        mask1_vis = mask1
    elif mask1.shape[0] == 1:
        mask1_vis = mask1[0]
    else:
        mask1_vis = mask1[..., 0]

    axes[0, 1].imshow(mask1_vis, cmap='gray')
    axes[0, 1].set_title('Mask 1')
    axes[0, 1].axis('off')

    # Overlay1
    if len(frame1_vis.shape) == 3 and frame1_vis.shape[-1] == 3:
        overlay1 = frame1_vis.copy()
        # Mask overlay (红色)
        mask1_bool = mask1_vis > (mask1_vis.max() * 0.5)
        overlay1[mask1_bool] = overlay1[mask1_bool] * 0.5 + np.array([0.5, 0, 0])
        axes[0, 2].imshow(overlay1)
        axes[0, 2].set_title('Frame 1 + Mask 1 Overlay')
        axes[0, 2].axis('off')
    else:
        axes[0, 2].axis('off')

    # Frame2
    if len(frame2.shape) == 3:
        if frame2.shape[0] in [1, 3]:  # (C, H, W)
            frame2_vis = frame2.transpose(1, 2, 0) if frame2.max() <= 1.0 else (frame2 / 255.0).transpose(1, 2, 0)
        else:  # (H, W, C)
            frame2_vis = frame2 if frame2.max() <= 1.0 else frame2 / 255.0

        if frame2_vis.shape[-1] == 1:
            frame2_vis = np.repeat(frame2_vis, 3, axis=-1)

        axes[1, 0].imshow(frame2_vis)
    else:
        axes[1, 0].imshow(frame2, cmap='gray')
    axes[1, 0].set_title('Frame 2')
    axes[1, 0].axis('off')

    # Mask2
    if len(mask2.shape) == 2:
        mask2_vis = mask2
    elif mask2.shape[0] == 1:
        mask2_vis = mask2[0]
    else:
        mask2_vis = mask2[..., 0]

    axes[1, 1].imshow(mask2_vis, cmap='gray')
    axes[1, 1].set_title('Mask 2')
    axes[1, 1].axis('off')

    # Overlay2
    if len(frame2_vis.shape) == 3 and frame2_vis.shape[-1] == 3:
        overlay2 = frame2_vis.copy()
        mask2_bool = mask2_vis > (mask2_vis.max() * 0.5)
        overlay2[mask2_bool] = overlay2[mask2_bool] * 0.5 + np.array([0.5, 0, 0])
        axes[1, 2].imshow(overlay2)
        axes[1, 2].set_title('Frame 2 + Mask 2 Overlay')
        axes[1, 2].axis('off')
    else:
        axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{basename}_visualization.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ 可视化完成，保存到: {output_dir}/{basename}_*.png")

    return True


def main():
    # SAV数据集路径
    sav_dir = "/data/xyc/formed_data/npz"
    output_dir = "/data/xyc/ANS/sav_visualization"

    print("="*80)
    print("SAV数据集NPZ文件可视化")
    print("="*80)
    print(f"\n数据目录: {sav_dir}")
    print(f"输出目录: {output_dir}")

    # 查找所有npz文件
    npz_files = glob.glob(f"{sav_dir}/**/masklet_data_*.npz", recursive=True)

    if not npz_files:
        print(f"\n❌ 未找到npz文件！")
        return

    print(f"\n找到 {len(npz_files)} 个npz文件")

    # 可视化前5个样本
    num_samples = min(5, len(npz_files))
    print(f"\n可视化前 {num_samples} 个样本...")

    for i, npz_path in enumerate(npz_files[:num_samples]):
        print(f"\n{'='*80}")
        print(f"样本 {i+1}/{num_samples}")
        print(f"{'='*80}")

        try:
            visualize_sav_sample(npz_path, output_dir)
        except Exception as e:
            print(f"❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print(f"✅ 可视化完成！")
    print(f"所有图像保存在: {output_dir}/")
    print(f"{'='*80}")

    # 打印说明
    print(f"\n检查要点:")
    print(f"1. Frame和Mask的尺寸是否一致")
    print(f"2. Mask是否为二值 (0或255)")
    print(f"3. Overlay图中红色区域是否与Mask对应")
    print(f"4. Frame1和Frame2是否为同一场景的不同帧")
    print(f"5. Mask1和Mask2是否标注了相同的物体/区域")


if __name__ == '__main__':
    main()
