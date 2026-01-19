import os
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import random

class MaskletNPZDataset(Dataset):
    """
    用于加载预处理好的NPZ文件的数据集类，每个NPZ文件包含两帧的图像和掩码数据
    """
    def __init__(
        self,
        npz_dir: str,
        prefix: str = "masklet_data",
        transform=None,
        shuffle: bool = False
    ):
        self.npz_dir = npz_dir
        self.prefix = prefix
        self.transform = transform

        # 获取所有符合条件的NPZ文件路径
        pattern = os.path.join(npz_dir, "**", f"{prefix}_*.npz")
        self.npz_files = glob.glob(pattern, recursive=True)

        if not self.npz_files:
            raise ValueError(f"在目录 {npz_dir} 中未找到前缀为 {prefix} 的NPZ文件")

        # 可选择是否打乱文件顺序
        if shuffle:
            random.shuffle(self.npz_files)

        print(f"找到 {len(self.npz_files)} 个NPZ文件")

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        try:
            # 加载NPZ文件
            npz_path = self.npz_files[idx]
            data = np.load(npz_path)

            # 获取数据并转换为张量
            frame1 = torch.from_numpy(data['frame1']).float()
            mask1 = torch.from_numpy(data['mask1']).float()
            frame2 = torch.from_numpy(data['frame2']).float()
            mask2 = torch.from_numpy(data['mask2']).float()

            # 应用数据增强（如果有）
            if self.transform:
                frame1, mask1 = self.transform(frame1, mask1)
                frame2, mask2 = self.transform(frame2, mask2)

            return (frame1, mask1), (frame2, mask2)

        except Exception as e:
            print(f"处理索引 {idx} (文件: {self.npz_files[idx]}) 时出错: {e}")
            return None
