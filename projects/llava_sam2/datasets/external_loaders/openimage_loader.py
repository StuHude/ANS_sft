import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# 使用与alpha_grit.py相同的标准化参数
PIXEL_MEAN = (0.48145466, 0.4578275, 0.40821073)
PIXEL_STD = (0.26862954, 0.26130258, 0.27577711)

class SegmentationDataset(Dataset):
    def __init__(self, annotation_csv, label_csv, image_dir, mask_dir, transform=None):
        self.data = pd.read_csv(annotation_csv)
        self.label_map = self._load_label_map(label_csv)
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        # 使用标准CLIP预处理
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(PIXEL_MEAN, PIXEL_STD),
        ])

        # 掩码变换（生成全白掩码）
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(0.5, 0.26)
        ])

    def _load_label_map(self, label_csv):
        label_df = pd.read_csv(label_csv)
        return dict(zip(label_df['LabelName'], label_df['DisplayName']))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_id = row['ImageID']
        mask_path = row['MaskPath']
        label_name = row['LabelName']

        # 加载图像
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        image = Image.open(image_path).convert("RGB")
        # 转换为tensor
        image = self.transform(image)

        # 加载掩码
        full_mask_path = os.path.join(self.mask_dir, mask_path)
        mask = Image.open(full_mask_path).convert("L")

        mask = self.mask_transform(mask)

        label_display_name = self.label_map.get(label_name, "Unknown")

        return image, mask, label_display_name
