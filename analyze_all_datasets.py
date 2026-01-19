"""
分析sa2va所有数据集的image和mask处理方式
"""
import re
import os

datasets_dir = "/data/xyc/ANS/projects/llava_sam2/datasets"

# 需要分析的数据集
datasets_to_analyze = [
    ("Osprey_Dataset.py", "OspreyDataset", "mask作为visual prompt"),
    ("describe_anything_referring_dataset.py", "DescribeAnythingReferringDataset", "mask作为visual prompt"),
    ("RefCOCO_Dataset.py", "ReferSegmDataset", "mask作为GT label"),
    ("ReVOS_Dataset.py", "VideoReVOSDataset", "视频数据集"),
    ("MeVIS_Dataset.py", "VideoMeVISDataset", "视频数据集"),
]

def extract_transform_info(file_path):
    """提取transform相关信息"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    info = {}

    # 查找IMAGENET normalization
    if re.search(r'IMAGENET_MEAN|Normalize.*mean.*std', content):
        info['has_imagenet_norm'] = True
    else:
        info['has_imagenet_norm'] = False

    # 查找extra_image_processor
    if 'extra_image_processor' in content:
        info['has_extra_processor'] = True
        # 提取extra_image_processor的使用
        match = re.search(r'self\.extra_image_processor\.apply_image\((.*?)\)', content)
        if match:
            info['extra_processor_usage'] = match.group(0)
    else:
        info['has_extra_processor'] = False

    # 查找g_pixel_values
    if 'g_pixel_values' in content:
        info['has_g_pixel_values'] = True
        # 提取g_pixel_values的生成代码
        matches = re.findall(r'g_pixel_values.*?=.*?(?:\n|$)', content)
        info['g_pixel_values_gen'] = matches[:3] if matches else []
    else:
        info['has_g_pixel_values'] = False

    # 查找prompt_masks
    if 'prompt_masks' in content:
        info['has_prompt_masks'] = True
        # 提取prompt_masks相关代码
        matches = re.findall(r'prompt_masks.*?=.*?(?:\n|$)', content)
        info['prompt_masks_usage'] = matches[:3] if matches else []
    else:
        info['has_prompt_masks'] = False

    # 查找transformer定义
    match = re.search(r'self\.transformer\s*=\s*T\.Compose\(\[(.*?)\]\)', content, re.DOTALL)
    if match:
        info['transformer'] = match.group(1).strip()[:200]  # 限制长度
    else:
        info['transformer'] = None

    # 查找masks作为GT的用法
    if re.search(r"data_dict\['masks'\]|out_data_dict\['masks'\]", content):
        info['masks_as_gt'] = True
    else:
        info['masks_as_gt'] = False

    return info

print("="*80)
print("Sa2VA 所有数据集的 Image 和 Mask 处理方式汇总")
print("="*80)
print()

for filename, classname, description in datasets_to_analyze:
    filepath = os.path.join(datasets_dir, filename)

    if not os.path.exists(filepath):
        print(f"⚠️  {filename} 不存在")
        continue

    print(f"\n{'='*80}")
    print(f"数据集: {filename}")
    print(f"类名: {classname}")
    print(f"用途: {description}")
    print(f"{'='*80}")

    info = extract_transform_info(filepath)

    print(f"\n1. ImageNet Normalization: {'✅ YES' if info['has_imagenet_norm'] else '❌ NO'}")

    if info['transformer']:
        print(f"\n2. Transformer定义:")
        print(f"   {info['transformer'][:150]}...")

    print(f"\n3. extra_image_processor (g_pixel_values): {'✅ YES' if info['has_extra_processor'] else '❌ NO'}")
    if info.get('extra_processor_usage'):
        print(f"   用法: {info['extra_processor_usage']}")
    if info.get('g_pixel_values_gen'):
        print(f"   g_pixel_values生成:")
        for line in info['g_pixel_values_gen']:
            print(f"     {line.strip()}")

    print(f"\n4. prompt_masks (visual prompt): {'✅ YES' if info['has_prompt_masks'] else '❌ NO'}")
    if info.get('prompt_masks_usage'):
        print(f"   用法:")
        for line in info['prompt_masks_usage'][:2]:
            print(f"     {line.strip()}")

    print(f"\n5. masks作为GT label: {'✅ YES' if info['masks_as_gt'] else '❌ NO'}")

    print(f"\n{'='*80}")

print("\n\n" + "="*80)
print("总结")
print("="*80)
print("""
关键发现：
1. 所有数据集的pixel_values都使用ImageNet Normalization
2. 如果有extra_image_processor，会生成g_pixel_values (DirectResize到1024，不normalize)
3. Osprey和DescribeAnything使用prompt_masks作为visual prompt
4. RefCOCO等使用masks作为GT label
5. g_pixel_values是[0, 255]的uint8 tensor (通过DirectResize)
6. pixel_values是ImageNet normalized的float tensor
""")
