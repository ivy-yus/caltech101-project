import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
import os

# 1. 设置路径 (由于我们在项目根目录下运行，所以是 ./caltech-101)
DATA_DIR = './caltech-101'
IMG_SIZE = 128 
BATCH_SIZE = 32

def main():
    # 检查文件夹是否存在
    if not os.path.exists(DATA_DIR):
        print(f"❌ 错误: 找不到数据集文件夹 '{DATA_DIR}'。请确保你把解压后的文件夹放在了正确的位置！")
        return

    print("✅ 找到数据集文件夹，开始进行数据预处理和划分...")

    # 2. 定义数据转换
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform_aug = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(), 
        transforms.RandomRotation(15),     
        transforms.ToTensor(),
        normalize,
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])

    # 3. 加载数据集
    full_dataset_train = datasets.ImageFolder(DATA_DIR, transform=train_transform_aug)
    full_dataset_eval = datasets.ImageFolder(DATA_DIR, transform=val_test_transform)

    targets = full_dataset_train.targets

    # 第一次划分：70% 训练集，30% 剩余
    train_idx, temp_idx = train_test_split(
        np.arange(len(targets)), 
        test_size=0.30, 
        stratify=targets, 
        random_state=42
    )

    temp_targets = [targets[i] for i in temp_idx]

    # 第二次划分：剩下的对半分，15% 验证，15% 测试
    val_idx, test_idx = train_test_split(
        temp_idx, 
        test_size=0.50, 
        stratify=temp_targets, 
        random_state=42
    )

    # 生成 Subset
    train_dataset = Subset(full_dataset_train, train_idx)
    val_dataset = Subset(full_dataset_eval, val_idx)
    test_dataset = Subset(full_dataset_eval, test_idx)

    print("-" * 50)
    print("📊 数据集划分结果 (70% / 15% / 15%):")
    print(f"   训练集 (Train):      {len(train_dataset)} 张图片")
    print(f"   验证集 (Validation): {len(val_dataset)} 张图片")
    print(f"   测试集 (Test):       {len(test_dataset)} 张图片")
    print(f"   总计类别数:          {len(full_dataset_train.classes)} 类")
    print("-" * 50)

    # 4. 创建 DataLoaders
    # Mac M3 Pro 可以使用多个 worker 加速读取，这里设为 4
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # 5. 🚀 核心测试环节：尝试抓取一个 Batch 的数据！
    print("🚀 正在尝试从 DataLoader 中读取一个批次 (Batch) 的数据...")
    try:
        # 获取一个批次的图片和标签
        images, labels = next(iter(train_loader))
        print("✅ 成功读取一个 Batch！")
        print(f"   图片的 Tensor 形状 (Batch Size, Channels, Height, Width): {images.shape}")
        print(f"   标签的 Tensor 形状: {labels.shape}")
        print("🎉 恭喜！你的数据预处理和加载管道已经完全打通，随时可以喂给模型了！")
    except Exception as e:
        print(f"❌ 读取 Batch 失败，错误信息: {e}")

if __name__ == '__main__':
    main()