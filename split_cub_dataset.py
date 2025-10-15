import os
import shutil
import random
from pathlib import Path

def split_cub_dataset(source_dir, target_dir, train_ratio=0.8):
    """
    划分CUB_200_2011数据集
    
    Args:
        source_dir: CUB_200_2011数据集根目录
        target_dir: 目标保存目录
        train_ratio: 训练集比例，默认0.8
    """
    # 创建目标目录
    train_dir = os.path.join(target_dir, 'train')
    val_dir = os.path.join(target_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # 读取images.txt文件，获取图片路径信息
    with open(os.path.join(source_dir, 'images.txt')) as f:
        image_paths = [line.strip().split()[1] for line in f.readlines()]
    
    # 读取train_test_split.txt文件，获取原始划分信息
    with open(os.path.join(source_dir, 'train_test_split.txt')) as f:
        split_labels = [int(line.strip().split()[1]) for line in f.readlines()]
    
    # 创建图片ID到类别的映射
    with open(os.path.join(source_dir, 'image_class_labels.txt')) as f:
        class_labels = [int(line.strip().split()[1]) for line in f.readlines()]
    
    # 读取类别名称
    with open(os.path.join(source_dir, 'classes.txt')) as f:
        class_names = [line.strip().split()[1] for line in f.readlines()]
    
    # 确保每个类别都有训练和验证样本
    class_images = {}
    for img_path, class_id in zip(image_paths, class_labels):
        class_name = class_names[class_id - 1]  # 类别ID从1开始
        if class_name not in class_images:
            class_images[class_name] = []
        class_images[class_name].append(img_path)
    
    # 对每个类别进行划分
    for class_name, images in class_images.items():
        # 随机打乱
        random.shuffle(images)
        
        # 计算分割点
        split_idx = int(len(images) * train_ratio)
        
        # 创建类别目录
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        
        # 复制训练集图片
        for img_path in images[:split_idx]:
            src = os.path.join(source_dir, 'images', img_path)
            dst = os.path.join(train_dir, class_name, os.path.basename(img_path))
            shutil.copy2(src, dst)
        
        # 复制验证集图片
        for img_path in images[split_idx:]:
            src = os.path.join(source_dir, 'images', img_path)
            dst = os.path.join(val_dir, class_name, os.path.basename(img_path))
            shutil.copy2(src, dst)
        
        print(f"处理类别 {class_name}: 训练集 {split_idx} 张, 验证集 {len(images) - split_idx} 张")

if __name__ == '__main__':
    # 设置随机种子以保证可重复性
    random.seed(42)
    
    # 设置路径
    source_dir = r'D:\CS\MyProjects\resources\Datasets\CUB_200_2011'  # CUB_200_2011数据集根目录
    target_dir = r'D:\CS\MyProjects\resources\Datasets\CUB_200_2011\CUB_split'     # 目标保存目录
    
    # 执行数据集划分
    split_cub_dataset(source_dir, target_dir)
    
    print("\n数据集划分完成！")
    print(f"训练集保存在: {os.path.join(target_dir, 'train')}")
    print(f"验证集保存在: {os.path.join(target_dir, 'val')}")
