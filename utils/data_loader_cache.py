## places365 dataloader with cache
import os
import torch
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image
import numpy as np
from copy import deepcopy

class Places365CacheDataset(Dataset):
    def __init__(self, data_dir, transform=None, cache_size=[224, 224], cache_path='./places365_cache', cache_boost=False):
        self.data_dir = data_dir
        self.transform = transform
        self.cache_size = cache_size
        self.cache_path = cache_path
        self.cache_boost = cache_boost
        
        # 创建基础数据集以获取文件路径和标签
        self.base_dataset = datasets.ImageFolder(data_dir)
        self.classes = self.base_dataset.classes
        self.class_to_idx = self.base_dataset.class_to_idx
        
        # 构建数据信息
        self.dataset = {
            "image_paths": [x[0] for x in self.base_dataset.imgs],
            "labels": [x[1] for x in self.base_dataset.imgs],
            "classes": self.classes,
            "class_to_idx": self.class_to_idx
        }
        
        # 管理缓存
        self.dataset = self.manage_cache()
        
        # 如果启用缓存加速，加载所有数据到内存
        if self.cache_boost:
            self.load_all_cache_to_memory()

    def manage_cache(self):
        # 创建缓存目录
        cache_folder = os.path.join(self.cache_path, f"{self.cache_size[0]}x{self.cache_size[1]}")
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        
        if not os.path.exists(cache_folder):
            return self.create_cache(cache_folder)
        return self.load_cache_info(cache_folder)
    
    def create_cache(self, cache_folder):
        print(f"Creating cache at {cache_folder}")
        os.makedirs(cache_folder, exist_ok=True)
        
        cached_dataset = deepcopy(self.dataset)
        cached_dataset["cache_paths"] = []
        
        # 预处理并缓存所有图像
        for i, (img_path, label) in enumerate(tqdm(zip(self.dataset["image_paths"], self.dataset["labels"]), 
                                                   total=len(self.dataset["image_paths"]))):
            # 读取并预处理图像
            image = Image.open(img_path).convert('RGB')
            image_tensor = self.preprocess_image(image)
            
            # 生成缓存文件路径
            rel_path = os.path.relpath(img_path, self.data_dir)
            cache_file_name = rel_path.replace(os.sep, '_').replace('.', '_') + '.pt'
            cache_file_path = os.path.join(cache_folder, cache_file_name)
            
            # 保存预处理后的张量（使用uint8类型减小文件大小）
            torch.save({'image': image_tensor, 'label': label}, cache_file_path)
            
            cached_dataset["cache_paths"].append(cache_file_path)
        
        # 保存缓存信息
        cache_info_path = os.path.join(cache_folder, 'cache_info.json')
        with open(cache_info_path, 'w') as f:
            json.dump({
                "image_paths": cached_dataset["image_paths"],
                "labels": cached_dataset["labels"],
                "cache_paths": cached_dataset["cache_paths"],
                "classes": cached_dataset["classes"],
                "class_to_idx": cached_dataset["class_to_idx"]
            }, f)
        
        print(f"Cache created with {len(cached_dataset['cache_paths'])} images")
        return cached_dataset
    
    def load_cache_info(self, cache_folder):
        cache_info_path = os.path.join(cache_folder, 'cache_info.json')
        if not os.path.exists(cache_info_path):
            return self.create_cache(cache_folder)
        
        with open(cache_info_path, 'r') as f:
            cache_info = json.load(f)
        
        print(f"Loaded cache info with {len(cache_info['cache_paths'])} images")
        return cache_info
    
    def preprocess_image(self, image):
        # 调整图像大小
        if hasattr(transforms, 'Resize'):
            resize_transform = transforms.Resize((self.cache_size[0], self.cache_size[1]))
        else:
            # 兼容旧版本torchvision
            resize_transform = transforms.RandomSizedCrop((self.cache_size[0], self.cache_size[1]))
        image = resize_transform(image)
        
        # 转换为张量
        tensor_transform = transforms.ToTensor()
        image_tensor = tensor_transform(image)
        # 转换为uint8降低文件大小
        image_tensor = (image_tensor * 255).clamp(0, 255).to(torch.uint8)
        
        return image_tensor
    
    def load_all_cache_to_memory(self):
        print("Loading all cached data to memory...")
        self.cached_data = []
        for cache_path in tqdm(self.dataset["cache_paths"]):
            data = torch.load(cache_path)
            self.cached_data.append(data)
        print(f"Loaded {len(self.cached_data)} items to memory")
    
    def __len__(self):
        return len(self.dataset["image_paths"])
    
    def __getitem__(self, idx):
        if self.cache_boost and hasattr(self, 'cached_data'):
            # 从内存中直接获取数据
            data = self.cached_data[idx]
            image_tensor = data['image']
            label = data['label']
        else:
            # 从磁盘加载缓存文件
            cache_path = self.dataset["cache_paths"][idx]
            data = torch.load(cache_path)
            image_tensor = data['image']
            label = data['label']
        
        # 将 uint8 转换回 float32 并归一化到 [0,1] 范围
        image_tensor = image_tensor.float() / 255.0

        # 应用额外的变换（如归一化、数据增强等）
        if self.transform:
            # 将张量转换回PIL图像以应用变换
            image_pil = transforms.ToPILImage()(image_tensor)
            image_tensor = self.transform(image_pil)
        
        return image_tensor, label

def get_places365_dataloaders_cache(data_path, batch_size=32, workers=4, cache_size=[224, 224], cache_boost=False):
    """
    创建带缓存的Places365数据加载器
    
    Args:
        data_path: 数据集根目录，应包含train和val子目录
        batch_size: 批次大小
        workers: 数据加载工作进程数
        cache_size: 缓存图像的尺寸 [height, width]
        cache_boost: 是否将所有数据加载到内存中
    
    Returns:
        train_loader, val_loader: 训练和验证数据加载器
    """
    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # 创建带缓存的训练数据集
    train_dataset = Places365CacheDataset(
        data_dir=traindir,
        transform=transforms.Compose([
            transforms.RandomResizedCrop(cache_size[0], scale=(0.8, 1.0)),
            # transforms.RandomCrop(cache_size[0], padding=32),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),  # 添加随机旋转
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 添加颜色抖动
            transforms.ToTensor(),
            normalize,
        ]),
        cache_size=cache_size,
        # cache_path=os.path.join(data_path, 'cache', 'train'),
        cache_path=os.path.abspath(os.path.join(data_path, 'cache', 'train')),
        cache_boost=cache_boost
    )
    
    # 创建带缓存的验证数据集
    val_dataset = Places365CacheDataset(
        valdir,
        transform=transforms.Compose([
            transforms.Resize(int(cache_size[0] * 1.2)),
            transforms.CenterCrop(cache_size[0]),
            transforms.ToTensor(),
            normalize,
        ]),
        cache_size=cache_size,
        # cache_path=os.path.join(data_path, 'cache', 'val'),
        cache_path=os.path.abspath(os.path.join(data_path, 'cache', 'val')),
        cache_boost=cache_boost
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size, 
        shuffle=True,
        num_workers=workers, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size, 
        shuffle=False,
        num_workers=workers, 
        pin_memory=True
    )
    
    return train_loader, val_loader


def get_places365_dataloaders_normal(data_path, batch_size=32, workers=4, cache_size=[224, 224]):
    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
        transforms.RandomResizedCrop(cache_size[0], scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize]))
    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(int(cache_size[0] * 1.2)),
        transforms.CenterCrop(cache_size[0]),
        transforms.ToTensor(),
        normalize]))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=torch.cuda.is_available(), drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=torch.cuda.is_available())

    return train_loader, val_loader


# 使用示例
if __name__ == "__main__":
    # 示例用法

    # # 一般正常加载
    # train_loader, val_loader = get_places365_dataloaders_normal(
    #     data_path='../Datasets/places365standard_easyformat/places365_standard',
    #     batch_size=32,
    #     workers=6,
    #     cache_size=[224, 224]
    # )

    # 加载cache来加速I/O
    train_loader, val_loader = get_places365_dataloaders_cache(
        data_path='../Datasets/places365standard_easyformat/places365_standard',
        batch_size=32,
        workers=6,
        cache_size=[224, 224],
        cache_boost=False  # 设置为True以获得最大速度提升
    )

    # 打印train_loader长度
    print(f"Train dataset length: {len(train_loader.dataset)}")
    print(f"Val dataset length: {len(val_loader.dataset)}")

    for i, (inputs, labels) in enumerate(train_loader):
        print(f"Batch {i}")
        # inputs = inputs.to('cuda')
        # labels = labels.to('cuda')
        # print(f"Batch {i}: {inputs.shape}")
        # print(f"Batch {i}: {labels.shape}")
    print('done.')