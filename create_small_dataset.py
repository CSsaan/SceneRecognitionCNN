import os
import shutil
from pathlib import Path
import random

subfolder_names = ['airfield', 'airplane_cabin', 'amusement_park', 'bakery-shop', 'bar', 'baseball_field', 'basketball_court-indoor', 'beach', 'beauty_salon', 'bedroom', 'botanical_garden', 'bowling_alley', 'boxing_ring', 'building_facade', 'bus_interior', 'bus_station-indoor', 'campus', 'church-outdoor', 'classroom', 'conference_room', 'desert-sand', 'desert_road', 'forest-broadleaf', 'forest_road', 'gas_station', 'gymnasium-indoor', 'harbor', 'hayfield', 'hospital_room', 'house', 'japanese_garden', 'kitchen', 'library-indoor', 'lighthouse', 'mountain', 'movie_theater-indoor', 'museum-indoor', 'natural_history_museum', 'office_building', 'office_cubicles', 'park', 'parking_garage-indoor', 'playground', 'railroad_track', 'restaurant', 'shopping_mall-indoor', 'snowfield', 'soccer_field', 'storage_room', 'subway_station-platform', 'supermarket', 'temple-asia', 'volleyball_court-outdoor', 'waterfall', 'water_park', 'wind_farm']

def create_small_dataset(source_dir, target_dir, images_per_class=200):
    """
    从大型图像分类数据集中为每个类别抽取指定数量的图像，创建小型数据集
    
    Args:
        source_dir (str): 原始数据集路径（包含各类别子文件夹）
        target_dir (str): 目标数据集路径
        images_per_class (int): 每个类别保留的图像数量，默认200
    """
    # 创建目标目录
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    
    # 遍历源目录中的每个类别文件夹
    for class_folder in os.listdir(source_dir):
        if class_folder not in subfolder_names:
            continue
        class_path = os.path.join(source_dir, class_folder)
        
        # 确保是目录而不是文件
        if os.path.isdir(class_path):
            print(f"处理类别: {class_folder}")
            
            # 创建目标类别目录
            target_class_path = os.path.join(target_dir, class_folder)
            Path(target_class_path).mkdir(parents=True, exist_ok=True)
            
            # 获取该类别下的所有图像文件
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]
            
            # 如果图像数量少于所需数量，则取全部图像
            if len(image_files) <= images_per_class:
                selected_images = image_files
                print(f"  类别 '{class_folder}' 图像不足{images_per_class}张，仅复制{len(image_files)}张")
            else:
                # 随机选择指定数量的图像
                selected_images = random.sample(image_files, images_per_class)
                print(f"  从{len(image_files)}张图像中随机选择{images_per_class}张")
            
            # 复制选中的图像到目标目录
            for image_file in selected_images:
                source_file = os.path.join(class_path, image_file)
                target_file = os.path.join(target_class_path, image_file)
                shutil.copy2(source_file, target_file)
    
    print("小型数据集创建完成！")

# 使用示例
if __name__ == "__main__":
    # 设置源目录和目标目录路径
    source_directory = "../Datasets/places365standard_easyformat/places365_standard/val"  # 原始数据集路径
    target_directory = "../Datasets/places365standard_easyformat/places365_standard/val_new_classes"  # 新的小型数据集路径
    
    # 创建小型数据集
    create_small_dataset(source_directory, target_directory, images_per_class=50)