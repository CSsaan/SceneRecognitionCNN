# Scene Recognition CNN

基于卷积神经网络的场景识别系统，用于自然场景环境分类任务。该系统支持多种深度学习模型架构，包括ResNet系列和DINOv3，并提供了高效的训练和推理流程。

## 目录

- [Scene Recognition CNN](#scene-recognition-cnn)
  - [目录](#目录)
  - [项目简介](#项目简介)
  - [主要特性](#主要特性)
  - [技术栈](#技术栈)
  - [项目结构](#项目结构)
  - [安装指南](#安装指南)
    - [使用UV（推荐）](#使用uv推荐)
    - [使用PIP](#使用pip)
  - [使用说明](#使用说明)
    - [准备数据集](#准备数据集)
    - [下载预训练权重](#下载预训练权重)
  - [配置文件](#配置文件)
  - [模型架构](#模型架构)
    - [ResNet Linear](#resnet-linear)
    - [DINOv3 Linear](#dinov3-linear)
  - [训练模型](#训练模型)
    - [训练选项](#训练选项)
  - [推理预测](#推理预测)
  - [性能优化](#性能优化)
    - [数据加载优化](#数据加载优化)
    - [混合精度训练](#混合精度训练)
    - [模型冻结](#模型冻结)
  - [可视化](#可视化)
  - [性能指标](#性能指标)
  - [许可证](#许可证)

## 项目简介

Scene Recognition CNN是一个专门用于自然场景识别的深度学习项目。它基于Places365数据集的标准分类任务，支持365个不同的场景类别识别。该项目实现了多种现代CNN架构，并提供了完整的训练、验证和推理流程。

## 主要特性

- 多种模型架构支持（ResNet系列、DINOv3）
- 高效的数据加载和缓存机制
- 混合精度训练支持
- TensorBoard可视化集成
- 模型冻结/微调功能
- 断点续训功能
- 详细的模型统计信息

## 技术栈

- Python >= 3.12
- PyTorch >= 2.8.0
- TorchVision >= 0.23.0
- Transformers >= 4.57.1
- OpenCV-Python >= 4.12.0.88
- TensorBoard >= 2.20.0
- TorchInfo >= 1.8.0
- THOP >= 0.1.1
- PyYAML >= 6.0.3
- TQDM >= 4.67.1

## 项目结构

``` tree
SceneRecognitionCNN/
├── configs/                    # 配置文件目录
│   ├── train_resnet18_params.yaml
│   └── train_dinov3_params.yaml
├── models/                     # 模型定义目录
│   ├── dinov3_linear.py
│   └── resnet_linear.py
├── utils/                      # 工具函数目录
│   ├── data_loader_cache.py    # 数据加载器与缓存机制
│   ├── losses.py               # 损失函数
│   ├── metrics.py              # 评估指标
│   └── model_statistics.py     # 模型统计工具
├── checkpoints/                # 模型检查点目录
├── runs/                       # TensorBoard日志目录
├── trainer.py                  # 训练主程序
├── infer.py                    # 推理脚本
├── pyproject.toml              # 项目依赖配置
└── README.md                   # 项目说明文档
```

## 安装指南

### 使用UV（推荐）

```bash
# 克隆项目
git clone https://github.com/CSsaan/SceneRecognitionCNN
cd SceneRecognitionCNN

# 安装依赖
uv sync
```

### 使用PIP

```bash
# 克隆项目
git clone <repository-url>
cd SceneRecognitionCNN

# 安装依赖
pip install -e .
```

## 使用说明

### 准备数据集

该项目使用Places365标准数据集，数据集应按以下结构组织：

``` tree
places365_standard/
├── train/
│   ├── airfield/
│   ├── airplane_cabin/
│   └── ...
└── val/
    ├── airfield/
    ├── airplane_cabin/
    └── ...
```

### 下载预训练权重

对于ResNet模型，可以从Places365项目下载预训练权重：

```bash
wget http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar -P checkpoints/weights/
```

对于DINOv3模型，可以从 Hugging Face 下载：[https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m/tree/main](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m/tree/main)
并放置于`checkpoints/weights/`目录下。

## 配置文件

项目使用YAML配置文件来管理训练参数。提供了两个示例配置文件：

1. [configs/train_resnet18_params.yaml](configs/train_resnet18_params.yaml) - ResNet18配置
2. [configs/train_dinov3_params.yaml](configs/train_dinov3_params.yaml) - DINOv3配置

主要配置项包括：

- `num_classes`: 类别数量（默认365）
- `img_size`: 输入图像尺寸（默认224）
- `dataset_dir`: 数据集路径
- `pretrained_weights`: 预训练权重路径
- `resume`: 恢复训练的检查点路径
- `arch`: 模型架构（resnet18, dinov3等）
- `epochs`: 训练轮数
- `lr`: 学习率
- `batch_size`: 批次大小
- `use_freeze_backbone`: 是否冻结骨干网络
- `use_amp`: 是否使用混合精度训练

## 模型架构

### ResNet Linear

基于ResNet的线性分类器，在ResNet基础上添加全连接层进行分类。支持预训练权重加载和骨干网络冻结功能。

### DINOv3 Linear

基于DINOv3视觉Transformer的线性分类器，使用Transformer的[CLS]标记输出进行分类。

## 训练模型

使用以下命令选择配置文件进行训练，在trainer.py中指定配置文件：

```python
parser.add_argument('--cfg', type=str, default='configs/train_resnet18_params.yaml') # 使用默认配置训练ResNet18模型
parser.add_argument('--cfg', type=str, default='configs/train_dinov3_params.yaml') # 使用DINOv3配置训练
```

### 训练选项

1. **迁移学习**：通过设置`use_freeze_backbone: True`冻结骨干网络，只训练分类头
2. **混合精度训练**：通过设置`use_amp: True`启用混合精度训练，提高训练速度
3. **断点续训**：通过设置`resume`参数指定检查点路径恢复训练
4. **评估模式**：通过设置`evaluate: True`只进行验证，不进行训练

## 推理预测

使用[infer.py](infer.py)进行单张图片预测：

## 性能优化

### 数据加载优化

项目实现了两种数据加载方式：

1. **普通数据加载**：使用标准PyTorch ImageFolder
2. **缓存数据加载**：预处理并缓存图像，大幅提高训练速度

通过设置`use_dataLoaderCache: True`启用缓存数据加载。

### 混合精度训练

通过设置`use_amp: True`启用混合精度训练，可以显著提高训练速度并减少显存占用。

### 模型冻结

通过设置`use_freeze_backbone: True`冻结骨干网络，仅训练分类头，适用于迁移学习场景。

## 可视化

项目集成了TensorBoard用于训练过程可视化：

```bash
tensorboard --logdir runs
```

TensorBoard中包含以下信息：

- 训练/验证损失曲线
- 训练/验证准确率曲线
- 学习率变化曲线
- 模型结构图

## 性能指标

模型在Places365数据集上的Top-1准确率约为69.29%（ResNet18）。

## 许可证

本项目仅供学习研究使用，请遵守相关数据集和模型的许可证协议。
