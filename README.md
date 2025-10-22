# Scene Recognition CNN

基于卷积神经网络的场景识别系统，用于自然场景环境分类任务。该系统支持多种深度学习模型架构，包括ResNet系列、DINOv3和Swin Transformer，并提供了高效的训练和推理流程。

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
    - [Swin Transformer](#swin-transformer)
  - [训练模型](#训练模型)
    - [训练选项](#训练选项)
  - [推理预测](#推理预测)
    - [PyTorch模型推理](#pytorch模型推理)
    - [ONNX模型推理](#onnx模型推理)
  - [模型导出](#模型导出)
    - [ONNX导出](#onnx导出)
    - [模型量化](#模型量化)
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

- 多种模型架构支持（ResNet系列、DINOv3、Swin Transformer）
- 高效的数据加载和缓存机制
- 混合精度训练支持
- TensorBoard可视化集成
- 模型冻结/微调功能
- 断点续训功能
- 详细的模型统计信息
- ONNX模型导出和推理支持
- 模型量化优化（动态量化、静态量化）
- 类激活映射（CAM）可视化

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
- ONNX >= 1.16.0
- ONNX Runtime >= 1.18.0

## 项目结构

``` tree
SceneRecognitionCNN/
├── configs/                    # 配置文件目录
│   ├── train_resnet18_params.yaml
│   ├── train_dinov3_params.yaml
│   └── train_swin_params.yaml
├── export/                     # 模型导出目录
│   ├── export_onnx.py          # ONNX导出脚本
│   ├── infer_onnx.py           # ONNX推理脚本
│   └── outputs/                # 导出模型目录
│       ├── *.onnx              # ONNX模型文件
├── models/                     # 模型定义目录
│   ├── dinov3_linear.py
│   ├── resnet_linear.py
│   └── swin_linear.py
├── utils/                      # 工具函数目录
│   ├── data_loader_cache.py    # 数据加载器与缓存机制
│   ├── losses.py               # 损失函数
│   ├── metrics.py              # 评估指标
│   └── model_statistics.py     # 模型统计工具
├── checkpoints/                # 模型检查点目录
├── docs/                       # 文档目录
├── runs/                       # TensorBoard日志目录
├── train.py                    # 训练主程序
├── trainDDP.py                 # 分布式训练
├── infer.py                    # 推理脚本
├── infer_unified.py            # 统一推理脚本（含可视化）
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

对于Swin Transformer模型，可以从官方仓库下载预训练权重。

## 配置文件

项目使用YAML配置文件来管理训练参数。提供了三个示例配置文件：

1. [configs/train_resnet18_params.yaml](configs/train_resnet18_params.yaml) - ResNet18配置
2. [configs/train_dinov3_params.yaml](configs/train_dinov3_params.yaml) - DINOv3配置
3. [configs/train_swin_params.yaml](configs/train_swin_params.yaml) - Swin Transformer配置

主要配置项包括：

- `num_classes`: 类别数量（默认365）
- `img_size`: 输入图像尺寸（默认224）
- `dataset_dir`: 数据集路径
- `pretrained_weights`: 预训练权重路径
- `resume`: 恢复训练的检查点路径
- `arch`: 模型架构（resnet18, dinov3, swin等）
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

### Swin Transformer

基于Swin Transformer的分类器，支持多种变体（Swin-Tiny, Swin-Small等）。使用窗口注意力机制和层级结构设计，具有良好的性能表现。

## 训练模型

使用以下命令选择配置文件进行训练，在 train.py / trainDDP.py 中指定配置文件：

```python
parser.add_argument('--cfg', type=str, default='configs/train_resnet18_params.yaml') # 使用默认配置训练ResNet18模型
parser.add_argument('--cfg', type=str, default='configs/train_dinov3_params.yaml') # 使用DINOv3配置训练
parser.add_argument('--cfg', type=str, default='configs/train_swin_params.yaml') # 使用Swin Transformer配置训练
```

### 训练选项

1. **迁移学习**：通过设置`use_freeze_backbone: True`冻结骨干网络，只训练分类头
2. **混合精度训练**：通过设置`use_amp: True`启用混合精度训练，提高训练速度
3. **断点续训**：通过设置`resume`参数指定检查点路径恢复训练
4. **评估模式**：通过设置`evaluate: True`只进行验证，不进行训练(此时会加载val数据集，要修改成需要的数据-直接改文件夹名称即可)

## 推理预测

### PyTorch模型推理

使用[infer.py](infer.py)进行单张图片预测：

```bash
python infer.py
```

使用[infer_unified.py](infer_unified.py)进行带热力图可视化的预测（包含类激活映射）：

```bash
python infer_unified.py
```

### ONNX模型推理

使用[export/infer_onnx.py](export/infer_onnx.py)进行ONNX模型推理：

```bash
python export/infer_onnx.py --model export/outputs/resnet18.onnx --source path/to/image.jpg
```

支持多种ONNX模型格式：

- 原始FP32模型
- 动态量化模型
- 静态量化模型

## 模型导出

### ONNX导出

使用[export/export_onnx.py](export/export_onnx.py)脚本将PyTorch模型导出为ONNX格式：

```bash
python export/export_onnx.py
```

该脚本会自动执行以下操作：

1. 加载训练好的PyTorch模型
2. 导出为ONNX格式
3. 验证导出的ONNX模型
4. 对模型进行量化优化

导出的模型将保存在[export/outputs/](export/outputs/)目录中。

### 模型量化

导出过程中会自动生成两种量化版本的ONNX模型：

1. **动态量化模型**：对权重进行量化，推理时动态计算激活值
2. **静态量化模型**：对权重和激活值都进行量化，需要校准数据

量化可以显著减小模型大小并提高推理速度，通常对精度影响很小。

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

[infer_unified.py](infer_unified.py)还支持类激活映射（CAM）可视化，可以生成热力图显示模型关注的图像区域。

## 性能指标

模型在Places365数据集上的Top-1准确率约为69.29%（ResNet18）。

## 许可证

本项目仅供学习研究使用，请遵守相关数据集和模型的许可证协议。
