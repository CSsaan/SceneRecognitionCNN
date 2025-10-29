import os
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import argparse
from PIL import Image
from transformers import AutoModel
import cv2
import numpy as np

from models import DinoV3Linear
from models import ResNetLinear
from models import FastVitLinear
from models.fastvit import fastvit_t8, fastvit_t12, fastvit_s12, fastvit_sa12, fastvit_sa24, fastvit_sa36, fastvit_ma36


def load_model(arch, num_classes, resume, device):
    """
    加载模型
    
    Args:
        arch (str): 模型架构
        num_classes (int): 类别数量
        resume (str): 模型权重路径
        device (str): 设备类型
    
    Returns:
        model: 加载好的模型
    """
    print(f"[Creating model '{arch}']")
    model = None
    if arch == 'dinov3':
        MODEL_NAME_OR_PATH = "./checkpoints/weights/dinov3-vits16-pretrain-lvd1689m"
        backbone = AutoModel.from_pretrained(MODEL_NAME_OR_PATH)
        model = DinoV3Linear(backbone, num_classes)
    elif arch.lower().startswith('resnet') or arch.lower().startswith('vgg'):
        backbone = models.__dict__[arch](num_classes=365)
        model = backbone # ResNetLinear(backbone, num_classes)  # freeze backbone
    elif arch.lower().startswith('fastvit'):
        # 根据架构名称选择合适的FastViT变体
        fastvit_models = {
            'fastvit_t8': fastvit_t8,
            'fastvit_t12': fastvit_t12,
            'fastvit_s12': fastvit_s12,
            'fastvit_sa12': fastvit_sa12,
            'fastvit_sa24': fastvit_sa24,
            'fastvit_sa36': fastvit_sa36,
            'fastvit_ma36': fastvit_ma36,
        }
        constructor = fastvit_models.get(arch.lower())
        if constructor is None:
            raise ValueError(f"Unsupported FastViT architecture '{arch}'")
        backbone = constructor()
        model = FastVitLinear(backbone, num_classes)
    else:
        raise ValueError(f"Unsupported architecture '{arch}'")

    # 将模型移动到指定设备
    model = model.to(device)

    # load the model weights
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume, map_location=device)  # 添加map_location参数
        # state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint.items()}  # checkpoint['state_dict']
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()} if 'state_dict' in checkpoint else {str.replace(k, 'module.', ''): v for k, v in checkpoint.items()}
        model.load_state_dict(state_dict, strict=False)
        model.eval()
    else:
        raise FileNotFoundError(f"No checkpoint found at '{resume}'")
    
    return model


def load_classes(categories_file_name):
    """
    加载类别标签
    
    Args:
        categories_file_name (str): 类别文件路径
    
    Returns:
        tuple: 类别元组
    """
    # load the class labels
    if not os.access(categories_file_name, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        os.system('wget ' + synset_url + ' -P ./docs/')
    classes = list()
    with open(categories_file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)
    return classes


def predict_image(model, img_path, classes, device, topk=5):
    """
    对图像进行预测
    
    Args:
        model: 模型
        img_path (str): 图像路径
        classes (tuple): 类别元组
        device (str): 设备类型
        topk (int): 返回前k个预测结果
    
    Returns:
        list: 预测结果列表
    """
    # load the image transformer
    centre_crop = trn.Compose([
        trn.Resize((224, 224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # load the test image
    img = Image.open(img_path)
    input_img = V(centre_crop(img).unsqueeze(0)).to(device)  # 将输入移动到指定设备

    # forward pass
    with torch.no_grad():  # 推理时禁用梯度计算以节省内存和提高速度
        logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    
    results = []
    for i in range(0, min(topk, len(classes))):
        results.append({
            'probability': probs[i].item(),
            'class_name': classes[idx[i]],
            'index': idx[i].item()
        })
    
    return results


def predict_video(model, video_path, classes, device, output_path, topk=3):
    """
    对视频进行预测并在每帧上显示分类结果
    
    Args:
        model: 模型
        video_path (str): 视频路径
        classes (tuple): 类别元组
        device (str): 设备类型
        output_path (str): 输出视频路径
        topk (int): 显示前k个预测结果
    """
    # load the image transformer
    centre_crop = trn.Compose([
        trn.Resize((224, 224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    print(f"Processing video: {total_frames} frames total")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 转换帧为PIL图像
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # 预处理图像
        input_img = V(centre_crop(img).unsqueeze(0)).to(device)
        
        # 模型推理
        with torch.no_grad():
            logit = model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        
        # 在帧上绘制结果
        display_frame = frame.copy()
        
        # 绘制顶部预测结果
        for i in range(min(topk, len(classes))):
            prob = probs[i].item()
            class_name = classes[idx[i]]
            
            # 构造显示文本
            text = f"{class_name}: {prob:.2f}"
            
            # 设置文本位置和样式
            y_pos = 30 + i * 30
            cv2.putText(display_frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 写入处理后的帧
        out.write(display_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:  # 每30帧打印一次进度
            print(f"Processed {frame_count}/{total_frames} frames")
    
    # 释放资源
    cap.release()
    out.release()
    print(f"Video processing completed. Output saved to {output_path}")


def benchmark_model(model, img_path, device, times=100):
    """
    对模型进行性能测试
    
    Args:
        model: 模型
        img_path (str): 图像路径
        device (str): 设备类型
        times (int): 测试次数
    
    Returns:
        float: 平均延迟时间(ms)
    """
    # load the image transformer
    centre_crop = trn.Compose([
        trn.Resize((224, 224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # load the test image
    img = Image.open(img_path)
    input_img = V(centre_crop(img).unsqueeze(0)).to(device)  # 将输入移动到指定设备
    
    # spend time
    import time
    total = 0
    with torch.no_grad():  # 推理时禁用梯度计算
        for _ in range(times):
            start = time.time()
            _ = model.forward(input_img)
            end = time.time()
            total += (end - start) * 1000
    return total / times


def main():
    parser = argparse.ArgumentParser(description='Scene Recognition Inference')
    parser.add_argument('--arch', type=str, default='dinov3', 
                        help='模型架构: dinov3, resnet18, swin, fastvit_xxx 等')
    parser.add_argument('--num_classes', type=int, default=56,
                        help='类别数量')
    parser.add_argument('--img_name', type=str, default='./test/input/snow.jpg',
                        help='测试图像路径')
    parser.add_argument('--video_name', type=str, default='./test/input/video_VLOG02_Newyork.mp4',
                        help='测试视频路径')
    parser.add_argument('--resume', type=str, default='./checkpoints/saved/weights_only_checkpoints/dinov3_Epoch60_83.911_pure.pth',
                        help='模型权重路径')
    parser.add_argument('--categories_file_name', type=str, default='./docs/categories_places56.txt',
                        help='类别标签文件路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='设备类型: cuda 或 cpu')
    parser.add_argument('--benchmark', type=bool, default=True,
                        help='是否进行性能测试')
    parser.add_argument('--benchmark_times', type=int, default=100,
                        help='性能测试次数')
    args = parser.parse_args()
    
    # 创建模型
    model = load_model(args.arch, args.num_classes, args.resume, args.device)
    
    # 加载类别
    classes = load_classes(args.categories_file_name)
    
    # 进行图像预测
    results = predict_image(model, args.img_name, classes, args.device)
    print('{} prediction on {} using {}'.format(args.arch, args.img_name, args.device))
    # 输出预测结果
    for i, result in enumerate(results):
        print('{:.3f} -> {}, idx: {}'.format(result['probability'], result['class_name'], result['index']))
    # 性能测试
    if args.benchmark:
        latency = benchmark_model(model, args.img_name, args.device, args.benchmark_times)
        print(f"Latency: {latency:.0f}ms")

    # 如果提供了视频路径，则处理视频
    if args.video_name and os.path.isfile(args.video_name):
        output_video_path = './test/output_' + os.path.basename(args.video_name)
        predict_video(model, args.video_name, classes, args.device, output_video_path)

if __name__ == '__main__':
    main()
