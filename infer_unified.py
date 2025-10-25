import os
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import numpy as np
import cv2
import argparse
from PIL import Image
from transformers import AutoModel

from models import DinoV3Linear
from models import ResNetLinear

# 全局变量用于存储特征
features_blobs = []


def hook_feature(module, input, output):
    """特征提取钩子函数"""
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))


def returnResnetCAM(feature_conv, weight_softmax, class_idx):
    """生成类激活映射"""
    size_upsample = (256, 256)
    nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


def returnDinoV3CAM(feature_conv):
    """生成类激活映射"""
    size_upsample = (256, 256)
    # features_blobs[0]应该包含last_hidden_state
    last_hidden = torch.from_numpy(feature_conv)
    last_hidden = last_hidden.unsqueeze(0)
    # 提取CLS token和patch tokens
    cls_token = last_hidden[:, 0, :]  # [B, 384]
    # cls_token = torch.ones_like(cls_token) # cls_token 值全为1 (即可查看原始patchs的热力图)
    patch_tokens = last_hidden[:, 5:, :]  # [B, 196, 384]
    # 3. 计算相似度：CLS 与每个 patch 的余弦相似度
    sims = torch.cosine_similarity(cls_token.unsqueeze(0), patch_tokens, dim=2)  # [B, 196]
    sims = sims.cpu().numpy().reshape(14, 14)  # 14x14 网格
    sims = (sims - sims.min()) / (sims.max() - sims.min())  # 归一化
    sims = np.uint8(255 * sims)
    return [cv2.resize(sims, size_upsample)]


def load_labels(file_name_category, file_name_IO, file_name_attribute):
    """加载所有标签"""
    # 场景类别相关
    if not os.access(file_name_category, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        os.system('wget ' + synset_url + ' -P ./docs/')
    classes = list()
    with open(file_name_category) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    # 室内/室外相关
    if not os.access(file_name_IO, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt'
        os.system('wget ' + synset_url + ' -P ./docs/')
    with open(file_name_IO) as f:
        lines = f.readlines()
        labels_IO = []
        for line in lines:
            items = line.rstrip().split()
            labels_IO.append(int(items[-1]) - 1)  # 0是室内，1是室外
    labels_IO = np.array(labels_IO)

    # 场景属性相关
    if not os.access(file_name_attribute, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/labels_sunattribute.txt'
        os.system('wget ' + synset_url + ' -P ./docs/')
    with open(file_name_attribute) as f:
        lines = f.readlines()
        labels_attribute = [item.rstrip() for item in lines]
    file_name_W = './docs/W_sceneattribute_wideresnet18.npy'
    if not os.access(file_name_W, os.W_OK):
        synset_url = 'http://places2.csail.mit.edu/models_places365/W_sceneattribute_wideresnet18.npy'
        os.system('wget ' + synset_url + ' -P ./docs/')
    W_attribute = np.load(file_name_W)

    return classes, labels_IO, labels_attribute, W_attribute


def load_model(arch, num_classes, resume, device):
    """加载模型"""
    global features_blobs
    features_blobs = []

    print(f"[Creating model '{arch}']")
    model = None
    if arch.lower().startswith('dinov3'):
        MODEL_NAME_OR_PATH = "./checkpoints/weights/dinov3-vits16-pretrain-lvd1689m"
        backbone = AutoModel.from_pretrained(MODEL_NAME_OR_PATH)
        model = DinoV3Linear(backbone, num_classes)
    elif arch.lower().startswith('resnet') or arch.lower().startswith('vgg'):
        backbone = models.__dict__[arch](num_classes=365)
        model = ResNetLinear(backbone, num_classes)
    else:
        raise ValueError(f"Unsupported architecture '{arch}'")

    # 将模型移动到指定设备
    model = model.to(device)

    # 加载模型权重
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume, map_location=device)  # 添加map_location参数
        # state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint.items()}  # checkpoint['state_dict']
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()} if 'state_dict' in checkpoint else {str.replace(k, 'module.', ''): v for k, v in checkpoint.items()}
        model.load_state_dict(state_dict, strict=False)
        model.eval()
    else:
        raise FileNotFoundError(f"No checkpoint found at '{resume}'")

    # 为每个指定层注册前向传播钩子
    if arch.lower().startswith('dinov3'):
        model.backbone.layer[-1].register_forward_hook(hook_feature)
    elif arch.lower().startswith('resnet') or arch.lower().startswith('vgg'):
        model.backbone.layer4.register_forward_hook(hook_feature)
        model.backbone.avgpool.register_forward_hook(hook_feature)
    else:
        raise ValueError(f"Unsupported architecture '{arch}'")

    return model


def predict_and_visualize(model, img_path, classes, labels_IO, labels_attribute, W_attribute, 
                         arch, device, topk=5, save_cam=True, cam_output_path='./test/cam-heatmap.jpg'):
    """
    对图像进行预测并可视化
    
    Args:
        model: 模型
        img_path (str): 图像路径
        classes (tuple): 类别标签
        labels_IO (np.array): 室内/室外标签
        labels_attribute (list): 属性标签
        W_attribute (np.array): 属性权重矩阵
        arch (str): 模型架构
        device (str): 设备类型
        topk (int): 返回前k个预测结果
        save_cam (bool): 是否保存CAM热力图
        cam_output_path (str): CAM热力图保存路径
    
    Returns:
        dict: 包含预测结果的字典
    """
    global features_blobs
    features_blobs = []  # 重置特征 blobs
    
    # 获取softmax权重
    params = list(model.parameters())
    weight_softmax = params[-2].data.cpu().numpy()  # 移动到CPU进行numpy操作
    weight_softmax[weight_softmax < 0] = 0

    # 加载图像变换
    tf = trn.Compose([
        trn.Resize((224, 224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载测试图像
    img = Image.open(img_path)
    input_img = V(tf(img).unsqueeze(0)).to(device)  # 将输入移动到指定设备

    # 前向传播
    with torch.no_grad():  # 推理时禁用梯度计算
        logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.cpu().numpy()  # 移动到CPU进行numpy操作
    idx = idx.cpu().numpy()

    # 输出室内/室外预测
    io_image = np.mean(labels_IO[idx[:10]])  # 根据前10个投票决定室内或室外
    environment_type = "indoor" if io_image < 0.5 else "outdoor"

    # 输出场景类别预测
    scene_categories = []
    for i in range(0, min(topk, len(classes))):
        scene_categories.append({
            'probability': probs[i],
            'class_name': classes[idx[i]],
            'index': idx[i]
        })

    # 初始化返回结果
    result = {
        'environment_type': environment_type,
        'scene_categories': scene_categories,
        'scene_attributes': [],
        'cam_saved': False
    }

    # 输出场景属性和生成类激活映射
    if len(features_blobs) > 0:
        if arch.lower().startswith('resnet') or arch.lower().startswith('vgg'):
            # 获取属性响应
            responses_attribute = W_attribute.dot(features_blobs[1])
            idx_a = np.argsort(responses_attribute)
            attributes = [labels_attribute[idx_a[i]] for i in range(-1, -10, -1)]
            result['scene_attributes'] = attributes

        # 生成类激活映射
        if save_cam:
            try:
                if arch.lower().startswith('dinov3'):
                    CAMs = returnDinoV3CAM(features_blobs[0])  # 获取热力图
                elif arch.lower().startswith('resnet') or arch.lower().startswith('vgg'):
                    CAMs = returnResnetCAM(features_blobs[0], weight_softmax, [idx[0]])  # 获取热力图
                else:
                    raise ValueError(f"Unsupported architecture '{arch}'")

                sims = CAMs[0]
                # 渲染CAM并输出
                img_cv2 = cv2.imread(img_path)
                if img_cv2 is not None:
                    height, width, _ = img_cv2.shape
                    heatmap = cv2.applyColorMap(cv2.resize(sims, (width, height)), cv2.COLORMAP_JET)
                    result_img = heatmap * 0.4 + img_cv2 * 0.5
                    cv2.imwrite(cam_output_path, result_img)
                    result['cam_saved'] = True
                    result['cam_path'] = cam_output_path
            except Exception as e:
                print(f"Error generating CAM: {e}")

    return result

def predict_video_with_cam(model, video_path, classes, labels_IO, arch, device, topk=3, cam_output_path='./test/cam-video.mp4'):
    """
    对视频进行预测并在每帧上显示分类结果和热力图
    
    Args:
        model: 模型
        video_path (str): 视频路径
        classes (tuple): 类别标签
        labels_IO (np.array): 室内/室外标签
        labels_attribute (list): 属性标签
        W_attribute (np.array): 属性权重矩阵
        arch (str): 模型架构
        device (str): 设备类型
        output_path (str): 输出视频路径
        topk (int): 返回前k个预测结果
        cam_output_path (str): CAM热力图视频保存路径
    """
    global features_blobs
    
    # 获取softmax权重 (用于ResNet/VGG的CAM)
    params = list(model.parameters())
    weight_softmax = None
    if arch.lower().startswith('resnet') or arch.lower().startswith('vgg'):
        weight_softmax = params[-2].data.cpu().numpy()
        weight_softmax[weight_softmax < 0] = 0

    # 加载图像变换
    tf = trn.Compose([
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
    out = cv2.VideoWriter(cam_output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    print(f"Processing video: {total_frames} frames total")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 重置特征 blobs
        features_blobs = []
        
        # 转换帧为PIL图像
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # 预处理图像
        input_img = V(tf(img_pil).unsqueeze(0)).to(device)
        
        # 前向传播
        with torch.no_grad():
            logit = model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        probs = probs.cpu().numpy()
        idx = idx.cpu().numpy()
        
        # 输出室内/室外预测
        io_image = np.mean(labels_IO[idx[:10]])
        environment_type = "indoor" if io_image < 0.5 else "outdoor"
        
        # 输出场景类别预测
        scene_categories = []
        for i in range(0, min(topk, len(classes))):
            scene_categories.append({
                'probability': probs[i],
                'class_name': classes[idx[i]],
                'index': idx[i]
            })
        
        # 复制原始帧用于显示
        display_frame = frame.copy()
        
        # 绘制顶部预测结果
        # 参考 #selectedCode 中的显示方式
        for i in range(min(topk, len(classes))):
            prob = probs[i]
            class_name = classes[idx[i]]
            
            # 构造显示文本
            text = f"{class_name}: {prob:.2f}"
            
            # 设置文本位置和样式
            y_pos = 30 + i * 30
            cv2.putText(display_frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 添加环境类型
        env_text = f"Environment: {environment_type}"
        cv2.putText(display_frame, env_text, (10, 30 + topk * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 生成并叠加CAM热力图
        if len(features_blobs) > 0 and len(scene_categories) > 0:
            try:
                if arch.lower().startswith('dinov3'):
                    CAMs = returnDinoV3CAM(features_blobs[0])
                elif arch.lower().startswith('resnet') or arch.lower().startswith('vgg'):
                    CAMs = returnResnetCAM(features_blobs[0], weight_softmax, [idx[0]])
                else:
                    raise ValueError(f"Unsupported architecture '{arch}'")

                if CAMs and len(CAMs) > 0:
                    sims = CAMs[0]
                    # 调整热力图大小以匹配原图
                    heatmap = cv2.applyColorMap(cv2.resize(sims, (width, height)), cv2.COLORMAP_JET)
                    # 将热力图叠加到原图上
                    display_frame = cv2.addWeighted(display_frame, 0.6, heatmap, 0.4, 0)
            except Exception as e:
                print(f"Error generating CAM for frame {frame_count}: {e}")
        
        # 写入处理后的帧
        out.write(display_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:  # 每30帧打印一次进度
            print(f"Processed {frame_count}/{total_frames} frames")
    
    # 释放资源
    cap.release()
    out.release()
    print(f"Video processing completed. Output saved to {cam_output_path}")

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
    # 加载图像变换
    tf = trn.Compose([
        trn.Resize((224, 224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载测试图像
    img = Image.open(img_path)
    input_img = V(tf(img).unsqueeze(0)).to(device)  # 将输入移动到指定设备

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
    parser = argparse.ArgumentParser(description='Scene Recognition Inference with Visualization')
    parser.add_argument('--arch', type=str, default='dinov3',
                        help='模型架构: dinov3, resnet18 等')
    parser.add_argument('--num_classes', type=int, default=56,
                        help='类别数量')
    parser.add_argument('--img_name', type=str, default='./test/input/snow.jpg',
                        help='测试图像路径')
    parser.add_argument('--video_name', type=str, default='./test/input/1311910610-1-192.mp4',
                        help='测试视频路径')
    parser.add_argument('--resume', type=str, default='./checkpoints/saved/weights_only_checkpoints/dinov3_Epoch60_83.911_pure.pth',
                        help='模型权重路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='设备类型: cuda 或 cpu')
    parser.add_argument('--categories_file', type=str, default='./docs/categories_places56.txt',
                        help='类别标签文件路径')
    parser.add_argument('--io_file', type=str, default='./docs/IO_places56.txt',
                        help='室内/室外标签文件路径')
    parser.add_argument('--attribute_file', type=str, default='./docs/labels_sunattribute.txt',
                        help='属性标签文件路径')
    parser.add_argument('--benchmark', type=bool, default=True,
                        help='是否进行性能测试')
    parser.add_argument('--benchmark_times', type=int, default=100,
                        help='性能测试次数')
    parser.add_argument('--save_cam', type=bool, default=True,
                        help='是否保存类激活映射图')
    parser.add_argument('--cam_output_path', type=str, default='./test/cam-heatmap.jpg',
                        help='CAM热力图保存路径')
    parser.add_argument('--topk', type=int, default=5,
                        help='显示前k个预测结果')
    args = parser.parse_args()

    # 加载标签
    classes, labels_IO, labels_attribute, W_attribute = load_labels(
        args.categories_file, args.io_file, args.attribute_file)

    # 加载模型
    model = load_model(args.arch, args.num_classes, args.resume, args.device)

    # 如果提供了视频路径，则处理视频
    if args.video_name:
        output_video_path = './test/output_' + os.path.basename(args.video_name)
        predict_video_with_cam(model, args.video_name, classes, labels_IO, args.arch, args.device, 3, output_video_path)

    # 进行预测和可视化
    result = predict_and_visualize(
        model, args.img_name, classes, labels_IO, labels_attribute, W_attribute,
        args.arch, args.device, args.topk, args.save_cam, args.cam_output_path)

    # 输出结果
    print('{} prediction on {} using {}'.format(args.arch, args.img_name, args.device))
    print('--TYPE OF ENVIRONMENT: {}'.format(result['environment_type']))
    print('--SCENE CATEGORIES:')
    for i, category in enumerate(result['scene_categories']):
        print('{:.3f} -> {}, idx: {}'.format(category['probability'], category['class_name'], category['index']))

    if result['scene_attributes']:
        print('--SCENE ATTRIBUTES:')
        print(', '.join(result['scene_attributes']))

    if result['cam_saved']:
        print('Class activation map is saved as {}'.format(result['cam_path']))

    # 性能测试
    if args.benchmark:
        latency = benchmark_model(model, args.img_name, args.device, args.benchmark_times)
        print(f"Latency: {latency:.0f}ms")


if __name__ == '__main__':
    main()