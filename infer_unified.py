import os
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import numpy as np
import cv2
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoConfig, get_cosine_schedule_with_warmup

from models.dinov3_linear import DinoV3Linear
from models.resnet_linear import ResNetLinear


# 配置参数
num_classes = 365
arch = 'resnet18'  # 可以改为 'dinov3'
img_name = r"D:\CS\MyProjects\resources\Datasets\CUB_200_2011\val\026.Bronzed_Cowbird\Bronzed_Cowbird_0022_796221.jpg"
resume = 'checkpoints/resnet18_Epoch20_69.292_pure.pth'

file_name_category = './docs/categories_places365.txt' # 场景类相关
file_name_IO = './docs/IO_places365.txt' # 室内/室外
file_name_attribute = './docs/labels_sunattribute.txt' # 场景属性

# 全局变量用于存储特征
features_blobs = []

def hook_feature(module, input, output):
    """特征提取钩子函数"""
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))

def returnCAM(feature_conv, weight_softmax, class_idx):
    """生成类激活映射"""
    size_upsample = (256, 256)
    nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def load_labels():
    """加载所有标签"""
    # 场景类别相关
    file_name_category = './docs/categories_places365.txt'
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
            labels_IO.append(int(items[-1]) -1) # 0是室内，1是室外
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

def load_model(arch, num_classes, resume):
    """加载模型"""
    global features_blobs
    features_blobs = []
    
    print(f"[Creating model '{arch}']")
    model = None
    if arch == 'dinov3':
        MODEL_NAME_OR_PATH = "./checkpoints/weights/dinov3-vits16-pretrain-lvd1689m"
        backbone = AutoModel.from_pretrained(MODEL_NAME_OR_PATH)
        model = DinoV3Linear(backbone, num_classes)
    elif arch.lower().startswith('resnet') or arch.lower().startswith('vgg'):
        backbone = models.__dict__[arch](num_classes=365)
        model = ResNetLinear(backbone, num_classes)
    else:
        raise ValueError(f"Unsupported architecture '{arch}'")

    # 加载模型权重
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint.items()} # checkpoint['state_dict']
        model.load_state_dict(state_dict, strict=False)
        model.eval()
    else:
        raise FileNotFoundError(f"No checkpoint found at '{resume}'")
    
    print(model)
    # 指定要注册钩子的层名
    features_names = ['layer4','avgpool']  # 这是ResNet的最后一个卷积层和平均池化层
    # 为每个指定层注册前向传播钩子
    model.backbone.layer4.register_forward_hook(hook_feature)
    model.backbone.avgpool.register_forward_hook(hook_feature)
    
    return model

def main():    
    # 加载标签
    classes, labels_IO, labels_attribute, W_attribute = load_labels()
    
    # 加载模型
    model = load_model(arch, num_classes, resume)
    
    # 获取softmax权重
    params = list(model.parameters())
    weight_softmax = params[-2].data.numpy()
    weight_softmax[weight_softmax<0] = 0
    
    # 加载图像变换
    tf = trn.Compose([
        trn.Resize((224,224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 加载测试图像
    img = Image.open(img_name)
    input_img = V(tf(img).unsqueeze(0))
    
    # 前向传播
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()
    
    # 输出室内/室外预测
    io_image = np.mean(labels_IO[idx[:10]])  # 根据前10个投票决定室内或室外
    if io_image < 0.5:
        print('--TYPE OF ENVIRONMENT: indoor')
    else:
        print('--TYPE OF ENVIRONMENT: outdoor')
    
    # 输出场景类别预测
    print('--SCENE CATEGORIES:')
    for i in range(0, 5):
        print(f'{probs[i]:.3f} -> {classes[idx[i]]}, idx: {idx[i]}')
    
    # 输出场景属性
    if len(features_blobs) > 0:
        responses_attribute = W_attribute.dot(features_blobs[1])
        idx_a = np.argsort(responses_attribute)
        print('--SCENE ATTRIBUTES:')
        print(', '.join([labels_attribute[idx_a[i]] for i in range(-1,-10,-1)]))
        
        # 生成类激活映射
        print('Class activation map is saved as cam.jpg')
        CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
        
        # 渲染CAM并输出
        img_cv2 = cv2.imread(img_name)
        height, width, _ = img_cv2.shape
        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.4 + img_cv2 * 0.5
        cv2.imwrite('cam-heatmap.jpg', result)

if __name__ == '__main__':
    main()
