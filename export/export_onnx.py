import onnx
import torch
from PIL import Image
# from onnxsim import simplify
from torchvision import transforms
import torchvision.models as models
from transformers import AutoImageProcessor, AutoModel, AutoConfig, get_cosine_schedule_with_warmup
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, quantize_static, QuantType, CalibrationMethod, quantize_dynamic
import numpy as np

import os
import sys
import random
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models import DinoV3Linear
from models import ResNetLinear


class PyTorchToONNXConverter:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def convert_to_onnx(self, input_shape, output_path, device):
        # 创建虚拟输入
        dummy_input = torch.randn(*input_shape).to(device)
        # 导出 ONNX 模型
        torch.onnx.export(
            self.model,              # pytorch网络模型
            dummy_input,             # 随机的模拟输入
            output_path,             # 导出的onnx文件位置
            export_params=True,      # 导出训练好的模型参数
            verbose=False,           # debug message
            do_constant_folding=True,# 是否进行常数折叠
            input_names=['input'],   # 为静态网络图中的输入节点设置别名，在进行onnx推理时，将input_names字段与输入数据绑定
            output_names=['output'], # 为输出节点设置别名
            opset_version=11,
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

# 数据预处理
val_transforms = transforms.Compose(
    [
        # Resize(256, interpolation="bilinear"),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Resize((224, 224)),
        # CenterCrop(224),
        transforms.Normalize((0.49372172, 0.46933405, 0.44654398), (0.30379174, 0.29378528, 0.30067085)),
    ]
)
 
# 数据批次读取器
def batch_reader(datas, batch_size):
    _datas = []
    length = len(datas)
    for i, data in enumerate(datas):
        if batch_size==1:
            yield {'input': data}
        elif (i+1) % batch_size==0:
            _datas.append(data)
            yield {'input': _datas}
            _datas = []
        elif i<length-1:
            _datas.append(data)
        else:
            _datas.append(data)
            yield {'input': _datas}
 
# 构建校准数据读取器
'''
    实质是一个迭代器
    get_next 方法返回一个如下样式的字典
    {
        输入 1: 数据 1, 
        ...
        输入 n: 数据 n
    }
    记录了模型的各个输入和其对应的经过预处理后的数据
'''
class DataReader(CalibrationDataReader):
    def __init__(self, datas, batch_size):
        self.datas = batch_reader(datas, batch_size)
 
    def get_next(self):
        return next(self.datas, None)



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载pytorch模型
    num_classes = 365
    arch = 'resnet18'
    img_name = r"D:\CS\MyProjects\resources\Datasets\CUB_200_2011\val\026.Bronzed_Cowbird\Bronzed_Cowbird_0022_796221.jpg"
    resume = './checkpoints/resnet18_Epoch20_69.292_pure.pth'
    output_path = './export/outputs'
    os.makedirs(output_path, exist_ok=True)

    # create the model
    print(f"[Creating model '{arch}']")
    model = None
    if arch == 'dinov3':
        MODEL_NAME_OR_PATH = "./checkpoints/weights/dinov3-vits16-pretrain-lvd1689m"
        backbone = AutoModel.from_pretrained(MODEL_NAME_OR_PATH)
        model = DinoV3Linear(backbone, num_classes)
    elif arch.lower().startswith('resnet') or arch.lower().startswith('vgg'):
        backbone = models.__dict__[arch](num_classes=365)
        model = ResNetLinear(backbone, num_classes) # freze backbone
    else:
        raise ValueError(f"Unsupported architecture '{arch}'")

    # load the model weights
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint.items()} # checkpoint['state_dict']
        model.load_state_dict(state_dict, strict=False)
        model.eval().to(device)
    else:
        raise FileNotFoundError(f"No checkpoint found at '{resume}'")

    # 导出ONNX模型
    converter = PyTorchToONNXConverter(model)
    input_shape = (1, 3, 224, 224)  # 示例输入大小:(None, 3, 256, 256) --> 模型输出大小:(None, 1, 256, 256)
    converter.convert_to_onnx(input_shape, f'{output_path}/{arch}.onnx', device)
    print("ONNX 模型导出完成！")

    # 检查导出的模型
    onnx_model = onnx.load(f'{output_path}/{arch}.onnx')
    onnx.checker.check_model(onnx_model)
    # print(onnx.printer.to_text(onnx_model.graph))
    print("ONNX 模型检查通过！")

    # # simplify onnx model(Need <= Python3.11)
    # save_path = f'{output_path}/{arch}_simplified.onnx'
    # onnx_model, check = simplify(onnx_model)
    # onnx.save(onnx_model, save_path)

    # 动态量化
    model_fp32 = f'{output_path}/{arch}.onnx'
    model_quant_dynamic = f'{output_path}/{arch}_quant_dynamic.onnx'
    quantize_dynamic(
        model_input=model_fp32,           # 输入模型
        model_output=model_quant_dynamic, # 输出模型
        weight_type=QuantType.QUInt8,     # 参数类型 Int8 / UInt8
    )
    print("ONNX 动态量化完成！")

    # TODO: 静态量化
    model_quant_static = f'{output_path}/{arch}_quant_static.onnx'
    img_dir = r'D:\CS\MyProjects\resources\Datasets\CUB_200_2011\train\002.Laysan_Albatross'
    img_num = 48
    datas = [
        val_transforms(
            Image.open(os.path.join(img_dir, img)).convert('RGB')
        ) for img in os.listdir(img_dir)[:img_num]
    ]
    datas = [data.numpy() for data in datas] # 将datas列表的每个元素转换为numpy类型
    data_reader = DataReader(datas, batch_size=2) # 实例化一个校准数据读取器
    quantize_static(
        model_input=model_fp32,              # 输入模型
        model_output=model_quant_static,     # 输出模型
        calibration_data_reader=data_reader, # 校准数据读取器
        quant_format= QuantFormat.QDQ,       # 量化格式 QDQ / QOperator
        activation_type=QuantType.QInt8,     # 激活类型 Int8 / UInt8
        weight_type=QuantType.QInt8,         # 参数类型 Int8 / UInt8
        calibrate_method=CalibrationMethod.MinMax, # 数据校准方法 MinMax / Entropy / Percentile
    )
    print("ONNX 静态量化完成！")
