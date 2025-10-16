import torch
import torch.nn as nn
from torchinfo import summary
from thop import profile, clever_format
import numpy as np

def count_parameters(model):
    """
    统计模型参数数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    params = {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': non_trainable_params
    }
    print(f"Total Parameters:     {params['total']:,} ({params['total']/1e6:.2f}M)")
    print(f"Trainable Parameters: {params['trainable']:,} ({params['trainable']/1e6:.2f}M)")
    print(f"Non-trainable Parameters: {params['non_trainable']:,} ({params['non_trainable']/1e6:.2f}M)")
    
    return params

def layer_wise_parameters(model):
    """
    按层统计参数数量
    """
    params_dict = {}
    total_params = 0
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0 and hasattr(module, 'weight'):
            params = sum(p.numel() for p in module.parameters())
            params_dict[name] = params
            total_params += params
            
    return params_dict, total_params

def calculate_flops(model, input_shape):
    """
    计算模型的FLOPs和参数数量
    """
    input_tensor = torch.randn(input_shape).to("cuda" if torch.cuda.is_available() else "cpu")
    macs, params = profile(model, inputs=(input_tensor, ), verbose=False)
    flops = macs * 2  # MACs to FLOPs
    return flops, params

def model_memory_usage(model, input_shape, batch_size=1):
    """
    估算模型内存使用量
    """
    # 参数内存 (bytes)
    param_count = sum(p.numel() for p in model.parameters())
    param_memory = param_count * 4  # float32 = 4 bytes
    
    # 梯度内存
    gradient_memory = param_memory
    
    # 缓冲区内存
    buffer_memory = sum(b.numel() for b in model.buffers()) * 4
    
    # 前向传播激活值内存估算
    dummy_input = torch.randn(input_shape)
    dummy_input = dummy_input.to(next(model.parameters()).device if next(model.parameters()).is_cuda else 'cpu')
    
    activation_memory = 0
    activations = []
    
    def hook_fn(module, input, output):
        if isinstance(input, tuple):
            for inp in input:
                if isinstance(inp, torch.Tensor):
                    activations.append(inp)
        if isinstance(output, torch.Tensor):
            activations.append(output)
    
    hooks = []
    for module in model.modules():
        hooks.append(module.register_forward_hook(hook_fn))
    
    try:
        with torch.no_grad():
            model(dummy_input)
            activation_memory = sum(act.numel() * 4 for act in activations)
    except:
        activation_memory = 0  # 如果无法计算则设为0
    
    for hook in hooks:
        hook.remove()
    
    total_memory = param_memory + gradient_memory + buffer_memory + activation_memory
    
    return {
        'parameters': param_memory,
        'gradients': gradient_memory,
        'buffers': buffer_memory,
        'activations': activation_memory,
        'total': total_memory
    }

def detailed_model_summary(model, input_shape):
    """
    生成详细的模型统计信息
    """    
    print("=" * 80)
    print("MODEL STATISTICS SUMMARY")
    print("=" * 80, "\n")
    
    # FLOPs计算
    print("-" * 80)
    params = count_parameters(model)
    try:
        flops, _ = calculate_flops(model, input_shape)
        flops_formatted, params_formatted = clever_format([flops, params['total']], "%.3f")
        print(f"Total FLOPs: {flops_formatted}")
        print(f"Total Params: {params_formatted}")
    except Exception as e:
        print(f"Could not calculate FLOPs: {str(e)}")
    print("-" * 80, "\n")
    
    # 按层统计
    print("-" * 80)
    layer_params, total_layer_params = layer_wise_parameters(model)
    print("Layer-wise Parameter Distribution:")
    sorted_layers = sorted(layer_params.items(), key=lambda x: x[1], reverse=True)
    for name, param_count in sorted_layers[:10]:  # 显示前10层
        if param_count > 0:
            percentage = (param_count / params['total']) * 100
            print(f"  {name:<40} {param_count:>12,} ({percentage:>6.2f}%)")
    
    if len(sorted_layers) > 10:
        remaining_params = sum(param_count for _, param_count in sorted_layers[10:])
        percentage = (remaining_params / params['total']) * 100
        print(f"  {'Other layers':<40} {remaining_params:>12,} ({percentage:>6.2f}%)")
    print("-" * 80, "\n")
    
    # 模型结构摘要
    print("-" * 80)
    try:
        print("Model Architecture Summary:")
        summary(model, input_shape, device="cuda" if torch.cuda.is_available() else "cpu")
    except Exception as e:
        print(f"Could not generate model summary: {str(e)}")
    print("-" * 80, "\n")

# 使用示例
if __name__ == "__main__":
    # 假设DINOv3_UNet_model已经定义
    # from MyUnet import DINOv3UNetMatting
    # model = DINOv3UNetMatting("./dinov3-vits16-pretrain-lvd1689m")
    
    # detailed_model_summary(model, (1, 3, 224, 224))
    pass