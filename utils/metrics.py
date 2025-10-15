import torch

class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def compute_accuracy(pred: torch.Tensor, target:torch.Tensor, topk: tuple = (1,)):
    maxk = max(topk)
    batch_size = target.shape[0]
    pred = pred.topk(maxk, 1)[-1]
    pred = pred.t()
    correct = pred == target.view(1, -1).expand_as(pred)
    return [correct[:k].reshape(-1).float().sum(0).item()*100. /  batch_size for k in topk]


if __name__ == '__main__':
    # 测试用例1：简单的4分类问题
    # ------------------------------------------------------------
    print("=== 测试用例1：简单4分类 ===")
    # 模拟batch_size=3的输出和标签
    # output: (batch_size, num_classes)
    output = torch.tensor([
        [0.1, 0.2, 0.3, 0.4],  # 样本1: 预测类别3(0.4最大)
        [0.5, 0.1, 0.21, 0.19],  # 样本2: 预测类别0(0.5最大)
        [0.19, 0.21, 0.5, 0.1]   # 样本3: 预测类别2(0.5最大)
    ])
    
    target = torch.tensor([3, 0, 1])  # 真实标签
    
    print("输出概率:")
    print(output)
    print("真实标签:", target)
    
    # 计算top1和top2准确率
    top1, top2 = compute_accuracy(output, target, topk=(1, 2))
    
    print(f"\n详细分析:")
    _, pred = output.topk(2, 1, True, True)
    print("预测结果 (top-2):")
    for i in range(len(target)):
        print(f"  样本{i}: 预测={pred[i].tolist()}, 真实={target[i]}")
    
    # 手动验证
    print(f"\n手动验证，期望:")
    print(f"  Top-1: 样本0(✓), 样本1(✓), 样本2(✗) => 2/3 = 66.67%")
    print(f"  Top-2: 样本0(✓), 样本1(✓), 样本2(✓) => 3/3 = 100.00%")
    
    print(f"\n函数结果:")
    print(f"  Top-1准确率: {top1:.2f}%")
    print(f"  Top-2准确率: {top2:.2f}%")
