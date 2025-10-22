import os
import time
import yaml
import shutil
import random
import datetime
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
from transformers import AutoImageProcessor, AutoModel, AutoConfig, get_cosine_schedule_with_warmup

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.metrics import AverageMeter, compute_accuracy
from utils.model_statistics import detailed_model_summary
from utils.data_loader_cache import get_places365_dataloaders_cache, get_places365_dataloaders_normal


def setup(rank, world_size):
    """设置分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = str(world_size)
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """清理分布式训练环境"""
    dist.destroy_process_group()
    
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class SceneRecognitionTrainer:
    def __init__(self, config):
        self.cfg = config
        self.start_epoch = 0
        self.best_prec1 = 0

        self.rank = dist.get_rank() if dist.is_initialized() else 0
        pid = os.getpid()
        print(f'current pid: {pid}')
        print(f'Current rank {self.rank}')
        device_id = self.rank % torch.cuda.device_count()

        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        self.scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available() and config.use_amp)


        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        # 只在rank 0上初始化TensorBoard writer
        if self.rank == 0:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.writer = SummaryWriter(log_dir=f'runs/places365_{current_time}')
        else:
            self.writer = None

        # 初始化模型、优化器和学习率调度器
        self._init_model()
        self._init_optimizer()
        self._init_scheduler()
        
        # 初始化数据加载器
        self._init_dataloaders()
        
        # 初始化损失函数
        self._init_criterion()
    
    def _init_model(self):
        """初始化模型"""
        print(f"[Creating model '{self.cfg.arch}']")
        model_name = self.cfg.arch.lower()
        if model_name == 'dinov3':
            from models import DinoV3Linear
            MODEL_NAME_OR_PATH = self.cfg.pretrained_weights # "./checkpoints/weights/dinov3-vits16-pretrain-lvd1689m"
            backbone = AutoModel.from_pretrained(MODEL_NAME_OR_PATH)
            self.model = DinoV3Linear(backbone, self.cfg.num_classes, freeze_backbone=self.cfg.use_freeze_backbone) # freze backbone
        elif model_name.startswith('resnet') or model_name.startswith('vgg'):
            from models import ResNetLinear
            backbone = models.__dict__[self.cfg.arch](num_classes=365) # , pretrained=True
            self.model = ResNetLinear(backbone, self.cfg.num_classes, freeze_backbone=self.cfg.use_freeze_backbone, pretrained_weights_path=self.cfg.pretrained_weights) # freze backbone
        elif model_name == 'swin':
            from models import SwinLinear
            from models.swin_transformer import build_model as build_swin_model
            backbone = build_swin_model(self.cfg)
            self.model = SwinLinear(backbone, self.cfg.num_classes, freeze_backbone=self.cfg.use_freeze_backbone, pretrained_weights_path=self.cfg.pretrained_weights) # freze backbone
        else:
            raise ValueError(f"Unsupported architecture '{self.cfg.arch}'")

        # 使用DDP包装模型而不是DataParallel
        self.model.to(self.device)
        if self.cfg.use_DDP:
            self.model = DDP(self.model, device_ids=[self.rank], find_unused_parameters=True)
    
    def _init_optimizer(self):
        """初始化优化器"""
        opt_lower = self.cfg.optimizer.lower()
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())

        if opt_lower == 'sgd':
            self.optimizer = torch.optim.SGD(parameters, self.cfg.lr, momentum=self.cfg.momentum, weight_decay=self.cfg.weight_decay)
        elif opt_lower == 'adamw':
            self.optimizer = torch.optim.AdamW(parameters, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        elif opt_lower == 'fused_adam':
            from apex.optimizers import FusedAdam
            self.optimizer = FusedAdam(parameters, eps=1e-8, betas=(0.9, 0.999), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        elif opt_lower == 'fused_lamb':
            from apex.optimizers import FusedLAMB
            self.optimizer = FusedLAMB(parameters, eps=1e-8, betas=(0.9, 0.999), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer '{self.cfg.optimizer}'")
    
    def _init_scheduler(self):
        """初始化学习率调度器"""
        scheduler_lower = self.cfg.scheduler.lower()
        if scheduler_lower == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif scheduler_lower == 'multistep':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30, 60, 90], gamma=0.1)
        elif scheduler_lower == 'linear':
            self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.0, total_iters=self.cfg.epochs)
        elif scheduler_lower == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.cfg.epochs, eta_min=self.cfg.lr * 0.001)
        elif scheduler_lower == 'cosine_warmup':
            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=int(0.05*self.cfg.epochs), num_training_steps=self.cfg.epochs)
        else:
            raise ValueError(f"Unsupported scheduler '{self.cfg.scheduler}'")
        
    def _init_dataloaders(self):
        """初始化数据加载器"""
        if self.cfg.use_dataLoaderCache:
            self.train_loader, self.val_loader = get_places365_dataloaders_cache(
                data_path=self.cfg.dataset_dir,
                batch_size=self.cfg.batch_size,
                workers=self.cfg.workers,
                cache_size=[self.cfg.img_size, self.cfg.img_size],
                cache_boost=False,
                use_distributed=self.cfg.use_DDP,
                rank=self.rank,
                world_size=self.cfg.world_size
            )
        else:
            self.train_loader, self.val_loader = get_places365_dataloaders_normal(
                data_path=self.cfg.dataset_dir,
                batch_size=self.cfg.batch_size,
                workers=self.cfg.workers,
                cache_size=[self.cfg.img_size, self.cfg.img_size],
                use_distributed=self.cfg.use_DDP,
                rank=self.rank,
                world_size=self.cfg.world_size
            )
        if self.rank == 0:
            print(f"[Using {len(self.train_loader)} batches for training, {len(self.val_loader)} for validation]")

    def _init_criterion(self):
        """初始化损失函数"""
        if self.cfg.label_smoothing > 0.0:
            from utils.losses import LabelSmoothCrossEntropy
            self.criterion_train = LabelSmoothCrossEntropy(smoothing=0.1).to(self.device)
        else:
            self.criterion_train = nn.CrossEntropyLoss().to(self.device)
        self.criterion_val = nn.CrossEntropyLoss().to(self.device)
        
    def load_checkpoint(self):
        """加载检查点"""
        if self.cfg.resume:
            if os.path.isfile(self.cfg.resume):
                print("=> loading checkpoint '{}'".format(self.cfg.resume))
                checkpoint = torch.load(self.cfg.resume)
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scaler.load_state_dict(checkpoint['scaler'])
                self.start_epoch = checkpoint['epoch']
                self.best_prec1 = checkpoint['best_prec1']
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
                print(f"=> loaded checkpoint '{self.cfg.resume}' (epoch {checkpoint['epoch']})")
            else:
                raise FileNotFoundError(f"No checkpoint found at '{self.cfg.resume}'")
                
    def log_model_summary(self):
        """记录模型摘要信息"""
        if self.cfg.use_DDP:
            return
        # 将模型结构写入TensorBoard
        try:
            if not self.cfg.arch.lower().startswith('swin'):
                sample_input = torch.randn(1, 3, self.cfg.img_size, self.cfg.img_size).to(self.device)
                self.writer.add_graph(self.model, sample_input)
                
        except Exception as e:
            print(f"Could not add model graph to TensorBoard: {str(e)}")
        # 打印模型统计参数
        detailed_model_summary(self.model, (1, 3, self.cfg.img_size, self.cfg.img_size))
        
    def train_epoch(self, epoch):
        """训练一个epoch"""
        # 设置sampler的epoch以确保每个epoch的shuffle不同
        if self.cfg.use_DDP and hasattr(self.train_loader, 'sampler'):
            self.train_loader.sampler.set_epoch(epoch)

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to train mode
        self.model.train()

        end = time.time()
        if self.rank == 0:  # 只在rank 0上显示进度条
            pbar_batch = tqdm(self.train_loader, desc='Training')
        else:
            pbar_batch = self.train_loader
        for i, (input, target) in enumerate(pbar_batch):
            # measure data loading time
            data_time.update(time.time() - end)

            input = input.to(self.device, non_blocking=True) 
            target = target.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # compute output with mixed precision
            with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available() and self.cfg.use_amp):
                output = self.model(input)
                loss = self.criterion_train(output, target)

            # measure accuracy and record loss
            prec1, prec5 = compute_accuracy(output.data, target, topk=(1, 5))

            # reduce loss & accuracy
            if self.cfg.use_DDP:
                reduced_loss = self._reduce_tensor(loss.item())
                reduced_top1 = self._reduce_tensor(prec1)
                reduced_top5 = self._reduce_tensor(prec5)
            else:
                reduced_loss = loss.item()
                reduced_top1 = prec1
                reduced_top5 = prec5

            losses.update(reduced_loss, input.size(0))
            top1.update(reduced_top1, input.size(0))
            top5.update(reduced_top5, input.size(0))

            # compute gradient and do SGD step
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # 耗时监控
            batch_time.update(time.time() - end)
            end = time.time()
            
            # 显示进度信息 (只在rank 0上)
            if self.rank == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                postfix = {
                    'Epoch': f'[{epoch}][{i}/{len(self.train_loader)}]',
                    'Time': f'{batch_time.val:.3f}({batch_time.avg:.3f})',
                    'Data': f'{data_time.val:.3f}({data_time.avg:.3f})',
                    'Loss': f'{losses.val:.4f}({losses.avg:.4f})',
                    'LR': f'{current_lr:.6f}',
                    'Prec@1': f'{top1.val:06.3f}({top1.avg:06.3f})',
                    'Prec@5': f'{top5.val:06.3f}({top5.avg:06.3f})'
                }
                pbar_batch.set_postfix(postfix) 
        
        # 将训练指标写入TensorBoard (只在rank 0上)
        if self.rank == 0 and self.writer is not None:
            self.writer.add_scalar('Train/Loss', losses.avg, epoch)
            self.writer.add_scalar('Train/Prec@1', top1.avg, epoch)
            self.writer.add_scalar('Train/Prec@5', top5.avg, epoch)
            self.writer.add_scalar('Train/Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
        
        return losses.avg, top1.avg, top5.avg

    def validate(self, epoch):
        """验证模型"""
        # 设置sampler的epoch
        if self.cfg.use_DDP and hasattr(self.val_loader, 'sampler'):
            self.val_loader.sampler.set_epoch(epoch)

        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        with torch.no_grad():
            if self.rank == 0:  # 只在rank 0上显示进度条
                pbar_batch = tqdm(self.val_loader, desc='Validating')
            else:
                pbar_batch = self.val_loader
            for i, (input, target) in enumerate(pbar_batch):
                input = input.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                # compute output
                output = self.model(input)
                loss = self.criterion_val(output, target)

                # measure accuracy and record loss
                prec1, prec5 = compute_accuracy(output.data, target, topk=(1, 5))

                # reduce loss & accuracy
                if self.cfg.use_DDP:
                    reduced_loss = self._reduce_tensor(loss.item())
                    reduced_top1 = self._reduce_tensor(prec1)
                    reduced_top5 = self._reduce_tensor(prec5)
                else:
                    reduced_loss = loss.item()
                    reduced_top1 = prec1
                    reduced_top5 = prec5

                losses.update(reduced_loss, input.size(0))
                top1.update(reduced_top1, input.size(0))
                top5.update(reduced_top5, input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if self.rank == 0:
                    postfix = {
                        'Test': f'[{i}/{len(self.val_loader)}]',
                        'Time': f'{batch_time.val:.3f}({batch_time.avg:.3f})',
                        'Loss': f'{losses.val:.4f}({losses.avg:.4f})',
                        'Prec@1': f'{top1.val:06.3f}({top1.avg:06.3f})',
                        'Prec@5': f'{top5.val:06.3f}({top5.avg:06.3f})'
                    }
                    pbar_batch.set_postfix(postfix)

            if self.rank == 0:
                print(f' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}')
                # 将验证指标写入TensorBoard
                if self.writer is not None:
                    self.writer.add_scalar('Val/Loss', losses.avg, epoch)
                    self.writer.add_scalar('Val/Prec@1', top1.avg, epoch)
                    self.writer.add_scalar('Val/Prec@5', top5.avg, epoch)

        return top1.avg

    def _reduce_tensor(self, tensor_data):
        rt = torch.tensor(tensor_data).to(self.device)
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= dist.get_world_size()
        return rt

    def save_checkpoint(self, epoch, prec1, is_best, filename='resnet18', epoch_interval=10):
        """保存检查点"""
        if self.rank != 0:
            return
        save_dir_pure = 'checkpoints/saved/weights_only_checkpoints/'
        save_dir_full = 'checkpoints/saved/full_checkpoints/'
        os.makedirs(save_dir_pure, exist_ok=True)
        os.makedirs(save_dir_full, exist_ok=True)
            
        state = {
            'epoch': epoch + 1, 
            'arch': self.cfg.arch, 
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
            'state_dict': self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict(), 
            'best_prec1': self.best_prec1
        }
        
        # 保存模型参数(纯模型结构，不支持断点续训)
        torch.save(state['state_dict'], save_dir_pure + f"{filename}_Epoch{state['epoch']}_{prec1:.3f}_pure.pth")
        # 保存模型参数(支持断点续训)
        torch.save(state, save_dir_full + f'{filename}_latest.pth')
        if is_best:
            shutil.copyfile(
                save_dir_full + f'{filename}_latest.pth', 
                save_dir_full + f"{filename}_best_Epoch{state['epoch']}_{prec1:.3f}.pth"
            )
        # 定期保存
        if state['epoch'] % epoch_interval == 0:
            shutil.copyfile(
                save_dir_full + f'{filename}_latest.pth', 
                save_dir_full + f"{filename}_epoch_Epoch{state['epoch']}_{prec1:.3f}.pth"
            )

    def train(self):
        """执行训练过程"""
        # 加载检查点（如果指定）
        self.load_checkpoint()
        
        # 记录模型摘要
        self.log_model_summary()
        
        # 如果只是评估模式
        if self.cfg.evaluate:
            self.best_prec1 = 0
            self.validate(0)
            if self.rank == 0 and self.writer is not None:
                self.writer.close()
            return

        # 训练循环
        for epoch in tqdm(range(self.start_epoch, self.cfg.epochs), desc='Epoch'):
            # 训练一个epoch
            _train_loss, _train_prec1, _train_prec5 = self.train_epoch(epoch)

            # 验证模型
            prec1 = self.validate(epoch)

            # 保存检查点
            is_best = prec1 > self.best_prec1
            self.best_prec1 = max(prec1, self.best_prec1)
            self.save_checkpoint(epoch, prec1, is_best, self.cfg.arch.lower())

            # 更新学习率
            self.scheduler.step()

        # 关闭TensorBoard writer
        if self.rank == 0 and self.writer is not None:
            self.writer.close()

def main_ddp(rank, cfg):
    """DDP主函数"""
    try:
        # 设置分布式训练环境
        setup(rank, cfg.world_size)
    
        # 设置随机种子
        seed_all(42)
        
        # 创建trainer实例
        trainer = SceneRecognitionTrainer(cfg)
        trainer.train()
    finally:
        # 清理环境
        cleanup()

def main(cfg):
    """主函数"""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=cfg.GPUs
    # 获取GPU数量
    world_size = torch.cuda.device_count()
    cfg.world_size = world_size
    
    # 如果只有一个GPU，即使配置要求DDP也不使用
    if cfg.world_size == 1:
        cfg.use_DDP = False
        print("Only one GPU available, DDP disabled.")
    

    if cfg.use_DDP:
        # 使用多GPU分布式训练
        mp.spawn(main_ddp, args=(cfg, ), nprocs=cfg.world_size, join=True)
    else:
        # 单GPU训练
        seed_all(42)
        trainer = SceneRecognitionTrainer(cfg)
        trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # train_resnet18_params.yaml | train_dinov3_params.yaml | train_swin_params.yaml
    parser.add_argument('--cfg', type=str, default='configs/train_dinov3_params.yaml')

    args = parser.parse_args()
    cfg = argparse.Namespace(**yaml.load(open(args.cfg), Loader=yaml.SafeLoader))
    
    main(cfg)
