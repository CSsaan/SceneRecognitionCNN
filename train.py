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

from utils.metrics import AverageMeter, compute_accuracy
from utils.model_statistics import detailed_model_summary
from utils.data_loader_cache import get_places365_dataloaders_cache, get_places365_dataloaders_normal


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SceneRecognitionTrainer:
    def __init__(self, config):
        self.cfg = config
        self.start_epoch = 0
        self.best_prec1 = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available() and config.use_amp)
        torch.backends.cudnn.benchmark = True
        
        # 初始化TensorBoard writer
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(log_dir=f'runs/places365_{current_time}')
        
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

        # DataParallel
        device_ids = list(range(torch.cuda.device_count()))
        print(f"正在使用 {torch.cuda.device_count()} 个 GPU 进行训练！")
        if self.cfg.arch.lower().startswith('alexnet') or self.cfg.arch.lower().startswith('vgg'):
            self.model.features = nn.DataParallel(self.model.features, device_ids=device_ids)
        elif torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model, device_ids=device_ids)
            
        self.model.to(self.device)
    
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
            raise ValueError(f"Unsupported optimizer '{self.optimizer}'")
    
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
            raise ValueError(f"Unsupported scheduler '{self.scheduler}'")
        
    def _init_dataloaders(self):
        """初始化数据加载器"""
        if self.cfg.use_dataLoaderCache:
            self.train_loader, self.val_loader = get_places365_dataloaders_cache(
                data_path=self.cfg.dataset_dir,
                batch_size=self.cfg.batch_size,
                workers=self.cfg.workers,
                cache_size=[self.cfg.img_size, self.cfg.img_size],
                cache_boost=False
            )
        else:
            self.train_loader, self.val_loader = get_places365_dataloaders_normal(
                data_path=self.cfg.dataset_dir,
                batch_size=self.cfg.batch_size,
                workers=self.cfg.workers,
                cache_size=[self.cfg.img_size, self.cfg.img_size]
            )
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
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to train mode
        self.model.train()

        end = time.time()
        pbar_batch = tqdm(self.train_loader, desc='Training')
        for i, (input, target) in enumerate(pbar_batch):
            # measure data loading time
            data_time.update(time.time() - end)

            input = input.to(self.device) 
            target = target.to(self.device)

            self.optimizer.zero_grad()

            # compute output with mixed precision
            with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available() and self.cfg.use_amp):
                output = self.model(input)
                loss = self.criterion_train(output, target)

            # measure accuracy and record loss
            prec1, prec5 = compute_accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1, input.size(0))
            top5.update(prec5, input.size(0))

            # compute gradient and do SGD step
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # 耗时监控
            batch_time.update(time.time() - end)
            end = time.time()
            
            # 显示进度信息
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
        
        # 将训练指标写入TensorBoard
        self.writer.add_scalar('Train/Loss', losses.avg, epoch)
        self.writer.add_scalar('Train/Prec@1', top1.avg, epoch)
        self.writer.add_scalar('Train/Prec@5', top5.avg, epoch)
        self.writer.add_scalar('Train/Learning_Rate', current_lr, epoch)
        
        return losses.avg, top1.avg, top5.avg

    def validate(self, epoch):
        """验证模型"""
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        with torch.no_grad():
            pbar_batch = tqdm(self.val_loader, desc='Validating')
            for i, (input, target) in enumerate(pbar_batch):
                input = input.to(self.device)
                target = target.to(self.device)

                # compute output
                output = self.model(input)
                loss = self.criterion_val(output, target)

                # measure accuracy and record loss
                prec1, prec5 = compute_accuracy(output.data, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(prec1, input.size(0))
                top5.update(prec5, input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                postfix = {
                    'Test': f'[{i}/{len(self.val_loader)}]',
                    'Time': f'{batch_time.val:.3f}({batch_time.avg:.3f})',
                    'Loss': f'{losses.val:.4f}({losses.avg:.4f})',
                    'Prec@1': f'{top1.val:06.3f}({top1.avg:06.3f})',
                    'Prec@5': f'{top5.val:06.3f}({top5.avg:06.3f})'
                }
                pbar_batch.set_postfix(postfix)

            print(f' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}')
        
            # 将验证指标写入TensorBoard
            self.writer.add_scalar('Val/Loss', losses.avg, epoch)
            self.writer.add_scalar('Val/Prec@1', top1.avg, epoch)
            self.writer.add_scalar('Val/Prec@5', top5.avg, epoch)

        return top1.avg

    def save_checkpoint(self, epoch, prec1, is_best, filename='resnet18', epoch_interval=10):
        """保存检查点"""
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
        self.writer.close()


def main(cfg):
    """主函数"""
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
