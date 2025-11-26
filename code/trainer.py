"""
trainer.py: 训练循环与日志系统

【第六阶段】训练循环与日志系统 (Training Loop & Operations)

【核心理念】

这不仅是简单的 for 循环，而是构建一个可监控、可中断、可恢复的训练引擎。
关键特性：

1. 健壮的训练工程
   ✓ 模块化架构：重复调用代码封装为类
   ✓ 完整的错误处理和恢复机制
   ✓ 详细的训练日志和指标可视化
   ✓ 中断恢复：支持 checkpoint 保存和加载
   ✓ 早停机制：验证集指标不上升时停止训练

2. 损失函数设计
   ✓ 类别不平衡处理：自动计算 class_weights
   ✓ 掩膜损失：仅对中心像素计算梯度
   ✓ 可选的焦点损失（Focal Loss）应对极端不平衡

3. 验证协议
   ✓ 多指标评估：Accuracy, Precision, Recall, F1-Score, IoU
   ✓ Debug 模式：在小样本上快速过拟合测试
   ✓ 详细的验证报告和混淆矩阵

【架构设计】

┌─────────────────────────────────────────────────────────────┐
│  Trainer 类                                                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  初始化：                                                    │
│  - 模型、优化器、损失函数、设备                              │
│  - 验证/测试集、指标计算器                                   │
│  - 日志记录器                                                │
│                                                              │
│  核心方法：                                                  │
│  1. train_epoch()       → 执行一个 epoch 的训练              │
│  2. validate()          → 在验证集上评估                     │
│  3. train()             → 完整的训练循环                     │
│  4. save_checkpoint()   → 保存模型权重                       │
│  5. load_checkpoint()   → 加载模型权重                       │
│  6. compute_class_weights() → 计算类别权重                   │
│                                                              │
│  辅助组件：                                                  │
│  - MaskedCrossEntropyLoss:  掩膜损失函数                     │
│  - MetricsCalculator:       指标计算器                       │
│  - TrainingLogger:          训练日志                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd  # 引入 pandas 用于输出 CSV
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


# ============================================================================
# 日志记录器
# ============================================================================

class TrainingLogger:
    """训练日志记录器"""
    
    def __init__(self, output_dir: Path, verbose: bool = True):
        """
        初始化日志记录器
        
        Args:
            output_dir: 输出目录
            verbose: 是否打印到控制台
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        
        self.log_file = self.output_dir / 'training_log.txt'
        self.metrics_file = self.output_dir / 'training_metrics.json'
        
        self.metrics_history = {
            'train_loss': [],
            'train_accuracy': [],
            'train_f1_macro': [],
            'train_f1_weighted': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1_macro': [],
            'val_f1_weighted': [],
            'val_iou': [],
        }
        
        self._setup_logging()
    
    @staticmethod
    def _setup_logging():
        """配置日志系统"""
        if not logging.getLogger(__name__).handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(levelname)s: %(message)s'
            )
            handler.setFormatter(formatter)
            logging.getLogger(__name__).addHandler(handler)
            logging.getLogger(__name__).setLevel(logging.INFO)
    
    def log(self, message: str, level: str = 'INFO'):
        """
        记录日志
        
        Args:
            message: 日志消息
            level: 日志级别
        """
        logger = logging.getLogger(__name__)
        
        if level == 'INFO':
            logger.info(message)
        elif level == 'WARNING':
            logger.warning(message)
        elif level == 'ERROR':
            logger.error(message)
        elif level == 'DEBUG':
            logger.debug(message)
        
        # 写入文件
        with open(self.log_file, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp}] {level}: {message}\n")
    
    def log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """记录指标"""
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
    
    def save_metrics(self):
        """保存指标到 JSON 文件"""
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def print_header(self, title: str):
        """打印标题"""
        if self.verbose:
            line = "=" * 80
            print(f"\n{line}")
            print(f"🚀 {title}")
            print(f"{line}\n")
        self.log(f"\n{'=' * 80}")
        self.log(title)
        self.log(f"{'=' * 80}")
    
    def print_epoch_summary(self, epoch: int, num_epochs: int, metrics: Dict[str, float], is_best: bool = False):
        """打印 epoch 摘要"""
        best_marker = " (↑ best)" if is_best else ""
        
        # ✨ 根据可用的指标判断分层还是标准模式
        if 'train_hierarchical_accuracy' in metrics:
            # 分层模式
            train_loss = metrics.get('train_loss', 0)
            train_major_acc = metrics.get('train_major_accuracy', 0)
            train_detail_acc = metrics.get('train_detail_accuracy', 0)
            train_hier_acc = metrics.get('train_hierarchical_accuracy', 0)
            
            val_loss = metrics.get('val_loss', 0)
            val_major_acc = metrics.get('val_major_accuracy', 0)
            val_detail_acc = metrics.get('val_detail_accuracy', 0)
            val_hier_acc = metrics.get('val_hierarchical_accuracy', 0)
            
            message = (
                f"Epoch {epoch}/{num_epochs}: "
                f"Train Loss={train_loss:.4f} "
                f"Major={train_major_acc:.1%} Detail={train_detail_acc:.1%} Hier={train_hier_acc:.1%} | "
                f"Val Loss={val_loss:.4f} "
                f"Major={val_major_acc:.1%} Detail={val_detail_acc:.1%} Hier={val_hier_acc:.1%}{best_marker}"
            )
        else:
            # 标准模式
            train_loss = metrics.get('train_loss', 0)
            train_acc = metrics.get('train_accuracy', 0)
            train_f1 = metrics.get('train_f1_macro', 0)
            
            val_loss = metrics.get('val_loss', 0)
            val_acc = metrics.get('val_accuracy', 0)
            val_f1 = metrics.get('val_f1_macro', 0)
            
            message = (
                f"Epoch {epoch}/{num_epochs}: "
                f"Train Loss={train_loss:.4f} Acc={train_acc:.1%} F1={train_f1:.4f} | "
                f"Val Loss={val_loss:.4f} Acc={val_acc:.1%} F1={val_f1:.4f}{best_marker}"
            )
        
        if self.verbose:
            print(message)
        
        self.log(message)


# ============================================================================
# 指标计算器
# ============================================================================

class MetricsCalculator:
    """指标计算器"""
    
    @staticmethod
    def compute_metrics(
        predictions: np.ndarray,
        targets: np.ndarray,
        num_classes: int,
        average_methods: List[str] = None,
    ) -> Dict[str, float]:
        """
        计算分类指标
        
        Args:
            predictions: 预测标签 (N,)
            targets: 真实标签 (N,)
            num_classes: 类别总数
            average_methods: 平均方法列表 ['macro', 'weighted']
        
        Returns:
            包含各项指标的字典
        """
        if average_methods is None:
            average_methods = ['macro', 'weighted']
        
        metrics = {
            'accuracy': float(accuracy_score(targets, predictions)),
        }
        
        # F1-Score（多种平均方式）
        for avg_method in average_methods:
            key = f'f1_{avg_method}'
            try:
                metrics[key] = float(f1_score(targets, predictions, average=avg_method, zero_division=0))
            except:
                metrics[key] = 0.0
        
        # Precision 和 Recall
        try:
            metrics['precision'] = float(precision_score(targets, predictions, average='weighted', zero_division=0))
            metrics['recall'] = float(recall_score(targets, predictions, average='weighted', zero_division=0))
        except:
            metrics['precision'] = 0.0
            metrics['recall'] = 0.0
        
        # IoU (Intersection over Union)
        metrics['iou'] = MetricsCalculator.compute_iou(predictions, targets, num_classes)
        
        return metrics
    
    @staticmethod
    def compute_iou(predictions: np.ndarray, targets: np.ndarray, num_classes: int) -> float:
        """
        计算 IoU (Intersection over Union)
        
        IoU = TP / (TP + FP + FN)
        
        Args:
            predictions: 预测标签 (N,)
            targets: 真实标签 (N,)
            num_classes: 类别总数
        
        Returns:
            平均 IoU
        """
        iou_list = []
        
        for class_id in range(num_classes):
            tp = np.sum((predictions == class_id) & (targets == class_id))
            fp = np.sum((predictions == class_id) & (targets != class_id))
            fn = np.sum((predictions != class_id) & (targets == class_id))
            
            denominator = tp + fp + fn
            if denominator > 0:
                iou = tp / denominator
                iou_list.append(iou)
        
        if iou_list:
            return float(np.mean(iou_list))
        else:
            return 0.0
    
    @staticmethod
    def compute_confusion_matrix(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """计算混淆矩阵"""
        return confusion_matrix(targets, predictions)


# ============================================================================
# 损失函数
# ============================================================================

class WeightedCrossEntropyLoss(nn.Module):
    """
    加权交叉熵损失
    
    用于处理类别不平衡问题。根据类别频率自动计算权重。
    """
    
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
    ):
        """
        初始化加权交叉熵损失
        
        Args:
            weight: 类别权重
            reduction: 归约方式
        """
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(
            weight=weight,
            reduction=reduction,
        )
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算损失"""
        return self.ce_loss(logits, targets)


# ============================================================================
# 训练器类
# ============================================================================

class Trainer:
    """
    训练器类
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        num_classes: Optional[int] = None,
        hierarchical_map: Optional[dict] = None,
        device: str = 'cuda',
        output_dir: Optional[Path] = None,
        verbose: bool = True,
    ):
        """
        初始化训练器
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.hierarchical_map = hierarchical_map
        self.device = torch.device(device)
        self.verbose = verbose
        
        # 确定num_classes
        if hierarchical_map is not None:
            # 使用分层映射计算总小类数
            self.num_classes = sum(
                len(info.get('detail_classes', {})) 
                for info in hierarchical_map.values()
            )
            self.is_hierarchical = True
        else:
            # 使用传入的 num_classes
            self.num_classes = num_classes if num_classes is not None else 8
            self.is_hierarchical = False
        
        # 设置输出目录
        if output_dir is None:
            output_dir = Path('./experiments/outputs')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化日志
        self.logger = TrainingLogger(self.output_dir, verbose=verbose)
        
        # 设备检查
        self.logger.log(f"使用设备: {self.device}")
        if self.device.type == 'cuda':
            self.logger.log(f"GPU 型号: {torch.cuda.get_device_name()}")
        
        # 打印分类模式
        if self.is_hierarchical:
            self.logger.log(f"使用分层分类模式：{len(hierarchical_map)} 个大类，共 {self.num_classes} 个小类")
        else:
            self.logger.log(f"使用标准分类模式：{self.num_classes} 个类别")
        
        # 模型放到设备
        self.model = self.model.to(self.device)
        
        # 计算类别权重
        if self.is_hierarchical:
            self.major_class_weights, self.detail_class_weights = self._compute_class_weights()
            
            # 构建 ID 到名称的映射，用于 CSV 输出
            self.major_id_to_name = {}
            self.detail_id_to_name = {}
            if self.hierarchical_map:
                # 1. 大类 ID -> Name
                self.major_id_to_name = {
                    info['major_id']: name 
                    for name, info in self.hierarchical_map.items()
                }
                # 2. 小类 ID -> Name
                for major_name, info in self.hierarchical_map.items():
                    for detail_name, detail_id in info['detail_classes'].items():
                        self.detail_id_to_name[detail_id] = detail_name
            self.logger.log("已构建类别名称映射表 (ID -> Name)")
            
        else:
            self.class_weights = self._compute_class_weights()
        
        # 初始化优化器和损失函数（延后到 train 方法）
        self.optimizer = None
        self.criterion = None
        self.criterion_major = None
        self.criterion_detail = None
        
        # 训练历史初始化
        if self.is_hierarchical:
            self.history = {
                'train_loss': [],
                'train_major_loss': [],
                'train_detail_loss': [],
                'train_major_accuracy': [],
                'train_detail_accuracy': [],
                'train_hierarchical_accuracy': [],
                'val_loss': [],
                'val_major_loss': [],
                'val_detail_loss': [],
                'val_major_accuracy': [],
                'val_detail_accuracy': [],
                'val_hierarchical_accuracy': [],
            }
        else:
            self.history = {
                'train_loss': [],
                'train_accuracy': [],
                'train_f1_macro': [],
                'train_f1_weighted': [],
                'val_loss': [],
                'val_accuracy': [],
                'val_f1_macro': [],
                'val_f1_weighted': [],
                'val_iou': [],
            }
        
        # 最佳模型追踪
        self.best_val_f1 = -np.inf
        self.best_epoch = 0
        self.patience_counter = 0
    
    def _compute_class_weights(self) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        计算类别权重以处理类别不平衡
        如果是分层模式，返回 (major_weights, detail_weights)
        """
        self.logger.log("[权重计算] 计算类别权重...")
        
        if self.is_hierarchical:
            # ====== 分层模式权重计算 ======
            num_major = len(self.hierarchical_map)
            num_detail = self.num_classes
            
            major_counts = np.zeros(num_major)
            detail_counts = np.zeros(num_detail)
            total_samples = 0
            
            for batch in tqdm(self.train_dataloader, desc="统计类别分布", disable=not self.verbose, leave=False):
                if isinstance(batch, dict):
                    m_labels = batch['major_label'].cpu().numpy()
                    d_labels = batch['detail_label'].cpu().numpy()
                    for l in m_labels: major_counts[l] += 1
                    for l in d_labels: detail_counts[l] += 1
                    total_samples += len(m_labels)
                else:
                    self.logger.log("警告: 分层模式下收到非字典格式 batch，跳过权重计算", 'WARNING')
                    return torch.ones(num_major).to(self.device), torch.ones(num_detail).to(self.device)
            
            # 计算大类权重
            major_weights = np.zeros(num_major)
            for c in range(num_major):
                if major_counts[c] > 0:
                    major_weights[c] = total_samples / (num_major * major_counts[c])
                else:
                    major_weights[c] = 1.0
            major_weights = major_weights / major_weights.mean()
            
            # 计算小类权重
            detail_weights = np.zeros(num_detail)
            for c in range(num_detail):
                if detail_counts[c] > 0:
                    detail_weights[c] = total_samples / (num_detail * detail_counts[c])
                else:
                    detail_weights[c] = 1.0
            detail_weights = detail_weights / detail_weights.mean()
            
            self.logger.log(f"[权重计算] 分层模式 - 大类权重形状: {major_weights.shape}, 小类权重形状: {detail_weights.shape}")
            
            return (
                torch.from_numpy(major_weights).float().to(self.device),
                torch.from_numpy(detail_weights).float().to(self.device)
            )

        else:
            # ====== 标准模式权重计算 ======
            label_counts = np.zeros(self.num_classes)
            total_samples = 0
            
            for batch in tqdm(
                self.train_dataloader,
                desc="统计类别分布",
                disable=not self.verbose,
                leave=False
            ):
                if isinstance(batch, dict):
                    labels = batch['label']
                else:
                    labels = batch[1]  # 假设是 (data, label) 元组
                
                labels = labels.cpu().numpy()
                for label in labels:
                    label_counts[label] += 1
                    total_samples += 1
            
            # 计算权重
            weights = np.zeros(self.num_classes)
            for c in range(self.num_classes):
                if label_counts[c] > 0:
                    # 反向加权
                    weights[c] = total_samples / (self.num_classes * label_counts[c])
                else:
                    weights[c] = 1.0  # 如果某类不存在，权重为 1
            
            # 归一化（使得平均权重为 1）
            weights = weights / weights.mean()
            
            # 打印类别分布
            self.logger.log("[权重计算] 类别分布:")
            for c in range(self.num_classes):
                count = int(label_counts[c])
                weight = weights[c]
                self.logger.log(f"  类别 {c}: {count:6d} 样本 | 权重: {weight:.4f}")
            
            return torch.from_numpy(weights).float().to(self.device)
    
    def train_epoch(self, epoch: int, num_epochs: int) -> Dict[str, float]:
        """执行一个 epoch 的训练"""
        if self.is_hierarchical:
            return self._train_epoch_hierarchical(epoch, num_epochs)
        else:
            return self._train_epoch_standard(epoch, num_epochs)
    
    def _train_epoch_standard(self, epoch: int, num_epochs: int) -> Dict[str, float]:
        """标准分类的训练"""
        self.model.train()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch}/{num_epochs}",
            disable=not self.verbose,
            leave=False
        )
        
        for batch_idx, batch in enumerate(pbar):
            # 获取数据
            if isinstance(batch, dict):
                dynamic = batch['dynamic'].to(self.device)
                static = batch['static'].to(self.device)
                labels = batch['label'].to(self.device)
            else:
                # 假设是元组格式
                dynamic, static, labels = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
            
            # 前向传播
            outputs = self.model(dynamic, static)
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            
            # 计算损失
            loss = self.criterion(logits, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 统计指标
            total_loss += loss.item()
            
            # 获取预测结果
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            targets = labels.cpu().numpy()
            
            all_predictions.extend(predictions)
            all_targets.extend(targets)
            
            # 更新进度条
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # 计算指标
        metrics = MetricsCalculator.compute_metrics(
            np.array(all_predictions),
            np.array(all_targets),
            self.num_classes,
            average_methods=['macro', 'weighted']
        )
        
        metrics['train_loss'] = total_loss / len(self.train_dataloader)
        metrics['train_accuracy'] = metrics.pop('accuracy')
        metrics['train_f1_macro'] = metrics.pop('f1_macro', 0.0)
        metrics['train_f1_weighted'] = metrics.pop('f1_weighted', 0.0)
        
        return metrics
    
    def _train_epoch_hierarchical(self, epoch: int, num_epochs: int) -> Dict[str, float]:
        """分层分类的训练"""
        self.model.train()
        
        total_loss = 0.0
        total_major_loss = 0.0
        total_detail_loss = 0.0
        all_major_preds = []
        all_major_targets = []
        all_detail_preds = []
        all_detail_targets = []
        
        pbar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch}/{num_epochs}",
            disable=not self.verbose,
            leave=False
        )
        
        # 损失权重（可从配置中读取）
        weight_major = 0.3
        weight_detail = 0.7
        
        for batch_idx, batch in enumerate(pbar):
            # 获取数据
            if isinstance(batch, dict):
                dynamic = batch['dynamic'].to(self.device)
                static = batch['static'].to(self.device)
                major_labels = batch['major_label'].to(self.device)
                detail_labels = batch['detail_label'].to(self.device)
            else:
                raise ValueError("分层分类必须使用字典格式的 batch")
            
            # 前向传播：必须传入 major_labels 以启用 Teacher Forcing
            outputs = self.model(dynamic, static, major_labels=major_labels)
            
            major_logits = outputs['major_logits']  # (B, num_major)
            detail_logits = outputs['detail_logits']  # (B, max_detail)
            
            # 计算两级损失
            loss_major = self.criterion_major(major_logits, major_labels)
            loss_detail = self.criterion_detail(detail_logits, detail_labels)
            
            # 加权组合
            loss = weight_major * loss_major + weight_detail * loss_detail
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 统计指标
            total_loss += loss.item()
            total_major_loss += loss_major.item()
            total_detail_loss += loss_detail.item()
            
            # 获取预测结果
            major_preds = torch.argmax(major_logits, dim=1).cpu().numpy()
            major_targets = major_labels.cpu().numpy()
            detail_preds = torch.argmax(detail_logits, dim=1).cpu().numpy()
            detail_targets = detail_labels.cpu().numpy()
            
            all_major_preds.extend(major_preds)
            all_major_targets.extend(major_targets)
            all_detail_preds.extend(detail_preds)
            all_detail_targets.extend(detail_targets)
            
            # 更新进度条
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # 计算指标
        major_metrics = MetricsCalculator.compute_metrics(
            np.array(all_major_preds),
            np.array(all_major_targets),
            len(self.hierarchical_map),
            average_methods=['macro']
        )
        
        detail_metrics = MetricsCalculator.compute_metrics(
            np.array(all_detail_preds),
            np.array(all_detail_targets),
            self.num_classes,
            average_methods=['macro']
        )
        
        # 层级准确率（大类和小类都预测正确）
        hierarchical_correct = (
            np.array(all_major_preds) == np.array(all_major_targets)
        ) & (
            np.array(all_detail_preds) == np.array(all_detail_targets)
        )
        hierarchical_accuracy = hierarchical_correct.mean()
        
        metrics = {
            'train_loss': total_loss / len(self.train_dataloader),
            'train_major_loss': total_major_loss / len(self.train_dataloader),
            'train_detail_loss': total_detail_loss / len(self.train_dataloader),
            'train_major_accuracy': major_metrics.get('accuracy', 0.0),
            'train_detail_accuracy': detail_metrics.get('accuracy', 0.0),
            'train_hierarchical_accuracy': hierarchical_accuracy,
        }
        
        return metrics
    
    def validate(self, epoch: Optional[int] = None) -> Dict[str, float]:
        """
        在验证集上评估模型
        """
        if self.val_dataloader is None:
            return {}
        
        if self.is_hierarchical:
            return self._validate_hierarchical(epoch=epoch)
        else:
            return self._validate_standard()
    
    def _validate_standard(self) -> Dict[str, float]:
        """标准分类的验证"""
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(
            self.val_dataloader,
            desc="验证",
            disable=not self.verbose,
            leave=False
        )
        
        with torch.no_grad():
            for batch in pbar:
                # 获取数据
                if isinstance(batch, dict):
                    dynamic = batch['dynamic'].to(self.device)
                    static = batch['static'].to(self.device)
                    labels = batch['label'].to(self.device)
                else:
                    dynamic, static, labels = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
                
                # 前向传播
                outputs = self.model(dynamic, static)
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs
                
                # 计算损失
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                # 获取预测结果
                predictions = torch.argmax(logits, dim=1).cpu().numpy()
                targets = labels.cpu().numpy()
                
                all_predictions.extend(predictions)
                all_targets.extend(targets)
        
        # 计算指标
        metrics = MetricsCalculator.compute_metrics(
            np.array(all_predictions),
            np.array(all_targets),
            self.num_classes,
            average_methods=['macro', 'weighted']
        )
        
        metrics['val_loss'] = total_loss / len(self.val_dataloader)
        metrics['val_accuracy'] = metrics.pop('accuracy')
        metrics['val_f1_macro'] = metrics.pop('f1_macro', 0.0)
        metrics['val_f1_weighted'] = metrics.pop('f1_weighted', 0.0)
        metrics['val_iou'] = metrics.pop('iou', 0.0)
        
        return metrics
    
    def _validate_hierarchical(self, epoch: Optional[int] = None) -> Dict[str, float]:
        """
        分层分类的验证
        【功能】支持输出预测结果到 CSV
        """
        self.model.eval()
        
        total_loss = 0.0
        total_major_loss = 0.0
        total_detail_loss = 0.0
        
        # 用于收集所有结果以便保存 CSV
        all_results = {
            'major_true': [], 'major_pred': [],
            'detail_true': [], 'detail_pred': []
        }
        
        pbar = tqdm(
            self.val_dataloader,
            desc="验证",
            disable=not self.verbose,
            leave=False
        )
        
        weight_major = 0.3
        weight_detail = 0.7
        
        with torch.no_grad():
            for batch in pbar:
                # 获取数据
                if isinstance(batch, dict):
                    dynamic = batch['dynamic'].to(self.device)
                    static = batch['static'].to(self.device)
                    major_labels = batch['major_label'].to(self.device)
                    detail_labels = batch['detail_label'].to(self.device)
                else:
                    raise ValueError("分层分类必须使用字典格式的 batch")
                
                # 前向传播 (验证不使用 major_labels)
                outputs = self.model(dynamic, static)
                major_logits = outputs['major_logits']
                detail_logits = outputs['detail_logits']
                
                # 计算两级损失
                loss_major = self.criterion_major(major_logits, major_labels)
                loss_detail = self.criterion_detail(detail_logits, detail_labels)
                loss = weight_major * loss_major + weight_detail * loss_detail
                
                total_loss += loss.item()
                total_major_loss += loss_major.item()
                total_detail_loss += loss_detail.item()
                
                # 获取预测结果
                major_preds = torch.argmax(major_logits, dim=1).cpu().numpy()
                major_targets = major_labels.cpu().numpy()
                detail_preds = torch.argmax(detail_logits, dim=1).cpu().numpy()
                detail_targets = detail_labels.cpu().numpy()
                
                # 收集结果
                all_results['major_true'].extend(major_targets)
                all_results['major_pred'].extend(major_preds)
                all_results['detail_true'].extend(detail_targets)
                all_results['detail_pred'].extend(detail_preds)
        
        # 保存 Debug 表格
        if epoch is not None:
            try:
                # 构造 DataFrame
                df = pd.DataFrame(all_results)
                
                # 映射 ID 为中文名称 (如果映射表存在)
                if hasattr(self, 'major_id_to_name') and self.major_id_to_name:
                    df['major_true_name'] = df['major_true'].map(self.major_id_to_name)
                    df['major_pred_name'] = df['major_pred'].map(self.major_id_to_name)
                    df['detail_true_name'] = df['detail_true'].map(self.detail_id_to_name)
                    df['detail_pred_name'] = df['detail_pred'].map(self.detail_id_to_name)
                    
                    # 调整列顺序
                    cols = ['major_true_name', 'major_pred_name', 'detail_true_name', 'detail_pred_name',
                            'major_true', 'major_pred', 'detail_true', 'detail_pred']
                    df = df[cols]
                
                # 增加一列判断是否正确
                df['major_correct'] = df['major_true'] == df['major_pred']
                df['detail_correct'] = df['detail_true'] == df['detail_pred']
                
                # 保存文件
                filename = f'val_predictions_epoch_{epoch}.csv'
                save_path = self.output_dir / filename
                df.to_csv(save_path, index=False, encoding='utf-8-sig')
                if self.verbose:
                    self.logger.log(f"📝 验证集预测结果已保存: {filename}")
                
            except Exception as e:
                self.logger.log(f"⚠️ 保存验证表格失败: {e}", level='WARNING')
        
        # 计算指标
        major_metrics = MetricsCalculator.compute_metrics(
            np.array(all_results['major_pred']),
            np.array(all_results['major_true']),
            len(self.hierarchical_map),
            average_methods=['macro']
        )
        
        detail_metrics = MetricsCalculator.compute_metrics(
            np.array(all_results['detail_pred']),
            np.array(all_results['detail_true']),
            self.num_classes,
            average_methods=['macro']
        )
        
        # 层级准确率
        hierarchical_correct = (
            np.array(all_results['major_pred']) == np.array(all_results['major_true'])
        ) & (
            np.array(all_results['detail_pred']) == np.array(all_results['detail_true'])
        )
        hierarchical_accuracy = hierarchical_correct.mean()
        
        metrics = {
            'val_loss': total_loss / len(self.val_dataloader),
            'val_major_loss': total_major_loss / len(self.val_dataloader),
            'val_detail_loss': total_detail_loss / len(self.val_dataloader),
            'val_major_accuracy': major_metrics.get('accuracy', 0.0),
            'val_detail_accuracy': detail_metrics.get('accuracy', 0.0),
            'val_hierarchical_accuracy': hierarchical_accuracy,
        }
        
        return metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存模型 checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'best_val_f1': self.best_val_f1,
        }
        
        # 保存最后的模型
        last_path = self.output_dir / 'last_model.pth'
        torch.save(checkpoint, last_path)
        
        # 保存最佳模型
        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.logger.log(f"💾 保存最佳模型: Epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path: Path) -> int:
        """加载模型 checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer and checkpoint.get('optimizer_state_dict'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.best_val_f1 = checkpoint.get('best_val_f1', -np.inf)
        epoch = checkpoint.get('epoch', 0)
        
        self.logger.log(f"✅ 加载 checkpoint: {checkpoint_path}")
        
        return epoch
    
    def train(
        self,
        num_epochs: int = 50,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 10,
        debug: bool = False,
        resume_from: Optional[Path] = None,
    ) -> Dict:
        """完整的训练循环"""
        # Debug 模式
        if debug:
            self.logger.print_header("开始 Overfit 测试 (Debug Mode)...")
            num_epochs = 10
            patience = 1000  # 禁用早停
        else:
            self.logger.print_header("开始训练")
        
        # 初始化优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # 初始化损失函数
        if self.is_hierarchical:
            self.criterion_major = WeightedCrossEntropyLoss(weight=self.major_class_weights)
            self.criterion_detail = WeightedCrossEntropyLoss(weight=self.detail_class_weights)
            self.logger.log("已初始化分层损失函数 (Major & Detail)")
        else:
            self.criterion = WeightedCrossEntropyLoss(weight=self.class_weights)
            self.logger.log("已初始化标准损失函数")
        
        self.logger.log(f"学习率: {learning_rate}")
        self.logger.log(f"权重衰减: {weight_decay}")
        self.logger.log(f"早停耐心: {patience} epochs")
        
        # 恢复训练（如果指定）
        start_epoch = 1
        if resume_from and resume_from.exists():
            start_epoch = self.load_checkpoint(resume_from) + 1
        
        # 训练循环
        start_time = time.time()
        
        for epoch in range(start_epoch, num_epochs + 1):
            # 训练一个 epoch
            train_metrics = self.train_epoch(epoch, num_epochs)
            
            # 验证
            val_metrics = self.validate(epoch=epoch) if self.val_dataloader else {}
            
            # 合并指标
            epoch_metrics = {**train_metrics, **val_metrics}
            
            # 记录指标
            for key, value in epoch_metrics.items():
                if key in self.history:
                    self.history[key].append(value)
            
            self.logger.log_metrics(epoch, epoch_metrics)
            
            # 检查是否是最佳模型
            if self.is_hierarchical:
                val_metric = val_metrics.get('val_hierarchical_accuracy', -np.inf)
            else:
                val_metric = val_metrics.get('val_f1_macro', -np.inf)
            
            is_best = val_metric > self.best_val_f1
            
            if is_best:
                self.best_val_f1 = val_metric
                self.best_epoch = epoch
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # 打印 epoch 摘要
            self.logger.print_epoch_summary(epoch, num_epochs, epoch_metrics, is_best=is_best)
            
            # 保存 checkpoint
            self.save_checkpoint(epoch, is_best=is_best)
            
            # 早停
            if self.patience_counter >= patience:
                self.logger.log(f"⏹️  早停：{patience} 个 epoch 无改进")
                break
        
        # 训练完成
        elapsed_time = time.time() - start_time
        
        self.logger.print_header("训练完成")
        self.logger.log(f"总耗时: {elapsed_time / 3600:.2f} 小时")
        
        if self.is_hierarchical:
            self.logger.log(f"最佳模型: Epoch {self.best_epoch} (Val 层级准确率: {self.best_val_f1:.4f})")
        else:
            self.logger.log(f"最佳模型: Epoch {self.best_epoch} (Val F1: {self.best_val_f1:.4f})")
        
        # 加载最佳模型
        best_model_path = self.output_dir / 'best_model.pth'
        if best_model_path.exists():
            self.load_checkpoint(best_model_path)
            self.logger.log("✅ 加载最佳模型权重")
        
        # 保存训练历史
        self.logger.save_metrics()
        self.logger.log(f"💾 训练历史已保存: {self.logger.metrics_file}")
        
        return self.history
    
    def test(self, test_loader=None) -> Dict[str, float]:
        """在测试集上评估模型"""
        if test_loader is None:
            test_loader = self.test_dataloader
        
        if test_loader is None:
            self.logger.log("❌ 未提供测试数据加载器")
            return {}
        
        if self.is_hierarchical:
            return self._test_hierarchical(test_loader)
        else:
            return self._test_standard(test_loader)
    
    def _test_standard(self, test_loader) -> Dict[str, float]:
        """标准分类的测试"""
        self.logger.print_header("测试阶段")
        
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        pbar = tqdm(
            test_loader,
            desc="测试",
            disable=not self.verbose,
        )
        
        with torch.no_grad():
            for batch in pbar:
                # 获取数据
                if isinstance(batch, dict):
                    dynamic = batch['dynamic'].to(self.device)
                    static = batch['static'].to(self.device)
                    labels = batch['label'].to(self.device)
                else:
                    dynamic, static, labels = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
                
                # 前向传播
                outputs = self.model(dynamic, static)
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs
                
                # 获取预测结果
                probabilities = F.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1).cpu().numpy()
                targets = labels.cpu().numpy()
                
                all_predictions.extend(predictions)
                all_targets.extend(targets)
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # 【新增】保存测试集结果到 CSV
        try:
            df = pd.DataFrame({
                'target': all_targets,
                'prediction': all_predictions
            })
            # 如果有概率值也可以保存（可选）
            # df['probability'] = np.max(all_probabilities, axis=1)
            
            save_path = self.output_dir / 'test_predictions.csv'
            df.to_csv(save_path, index=False, encoding='utf-8-sig')
            self.logger.log(f"📝 测试集预测结果已保存: {save_path}")
        except Exception as e:
            self.logger.log(f"⚠️ 保存测试结果CSV失败: {e}", level='WARNING')
        
        # 计算指标
        metrics = MetricsCalculator.compute_metrics(
            np.array(all_predictions),
            np.array(all_targets),
            self.num_classes,
        )
        
        # 打印结果
        self.logger.log("\n📊 测试结果:")
        self.logger.log(f"  Accuracy: {metrics['accuracy']:.4f}")
        self.logger.log(f"  Precision: {metrics['precision']:.4f}")
        self.logger.log(f"  Recall: {metrics['recall']:.4f}")
        self.logger.log(f"  F1 (Macro): {metrics.get('f1_macro', 0):.4f}")
        self.logger.log(f"  F1 (Weighted): {metrics.get('f1_weighted', 0):.4f}")
        self.logger.log(f"  IoU: {metrics['iou']:.4f}")
        
        # 混淆矩阵
        cm = MetricsCalculator.compute_confusion_matrix(
            np.array(all_predictions),
            np.array(all_targets)
        )
        
        # 保存混淆矩阵
        cm_file = self.output_dir / 'confusion_matrix.npy'
        np.save(cm_file, cm)
        self.logger.log(f"💾 混淆矩阵已保存: {cm_file}")
        
        return metrics
    
    def _test_hierarchical(self, test_loader) -> Dict[str, float]:
        """分层分类的测试"""
        self.logger.print_header("测试阶段 (分层分类)")
        
        self.model.eval()
        
        # 用于收集结果
        all_results = {
            'major_true': [], 'major_pred': [],
            'detail_true': [], 'detail_pred': []
        }
        
        pbar = tqdm(
            test_loader,
            desc="测试",
            disable=not self.verbose,
        )
        
        with torch.no_grad():
            for batch in pbar:
                # 获取数据
                if isinstance(batch, dict):
                    dynamic = batch['dynamic'].to(self.device)
                    static = batch['static'].to(self.device)
                    major_labels = batch['major_label'].to(self.device)
                    detail_labels = batch['detail_label'].to(self.device)
                else:
                    raise ValueError("分层分类必须使用字典格式的 batch")
                
                # 前向传播
                outputs = self.model(dynamic, static)
                major_logits = outputs['major_logits']
                detail_logits = outputs['detail_logits']
                
                # 获取预测结果
                major_preds = torch.argmax(major_logits, dim=1).cpu().numpy()
                major_targets = major_labels.cpu().numpy()
                detail_preds = torch.argmax(detail_logits, dim=1).cpu().numpy()
                detail_targets = detail_labels.cpu().numpy()
                
                # 收集结果
                all_results['major_true'].extend(major_targets)
                all_results['major_pred'].extend(major_preds)
                all_results['detail_true'].extend(detail_targets)
                all_results['detail_pred'].extend(detail_preds)
        
        # 【新增】保存测试结果表格
        try:
            df = pd.DataFrame(all_results)
            
            # 映射 ID 为中文名称 (如果映射表存在)
            if hasattr(self, 'major_id_to_name') and self.major_id_to_name:
                df['major_true_name'] = df['major_true'].map(self.major_id_to_name)
                df['major_pred_name'] = df['major_pred'].map(self.major_id_to_name)
                df['detail_true_name'] = df['detail_true'].map(self.detail_id_to_name)
                df['detail_pred_name'] = df['detail_pred'].map(self.detail_id_to_name)
                
                # 调整列顺序
                cols = ['major_true_name', 'major_pred_name', 'detail_true_name', 'detail_pred_name',
                        'major_true', 'major_pred', 'detail_true', 'detail_pred']
                df = df[cols]
            
            # 增加一列判断是否正确
            df['major_correct'] = df['major_true'] == df['major_pred']
            df['detail_correct'] = df['detail_true'] == df['detail_pred']
            
            # 保存文件
            save_path = self.output_dir / 'test_predictions.csv'
            df.to_csv(save_path, index=False, encoding='utf-8-sig')
            self.logger.log(f"📝 测试集预测结果已保存: {save_path}")
            
        except Exception as e:
            self.logger.log(f"⚠️ 保存测试表格失败: {e}", level='WARNING')

        # 计算指标
        major_metrics = MetricsCalculator.compute_metrics(
            np.array(all_results['major_pred']),
            np.array(all_results['major_true']),
            len(self.hierarchical_map),
        )
        
        detail_metrics = MetricsCalculator.compute_metrics(
            np.array(all_results['detail_pred']),
            np.array(all_results['detail_true']),
            self.num_classes,
        )
        
        # 层级准确率
        hierarchical_correct = (
            np.array(all_results['major_pred']) == np.array(all_results['major_true'])
        ) & (
            np.array(all_results['detail_pred']) == np.array(all_results['detail_true'])
        )
        hierarchical_accuracy = hierarchical_correct.mean()
        
        # 打印结果
        self.logger.log("\n📊 测试结果 (分层分类):")
        self.logger.log(f"  大类准确率: {major_metrics['accuracy']:.4f}")
        self.logger.log(f"  小类准确率: {detail_metrics['accuracy']:.4f}")
        self.logger.log(f"  层级准确率: {hierarchical_accuracy:.4f}")
        
        metrics = {
            'major_accuracy': major_metrics['accuracy'],
            'detail_accuracy': detail_metrics['accuracy'],
            'hierarchical_accuracy': hierarchical_accuracy,
            'major_f1': major_metrics.get('f1_macro', 0.0),
            'detail_f1': detail_metrics.get('f1_macro', 0.0),
        }
        
        return metrics