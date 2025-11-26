"""
trainer.py: 通用模型训练器

支持指定训练目标（大类或小类）和标签映射。
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional, List, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

class Trainer:
    """
    通用训练器，用于训练任意一个 DualStreamSpatio_TemporalFusionNetwork 实例。
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        num_classes: int = 2,
        target_key: str = 'label',       # 关键参数：告诉 Trainer 从 batch 中取哪个作为标签
        label_mapping: Dict[int, int] = None, # 关键参数：用于将全局ID映射为本地ID (0 ~ num_classes-1)
        device: str = 'cuda',
        output_dir: Optional[Path] = None,
        verbose: bool = True,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.num_classes = num_classes
        self.target_key = target_key
        self.label_mapping = label_mapping
        self.device = torch.device(device)
        self.verbose = verbose
        self.output_dir = Path(output_dir) if output_dir else Path('./experiments/outputs')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = self.model.to(self.device)
        
        # 计算权重时需要考虑映射
        self.class_weights = self._compute_class_weights()
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        
        self.best_val_f1 = -np.inf
        self.best_epoch = 0
        
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)

    def _get_labels_from_batch(self, batch):
        """辅助函数：从 batch 中提取标签并应用映射"""
        if isinstance(batch, dict):
            raw_labels = batch[self.target_key]
        else:
            # 假设 tuple 格式最后是 label，这在现在的 dataset 中不太可能，主要是 dict
            raw_labels = batch[-1]
            
        raw_labels = raw_labels.to(self.device)
        
        # 如果有映射表（用于训练小类子模型时，将全局ID映射回 0~N）
        if self.label_mapping is not None:
            # 使用 torch.tensor 的 apply 或者 map 比较慢，建议预处理
            # 这里为了通用性，在 GPU 上做 lookup
            # 注意：这假设 label_mapping 覆盖了 batch 中所有出现的 label
            mapped_labels = torch.zeros_like(raw_labels)
            for global_id, local_id in self.label_mapping.items():
                mapped_labels[raw_labels == global_id] = local_id
            return mapped_labels
        
        return raw_labels

    def _compute_class_weights(self):
        """计算类别权重"""
        print(f"⚖️  正在计算类别权重 (Target: {self.target_key})...")
        label_counts = np.zeros(self.num_classes)
        total = 0
        
        # 遍历一遍数据
        for batch in tqdm(self.train_dataloader, desc="Stat Weights", leave=False):
            labels = self._get_labels_from_batch(batch).cpu().numpy()
            for l in labels:
                if 0 <= l < self.num_classes:
                    label_counts[l] += 1
                    total += 1
        
        weights = np.zeros(self.num_classes)
        for c in range(self.num_classes):
            if label_counts[c] > 0:
                weights[c] = total / (self.num_classes * label_counts[c])
            else:
                weights[c] = 1.0
        
        # 归一化
        weights = weights / weights.mean()
        return torch.from_numpy(weights).float().to(self.device)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        all_preds, all_targets = [], []
        
        pbar = tqdm(self.train_dataloader, desc=f"Ep {epoch} Train", leave=False)
        for batch in pbar:
            # 1. 准备数据
            dynamic = batch['dynamic'].to(self.device)
            static = batch['static'].to(self.device)
            labels = self._get_labels_from_batch(batch) # 获取映射后的标签
            
            # 2. 前向
            outputs = self.model(dynamic, static)
            logits = outputs['logits']
            
            # 3. 损失
            loss = self.criterion(logits, labels)
            
            # 4. 反向
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        metrics = self._compute_metrics(all_preds, all_targets)
        metrics['loss'] = total_loss / len(self.train_dataloader)
        return metrics

    def validate(self):
        if not self.val_dataloader: return {}
        self.model.eval()
        total_loss = 0
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Val", leave=False):
                dynamic = batch['dynamic'].to(self.device)
                static = batch['static'].to(self.device)
                labels = self._get_labels_from_batch(batch)
                
                outputs = self.model(dynamic, static)
                loss = self.criterion(outputs['logits'], labels)
                
                total_loss += loss.item()
                preds = torch.argmax(outputs['logits'], dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
        
        metrics = self._compute_metrics(all_preds, all_targets)
        metrics['loss'] = total_loss / len(self.val_dataloader)
        return metrics

    def _compute_metrics(self, preds, targets):
        return {
            'accuracy': accuracy_score(targets, preds),
            'f1_macro': f1_score(targets, preds, average='macro', zero_division=0)
        }

    def train(self, num_epochs=50, lr=1e-3, patience=10, weight_decay=1e-4):
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        patience_counter = 0
        
        print(f"🚀 开始训练 (Epochs: {num_epochs}, Target: {self.target_key})")
        
        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            
            print(f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f} Acc={train_metrics['accuracy']:.4f} | "
                  f"Val Loss={val_metrics.get('loss',0):.4f} Acc={val_metrics.get('accuracy',0):.4f} F1={val_metrics.get('f1_macro',0):.4f}")
            
            # 保存最佳
            val_f1 = val_metrics.get('f1_macro', 0)
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_epoch = epoch
                patience_counter = 0
                torch.save(self.model.state_dict(), self.output_dir / 'best_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"⏹️ 早停 (Patience {patience})")
                break
        
        # 加载最佳模型
        best_path = self.output_dir / 'best_model.pth'
        if best_path.exists():
            self.model.load_state_dict(torch.load(best_path))
            print(f"✅ 已加载最佳模型 (Epoch {self.best_epoch}, F1={self.best_val_f1:.4f})")