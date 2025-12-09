import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import numpy as np
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import json

# ============================================================================
# 标签平滑交叉熵损失 (Label Smoothing)
# ============================================================================
class LabelSmoothingLoss(nn.Module):
    """
    带标签平滑的交叉熵损失
    减少模型对预测的过度自信，提高泛化性能
    """
    def __init__(self, num_classes, smoothing=0.1, reduction='mean', weight=None, device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction
        self.device = device
        if weight is not None:
            self.weight = weight.to(device) if isinstance(weight, torch.Tensor) else torch.tensor(weight, device=device, dtype=torch.float)
        else:
            self.weight = None
    
    def forward(self, pred, target):
        """
        pred: (B, C) logits
        target: (B,) target indices
        """
        pred = pred.log_softmax(dim=-1)
        
        with torch.no_grad():
            # 创建平滑的target分布
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        # 计算KL散度
        loss = torch.sum(-true_dist * pred, dim=-1)
        
        # 应用类别权重
        if self.weight is not None:
            loss = loss * self.weight[target]
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# ============================================================================
# Focal Loss 定义
# ============================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss: 降低易分样本权重，关注难分样本
    Gamma: 聚焦参数 (默认2.0)
    Alpha: 类别平衡参数 (可以是列表或Tensor)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', device='cuda'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.device = device
        
        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32).to(device)
            else:
                self.alpha = alpha.to(device)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ============================================================================
# Trainer 类
# ============================================================================
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader = None,
        num_classes: int = 2,
        device: str = 'cuda',
        output_dir: str = './output',
        class_weights: torch.Tensor = None,
        target_key: str = 'label',
        verbose: bool = True,
        label_mapping: dict = None,
        use_focal_loss: bool = True,
        label_smoothing: float = 0.1  # 新增参数
    ):
        self.model = model.to(device)
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.test_loader = test_dataloader
        self.num_classes = num_classes
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.target_key = target_key
        self.label_mapping = label_mapping

        # 日志 Handler
        if not any(isinstance(h, logging.FileHandler) for h in self.logger.handlers):
            log_path = self.output_dir / "train.log"
            file_handler = logging.FileHandler(log_path, encoding='utf-8')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # 损失函数选择
        if use_focal_loss:
            self.logger.info(f" 🔧使用 Focal Loss (标签平滑={label_smoothing}) 处理难分样本")
            self.criterion = FocalLoss(alpha=class_weights, gamma=2.0, device=device)
        else:
            self.logger.info(f"🔧 使用 CrossEntropy Loss (标签平滑={label_smoothing})")
            # 使用标签平滑而不是直接CrossEntropy
            self.criterion = LabelSmoothingLoss(num_classes=num_classes, smoothing=label_smoothing, weight=class_weights, device=device)
            
        self.scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        if hasattr(torch.amp, 'GradScaler'):
             self.scaler = torch.amp.GradScaler('cuda')

        self.optimizer = None
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

    def mixup_data(self, x_dyn, x_sta, y, alpha=0.4):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        batch_size = x_dyn.size(0)
        index = torch.randperm(batch_size).to(self.device)
        mixed_dyn = lam * x_dyn + (1 - lam) * x_dyn[index, :]
        mixed_sta = lam * x_sta + (1 - lam) * x_sta[index, :]
        y_a, y_b = y, y[index]
        return mixed_dyn, mixed_sta, y_a, y_b, lam

    def _get_labels(self, batch):
        labels = batch[self.target_key].to(self.device)
        if self.label_mapping:
            cpu_labels = labels.cpu().numpy()
            local_labels = np.array([self.label_mapping.get(x, 0) for x in cpu_labels])
            labels = torch.from_numpy(local_labels).to(self.device).long()
        return labels

    def train(self, num_epochs=50, learning_rate=1e-3, weight_decay=1e-4, patience=10, debug=False, resume_from=None, accumulation_steps=1):
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # 改进：Linear Warmup + Cosine Annealing (比CosineAnnealingWarmRestarts更稳定)
        total_steps = num_epochs * len(self.train_loader)
        warmup_steps = len(self.train_loader) * 2  # 前2个epoch做warmup
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(num_epochs - current_step / len(self.train_loader)) / float(max(1, num_epochs)))
        
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        start_epoch = 0
        if resume_from and resume_from.exists():
            checkpoint = torch.load(resume_from)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            self.logger.info(f"🔄 从 Epoch {start_epoch} 恢复训练")

        no_improve_count = 0
        
        for epoch in range(start_epoch, num_epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            self.optimizer.zero_grad()
            
            for i, batch in enumerate(self.train_loader):
                if not batch: continue
                
                dyn = batch['dynamic'].to(self.device)
                sta = batch['static'].to(self.device)
                labels = self._get_labels(batch)
                
                # 混合精度上下文
                autocast_ctx = torch.amp.autocast('cuda') if hasattr(torch.amp, 'autocast') else torch.cuda.amp.autocast()

                # [修正] 降低 Mixup 触发概率 (0.5 -> 0.2) 和强度，以减轻欠拟合
                if not debug and np.random.rand() < 0.2:
                    # 显式降低 alpha 为 0.2
                    dyn, sta, targets_a, targets_b, lam = self.mixup_data(dyn, sta, labels, alpha=0.2)
                    with autocast_ctx:
                        outputs = self.model(dyn, sta)
                        loss = lam * self.criterion(outputs['logits'], targets_a) + (1 - lam) * self.criterion(outputs['logits'], targets_b)
                else:
                    with autocast_ctx:
                        outputs = self.model(dyn, sta)
                        loss = self.criterion(outputs['logits'], labels)
                
                loss = loss / accumulation_steps
                self.scaler.scale(loss).backward()
                
                if (i + 1) % accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                
                train_loss += loss.item() * accumulation_steps
                with torch.no_grad():
                    preds = torch.argmax(outputs['logits'], dim=1)
                    train_correct += (preds == labels).sum().item()
                    train_total += labels.size(0)
                
                if debug and i >= 5: break
            
            # 更新学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            scheduler.step()
            
            avg_train_loss = train_loss / len(self.train_loader)
            avg_train_acc = train_correct / train_total if train_total > 0 else 0.0
            val_metrics = self.evaluate(self.val_loader)
            
            self.history['train_loss'].append(avg_train_loss)
            self.history['train_acc'].append(avg_train_acc)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1_macro'])

            history_path = self.output_dir / "training_history.json"
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=4)
            
            log_msg = (
                f"Epoch {epoch+1}/{num_epochs} [LR={current_lr:.6f}]: "
                f"Train Loss={avg_train_loss:.4f} | "
                f"Train Acc={avg_train_acc:.4f} | "
                f"Val Loss={val_metrics['loss']:.4f} | "
                f"Val Acc={val_metrics['accuracy']:.4f} | "
                f"Val F1={val_metrics['f1_macro']:.4f}"
            )

            if self.verbose:
                self.logger.info(log_msg)
            
            if val_metrics['f1_macro'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1_macro']
                self.best_epoch = epoch + 1
                no_improve_count = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_f1': self.best_val_f1
                }, self.output_dir / "best_model.pth")
                self.logger.info(f"💾 保存最佳模型 (F1: {self.best_val_f1:.4f})")
            else:
                no_improve_count += 1
                
            if no_improve_count >= patience:
                self.logger.info(f"🛑 Early stopping at epoch {epoch+1}")
                break
                
        return self.history

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                if not batch: continue
                dyn = batch['dynamic'].to(self.device)
                sta = batch['static'].to(self.device)
                labels = self._get_labels(batch)
                
                outputs = self.model(dyn, sta)
                loss = self.criterion(outputs['logits'], labels)
                total_loss += loss.item()
                
                probs = outputs['probabilities']
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        if len(all_labels) == 0:
            return {'loss': 0, 'accuracy': 0, 'f1_macro': 0, 'preds': [], 'labels': []}

        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        return {
            'loss': total_loss / len(dataloader) if len(dataloader) > 0 else 0,
            'accuracy': accuracy,
            'f1_macro': f1,
            'preds': all_preds,
            'labels': all_labels
        }
    
    def test(self):
        if self.test_loader is None:
            self.logger.warning("⚠️ 没有提供测试集 DataLoader")
            return {}

        best_path = self.output_dir / "best_model.pth"
        if best_path.exists():
            checkpoint = torch.load(best_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"🧪 加载最佳模型 (Epoch {checkpoint['epoch']+1}) 进行测试")
        
        metrics = self.evaluate(self.test_loader)
        cm = confusion_matrix(metrics['labels'], metrics['preds'])
        np.save(self.output_dir / "confusion_matrix.npy", cm)

        report = classification_report(metrics['labels'], metrics['preds'], digits=4, zero_division=0)
        print("\nTest Report:")
        print(report)
        self.logger.info("\nTest Report:\n" + report)

        return metrics