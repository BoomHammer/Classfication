import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import logging
import numpy as np
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import logging
import numpy as np
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import json

# ============================================================================
# 标签平滑交叉熵损失 (Label Smoothing) - 支持语义分割
# ============================================================================
class LabelSmoothingLoss(nn.Module):
    """
    带标签平滑的交叉熵损失
    减少模型对预测的过度自信，提高泛化性能
    支持 2D (分类) 和 4D (分割) 输入
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
        pred: (B, C) logits 或 (B, C, H, W) logits
        target: (B,) target 或 (B, H, W) target
        """
        # 处理分割任务的维度: (B, C, H, W) -> (N, C)
        if pred.dim() == 4:
            # [修复] 使用 reshape 替代 view 以兼容非连续张量
            pred = pred.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes)
            target = target.reshape(-1)
            
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
            weight_t = self.weight[target]
            weight_t = weight_t / (weight_t.max() + 1e-8)
            loss = loss * weight_t
        
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
        # inputs: (B, C, H, W) or (B, C)
        # targets: (B, H, W) or (B)
        
        # 计算交叉熵 (reduction='none' 保留维度)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 防止数值溢出
        ce_loss = torch.clamp(ce_loss, min=1e-6, max=100.0)
        
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # 应用权重
        if self.alpha is not None:
            # alpha[targets] 会自动处理 broadcast
            weight_t = self.alpha[targets]
            weight_t = weight_t / (weight_t.max() + 1e-8)
            focal_loss = focal_loss * weight_t

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
        label_smoothing: float = 0.1,
        model_init_params: dict = None
    ):
        self.model = model.to(device)
        self.model_init_params = model_init_params or {}
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
            self.logger.info(f" 🔧使用 Focal Loss (标签平滑={label_smoothing})")
            self.criterion = FocalLoss(alpha=class_weights, gamma=2.0, device=device)
        else:
            self.logger.info(f"🔧 使用 CrossEntropy Loss (标签平滑={label_smoothing})")
            self.criterion = LabelSmoothingLoss(num_classes=num_classes, smoothing=label_smoothing, weight=class_weights, device=device)
            
        self.scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        if hasattr(torch.amp, 'GradScaler'):
             self.scaler = torch.amp.GradScaler('cuda')

        self.optimizer = None
        self.best_val_f1 = 0.0
        self.history = {'train_loss': [], 'train_acc': [], 'train_f1': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

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
            local_labels = []
            for x in cpu_labels:
                if x in self.label_mapping:
                    local_labels.append(self.label_mapping[x])
                else:
                    local_labels.append(min(self.label_mapping.values()))
            local_labels = np.array(local_labels)
            labels = torch.from_numpy(local_labels).to(self.device).long()
        return labels
    
    def _expand_labels_if_needed(self, logits, labels):
        """
        如果模型输出是 4D (分割图)，而标签是 1D (标量)，
        则将标签扩展为 3D (B, H, W)
        """
        if logits.dim() == 4 and labels.dim() == 1:
            B, C, H, W = logits.shape
            # 扩展标签: [B] -> [B, 1, 1] -> [B, H, W]
            return labels.view(B, 1, 1).expand(B, H, W)
        return labels

    def train(self, num_epochs=50, learning_rate=1e-3, weight_decay=1e-4, patience=10, debug=False, resume_from=None, accumulation_steps=1):
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Linear Warmup + Cosine Annealing
        total_steps = num_epochs * len(self.train_loader)
        warmup_steps = len(self.train_loader) * 2
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(num_epochs - current_step / len(self.train_loader)) / float(max(1, num_epochs)))
        
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        start_epoch = 0
        if resume_from and resume_from.exists():
            checkpoint = torch.load(resume_from, weights_only=True)
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
            train_preds = []
            train_labels = []
            
            self.optimizer.zero_grad()
            
            for i, batch in enumerate(self.train_loader):
                if not batch: continue
                
                dyn = batch['dynamic'].to(self.device)
                sta = batch['static'].to(self.device)
                labels = self._get_labels(batch)
                
                autocast_ctx = torch.amp.autocast('cuda') if hasattr(torch.amp, 'autocast') else torch.cuda.amp.autocast()

                if not debug and np.random.rand() < 0.2:
                    # Mixup
                    dyn, sta, targets_a, targets_b, lam = self.mixup_data(dyn, sta, labels, alpha=0.2)
                    with autocast_ctx:
                        outputs = self.model(dyn, sta)
                        logits = outputs['logits']
                        
                        targets_a = self._expand_labels_if_needed(logits, targets_a)
                        targets_b = self._expand_labels_if_needed(logits, targets_b)
                        
                        loss = lam * self.criterion(logits, targets_a) + (1 - lam) * self.criterion(logits, targets_b)
                else:
                    with autocast_ctx:
                        outputs = self.model(dyn, sta)
                        logits = outputs['logits']
                        
                        labels = self._expand_labels_if_needed(logits, labels)
                        
                        loss = self.criterion(logits, labels)
                
                loss = loss / accumulation_steps
                self.scaler.scale(loss).backward()
                
                if (i + 1) % accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                
                train_loss += loss.item() * accumulation_steps
                
                with torch.no_grad():
                    logits = outputs['logits']
                    preds = torch.argmax(logits, dim=1)
                    
                    if preds.dim() == 3 and labels.dim() == 1:
                         # 仅用于统计：将 label 扩展来对比
                         labels_exp = labels.view(-1, 1, 1).expand_as(preds)
                         train_correct += (preds == labels_exp).sum().item()
                         train_total += preds.numel()
                    else:
                         train_correct += (preds == labels).sum().item()
                         train_total += labels.numel()
                    
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
                logits = outputs['logits']
                
                # [修复] 检查并扩展标签
                labels_spatial = self._expand_labels_if_needed(logits, labels)
                
                loss = self.criterion(logits, labels_spatial)
                total_loss += loss.item()
                
                probs = outputs['probabilities']
                preds = torch.argmax(probs, dim=1) # (B, H, W)
                
                # 收集结果用于计算 Metrics
                if preds.dim() == 3:
                     if labels.dim() == 1:
                        labels = labels.view(-1, 1, 1).expand_as(preds)
                     
                     # [关键修复] 使用 reshape 而不是 view，因为 expand 后的张量在 flatten 时可能不连续
                     all_preds.extend(preds.reshape(-1).cpu().numpy())
                     all_labels.extend(labels.reshape(-1).cpu().numpy())
                else:
                     all_preds.extend(preds.cpu().numpy())
                     all_labels.extend(labels.cpu().numpy())
        
        if len(all_labels) == 0:
            return {'loss': 0, 'accuracy': 0, 'f1_macro': 0, 'preds': [], 'labels': []}

        # 转为 numpy
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        
        accuracy = np.mean(y_true == y_pred)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        return {
            'loss': total_loss / len(dataloader) if len(dataloader) > 0 else 0,
            'accuracy': accuracy,
            'f1_macro': f1,
            'preds': [], 
            'labels': []
        }
    
    def test(self):
        if self.test_loader is None:
            self.logger.warning("⚠️ 没有提供测试集 DataLoader")
            return {}

        best_path = self.output_dir / "best_model.pth"
        if best_path.exists():
            checkpoint = torch.load(best_path, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"🧪 加载最佳模型 (Epoch {checkpoint['epoch']+1}) 进行测试")
        
        metrics = self.evaluate(self.test_loader)
        self.logger.info(f"Test Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1_macro']:.4f}")
        return metrics
    
    # ============================================================================
    # Stratified K-Fold 交叉验证
    # ============================================================================
    def train_with_kfold(self, 
                         dataset, 
                         num_epochs=50, 
                         learning_rate=1e-3, 
                         weight_decay=1e-4, 
                         patience=10,
                         n_splits=5,
                         random_state=42,
                         debug=False,
                         accumulation_steps=1,
                         batch_size=None):
        self.logger.info(f"🔄 开始 Stratified {n_splits}-Fold 交叉验证")
        
        if batch_size is None:
            batch_size = self.train_loader.batch_size if self.train_loader else 32
        
        all_labels = []
        for idx in range(len(dataset)):
            batch = dataset[idx]
            label = batch[self.target_key]
            if isinstance(label, torch.Tensor):
                label = label.item()
            all_labels.append(label)
        all_labels = np.array(all_labels)
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        kfold_results = {
            'fold_histories': [],
            'fold_metrics': [],
            'mean_metrics': {},
            'std_metrics': {}
        }
        
        fold_accuracies = []
        fold_f1_scores = []
        fold_losses = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(dataset)), all_labels)):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"⏳ 第 {fold+1}/{n_splits} 折训练")
            self.logger.info(f"{'='*60}")
            self.logger.info(f"   训练集大小: {len(train_idx)}, 验证集大小: {len(val_idx)}")
            
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=getattr(dataset, 'collate_fn', None))
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=getattr(dataset, 'collate_fn', None))
            
            original_train_loader = self.train_loader
            original_val_loader = self.val_loader
            self.train_loader = train_loader
            self.val_loader = val_loader
            
            self.model = self.model.__class__(**self._get_model_init_params()).to(self.device)
            self.best_val_f1 = 0.0
            self.history = {'train_loss': [], 'train_acc': [], 'train_f1': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
            
            fold_output_dir = self.output_dir / f"fold_{fold+1}"
            fold_output_dir.mkdir(parents=True, exist_ok=True)
            original_output_dir = self.output_dir
            self.output_dir = fold_output_dir
            
            try:
                history = self.train(
                    num_epochs=num_epochs,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    patience=patience,
                    debug=debug,
                    accumulation_steps=accumulation_steps
                )
                
                best_path = fold_output_dir / "best_model.pth"
                if best_path.exists():
                    checkpoint = torch.load(best_path, weights_only=True)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                
                val_metrics = self.evaluate(val_loader)
                
                fold_accuracies.append(val_metrics['accuracy'])
                fold_f1_scores.append(val_metrics['f1_macro'])
                fold_losses.append(val_metrics['loss'])
                
                fold_result = {
                    'fold': fold + 1,
                    'val_accuracy': val_metrics['accuracy'],
                    'val_f1': val_metrics['f1_macro'],
                    'val_loss': val_metrics['loss']
                }
                kfold_results['fold_metrics'].append(fold_result)
                kfold_results['fold_histories'].append(history)
                
                self.logger.info(f"✅ 第 {fold+1} 折完成 - Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_macro']:.4f}")
                
            finally:
                self.train_loader = original_train_loader
                self.val_loader = original_val_loader
                self.output_dir = original_output_dir
        
        mean_accuracy = np.mean(fold_accuracies)
        std_accuracy = np.std(fold_accuracies)
        mean_f1 = np.mean(fold_f1_scores)
        std_f1 = np.std(fold_f1_scores)
        mean_loss = np.mean(fold_losses)
        std_loss = np.std(fold_losses)
        
        kfold_results['mean_metrics'] = {
            'accuracy': float(mean_accuracy),
            'accuracy_std': float(std_accuracy),
            'f1_macro': float(mean_f1),
            'f1_macro_std': float(std_f1),
            'loss': float(mean_loss),
            'loss_std': float(std_loss)
        }
        
        kfold_results_path = self.output_dir / "kfold_results.json"
        with open(kfold_results_path, 'w', encoding='utf-8') as f:
            serializable_results = {
                'fold_metrics': kfold_results['fold_metrics'],
                'mean_metrics': kfold_results['mean_metrics']
            }
            json.dump(serializable_results, f, indent=4)
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🎯 K-Fold 结果: Acc {mean_accuracy:.4f}±{std_accuracy:.4f}, F1 {mean_f1:.4f}±{std_f1:.4f}")
        self.logger.info(f"{'='*60}\n")
        
        return kfold_results
    
    def _get_model_init_params(self):
        return self.model_init_params

    def predict_with_ensemble(self, dataloader, n_splits=5, method='voting'):
        # 简单占位
        pass
    # ============================================================================
    # Ensemble 预测（用于 K-Fold 模型）
    # ============================================================================
    def predict_with_ensemble(self, dataloader, n_splits=5, method='voting'):
        """
        使用 K-Fold 训练的多个模型进行 Ensemble 预测
        
        参数:
            dataloader: 预测数据的 DataLoader
            n_splits: K-Fold 的折数，对应保存的模型数量
            method: 预测方法
                - 'voting': 多数投票（分类问题）
                - 'averaging': 概率平均（推荐）
        
        返回:
            ensemble_preds: 集成后的预测标签
            ensemble_probs: 集成后的预测概率
            all_fold_probs: 所有fold的预测概率 (n_folds, batch_size, num_classes)
        """
        self.logger.info(f"🎯 使用 {n_splits} 个模型进行 Ensemble 预测 (method={method})")
        
        all_fold_outputs = []  # 存储所有fold的输出
        
        for fold_idx in range(1, n_splits + 1):
            fold_dir = self.output_dir / f"fold_{fold_idx}"
            model_path = fold_dir / "best_model.pth"
            
            if not model_path.exists():
                self.logger.warning(f"⚠️ 模型文件不存在: {model_path}")
                continue
            
            # 加载当前fold的模型
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # 进行预测
            self.model.eval()
            fold_probs = []
            fold_preds = []
            
            with torch.no_grad():
                for batch in dataloader:
                    if not batch: continue
                    
                    dyn = batch['dynamic'].to(self.device)
                    sta = batch['static'].to(self.device)
                    
                    outputs = self.model(dyn, sta)
                    probs = outputs['probabilities']  # (batch_size, num_classes)
                    
                    fold_probs.append(probs.cpu().numpy())
            
            fold_probs = np.concatenate(fold_probs, axis=0)  # (total_samples, num_classes)
            all_fold_outputs.append(fold_probs)
            
            self.logger.info(f"✅ Fold {fold_idx} 预测完成")
        
        all_fold_outputs = np.array(all_fold_outputs)  # (n_folds, total_samples, num_classes)
        
        if method == 'averaging':
            # 概率平均
            ensemble_probs = np.mean(all_fold_outputs, axis=0)  # (total_samples, num_classes)
        elif method == 'voting':
            # 多数投票
            fold_preds = np.argmax(all_fold_outputs, axis=2)  # (n_folds, total_samples)
            ensemble_preds_list = []
            for sample_idx in range(fold_preds.shape[1]):
                votes = fold_preds[:, sample_idx]
                # 获取投票最多的标签
                unique, counts = np.unique(votes, return_counts=True)
                ensemble_preds_list.append(unique[np.argmax(counts)])
            
            ensemble_preds = np.array(ensemble_preds_list)
            # 转换为概率分布（one-hot）
            ensemble_probs = np.zeros((ensemble_preds.shape[0], self.num_classes))
            for i, pred in enumerate(ensemble_preds):
                ensemble_probs[i, pred] = 1.0
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        # 从概率获取预测标签
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        
        self.logger.info(f"🎯 Ensemble 预测完成")
        
        return ensemble_preds, ensemble_probs, all_fold_outputs
    
    def evaluate_with_ensemble(self, dataloader, n_splits=5, method='averaging'):
        """
        使用 Ensemble 模型在验证/测试集上进行评估
        
        参数:
            dataloader: 验证数据的 DataLoader
            n_splits: K-Fold 的折数
            method: 预测方法（'voting' 或 'averaging'）
        
        返回:
            metrics: 包含准确率、F1等指标的字典
            predictions: 预测结果（包含标签和概率）
        """
        ensemble_preds, ensemble_probs, _ = self.predict_with_ensemble(dataloader, n_splits, method)
        
        # 收集真实标签
        all_labels = []
        with torch.no_grad():
            for batch in dataloader:
                if not batch: continue
                labels = self._get_labels(batch)
                all_labels.extend(labels.cpu().numpy())
        
        all_labels = np.array(all_labels)
        
        # 计算指标
        accuracy = np.mean(ensemble_preds == all_labels)
        f1 = f1_score(all_labels, ensemble_preds, average='macro', zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1,
            'preds': ensemble_preds,
            'probs': ensemble_probs,
            'labels': all_labels
        }
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🎯 Ensemble 模型评估结果 ({method} 方法)")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"准确率: {accuracy:.4f}")
        self.logger.info(f"F1 分数: {f1:.4f}")
        self.logger.info(f"{'='*60}\n")
        
        return metrics