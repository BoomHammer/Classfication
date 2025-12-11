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
        
        # [改进] 应用类别权重时进行归一化，防止loss爆炸
        if self.weight is not None:
            weight_t = self.weight[target]
            # [关键修复] 归一化权重
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
        # [修复] 计算交叉熵损失，不使用 alpha 权重（权重在 Focal 中已隐含处理）
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # [防护] 限制ce_loss的范围，防止数值溢出
        ce_loss = torch.clamp(ce_loss, min=1e-6, max=100.0)
        
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # [改进] 在这里应用权重，而不是在ce_loss计算中
        if self.alpha is not None:
            weight_t = self.alpha[targets]
            # [关键修复] 归一化权重，防止loss过大
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
        label_smoothing: float = 0.1,  # 新增参数
        model_init_params: dict = None  # 新增：保存模型初始化参数
    ):
        self.model = model.to(device)
        self.model_init_params = model_init_params or {}  # 保存模型初始化参数
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
            # [改进] 确保所有标签都能被正确映射，否则打印警告
            local_labels = []
            for x in cpu_labels:
                if x in self.label_mapping:
                    local_labels.append(self.label_mapping[x])
                else:
                    # [防护] 如果找不到映射，使用映射表中的第一个有效值
                    print(f"⚠️ 警告: 标签 {x} 未在映射表中找到，已跳过或使用默认值")
                    local_labels.append(min(self.label_mapping.values()))
            
            local_labels = np.array(local_labels)
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
            train_preds = []
            train_labels = []
            
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
                    train_preds.extend(preds.cpu().numpy())
                    train_labels.extend(labels.cpu().numpy())
                
                if debug and i >= 5: break
            
            # 更新学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            scheduler.step()
            
            avg_train_loss = train_loss / len(self.train_loader)
            avg_train_acc = train_correct / train_total if train_total > 0 else 0.0
            train_f1 = f1_score(train_labels, train_preds, average='macro', zero_division=0)
            val_metrics = self.evaluate(self.val_loader)
            
            self.history['train_loss'].append(avg_train_loss)
            self.history['train_acc'].append(avg_train_acc)
            self.history['train_f1'].append(train_f1)
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
                f"Train F1={train_f1:.4f} | "
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
        """
        使用 Stratified K-Fold 交叉验证训练模型
        
        参数:
            dataset: 完整的数据集 (PointTimeSeriesDataset)
            num_epochs: 每一折的训练轮数
            learning_rate: 学习率
            weight_decay: 权重衰减
            patience: 早停耐心值
            n_splits: K折数（默认5）
            random_state: 随机种子
            debug: 调试模式
            accumulation_steps: 梯度累积步数
            batch_size: 批大小（如果为None，从self.train_loader获取）
            
        返回:
            kfold_results: 包含所有折的训练结果和平均指标
        """
        self.logger.info(f"🔄 开始 Stratified {n_splits}-Fold 交叉验证")
        
        # 获取 batch_size
        if batch_size is None:
            if self.train_loader is not None:
                batch_size = self.train_loader.batch_size
            else:
                batch_size = 32  # 默认值
        
        # 提取所有标签用于分层
        all_labels = []
        for idx in range(len(dataset)):
            batch = dataset[idx]
            label = batch[self.target_key]
            if isinstance(label, torch.Tensor):
                label = label.item()
            all_labels.append(label)
        all_labels = np.array(all_labels)
        
        # 初始化 Stratified K-Fold
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
            
            # 创建子集
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            
            # 创建 DataLoader
            train_loader = DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=getattr(dataset, 'collate_fn', None)
            )
            val_loader = DataLoader(
                val_subset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=getattr(dataset, 'collate_fn', None)
            )
            
            # 保存原始 DataLoader
            original_train_loader = self.train_loader
            original_val_loader = self.val_loader
            
            # 替换为当前折的 DataLoader
            self.train_loader = train_loader
            self.val_loader = val_loader
            
            # 重置模型和优化器
            self.model = self.model.__class__(**self._get_model_init_params()).to(self.device)
            self.best_val_f1 = 0.0
            self.best_epoch = 0
            self.history = {'train_loss': [], 'train_acc': [], 'train_f1': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
            
            # 为当前折创建输出目录
            fold_output_dir = self.output_dir / f"fold_{fold+1}"
            fold_output_dir.mkdir(parents=True, exist_ok=True)
            original_output_dir = self.output_dir
            self.output_dir = fold_output_dir
            
            try:
                # 训练当前折
                history = self.train(
                    num_epochs=num_epochs,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    patience=patience,
                    debug=debug,
                    accumulation_steps=accumulation_steps
                )
                
                # 评估当前折
                best_path = fold_output_dir / "best_model.pth"
                if best_path.exists():
                    checkpoint = torch.load(best_path)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                
                val_metrics = self.evaluate(val_loader)
                
                fold_accuracies.append(val_metrics['accuracy'])
                fold_f1_scores.append(val_metrics['f1_macro'])
                fold_losses.append(val_metrics['loss'])
                
                fold_result = {
                    'fold': fold + 1,
                    'train_history': history,
                    'val_accuracy': val_metrics['accuracy'],
                    'val_f1': val_metrics['f1_macro'],
                    'val_loss': val_metrics['loss']
                }
                kfold_results['fold_metrics'].append(fold_result)
                kfold_results['fold_histories'].append(history)
                
                self.logger.info(f"✅ 第 {fold+1} 折完成 - Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_macro']:.4f}")
                
            finally:
                # 恢复原始 DataLoader 和输出目录
                self.train_loader = original_train_loader
                self.val_loader = original_val_loader
                self.output_dir = original_output_dir
        
        # 计算平均指标
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
        
        kfold_results['std_metrics'] = {
            'accuracy_std': float(std_accuracy),
            'f1_macro_std': float(std_f1),
            'loss_std': float(std_loss)
        }
        
        # 保存 K-Fold 结果
        kfold_results_path = self.output_dir / "kfold_results.json"
        with open(kfold_results_path, 'w', encoding='utf-8') as f:
            # 只保存可序列化的部分
            serializable_results = {
                'fold_metrics': kfold_results['fold_metrics'],
                'mean_metrics': kfold_results['mean_metrics'],
                'std_metrics': kfold_results['std_metrics']
            }
            json.dump(serializable_results, f, indent=4)
        
        # 打印最终结果
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🎯 K-Fold 交叉验证最终结果 ({n_splits}-Fold)")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"平均准确率: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        self.logger.info(f"平均 F1 分数: {mean_f1:.4f} ± {std_f1:.4f}")
        self.logger.info(f"平均损失: {mean_loss:.4f} ± {std_loss:.4f}")
        self.logger.info(f"{'='*60}\n")
        
        return kfold_results
    
    def _get_model_init_params(self):
        """获取模型初始化参数（用于重新初始化）"""
        return self.model_init_params

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
            checkpoint = torch.load(model_path, map_location=self.device)
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