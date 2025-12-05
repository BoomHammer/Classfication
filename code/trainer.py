import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, f1_score, confusion_matrix

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
        label_mapping: dict = None
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
        self.target_key = target_key
        self.label_mapping = label_mapping
        
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        # [修复] 使用 torch.amp.GradScaler 替代 torch.cuda.amp.GradScaler
        self.scaler = torch.amp.GradScaler('cuda')
        self.optimizer = None
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

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
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
            self.optimizer.zero_grad()
            
            for i, batch in enumerate(pbar):
                if not batch: continue
                
                dyn = batch['dynamic'].to(self.device)
                sta = batch['static'].to(self.device)
                labels = self._get_labels(batch)
                
                if not debug and np.random.rand() < 0.5:
                    dyn, sta, targets_a, targets_b, lam = self.mixup_data(dyn, sta, labels)
                    # [修复] 使用 torch.amp.autocast
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(dyn, sta)
                        loss = lam * self.criterion(outputs['logits'], targets_a) + (1 - lam) * self.criterion(outputs['logits'], targets_b)
                else:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(dyn, sta)
                        loss = self.criterion(outputs['logits'], labels)
                
                loss = loss / accumulation_steps
                self.scaler.scale(loss).backward()
                
                if (i + 1) % accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                
                train_loss += loss.item() * accumulation_steps
                pbar.set_postfix({'loss': f"{loss.item() * accumulation_steps:.4f}"})
                
                if debug and i >= 5: break
            
            avg_train_loss = train_loss / len(self.train_loader)
            val_metrics = self.evaluate(self.val_loader)
            
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1_macro'])
            
            if self.verbose:
                print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={val_metrics['loss']:.4f}, Val Acc={val_metrics['accuracy']:.4f}, Val F1={val_metrics['f1_macro']:.4f}")
            
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
        
        print("\nTest Report:")
        print(classification_report(metrics['labels'], metrics['preds'], digits=4, zero_division=0))
        return metrics