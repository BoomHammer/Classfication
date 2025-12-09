# 遥感分类精度提升分析报告

## 📊 现状诊断

### 1. 当前性能指标 (20251209_1613 运行)
- **大类模型**: F1 ≈ 0.42-0.46 (最高 0.4627)
- **小类模型**: F1 ≈ 0.20-0.30 (显著更低)
- **问题**: 小类精度远低于大类，说明细粒度特征学习不足

### 2. 核心问题识别

#### 问题1: **Loss 函数数值爆炸** ⚠️ 严重
```log
Train Loss=11285664.3824 | Train Acc=0.1397  # 日志中可见
Train Loss=10937920.7059 | Train Acc=0.1397  # 完全失控的Loss
```
**根本原因**: 模型输出的logits未经过正确的数值缩放处理

#### 问题2: **小类样本严重不足**
- 大类内样本分布极不均衡
- 某些小类可能仅有5-10个样本
- 小Batch导致BatchNorm无效且梯度噪声大

#### 问题3: **特征表示能力弱**
- 时间序列数据（12个时步）可能不足以捕捉植被变化
- 动态+静态特征融合方式不够精细
- 空间编码器可能过于简化

#### 问题4: **数据增强缺失**
- 代码中找不到任何数据增强逻辑
- 遥感时序数据最适合的增强方式（时间扭曲、光谱增强等）未实现

#### 问题5: **训练策略不当**
- 学习率调度可能不合理
- Early Stopping的耐心参数可能太低
- Focal Loss强度可能设置不当

---

## 💡 快速修复方案 (优先级排序)

### **方案1: 修复Loss数值爆炸 (最关键)**

**症状**: Loss值达到百万级别

**根本原因**: 
1. 模型输出logits缺乏数值稳定性
2. 可能存在未正确处理的NaN/Inf值
3. 权重初始化不当或梯度爆炸

**修复代码**:

```python
# trainer.py 中修改 Focal Loss 实现
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight
    
    def forward(self, inputs, targets):
        # 1. 数值稳定的Softmax
        inputs = inputs - inputs.max(dim=1, keepdim=True)[0]  # 防止溢出
        
        # 2. 计算概率
        p = F.softmax(inputs, dim=1)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        
        # 3. 获取目标类别的概率
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # 4. 计算Focal Loss
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        # 5. 检查并过滤异常值
        focal_loss = torch.clamp(focal_loss, min=0, max=1e6)
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()
```

**验证**:
```bash
# 训练后检查Loss是否在合理范围 (0.5-3.0)
grep "Train Loss" major_model/train.log | head -20
```

---

### **方案2: 实现时序数据增强**

**为什么必需**: 遥感时序数据量小，增强可大幅提升泛化性能

**实现**: 创建新文件 `code/data_augmentation.py`

```python
import torch
import torch.nn as nn
import numpy as np

class TemporalAugmentation:
    """时间序列增强"""
    
    @staticmethod
    def temporal_warp(x, num_segments=3, max_warp=0.2):
        """时间序列弯曲增强"""
        B, T, C, H, W = x.shape
        t = torch.linspace(0, 1, T, device=x.device)
        
        # 生成随机弯曲
        warp_t = t.clone().unsqueeze(0)
        for _ in range(num_segments):
            segment_start = np.random.randint(0, T-1)
            segment_end = np.random.randint(segment_start+1, T)
            warp_scale = 1 + np.random.uniform(-max_warp, max_warp)
            
            segment_mask = (warp_t >= segment_start/T) & (warp_t <= segment_end/T)
            warp_t[segment_mask] *= warp_scale
        
        # 插值采样
        warp_t = torch.clamp(warp_t, 0, 1)
        warp_indices = (warp_t * (T-1)).long()
        
        return x[:, warp_indices.squeeze(0)]
    
    @staticmethod
    def spectrum_jitter(x, std=0.01):
        """光谱抖动增强"""
        noise = torch.randn_like(x) * std
        return torch.clamp(x + noise, 0, 1)
    
    @staticmethod
    def temporal_dropout(x, drop_rate=0.1):
        """时间段dropout"""
        B, T, C, H, W = x.shape
        drop_frames = int(T * drop_rate)
        
        drop_indices = np.random.choice(T, drop_frames, replace=False)
        mask = torch.ones(T, device=x.device)
        mask[drop_indices] = 0
        
        x_aug = x.clone()
        x_aug[:, drop_indices] = x_aug[:, drop_indices].roll(1, dims=1)
        return x_aug

class PointTimeSeriesDatasetWithAugmentation:
    """在 PointTimeSeriesDataset 基础上添加增强"""
    
    def __init__(self, *args, augmentation_prob=0.5, **kwargs):
        # ... 继承原有初始化 ...
        self.augmentation_prob = augmentation_prob
        self.aug = TemporalAugmentation()
    
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)  # 获取原始样本
        
        if np.random.random() < self.augmentation_prob:
            # 随机选择增强方式
            aug_type = np.random.choice(['warp', 'jitter', 'dropout'])
            
            x_dyn = sample['x_dynamic']
            if aug_type == 'warp':
                x_dyn = self.aug.temporal_warp(x_dyn.unsqueeze(0)).squeeze(0)
            elif aug_type == 'jitter':
                x_dyn = self.aug.spectrum_jitter(x_dyn)
            elif aug_type == 'dropout':
                x_dyn = self.aug.temporal_dropout(x_dyn.unsqueeze(0)).squeeze(0)
            
            sample['x_dynamic'] = x_dyn
        
        return sample
```

**集成到main.py**:
```python
# 在 PointTimeSeriesDataset 初始化后
from data_augmentation import PointTimeSeriesDatasetWithAugmentation

# 替换原来的数据集
full_train_dataset = PointTimeSeriesDatasetWithAugmentation(
    config, encoder, split='train', 
    augmentation_prob=0.5  # 50% 概率增强
)
```

---

### **方案3: 改进小样本处理**

**问题**: 某些小类样本极少，造成训练不稳定

**解决方案A: 样本重采样 (Oversampling)**

```python
# 在 main.py 的小类模型训练部分
from torch.utils.data import WeightedRandomSampler

def create_balanced_sampler(dataset, num_classes):
    """为不平衡数据集创建加权采样器"""
    labels = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        labels.append(sample['label'].item())
    
    class_counts = np.bincount(labels, minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[labels]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )

# 使用方式
sampler = create_balanced_sampler(train_subset, num_sub_classes)
train_loader = DataLoader(
    train_subset,
    batch_size=detail_cfg['batch_size'],
    sampler=sampler,  # 使用加权采样
    collate_fn=collate_fn,
    **common_cfg
)
```

**解决方案B: Mixup混合**

```python
# trainer.py 中添加
def mixup_batch(self, x_dyn, x_sta, y, alpha=0.4):
    """Mixup数据增强"""
    batch_size = y.size(0)
    index = torch.randperm(batch_size)
    
    lam = np.random.beta(alpha, alpha)
    
    mixed_x_dyn = lam * x_dyn + (1 - lam) * x_dyn[index]
    mixed_x_sta = lam * x_sta + (1 - lam) * x_sta[index]
    
    y_a, y_b = y, y[index]
    return mixed_x_dyn, mixed_x_sta, y_a, y_b, lam

# 训练循环中使用
mixed_x_dyn, mixed_x_sta, y_a, y_b, lam = self.mixup_batch(
    x_dyn, x_sta, y
)
outputs = self.model(mixed_x_dyn, mixed_x_sta)
loss = lam * self.criterion(outputs, y_a) + \
       (1 - lam) * self.criterion(outputs, y_b)
```

---

### **方案4: 改进模型架构**

**当前问题**: 空间编码器可能过于简单

**改进方案**: 增强特征提取

```python
# model_architecture.py

class EnhancedSpatialEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, dropout=0.15):
        super().__init__()
        
        # 多尺度卷积分支
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Squeeze-Excitation模块
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim * 3, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim * 3, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.output_projection = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 多尺度特征
        f1 = self.conv1x1(x)
        f3 = self.conv3x3(x)
        f5 = self.conv5x5(x)
        
        # 特征融合
        f_concat = torch.cat([f1, f3, f5], dim=1)
        
        # SE注意力
        se_weights = self.se(f_concat)
        f_weighted = f_concat * se_weights
        
        # 最终投影
        output = self.output_projection(f_weighted)
        
        return output
```

---

### **方案5: 更智能的学习率调度**

**问题**: 固定或简单线性调度可能不适合小样本

**解决方案**: 使用Warmup + CosineAnnealing

```python
# trainer.py
def get_scheduler(optimizer, num_epochs, len_train_loader):
    """创建学习率调度器"""
    total_steps = num_epochs * len_train_loader
    warmup_steps = int(0.1 * total_steps)  # 10% warmup
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / 
                       float(max(1, total_steps - warmup_steps)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda
    )
    return scheduler
```

---

## 🚀 完整实施计划

### 第1阶段 (今天 - 立即): 修复Loss爆炸
```bash
# 1. 修改 trainer.py 中的 FocalLoss 实现
# 2. 重新训练大类模型
# 3. 检查 Loss 曲线是否正常化
```

### 第2阶段 (明天): 添加数据增强
```bash
# 1. 创建 data_augmentation.py
# 2. 集成到 main.py
# 3. 对比有/无增强的训练结果
```

### 第3阶段: 改进小样本处理
```bash
# 1. 实现加权采样器
# 2. 添加 Mixup 增强
# 3. 监控小类精度提升
```

### 第4阶段: 模型架构优化
```bash
# 1. 替换空间编码器
# 2. 调整融合策略
# 3. 重新训练并对比
```

---

## 📈 预期改进

| 方面 | 当前 | 修复后 | 目标 |
|------|------|--------|------|
| 大类F1 | 0.46 | 0.55+ | 0.65+ |
| 小类F1 | 0.25 | 0.40+ | 0.55+ |
| Loss稳定性 | ❌ 爆炸 | ✅ 正常 | ✅ 收敛 |
| 过拟合 | 中等 | 低 | 最小 |

---

## 📝 关键提示

1. **保存baseline**: 在修改前备份当前最佳模型
2. **逐步验证**: 每个修改后都要训练并对比结果
3. **监控指标**: 关注 Loss/F1/Acc 三个维度
4. **小类优先**: 大类已经不错，重点改进小类
5. **数据质量**: 检查CSV标签和影像数据是否有问题

---

## ❓ 可选深度优化

如果上述方案效果不理想，可继续尝试:

1. **分离大小类训练**: 大小类使用不同的学习率和策略
2. **类条件批归一化**: 按类别分组进行BatchNorm
3. **自适应权重**: 根据验证集实时调整类别权重
4. **集合模型**: 多个模型的投票/平均
5. **半监督学习**: 利用未标注数据(如果有的话)

