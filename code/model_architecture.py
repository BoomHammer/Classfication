"""
model_architecture.py: 双流时空融合网络 (Dual-Stream Spatio-Temporal Fusion Network)

【架构设计】
这是一个通用的分类骨干网络。
在“先分大类，后分小类”的策略中，这个同一个网络结构会被实例化多次：
1. 一次用于大类分类器（输出层节点数 = 大类数量）
2. 多次用于各个大类下的小类分类器（输出层节点数 = 该大类下的小类数量）

输入：
  - dynamic: (B, T, C, H, W)    # 动态影像时间序列
  - static: (B, S, H, W)        # 静态影像
输出：
  - logits: (B, num_classes)    # 类别概率
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

# ============================================================================
# 基础组件：位置编码、L-TAE、空间编码器 (保持不变)
# ============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class LightweightTemporalAttentionEncoder(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, temporal_steps: int = 12, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.pos_encoding = PositionalEncoding(d_model=in_channels, max_len=temporal_steps, dropout=dropout)
        self.temporal_attention_net = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.output_projection = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T, C, H, W = x.shape
        x_global = x.mean(dim=[3, 4], keepdim=False)
        x_flat = x_global.reshape(-1, C)
        x_pos = self.pos_encoding(x_flat.unsqueeze(1)).squeeze(1).reshape(B, T, C)
        
        attention_scores = self.temporal_attention_net(x_pos).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        weighted_x = x * attention_weights.reshape(B, T, 1, 1, 1)
        aggregated = weighted_x.sum(dim=1)
        output_features = self.output_projection(aggregated)
        
        return {'aggregated': output_features, 'attention_weights': attention_weights}

class SpatialEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 64, patch_size: int = 64, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2), nn.ReLU(inplace=True), nn.Dropout2d(dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True), nn.Dropout2d(dropout),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True), nn.Dropout2d(dropout),
        )
        self.residual_1x1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=4)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        return x3 + self.residual_1x1(x)

# ============================================================================
# 分支定义
# ============================================================================

class DynamicStreamBranch(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, fusion_dim: int, patch_size: int, temporal_steps: int, dropout: float):
        super().__init__()
        self.spatial_encoder = SpatialEncoder(in_channels, hidden_dim, patch_size, dropout)
        self.temporal_encoder = LightweightTemporalAttentionEncoder(hidden_dim, patch_size // 4, temporal_steps, hidden_dim, dropout)
        self.fusion_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(hidden_dim, fusion_dim), nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T, C, H, W = x.shape
        x_spatial = torch.stack([self.spatial_encoder(x[:, t]) for t in range(T)], dim=1)
        temp_out = self.temporal_encoder(x_spatial)
        return {'embeddings': self.fusion_net(temp_out['aggregated']), 'attention_weights': temp_out['attention_weights']}

class StaticStreamBranch(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, fusion_dim: int, patch_size: int, dropout: float):
        super().__init__()
        self.spatial_encoder = SpatialEncoder(in_channels, hidden_dim, patch_size, dropout)
        self.fusion_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(hidden_dim, fusion_dim), nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {'embeddings': self.fusion_net(self.spatial_encoder(x))}

# ============================================================================
# 主模型：双流时空融合网络
# ============================================================================

class DualStreamSpatio_TemporalFusionNetwork(nn.Module):
    """
    标准的双流时空融合网络。
    可用于大类分类（num_classes = 大类数），
    也可用于小类分类（num_classes = 某大类下的小类数）。
    """
    
    def __init__(
        self,
        in_channels_dynamic: int,
        in_channels_static: int,
        num_classes: int,
        patch_size: int = 64,
        temporal_steps: int = 12,
        hidden_dim: int = 64,
        fusion_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # 动态分支
        self.dynamic_stream = DynamicStreamBranch(
            in_channels=in_channels_dynamic, hidden_dim=hidden_dim, fusion_dim=fusion_dim,
            patch_size=patch_size, temporal_steps=temporal_steps, dropout=dropout,
        )
        
        # 静态分支
        self.static_stream = StaticStreamBranch(
            in_channels=in_channels_static, hidden_dim=hidden_dim, fusion_dim=fusion_dim,
            patch_size=patch_size, dropout=dropout,
        )
        
        # 融合与分类头
        self.fusion_head = nn.Sequential(
            nn.Linear(2 * fusion_dim, fusion_dim),
            nn.GroupNorm(8, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, dynamic: torch.Tensor, static: torch.Tensor, return_aux: bool = False) -> Dict[str, torch.Tensor]:
        dyn_out = self.dynamic_stream(dynamic)
        sta_out = self.static_stream(static)
        
        combined = torch.cat([dyn_out['embeddings'], sta_out['embeddings']], dim=1)
        logits = self.fusion_head(combined)
        
        result = {'logits': logits, 'probabilities': F.softmax(logits, dim=1)}
        
        if return_aux:
            result['auxiliary'] = {
                'dynamic_attention': dyn_out['attention_weights'],
            }
        
        return result

    def get_model_summary(self) -> Dict:
        return {
            'model_name': self.__class__.__name__,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'num_classes': self.num_classes
        }