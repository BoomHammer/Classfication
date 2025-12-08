"""
model_architecture.py: 双流时空融合网络 (Dual-Stream Spatio-Temporal Fusion Network) - 改进版
改进点：
1. 引入 Residual Block 增强 SpatialEncoder
2. 引入 Gated Fusion 机制优化动静态特征融合
3. [本次修正] L-TAE 使用中心像素计算注意力，避免背景噪声干扰
4. [本次修正] 调整默认 Dropout 防止欠拟合
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

# ============================================================================
# 基础组件：位置编码、L-TAE
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
    def __init__(self, in_channels: int, patch_size: int, temporal_steps: int = 12, hidden_dim: int = 64, dropout: float = 0.1):
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
        
        # [改进] 使用中心区域特征来计算时间注意力，而不是全图平均
        # 避免边缘的异质背景（如农田、裸地）干扰对中心目标（如草甸）的判断
        center_h, center_w = H // 2, W // 2
        # 取中心像素特征 (B, T, C)
        x_center = x[:, :, :, center_h, center_w]
        
        # 将中心特征用于计算 Attention Score
        x_flat = x_center.reshape(-1, C)
        x_pos = self.pos_encoding(x_flat.unsqueeze(1)).squeeze(1).reshape(B, T, C)
        
        attention_scores = self.temporal_attention_net(x_pos).squeeze(-1) # (B, T)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Temporal Aggregation: 使用计算出的权重对整个 Patch 进行加权
        weighted_x = x * attention_weights.reshape(B, T, 1, 1, 1)
        aggregated = weighted_x.sum(dim=1) # (B, C, H, W)
        output_features = self.output_projection(aggregated)
        
        return {'aggregated': output_features, 'attention_weights': attention_weights}

# ============================================================================
# 改进组件：ResBlock Spatial Encoder & Gated Fusion
# ============================================================================

class ResidualBlock(nn.Module):
    """标准的 ResNet BasicBlock"""
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class SpatialEncoder(nn.Module):
    """改进后的空间编码器，使用残差结构"""
    def __init__(self, in_channels: int, hidden_dim: int = 64, patch_size: int = 64, dropout: float = 0.1):
        super().__init__()
        # 初始卷积
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True)
        )
        # 残差层
        self.layer1 = ResidualBlock(hidden_dim // 2, hidden_dim, stride=2, dropout=dropout)
        self.layer2 = ResidualBlock(hidden_dim, hidden_dim, stride=2, dropout=dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class GatedFusion(nn.Module):
    """门控融合模块：自动学习动态和静态特征的权重"""
    def __init__(self, fusion_dim: int):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Sigmoid()
        )
        
    def forward(self, dynamic_emb, static_emb):
        # dynamic_emb: (B, fusion_dim)
        # static_emb:  (B, fusion_dim)
        combined = torch.cat([dynamic_emb, static_emb], dim=1)
        z = self.gate_net(combined)
        # 融合: z * dynamic + (1-z) * static
        return z * dynamic_emb + (1 - z) * static_emb

# ============================================================================
# 分支定义
# ============================================================================

class DynamicStreamBranch(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, fusion_dim: int, patch_size: int, temporal_steps: int, dropout: float):
        super().__init__()
        self.spatial_encoder = SpatialEncoder(in_channels, hidden_dim, patch_size, dropout)
        # 注意：SpatialEncoder 现在输出通道数是 hidden_dim
        self.temporal_encoder = LightweightTemporalAttentionEncoder(hidden_dim, patch_size // 4, temporal_steps, hidden_dim, dropout)
        self.fusion_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(),
            nn.Linear(hidden_dim, fusion_dim), 
            nn.LayerNorm(fusion_dim), # 改用 LayerNorm 适应 Flatten 后的特征
            nn.ReLU(inplace=True), 
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T, C, H, W = x.shape
        # 共享空间编码器权重处理所有时间步
        # (B*T, C, H, W) -> (B*T, hidden_dim, H', W')
        x_reshaped = x.view(B * T, C, H, W)
        x_spatial = self.spatial_encoder(x_reshaped)
        _, C_new, H_new, W_new = x_spatial.shape
        x_spatial = x_spatial.view(B, T, C_new, H_new, W_new)
        
        temp_out = self.temporal_encoder(x_spatial)
        return {'embeddings': self.fusion_net(temp_out['aggregated']), 'attention_weights': temp_out['attention_weights']}

class StaticStreamBranch(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, fusion_dim: int, patch_size: int, dropout: float):
        super().__init__()
        self.spatial_encoder = SpatialEncoder(in_channels, hidden_dim, patch_size, dropout)
        self.fusion_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(),
            nn.Linear(hidden_dim, fusion_dim), 
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True), 
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {'embeddings': self.fusion_net(self.spatial_encoder(x))}

# ============================================================================
# 主模型：双流时空融合网络
# ============================================================================

class DualStreamSpatio_TemporalFusionNetwork(nn.Module):
    def __init__(
        self,
        in_channels_dynamic: int,
        in_channels_static: int,
        num_classes: int,
        patch_size: int = 64,
        temporal_steps: int = 12,
        hidden_dim: int = 128, # [默认值建议] 保持足够容量
        fusion_dim: int = 128,
        dropout: float = 0.1,  # [默认值建议] 降低默认 Dropout 以缓解欠拟合 (原0.2/0.3)
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
        
        # 门控融合
        self.gated_fusion = GatedFusion(fusion_dim)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim), 
            nn.BatchNorm1d(fusion_dim),
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
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, dynamic: torch.Tensor, static: torch.Tensor, return_aux: bool = False) -> Dict[str, torch.Tensor]:
        dyn_out = self.dynamic_stream(dynamic)
        sta_out = self.static_stream(static)
        
        # 使用门控融合替代简单的 concat
        fused_embedding = self.gated_fusion(dyn_out['embeddings'], sta_out['embeddings'])
        
        logits = self.classifier(fused_embedding)
        
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