"""
model_architecture.py: 双流时空融合网络 (语义分割版)
修改说明：
1. 移除了 Flatten 和 Global Pooling，保留空间维度。
2. 引入 Decoder 模块，将下采样后的特征恢复到原始分辨率。
3. L-TAE 改为 Pixel-wise 模式，对每个像素计算时间注意力。
4. 输出形状从 (B, NumClasses) 变为 (B, NumClasses, H, W)。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

# ============================================================================
# 基础组件：位置编码
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
        # 调整形状以适应 (Batch*H*W, T, C) 或 (Batch, T, C)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, C)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# ============================================================================
# 核心组件：Pixel-wise L-TAE (支持空间特征图)
# ============================================================================

class PixelWiseLTAE(nn.Module):
    """
    像素级轻量级时间注意力编码器
    输入: (B, T, C, H, W)
    输出: (B, C_out, H, W)
    """
    def __init__(self, in_channels: int, temporal_steps: int = 12, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.in_channels = in_channels
        
        # 位置编码处理 (我们将空间展平放到 Batch 维度来处理时间序列)
        self.pos_encoding = PositionalEncoding(d_model=in_channels, max_len=temporal_steps, dropout=dropout)
        
        # 注意力计算 MLP (使用 1x1 卷积代替 Linear 以保留空间处理能力，或者 reshape 后用 Linear)
        # 这里为了效率，我们 reshape 为 (B*H*W, T, C) 统一处理
        self.attention_net = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1), # 输出 Attention Score
        )
        
        # 输出投影 (1x1 Conv)
        self.output_projection = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T, C, H, W = x.shape
        
        # 1. 变形: (B, T, C, H, W) -> (B, H, W, T, C) -> (B*H*W, T, C)
        # 这样我们可以对每个像素独立应用时间注意力
        x_pixel = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)
        
        # 2. 加上位置编码
        x_pos = self.pos_encoding(x_pixel)
        
        # 3. 计算注意力分数: (B*H*W, T, 1) -> (B*H*W, T)
        attn_scores = self.attention_net(x_pos).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1) # 在时间维度归一化
        
        # 4. 加权聚合: (B*H*W, T, 1) * (B*H*W, T, C) -> Sum over T -> (B*H*W, C)
        attn_weights_expanded = attn_weights.unsqueeze(-1)
        weighted_x = x_pixel * attn_weights_expanded
        aggregated = weighted_x.sum(dim=1)
        
        # 5. 还原空间维度: (B*H*W, C) -> (B, H, W, C) -> (B, C, H, W)
        aggregated = aggregated.view(B, H, W, C).permute(0, 3, 1, 2)
        attn_weights_map = attn_weights.view(B, H, W, T).permute(0, 3, 1, 2) # (B, T, H, W)
        
        # 6. 输出投影 (Conv2d 1x1)
        output_features = self.output_projection(aggregated)
        
        return {'aggregated': output_features, 'attention_weights': attn_weights_map}

# ============================================================================
# 空间编码器 (Fully Convolutional)
# ============================================================================

class ResidualBlock(nn.Module):
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
    """
    提取空间特征，带有下采样
    Input: (B, C, H, W) -> Output: (B, Hidden, H/4, W/4)
    """
    def __init__(self, in_channels: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True)
        )
        # 下采样层 (总共下采样 4 倍)
        self.layer1 = ResidualBlock(hidden_dim // 2, hidden_dim, stride=2, dropout=dropout) # H/2
        self.layer2 = ResidualBlock(hidden_dim, hidden_dim, stride=2, dropout=dropout)      # H/4
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x

# ============================================================================
# 分支定义 (保持特征图结构)
# ============================================================================

class DynamicStreamBranch(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, fusion_dim: int, temporal_steps: int, dropout: float):
        super().__init__()
        self.spatial_encoder = SpatialEncoder(in_channels, hidden_dim, dropout)
        # 注意: L-TAE 输入维度现在是 hidden_dim
        self.temporal_encoder = PixelWiseLTAE(hidden_dim, temporal_steps, hidden_dim, dropout)
        
        # 融合前的投影 (使用 1x1 Conv)
        self.projection = nn.Sequential(
            nn.Conv2d(hidden_dim, fusion_dim, kernel_size=1),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T, C, H, W = x.shape
        # 处理每一帧的空间特征
        x_reshaped = x.view(B * T, C, H, W)
        x_spatial = self.spatial_encoder(x_reshaped) # (B*T, hidden, H/4, W/4)
        
        # 恢复时间维度
        _, C_new, H_new, W_new = x_spatial.shape
        x_spatial = x_spatial.view(B, T, C_new, H_new, W_new)
        
        # 时间聚合 (Pixel-wise)
        temp_out = self.temporal_encoder(x_spatial) # (B, hidden, H/4, W/4)
        
        return {'embeddings': self.projection(temp_out['aggregated']), 'attention_weights': temp_out['attention_weights']}

class StaticStreamBranch(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, fusion_dim: int, dropout: float):
        super().__init__()
        self.spatial_encoder = SpatialEncoder(in_channels, hidden_dim, dropout)
        
        self.projection = nn.Sequential(
            nn.Conv2d(hidden_dim, fusion_dim, kernel_size=1),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: (B, C, H, W) -> (B, hidden, H/4, W/4)
        spatial_out = self.spatial_encoder(x)
        return {'embeddings': self.projection(spatial_out)}

# ============================================================================
# 解码器 (Decoder) - 用于语义分割
# ============================================================================

class Decoder(nn.Module):
    """
    简单解码器：将下采样 4 倍的特征图恢复到原始分辨率
    Structure: Upsample -> Conv -> Upsample -> Conv
    """
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        
        # Stage 1: H/4 -> H/2
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
        
        # Stage 2: H/2 -> H
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

    def forward(self, x):
        x = self.up1(x)
        x = self.conv1(x)
        x = self.up2(x)
        x = self.conv2(x)
        return x

class GatedFusion(nn.Module):
    """像素级门控融合 (使用 1x1 Conv)"""
    def __init__(self, fusion_dim: int):
        super().__init__()
        # 使用 1x1 Conv 代替 Linear
        self.gate_net = nn.Sequential(
            nn.Conv2d(fusion_dim * 2, fusion_dim, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, dynamic_emb, static_emb):
        combined = torch.cat([dynamic_emb, static_emb], dim=1)
        z = self.gate_net(combined)
        return z * dynamic_emb + (1 - z) * static_emb

# ============================================================================
# 主模型：双流时空融合分割网络
# ============================================================================

class DualStreamSpatio_TemporalFusionNetwork(nn.Module):
    def __init__(
        self,
        in_channels_dynamic: int,
        in_channels_static: int,
        num_classes: int,
        temporal_steps: int = 12,
        hidden_dim: int = 128, 
        fusion_dim: int = 128,
        dropout: float = 0.1,
        classifier_hidden_dims: list = None, # 这里作为解码器前的中间层
        **kwargs # 兼容旧接口
    ):
        super().__init__()
        self.num_classes = num_classes
        
        # 1. 动态分支
        self.dynamic_stream = DynamicStreamBranch(
            in_channels=in_channels_dynamic, hidden_dim=hidden_dim, fusion_dim=fusion_dim,
            temporal_steps=temporal_steps, dropout=dropout,
        )
        
        # 2. 静态分支
        self.static_stream = StaticStreamBranch(
            in_channels=in_channels_static, hidden_dim=hidden_dim, fusion_dim=fusion_dim,
            dropout=dropout,
        )
        
        # 3. 融合模块
        self.gated_fusion = GatedFusion(fusion_dim)
        
        # 4. 解码器 (Decoder) + 分类头
        # 融合后的特征图大小是 (B, fusion_dim, H/4, W/4)
        # 我们先解码回 (B, 64, H, W)，然后输出分类
        
        decoder_hidden = 64
        self.decoder = Decoder(in_channels=fusion_dim, out_channels=decoder_hidden, dropout=dropout)
        
        # 最终像素分类器 (1x1 Conv)
        self.classifier = nn.Conv2d(decoder_hidden, num_classes, kernel_size=1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
    
    def forward(self, dynamic: torch.Tensor, static: torch.Tensor, return_aux: bool = False) -> Dict[str, torch.Tensor]:
        # 1. 编码
        dyn_out = self.dynamic_stream(dynamic)  # (B, fusion_dim, H/4, W/4)
        sta_out = self.static_stream(static)    # (B, fusion_dim, H/4, W/4)
        
        # 2. 融合
        fused_embedding = self.gated_fusion(dyn_out['embeddings'], sta_out['embeddings'])
        
        # 3. 解码 (上采样)
        decoded_features = self.decoder(fused_embedding) # (B, decoder_hidden, H, W)
        
        # 4. 分类
        logits = self.classifier(decoded_features) # (B, num_classes, H, W)
        
        result = {'logits': logits, 'probabilities': F.softmax(logits, dim=1)}
        
        if return_aux:
            result['auxiliary'] = {
                'dynamic_attention': dyn_out['attention_weights'],
            }
        
        return result

    def get_model_summary(self) -> Dict:
        return {
            'model_name': self.__class__.__name__ + " (Segmentation)",
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'num_classes': self.num_classes
        }