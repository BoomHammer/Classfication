"""
model_architecture.py: 双流时空融合网络 (Dual-Stream Spatio-Temporal Fusion Network)

【架构设计】

该模块实现了一个针对遥感影像时间序列分类的先进神经网络架构。核心创新包括：

1. 轻量级时间注意力编码器 (L-TAE, Lightweight Temporal Attention Encoder)
   - 自动学习不同时间步的重要性
   - 例如：收获季节的影像权重高于休耕期
   - 相比 Transformer 计算量更小，相比 LSTM/GRU 更易并行化

2. 双流架构 (Dual-Stream Architecture)
   - 动态流：处理随季节变化的特征（如植被指数、湿度）
   - 静态流：处理不变的背景特征（如地形高程、坡度）
   - 特征融合层：级联融合两路特征

3. 空间-时间分解设计 (Spatial-Temporal Factorization)
   - 空间编码器：共享权重的CNN处理每个时间步
   - 时间编码器：L-TAE学习时间依赖性
   - 分解优势：减少参数量 + 提高泛化性能

【理论基础】

L-TAE 论文：
  "Lightweight Temporal Attention Encoding for Satellite Image Time Series 
   Land Cover Classification" (Marc Rußwurm et al., 2021)
  https://arxiv.org/abs/2103.03941

核心思想：
  对于时间序列 {f_1, f_2, ..., f_T}，传统注意力计算所有T² 对交互。
  L-TAE 改进：
  1. 使用位置编码 (Positional Encoding) 编码时间位置
  2. 单头注意力 (Single-Head Attention) 而非多头，减少参数
  3. 轻量级设计：仅学习时间权重，不学习特征转换

【输入输出格式】

输入：
  - dynamic: (B, T, C, H, W)    # 动态影像时间序列
  - static: (B, S, H, W)        # 静态影像
  其中 B=批大小, T=时间步数, C=动态通道数, S=静态通道数, H/W=空间分辨率

输出：
  - logits: (B, num_classes)    # 类别概率
  - aux: dict                   # 辅助信息（如注意力权重）

【关键参数（从配置文件自动检测）】

- in_channels_dynamic: 动态影像通道数（由RasterCrawler检测）
- in_channels_static: 静态影像通道数（由RasterCrawler检测）
- num_classes: 类别总数（由LabelEncoder检测）
- temporal_steps: 时间步数（固定为12个月）
- patch_size: 空间分辨率（固定为64）
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


# ============================================================================
# 位置编码 (Positional Encoding)
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    位置编码模块，用于编码时间维度信息
    """
    
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 初始化位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)
        )
        
        # 填充位置编码
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置用正弦
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])  # 奇数位置用余弦
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        # 注册为缓冲区（不作为模型参数）
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ============================================================================
# 轻量级时间注意力编码器 (L-TAE)
# ============================================================================

class LightweightTemporalAttentionEncoder(nn.Module):
    """
    轻量级时间注意力编码器 (L-TAE)
    """
    
    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        temporal_steps: int = 12,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.temporal_steps = temporal_steps
        self.hidden_dim = hidden_dim
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(
            d_model=in_channels,
            max_len=temporal_steps,
            dropout=dropout
        )
        
        # 时间注意力学习模块
        self.temporal_attention_net = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T, C, H, W = x.shape
        
        # 步骤1：计算全局时间权重
        x_global = x.mean(dim=[3, 4], keepdim=False)  # (B, T, C)
        
        # 步骤2：添加位置编码
        x_flat = x_global.reshape(-1, C)  # (B*T, C)
        x_pos = self.pos_encoding(x_flat.unsqueeze(1)).squeeze(1)  # (B*T, C)
        x_pos = x_pos.reshape(B, T, C)  # (B, T, C)
        
        # 步骤3：学习时间注意力权重
        attention_scores = self.temporal_attention_net(x_pos)  # (B, T, 1)
        attention_scores = attention_scores.squeeze(-1)  # (B, T)
        attention_weights = F.softmax(attention_scores, dim=1)  # (B, T)
        
        # 步骤4：使用注意力权重聚合时间维度
        weighted_x = x * attention_weights.reshape(B, T, 1, 1, 1)  # (B, T, C, H, W)
        aggregated = weighted_x.sum(dim=1)  # (B, C, H, W)
        
        # 步骤5：特征投影
        output_features = self.output_projection(aggregated)  # (B, hidden_dim, H, W)
        
        return {
            'aggregated': output_features,
            'attention_weights': attention_weights,
            'intermediate': aggregated,
        }


# ============================================================================
# 空间编码器 (Spatial Encoder)
# ============================================================================

class SpatialEncoder(nn.Module):
    """
    空间特征编码器
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        patch_size: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        
        self.residual_1x1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=4)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        residual = self.residual_1x1(x)
        output = x3 + residual
        return output


# ============================================================================
# 动态流分支 (Dynamic Stream)
# ============================================================================

class DynamicStreamBranch(nn.Module):
    """
    动态流分支
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        fusion_dim: int = 128,
        patch_size: int = 64,
        temporal_steps: int = 12,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.fusion_dim = fusion_dim
        self.temporal_steps = temporal_steps
        
        self.spatial_encoder = SpatialEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            patch_size=patch_size,
            dropout=dropout,
        )
        
        self.temporal_encoder = LightweightTemporalAttentionEncoder(
            in_channels=hidden_dim,
            patch_size=patch_size // 4,
            temporal_steps=temporal_steps,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        
        self.fusion_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T, C, H, W = x.shape
        
        x_spatial = []
        for t in range(T):
            feat = self.spatial_encoder(x[:, t, :, :, :])
            x_spatial.append(feat)
        
        x_spatial = torch.stack(x_spatial, dim=1)
        
        temporal_output = self.temporal_encoder(x_spatial)
        x_temporal = temporal_output['aggregated']
        attention_weights = temporal_output['attention_weights']
        
        embeddings = self.fusion_net(x_temporal)
        
        return {
            'embeddings': embeddings,
            'spatial_features': x_spatial,
            'temporal_features': x_temporal,
            'attention_weights': attention_weights,
        }


# ============================================================================
# 静态流分支 (Static Stream)
# ============================================================================

class StaticStreamBranch(nn.Module):
    """
    静态流分支
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        fusion_dim: int = 128,
        patch_size: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.fusion_dim = fusion_dim
        
        self.spatial_encoder = SpatialEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            patch_size=patch_size,
            dropout=dropout,
        )
        
        self.fusion_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        spatial_features = self.spatial_encoder(x)
        embeddings = self.fusion_net(spatial_features)
        
        return {
            'embeddings': embeddings,
            'spatial_features': spatial_features,
        }


# ============================================================================
# 双流融合与分类头
# ============================================================================

class DualStreamSpatio_TemporalFusionNetwork(nn.Module):
    """
    双流时空融合网络 (DSSTFN) - 标准版
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
        
        self.in_channels_dynamic = in_channels_dynamic
        self.in_channels_static = in_channels_static
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.temporal_steps = temporal_steps
        self.hidden_dim = hidden_dim
        self.fusion_dim = fusion_dim
        
        self.dynamic_stream = DynamicStreamBranch(
            in_channels=in_channels_dynamic,
            hidden_dim=hidden_dim,
            fusion_dim=fusion_dim,
            patch_size=patch_size,
            temporal_steps=temporal_steps,
            dropout=dropout,
        )
        
        self.static_stream = StaticStreamBranch(
            in_channels=in_channels_static,
            hidden_dim=hidden_dim,
            fusion_dim=fusion_dim,
            patch_size=patch_size,
            dropout=dropout,
        )
        
        combined_dim = 2 * fusion_dim
        
        self.fusion_head = nn.Sequential(
            nn.Linear(combined_dim, fusion_dim),
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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        dynamic: torch.Tensor,
        static: torch.Tensor,
        return_aux: bool = False,
    ) -> Dict[str, torch.Tensor]:
        
        dynamic_output = self.dynamic_stream(dynamic)
        dynamic_embeddings = dynamic_output['embeddings']
        
        static_output = self.static_stream(static)
        static_embeddings = static_output['embeddings']
        
        combined_features = torch.cat(
            [dynamic_embeddings, static_embeddings],
            dim=1
        )
        
        logits = self.fusion_head(combined_features)
        probabilities = F.softmax(logits, dim=1)
        
        result = {
            'logits': logits,
            'probabilities': probabilities,
        }
        
        if return_aux:
            result['auxiliary'] = {
                'dynamic_attention_weights': dynamic_output['attention_weights'],
                'dynamic_embeddings': dynamic_embeddings,
                'static_embeddings': static_embeddings,
                'dynamic_spatial_features': dynamic_output['spatial_features'],
                'static_spatial_features': static_output['spatial_features'],
                'dynamic_temporal_features': dynamic_output['temporal_features'],
            }
        
        return result
    
    def get_model_summary(self) -> Dict:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'configuration': {
                'in_channels_dynamic': self.in_channels_dynamic,
                'in_channels_static': self.in_channels_static,
                'num_classes': self.num_classes,
                'patch_size': self.patch_size,
                'temporal_steps': self.temporal_steps,
                'hidden_dim': self.hidden_dim,
                'fusion_dim': self.fusion_dim,
            }
        }


def create_model(in_channels_dynamic, in_channels_static, num_classes, config=None):
    if config is None: config = {}
    return DualStreamSpatio_TemporalFusionNetwork(
        in_channels_dynamic, in_channels_static, num_classes,
        patch_size=config.get('patch_size', 64),
        temporal_steps=config.get('temporal_steps', 12),
        hidden_dim=config.get('hidden_dim', 64),
        fusion_dim=config.get('fusion_dim', 128),
        dropout=config.get('dropout', 0.2),
    )


# ============================================================================
# 分层分类模型 (Hierarchical Classification Network)
# ============================================================================

class HierarchicalDualStreamNetwork(nn.Module):
    """
    分层分类双流网络
    
    【修复说明】
    1. 修正了输出 Logits 的形状问题。现在返回 [Batch, Total_Detail_Classes] 的全局 Logits。
       未被选中的小类位置填充为 -inf，以保证 CrossEntropyLoss 正常工作。
    2. 支持 Teacher Forcing：在训练时使用真实的大类标签进行路由。
    """
    
    def __init__(
        self,
        in_channels_dynamic: int,
        in_channels_static: int,
        hierarchical_map: Dict,
        patch_size: int = 64,
        temporal_steps: int = 12,
        hidden_dim: int = 64,
        fusion_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.in_channels_dynamic = in_channels_dynamic
        self.in_channels_static = in_channels_static
        self.hierarchical_map = hierarchical_map
        self.patch_size = patch_size
        self.temporal_steps = temporal_steps
        self.hidden_dim = hidden_dim
        self.fusion_dim = fusion_dim
        
        # =====================================================================
        # 分析层级结构
        # =====================================================================
        self.num_major_classes = len(hierarchical_map)
        self.major_ids_sorted = sorted(hierarchical_map.keys())
        self.major_id_to_idx = {major_id: idx for idx, major_id in enumerate(self.major_ids_sorted)}
        self.major_idx_to_id = {idx: major_id for idx, major_id in enumerate(self.major_ids_sorted)}
        
        self.num_detail_classes_per_major = {}
        self.has_single_detail_class = {}
        
        # 计算总的小类数量，用于构建全局 Logits
        all_detail_ids = set()
        
        for major_id, major_info in hierarchical_map.items():
            detail_classes = major_info.get('detail_classes', {})
            num_detail = len(detail_classes)
            self.num_detail_classes_per_major[major_id] = num_detail
            self.has_single_detail_class[major_id] = (num_detail == 1)
            all_detail_ids.update(detail_classes.values())
            
        self.total_detail_classes = len(all_detail_ids) if all_detail_ids else 0
        print(f"[HierarchicalNetwork] 总小类数量: {self.total_detail_classes}")
        
        # =====================================================================
        # 构建共享编码器
        # =====================================================================
        self.dynamic_stream = DynamicStreamBranch(
            in_channels=in_channels_dynamic,
            hidden_dim=hidden_dim,
            fusion_dim=fusion_dim,
            patch_size=patch_size,
            temporal_steps=temporal_steps,
            dropout=dropout,
        )
        
        self.static_stream = StaticStreamBranch(
            in_channels=in_channels_static,
            hidden_dim=hidden_dim,
            fusion_dim=fusion_dim,
            patch_size=patch_size,
            dropout=dropout,
        )
        
        combined_dim = 2 * fusion_dim
        
        # =====================================================================
        # 构建分类器
        # =====================================================================
        # 第一级：大类
        self.classifier_level1 = nn.Sequential(
            nn.Linear(combined_dim, fusion_dim),
            nn.GroupNorm(8, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, self.num_major_classes),
        )
        
        # 第二级：小类（列表）
        self.classifier_level2_list = nn.ModuleList()
        
        for major_id in sorted(hierarchical_map.keys()):
            num_detail = self.num_detail_classes_per_major[major_id]
            
            if num_detail == 1:
                self.classifier_level2_list.append(None)
            else:
                classifier = nn.Sequential(
                    nn.Linear(combined_dim, fusion_dim),
                    nn.GroupNorm(8, fusion_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(fusion_dim, num_detail),
                )
                self.classifier_level2_list.append(classifier)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        dynamic: torch.Tensor,
        static: torch.Tensor,
        major_labels: Optional[torch.Tensor] = None,  # 新增：用于 Teacher Forcing
        return_aux: bool = False,
    ) -> Dict[str, torch.Tensor]:
        
        batch_size = dynamic.size(0)
        device = dynamic.device
        
        # 步骤1：编码
        dynamic_output = self.dynamic_stream(dynamic)
        dynamic_embeddings = dynamic_output['embeddings']
        
        static_output = self.static_stream(static)
        static_embeddings = static_output['embeddings']
        
        combined_features = torch.cat(
            [dynamic_embeddings, static_embeddings],
            dim=1
        )
        
        # 步骤2：第一级分类
        major_logits = self.classifier_level1(combined_features)
        major_probs = F.softmax(major_logits, dim=1)
        major_preds = major_logits.argmax(dim=1)
        
        # 步骤3：第二级分类（关键修正）
        # 确定路由索引：训练时使用 Teacher Forcing (如有)，否则使用预测值
        if self.training and major_labels is not None:
            routing_indices = major_labels
        else:
            routing_indices = major_preds
        
        # 创建全局 Logits 张量，初始化为极小值（相当于概率为0）
        # 形状为 [B, Total_Detail_Classes]，可以与全局权重 [Total] 兼容
        global_detail_logits = torch.full(
            (batch_size, self.total_detail_classes), 
            -1e9, 
            device=device
        )
        
        detail_preds_list = []
        
        for b in range(batch_size):
            # 获取当前样本的大类索引（在 classifier_list 中的索引）
            major_idx = routing_indices[b].item()
            
            # 如果是预测值，需要确保不越界（虽然argmax不会越界，但防万一）
            if major_idx >= len(self.major_idx_to_id):
                major_idx = 0
                
            major_id = self.major_idx_to_id[major_idx]
            
            # 获取该大类对应的真实小类全局ID列表
            detail_classes_map = self.hierarchical_map[major_id]['detail_classes']
            # 注意：分类器的输出是按照 key 排序还是 value 排序？
            # 这里的约定是：detail_classes 的 value 是全局 ID
            # 假设 LabelEncoder 是按名称排序分配的，或者是按某种顺序。
            # 这里的关键是：分类器输出的第 i 个节点对应哪个全局 ID？
            # 我们假设分类器是按照 detail_classes.keys() 排序后的顺序输出的
            sorted_detail_names = sorted(detail_classes_map.keys(), key=lambda x: detail_classes_map[x])
            current_global_ids = [detail_classes_map[name] for name in sorted_detail_names]
            
            if self.has_single_detail_class[major_id]:
                # 只有1个小类，直接给该 ID 一个高置信度
                unique_id = current_global_ids[0]
                global_detail_logits[b, unique_id] = 10.0  # Logit = 10.0 -> Sigmoid ~ 1.0
                detail_preds_list.append(unique_id)
            else:
                # 多个小类，运行分类器
                classifier_idx = self.major_id_to_idx[major_id]
                classifier = self.classifier_level2_list[classifier_idx]
                
                # 推理
                sample_features = combined_features[b:b+1]
                local_logits = classifier(sample_features)  # (1, num_sub_classes)
                
                # 将局部 Logits 映射到全局 Logits 张量
                # local_logits[0, i] 对应 current_global_ids[i]
                for i, global_id in enumerate(current_global_ids):
                    global_detail_logits[b, global_id] = local_logits[0, i]
                
                # 预测
                local_pred = local_logits.argmax(dim=1).item()
                global_pred = current_global_ids[local_pred]
                detail_preds_list.append(global_pred)
        
        detail_preds = torch.tensor(detail_preds_list, dtype=torch.long, device=device)
        
        result = {
            'major_logits': major_logits,
            'major_probs': major_probs,
            'major_preds': major_preds,
            'detail_logits': global_detail_logits,  # 现在是 [B, 32]
            'detail_preds': detail_preds,
        }
        
        if return_aux:
            result['auxiliary'] = {
                'dynamic_attention_weights': dynamic_output['attention_weights'],
                'dynamic_embeddings': dynamic_embeddings,
                'static_embeddings': static_embeddings,
            }
        
        return result

    def get_model_summary(self) -> Dict:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'configuration': {
                'in_channels_dynamic': self.in_channels_dynamic,
                'in_channels_static': self.in_channels_static,
                'num_major_classes': self.num_major_classes,
                'total_detail_classes': self.total_detail_classes,
                'hidden_dim': self.hidden_dim,
                'fusion_dim': self.fusion_dim,
            }
        }
