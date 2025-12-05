"""
point_timeseries_dataset.py: 极速版 (加载 .pt 文件) - 修复静态通道检测
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import logging
import json
import numpy as np

class PointTimeSeriesDataset(Dataset):
    def __init__(self, config, encoder, crawler=None, split='train', split_ratio=(0.8, 0.2, 0.0), seed=42, verbose=True, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.processed_dir = config.get_resolved_path('data_dir') / "processed_tensors"
        self.encoder = encoder
        
        # 验证预处理数据是否存在
        if not self.processed_dir.exists():
            raise FileNotFoundError(f"❌ 找不到预处理数据: {self.processed_dir}\n请先运行 code/preprocess_dataset.py")

        # 1. 加载元数据
        meta_path = self.processed_dir / "dataset_metadata.json"
        self.channel_map = {}
        self.num_channels = 0     # 动态通道数
        self.static_channels = [] # 静态通道名
        self.num_static_channels = 0 # 静态通道数

        if meta_path.exists():
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            self.channel_map = meta.get('channel_map', {})
            self.num_channels = meta.get('num_channels', len(self.channel_map))
            self.static_channels = meta.get('static_channels', [])
            self.num_static_channels = len(self.static_channels)
        else:
            if verbose:
                self.logger.warning("⚠️ 未找到 dataset_metadata.json，将通过样本自动推断通道数。")

        # 2. 获取所有可用的样本ID
        all_files = list(self.processed_dir.glob("*.pt"))
        all_indices = []
        for f in all_files:
            if f.stem.isdigit():
                all_indices.append(int(f.stem))
        
        if len(all_indices) == 0:
            raise RuntimeError("预处理目录是空的！")

        # 3. 划分数据集
        self.indices = self._split_indices(all_indices, split, split_ratio, seed)
        if verbose:
            self.logger.info(f"[{split.upper()}] 加载 {len(self.indices)} 个预处理样本")
        
        # 4. 自动推断通道数 (如果元数据缺失或不完整)
        # 即使有元数据，校验一下也是安全的
        if len(self.indices) > 0 and (self.num_channels == 0 or self.num_static_channels == 0):
            try:
                # 显式 weights_only=False 消除警告
                sample_0 = torch.load(self.processed_dir / f"{self.indices[0]}.pt", weights_only=False)
                
                # 动态通道检测
                if self.num_channels == 0:
                    self.num_channels = sample_0['dynamic'].shape[1]
                
                # 静态通道检测
                if self.num_static_channels == 0:
                    if 'static' in sample_0:
                        self.num_static_channels = sample_0['static'].shape[0]
                    else:
                        self.num_static_channels = 0 # 甚至可能是 0
                
                if verbose:
                    self.logger.info(f"🔍 自动检测通道数: Dynamic={self.num_channels}, Static={self.num_static_channels}")
            except Exception as e:
                self.logger.warning(f"⚠️ 无法通过样本推断通道数: {e}")
        
        if encoder:
            self.points_df = encoder.get_geodataframe().reset_index(drop=True)

    def _split_indices(self, available_indices, split, ratio, seed):
        available_indices = np.array(sorted(available_indices))
        n = len(available_indices)
        
        np.random.seed(seed)
        shuffled = np.random.permutation(available_indices)
        
        n_train = int(n * ratio[0])
        n_val = int(n * ratio[1])
        
        if split == 'train': return shuffled[:n_train]
        elif split == 'val': return shuffled[n_train:n_train+n_val]
        else: return shuffled[n_train+n_val:]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_id = self.indices[idx]
        file_path = self.processed_dir / f"{file_id}.pt"
        
        data = torch.load(file_path, weights_only=False)
        
        # 动态生成 Mask
        T = data['dynamic'].shape[0]
        data['mask'] = torch.ones(T, dtype=torch.bool)
        
        return data

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return {}

    max_t = max([b['dynamic'].shape[0] for b in batch])
    C, H, W = batch[0]['dynamic'].shape[1:]
    
    padded_dyn = []
    padded_mask = []
    padded_dates = []
    
    for b in batch:
        t = b['dynamic'].shape[0]
        pad_len = max_t - t
        
        if pad_len > 0:
            p_d = torch.cat([b['dynamic'], torch.zeros(pad_len, C, H, W)], dim=0)
            p_m = torch.cat([b['mask'], torch.zeros(pad_len, dtype=torch.bool)], dim=0)
            p_dt = torch.cat([b['dates'], torch.zeros(pad_len, dtype=torch.long)], dim=0)
        else:
            p_d = b['dynamic']
            p_m = b['mask']
            p_dt = b['dates']
            
        padded_dyn.append(p_d)
        padded_mask.append(p_m)
        padded_dates.append(p_dt)
        
    return {
        'dynamic': torch.stack(padded_dyn),
        'static': torch.stack([b['static'] for b in batch]),
        'label': torch.stack([b['label'] for b in batch]),
        'major_label': torch.stack([b['major_label'] for b in batch]),
        'detail_label': torch.stack([b['detail_label'] for b in batch]),
        'mask': torch.stack(padded_mask),
        'dates': torch.stack(padded_dates),
        'coords': [b['coords'] for b in batch],
        'metadata': [{'sample_id': b['sample_id']} for b in batch] 
    }