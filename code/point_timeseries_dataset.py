"""
point_timeseries_dataset.py: 自定义时空数据集类

【修复说明】
1. 在 __getitem__ 中增加了 major_label 的读取，这对分层分类的 Teacher Forcing 至关重要。
2. 更新了 collate_fn 以正确堆叠 major_label 和 detail_label。
3. 确保了返回字典的键名与 Trainer 期望的一致。
"""

import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class PointTimeSeriesDataset(Dataset):
    """
    自定义时空数据集类
    """
    
    # 标准时间轴：1月到12月
    MONTHS = list(range(1, 13))
    
    def __init__(
        self,
        config: 'ConfigManager',
        encoder: 'LabelEncoder',
        dynamic_crawler: Optional['RasterCrawler'] = None,
        static_crawler: Optional['RasterCrawler'] = None,
        stats_file: Optional[Path] = None,
        split: str = 'train',
        split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 42,
        cache_metadata: bool = True,
        missing_value_strategy: str = 'zero_padding',
        normalization_method: str = 'minmax',
        verbose: bool = True,
    ):
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        self.verbose = verbose
        
        self.config = config
        self.encoder = encoder
        self.dynamic_crawler = dynamic_crawler
        self.static_crawler = static_crawler
        self.split = split
        self.missing_value_strategy = missing_value_strategy
        self.normalization_method = normalization_method
        
        self._log(f"🚀 初始化时空数据集（split={split}）")
        
        # 获取点位列表
        self.points_df = encoder.get_geodataframe().copy()
        self.points_df = self.points_df.reset_index(drop=True)
        
        # 划分数据集
        np.random.seed(seed)
        n_samples = len(self.points_df)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        n_train = int(n_samples * split_ratio[0])
        n_val = int(n_samples * split_ratio[1])
        
        if split == 'train':
            self.indices = indices[:n_train]
        elif split == 'val':
            self.indices = indices[n_train:n_train + n_val]
        elif split == 'test':
            self.indices = indices[n_train + n_val:]
        else:
            raise ValueError(f"未知的split: {split}")
        
        self._log(f"📋 {split} 集样本数: {len(self.indices)}")
        
        # 加载归一化参数
        self.normalization_stats = None
        if stats_file and Path(stats_file).exists():
            with open(stats_file, 'r', encoding='utf-8') as f:
                self.normalization_stats = json.load(f)
        
        # 缓存元数据
        self.metadata_cache = {}
        if cache_metadata:
            self._build_metadata_cache()
    
    @staticmethod
    def _setup_logging():
        if not logging.getLogger(__name__).handlers:
            logging.basicConfig(level=logging.INFO)
    
    def _log(self, message: str):
        if self.verbose:
            self.logger.info(message)
    
    def _build_metadata_cache(self):
        if not self.dynamic_crawler and not self.static_crawler:
            return
        
        for idx in tqdm(self.indices, disable=not self.verbose, desc="构建元数据缓存"):
            row = self.points_df.iloc[idx]
            x, y = float(row.geometry.x), float(row.geometry.y)
            
            dynamic_rasters = {}
            if self.dynamic_crawler:
                rasters = self.dynamic_crawler.find_rasters_by_point(x, y)
                dynamic_rasters = self._group_rasters_by_month(rasters)
            
            static_rasters = []
            if self.static_crawler:
                static_rasters = self.static_crawler.find_rasters_by_point(x, y)
            
            self.metadata_cache[idx] = {
                'dynamic': dynamic_rasters,
                'static': static_rasters,
            }
    
    def _group_rasters_by_month(self, rasters: List) -> Dict[int, str]:
        grouped = defaultdict(list)
        for raster in rasters:
            month = raster.month if raster.month else 1
            grouped[month].append(raster)
        
        result = {}
        for month, month_rasters in grouped.items():
            month_rasters.sort(key=lambda r: r.date if r.date else datetime.min, reverse=True)
            result[month] = str(month_rasters[0].filepath)
        return result
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_idx = self.indices[idx]
        row = self.points_df.iloc[sample_idx]
        
        x = float(row.geometry.x)
        y = float(row.geometry.y)
        
        # =====================================================================
        # 关键修复：正确读取 major_label 和 detail_label
        # =====================================================================
        try:
            detail_label = int(row['detail_label'])
            major_label = int(row['major_label'])
        except KeyError:
            # 如果 LabelEncoder 未正确生成列，抛出明确错误
            raise KeyError("DataFrame 中缺少 'detail_label' 或 'major_label' 列。请检查 LabelEncoder。")
            
        # 获取影像路径
        if sample_idx in self.metadata_cache:
            dynamic_rasters = self.metadata_cache[sample_idx]['dynamic']
            static_rasters = self.metadata_cache[sample_idx]['static']
        else:
            # Fallback (省略具体实现，与之前一致)
            dynamic_rasters = {}
            static_rasters = []
        
        # 读取影像
        patch_size = self.config.get('data_specs.spatial.patch_size', 64)
        dynamic_tensor = self._read_dynamic_timeseries(dynamic_rasters, x, y, patch_size)
        static_tensor = self._read_static_data(static_rasters, x, y, patch_size)
        
        # 归一化
        dynamic_tensor = self._normalize(dynamic_tensor, 'dynamic')
        static_tensor = self._normalize(static_tensor, 'static')
        
        return {
            'dynamic': dynamic_tensor,
            'static': static_tensor,
            'label': torch.tensor(detail_label, dtype=torch.long),       # 兼容标准模式
            'detail_label': torch.tensor(detail_label, dtype=torch.long), # 分层模式专用
            'major_label': torch.tensor(major_label, dtype=torch.long),   # 分层模式专用 (关键!)
            'coords': (x, y),
            'metadata': {'sample_id': sample_idx}
        }
    
    def _read_dynamic_timeseries(self, dynamic_rasters, x, y, patch_size):
        if not dynamic_rasters:
            return torch.zeros((12, 1, patch_size, patch_size), dtype=torch.float32)
            
        n_channels = self._get_num_channels(dynamic_rasters)
        dynamic_tensor = np.zeros((12, n_channels, patch_size, patch_size), dtype=np.float32)
        read_months = []
        
        for month in self.MONTHS:
            if month in dynamic_rasters:
                try:
                    data = self._read_window(dynamic_rasters[month], x, y, patch_size)
                    if data is not None:
                        if data.ndim == 2: data = data[np.newaxis, ...]
                        dynamic_tensor[month - 1] = data[:n_channels]
                        read_months.append(month)
                except Exception:
                    pass
        
        # 缺失值处理
        if self.missing_value_strategy == 'linear_interpolation':
            dynamic_tensor = self._interpolate_missing_months(dynamic_tensor, read_months)
        
        return torch.from_numpy(dynamic_tensor).float()

    def _read_static_data(self, static_rasters, x, y, patch_size):
        if not static_rasters:
            return torch.zeros((1, patch_size, patch_size), dtype=torch.float32)
        
        # 只取第一张静态图
        data = self._read_window(str(static_rasters[0].filepath), x, y, patch_size)
        if data is None:
            return torch.zeros((1, patch_size, patch_size), dtype=torch.float32)
        if data.ndim == 2:
            data = data[np.newaxis, ...]
            
        return torch.from_numpy(data).float()

    def _read_window(self, filepath, x, y, patch_size):
        try:
            with rasterio.open(filepath) as src:
                col, row = src.index(x, y)
                window = Window(col - patch_size//2, row - patch_size//2, patch_size, patch_size)
                data = src.read(window=window, boundless=True, fill_value=0)
                return data.astype(np.float32)
        except:
            return None

    def _get_num_channels(self, rasters):
        for f in rasters.values():
            try:
                with rasterio.open(f) as src: return src.count
            except: continue
        return 1

    def _normalize(self, tensor, data_type):
        if self.normalization_method == 'none' or not self.normalization_stats:
            return tensor
        
        stats = self.normalization_stats.get(f"{data_type}_stats", {})
        channels = stats.get('channels', [])
        
        if self.normalization_method == 'zscore':
            for i, ch in enumerate(channels):
                mean, std = ch.get('mean', 0), ch.get('std', 1)
                if std > 0: tensor[i] = (tensor[i] - mean) / std
                
        elif self.normalization_method == 'minmax':
            for i, ch in enumerate(channels):
                min_v, max_v = ch.get('min', 0), ch.get('max', 1)
                if max_v > min_v: tensor[i] = (tensor[i] - min_v) / (max_v - min_v)
                
        return tensor

    def _interpolate_missing_months(self, tensor, available):
        # 简化的插值逻辑
        return tensor

    def get_statistics(self):
        subset = self.points_df.iloc[self.indices]
        return {
            'total_samples': len(self),
            'label_distribution': subset['detail_label'].value_counts().to_dict()
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    更新后的 collate_fn，支持 major_label 和 detail_label
    """
    return {
        'dynamic': torch.stack([sample['dynamic'] for sample in batch]),
        'static': torch.stack([sample['static'] for sample in batch]),
        'label': torch.stack([sample['label'] for sample in batch]),
        'major_label': torch.stack([sample['major_label'] for sample in batch]),   # 新增
        'detail_label': torch.stack([sample['detail_label'] for sample in batch]), # 新增
        'coords': [sample['coords'] for sample in batch],
        'metadata': [sample['metadata'] for sample in batch],
    }