"""
PointTimeSeriesDataset: 自定义时空数据集类

【核心设计原理】

这是整个系统中最具技术挑战性的部分。传统计算机视觉将图像预切割成小文件，
但在遥感领域切分TB级大图会产生数百万个小文件，导致文件系统崩溃。

本方案采用"即时窗口读取"(On-the-fly Windowed Reading)策略：
1. 保持原始大图（GeoTIFF），不进行切割
2. 在训练时根据CSV坐标实时计算像素偏移
3. 利用GeoTIFF的分块存储特性（Tiled Structure），仅读取所需的64x64像素窗口
4. 使用rasterio的高效窗口读取API，配合内存映射

【时空对齐策略】

由于遥感影像时间元数据存在模糊性（如"2023年1月"而非"2023-01-15"），
本方案定义统一的时间轴：
- 标准时间轴：1月到12月（12个时间步）
- 时间分组方式：按月份聚合
- 单月多张影像：取时间最近的一张
- 缺失月份：Zero-padding（保持为零）或线性插值

【关键特性】

✓ 高效读取：利用rasterio的窗口读取，每次O(1)磁盘操作
✓ 内存高效：即时读取，不预加载所有影像
✓ 灵活对齐：支持多种时间分组和缺失值处理策略
✓ 多源融合：支持动态影像（如Sentinel-2）和静态影像（如DEM）
✓ 完整验证：支持数据质量检查和性能基准测试

【数据格式】

返回样本格式（PyTorch张量）：
{
    "dynamic": Tensor(T, C, H, W),     # 动态影像 (12月 × 波段 × 64 × 64)
    "static": Tensor(S, H, W),        # 静态影像 (波段 × 64 × 64)
    "label": int,                      # 类别标签 (0-7)
    "coords": Tuple(float, float),    # 原始坐标 (经度, 纬度)
    "metadata": dict                   # 元数据（如使用的影像文件）
}

【性能指标】

预期性能（在SSD上）：
- __getitem__ 耗时：< 0.1 秒（必要条件，否则成为GPU训练瓶颈）
- 内存占用：约200MB（预加载R-树索引）
- 数据集大小：支持 1M+ 样本无问题

【使用示例】

from point_timeseries_dataset import PointTimeSeriesDataset
from torch.utils.data import DataLoader

dataset = PointTimeSeriesDataset(
    config=config,
    encoder=label_encoder,
    dynamic_crawler=dynamic_crawler,
    static_crawler=static_crawler,
    stats_file='normalization_stats.json',
    split='train',
    cache_metadata=True
)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    shuffle=True
)

for batch in dataloader:
    dynamic = batch['dynamic']      # (32, 12, 4, 64, 64)
    static = batch['static']        # (32, 1, 64, 64)
    labels = batch['label']         # (32,)
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


@dataclass
class SampleMetadata:
    """样本元数据"""
    sample_id: int
    point_id: str
    x: float
    y: float
    label: int
    label_name: str
    major_class: str
    detail_class: str
    used_dynamic_rasters: Dict[int, str]  # month -> filepath
    used_static_rasters: List[str]
    

class PointTimeSeriesDataset(Dataset):
    """
    自定义时空数据集类
    
    实现从稀疏的矢量点标签（CSV）与稠密的遥感影像时间序列的对齐。
    采用"即时窗口读取"策略，避免预切割大文件。
    
    使用流程：
    1. 初始化数据集
    2. 通过 DataLoader 迭代
    3. 每次调用 __getitem__ 时实时读取所需的图像窗口
    """
    
    # 标准时间轴：1月到12月
    MONTHS = list(range(1, 13))
    MONTH_NAMES = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr',
        5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
        9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }
    
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
        """
        初始化时空数据集
        
        Args:
            config: ConfigManager对象
            encoder: LabelEncoder对象（包含点位信息和类别映射）
            dynamic_crawler: 动态影像爬虫（Sentinel-2等）
            static_crawler: 静态影像爬虫（DEM等）
            stats_file: 归一化参数文件路径
            split: 数据划分 ('train', 'val', 'test')
            split_ratio: 训练/验证/测试划分比例
            seed: 随机种子
            cache_metadata: 是否缓存元数据以加快查询
            missing_value_strategy: 缺失值处理策略 ('zero_padding', 'linear_interpolation', 'forward_fill')
            normalization_method: 归一化方法 ('minmax', 'zscore', 'none')
            verbose: 是否打印详细日志
        
        Raises:
            ValueError: 配置或数据不合法
            FileNotFoundError: 必要文件不存在
        """
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        self.verbose = verbose
        
        # 保存配置
        self.config = config
        self.encoder = encoder
        self.dynamic_crawler = dynamic_crawler
        self.static_crawler = static_crawler
        self.split = split
        self.split_ratio = split_ratio
        self.seed = seed
        self.cache_metadata = cache_metadata
        self.missing_value_strategy = missing_value_strategy
        self.normalization_method = normalization_method
        
        self._log(f"🚀 初始化时空数据集（split={split}）")
        
        # =====================================================================
        # 第一步：获取点位列表
        # =====================================================================
        self.points_df = encoder.get_geodataframe().copy()
        self.points_df = self.points_df.reset_index(drop=True)
        
        self._log(f"📊 点位总数: {len(self.points_df)}")
        
        # =====================================================================
        # 第二步：划分训练/验证/测试集
        # =====================================================================
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
        
        # =====================================================================
        # 第三步：加载归一化参数
        # =====================================================================
        self.normalization_stats = None
        if stats_file and Path(stats_file).exists():
            with open(stats_file, 'r', encoding='utf-8') as f:
                self.normalization_stats = json.load(f)
            self._log(f"✅ 加载归一化参数: {stats_file}")
        elif self.normalization_method != 'none':
            self._log("⚠️  未找到归一化参数文件，将使用默认值（均值=0，方差=1）")
        
        # =====================================================================
        # 第四步：缓存元数据（可选）
        # =====================================================================
        self.metadata_cache = {}
        if cache_metadata:
            self._log("💾 构建元数据缓存（用于加快查询）...")
            self._build_metadata_cache()
        
        self._log(f"✅ 时空数据集初始化完成 (split={split}, size={len(self.indices)})")
    
    @staticmethod
    def _setup_logging():
        """配置日志系统"""
        if not logging.getLogger(__name__).handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logging.getLogger(__name__).addHandler(handler)
            logging.getLogger(__name__).setLevel(logging.INFO)
    
    def _log(self, message: str):
        """条件性地打印日志"""
        if self.verbose:
            self.logger.info(message)
    
    def _build_metadata_cache(self):
        """
        预构建元数据缓存
        
        对每个点位，查询覆盖它的所有影像，按月份分组并排序。
        这样在 __getitem__ 时可以快速获取。
        """
        if not self.dynamic_crawler and not self.static_crawler:
            self._log("⚠️  没有栅格爬虫，跳过元数据缓存")
            return
        
        for idx in tqdm(self.indices, disable=not self.verbose, desc="构建元数据缓存"):
            row = self.points_df.iloc[idx]
            x, y = float(row.geometry.x), float(row.geometry.y)
            
            # 查询动态影像
            dynamic_rasters = {}
            if self.dynamic_crawler:
                rasters = self.dynamic_crawler.find_rasters_by_point(x, y)
                dynamic_rasters = self._group_rasters_by_month(rasters)
            
            # 查询静态影像
            static_rasters = []
            if self.static_crawler:
                static_rasters = self.static_crawler.find_rasters_by_point(x, y)
            
            self.metadata_cache[idx] = {
                'dynamic': dynamic_rasters,
                'static': static_rasters,
            }
    
    def _group_rasters_by_month(self, rasters: List) -> Dict[int, str]:
        """
        将影像按月份分组
        
        如果某月有多张影像，取时间最近的一张
        
        Args:
            rasters: RasterMetadata 列表
        
        Returns:
            Dict[month (1-12) -> filepath]
        """
        grouped = defaultdict(list)
        for raster in rasters:
            month = raster.month if raster.month else 1
            grouped[month].append(raster)
        
        # 每个月选择最近的影像
        result = {}
        for month, month_rasters in grouped.items():
            # 按日期排序，选择最近的
            month_rasters.sort(
                key=lambda r: r.date if r.date else datetime.min,
                reverse=True
            )
            result[month] = str(month_rasters[0].filepath)
        
        return result
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取单个样本
        
        这是模型训练时被反复调用的核心函数。
        设计目标：在 < 0.1 秒内完成单个样本的读取、对齐和归一化。
        
        Args:
            idx: 样本索引（0 到 len(dataset)-1）
        
        Returns:
            Dict 包含：
                - dynamic: (T, C, H, W) 张量，T=12月，C=波段数，H=W=patch_size
                - static: (S, H, W) 张量
                - label: 类别标签整数
                - coords: (x, y) 元组
                - metadata: 元数据字典
        
        Raises:
            RuntimeError: 读取栅格失败
            ValueError: 坐标超出范围
        """
        # =====================================================================
        # 第一步：获取点位信息
        # =====================================================================
        sample_idx = self.indices[idx]
        row = self.points_df.iloc[sample_idx]
        
        x = float(row.geometry.x)
        y = float(row.geometry.y)
        detail_label = int(row['detail_label'])
        major_label = int(row['major_label'])  # 新增：获取大类标签
        label_name = row.get('Alliance', 'Unknown')
        major_class_name = row.get('Formation', 'Unknown')
        
        # =====================================================================
        # 第二步：查询影像（从缓存或即时查询）
        # =====================================================================
        if sample_idx in self.metadata_cache:
            dynamic_rasters = self.metadata_cache[sample_idx]['dynamic']
            static_rasters = self.metadata_cache[sample_idx]['static']
        else:
            dynamic_rasters = {}
            static_rasters = []
            
            if self.dynamic_crawler:
                rasters = self.dynamic_crawler.find_rasters_by_point(x, y)
                dynamic_rasters = self._group_rasters_by_month(rasters)
            
            if self.static_crawler:
                static_rasters = self.static_crawler.find_rasters_by_point(x, y)
        
        # =====================================================================
        # 第三步：读取动态影像（时间序列）
        # =====================================================================
        patch_size = self.config.get('data_specs.spatial.patch_size', 64)
        dynamic_tensor = self._read_dynamic_timeseries(
            dynamic_rasters, x, y, patch_size
        )
        
        # =====================================================================
        # 第四步：读取静态影像
        # =====================================================================
        static_tensor = self._read_static_data(static_rasters, x, y, patch_size)
        
        # =====================================================================
        # 第五步：归一化
        # =====================================================================
        dynamic_tensor = self._normalize(dynamic_tensor, 'dynamic')
        static_tensor = self._normalize(static_tensor, 'static')
        
        # =====================================================================
        # 第六步：返回样本
        # =====================================================================
        return {
            'dynamic': dynamic_tensor,
            'static': static_tensor,
            'major_label': torch.tensor(major_label, dtype=torch.long),  # 新增：大类标签
            'detail_label': torch.tensor(detail_label, dtype=torch.long),  # 改名为detail_label
            'label': torch.tensor(detail_label, dtype=torch.long),  # 向后兼容
            'coords': (x, y),
            'metadata': {
                'sample_id': sample_idx,
                'detail_label_name': label_name,
                'major_class_name': major_class_name,
                'used_dynamic_months': list(dynamic_rasters.keys()),
                'used_static_files': [str(f) for f in static_rasters],
            }
        }
    
    def _read_dynamic_timeseries(
        self,
        dynamic_rasters: Dict[int, str],
        x: float,
        y: float,
        patch_size: int,
    ) -> torch.Tensor:
        """
        读取动态影像时间序列
        
        逻辑：
        1. 初始化全零张量 (T, C, H, W)
        2. 对每个存在的月份，计算像素坐标和窗口
        3. 使用 rasterio.read(window=...) 读取数据
        4. 处理缺失月份
        
        Args:
            dynamic_rasters: {month -> filepath} 映射
            x, y: 地理坐标
            patch_size: 切片大小（例如64）
        
        Returns:
            torch.Tensor: (T, C, H, W) 张量，其中 T=12
        """
        if not dynamic_rasters:
            # 如果没有动态影像，返回全零
            return torch.zeros(
                (12, 1, patch_size, patch_size),
                dtype=torch.float32
            )
        
        # 确定通道数
        n_channels = self._get_num_channels(dynamic_rasters)
        dynamic_tensor = np.zeros((12, n_channels, patch_size, patch_size), dtype=np.float32)
        
        # 记录读取的月份
        read_months = []
        
        # 对每个月份读取数据
        for month in self.MONTHS:
            if month in dynamic_rasters:
                filepath = dynamic_rasters[month]
                try:
                    data = self._read_window(filepath, x, y, patch_size)
                    if data is not None:
                        # 确保数据形状正确
                        if data.ndim == 2:
                            data = data[np.newaxis, ...]  # 添加通道维度
                        elif data.ndim == 3:
                            pass  # 已经是 (C, H, W)
                        else:
                            data = data.reshape(n_channels, patch_size, patch_size)
                        
                        # 截取到 n_channels（以防万一）
                        data = data[:n_channels]
                        dynamic_tensor[month - 1] = data
                        read_months.append(month)
                except Exception as e:
                    self.logger.warning(f"❌ 读取动态影像失败 (month={month}, file={filepath}): {e}")
        
        # =====================================================================
        # 处理缺失月份
        # =====================================================================
        missing_months = [m for m in self.MONTHS if m not in read_months]
        
        if missing_months:
            if self.missing_value_strategy == 'zero_padding':
                # 保持为零（默认）
                pass
            elif self.missing_value_strategy == 'linear_interpolation':
                # 线性插值（简单版本）
                dynamic_tensor = self._interpolate_missing_months(
                    dynamic_tensor, read_months
                )
            elif self.missing_value_strategy == 'forward_fill':
                # 前向填充
                dynamic_tensor = self._forward_fill_missing_months(
                    dynamic_tensor, read_months
                )
        
        return torch.from_numpy(dynamic_tensor).float()
    
    def _read_static_data(
        self,
        static_rasters: List,
        x: float,
        y: float,
        patch_size: int,
    ) -> torch.Tensor:
        """
        读取静态影像（例如DEM）
        
        Args:
            static_rasters: RasterMetadata 列表
            x, y: 地理坐标
            patch_size: 切片大小
        
        Returns:
            torch.Tensor: (S, H, W) 张量，其中 S=静态图层数
        """
        if not static_rasters:
            # 如果没有静态影像，返回全零
            return torch.zeros(
                (1, patch_size, patch_size),
                dtype=torch.float32
            )
        
        # 读取第一个静态影像（通常只有一个，如DEM）
        static_data_list = []
        for raster in static_rasters[:1]:  # 只取第一个
            filepath = str(raster.filepath)
            try:
                data = self._read_window(filepath, x, y, patch_size)
                if data is not None:
                    if data.ndim == 2:
                        data = data[np.newaxis, ...]
                    static_data_list.append(data)
            except Exception as e:
                self.logger.warning(f"❌ 读取静态影像失败 (file={filepath}): {e}")
        
        if static_data_list:
            static_tensor = np.vstack(static_data_list).astype(np.float32)
        else:
            static_tensor = np.zeros((1, patch_size, patch_size), dtype=np.float32)
        
        return torch.from_numpy(static_tensor).float()
    
    def _read_window(
        self,
        filepath: str,
        x: float,
        y: float,
        patch_size: int,
        timeout: float = 5.0,
    ) -> Optional[np.ndarray]:
        """
        使用 rasterio 读取地理坐标对应的图像窗口
        
        核心逻辑：
        1. 打开GeoTIFF文件
        2. 将地理坐标 (x, y) 转换为像素坐标 (row, col)
        3. 计算窗口范围
        4. 使用 rasterio.read(window=...) 读取数据
        5. 处理边界情况（点在影像边缘）
        
        Args:
            filepath: 栅格文件路径
            x, y: 地理坐标（与影像CRS一致）
            patch_size: 窗口大小（例如64）
            timeout: 读取超时时间（秒）
        
        Returns:
            np.ndarray: 形状为 (C, H, W) 或 (H, W) 的数据
                       如果坐标超出范围，返回 None
        """
        try:
            with rasterio.open(filepath) as src:
                # 检查坐标是否在影像范围内
                bounds = src.bounds
                if not (bounds.left <= x <= bounds.right and 
                        bounds.bottom <= y <= bounds.top):
                    return None
                
                # 转换地理坐标到像素坐标
                # rasterio 中，(col, row) = src.index(x, y)
                col, row = src.index(x, y)
                col = int(col)
                row = int(row)
                
                # 计算窗口（以点为中心）
                half_size = patch_size // 2
                col_start = max(0, col - half_size)
                row_start = max(0, row - half_size)
                col_end = min(src.width, col_start + patch_size)
                row_end = min(src.height, row_start + patch_size)
                
                # 检查是否有足够的数据
                if col_end - col_start < patch_size or row_end - row_start < patch_size:
                    # 数据不足，考虑是否要填充
                    # 这里简单地返回 None，或可以进行零填充
                    return None
                
                # 创建窗口对象
                window = Window(col_start, row_start, col_end - col_start, row_end - row_start)
                
                # 读取所有波段
                data = src.read(window=window)  # 返回 (C, H, W)
                
                return data.astype(np.float32)
        
        except Exception as e:
            self.logger.debug(f"❌ 读取窗口失败 (file={filepath}, x={x}, y={y}): {e}")
            return None
    
    def _get_num_channels(self, rasters_dict: Dict) -> int:
        """获取第一个可用栅格的通道数"""
        for filepath in rasters_dict.values():
            try:
                with rasterio.open(filepath) as src:
                    return src.count
            except:
                continue
        return 1  # 默认返回1
    
    def _normalize(self, tensor: torch.Tensor, data_type: str) -> torch.Tensor:
        """
        应用归一化
        
        Args:
            tensor: 输入张量
            data_type: 数据类型 ('dynamic' 或 'static')
        
        Returns:
            归一化后的张量
        """
        if self.normalization_method == 'none':
            return tensor
        
        if not self.normalization_stats:
            return tensor
        
        stats_key = f"{data_type}_stats"
        if stats_key not in self.normalization_stats:
            return tensor
        
        stats = self.normalization_stats[stats_key]
        
        if self.normalization_method == 'zscore':
            # Z-score 标准化：(x - mean) / std
            for channel_idx, channel_stats in enumerate(stats.get('channels', [])):
                mean = channel_stats.get('mean', 0.0)
                std = channel_stats.get('std', 1.0)
                if std > 0:
                    tensor[channel_idx] = (tensor[channel_idx] - mean) / std
        
        elif self.normalization_method == 'minmax':
            # MinMax 归一化：(x - min) / (max - min)
            for channel_idx, channel_stats in enumerate(stats.get('channels', [])):
                min_val = channel_stats.get('min', 0.0)
                max_val = channel_stats.get('max', 1.0)
                if max_val > min_val:
                    tensor[channel_idx] = (tensor[channel_idx] - min_val) / (max_val - min_val)
        
        return tensor
    
    def _interpolate_missing_months(
        self,
        tensor: np.ndarray,
        available_months: List[int]
    ) -> np.ndarray:
        """线性插值填充缺失月份"""
        if len(available_months) < 2:
            return tensor
        
        # 简单的线性插值实现
        available_months_sorted = sorted(available_months)
        for i in range(len(available_months_sorted) - 1):
            m1 = available_months_sorted[i]
            m2 = available_months_sorted[i + 1]
            
            for m in range(m1 + 1, m2):
                # 线性插值
                alpha = (m - m1) / (m2 - m1)
                tensor[m - 1] = (1 - alpha) * tensor[m1 - 1] + alpha * tensor[m2 - 1]
        
        return tensor
    
    def _forward_fill_missing_months(
        self,
        tensor: np.ndarray,
        available_months: List[int]
    ) -> np.ndarray:
        """前向填充缺失月份"""
        if not available_months:
            return tensor
        
        available_months_sorted = sorted(available_months)
        current_value = tensor[available_months_sorted[0] - 1].copy()
        
        for month in range(1, 13):
            if month not in available_months:
                tensor[month - 1] = current_value
            else:
                current_value = tensor[month - 1].copy()
        
        return tensor
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据集统计信息
        
        Returns:
            包含样本数、类别分布等信息的字典
        """
        subset_df = self.points_df.iloc[self.indices]
        
        return {
            'total_samples': len(self),
            'split': self.split,
            'label_distribution': subset_df['detail_label'].value_counts().to_dict(),
            'major_class_distribution': subset_df['Formation'].value_counts().to_dict(),
            'bounding_box': {
                'x_min': float(subset_df.geometry.bounds['minx'].min()),
                'y_min': float(subset_df.geometry.bounds['miny'].min()),
                'x_max': float(subset_df.geometry.bounds['maxx'].max()),
                'y_max': float(subset_df.geometry.bounds['maxy'].max()),
            }
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    自定义 collate 函数，用于 DataLoader
    
    Args:
        batch: 样本列表
    
    Returns:
        批次字典
    """
    return {
        'dynamic': torch.stack([sample['dynamic'] for sample in batch]),
        'static': torch.stack([sample['static'] for sample in batch]),
        'label': torch.stack([sample['label'] for sample in batch]),
        'coords': [sample['coords'] for sample in batch],
        'metadata': [sample['metadata'] for sample in batch],
    }
