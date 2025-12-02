"""
data_loader.py: 整合入口
"""
import torch
from torch.utils.data import DataLoader
from raster_crawler import RasterCrawler
from stats_calculator import StatsCalculator
from point_timeseries_dataset import PointTimeSeriesDataset, collate_fn

class DataLoaderManager:
    def __init__(self, config, encoder):
        self.config = config
        self.encoder = encoder
        
        # 1. 初始化爬虫
        self.crawler = RasterCrawler(config)
        
        # 2. 检查并计算统计量 (如果不存在)
        stats_file = config.get_experiment_output_dir() / 'normalization_stats.json'
        if not stats_file.exists():
            print("正在计算全局归一化统计量 (这可能需要几分钟)...")
            calc = StatsCalculator(config)
            calc.compute_global_stats(self.crawler, sampling_rate=0.1) # 10% 采样
            calc.save_stats()

    def create_loaders(self):
        # 创建 Dataset
        train_ds = PointTimeSeriesDataset(
            self.config, self.encoder, self.crawler, split='train'
        )
        val_ds = PointTimeSeriesDataset(
            self.config, self.encoder, self.crawler, split='val'
        )
        
        # 创建 Loader
        train_loader = DataLoader(
            train_ds, 
            batch_size=self.config.get('training.batch_size', 16),
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0 # Windows下调试建议设为0
        )
        
        val_loader = DataLoader(
            val_ds, 
            batch_size=self.config.get('training.batch_size', 16),
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        # 获取输入通道数 (用于模型初始化)
        input_dim = train_ds.num_channels
        
        return train_loader, val_loader, input_dim