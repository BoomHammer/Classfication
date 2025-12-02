"""
stats_calculator.py: 统计计算器 (Super-Channel 适配版)
"""
import json
import logging
import random
import numpy as np
import rasterio
from tqdm import tqdm
from collections import defaultdict
from dataclasses import dataclass

@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    M2: float = 0.0
    
    def update(self, val_array):
        # Welford 算法批量更新
        val_array = val_array.flatten().astype(np.float64)
        n = len(val_array)
        if n == 0: return
        new_mean = np.mean(val_array)
        new_M2 = np.sum((val_array - new_mean)**2)
        
        delta = new_mean - self.mean
        new_count = self.count + n
        
        # 合并方差
        self.M2 += new_M2 + delta**2 * self.count * n / new_count
        self.mean += delta * n / new_count
        self.count = new_count

    @property
    def std(self):
        return np.sqrt(self.M2 / self.count) if self.count > 0 else 0.0

class StatsCalculator:
    def __init__(self, config):
        self.config = config
        self.output_dir = config.get_experiment_output_dir()
        self.logger = logging.getLogger(__name__)
        
        # 存储每个变量的统计器: {'GPP': RunningStats(), 'NDVI': ...}
        self.var_stats = defaultdict(RunningStats) 
        self.dynamic_channel_order = [] 

    def compute_global_stats(self, crawler, sampling_rate=0.1):
        """
        计算全局统计量 (按变量聚合)
        """
        # 1. 获取超级通道定义 (确定顺序)
        def_dict = crawler.get_super_channel_definition()
        # 将 map 反转: {0: 'GPP', 1: 'NDVI'} 方便排序输出
        self.dynamic_channel_order = sorted(def_dict['channel_map'].keys())
        
        # 2. 按变量分组文件
        var_files = defaultdict(list)
        all_rasters = crawler.get_all_rasters()
        for r in all_rasters:
            if r.variable:
                var_files[r.variable].append(r)
        
        self.logger.info(f"统计计算: 共有 {len(self.dynamic_channel_order)} 个变量通道")
        
        # 3. 对每个变量进行采样统计
        for var_name in self.dynamic_channel_order:
            files = var_files.get(var_name, [])
            if not files: continue
            
            # 采样
            k = max(1, int(len(files) * sampling_rate))
            sampled = random.sample(files, k)
            
            stats = self.var_stats[var_name]
            pbar = tqdm(sampled, desc=f"统计 {var_name}", leave=False)
            
            for meta in pbar:
                try:
                    with rasterio.open(meta.filepath) as src:
                        data = src.read(1) # 假设单波段文件
                        # 过滤无效值 (假设 0 是 nodata，视情况修改)
                        valid_data = data[data != 0] 
                        stats.update(valid_data)
                except Exception as e:
                    pass
            
            self.logger.info(f"  - {var_name}: Mean={stats.mean:.4f}, Std={stats.std:.4f}")

    def save_stats(self, filename='normalization_stats.json'):
        # 按照 channel_map 的顺序导出 mean 和 std 列表
        means = []
        stds = []
        
        for var in self.dynamic_channel_order:
            s = self.var_stats[var]
            means.append(float(s.mean))
            # 避免 std 为 0 导致除零错误
            stds.append(float(s.std) if s.std > 1e-6 else 1.0)
            
        output = {
            "dynamic_stats": {
                "channels": [{"mean": m, "std": s, "name": n} for m, s, n in zip(means, stds, self.dynamic_channel_order)]
            },
            # 静态影像如果有，需类似处理。这里简化为空或默认
            "static_stats": {"channels": []} 
        }
        
        with open(self.output_dir / filename, 'w') as f:
            json.dump(output, f, indent=2)