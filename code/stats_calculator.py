"""
stats_calculator.py: 统计计算器 (修复版 - 包含静态数据全量统计)
"""
import json
import logging
import random
import numpy as np
import rasterio
from tqdm import tqdm
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    M2: float = 0.0
    
    def update(self, val_array):
        val_array = val_array.flatten().astype(np.float64)
        n = len(val_array)
        if n == 0: return
        new_mean = np.mean(val_array)
        new_M2 = np.sum((val_array - new_mean)**2)
        delta = new_mean - self.mean
        new_count = self.count + n
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
        
        self.dynamic_stats = defaultdict(RunningStats)
        self.static_stats = defaultdict(RunningStats)
        self.dynamic_channel_order = [] 
        self.static_channel_order = []

    def compute_all_stats(self, dynamic_crawler, static_crawler, sampling_rate=0.1):
        self._compute_dynamic(dynamic_crawler, sampling_rate)
        self._compute_static(static_crawler) # 静态数据不采样，全量计算
        self.save_stats()

    def _compute_dynamic(self, crawler, sampling_rate):
        """计算动态影像统计量 (采样)"""
        def_dict = crawler.get_super_channel_definition()
        self.dynamic_channel_order = sorted(def_dict['channel_map'].keys())
        
        var_files = defaultdict(list)
        for r in crawler.get_all_rasters():
            if r.variable:
                var_files[r.variable].append(r)
        
        self.logger.info(f"📊 正在计算动态变量统计量 ({len(self.dynamic_channel_order)} 个变量)...")
        for i, var_name in enumerate(self.dynamic_channel_order):
            files = var_files.get(var_name, [])
            if not files: continue
            
            k = max(1, int(len(files) * sampling_rate))
            sampled = random.sample(files, k)
            stats = self.dynamic_stats[var_name]
            
            self.logger.info(f"  [{i+1}/{len(self.dynamic_channel_order)}] 计算 {var_name} (采样 {len(sampled)} 张)...")
            
            for meta in sampled:
                try:
                    with rasterio.open(meta.filepath) as src:
                        data = src.read(1)
                        valid_data = data[data != 0] # 假设 0 是 nodata
                        stats.update(valid_data)
                except: pass
            
            # 计算完一个变量后，输出结果
            self.logger.info(f"    -> {var_name}: Mean={stats.mean:.4f}, Std={stats.std:.4f}")

    def _compute_static(self, crawler):
        """计算静态影像统计量 (全量)"""
        # 获取所有静态文件
        rasters = crawler.get_all_rasters()
        # 静态文件通常按文件名作为变量名 (如 DEM.tif -> DEM)
        # 这里我们按文件名排序以保证顺序一致
        rasters.sort(key=lambda x: x.filepath.stem)
        
        self.static_channel_order = [r.filepath.stem for r in rasters]
        self.logger.info(f"🏔️ 正在计算静态变量统计量 ({len(rasters)} 个文件)...")
        
        for r in rasters:
            var_name = r.filepath.stem
            stats = self.static_stats[var_name]
            
            self.logger.info(f"  - 读取全量文件: {r.filename} ...")
            try:
                with rasterio.open(r.filepath) as src:
                    # 静态数据可能很大，分块读取或者读整个(内存允许的话)
                    # 考虑到静态数据通常只有一景，尝试直接读取
                    data = src.read(1)
                    # 静态数据处理 nodata (通常 DEM 的 nodata 是 -9999 或 -32768)
                    if src.nodata is not None:
                        valid_data = data[data != src.nodata]
                    else:
                        valid_data = data # 无法判断则全部计算
                    
                    # 再次过滤可能的填充值 (如坡度 < 0)
                    if 'slope' in var_name.lower():
                        valid_data = valid_data[valid_data >= 0]
                        
                    stats.update(valid_data)
                self.logger.info(f"    {var_name}: Mean={stats.mean:.4f}, Std={stats.std:.4f}")
            except Exception as e:
                self.logger.error(f"    计算 {var_name} 失败: {e}")

    def save_stats(self, filename='normalization_stats.json'):
        output = {
            "dynamic_stats": {
                "channels": [
                    {"mean": float(self.dynamic_stats[n].mean), 
                     "std": float(self.dynamic_stats[n].std) if self.dynamic_stats[n].std > 1e-6 else 1.0, 
                     "name": n} 
                    for n in self.dynamic_channel_order
                ]
            },
            "static_stats": {
                "channels": [
                    {"mean": float(self.static_stats[n].mean), 
                     "std": float(self.static_stats[n].std) if self.static_stats[n].std > 1e-6 else 1.0, 
                     "name": n} 
                    for n in self.static_channel_order
                ]
            }
        }
        
        with open(self.output_dir / filename, 'w') as f:
            json.dump(output, f, indent=2)
        self.logger.info(f"✅ 统计量已保存: {filename}")