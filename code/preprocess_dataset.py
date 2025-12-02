"""
preprocess_dataset.py: 离线数据预处理脚本 (修复版)
功能：将多源异构 TIFF 数据对齐并转换为 .pt 文件存储，并保存元数据。
"""

import sys
import shutil
import logging
import json
from pathlib import Path
import torch
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
from collections import defaultdict
from datetime import date

# 导入本地模块
sys.path.insert(0, str(Path(__file__).parent))
from config_manager import ConfigManager
from label_encoder import LabelEncoder
from raster_crawler import RasterCrawler

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Preprocessor")

class DataPreprocessor:
    def __init__(self, config_path):
        self.config = ConfigManager(config_path)
        self.output_dir = self.config.get_resolved_path('data_dir') / "processed_tensors"
        self.stats_file = self.config.get_experiment_output_dir() / 'normalization_stats.json'
        
        # 初始化组件
        logger.info("初始化组件...")
        self.encoder = LabelEncoder(config=self.config)
        self.crawler = RasterCrawler(config=self.config)
        self.points_df = self.encoder.get_geodataframe().reset_index(drop=True)
        
        # 获取超级通道定义
        def_dict = self.crawler.get_super_channel_definition()
        self.channel_map = def_dict['channel_map']
        self.timeline = def_dict['timeline']
        self.num_channels = len(self.channel_map)
        
        # 加载归一化统计量
        if self.stats_file.exists():
            with open(self.stats_file, 'r', encoding='utf-8') as f:
                self.stats = json.load(f)
        else:
            logger.warning("⚠️ 未找到统计文件，将跳过归一化！建议先运行 stats_calculator.py")
            self.stats = None

    def _get_file_map(self, point_geom):
        """为单个点构建时间-文件映射"""
        x, y = point_geom.x, point_geom.y
        rasters = self.crawler.find_rasters_by_point(x, y)
        
        daily = defaultdict(dict)
        monthly = defaultdict(dict)
        
        for r in rasters:
            if not r.variable: continue
            if r.is_monthly:
                monthly[(r.date.year, r.date.month)][r.variable] = str(r.filepath)
            elif r.date:
                daily[r.date][r.variable] = str(r.filepath)
        
        aligned_map = {}
        for t, target_date in enumerate(self.timeline):
            step_files = {}
            m_key = (target_date.year, target_date.month)
            for var, ch_idx in self.channel_map.items():
                path = None
                if var in daily.get(target_date, {}):
                    path = daily[target_date][var]
                elif var in monthly.get(m_key, {}):
                    path = monthly[m_key][var]
                if path: step_files[ch_idx] = path
            if step_files: aligned_map[t] = step_files
            
        return aligned_map

    def process_all(self):
        """执行预处理"""
        patch_size = self.config.get('data_specs.spatial.patch_size', 64)
        max_len = self.config.get('data.max_len', 60)
        
        # 清理旧数据
        if self.output_dir.exists():
            logger.warning(f"清理旧数据: {self.output_dir}")
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True)

        logger.info(f"开始处理 {len(self.points_df)} 个样本...")
        logger.info(f"目标: {self.output_dir}")
        logger.info(f"参数: Patch={patch_size}, MaxLen={max_len}, Channels={self.num_channels}")

        success_count = 0
        
        for idx, row in tqdm(self.points_df.iterrows(), total=len(self.points_df)):
            file_map = self._get_file_map(row.geometry)
            
            # 1. 时间步筛选与截断
            active_steps = sorted(file_map.keys())
            if len(active_steps) > max_len:
                indices = np.linspace(0, len(active_steps)-1, max_len, dtype=int)
                active_steps = [active_steps[i] for i in indices]
            
            T_actual = len(active_steps)
            if T_actual == 0: continue

            dyn_tensor = np.zeros((T_actual, self.num_channels, patch_size, patch_size), dtype=np.float32)
            dates = np.zeros(T_actual, dtype=int)
            
            # 2. 读取数据
            for k, t_idx in enumerate(active_steps):
                date_obj = self.timeline[t_idx]
                dates[k] = date_obj.timetuple().tm_yday
                
                step_files = file_map[t_idx]
                for ch_idx, fpath in step_files.items():
                    try:
                        with rasterio.open(fpath) as src:
                            r, c = src.index(row.geometry.x, row.geometry.y)
                            w = Window(c - patch_size//2, r - patch_size//2, patch_size, patch_size)
                            data = src.read(1, window=w, boundless=True, fill_value=0)
                            dyn_tensor[k, ch_idx] = data
                    except: pass

            # 3. 归一化
            if self.stats:
                stats_list = self.stats.get('dynamic_stats', {}).get('channels', [])
                for ch in range(self.num_channels):
                    if ch < len(stats_list):
                        mu = stats_list[ch]['mean']
                        sigma = stats_list[ch]['std']
                        if sigma > 1e-6:
                            dyn_tensor[:, ch] = (dyn_tensor[:, ch] - mu) / sigma

            # 4. 保存为 .pt 文件
            sample_data = {
                'dynamic': torch.from_numpy(dyn_tensor).float(),
                'static': torch.zeros(1, patch_size, patch_size),
                'label': torch.tensor(int(row.detail_label)).long(),
                'major_label': torch.tensor(int(row.major_label)).long(),
                'detail_label': torch.tensor(int(row.detail_label)).long(),
                'dates': torch.from_numpy(dates).long(),
                'coords': (row.geometry.x, row.geometry.y),
                'sample_id': idx
            }
            
            torch.save(sample_data, self.output_dir / f"{idx}.pt")
            success_count += 1

        # [新增] 保存元数据文件，供 Dataset 读取 channel_map
        metadata = {
            'channel_map': self.channel_map,
            'num_channels': self.num_channels,
            'timeline_start': self.timeline[0].isoformat() if self.timeline else None,
            'timeline_end': self.timeline[-1].isoformat() if self.timeline else None
        }
        with open(self.output_dir / "dataset_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"✅ 预处理完成！成功转换 {success_count} 个样本。")
        logger.info(f"📄 元数据已保存: dataset_metadata.json")

if __name__ == "__main__":
    config_path = Path(__file__).parent / 'config.yaml'
    preprocessor = DataPreprocessor(str(config_path))
    preprocessor.process_all()