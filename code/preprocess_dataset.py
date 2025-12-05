"""
preprocess_dataset.py: 离线数据预处理脚本 (修复版 - 真正读取静态数据 - 修复 LabelEncoder 属性错误)
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

sys.path.insert(0, str(Path(__file__).parent))
from config_manager import ConfigManager
from label_encoder import LabelEncoder
from raster_crawler import RasterCrawler
from stats_calculator import StatsCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Preprocessor")

class DataPreprocessor:
    def __init__(self, config_path):
        self.config = ConfigManager(config_path)
        self.output_dir = self.config.get_resolved_path('data_dir') / "processed_tensors"
        self.stats_file = self.config.get_experiment_output_dir() / 'normalization_stats.json'
        
        self.encoder = LabelEncoder(config=self.config)
        self.points_df = self.encoder.get_geodataframe().reset_index(drop=True)
        
        # 1. 动态爬虫
        self.dyn_crawler = RasterCrawler(config=self.config)
        def_dict = self.dyn_crawler.get_super_channel_definition()
        self.dyn_channel_map = def_dict['channel_map']
        self.timeline = def_dict['timeline']
        self.num_dyn_channels = len(self.dyn_channel_map)
        
        # 2. 静态爬虫 (匹配所有 tif)
        # 注意：这里需要确保 raster_crawler.py 已经更新支持 raster_dir 参数
        self.static_crawler = RasterCrawler(
            config=self.config,
            raster_dir=self.config.get_resolved_path('static_images_dir'),
            filename_pattern='.*', # 匹配任何文件名
            file_extensions=('.tif', '.tiff')
        )
        self.static_files = sorted(self.static_crawler.get_all_rasters(), key=lambda x: x.filepath.stem)
        self.static_channel_names = [f.filepath.stem for f in self.static_files]
        self.num_static_channels = len(self.static_files)
        
        logger.info(f"动态通道: {self.num_dyn_channels} (GPP, NDVI...)")
        logger.info(f"静态通道: {self.num_static_channels} ({self.static_channel_names})")

        # 3. 计算或加载统计量
        self.stats_calc = StatsCalculator(self.config)
        if not self.stats_file.exists():
            logger.warning("⚠️ 统计文件不存在，正在计算...")
            self.stats_calc.compute_all_stats(self.dyn_crawler, self.static_crawler)
        
        with open(self.stats_file, 'r', encoding='utf-8') as f:
            self.stats = json.load(f)

    def _get_file_map(self, point_geom):
        """为单个点构建动态数据映射"""
        x, y = point_geom.x, point_geom.y
        rasters = self.dyn_crawler.find_rasters_by_point(x, y)
        daily = defaultdict(dict)
        for r in rasters:
            if r.variable and r.date: daily[r.date][r.variable] = str(r.filepath)
        
        aligned_map = {}
        for t, target_date in enumerate(self.timeline):
            step_files = {}
            if target_date in daily:
                for var, ch_idx in self.dyn_channel_map.items():
                    if var in daily[target_date]:
                        step_files[ch_idx] = daily[target_date][var]
            if step_files: aligned_map[t] = step_files
        return aligned_map

    def process_all(self):
        patch_size = self.config.get('data_specs.spatial.patch_size', 64)
        max_len = self.config.get('data.max_len', 60)
        
        if self.output_dir.exists(): shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True)

        logger.info(f"开始处理 {len(self.points_df)} 个样本...")
        success_count = 0
        
        # 预先获取静态数据的统计量
        s_means = [c['mean'] for c in self.stats['static_stats']['channels']]
        s_stds = [c['std'] for c in self.stats['static_stats']['channels']]
        
        d_means = [0] * self.num_dyn_channels
        d_stds = [1] * self.num_dyn_channels
        for c in self.stats['dynamic_stats']['channels']:
            if c['name'] in self.dyn_channel_map:
                idx = self.dyn_channel_map[c['name']]
                d_means[idx] = c['mean']
                d_stds[idx] = c['std']

        for idx, row in tqdm(self.points_df.iterrows(), total=len(self.points_df)):
            # --- 处理动态数据 ---
            file_map = self._get_file_map(row.geometry)
            active_steps = sorted(file_map.keys())
            
            # 时间采样/截断
            if len(active_steps) > max_len:
                indices = np.linspace(0, len(active_steps)-1, max_len, dtype=int)
                active_steps = [active_steps[i] for i in indices]
            
            T = len(active_steps)
            if T == 0: continue # 跳过无动态数据的点

            dyn_tensor = np.zeros((T, self.num_dyn_channels, patch_size, patch_size), dtype=np.float32)
            dates = np.zeros(T, dtype=int)

            for k, t_idx in enumerate(active_steps):
                dates[k] = self.timeline[t_idx].timetuple().tm_yday
                for ch_idx, fpath in file_map[t_idx].items():
                    try:
                        with rasterio.open(fpath) as src:
                            r, c = src.index(row.geometry.x, row.geometry.y)
                            w = Window(c - patch_size//2, r - patch_size//2, patch_size, patch_size)
                            data = src.read(1, window=w, boundless=True, fill_value=0)
                            # 动态归一化
                            if d_stds[ch_idx] > 1e-6:
                                data = (data - d_means[ch_idx]) / d_stds[ch_idx]
                            dyn_tensor[k, ch_idx] = data
                    except: pass
            
            # --- 处理静态数据 ---
            static_tensor = np.zeros((self.num_static_channels, patch_size, patch_size), dtype=np.float32)
            
            for i, r_meta in enumerate(self.static_files):
                try:
                    with rasterio.open(r_meta.filepath) as src:
                        r, c = src.index(row.geometry.x, row.geometry.y)
                        w = Window(c - patch_size//2, r - patch_size//2, patch_size, patch_size)
                        data = src.read(1, window=w, boundless=True, fill_value=src.nodata if src.nodata else 0)
                        
                        # 处理 nodata
                        if src.nodata is not None:
                            mask = (data == src.nodata)
                            data = data.astype(np.float32)
                            data[mask] = s_means[i] # 填充均值
                        
                        # 静态归一化
                        if s_stds[i] > 1e-6:
                            data = (data - s_means[i]) / s_stds[i]
                        
                        static_tensor[i] = data
                except Exception as e:
                    logger.warning(f"静态文件读取失败 {r_meta.filename}: {e}")

            # --- 保存 ---
            sample_data = {
                'dynamic': torch.from_numpy(dyn_tensor).float(),
                'static': torch.from_numpy(static_tensor).float(),
                'label': torch.tensor(int(row.detail_label)).long(),
                'major_label': torch.tensor(int(row.major_label)).long(),
                'detail_label': torch.tensor(int(row.detail_label)).long(),
                'dates': torch.from_numpy(dates).long(),
                'coords': (row.geometry.x, row.geometry.y),
                'sample_id': idx
            }
            torch.save(sample_data, self.output_dir / f"{idx}.pt")
            success_count += 1

        # 保存检测到的参数，供训练脚本使用
        # 【修复】使用 detailed_labels_map 替代不存在的 detailed_labels
        detected_params = {
            'num_classes': len(self.encoder.detailed_labels_map), 
            'dynamic_channels': self.num_dyn_channels,
            'static_channels': self.num_static_channels
        }
        with open(self.config.get_experiment_output_dir() / 'detected_parameters.json', 'w') as f:
            json.dump(detected_params, f)
            
        # 保存元数据
        metadata = {
            'channel_map': self.dyn_channel_map,
            'static_channels': self.static_channel_names,
            'num_channels': self.num_dyn_channels
        }
        with open(self.output_dir / "dataset_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"✅ 预处理完成！成功 {success_count} 个样本。")
        logger.info(f"   - 动态特征: {self.num_dyn_channels} 维")
        logger.info(f"   - 静态特征: {self.num_static_channels} 维 (已归一化)")

if __name__ == "__main__":
    config_path = Path(__file__).parent / 'config.yaml'
    preprocessor = DataPreprocessor(str(config_path))
    preprocessor.process_all()