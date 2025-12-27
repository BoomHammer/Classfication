#!/usr/bin/env python3
"""
predict_raster.py: 植被图栅格预测脚本

功能：
1. 读取训练好的分层模型（大类+小类）。
2. 扫描 data/raster 下的动静态影像，构建与训练一致的时间序列输入。
3. 使用滑动窗口进行全图预测。
4. 输出带有地理坐标系的 TIF 结果图。

使用方式：
python code/predict_raster.py --run_dir experiments/outputs/XXXXXXXX_XXXX_EXP_2023_001
"""

import os
import sys
import json
import torch
import logging
import argparse
import rasterio
import numpy as np
from pathlib import Path
from tqdm import tqdm
from rasterio.windows import Window
from collections import defaultdict

# 导入本地模块
sys.path.insert(0, str(Path(__file__).parent))
from config_manager import ConfigManager
from model_architecture import DualStreamSpatio_TemporalFusionNetwork
from raster_crawler import RasterCrawler

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RasterPredictor")

class RasterPredictor:
    def __init__(self, run_dir, device=None):
        self.run_dir = Path(run_dir)
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. 加载配置
        self.config_path = self.run_dir / "config_used.yaml"
        self.config = ConfigManager(str(self.config_path), create_experiment_dir=False)
        
        # 2. 加载元数据
        self._load_metadata()
        
        # 3. 扫描影像数据
        self._scan_raster_data()
        
        # 4. 加载统计量 (归一化参数)
        self._load_normalization_stats()
        
        # 5. 加载所有模型
        self._load_models()

    def _load_metadata(self):
        """加载映射关系和通道定义"""
        # 标签映射
        with open(self.run_dir / 'detailed_labels_map.json', 'r', encoding='utf-8') as f:
            self.detailed_map = json.load(f)
        self.inverse_detailed_map = {v: k for k, v in self.detailed_map.items()}
        
        with open(self.run_dir / 'major_labels_map.json', 'r', encoding='utf-8') as f:
            self.major_map = json.load(f)
            
        # 数据集参数
        with open(self.run_dir / 'detected_parameters.json', 'r') as f:
            self.params = json.load(f)
            
        with open(self.run_dir.parent.parent / 'data/processed_tensors/dataset_metadata.json', 'r') as f:
            self.dataset_meta = json.load(f)
            
        self.dyn_channel_map = self.dataset_meta['channel_map']
        self.static_channel_names = self.dataset_meta['static_channels']
        self.num_dyn_channels = self.params['dynamic_channels']
        self.num_static_channels = self.params['static_channels']
        
        logger.info(f"配置加载完毕: {len(self.detailed_map)} 个小类, {len(self.major_map)} 个大类")

    def _scan_raster_data(self):
        """使用 RasterCrawler 扫描影像，构建输入索引"""
        data_dir = self.config.get_resolved_path('data_dir')
        raster_root = data_dir / "raster"
        
        logger.info(f"正在扫描影像目录: {raster_root}")
        
        # 使用爬虫扫描所有文件
        crawler = RasterCrawler(self.config, raster_dir=raster_root, file_extensions=['.tif', '.tiff'])
        all_rasters = crawler.get_all_rasters()
        
        # 1. 寻找基准影像 (第一个 NDVI)
        self.ref_meta = None
        for r in all_rasters:
            if r.variable == 'NDVI' and r.date is not None:
                self.ref_meta = r
                break
        
        if not self.ref_meta:
            raise FileNotFoundError("未在 dynamic 目录中找到任何 NDVI 影像作为基准参考！")
            
        logger.info(f"基准影像: {self.ref_meta.filename} (Size: {self.ref_meta.width}x{self.ref_meta.height})")

        # 2. 构建动态数据映射: Date -> Channel Index -> FilePath
        self.dyn_files = defaultdict(dict)
        dates_set = set()
        
        for r in all_rasters:
            if r.variable in self.dyn_channel_map and r.date is not None:
                ch_idx = self.dyn_channel_map[r.variable]
                self.dyn_files[r.date][ch_idx] = r.filepath
                dates_set.add(r.date)
        
        self.timeline = sorted(list(dates_set))
        # 限制时间步长 (与训练保持一致，例如取最近 N 个或所有)
        # 这里假设使用所有扫描到的时间点，按时间排序
        logger.info(f"动态数据: 发现 {len(self.timeline)} 个时间点")

        # 3. 构建静态数据映射: Channel Index -> FilePath
        self.static_files = {}
        for r in all_rasters:
            # 静态文件通常没有日期，或者变量名在 static_channels 列表中
            # 这里简单通过文件名或变量名匹配
            if r.variable in self.static_channel_names:
                # 找到它在 static_channels 列表中的索引
                idx = self.static_channel_names.index(r.variable)
                self.static_files[idx] = r.filepath
            elif r.filename.replace('.tif', '') in self.static_channel_names:
                 idx = self.static_channel_names.index(r.filename.replace('.tif', ''))
                 self.static_files[idx] = r.filepath

        # 检查静态文件完整性
        missing_static = [self.static_channel_names[i] for i in range(self.num_static_channels) if i not in self.static_files]
        if missing_static:
            logger.warning(f"缺失静态文件: {missing_static}，将用0填充")

    def _load_normalization_stats(self):
        """加载归一化统计量"""
        stats_file = self.run_dir / 'normalization_stats.json'
        if not stats_file.exists():
            logger.warning("未找到 normalization_stats.json，将不进行归一化（可能导致预测错误）")
            self.d_means, self.d_stds = [0]*self.num_dyn_channels, [1]*self.num_dyn_channels
            self.s_means, self.s_stds = [0]*self.num_static_channels, [1]*self.num_static_channels
            return

        with open(stats_file, 'r') as f:
            stats = json.load(f)
            
        # 映射动态统计量
        self.d_means = [0.0] * self.num_dyn_channels
        self.d_stds = [1.0] * self.num_dyn_channels
        for c in stats['dynamic_stats']['channels']:
            if c['name'] in self.dyn_channel_map:
                idx = self.dyn_channel_map[c['name']]
                self.d_means[idx] = c['mean']
                self.d_stds[idx] = c['std']

        # 映射静态统计量
        self.s_means = [0.0] * self.num_static_channels
        self.s_stds = [1.0] * self.num_static_channels
        # 静态统计量在 list 中按顺序存储，或者按 name 匹配
        # 假设 stats['static_stats']['channels'] 的顺序可能不对应 static_channel_names
        # 最好按 name 匹配
        for c in stats['static_stats']['channels']:
            if c['name'] in self.static_channel_names:
                idx = self.static_channel_names.index(c['name'])
                self.s_means[idx] = c['mean']
                self.s_stds[idx] = c['std']
        
        # 转为 Tensor 方便广播计算
        self.d_means = torch.tensor(self.d_means).view(1, 1, -1, 1, 1).float() # (B, T, C, H, W)
        self.d_stds = torch.tensor(self.d_stds).view(1, 1, -1, 1, 1).float()
        self.s_means = torch.tensor(self.s_means).view(1, -1, 1, 1).float()    # (B, C, H, W)
        self.s_stds = torch.tensor(self.s_stds).view(1, -1, 1, 1).float()

    def _load_one_model(self, model_path, num_classes):
        """辅助函数：加载单个模型"""
        model = DualStreamSpatio_TemporalFusionNetwork(
            in_channels_dynamic=self.num_dyn_channels,
            in_channels_static=self.num_static_channels,
            num_classes=num_classes,
            classifier_hidden_dims=self.config.get('model.classifier.hidden_dims', [128, 64, 32])
        )
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            logger.error(f"加载模型失败 {model_path}: {e}")
            return None

    def _load_models(self):
        """加载大类模型和所有小类模型"""
        logger.info("正在加载模型权重...")
        
        # 1. 加载大类模型 (使用 Fold 1)
        major_path = self.run_dir / 'major_model' / 'fold_1' / 'best_model.pth'
        self.major_model = self._load_one_model(major_path, len(self.major_map))
        if not self.major_model:
            raise RuntimeError("无法加载大类模型！")

        # 2. 加载小类模型字典
        self.detail_models = {}
        for major_name, major_id in self.major_map.items():
            sub_dir = self.run_dir / f"detail_model_{major_id}_{major_name}"
            # 读取局部映射，确定 num_classes
            map_file = sub_dir / "class_mapping.json"
            model_file = sub_dir / "fold_1" / "best_model.pth"
            
            if map_file.exists() and model_file.exists():
                with open(map_file, 'r') as f:
                    mapping = json.load(f)
                local_map = mapping['local_to_global_map']
                # 小类模型输出维度 = 局部类别数
                sub_model = self._load_one_model(model_file, len(local_map))
                
                if sub_model:
                    # 存储模型和局部ID转全局ID的映射
                    self.detail_models[major_id] = {
                        'model': sub_model,
                        'map': {int(k): int(v) for k, v in local_map.items()}
                    }
                    logger.info(f"   + Loaded Detail Model: {major_name} (ID: {major_id})")

    def _read_data_window(self, window):
        """读取指定窗口的所有数据并组装成 Tensor"""
        w, h = int(window.width), int(window.height)
        
        # --- 1. 读取动态数据 (T, C, H, W) ---
        T = len(self.timeline)
        dyn_data = np.zeros((T, self.num_dyn_channels, h, w), dtype=np.float32)
        
        # 优化：不逐个打开文件，而是按时间循环
        for t_idx, date_obj in enumerate(self.timeline):
            if date_obj in self.dyn_files:
                for ch_idx, fpath in self.dyn_files[date_obj].items():
                    try:
                        with rasterio.open(fpath) as src:
                            # 确保窗口在图像范围内，处理边界
                            src_window = window.intersection(Window(0, 0, src.width, src.height))
                            data = src.read(1, window=src_window, boundless=True, fill_value=0)
                            
                            # 如果 read 出来的尺寸小于窗口 (边缘情况)，需要 resize 或 pad ?
                            # boundless=True 已经处理了填充，但 shape 必须对
                            if data.shape != (h, w):
                                temp = np.zeros((h, w), dtype=np.float32)
                                temp[:data.shape[0], :data.shape[1]] = data
                                data = temp
                                
                            dyn_data[t_idx, ch_idx] = data
                    except Exception as e:
                        pass # 缺失通道填充0

        # --- 2. 读取静态数据 (C, H, W) ---
        sta_data = np.zeros((self.num_static_channels, h, w), dtype=np.float32)
        for ch_idx, fpath in self.static_files.items():
            try:
                with rasterio.open(fpath) as src:
                    src_window = window.intersection(Window(0, 0, src.width, src.height))
                    data = src.read(1, window=src_window, boundless=True, fill_value=0)
                    if data.shape != (h, w):
                        temp = np.zeros((h, w), dtype=np.float32)
                        temp[:data.shape[0], :data.shape[1]] = data
                        data = temp
                    sta_data[ch_idx] = data
            except:
                pass

        # 转 Tensor
        dyn_tensor = torch.from_numpy(dyn_data).float().unsqueeze(0) # (1, T, C, H, W)
        sta_tensor = torch.from_numpy(sta_data).float().unsqueeze(0) # (1, C, H, W)
        
        return dyn_tensor, sta_tensor

    def predict_all(self, output_path, patch_size=64):
        """主预测循环"""
        output_path = Path(output_path)
        
        # 打开基准影像读取地理信息
        with rasterio.open(self.ref_meta.filepath) as src:
            profile = src.profile.copy()
            width, height = src.width, src.height
        
        # 更新 Profile
        profile.update(dtype=rasterio.int32, count=1, compress='lzw', nodata=-1)
        
        # 计算块的网格
        # 这里使用非重叠窗口，直接平铺。如果需要消除拼接缝隙，需要更复杂的重叠+中心裁剪策略
        # 模型 patch_size 一般较小 (32/64)，直接作为 stride
        stride = patch_size
        windows = []
        for r in range(0, height, stride):
            for c in range(0, width, stride):
                w = min(stride, width - c)
                h = min(stride, height - r)
                windows.append(Window(c, r, w, h))
                
        logger.info(f"开始预测: 图像大小 {width}x{height}, 总块数 {len(windows)}")
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            for win in tqdm(windows, desc="Processing Tiles"):
                # 1. 读取数据
                dyn, sta = self._read_data_window(win) # Returns CPU tensors
                
                # 2. 归一化 (使用 CPU 广播计算，避免 GPU 显存碎块化)
                if self.d_stds is not None:
                    # 确保 d_means 在 CPU
                    d_means_cpu = self.d_means.to('cpu')
                    d_stds_cpu = self.d_stds.to('cpu')
                    s_means_cpu = self.s_means.to('cpu')
                    s_stds_cpu = self.s_stds.to('cpu')
                    
                    dyn = (dyn - d_means_cpu) / (d_stds_cpu + 1e-6)
                    sta = (sta - s_means_cpu) / (s_stds_cpu + 1e-6)

                # 3. 移至 GPU
                dyn = dyn.to(self.device)
                sta = sta.to(self.device)
                
                # 4. 初始化输出 Patch (B, H, W)
                B, _, _, H_curr, W_curr = dyn.shape
                final_patch = torch.full((H_curr, W_curr), -1, dtype=torch.long, device=self.device)
                
                # ================= 级联预测逻辑 =================
                
                # Step A: 大类预测
                with torch.no_grad():
                    major_out = self.major_model(dyn, sta)
                    major_probs = torch.softmax(major_out['probabilities'], dim=1)
                    # (B, num_major) -> (B, num_major, 1, 1) -> 无法直接得到空间图
                    # 等等！您的模型是 DualStreamSpatio_TemporalFusionNetwork
                    # 它的输出是在 fusion 后接 classifier (Global Average Pooling 之后的)
                    # 还是说它是分割模型？
                    # 查看 model_architecture.py:
                    # fusion_net 中包含 nn.AdaptiveAvgPool2d((1, 1))
                    # 这意味着模型输出的是整个 Patch 的一个分类标签，而不是 Dense Map (H, W)！
                    
                    # 修正：如果模型是 Patch Classification (一个 Patch 一个标签)
                    # 我们只能给这个窗口内的所有像素赋同一个值。
                    # 这就是为什么 patch_size 只有 32 或 64 的原因。
                    
                    major_pred_label = torch.argmax(major_probs, dim=1).item() # 标量
                
                # Step B: 小类预测
                if major_pred_label in self.detail_models:
                    detail_entry = self.detail_models[major_pred_label]
                    model_d = detail_entry['model']
                    mapping_d = detail_entry['map']
                    
                    with torch.no_grad():
                        detail_out = model_d(dyn, sta)
                        detail_probs = torch.softmax(detail_out['probabilities'], dim=1)
                        local_pred = torch.argmax(detail_probs, dim=1).item()
                        
                    # 映射回全局 ID
                    global_id = mapping_d.get(local_pred, -1)
                    
                    # 填充整个 Patch
                    final_patch[:] = global_id
                else:
                    # 如果没有对应小类模型，或大类预测错误，可以填大类ID或 Nodata
                    # 这里填 nodata (-1) 或者某种指示值
                    final_patch[:] = -1 
                
                # ==============================================
                
                # 写入结果
                dst.write(final_patch.cpu().numpy().astype(rasterio.int32), 1, window=win)

        logger.info(f"✅ 预测完成！结果已保存至: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="生成植被分类栅格图")
    parser.add_argument('--run_dir', type=str, required=True, help='实验输出目录')
    parser.add_argument('--output', type=str, default='vegetation_map.tif', help='输出文件名')
    parser.add_argument('--patch_size', type=int, default=None, help='推理块大小，默认读取配置文件')
    args = parser.parse_args()
    
    # 确定 Patch Size
    predictor = RasterPredictor(args.run_dir)
    
    p_size = args.patch_size
    if p_size is None:
        p_size = predictor.config.get('data_specs.spatial.patch_size', 32)
        
    out_file = Path(args.run_dir) / args.output
    predictor.predict_all(out_file, patch_size=p_size)

if __name__ == "__main__":
    main()