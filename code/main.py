#!/usr/bin/env python3
"""
main.py: “先大类，后小类” 分层训练流水线 (修复通道检测版)
"""

import sys
import json
import logging
import multiprocessing
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np

# 导入本地模块
sys.path.insert(0, str(Path(__file__).parent))
from config_manager import ConfigManager
from label_encoder import LabelEncoder
from raster_crawler import RasterCrawler
from point_timeseries_dataset import PointTimeSeriesDataset, collate_fn
from model_architecture import DualStreamSpatio_TemporalFusionNetwork
from trainer import Trainer
from stats_calculator import StatsCalculator

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

def get_subset_indices(dataset, filter_func):
    """辅助函数：遍历数据集，返回满足条件的局部索引列表"""
    indices = []
    df = dataset.points_df 
    for local_idx, global_idx in enumerate(dataset.indices):
        row = df.iloc[global_idx]
        if filter_func(row):
            indices.append(local_idx) 
    return indices

def main():
    setup_logging()
    print("="*60)
    print("🚀 启动分层训练流水线 (Auto-Channel Detect)")
    print("="*60)

    # 1. 加载配置
    config = ConfigManager(str(Path(__file__).parent / 'config.yaml'))
    output_dir = config.get_experiment_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取超参数
    major_cfg = config.get('train.major_model', {
        'epochs': 30, 'batch_size': 32, 'learning_rate': 1e-3, 'weight_decay': 1e-4, 'patience': 10
    })
    detail_cfg = config.get('train.detail_model', {
        'epochs': 40, 'batch_size': 16, 'learning_rate': 1e-3, 'weight_decay': 1e-4, 'patience': 10, 'min_samples': 5
    })
    common_cfg = {
        'num_workers': config.get('train.num_workers', 0),
        'pin_memory': True if torch.cuda.is_available() else False
    }

    # 2. 初始化组件
    encoder = LabelEncoder(config=config)
    
    # 3. 自动归一化计算 (仅当文件不存在时)
    stats_file = output_dir / 'normalization_stats.json'
    if not stats_file.exists():
        # 检查是否在之前的运行目录中有（可选优化，这里直接从tiff计算更稳）
        print("\n📊 正在计算全局统计量 (动态+静态)...")
        dyn_crawler = RasterCrawler(config=config)
        static_crawler = RasterCrawler(
            config=config, 
            raster_dir=config.get_resolved_path('static_images_dir'),
            filename_pattern='.*',
            file_extensions=('.tif', '.tiff')
        )
        calculator = StatsCalculator(config=config)
        calculator.compute_all_stats(dyn_crawler, static_crawler, sampling_rate=0.2) 
        print("✅ 统计量计算完成。")
        del dyn_crawler, static_crawler, calculator

    # 4. 加载数据集
    print("\n📦 加载预处理数据集...")
    try:
        full_train_dataset = PointTimeSeriesDataset(config, encoder, split='train')
        full_val_dataset = PointTimeSeriesDataset(config, encoder, split='val')
    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}")
        print("💡 请先运行: python code/preprocess_dataset.py")
        sys.exit(1)
    
    # 5. [关键修改] 直接从数据集获取通道参数，不再依赖可能丢失的json文件
    dyn_ch = full_train_dataset.num_channels
    sta_ch = full_train_dataset.num_static_channels
    
    print(f"   动态通道数: {dyn_ch}")
    print(f"   静态通道数: {sta_ch}")
    
    if sta_ch == 0:
        print("⚠️ 警告：检测到静态通道数为 0，请检查 preprocess_dataset.py 是否正确读取了静态数据。")
        # 如果确实是0，为了防止模型报错，可能需要特殊处理，但这里先让它跑，看是否报错

    major_map = encoder.get_major_labels_map()
    hierarchical_map = encoder.get_hierarchical_map()

    # =========================================================================
    # 阶段 A: 训练大类模型 (Major Model)
    # =========================================================================
    print("\n" + "="*60)
    print("🏗️  [阶段 A] 训练大类分类模型")
    print("="*60)
    
    major_model_dir = output_dir / "major_model"
    major_model = DualStreamSpatio_TemporalFusionNetwork(
        in_channels_dynamic=dyn_ch,
        in_channels_static=sta_ch,
        num_classes=len(major_map)
    )
    
    major_trainer = Trainer(
        model=major_model,
        train_dataloader=DataLoader(full_train_dataset, shuffle=True, batch_size=major_cfg['batch_size'], collate_fn=collate_fn, **common_cfg),
        val_dataloader=DataLoader(full_val_dataset, shuffle=False, batch_size=major_cfg['batch_size'], collate_fn=collate_fn, **common_cfg),
        num_classes=len(major_map),
        target_key='major_label',
        output_dir=major_model_dir
    )
    
    major_trainer.train(
        num_epochs=major_cfg['epochs'],
        learning_rate=major_cfg['learning_rate'],
        weight_decay=major_cfg['weight_decay'],
        patience=major_cfg['patience']
    )
    print(f"✅ 大类模型保存于: {major_model_dir}")

    # =========================================================================
    # 阶段 B: 训练小类模型 (Detail Models)
    # =========================================================================
    print("\n" + "="*60)
    print("🏗️  [阶段 B] 训练各分支小类模型")
    print("="*60)

    for major_name, major_id in major_map.items():
        print(f"\n👉 处理大类: {major_name} (ID: {major_id})")
        
        sub_info = hierarchical_map[major_name]
        detail_classes_map = sub_info['detail_classes']
        num_sub_classes = len(detail_classes_map)
        
        if num_sub_classes <= 1:
            print(f"   ⚠️ 该大类仅有 {num_sub_classes} 个小类，跳过。")
            continue

        sorted_details = sorted(detail_classes_map.items(), key=lambda x: x[1])
        global_to_local = {gid: lidx for lidx, (_, gid) in enumerate(sorted_details)}
        local_to_global = {lidx: gid for lidx, (_, gid) in enumerate(sorted_details)}
            
        train_indices = get_subset_indices(full_train_dataset, lambda row: row['major_label'] == major_id)
        val_indices = get_subset_indices(full_val_dataset, lambda row: row['major_label'] == major_id)
        
        print(f"   样本数: Train {len(train_indices)} | Val {len(val_indices)}")
        
        if len(train_indices) < detail_cfg.get('min_samples', 5):
            print("   ⚠️ 样本不足，跳过。")
            continue

        train_subset = Subset(full_train_dataset, train_indices)
        val_subset = Subset(full_val_dataset, val_indices)
        
        sub_model_dir = output_dir / f"detail_model_{major_id}_{major_name}"
        sub_model = DualStreamSpatio_TemporalFusionNetwork(
            in_channels_dynamic=dyn_ch,
            in_channels_static=sta_ch,
            num_classes=num_sub_classes
        )
        
        sub_trainer = Trainer(
            model=sub_model,
            train_dataloader=DataLoader(train_subset, shuffle=True, batch_size=detail_cfg['batch_size'], collate_fn=collate_fn, **common_cfg),
            val_dataloader=DataLoader(val_subset, shuffle=False, batch_size=detail_cfg['batch_size'], collate_fn=collate_fn, **common_cfg),
            num_classes=num_sub_classes,
            target_key='detail_label',
            label_mapping=global_to_local,
            output_dir=sub_model_dir
        )
        
        sub_trainer.train(
            num_epochs=detail_cfg['epochs'],
            learning_rate=detail_cfg['learning_rate'],
            weight_decay=detail_cfg['weight_decay'],
            patience=detail_cfg['patience']
        )
        
        with open(sub_model_dir / 'class_mapping.json', 'w', encoding='utf-8') as f:
            json.dump({
                'major_class': major_name,
                'major_id': major_id,
                'local_to_global_map': local_to_global, 
                'global_to_local_map': global_to_local
            }, f, ensure_ascii=False, indent=2)
            
        print(f"   ✅ {major_name} 小类模型完成。")

    print("\n" + "="*60)
    print("🎉 训练流水线结束！")
    print("="*60)

if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()