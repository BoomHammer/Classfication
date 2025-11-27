#!/usr/bin/env python3
"""
main.py: “先大类，后小类” 分层训练流水线 (配置分离版)

更新内容：
1. 真正从 config.yaml 读取训练超参数。
2. 支持大类 (major_model) 和小类 (detail_model) 使用不同的超参数 (epochs, lr, batch_size 等)。
3. 保持了之前的自动归一化和多进程 DataLoader 优化。
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
    """
    辅助函数：遍历数据集，返回满足 filter_func 条件的局部索引列表。
    """
    indices = []
    df = dataset.encoder.get_dataframe()
    for local_idx, global_idx in enumerate(dataset.indices):
        row = df.iloc[global_idx]
        if filter_func(row):
            indices.append(local_idx) 
    return indices

def main():
    setup_logging()
    print("="*60)
    print("🚀 启动分层训练流水线 (配置分离 & 自动参数版)")
    print("="*60)

    # 1. 加载配置
    config = ConfigManager(str(Path(__file__).parent / 'config.yaml'))
    output_dir = config.get_experiment_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取分离的超参数配置
    # 如果配置文件里没写，这里提供了默认的 fallback 值，防止报错
    major_cfg = config.get('train.major_model', {
        'epochs': 30, 'batch_size': 32, 'learning_rate': 1e-3, 'weight_decay': 1e-4, 'patience': 10
    })
    detail_cfg = config.get('train.detail_model', {
        'epochs': 40, 'batch_size': 16, 'learning_rate': 1e-3, 'weight_decay': 1e-4, 'patience': 10, 'min_samples': 5
    })
    
    common_cfg = {
        'num_workers': config.get('train.num_workers', min(8, multiprocessing.cpu_count())),
        'pin_memory': True if torch.cuda.is_available() else False
    }

    print("\n📋 训练配置加载:")
    print(f"   [大类参数] Epochs: {major_cfg['epochs']}, BS: {major_cfg['batch_size']}, LR: {major_cfg['learning_rate']}")
    print(f"   [小类参数] Epochs: {detail_cfg['epochs']}, BS: {detail_cfg['batch_size']}, LR: {detail_cfg['learning_rate']}")
    print(f"   [系统参数] Workers: {common_cfg['num_workers']}, Pin Memory: {common_cfg['pin_memory']}")

    # 2. 初始化组件
    encoder = LabelEncoder(config=config)
    dynamic_crawler = RasterCrawler(config=config, raster_dir=config.get_resolved_path('dynamic_images_dir'), filename_pattern=config.get('data_specs.raster_crawler.filename_pattern'), file_extensions=['.tif'])
    static_crawler = RasterCrawler(config=config, raster_dir=config.get_resolved_path('static_images_dir'), filename_pattern=config.get('data_specs.raster_crawler.filename_pattern'), file_extensions=['.tif'])
    
    # 3. 自动归一化计算 (如果不存在)
    stats_file = output_dir / 'normalization_stats.json'
    if not stats_file.exists():
        print("\n📊 未检测到统计文件，开始计算全局统计量...")
        calculator = StatsCalculator(config=config, dynamic_channel_names=None, static_channel_names=None)
        d_rasters = dynamic_crawler.get_all_rasters()
        s_rasters = static_crawler.get_all_rasters()
        calculator.compute_global_stats(dynamic_rasters=d_rasters, static_rasters=s_rasters, sampling_rate=0.2)
        calculator.save_stats('normalization_stats.json')
    else:
        print(f"\n✅ 检测到统计文件: {stats_file.name}，跳过计算。")

    # 获取通道数
    dyn_ch = dynamic_crawler.detect_num_channels()['most_common']
    sta_ch = static_crawler.detect_num_channels()['most_common']
    
    # 4. 初始化全量数据集
    print("\n📦 初始化全量数据集...")
    full_train_dataset = PointTimeSeriesDataset(config, encoder, dynamic_crawler, static_crawler, split='train', cache_metadata=True, verbose=False)
    full_val_dataset = PointTimeSeriesDataset(config, encoder, dynamic_crawler, static_crawler, split='val', cache_metadata=True, verbose=False)
    
    major_map = encoder.get_major_labels_map()
    hierarchical_map = encoder.get_hierarchical_map()

    # =========================================================================
    # 阶段 A: 训练大类模型 (使用 major_cfg)
    # =========================================================================
    print("\n" + "="*60)
    print("🏗️  [阶段 A] 训练大类分类模型 (Major Model)")
    print("="*60)
    
    major_model_dir = output_dir / "major_model"
    major_model = DualStreamSpatio_TemporalFusionNetwork(
        in_channels_dynamic=dyn_ch,
        in_channels_static=sta_ch,
        num_classes=len(major_map)
    )
    
    # 构造大类专用的 DataLoader
    major_loader_args = {
        'batch_size': major_cfg['batch_size'],
        'num_workers': common_cfg['num_workers'],
        'pin_memory': common_cfg['pin_memory'],
        'collate_fn': collate_fn
    }
    
    major_trainer = Trainer(
        model=major_model,
        train_dataloader=DataLoader(full_train_dataset, shuffle=True, **major_loader_args),
        val_dataloader=DataLoader(full_val_dataset, shuffle=False, **major_loader_args),
        num_classes=len(major_map),
        target_key='major_label', 
        output_dir=major_model_dir
    )
    
    major_trainer.train(
        num_epochs=major_cfg['epochs'],
        lr=major_cfg['learning_rate'],
        weight_decay=major_cfg['weight_decay'],
        patience=major_cfg['patience']
    )
    print(f"✅ 大类模型训练完成，保存于: {major_model_dir}")

    # =========================================================================
    # 阶段 B: 训练各个小类模型 (使用 detail_cfg)
    # =========================================================================
    print("\n" + "="*60)
    print("🏗️  [阶段 B] 训练各分支小类模型 (Detail Models)")
    print("="*60)

    # 构造小类专用的 DataLoader 参数
    detail_loader_args = {
        'batch_size': detail_cfg['batch_size'], # 使用小类配置的 BatchSize
        'num_workers': common_cfg['num_workers'],
        'pin_memory': common_cfg['pin_memory'],
        'collate_fn': collate_fn
    }

    for major_name, major_id in major_map.items():
        print(f"\n👉 正在处理大类: {major_name} (ID: {major_id})")
        
        # 获取小类信息
        sub_info = hierarchical_map[major_name]
        detail_classes_map = sub_info['detail_classes']
        num_sub_classes = len(detail_classes_map)
        
        if num_sub_classes <= 1:
            print(f"   ⚠️ 该大类仅有 {num_sub_classes} 个小类，跳过训练。")
            continue
            
        print(f"   包含小类: {list(detail_classes_map.keys())} (共 {num_sub_classes} 个)")

        # 构建映射
        sorted_details = sorted(detail_classes_map.items(), key=lambda x: x[1])
        global_to_local = {gid: lidx for lidx, (_, gid) in enumerate(sorted_details)}
        local_to_global = {lidx: gid for lidx, (_, gid) in enumerate(sorted_details)}
            
        # 筛选数据子集
        train_indices = get_subset_indices(full_train_dataset, lambda row: row['major_label'] == major_id)
        val_indices = get_subset_indices(full_val_dataset, lambda row: row['major_label'] == major_id)
        
        print(f"   样本数量: 训练集 {len(train_indices)} | 验证集 {len(val_indices)}")
        
        # 使用配置中的最小样本数限制
        min_samples = detail_cfg.get('min_samples', 5)
        if len(train_indices) < min_samples:
            print(f"   ⚠️ 样本过少 (<{min_samples})，跳过训练。")
            continue

        train_subset = Subset(full_train_dataset, train_indices)
        val_subset = Subset(full_val_dataset, val_indices)
        
        # 初始化子模型
        sub_model_dir = output_dir / f"detail_model_{major_id}_{major_name}"
        sub_model = DualStreamSpatio_TemporalFusionNetwork(
            in_channels_dynamic=dyn_ch,
            in_channels_static=sta_ch,
            num_classes=num_sub_classes
        )
        
        # 训练子模型
        sub_trainer = Trainer(
            model=sub_model,
            train_dataloader=DataLoader(train_subset, shuffle=True, **detail_loader_args),
            val_dataloader=DataLoader(val_subset, shuffle=False, **detail_loader_args),
            num_classes=num_sub_classes,
            target_key='detail_label',
            label_mapping=global_to_local,
            output_dir=sub_model_dir
        )
        
        sub_trainer.train(
            num_epochs=detail_cfg['epochs'],
            lr=detail_cfg['learning_rate'],
            weight_decay=detail_cfg['weight_decay'],
            patience=detail_cfg['patience']
        )
        
        # 保存映射
        mapping_info = {
            'major_class': major_name,
            'major_id': major_id,
            'local_to_global_map': local_to_global, 
            'global_to_local_map': global_to_local
        }
        with open(sub_model_dir / 'class_mapping.json', 'w', encoding='utf-8') as f:
            json.dump(mapping_info, f, ensure_ascii=False, indent=2)
            
        print(f"   ✅ {major_name} 小类模型训练完成。")

    print("\n" + "="*60)
    print("🎉 所有模型训练结束！")
    print("="*60)

if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()