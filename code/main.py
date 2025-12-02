#!/usr/bin/env python3
"""
main.py: “先大类，后小类” 分层训练流水线 (适配 Super-Channel 策略二)

【核心变更】
1. 移除了独立的 static_crawler，统一使用新的 RasterCrawler 进行多源异构数据管理。
2. 适配了新的 PointTimeSeriesDataset 接口 (基于 Super-Channel 对齐)。
3. 适配了新的 StatsCalculator 接口 (基于变量名聚合)。
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
    # 注意：这里假设 encoder.get_dataframe() 返回的顺序与 dataset.indices 的全局顺序一致
    # dataset.points_df 是在 init 时 copy 过来的，所以直接用 dataset.points_df 更安全
    df = dataset.points_df 
    
    for local_idx, global_idx in enumerate(dataset.indices):
        row = df.iloc[global_idx]
        if filter_func(row):
            indices.append(local_idx) 
    return indices

def main():
    setup_logging()
    print("="*60)
    print("🚀 启动分层训练流水线 (Super-Channel 融合版)")
    print("="*60)

    # 1. 加载配置
    config = ConfigManager(str(Path(__file__).parent / 'config.yaml'))
    output_dir = config.get_experiment_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取超参数 (带默认值)
    major_cfg = config.get('train.major_model', {
        'epochs': 30, 'batch_size': 32, 'learning_rate': 1e-3, 'weight_decay': 1e-4, 'patience': 10
    })
    detail_cfg = config.get('train.detail_model', {
        'epochs': 40, 'batch_size': 16, 'learning_rate': 1e-3, 'weight_decay': 1e-4, 'patience': 10, 'min_samples': 5
    })
    common_cfg = {
        'num_workers': config.get('train.num_workers', 0), # Windows调试建议设为0
        'pin_memory': True if torch.cuda.is_available() else False
    }

    print("\n📋 训练配置:")
    print(f"   [大类] Epochs: {major_cfg['epochs']}, BS: {major_cfg['batch_size']}, LR: {major_cfg['learning_rate']}")
    print(f"   [小类] Epochs: {detail_cfg['epochs']}, BS: {detail_cfg['batch_size']}, LR: {detail_cfg['learning_rate']}")

    # 2. 初始化核心组件
    encoder = LabelEncoder(config=config)
    
    # [说明] 训练阶段不需要 crawler，因为直接读取预处理后的 .pt 文件
    # 但如果缺少统计文件，下面会临时创建一个 crawler 来计算
    crawler = None 

    # 3. 自动归一化计算 (如果不存在)
    stats_file = output_dir / 'normalization_stats.json'
    if not stats_file.exists():
        print("\n📊 未检测到统计文件，正在从原始 TIFF 计算全局统计量...")
        print("   (这一步可能需要几分钟，计算完成后将永久保存)")
        
        # [核心修复] 临时初始化爬虫，仅用于统计计算
        # 必须传入 config 才能找到 dynamic_images_dir
        temp_crawler = RasterCrawler(config=config)
        
        calculator = StatsCalculator(config=config)
        # 使用 temp_crawler 进行计算
        calculator.compute_global_stats(temp_crawler, sampling_rate=0.2) 
        calculator.save_stats('normalization_stats.json')
        
        print("✅ 统计量计算完成并保存。")
        # 释放内存
        del temp_crawler
        del calculator
    else:
        print(f"\n✅ 检测到统计文件: {stats_file.name}，跳过计算。")

    # 4. 初始化全量数据集
    print("\n📦 加载预处理数据集...")
    try:
        full_train_dataset = PointTimeSeriesDataset(config, encoder, crawler=None, split='train')
        full_val_dataset = PointTimeSeriesDataset(config, encoder, crawler=None, split='val')
    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}")
        print("💡 请先运行: python code/preprocess_dataset.py")
        sys.exit(1)
    
    # 获取通道数信息
    dyn_ch = full_train_dataset.num_channels
    # [注意] 目前 PointTimeSeriesDataset 对静态数据使用占位符 (zeros)，通道数为 1
    # 如果后续你完善了静态数据逻辑，这里需要修改
    sta_ch = 1 
    
    print(f"   动态通道数: {dyn_ch} (包含变量: {list(full_train_dataset.channel_map.keys())})")
    print(f"   静态通道数: {sta_ch} (Placeholder)")

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
    print(f"✅ 大类模型保存于: {major_model_dir}")

    # =========================================================================
    # 阶段 B: 训练小类模型 (Detail Models)
    # =========================================================================
    print("\n" + "="*60)
    print("🏗️  [阶段 B] 训练各分支小类模型")
    print("="*60)

    detail_loader_args = {
        'batch_size': detail_cfg['batch_size'],
        'num_workers': common_cfg['num_workers'],
        'pin_memory': common_cfg['pin_memory'],
        'collate_fn': collate_fn
    }

    for major_name, major_id in major_map.items():
        print(f"\n👉 处理大类: {major_name} (ID: {major_id})")
        
        sub_info = hierarchical_map[major_name]
        detail_classes_map = sub_info['detail_classes']
        num_sub_classes = len(detail_classes_map)
        
        if num_sub_classes <= 1:
            print(f"   ⚠️ 该大类仅有 {num_sub_classes} 个小类，跳过。")
            continue

        # 映射构建
        sorted_details = sorted(detail_classes_map.items(), key=lambda x: x[1])
        global_to_local = {gid: lidx for lidx, (_, gid) in enumerate(sorted_details)}
        local_to_global = {lidx: gid for lidx, (_, gid) in enumerate(sorted_details)}
            
        # 筛选子集
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