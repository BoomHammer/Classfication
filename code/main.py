#!/usr/bin/env python3
"""
main.py: “先大类，后小类” 分层训练流水线

逻辑流程：
1. 准备全量数据。
2. 【阶段A】训练“大类分类器” (Major Class Model)
   - 数据：全量数据
   - 标签：major_label
   - 输出：major_model.pth
3. 【阶段B】训练多个“小类分类器” (Detail Class Models)
   - 循环遍历每个大类 ID
   - 筛选数据：仅保留属于当前大类的样本
   - 标签：detail_label (需重映射为 0~N)
   - 输出：detail_model_major_{id}.pth
"""

import sys
import json
import logging
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

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

def get_subset_indices(dataset, filter_func):
    """
    辅助函数：遍历数据集，返回满足 filter_func 条件的局部索引列表。
    
    修正说明：
    之前直接遍历 df 返回的是全局索引，会导致 Subset 越界。
    现在遍历 dataset.indices (当前 split 的全局索引列表)，并返回 local_idx (枚举索引)。
    """
    indices = []
    # print("  🔍 正在筛选数据子集...") # 减少刷屏
    
    # 获取原始的完整 DataFrame
    df = dataset.encoder.get_dataframe()
    
    # dataset.indices 存储了当前 split (如训练集) 对应在 DataFrame 中的全局行号
    # 我们需要返回 dataset 内部的局部索引 (0 ~ len(dataset)-1)
    # enumerate 的 local_idx 就是我们要传给 Subset 的索引
    for local_idx, global_idx in enumerate(dataset.indices):
        # 使用 iloc 通过行号访问原始数据
        row = df.iloc[global_idx]
        if filter_func(row):
            indices.append(local_idx) # 注意：这里必须存 local_idx
            
    return indices

def main():
    setup_logging()
    print("="*60)
    print("🚀 启动分层训练流水线 (Coarse-to-Fine)")
    print("="*60)

    # 1. 配置与数据准备
    config = ConfigManager(str(Path(__file__).parent / 'config.yaml'))
    output_dir = config.get_experiment_output_dir()
    
    # 初始化组件
    encoder = LabelEncoder(config=config)
    dynamic_crawler = RasterCrawler(config=config, raster_dir=config.get_resolved_path('dynamic_images_dir'), filename_pattern=config.get('data_specs.raster_crawler.filename_pattern'), file_extensions=['.tif'])
    static_crawler = RasterCrawler(config=config, raster_dir=config.get_resolved_path('static_images_dir'), filename_pattern=config.get('data_specs.raster_crawler.filename_pattern'), file_extensions=['.tif'])
    
    # 获取通道数
    dyn_ch = dynamic_crawler.detect_num_channels()['most_common']
    sta_ch = static_crawler.detect_num_channels()['most_common']
    
    # 初始化全量数据集
    print("\n📦 初始化全量数据集...")
    full_train_dataset = PointTimeSeriesDataset(config, encoder, dynamic_crawler, static_crawler, split='train', cache_metadata=True, verbose=False)
    full_val_dataset = PointTimeSeriesDataset(config, encoder, dynamic_crawler, static_crawler, split='val', cache_metadata=True, verbose=False)
    
    major_map = encoder.get_major_labels_map()
    hierarchical_map = encoder.get_hierarchical_map()

    # =========================================================================
    # 阶段 A: 训练大类模型 (Major Model)
    # =========================================================================
    print("\n" + "="*60)
    print("🏗️  [阶段 A] 训练大类分类模型 (Major Model)")
    print("="*60)
    
    major_model_dir = output_dir / "major_model"
    major_model = DualStreamSpatio_TemporalFusionNetwork(
        in_channels_dynamic=dyn_ch,
        in_channels_static=sta_ch,
        num_classes=len(major_map) # 输出节点数 = 大类数
    )
    
    major_trainer = Trainer(
        model=major_model,
        train_dataloader=DataLoader(full_train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn),
        val_dataloader=DataLoader(full_val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn),
        num_classes=len(major_map),
        target_key='major_label', # 告诉 Trainer 取 batch['major_label']
        output_dir=major_model_dir
    )
    
    # 你可以根据需要取消注释这一行来跳过大类训练（如果已经训练好了）
    major_trainer.train(num_epochs=30) 
    print(f"✅ 大类模型训练完成，保存于: {major_model_dir}")

    # =========================================================================
    # 阶段 B: 训练各个小类模型 (Detail Models)
    # =========================================================================
    print("\n" + "="*60)
    print("🏗️  [阶段 B] 训练各分支小类模型 (Detail Models)")
    print("="*60)

    for major_name, major_id in major_map.items():
        print(f"\n👉 正在处理大类: {major_name} (ID: {major_id})")
        
        # 1. 获取该大类下的小类信息
        sub_info = hierarchical_map[major_name]
        detail_classes_map = sub_info['detail_classes'] # {小类名: 全局ID}
        num_sub_classes = len(detail_classes_map)
        
        if num_sub_classes <= 1:
            print(f"   ⚠️ 该大类仅有 {num_sub_classes} 个小类，跳过训练。")
            continue
            
        print(f"   包含小类: {list(detail_classes_map.keys())} (共 {num_sub_classes} 个)")

        # 2. 构建本地映射 (Local ID Mapping)
        sorted_details = sorted(detail_classes_map.items(), key=lambda x: x[1]) # 按全局ID排序
        
        global_to_local = {}
        local_to_global = {}
        for local_idx, (d_name, global_id) in enumerate(sorted_details):
            global_to_local[global_id] = local_idx
            local_to_global[local_idx] = global_id
            
        # 3. 筛选数据子集 (Subset)
        print("   🔍 正在筛选数据子集...")
        # 注意：这里调用的是修复后的 get_subset_indices
        train_indices = get_subset_indices(full_train_dataset, lambda row: row['major_label'] == major_id)
        val_indices = get_subset_indices(full_val_dataset, lambda row: row['major_label'] == major_id)
        
        print(f"   样本数量: 训练集 {len(train_indices)} | 验证集 {len(val_indices)}")
        
        if len(train_indices) < 5:
            print("   ⚠️ 样本过少，跳过训练。")
            continue

        train_subset = Subset(full_train_dataset, train_indices)
        val_subset = Subset(full_val_dataset, val_indices)
        
        # 4. 初始化子模型
        sub_model_dir = output_dir / f"detail_model_{major_id}_{major_name}"
        sub_model = DualStreamSpatio_TemporalFusionNetwork(
            in_channels_dynamic=dyn_ch,
            in_channels_static=sta_ch,
            num_classes=num_sub_classes # 输出节点数 = 本地小类数
        )
        
        # 5. 训练子模型
        sub_trainer = Trainer(
            model=sub_model,
            train_dataloader=DataLoader(train_subset, batch_size=16, shuffle=True, collate_fn=collate_fn), # 子集可能较小，BatchSize减小
            val_dataloader=DataLoader(val_subset, batch_size=16, shuffle=False, collate_fn=collate_fn),
            num_classes=num_sub_classes,
            target_key='detail_label', # 取小类标签
            label_mapping=global_to_local, # 传入映射表，Trainer会自动将全局ID转为本地0~N
            output_dir=sub_model_dir
        )
        
        sub_trainer.train(num_epochs=40)
        
        # 6. 保存子模型的映射关系，以便推理时使用
        mapping_info = {
            'major_class': major_name,
            'major_id': major_id,
            'local_to_global_map': local_to_global, # 推理输出 0 -> 对应的全局ID
            'global_to_local_map': global_to_local
        }
        with open(sub_model_dir / 'class_mapping.json', 'w', encoding='utf-8') as f:
            json.dump(mapping_info, f, ensure_ascii=False, indent=2)
            
        print(f"   ✅ {major_name} 小类模型训练完成。")

    print("\n" + "="*60)
    print("🎉 所有模型训练结束！")
    print("="*60)

if __name__ == '__main__':
    main()