#!/usr/bin/env python3
"""
main.py: “先大类，后小类” 分层训练流水线 (修复 BatchNorm 单样本 Batch 问题)
"""

import sys
import json
import logging
import multiprocessing
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
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

def compute_class_weights(dataset, label_key, num_classes):
    """计算类别权重，用于处理类不平衡问题
    [改进] 使用平衡权重公式，并进行归一化防止loss爆炸
    权重公式: w_i = (1 - beta) / (1 - beta^{n_i})，其中 beta = (N-1)/N
    """
    class_counts = np.zeros(num_classes)
    
    # 处理 Subset 对象：提取原始 dataset 和索引映射
    if hasattr(dataset, 'dataset'):
        # Subset 对象
        original_dataset = dataset.dataset
        original_indices = original_dataset.indices
        subset_indices = dataset.indices
        df = original_dataset.points_df
        
        # Subset中的indices是original_dataset中的局部索引
        # 需要映射到original_dataset的global_idx
        for local_idx in subset_indices:
            global_idx = original_indices[local_idx]
            row = df.iloc[global_idx]
            label = int(row[label_key])
            if 0 <= label < num_classes:
                class_counts[label] += 1
    else:
        # 原始 Dataset 对象
        df = dataset.points_df
        for local_idx, global_idx in enumerate(dataset.indices):
            row = df.iloc[global_idx]
            label = int(row[label_key])
            if 0 <= label < num_classes:
                class_counts[label] += 1
    
    # [改进] 使用平衡权重公式
    total_samples = class_counts.sum()
    
    # 方案1：简单反向频率权重（稳定版本）
    # 权重 = 平均样本数 / 该类样本数
    avg_count = total_samples / (num_classes + 1e-6)
    weights = np.ones(num_classes)
    for i in range(num_classes):
        if class_counts[i] > 0:
            weights[i] = avg_count / class_counts[i]
        else:
            weights[i] = 1.0  # 类别不存在时设为1.0
    
    # [关键修复] 归一化权重，使得平均权重为1，防止loss过大
    weights = weights / (weights.mean() + 1e-8)
    
    # [防护] 限制权重范围 [0.1, 10.0]，防止极端不平衡类的权重过大
    weights = np.clip(weights, 0.1, 10.0)
    
    weights = torch.from_numpy(weights).float()
    
    return weights

def main():
    setup_logging()
    print("="*60)
    print("🚀 启动分层训练流水线 (Fix: BatchNorm Drop Last)")
    print("="*60)

    # 1. 加载配置
    config = ConfigManager(str(Path(__file__).parent / 'config.yaml'), create_experiment_dir=True)
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
    
    # 3. 自动归一化计算
    stats_file = output_dir / 'normalization_stats.json'
    if not stats_file.exists():
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
    
    # 5. 获取通道参数
    dyn_ch = full_train_dataset.num_channels
    sta_ch = full_train_dataset.num_static_channels
    
    print(f"   动态通道数: {dyn_ch}")
    print(f"   静态通道数: {sta_ch}")
    
    if sta_ch == 0:
        print("⚠️ 警告：检测到静态通道数为 0，请检查 preprocess_dataset.py 是否正确读取了静态数据。")

    major_map = encoder.get_major_labels_map()
    hierarchical_map = encoder.get_hierarchical_map()

    # =========================================================================
    # 阶段 A: 训练大类模型 (Major Model) - K-Fold 交叉验证
    # =========================================================================
    print("\n" + "="*60)
    print("🏗️  [阶段 A] 训练大类分类模型 (K-Fold 交叉验证)")
    print("="*60)
    
    # 读取 K-Fold 配置
    kfold_config = config.get('train.kfold', {})
    major_kfold_n_splits = kfold_config.get('n_splits', 5)
    major_kfold_random_state = kfold_config.get('random_state', 42)
    
    # 计算大类权重
    major_weights = compute_class_weights(full_train_dataset, 'major_label', len(major_map))
    print(f"📊 大类权重: {major_weights.tolist()}")
    
    major_label_smoothing = major_cfg.get('label_smoothing', config.get('model.label_smoothing', 0.05))
    major_model_dir = output_dir / "major_model"
    
    print(f"\n📊 启用 K-Fold 交叉验证 (n_splits={major_kfold_n_splits})")
    major_model = DualStreamSpatio_TemporalFusionNetwork(
        in_channels_dynamic=dyn_ch,
        in_channels_static=sta_ch,
        num_classes=len(major_map),
        dropout=config.get('model.dropout', 0.25),
        classifier_hidden_dims=config.get('model.classifier.hidden_dims', [128, 64, 32])
    )
    
    major_trainer = Trainer(
        model=major_model,
        train_dataloader=None,  # K-Fold 内部会创建
        val_dataloader=None,
        num_classes=len(major_map),
        target_key='major_label',
        output_dir=major_model_dir,
        class_weights=major_weights,
        use_focal_loss=True,
        label_smoothing=major_label_smoothing,
        model_init_params={  # 传入模型初始化参数
            'in_channels_dynamic': dyn_ch,
            'in_channels_static': sta_ch,
            'num_classes': len(major_map),
            'dropout': config.get('model.dropout', 0.25),
            'classifier_hidden_dims': config.get('model.classifier.hidden_dims', [128, 64, 32])
        }
    )
    
    kfold_results = major_trainer.train_with_kfold(
        dataset=full_train_dataset,
        num_epochs=major_cfg['epochs'],
        learning_rate=major_cfg['learning_rate'],
        weight_decay=major_cfg['weight_decay'],
        patience=major_cfg['patience'],
        n_splits=major_kfold_n_splits,
        random_state=major_kfold_random_state,
        debug=False,
        accumulation_steps=1,
        batch_size=major_cfg['batch_size']
    )
    
    print(f"✅ 大类模型 K-Fold 训练完成")
    print(f"   平均精度: {kfold_results['mean_metrics'].get('accuracy', 0):.4f} ± {kfold_results['std_metrics'].get('accuracy_std', 0):.4f}")
    print(f"✅ 大类模型保存于: {major_model_dir}")

    # =========================================================================
    # 阶段 B: 训练小类模型 (Detail Models) - K-Fold 交叉验证
    # =========================================================================
    print("\n" + "="*60)
    print("🏗️  [阶段 B] 训练各分支小类模型 (K-Fold 交叉验证)")
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
        
        # 计算当前小类的类别权重
        detail_weights = compute_class_weights(train_subset, 'detail_label', num_sub_classes)
        print(f"   📊 小类权重: {detail_weights.tolist()}")
        
        detail_label_smoothing = detail_cfg.get('label_smoothing', config.get('model.label_smoothing', 0.1))
        
        sub_model_dir = output_dir / f"detail_model_{major_id}_{major_name}"
        
        # 检查样本是否充足进行 K-Fold
        if len(train_indices) >= major_kfold_n_splits:
            # 样本充足，使用 K-Fold
            print(f"   📊 启用 K-Fold 交叉验证 (n_splits={major_kfold_n_splits})")
            
            sub_model = DualStreamSpatio_TemporalFusionNetwork(
                in_channels_dynamic=dyn_ch,
                in_channels_static=sta_ch,
                num_classes=num_sub_classes,
                dropout=config.get('model.dropout', 0.25),
                classifier_hidden_dims=config.get('model.classifier.hidden_dims', [128, 64, 32])
            )
            
            sub_trainer = Trainer(
                model=sub_model,
                train_dataloader=None,  # K-Fold 内部会创建
                val_dataloader=None,
                num_classes=num_sub_classes,
                target_key='detail_label',
                label_mapping=global_to_local,
                output_dir=sub_model_dir,
                class_weights=detail_weights,
                use_focal_loss=True,
                label_smoothing=detail_label_smoothing,
                model_init_params={  # 传入模型初始化参数
                    'in_channels_dynamic': dyn_ch,
                    'in_channels_static': sta_ch,
                    'num_classes': num_sub_classes,
                    'dropout': config.get('model.dropout', 0.25),
                    'classifier_hidden_dims': config.get('model.classifier.hidden_dims', [128, 64, 32])
                }
            )
            
            kfold_results = sub_trainer.train_with_kfold(
                dataset=train_subset,
                num_epochs=detail_cfg['epochs'],
                learning_rate=detail_cfg['learning_rate'],
                weight_decay=detail_cfg['weight_decay'],
                patience=detail_cfg['patience'],
                n_splits=major_kfold_n_splits,
                random_state=major_kfold_random_state,
                debug=False,
                accumulation_steps=1,
                batch_size=detail_cfg['batch_size']
            )
            
            print(f"   ✅ K-Fold 训练完成 | 平均精度: {kfold_results['mean_metrics'].get('accuracy', 0):.4f}")
        else:
            # 样本不足，自动降级到常规训练
            print(f"   ⏭️  样本数({len(train_indices)}) < K-Fold 折数({major_kfold_n_splits})，使用常规训练")
            
            sub_model = DualStreamSpatio_TemporalFusionNetwork(
                in_channels_dynamic=dyn_ch,
                in_channels_static=sta_ch,
                num_classes=num_sub_classes,
                dropout=config.get('model.dropout', 0.25),
                classifier_hidden_dims=config.get('model.classifier.hidden_dims', [128, 64, 32])
            )
            
            # 动态决定是否 drop_last
            use_drop_last = len(train_indices) > detail_cfg['batch_size']
            
            sub_trainer = Trainer(
                model=sub_model,
                train_dataloader=DataLoader(
                    train_subset, 
                    shuffle=True, 
                    batch_size=detail_cfg['batch_size'], 
                    collate_fn=collate_fn, 
                    drop_last=use_drop_last,
                    **common_cfg
                ),
                val_dataloader=DataLoader(
                    val_subset, 
                    shuffle=False, 
                    batch_size=detail_cfg['batch_size'], 
                    collate_fn=collate_fn, 
                    **common_cfg
                ),
                num_classes=num_sub_classes,
                target_key='detail_label',
                label_mapping=global_to_local,
                output_dir=sub_model_dir,
                class_weights=detail_weights,
                use_focal_loss=True,
                label_smoothing=detail_label_smoothing
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