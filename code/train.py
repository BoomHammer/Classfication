#!/usr/bin/env python3
"""
train.py: 训练主程序 (修复版 - 自动权重与参数)
"""

import sys
import json
import argparse
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from config_manager import ConfigManager
from point_timeseries_dataset import PointTimeSeriesDataset, collate_fn
from model_architecture import DualStreamSpatio_TemporalFusionNetwork
from trainer import Trainer

def calculate_class_weights(dataset, num_classes):
    """计算类别权重 (Inverse Frequency)"""
    print("⚖️ 正在计算类别权重...")
    # 从 Dataset 的 points_df 中直接获取标签列
    # 注意：label 字段名取决于 Dataset 初始化时设定的 target_col
    # 这里假设我们训练的是 dataset.label_col 指定的列
    all_labels = []
    # 稍微 trick 一下：Dataset 已经把 df 存在 self.points_df
    # 我们根据 split 筛选
    indices = dataset.indices
    # dataset.points_df 是完整的 dataframe
    # indices 是 numpy array
    subset_df = dataset.points_df.iloc[indices]
    
    # 确定当前训练的目标列 (major 或 detail)
    # 我们可以通过读取第一个样本的 'label' 来确认，或者假设是 detail
    # 但为了稳健，我们统计 dataset[i]['label']
    # 为了速度，直接用 DataFrame
    # 假设 Dataset 正确设置了当前任务的标签
    
    # 简易方案：遍历 dataset (稍微慢点但稳)
    # 或者直接用 DataFrame 的分布
    counts = Counter()
    # 假设 points_df 里的列是 encoder 处理过的
    # 这里我们只取前 1000 个样本做估计，或者全部
    labels = [dataset[i]['label'].item() for i in range(len(dataset))]
    counts.update(labels)
    
    total = sum(counts.values())
    weights = torch.zeros(num_classes)
    for cls_idx in range(num_classes):
        count = counts[cls_idx]
        if count > 0:
            weights[cls_idx] = total / (len(counts) * count)
        else:
            weights[cls_idx] = 1.0 # 没出现的类给 1
            
    print(f"   类别分布: {dict(counts)}")
    print(f"   计算权重: {weights.numpy().round(3)}")
    return weights

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32) # 默认为 32
    parser.add_argument('--accum_steps', type=int, default=2) # 默认累积2步 -> 效能64
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # 1. 配置
    config_path = Path(__file__).parent / args.config
    config = ConfigManager(str(config_path))
    output_dir = config.get_experiment_output_dir()
    
    # 2. 自动检测参数
    param_file = output_dir / 'detected_parameters.json'
    if not param_file.exists():
        print("❌ 请先运行 preprocess_dataset.py")
        return
    with open(param_file, 'r') as f:
        params = json.load(f)
    
    # 3. 数据集
    print("📊 加载数据集...")
    train_ds = PointTimeSeriesDataset(config, None, split='train', split_ratio=[0.7, 0.15, 0.15])
    val_ds = PointTimeSeriesDataset(config, None, split='val', split_ratio=[0.7, 0.15, 0.15])
    test_ds = PointTimeSeriesDataset(config, None, split='test', split_ratio=[0.7, 0.15, 0.15])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    # 4. 计算权重
    class_weights = calculate_class_weights(train_ds, params['num_classes'])

    # 5. 模型
    print(f"🏗️ 构建模型 (Dynamic: {params['dynamic_channels']}, Static: {params['static_channels']})...")
    model = DualStreamSpatio_TemporalFusionNetwork(
        in_channels_dynamic=params['dynamic_channels'],
        in_channels_static=params['static_channels'],
        num_classes=params['num_classes'],
        hidden_dim=config.get('model.hidden_dim', 64),
        dropout=config.get('model.dropout', 0.2)
    )

    # 6. 训练
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        num_classes=params['num_classes'],
        class_weights=class_weights, # 传入权重
        output_dir=output_dir
    )
    
    trainer.train(
        num_epochs=args.epochs,
        learning_rate=args.lr,
        accumulation_steps=args.accum_steps, # 传入累积步数
        debug=args.debug
    )
    
    trainer.test()

if __name__ == '__main__':
    main()