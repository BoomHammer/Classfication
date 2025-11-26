#!/usr/bin/env python3
"""
train.py: 完整的训练流程

【第六阶段】训练循环与日志系统

该脚本执行：
1. 加载已准备好的数据集
2. 初始化模型
3. 执行训练（支持 Debug 模式快速过拟合测试）
4. 评估测试集性能
5. 生成训练报告

运行方式：
    # 正常训练
    python train.py
    
    # Debug 模式（快速验证模型学习能力）
    python train.py --debug
    
    # 断点续训
    python train.py --resume_from ./experiments/outputs/.../last_model.pth
    
    # 自定义参数
    python train.py --epochs 100 --lr 1e-3 --batch_size 32

输出文件：
    experiments/outputs/{timestamp}_{experiment_id}/
    ├── best_model.pth              # 最佳模型权重
    ├── last_model.pth              # 最后一个 checkpoint
    ├── training_log.txt            # 训练日志
    ├── training_metrics.json        # 训练指标
    ├── confusion_matrix.npy         # 测试集混淆矩阵
    ├── training_report.json         # 最终训练报告
    └── model_summary.txt            # 模型信息汇总
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader

# 导入本地模块
sys.path.insert(0, str(Path(__file__).parent))

from config_manager import ConfigManager
from label_encoder import LabelEncoder
from raster_crawler import RasterCrawler
from point_timeseries_dataset import PointTimeSeriesDataset, collate_fn
from model_architecture import DualStreamSpatio_TemporalFusionNetwork
from trainer import Trainer


# ============================================================================
# 工具函数
# ============================================================================

def load_or_prepare_data(config: ConfigManager, force_recompute: bool = False):
    """
    加载或准备数据集
    
    Args:
        config: ConfigManager 对象
        force_recompute: 是否强制重新计算
    
    Returns:
        (train_loader, val_loader, test_loader, num_classes)
    """
    print("\n" + "=" * 80)
    print("📊 加载数据集...")
    print("=" * 80 + "\n")
    
    # 从之前的阶段检查必要文件
    output_dir = config.get_experiment_output_dir()
    
    required_files = [
        'normalization_stats.json',
        'dataset_info.json',
        'detected_parameters.json',
    ]
    
    for filename in required_files:
        filepath = Path(output_dir) / filename
        if not filepath.exists():
            print(f"❌ 必要文件不存在: {filename}")
            print(f"   请先运行 python main.py 完成数据准备")
            return None
    
    # 加载自动检测的参数
    with open(Path(output_dir) / 'detected_parameters.json', 'r') as f:
        params = json.load(f)
    
    num_classes = params['num_classes']
    dynamic_channels = params['dynamic_channels']
    static_channels = params['static_channels']
    
    print(f"✅ 自动检测参数:")
    print(f"   - 类别数: {num_classes}")
    print(f"   - 动态通道数: {dynamic_channels}")
    print(f"   - 静态通道数: {static_channels}\n")
    
    # 初始化标签编码器和爬虫
    encoder = LabelEncoder(config=config)
    
    dynamic_crawler = RasterCrawler(
        config=config,
        raster_dir=config.get_resolved_path('dynamic_images_dir'),
        filename_pattern=config.get('data_specs.raster_crawler.filename_pattern'),
        file_extensions=tuple(config.get('data_specs.raster_crawler.file_extensions', ['.tif', '.tiff', '.jp2'])),
    )
    
    static_crawler = RasterCrawler(
        config=config,
        raster_dir=config.get_resolved_path('static_images_dir'),
        filename_pattern=config.get('data_specs.raster_crawler.filename_pattern'),
        file_extensions=tuple(config.get('data_specs.raster_crawler.file_extensions', ['.tif', '.tiff', '.jp2'])),
    )
    
    # 初始化数据集
    print("初始化数据集...")
    
    stats_file = Path(output_dir) / 'normalization_stats.json'
    split_ratio = tuple(config.get('train.split_ratio', (0.7, 0.15, 0.15)))
    
    train_dataset = PointTimeSeriesDataset(
        config=config,
        encoder=encoder,
        dynamic_crawler=dynamic_crawler,
        static_crawler=static_crawler,
        stats_file=str(stats_file) if stats_file.exists() else None,
        split='train',
        split_ratio=split_ratio,
        seed=config.get('train.seed', 42),
        cache_metadata=True,
        verbose=False,
    )
    
    val_dataset = PointTimeSeriesDataset(
        config=config,
        encoder=encoder,
        dynamic_crawler=dynamic_crawler,
        static_crawler=static_crawler,
        stats_file=str(stats_file) if stats_file.exists() else None,
        split='val',
        split_ratio=split_ratio,
        seed=config.get('train.seed', 42),
        cache_metadata=True,
        verbose=False,
    )
    
    test_dataset = PointTimeSeriesDataset(
        config=config,
        encoder=encoder,
        dynamic_crawler=dynamic_crawler,
        static_crawler=static_crawler,
        stats_file=str(stats_file) if stats_file.exists() else None,
        split='test',
        split_ratio=split_ratio,
        seed=config.get('train.seed', 42),
        cache_metadata=True,
        verbose=False,
    )
    
    print(f"✅ 数据集加载完成:")
    print(f"   - 训练集: {len(train_dataset)} 样本")
    print(f"   - 验证集: {len(val_dataset)} 样本")
    print(f"   - 测试集: {len(test_dataset)} 样本\n")
    
    # 创建数据加载器
    batch_size = config.get('train.batch_size', 32)
    num_workers = config.get('train.num_workers', 0)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    return train_loader, val_loader, test_loader, num_classes, dynamic_channels, static_channels


def create_model(
    num_classes: int,
    dynamic_channels: int,
    static_channels: int,
    config: ConfigManager,
):
    """
    创建模型
    
    Args:
        num_classes: 类别数
        dynamic_channels: 动态通道数
        static_channels: 静态通道数
        config: 配置对象
    
    Returns:
        模型实例
    """
    print("\n" + "=" * 80)
    print("🏗️  构建模型...")
    print("=" * 80 + "\n")
    
    model = DualStreamSpatio_TemporalFusionNetwork(
        in_channels_dynamic=dynamic_channels,
        in_channels_static=static_channels,
        num_classes=num_classes,
        patch_size=config.get('data_specs.spatial.patch_size', 64),
        temporal_steps=12,
        hidden_dim=config.get('model.hidden_dim', 64),
        fusion_dim=config.get('model.fusion_dim', 128),
        dropout=config.get('model.dropout', 0.2),
    )
    
    summary = model.get_model_summary()
    print(f"✅ 模型构建成功")
    print(f"   - 模型名称: {summary['model_name']}")
    print(f"   - 总参数数: {summary['total_parameters']:,}")
    print(f"   - 可训练参数: {summary['trainable_parameters']:,}\n")
    
    return model


def main():
    """主程序入口"""
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练遥感影像分类模型')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心数')
    parser.add_argument('--debug', action='store_true', help='Debug 模式')
    parser.add_argument('--resume_from', type=str, default=None, help='从指定 checkpoint 恢复')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    
    args = parser.parse_args()
    
    # =========================================================================
    # 第一步：加载配置
    # =========================================================================
    print("\n" + "=" * 80)
    print("📋 加载配置...")
    print("=" * 80 + "\n")
    
    config_path = Path(__file__).parent / args.config
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return 1
    
    config = ConfigManager(str(config_path))
    output_dir = config.get_experiment_output_dir()
    
    print(f"✅ 配置加载成功")
    print(f"   - 输出目录: {output_dir}\n")
    
    # =========================================================================
    # 第二步：加载数据集
    # =========================================================================
    data_result = load_or_prepare_data(config)
    if data_result is None:
        return 1
    
    train_loader, val_loader, test_loader, num_classes, dynamic_channels, static_channels = data_result
    
    # =========================================================================
    # 第三步：创建模型
    # =========================================================================
    model = create_model(num_classes, dynamic_channels, static_channels, config)
    
    # =========================================================================
    # 第四步：初始化训练器
    # =========================================================================
    print("\n" + "=" * 80)
    print("🎓 初始化训练器...")
    print("=" * 80 + "\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        num_classes=num_classes,
        device=device,
        output_dir=output_dir,
        verbose=True,
    )
    
    print(f"✅ 训练器初始化完成\n")
    
    # =========================================================================
    # 第五步：执行训练
    # =========================================================================
    try:
        resume_from = None
        if args.resume_from:
            resume_from = Path(args.resume_from)
        
        history = trainer.train(
            num_epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            debug=args.debug,
            resume_from=resume_from,
        )
        
    except KeyboardInterrupt:
        print("\n\n⏸️  训练被中断")
        return 0
    except Exception as e:
        print(f"\n❌ 训练过程出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # =========================================================================
    # 第六步：测试集评估
    # =========================================================================
    print("\n")
    test_metrics = trainer.test()
    
    # =========================================================================
    # 第七步：生成最终报告
    # =========================================================================
    print("\n" + "=" * 80)
    print("📊 生成最终报告...")
    print("=" * 80 + "\n")
    
    final_report = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'config_file': str(config_path),
            'output_directory': str(output_dir),
        },
        'model_info': {
            'num_classes': num_classes,
            'dynamic_channels': dynamic_channels,
            'static_channels': static_channels,
        },
        'training_config': {
            'num_epochs': args.epochs,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'patience': args.patience,
            'debug_mode': args.debug,
        },
        'dataset_info': {
            'train_size': len(train_loader.dataset),
            'val_size': len(val_loader.dataset),
            'test_size': len(test_loader.dataset),
        },
        'training_history': history,
        'test_metrics': test_metrics,
        'best_model': {
            'epoch': trainer.best_epoch,
            'val_f1_score': float(trainer.best_val_f1),
        }
    }
    
    # 保存报告
    report_file = Path(output_dir) / 'training_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 最终报告已保存: {report_file}")
    
    # 打印摘要
    print("\n" + "=" * 80)
    print("📋 训练摘要")
    print("=" * 80 + "\n")
    
    print(f"数据集:")
    print(f"  - 训练集: {len(train_loader.dataset)} 样本")
    print(f"  - 验证集: {len(val_loader.dataset)} 样本")
    print(f"  - 测试集: {len(test_loader.dataset)} 样本")
    print(f"\n最佳模型:")
    print(f"  - Epoch: {trainer.best_epoch}")
    print(f"  - 验证 F1-Score: {trainer.best_val_f1:.4f}")
    print(f"\n测试结果:")
    print(f"  - Accuracy: {test_metrics.get('accuracy', 0):.4f}")
    print(f"  - F1 (Macro): {test_metrics.get('f1_macro', 0):.4f}")
    print(f"  - F1 (Weighted): {test_metrics.get('f1_weighted', 0):.4f}")
    print(f"  - IoU: {test_metrics.get('iou', 0):.4f}")
    print(f"\n输出目录: {output_dir}")
    print(f"\n📁 重要文件:")
    print(f"  - {output_dir}/best_model.pth              (最佳模型)")
    print(f"  - {output_dir}/training_log.txt            (训练日志)")
    print(f"  - {output_dir}/training_metrics.json        (训练指标)")
    print(f"  - {output_dir}/training_report.json         (最终报告)")
    print(f"  - {output_dir}/confusion_matrix.npy         (混淆矩阵)")
    
    print("\n" + "=" * 80)
    print("✅ 训练完成!")
    print("=" * 80 + "\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
