#!/usr/bin/env python3
"""
main.py: 数据处理流水线主入口

该脚本实现了完整的六阶段流程：
1. 第一阶段：配置管理与基础设施搭建 (Configuration & Infrastructure)
2. 第二阶段：数据清洗与时空索引构建 (Data Ingestion & Indexing)
3. 第三阶段：在线统计与数据归一化 (Statistical Analysis & Normalization)
4. 第四阶段：自定义时空数据集构建 (Custom Dataset Implementation)
5. 第五阶段：模型架构设计 (Model Architecture)
6. 第六阶段：训练循环与日志系统 (Training Loop & Logging)

运行方式：
    python main.py

输出文件位置：
    experiments/outputs/{timestamp}_{experiment_id}/
        ├── config_used.yaml                 # 使用的配置文件副本
        ├── data_inventory.csv               # 数据清单
        ├── verification_report.json         # 详细验证报告
        ├── data_summary.txt                 # 文本摘要报告
        ├── detailed_labels_map.json         # 详细类别映射
        ├── major_labels_map.json            # 大类映射
        ├── hierarchical_labels_map.json     # 层级映射
        ├── labels_geodata.geojson           # GeoJSON格式的标签
        ├── rasters_metadata.json            # 栅格元数据
        ├── rasters_summary.json             # 栅格汇总
        └── normalization_stats.json         # 归一化参数

【流程设计】
=============================================================================

第一阶段：配置管理与基础设施搭建
  ↓
  加载配置文件 → 路径验证 → 创建实验输出目录 → 配置冻结
  输出：config_used.yaml

第二阶段：数据清洗与时空索引构建
  ↓
  初始化标签编码器 → 读取CSV标签 → 生成类别映射 → 投影转换
  ↓
  初始化栅格爬虫 → 扫描影像文件 → 解析时间元数据 → 构建R-树索引
  ↓
  生成数据清单 → 验证CRS一致性 → 生成验证报告
  输出：
    - detailed_labels_map.json
    - major_labels_map.json
    - data_inventory.csv
    - verification_report.json
    - rasters_metadata.json

第三阶段：在线统计与数据归一化
  ↓
  采样栅格文件 → Welford增量算法计算统计量 → 保存归一化参数
  输出：normalization_stats.json

第四阶段：自定义时空数据集构建
  ↓
  继承Dataset类 → 时间轴对齐 → 窗口读取优化 → 缺失值处理 → 自动归一化
  ↓
  训练/验证/测试集划分 → 数据验证 → 性能基准测试
  输出：dataset_info.json

第五阶段：模型架构设计
  ↓
  构建双流融合网络 → 显示模型摘要 → 前向传播测试 → 时间注意力可视化
  输出：model_architecture.json

第六阶段：训练循环与日志系统
  ↓
  初始化训练器 → 执行训练循环 → 动态监控 → 检查点保存 → 生成报告
  ↓
  多指标评估（Accuracy, F1, IoU, Precision, Recall, 混淆矩阵）
  ↓
  可视化结果 → 自动分析报告
  输出：
    - training_report.json
    - best_model.pth
    - metrics_curves.png
    - confusion_matrix.png

=============================================================================

【关键特性】

✓ 模块化架构：各个组件职责清晰、松耦合
✓ 错误处理：快速失败机制，及时反馈问题
✓ 流式处理：Welford算法支持大规模数据
✓ 完整日志：每个阶段都有详细的进度和状态输出
✓ 数据可追溯：完整的元数据和验证报告
✓ 自动化输出：所有中间和最终结果自动保存

【系统要求】

Python >= 3.8

必要包：
  - yaml
  - pandas
  - geopandas
  - numpy
  - rasterio
  - rtree
  - shapely
  - tqdm

【使用示例】

$ python main.py
[INFO] ================================================== =============================
[INFO] 🚀 地理空间数据处理流水线启动
[INFO] ================================================== =============================
[INFO]
[INFO] 📋 [阶段1] 加载配置文件...
[INFO]   ✅ 配置加载成功
[INFO]
[INFO] 📋 [阶段2] 数据清洗与时空索引构建...
[INFO]   ✅ 标签处理完成 (2500个样本)
[INFO]   ✅ 影像索引完成 (15000个文件)
[INFO]
[INFO] 📋 [阶段3] 在线统计与数据归一化...
[INFO]   ✅ 统计量计算完成
[INFO]
[INFO] ================================================== =============================
[INFO] ✅ 全流程完成！
[INFO] ================================================== =============================
"""

import sys
import logging
import json
from pathlib import Path

# 确保能导入本地模块
sys.path.insert(0, str(Path(__file__).parent))

from config_manager import ConfigManager
from label_encoder import LabelEncoder
from raster_crawler import RasterCrawler
from data_preprocessor import DataPreprocessor
from stats_calculator import StatsCalculator
from point_timeseries_dataset import PointTimeSeriesDataset
from trainer import Trainer


def setup_logging():
    """配置全局日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s',
        stream=sys.stdout
    )


def print_header(title):
    """打印标题"""
    print("\n" + "=" * 80)
    print(f"🚀 {title}")
    print("=" * 80)


def print_section(num, title):
    """打印阶段标题"""
    print(f"\n📋 [阶段{num}] {title}")


def print_success(message):
    """打印成功消息"""
    print(f"✅ {message}")


def print_warning(message):
    """打印警告消息"""
    print(f"⚠️  {message}")


def print_error(message):
    """打印错误消息"""
    print(f"❌ {message}")


def phase_1_configuration():
    """
    第一阶段：配置管理与基础设施搭建
    
    功能：
    1. 加载配置文件
    2. 验证所有关键路径
    3. 创建实验输出目录
    4. 冻结配置（只读保护）
    
    Returns:
        ConfigManager: 配置对象
    """
    print_section(1, "配置管理与基础设施搭建")
    
    try:
        # 定位配置文件
        config_path = Path(__file__).parent / 'config.yaml'
        
        if not config_path.exists():
            print_error(f"配置文件不存在: {config_path}")
            return None
        
        print(f"📂 配置文件: {config_path}")
        
        # 初始化配置管理器（包含路径验证和实验输出目录创建）
        config = ConfigManager(str(config_path))
        
        print_success("配置加载成功")
        print(f"  项目名: {config.get('project_name')}")
        print(f"  实验ID: {config.get('experiment_id')}")
        print(f"  输出目录: {config.get_experiment_output_dir()}")
        
        return config
        
    except Exception as e:
        print_error(f"配置阶段失败: {e}")
        return None


def phase_2_data_ingestion(config):
    """
    第二阶段：数据清洗与时空索引构建
    
    功能：
    1. 初始化标签编码器（CSV → 类别映射）
    2. 初始化栅格爬虫（影像文件扫描与时间解析）
    3. 执行数据验证和清单生成
    
    Args:
        config: ConfigManager对象
    
    Returns:
        Tuple[LabelEncoder, RasterCrawler, RasterCrawler]: 标签编码器和两个栅格爬虫
    """
    print_section(2, "数据清洗与时空索引构建 (Data Ingestion & Indexing)")
    
    try:
        # 步骤 1: 初始化标签编码器
        print("\n  📝 初始化标签编码器...")
        try:
            encoder = LabelEncoder(config=config)
            stats = encoder.get_statistics()
            hierarchical_map = encoder.get_hierarchical_map()
            
            # 自动检测类别数
            num_detailed_classes = len(stats['detailed_class_distribution'])
            num_major_classes = len(stats['major_class_distribution'])
            
            print_success(f"标签编码完成 ({stats['total_samples']} 个样本)")
            print(f"    详细类别: {num_detailed_classes} 个")
            print(f"      {stats['detailed_class_distribution']}")
            print(f"    大类数: {num_major_classes} 个")
            print(f"      {stats['major_class_distribution']}")
            
            # 分析层级结构
            print(f"\n  📊 层级结构分析:")
            for major_id, major_info in sorted(hierarchical_map.items()):
                major_name = major_info.get('name', f'Major_{major_id}')
                num_detail = len(major_info.get('detail_classes', {}))
                skip_msg = " ⚡(仅1个小类)" if num_detail == 1 else ""
                print(f"    大类 {major_id}: {major_name} - {num_detail} 个小类{skip_msg}")
        except Exception as e:
            print_error(f"标签编码失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, None, None, None
        
        # 步骤 2: 初始化栅格爬虫 - 动态影像
        print("\n  📚 初始化栅格爬虫 (动态影像)...")
        dynamic_crawler = None
        dynamic_channels = None
        try:
            filename_pattern = config.get('data_specs.raster_crawler.filename_pattern')
            dynamic_crawler = RasterCrawler(
                config=config,
                raster_dir=config.get_resolved_path('dynamic_images_dir'),
                filename_pattern=filename_pattern,
                file_extensions=tuple(config.get(
                    'data_specs.raster_crawler.file_extensions',
                    ['.tif', '.tiff', '.jp2']
                )),
            )
            print_success(f"动态影像爬虫初始化 ({dynamic_crawler.get_raster_count()} 个文件)")
            
            # 自动检测波段数
            channel_info = dynamic_crawler.detect_num_channels(sample_size=5)
            dynamic_channels = channel_info['most_common']
            print(f"    自动检测: {dynamic_channels} 个波段")
            if 'warning' in channel_info:
                print_warning(f"    {channel_info['warning']}")
            
            # 自动检测坐标系
            print(f"    检测坐标系...")
            crs_info = dynamic_crawler.detect_crs()
            if crs_info['most_common_crs']:
                print(f"    坐标系: {crs_info['most_common_crs']}")
                if not crs_info['is_consistent']:
                    print_warning(f"    {crs_info['warning']}")
                if crs_info.get('recommendation'):
                    print(f"    💡 {crs_info['recommendation']}")

        except Exception as e:
            print_warning(f"动态影像爬虫初始化失败: {e}")
        
        # 步骤 3: 初始化栅格爬虫 - 静态影像
        print("\n  📚 初始化栅格爬虫 (静态影像)...")
        static_crawler = None
        static_channels = None
        try:
            static_crawler = RasterCrawler(
                config=config,
                raster_dir=config.get_resolved_path('static_images_dir'),
                filename_pattern=config.get('data_specs.raster_crawler.filename_pattern'),
                file_extensions=tuple(config.get(
                    'data_specs.raster_crawler.file_extensions',
                    ['.tif', '.tiff', '.jp2']
                )),
            )
            print_success(f"静态影像爬虫初始化 ({static_crawler.get_raster_count()} 个文件)")
            
            # 自动检测波段数
            channel_info = static_crawler.detect_num_channels(sample_size=5)
            static_channels = channel_info['most_common']
            print(f"    自动检测: {static_channels} 个波段")
            if 'warning' in channel_info:
                print_warning(f"    {channel_info['warning']}")
            
            # 自动检测坐标系
            print(f"    检测坐标系...")
            crs_info = static_crawler.detect_crs()
            if crs_info['most_common_crs']:
                print(f"    坐标系: {crs_info['most_common_crs']}")
                if not crs_info['is_consistent']:
                    print_warning(f"    {crs_info['warning']}")
                if crs_info.get('recommendation'):
                    print(f"    💡 {crs_info['recommendation']}")

            
            # 自动检测波段数
            channel_info = static_crawler.detect_num_channels(sample_size=5)
            static_channels = channel_info['most_common']
            print(f"    自动检测: {static_channels} 个波段")
            if 'warning' in channel_info:
                print_warning(f"    {channel_info['warning']}")
        except Exception as e:
            print_warning(f"静态影像爬虫初始化失败: {e}")
        
        # 步骤 4: 数据预处理和验证
        print("\n  🔍 执行数据验证...")
        try:
            preprocessor = DataPreprocessor(
                config=config,
                encoder=encoder,
                dynamic_crawler=dynamic_crawler,
                static_crawler=static_crawler,
            )
            preprocessor.run()
            print_success("数据验证完成")
        except Exception as e:
            print_error(f"数据验证失败: {e}")
            return encoder, dynamic_crawler, static_crawler, hierarchical_map, dynamic_channels, static_channels
        
        return encoder, dynamic_crawler, static_crawler, hierarchical_map, dynamic_channels, static_channels
        
    except Exception as e:
        print_error(f"第二阶段失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None


def phase_3_statistical_analysis(config, dynamic_crawler, static_crawler):
    """
    第三阶段：在线统计与数据归一化
    
    功能：
    1. 采样栅格文件
    2. 使用 Welford 增量算法计算统计量
    3. 保存归一化参数
    
    Args:
        config: ConfigManager对象
        dynamic_crawler: 动态影像爬虫
        static_crawler: 静态影像爬虫
    
    Returns:
        bool: 是否成功
    """
    print_section(3, "在线统计与数据归一化 (Statistical Analysis & Normalization)")
    
    try:
        if not dynamic_crawler and not static_crawler:
            print_warning("未找到任何影像爬虫，跳过统计阶段")
            return True
        
        # 初始化统计计算器
        print("\n  📊 初始化统计计算器...")
        calculator = StatsCalculator(
            config=config,
            dynamic_channel_names=['Band_0'],  # 实际通道数会自动检测
            static_channel_names=['Band_0'],
        )
        print_success("统计计算器已初始化")
        
        # 获取栅格列表
        dynamic_rasters = dynamic_crawler.get_all_rasters() if dynamic_crawler else None
        static_rasters = static_crawler.get_all_rasters() if static_crawler else None
        
        if not dynamic_rasters and not static_rasters:
            print_warning("未找到任何影像文件")
            return True
        
        # 计算统计量
        print("\n  🧮 计算全局统计量 (Welford增量算法)...")
        try:
            calculator.compute_global_stats(
                dynamic_rasters=dynamic_rasters,
                static_rasters=static_rasters,
                sampling_rate=0.6,  # 采样 60%
            )
            print_success("统计量计算完成")
            
            # 获取并显示统计参数
            params = calculator.get_normalization_params()
            if 'dynamic' in params:
                print(f"\n  📊 动态影像统计:")
                print(f"     Mean: {params['dynamic']['mean']}")
                print(f"     Std:  {params['dynamic']['std']}")
            if 'static' in params:
                print(f"\n  📊 静态影像统计:")
                print(f"     Mean: {params['static']['mean']}")
                print(f"     Std:  {params['static']['std']}")
        except Exception as e:
            print_error(f"统计量计算失败: {e}")
            return False
        
        # 保存统计量
        print("\n  💾 保存归一化参数...")
        try:
            calculator.save_stats('normalization_stats.json')
            print_success("归一化参数已保存")
        except Exception as e:
            print_error(f"保存统计量失败: {e}")
            return False
        
        return True
        
    except Exception as e:
        print_error(f"第三阶段失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def phase_4_dataset_construction(config, encoder, dynamic_crawler, static_crawler):
    """
    第四阶段：自定义时空数据集构建 (Custom Dataset Implementation)
    
    功能：
    1. 继承 torch.utils.data.Dataset 实现点-时序对齐
    2. 利用 rasterio 窗口读取实现高效数据加载
    3. 构建训练/验证/测试集划分
    4. 验证数据集完整性和性能
    
    关键特性：
    ✓ 即时窗口读取（On-the-fly Windowed Reading）
    ✓ 标准时间轴对齐（按月份聚合）
    ✓ 灵活的缺失值处理策略
    ✓ 自动归一化
    ✓ 高效的R树空间索引
    
    Args:
        config: ConfigManager对象
        encoder: LabelEncoder对象
        dynamic_crawler: 动态影像爬虫
        static_crawler: 静态影像爬虫
    
    Returns:
        Tuple[PointTimeSeriesDataset, PointTimeSeriesDataset, PointTimeSeriesDataset]: 
            训练、验证、测试数据集
    """
    print_section(4, "自定义时空数据集构建 (Custom Dataset Implementation)")
    
    try:
        # 步骤 1: 确认归一化参数文件
        output_dir = config.get_experiment_output_dir()
        stats_file = Path(output_dir) / 'normalization_stats.json'
        
        if not stats_file.exists():
            print_warning(f"归一化参数文件不存在: {stats_file}")
            print_warning("数据将以原始值返回，不进行归一化")
        else:
            print(f"✓ 归一化参数文件: {stats_file}")
        
        # 步骤 2: 初始化训练集
        print("\n  🎓 初始化训练集...")
        try:
            train_dataset = PointTimeSeriesDataset(
                config=config,
                encoder=encoder,
                dynamic_crawler=dynamic_crawler,
                static_crawler=static_crawler,
                stats_file=str(stats_file) if stats_file.exists() else None,
                split='train',
                split_ratio=config.get('train.split_ratio', (0.7, 0.15, 0.15)),
                seed=config.get('train.seed', 42),
                cache_metadata=True,
                missing_value_strategy=config.get('data_specs.temporal.missing_value_strategy', 'zero_padding'),
                normalization_method=config.get('data_specs.temporal.normalization_method', 'zscore'),
                verbose=False,
            )
            train_stats = train_dataset.get_statistics()
            print_success(f"训练集初始化完成 ({len(train_dataset)} 个样本)")
            print(f"  类别分布: {train_stats['label_distribution']}")
        except Exception as e:
            print_error(f"训练集初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
        
        # 步骤 3: 初始化验证集
        print("\n  📊 初始化验证集...")
        try:
            val_dataset = PointTimeSeriesDataset(
                config=config,
                encoder=encoder,
                dynamic_crawler=dynamic_crawler,
                static_crawler=static_crawler,
                stats_file=str(stats_file) if stats_file.exists() else None,
                split='val',
                split_ratio=config.get('train.split_ratio', (0.7, 0.15, 0.15)),
                seed=config.get('train.seed', 42),
                cache_metadata=True,
                verbose=False,
            )
            val_stats = val_dataset.get_statistics()
            print_success(f"验证集初始化完成 ({len(val_dataset)} 个样本)")
        except Exception as e:
            print_error(f"验证集初始化失败: {e}")
            return None, None, None
        
        # 步骤 4: 初始化测试集
        print("\n  🧪 初始化测试集...")
        try:
            test_dataset = PointTimeSeriesDataset(
                config=config,
                encoder=encoder,
                dynamic_crawler=dynamic_crawler,
                static_crawler=static_crawler,
                stats_file=str(stats_file) if stats_file.exists() else None,
                split='test',
                split_ratio=config.get('train.split_ratio', (0.7, 0.15, 0.15)),
                seed=config.get('train.seed', 42),
                cache_metadata=True,
                verbose=False,
            )
            test_stats = test_dataset.get_statistics()
            print_success(f"测试集初始化完成 ({len(test_dataset)} 个样本)")
        except Exception as e:
            print_error(f"测试集初始化失败: {e}")
            return None, None, None
        
        # 步骤 5: 性能基准测试（快速检查）
        print("\n  ⚡ 性能基准测试（抽样10个样本）...")
        try:
            import time
            times = []
            for i in range(min(10, len(train_dataset))):
                start = time.time()
                _ = train_dataset[i]
                times.append(time.time() - start)
            
            avg_time = sum(times) / len(times)
            max_time = max(times)
            
            print(f"  平均耗时: {avg_time:.4f}s")
            print(f"  最大耗时: {max_time:.4f}s")
            
            if avg_time < 0.1:
                print_success(f"性能满足要求 (< 0.1s)")
            elif avg_time < 0.5:
                print_warning(f"性能略慢，但可以接受 (< 0.5s)")
            else:
                print_warning(f"性能可能成为瓶颈 (> 0.5s)，考虑优化TIFF存储格式")
        except Exception as e:
            print_warning(f"性能测试异常: {e}")
        
        # 步骤 6: 保存数据集元数据
        print("\n  💾 保存数据集元数据...")
        try:
            dataset_info = {
                'train': {
                    'size': len(train_dataset),
                    'statistics': train_stats,
                },
                'val': {
                    'size': len(val_dataset),
                    'statistics': val_stats,
                },
                'test': {
                    'size': len(test_dataset),
                    'statistics': test_stats,
                },
                'configuration': {
                    'patch_size': config.get('data_specs.spatial.patch_size', 64),
                    'missing_value_strategy': config.get('data_specs.temporal.missing_value_strategy', 'zero_padding'),
                    'normalization_method': config.get('data_specs.temporal.normalization_method', 'zscore'),
                }
            }
            
            dataset_info_file = Path(output_dir) / 'dataset_info.json'
            with open(dataset_info_file, 'w', encoding='utf-8') as f:
                json.dump(dataset_info, f, ensure_ascii=False, indent=2)
            
            print_success(f"数据集元数据已保存: {dataset_info_file}")
        except Exception as e:
            print_warning(f"保存数据集元数据失败: {e}")
        
        print_success("第四阶段完成")
        return train_dataset, val_dataset, test_dataset
        
    except Exception as e:
        print_error(f"第四阶段失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def phase_5_model_architecture(config, hierarchical_map, dynamic_channels, static_channels):
    """
    第五阶段：模型架构设计 (Model Architecture)
    
    功能：
    1. 构建分层分类双流网络
    2. 显示模型摘要与参数统计
    3. 验证模型可以正常前向传播
    
    Args:
        config: ConfigManager对象
        hierarchical_map: 分层映射字典
        dynamic_channels: 动态影像通道数
        static_channels: 静态影像通道数
    
    Returns:
        模型实例，或None如果构建失败
    """
    print_section(5, "模型架构设计 (Model Architecture)")
    
    try:
        # 导入模型
        from model_architecture import HierarchicalDualStreamNetwork
        
        # 步骤 1: 初始化模型
        print("\n  🏗️  构建分层分类双流网络...")
        try:
            model = HierarchicalDualStreamNetwork(
                in_channels_dynamic=dynamic_channels if dynamic_channels else 4,
                in_channels_static=static_channels if static_channels else 1,
                hierarchical_map=hierarchical_map,
                patch_size=config.get('data_specs.spatial.patch_size', 64),
                temporal_steps=12,  # 固定为12个月
                hidden_dim=config.get('model.hidden_dim', 64),
                dropout=config.get('model.dropout', 0.2),
            )
            print_success("模型构建成功")
        except Exception as e:
            print_error(f"模型构建失败: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        # 步骤 2: 显示模型摘要
        print("\n  📊 模型摘要:")
        try:
            summary = model.get_model_summary()
            print(f"    模型名称: {summary['model_name']}")
            print(f"    总参数数: {summary['total_parameters']:,}")
            print(f"    可训练参数: {summary['trainable_parameters']:,}")
            print(f"\n    模型配置:")
            for key, value in summary['configuration'].items():
                print(f"      - {key}: {value}")
        except Exception as e:
            print_warning(f"获取模型摘要失败: {e}")
        
        # 步骤 3: 模型前向传播测试
        print("\n  ⚡ 执行前向传播测试...")
        try:
            import torch
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            # 创建虚拟输入
            batch_size = 4
            dynamic_dummy = torch.randn(
                batch_size,
                12,  # 时间步
                dynamic_channels if dynamic_channels else 4,
                64,  # patch_size
                64,
                device=device
            )
            static_dummy = torch.randn(
                batch_size,
                static_channels if static_channels else 1,
                64,
                64,
                device=device
            )
            
            # 前向传播
            with torch.no_grad():
                output = model(dynamic_dummy, static_dummy, return_aux=True)
            
            major_logits = output['major_logits']
            detail_logits = output['detail_logits']
            major_preds = output['major_preds']
            detail_preds = output['detail_preds']
            
            print_success(f"前向传播成功")
            print(f"    大类输出形状: {tuple(major_logits.shape)}")
            print(f"    小类输出形状: {tuple(detail_logits.shape)}")
            print(f"    大类预测: {major_preds}")
            print(f"    小类预测: {detail_preds}")
            
            # 显示注意力权重
            if 'auxiliary' in output:
                attn_weights = output['auxiliary']['dynamic_attention_weights']
                print(f"\n  📈 时间注意力权重 (平均值):")
                months = ['1月', '2月', '3月', '4月', '5月', '6月',
                         '7月', '8月', '9月', '10月', '11月', '12月']
                avg_weights = attn_weights.mean(dim=0).cpu().numpy()
                for i, (month, weight) in enumerate(zip(months, avg_weights)):
                    bar = '█' * int(weight * 50)
                    print(f"    {month}: {weight:.4f} {bar}")
        
        except Exception as e:
            print_error(f"前向传播测试失败: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        # 步骤 4: 保存模型架构信息
        print("\n  💾 保存模型架构信息...")
        try:
            output_dir = config.get_experiment_output_dir()
            
            model_info = {
                'architecture': 'HierarchicalDualStreamNetwork',
                'summary': summary,
                'configuration': {
                    'in_channels_dynamic': dynamic_channels if dynamic_channels else 4,
                    'in_channels_static': static_channels if static_channels else 1,
                    'hierarchical_map': hierarchical_map,
                    'patch_size': 64,
                    'temporal_steps': 12,
                    'hidden_dim': config.get('model.hidden_dim', 64),
                    'dropout': config.get('model.dropout', 0.2),
                },
                'device': str(device),
                'attention_weights': {
                    'months': ['January', 'February', 'March', 'April', 'May', 'June',
                              'July', 'August', 'September', 'October', 'November', 'December'],
                    'average_weights': avg_weights.tolist() if 'avg_weights' in locals() else None,
                }
            }
            
            model_info_file = Path(output_dir) / 'model_architecture.json'
            with open(model_info_file, 'w', encoding='utf-8') as f:
                json.dump(model_info, f, ensure_ascii=False, indent=2)
            
            print_success(f"模型架构信息已保存: {model_info_file}")
        
        except Exception as e:
            print_warning(f"保存模型架构信息失败: {e}")
        
        print_success("第五阶段完成")
        return model
        
    except ImportError as e:
        print_error(f"导入模型模块失败: {e}")
        print_warning("请确保已安装 torch 和其他依赖")
        return None
    except Exception as e:
        print_error(f"第五阶段失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def phase_6_training_loop(config, model, train_dataset, val_dataset, test_dataset, output_dir, hierarchical_map):
    """
    第六阶段：训练循环与日志系统 (Training Loop & Logging)
    
    功能：
    1. 初始化 Trainer 类（支持分层分类）
    2. 执行完整训练循环
    3. 进行验证和测试
    4. 生成训练报告
    
    Args:
        config: ConfigManager对象
        model: 分层分类神经网络模型
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        test_dataset: 测试数据集
        output_dir: 输出目录
        hierarchical_map: 分层映射字典
    
    Returns:
        bool: 训练是否成功
    """
    print_section(6, "训练循环与日志系统 (Training Loop & Logging)")
    
    try:
        import torch
        from torch.utils.data import DataLoader
        
        # 步骤 1: 检查训练配置
        print("\n  ⚙️  检查训练配置...")
        try:
            epochs = config.get('train.epochs', 50)
            batch_size = config.get('train.batch_size', 32)
            lr = config.get('train.lr', 1e-3)
            weight_decay = config.get('train.weight_decay', 1e-4)
            patience = config.get('train.patience', 10)
            
            print(f"    Epochs: {epochs}")
            print(f"    Batch Size: {batch_size}")
            print(f"    Learning Rate: {lr}")
            print(f"    Weight Decay: {weight_decay}")
            print(f"    Patience (Early Stopping): {patience}")
            print_success("训练配置检查完成")
        except Exception as e:
            print_error(f"配置检查失败: {e}")
            return False
        
        # 步骤 2: 初始化数据加载器
        print("\n  📦 初始化数据加载器...")
        try:
            # 注意：将 num_workers 设置为 0 避免多进程序列化问题
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,  # 强制单进程模式
                pin_memory=True if torch.cuda.is_available() else False,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,  # 强制单进程模式
                pin_memory=True if torch.cuda.is_available() else False,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,  # 强制单进程模式
                pin_memory=True if torch.cuda.is_available() else False,
            )
            print_success(f"数据加载器初始化完成")
            print(f"  训练批次: {len(train_loader)}")
            print(f"  验证批次: {len(val_loader)}")
            print(f"  测试批次: {len(test_loader)}")
        except Exception as e:
            print_error(f"数据加载器初始化失败: {e}")
            return False
        
        # 步骤 3: 初始化 Trainer
        print("\n  🎓 初始化 Trainer...")
        try:
            # 确定计算设备
            use_cuda = torch.cuda.is_available()
            device = 'cuda' if use_cuda else 'cpu'
            
            if use_cuda:
                print(f"    🖥️  GPU 可用: {torch.cuda.get_device_name()}")
            else:
                print(f"    ⚠️  GPU 不可用，使用 CPU（速度会很慢）")
            
            # 使用分层映射初始化训练器
            trainer = Trainer(
                model=model,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                hierarchical_map=hierarchical_map,  # ✅ 传递分层映射
                device=device,
                output_dir=str(output_dir),
                verbose=True,
            )
            print_success("Trainer 初始化完成")
            print(f"  大类数: {len(hierarchical_map)}")
            print(f"  设备: {trainer.device}")
            print(f"  输出目录: {trainer.output_dir}")
        except Exception as e:
            print_error(f"Trainer 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 步骤 4: 执行训练
        print("\n  🚀 开始训练...")
        try:
            trainer.train(
                num_epochs=epochs,
                learning_rate=lr,
                weight_decay=weight_decay,
                patience=patience,
            )
            print_success("训练完成")
        except Exception as e:
            print_error(f"训练过程出错: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 步骤 5: 加载最佳模型并进行测试
        print("\n  🧪 加载最佳模型进行测试...")
        try:
            best_model_path = Path(output_dir) / 'best_model.pth'
            if best_model_path.exists():
                trainer.load_checkpoint(str(best_model_path))
                print_success(f"最佳模型已加载: {best_model_path}")
            else:
                print_warning("最佳模型不存在，使用当前模型进行测试")
            
            # 执行测试
            test_metrics = trainer.test(test_loader)
            print_success("测试完成")
            print(f"\n  测试集结果:")
            print(f"    大类准确率: {test_metrics.get('major_accuracy', 0):.4f}")
            print(f"    小类准确率: {test_metrics.get('detail_accuracy', 0):.4f}")
            print(f"    层级准确率: {test_metrics.get('hierarchical_accuracy', 0):.4f}")
            print(f"    大类F1-Score: {test_metrics.get('major_f1', 0):.4f}")
            print(f"    小类F1-Score: {test_metrics.get('detail_f1', 0):.4f}")
        except Exception as e:
            print_warning(f"测试过程出错: {e}")
            import traceback
            traceback.print_exc()
        
        # 步骤 6: 生成训练报告
        print("\n  📊 生成训练报告...")
        try:
            report_file = Path(output_dir) / 'training_report.json'
            training_info = {
                'status': 'completed',
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': lr,
                'weight_decay': weight_decay,
                'patience': patience,
                'model_name': 'HierarchicalDualStreamNetwork',
                'device': str(trainer.device),
                'train_samples': len(train_dataset),
                'val_samples': len(val_dataset),
                'test_samples': len(test_dataset),
                'test_metrics': test_metrics if 'test_metrics' in locals() else {},
                'hierarchical_map': hierarchical_map,
            }
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(training_info, f, ensure_ascii=False, indent=2)
            
            print_success(f"训练报告已保存: {report_file}")
        except Exception as e:
            print_warning(f"生成训练报告失败: {e}")
        
        print_success("第六阶段完成")
        return True
        
    except Exception as e:
        print_error(f"第六阶段失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主程序入口"""
    setup_logging()
    
    print_header("地理空间数据处理流水线")
    
    try:
        # =====================================================================
        # 第一阶段：配置管理与基础设施搭建
        # =====================================================================
        config = phase_1_configuration()
        if config is None:
            return 1
        
        # =====================================================================
        # 第二阶段：数据清洗与时空索引构建
        # =====================================================================
        encoder, dynamic_crawler, static_crawler, hierarchical_map, dynamic_channels, static_channels = phase_2_data_ingestion(config)
        
        if encoder is None:
            return 1
        
        # =====================================================================
        # 第三阶段：在线统计与数据归一化
        # =====================================================================
        if not phase_3_statistical_analysis(config, dynamic_crawler, static_crawler):
            return 1
        
        # =====================================================================
        # 第四阶段：自定义时空数据集构建
        # =====================================================================
        train_dataset, val_dataset, test_dataset = phase_4_dataset_construction(
            config, encoder, dynamic_crawler, static_crawler
        )
        
        if train_dataset is None:
            return 1
        
        # =====================================================================
        # 第五阶段：模型架构设计
        # =====================================================================
        model = phase_5_model_architecture(
            config, hierarchical_map, dynamic_channels, static_channels
        )
        
        if model is None:
            return 1
        
        # =====================================================================
        # 第六阶段：训练循环与日志系统
        # =====================================================================
        output_dir = config.get_experiment_output_dir()
        success = phase_6_training_loop(
            config, model, train_dataset, val_dataset, test_dataset, output_dir, hierarchical_map
        )
        
        if not success:
            return 1
        
        # =====================================================================
        # 完成
        # =====================================================================
        print_header("✅ 完整流水线执行成功！")
        
        print(f"\n📁 输出文件位置: {output_dir}")
        print(f"\n📋 生成文件清单:")
        print(f"  ├── config_used.yaml                # 配置副本")
        print(f"  ├── data_inventory.csv              # 数据清单")
        print(f"  ├── verification_report.json        # 验证报告")
        print(f"  ├── data_summary.txt                # 摘要")
        print(f"  ├── detailed_labels_map.json        # 详细类别映射")
        print(f"  ├── major_labels_map.json           # 大类映射")
        print(f"  ├── hierarchical_labels_map.json    # 层级映射")
        print(f"  ├── labels_geodata.geojson          # GeoJSON标签")
        print(f"  ├── rasters_metadata.json           # 栅格元数据")
        print(f"  ├── rasters_summary.json            # 栅格汇总")
        print(f"  ├── normalization_stats.json        # 归一化参数")
        print(f"  ├── dataset_info.json               # 数据集元数据")
        print(f"  ├── model_architecture.json         # 模型架构信息")
        print(f"  ├── detected_parameters.json        # 自动检测参数")
        print(f"  ├── training_report.json            # 训练报告")
        print(f"  ├── best_model.pth                  # 最佳模型权重")
        print(f"  ├── last_model.pth                  # 最后一个模型")
        print(f"  ├── training_log.txt                # 训练日志")
        print(f"  └── metrics_curves.png              # 指标曲线图")
        
        # 保存自动检测到的参数
        print(f"\n📊 自动检测参数:")
        detected_params = {
            'num_major_classes': len(hierarchical_map),
            'hierarchical_map': hierarchical_map,
            'dynamic_channels': dynamic_channels,
            'static_channels': static_channels,
        }
        print(f"  ├── 大类数: {len(hierarchical_map)}")
        for major_id, major_info in sorted(hierarchical_map.items()):
            num_detail = len(major_info.get('detail_classes', {}))
            print(f"  │  ├── {major_info.get('name', f'Major_{major_id}')}: {num_detail} 个小类")
        print(f"  ├── 动态影像波段数: {dynamic_channels}")
        print(f"  └── 静态影像波段数: {static_channels}")
        
        # 保存到文件
        detected_file = Path(output_dir) / 'detected_parameters.json'
        with open(detected_file, 'w', encoding='utf-8') as f:
            json.dump(detected_params, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 自动检测参数已保存: {detected_file}")
        
        # 打印数据集统计信息
        print(f"\n📊 数据集统计:")
        print(f"  ├── 训练集: {len(train_dataset)} 个样本")
        print(f"  ├── 验证集: {len(val_dataset)} 个样本")
        print(f"  └── 测试集: {len(test_dataset)} 个样本")
        
        print("\n" + "=" * 80)
        print("🎉 六阶段完整流水线执行成功！")
        print("=" * 80 + "\n")
        
        return 0
        
    except Exception as e:
        print_error(f"流水线执行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
