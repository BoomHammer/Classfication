#!/usr/bin/env python3
"""
preprocess.py: 数据预处理主脚本

执行流程：
1. 加载配置
2. 初始化所有数据处理组件
3. 执行完整的数据验证流程
4. 生成数据清单和报告

运行方式：
    python preprocess.py

输出文件：
    - data_inventory.csv       # 数据清单
    - verification_report.json # 验证报告
    - data_summary.txt         # 摘要报告
    - detailed_labels_map.json # 详细类别映射
    - major_labels_map.json    # 大类映射
    - hierarchical_labels_map.json # 层级映射
    - rasters_metadata.json    # 影像元数据
    - rasters_summary.json     # 影像汇总
"""

import sys
import logging
from pathlib import Path

# 确保能够导入本地模块
sys.path.insert(0, str(Path(__file__).parent))

from config_manager import ConfigManager
from label_encoder import LabelEncoder
from raster_crawler import RasterCrawler
from data_preprocessor import DataPreprocessor


def setup_logging():
    """配置全局日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s'
    )


def main():
    """主程序入口"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # =====================================================
        # 第一步：加载配置
        # =====================================================
        logger.info("=" * 80)
        logger.info("🚀 数据预处理流程启动")
        logger.info("=" * 80)
        logger.info("\n📋 [第一步] 加载配置文件...")
        
        config_path = Path(__file__).parent / 'config.yaml'
        config = ConfigManager(str(config_path))
        
        logger.info(f"  ✅ 配置加载成功")
        logger.info(f"  📂 项目: {config.get('project_name')}")
        logger.info(f"  📊 实验ID: {config.get('experiment_id')}")
        logger.info(f"  📁 输出目录: {config.get_experiment_output_dir()}\n")
        
        # =====================================================
        # 第二步：初始化数据处理组件
        # =====================================================
        logger.info("📋 [第二步] 初始化数据处理组件...")
        
        # 初始化标签编码器
        logger.info("  📝 初始化 LabelEncoder...")
        encoder = LabelEncoder(config=config)
        logger.info(f"    ✅ 加载了 {encoder.get_statistics()['total_samples']} 个标签样本")
        
        # 初始化动态影像爬虫
        dynamic_crawler = None
        try:
            logger.info("  📚 初始化动态影像爬虫...")
            dynamic_crawler = RasterCrawler(
                config=config,
                raster_dir=config.get_resolved_path('dynamic_images_dir'),
                filename_pattern=config.get('data_specs.raster_crawler.filename_pattern'),
                file_extensions=tuple(config.get('data_specs.raster_crawler.file_extensions', ['.tif', '.tiff', '.jp2'])),
            )
            logger.info(f"    ✅ 扫描了 {dynamic_crawler.get_raster_count()} 个动态影像文件")
        except Exception as e:
            logger.warning(f"    ⚠️  动态影像爬虫初始化失败: {e}")
        
        # 初始化静态影像爬虫
        static_crawler = None
        try:
            logger.info("  📚 初始化静态影像爬虫...")
            static_crawler = RasterCrawler(
                config=config,
                raster_dir=config.get_resolved_path('static_images_dir'),
                filename_pattern=config.get('data_specs.raster_crawler.filename_pattern'),
                file_extensions=tuple(config.get('data_specs.raster_crawler.file_extensions', ['.tif', '.tiff', '.jp2'])),
            )
            logger.info(f"    ✅ 扫描了 {static_crawler.get_raster_count()} 个静态影像文件")
        except Exception as e:
            logger.warning(f"    ⚠️  静态影像爬虫初始化失败: {e}")
        
        logger.info("  ✅ 所有组件初始化完成\n")
        
        # =====================================================
        # 第三步：运行数据预处理和验证
        # =====================================================
        logger.info("📋 [第三步] 执行数据验证和清单生成...\n")
        
        preprocessor = DataPreprocessor(
            config=config,
            encoder=encoder,
            dynamic_crawler=dynamic_crawler,
            static_crawler=static_crawler,
        )
        preprocessor.run()
        
        # =====================================================
        # 完成
        # =====================================================
        logger.info("=" * 80)
        logger.info("✅ 数据预处理完成！")
        logger.info("=" * 80)
        logger.info("\n📁 输出文件位置:")
        output_dir = config.get_experiment_output_dir()
        logger.info(f"  {output_dir}/")
        logger.info(f"    ├── data_inventory.csv          # 📊 数据清单（关键文件）")
        logger.info(f"    ├── verification_report.json    # 📋 完整验证报告")
        logger.info(f"    ├── data_summary.txt            # 📄 摘要报告")
        logger.info(f"    ├── detailed_labels_map.json    # 🏷️  详细类别映射")
        logger.info(f"    ├── major_labels_map.json       # 🏷️  大类映射")
        logger.info(f"    ├── hierarchical_labels_map.json # 🏷️  层级映射")
        logger.info(f"    ├── rasters_metadata.json       # 🗺️  影像元数据")
        logger.info(f"    └── rasters_summary.json        # 🗺️  影像汇总\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
