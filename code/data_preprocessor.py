"""
DataPreprocessor: 数据预处理与验证模块

功能：
1. 整合 ConfigManager、LabelEncoder、RasterCrawler
2. 执行数据质量验证（坐标检查、CRS一致性等）
3. 生成数据清单文件（data_inventory.csv）
4. 生成验证报告（verification_report.json）
5. 输出详细的控制台日志

使用示例：
    preprocessor = DataPreprocessor(config=config)
    preprocessor.run()
"""

import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np


class DataPreprocessor:
    """
    数据预处理类
    
    执行完整的数据验证和清单生成流程
    """
    
    def __init__(
        self,
        config: 'ConfigManager',
        encoder: Optional['LabelEncoder'] = None,
        dynamic_crawler: Optional['RasterCrawler'] = None,
        static_crawler: Optional['RasterCrawler'] = None,
    ):
        """
        初始化数据预处理器
        
        Args:
            config: ConfigManager 对象
            encoder: LabelEncoder 对象（如果为 None，则创建新实例）
            dynamic_crawler: 动态影像爬虫（如果为 None，则创建新实例）
            static_crawler: 静态影像爬虫（如果为 None，则创建新实例）
        """
        self._setup_logging()
        logger = logging.getLogger(__name__)
        
        logger.info("=" * 80)
        logger.info("🚀 数据预处理器启动")
        logger.info("=" * 80)
        
        self.config = config
        self.output_dir = config.get_experiment_output_dir()
        
        # 初始化各组件
        logger.info("\n📋 [阶段1] 初始化数据处理组件...")
        
        if encoder is None:
            from label_encoder import LabelEncoder
            logger.info("  📝 初始化 LabelEncoder...")
            self.encoder = LabelEncoder(config=config)
        else:
            self.encoder = encoder
            logger.info("  ✅ 使用已有的 LabelEncoder")
        
        if dynamic_crawler is None:
            from raster_crawler import RasterCrawler
            logger.info("  📚 初始化动态影像爬虫...")
            try:
                self.dynamic_crawler = RasterCrawler(
                    config=config,
                    raster_dir=config.get_resolved_path('dynamic_images_dir'),
                    filename_pattern=config.get('data_specs.raster_crawler.filename_pattern'),
                    file_extensions=tuple(config.get('data_specs.raster_crawler.file_extensions', ['.tif', '.tiff', '.jp2'])),
                )
            except Exception as e:
                logger.warning(f"  ⚠️  动态影像爬虫初始化失败: {e}")
                self.dynamic_crawler = None
        else:
            self.dynamic_crawler = dynamic_crawler
            logger.info("  ✅ 使用已有的动态影像爬虫")
        
        if static_crawler is None:
            from raster_crawler import RasterCrawler
            logger.info("  📚 初始化静态影像爬虫...")
            try:
                self.static_crawler = RasterCrawler(
                    config=config,
                    raster_dir=config.get_resolved_path('static_images_dir'),
                    filename_pattern=config.get('data_specs.raster_crawler.filename_pattern'),
                    file_extensions=tuple(config.get('data_specs.raster_crawler.file_extensions', ['.tif', '.tiff', '.jp2'])),
                )
            except Exception as e:
                logger.warning(f"  ⚠️  静态影像爬虫初始化失败: {e}")
                self.static_crawler = None
        else:
            self.static_crawler = static_crawler
            logger.info("  ✅ 使用已有的静态影像爬虫")
        
        # 数据存储
        self.labels_gdf = self.encoder.get_geodataframe()
        self.verification_results = {}
        self.data_inventory = []
        
        logger.info("  ✅ 所有组件初始化完成\n")
    
    @staticmethod
    def _setup_logging():
        """配置日志系统"""
        if not logging.getLogger(__name__).handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '[%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
            logger.addHandler(handler)
    
    def run(self):
        """
        执行完整的数据预处理流程
        
        包括：
        1. 标签数据验证
        2. 影像索引验证
        3. CRS一致性检查
        4. 生成数据清单
        5. 生成验证报告
        """
        logger = logging.getLogger(__name__)
        
        try:
            # 阶段1：标签处理验证
            self._verify_labels()
            
            # 阶段2：动态影像验证
            if self.dynamic_crawler:
                self._verify_dynamic_rasters()
            
            # 阶段3：静态影像验证
            if self.static_crawler:
                self._verify_static_rasters()
            
            # 阶段4：CRS一致性检查
            self._verify_crs_consistency()
            
            # 阶段5：生成数据清单
            self._generate_data_inventory()
            
            # 阶段6：保存报告
            self._save_reports()
            
            logger.info("\n" + "=" * 80)
            logger.info("✅ 数据预处理完成！")
            logger.info("=" * 80 + "\n")
            
        except Exception as e:
            logger.error(f"\n❌ 数据预处理失败: {e}")
            traceback.print_exc()
            raise
    
    def _verify_labels(self):
        """
        [阶段1] 标签数据验证
        
        检查项：
        1. 原始点数
        2. 有效点数（剔除坐标异常）
        3. 类别分布
        4. 投影信息
        """
        logger = logging.getLogger(__name__)
        logger.info("\n📋 [阶段1] 标签处理验证...")
        
        # 原始点数
        original_count = len(self.labels_gdf)
        logger.info(f"  原始点数: {original_count}")
        
        # 检查坐标有效性
        invalid_coords = []
        for idx, row in self.labels_gdf.iterrows():
            x, y = row['x'], row['y']
            # 检查NaN和极端值
            if pd.isna(x) or pd.isna(y) or np.isinf(x) or np.isinf(y):
                invalid_coords.append(idx)
        
        valid_count = original_count - len(invalid_coords)
        logger.info(f"  有效点数: {valid_count} (剔除坐标异常点 {len(invalid_coords)})")
        
        # 类别分布 - 使用编码器的配置列名
        major_class_col = self.encoder.major_class_col
        detail_class_col = self.encoder.detail_class_col
        
        major_dist = self.labels_gdf[major_class_col].value_counts().to_dict() if major_class_col in self.labels_gdf.columns else {}
        detailed_dist = self.labels_gdf[detail_class_col].value_counts().to_dict() if detail_class_col in self.labels_gdf.columns else {}
        
        logger.info(f"  大类数量: {len(major_dist)}")
        logger.info(f"  详细类别数: {len(detailed_dist)}")
        
        # 保存验证结果
        self.verification_results['labels'] = {
            'original_count': original_count,
            'valid_count': valid_count,
            'invalid_count': len(invalid_coords),
            'invalid_indices': invalid_coords,
            'major_classes': len(major_dist),
            'detailed_classes': len(detailed_dist),
            'major_distribution': major_dist,
            'detailed_distribution': detailed_dist,
            'target_crs': str(self.labels_gdf.crs),
        }
        
        # 类别映射
        detailed_map = self.encoder.get_detailed_labels_map()
        major_map = self.encoder.get_major_labels_map()
        
        logger.info(f"  类别映射: {detailed_map} -> 已保存至 detailed_labels_map.json")
        logger.info(f"  大类映射: {major_map} -> 已保存至 major_labels_map.json")
        
        logger.info("  ✅ 标签验证完成\n")
    
    def _verify_dynamic_rasters(self):
        """
        [阶段2] 动态影像验证
        
        检查项：
        1. 扫描文件数
        2. 成功解析的文件数
        3. 时间跨度
        4. 投影一致性
        """
        logger = logging.getLogger(__name__)
        logger.info("📋 [阶段2] 动态影像索引验证...")
        
        if not self.dynamic_crawler:
            logger.warning("  ⚠️  动态影像爬虫不可用")
            return
        
        rasters = self.dynamic_crawler.get_all_rasters()
        total_files = len(rasters)
        
        logger.info(f"  扫描文件数: {total_files}")
        
        # 统计成功解析的文件
        # 成功解析的标准：有 year 或 month 信息（不一定要有具体日期）
        # PR2020.tif 返回 date=None, year=2020, month=None，应该被认为是成功解析
        parsed_count = sum(1 for r in rasters if r.year is not None)
        unparsed_count = total_files - parsed_count
        
        logger.info(f"  成功解析时间元数据: {parsed_count} ({unparsed_count}个文件命名不规范被跳过)")
        
        # 时间跨度（基于有效的年份和日期信息）
        dates = [r.date for r in rasters if r.date is not None]
        if dates:
            min_date = min(dates)
            max_date = max(dates)
            logger.info(f"  时间跨度（日期）: {min_date.strftime('%Y-%m')} 至 {max_date.strftime('%Y-%m')}")
        
        # 统计所有有效的年份
        years = [r.year for r in rasters if r.year is not None]
        if years:
            min_year = min(years)
            max_year = max(years)
            logger.info(f"  年份范围: {min_year} 至 {max_year}")
        
        # 投影统计
        crs_set = set(r.crs for r in rasters)
        logger.info(f"  投影类型: {crs_set}")
        
        self.verification_results['dynamic_rasters'] = {
            'total_files': total_files,
            'parsed_files': parsed_count,
            'unparsed_files': unparsed_count,
            'time_range': {
                'start': min_date.isoformat() if dates else None,
                'end': max_date.isoformat() if dates else None,
            },
            'year_range': {
                'min': min_year if years else None,
                'max': max_year if years else None,
            },
            'crs_distribution': list(crs_set),
        }
        
        logger.info("  ✅ 动态影像验证完成\n")
    
    def _verify_static_rasters(self):
        """
        [阶段3] 静态影像验证
        
        检查项：
        1. 静态影像文件数
        2. 投影信息
        """
        logger = logging.getLogger(__name__)
        logger.info("📋 [阶段3] 静态影像索引验证...")
        
        if not self.static_crawler:
            logger.warning("  ⚠️  静态影像爬虫不可用")
            return
        
        rasters = self.static_crawler.get_all_rasters()
        total_files = len(rasters)
        
        logger.info(f"  扫描文件数: {total_files}")
        
        # 投影统计
        crs_set = set(r.crs for r in rasters)
        logger.info(f"  投影类型: {crs_set}")
        
        self.verification_results['static_rasters'] = {
            'total_files': total_files,
            'crs_distribution': list(crs_set),
        }
        
        logger.info("  ✅ 静态影像验证完成\n")
    
    def _verify_crs_consistency(self):
        """
        [CRS一致性检查与自动检测]
        
        功能：
        1. 使用 CRS 管理器自动检测所有影像的坐标系
        2. 验证坐标系一致性
        3. 生成详细的 CRS 报告
        4. 提供重投影建议
        """
        logger = logging.getLogger(__name__)
        logger.info("📋 [CRS一致性检查与自动检测]...")
        
        from crs_manager import CRSManager
        crs_manager = CRSManager(self.config)
        
        # 目标CRS和配置
        target_crs = self.config.get('data_specs.spatial.target_crs')
        auto_reproject = self.config.get('data_specs.spatial.auto_reproject', False)
        
        logger.info(f"  📍 目标投影: {target_crs}")
        logger.info(f"  🔄 自动重投影: {'启用' if auto_reproject else '禁用'}")
        
        crs_issues = []
        
        # 检查动态影像
        if self.dynamic_crawler:
            logger.info("\n  📚 检查动态影像坐标系...")
            dynamic_rasters = self.dynamic_crawler.get_all_rasters()
            
            if dynamic_rasters:
                # 使用 CRS 管理器检测一致性
                dynamic_filepaths = [r.filepath for r in dynamic_rasters]
                crs_validation = crs_manager.validate_multiple_crs(
                    dynamic_filepaths,
                    verbose=True
                )
                
                # 检查是否与目标 CRS 匹配
                if not crs_validation['is_consistent']:
                    logger.warning(f"  ⚠️  动态影像坐标系不一致")
                    for fp in crs_validation['inconsistent_files'][:3]:
                        crs_issues.append({
                            'type': 'dynamic',
                            'file': Path(fp).name,
                            'crs': crs_validation['crs_details'].get(fp, 'UNKNOWN'),
                            'expected': target_crs
                        })
                
                # 检查与目标的匹配
                if crs_validation['most_common_crs'] and crs_validation['most_common_crs'] != target_crs:
                    logger.warning(
                        f"  ⚠️  动态影像坐标系 ({crs_validation['most_common_crs']}) "
                        f"与目标坐标系 ({target_crs}) 不匹配"
                    )
                    
                    if auto_reproject:
                        logger.info(f"  🔄 建议进行自动重投影")
                else:
                    logger.info(f"  ✅ 动态影像坐标系验证通过")
        
        # 检查静态影像
        if self.static_crawler:
            logger.info("\n  📚 检查静态影像坐标系...")
            static_rasters = self.static_crawler.get_all_rasters()
            
            if static_rasters:
                # 使用 CRS 管理器检测一致性
                static_filepaths = [r.filepath for r in static_rasters]
                crs_validation = crs_manager.validate_multiple_crs(
                    static_filepaths,
                    verbose=True
                )
                
                # 检查是否与目标 CRS 匹配
                if not crs_validation['is_consistent']:
                    logger.warning(f"  ⚠️  静态影像坐标系不一致")
                    for fp in crs_validation['inconsistent_files'][:3]:
                        crs_issues.append({
                            'type': 'static',
                            'file': Path(fp).name,
                            'crs': crs_validation['crs_details'].get(fp, 'UNKNOWN'),
                            'expected': target_crs
                        })
                
                # 检查与目标的匹配
                if crs_validation['most_common_crs'] and crs_validation['most_common_crs'] != target_crs:
                    logger.warning(
                        f"  ⚠️  静态影像坐标系 ({crs_validation['most_common_crs']}) "
                        f"与目标坐标系 ({target_crs}) 不匹配"
                    )
                    
                    if auto_reproject:
                        logger.info(f"  🔄 建议进行自动重投影")
                else:
                    logger.info(f"  ✅ 静态影像坐标系验证通过")
        
        # 生成报告
        if self.dynamic_crawler:
            self.dynamic_crawler.save_crs_report()
        if self.static_crawler:
            self.static_crawler.save_crs_report()
        
        if crs_issues:
            logger.warning(f"\n  ⚠️  发现 {len(crs_issues)} 个坐标系问题")
        else:
            logger.info(f"\n  ✅ 所有文件坐标系验证通过")
        
        self.verification_results['crs_consistency'] = {
            'target_crs': target_crs,
            'auto_reproject': auto_reproject,
            'issues_count': len(crs_issues),
            'issues': crs_issues[:10],  # 保存前10个问题
        }
        
        logger.info("  ✅ CRS 检查完成\n")
    
    def _generate_data_inventory(self):
        """
        [阶段5] 生成数据清单
        
        创建 data_inventory.csv，包含所有文件的详细信息：
        - 文件路径
        - 文件类型（动态/静态）
        - 解析的日期
        - 投影系统
        - 空间范围（min_x, min_y, max_x, max_y）
        """
        logger = logging.getLogger(__name__)
        logger.info("📋 [阶段5] 生成数据清单...")
        
        inventory_list = []
        
        # 添加动态影像
        if self.dynamic_crawler:
            rasters = self.dynamic_crawler.get_all_rasters()
            for r in rasters:
                inventory_list.append({
                    'file_path': str(r.filepath),
                    'type': 'Dynamic',
                    'filename': r.filename,
                    'parsed_date': r.date.isoformat() if r.date else 'N/A',
                    'year': r.year,
                    'month': r.month,
                    'epsg': r.crs.split(':')[-1] if ':' in str(r.crs) else r.crs,
                    'min_x': r.bounds[0],
                    'min_y': r.bounds[1],
                    'max_x': r.bounds[2],
                    'max_y': r.bounds[3],
                    'width': r.width,
                    'height': r.height,
                    'resolution_x': r.resolution[0],
                    'resolution_y': r.resolution[1],
                })
        
        # 添加静态影像
        if self.static_crawler:
            rasters = self.static_crawler.get_all_rasters()
            for r in rasters:
                inventory_list.append({
                    'file_path': str(r.filepath),
                    'type': 'Static',
                    'filename': r.filename,
                    'parsed_date': 'N/A',
                    'year': None,
                    'month': None,
                    'epsg': r.crs.split(':')[-1] if ':' in str(r.crs) else r.crs,
                    'min_x': r.bounds[0],
                    'min_y': r.bounds[1],
                    'max_x': r.bounds[2],
                    'max_y': r.bounds[3],
                    'width': r.width,
                    'height': r.height,
                    'resolution_x': r.resolution[0],
                    'resolution_y': r.resolution[1],
                })
        
        self.data_inventory = inventory_list
        
        # 创建 DataFrame
        inventory_df = pd.DataFrame(inventory_list)
        
        # 保存为 CSV
        inventory_path = self.output_dir / 'data_inventory.csv'
        inventory_df.to_csv(inventory_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"  文件总数: {len(inventory_list)}")
        logger.info(f"  动态影像: {sum(1 for item in inventory_list if item['type'] == 'Dynamic')}")
        logger.info(f"  静态影像: {sum(1 for item in inventory_list if item['type'] == 'Static')}")
        logger.info(f"  ✅ 数据清单已保存: {inventory_path}\n")
    
    def _save_reports(self):
        """
        [阶段6] 保存验证报告
        
        生成以下文件：
        1. verification_report.json - 完整的验证报告
        2. data_summary.txt - 人类可读的摘要报告
        """
        logger = logging.getLogger(__name__)
        logger.info("📋 [阶段6] 保存验证报告...")
        
        # 保存完整的验证结果
        report_path = self.output_dir / 'verification_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.verification_results, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"  ✅ 验证报告已保存: {report_path}")
        
        # 生成摘要报告
        summary_path = self.output_dir / 'data_summary.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("📊 数据预处理验证报告\n")
            f.write("=" * 80 + "\n\n")
            
            # 标签统计
            if 'labels' in self.verification_results:
                labels = self.verification_results['labels']
                f.write("【标签数据统计】\n")
                f.write(f"  原始点数: {labels['original_count']}\n")
                f.write(f"  有效点数: {labels['valid_count']} (剔除异常点: {labels['invalid_count']})\n")
                f.write(f"  大类数: {labels['major_classes']}\n")
                f.write(f"  详细类别数: {labels['detailed_classes']}\n")
                f.write(f"  投影系统: {labels['target_crs']}\n")
                f.write(f"  大类分布: {labels['major_distribution']}\n")
                f.write(f"  详细分布: {labels['detailed_distribution']}\n\n")
            
            # 动态影像统计
            if 'dynamic_rasters' in self.verification_results:
                dynamic = self.verification_results['dynamic_rasters']
                f.write("【动态影像统计】\n")
                f.write(f"  总文件数: {dynamic['total_files']}\n")
                f.write(f"  成功解析: {dynamic['parsed_files']}\n")
                f.write(f"  解析失败: {dynamic['unparsed_files']}\n")
                if dynamic['time_range']['start']:
                    f.write(f"  时间范围: {dynamic['time_range']['start']} ~ {dynamic['time_range']['end']}\n")
                f.write(f"  投影类型: {dynamic['crs_distribution']}\n\n")
            
            # 静态影像统计
            if 'static_rasters' in self.verification_results:
                static = self.verification_results['static_rasters']
                f.write("【静态影像统计】\n")
                f.write(f"  总文件数: {static['total_files']}\n")
                f.write(f"  投影类型: {static['crs_distribution']}\n\n")
            
            # CRS一致性检查
            if 'crs_consistency' in self.verification_results:
                crs = self.verification_results['crs_consistency']
                f.write("【CRS一致性检查】\n")
                f.write(f"  目标投影: {crs['target_crs']}\n")
                f.write(f"  不一致文件数: {crs['issues_count']}\n")
                if crs['issues']:
                    f.write(f"  问题详情:\n")
                    for issue in crs['issues']:
                        f.write(f"    - {issue['file']}: {issue['crs']}\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"  ✅ 摘要报告已保存: {summary_path}")
        logger.info("  ✅ 所有报告已保存\n")


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    try:
        from config_manager import ConfigManager
        
        print("=" * 80)
        print("🚀 数据预处理脚本")
        print("=" * 80)
        
        # 初始化配置
        config = ConfigManager('./config.yaml')
        
        # 创建并运行预处理器
        preprocessor = DataPreprocessor(config=config)
        preprocessor.run()
        
        print("\n" + "=" * 80)
        print("✅ 数据预处理完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        traceback.print_exc()
        sys.exit(1)
