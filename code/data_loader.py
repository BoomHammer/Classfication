"""
DataLoader: 整合数据加载模块

功能：
1. 整合 ConfigManager、LabelEncoder、RasterCrawler
2. 为每个标签点关联时间序列栅格
3. 生成用于模型训练的数据索引
4. 支持时间序列采样和数据增强
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import pandas as pd
import geopandas as gpd
import numpy as np


class DataLoader:
    """
    数据加载类
    
    整合所有组件：
    1. ConfigManager - 配置管理
    2. LabelEncoder - 标签编码
    3. RasterCrawler - 影像索引
    
    功能：
    - 为每个标签点关联时间序列栅格
    - 生成训练数据索引
    - 支持数据采样和划分
    
    使用示例：
        loader = DataLoader(config=config)
        train_index, val_index = loader.create_train_val_split(test_size=0.2)
        sample = loader.get_sample(sample_id=0)
    """
    
    def __init__(
        self,
        config: 'ConfigManager',
        encoder: Optional['LabelEncoder'] = None,
        crawler: Optional['RasterCrawler'] = None,
    ):
        """
        初始化数据加载器
        
        Args:
            config: ConfigManager 对象
            encoder: LabelEncoder 对象（如果为 None，则创建新实例）
            crawler: RasterCrawler 对象（如果为 None，则创建新实例）
        """
        self._setup_logging()
        logger = logging.getLogger(__name__)
        
        self.config = config
        self.output_dir = config.get_experiment_output_dir()
        
        logger.info("📊 初始化 DataLoader...")
        
        # 初始化或使用提供的 encoder
        if encoder is None:
            logger.info("📝 初始化 LabelEncoder...")
            try:
                from label_encoder import LabelEncoder
                self.encoder = LabelEncoder(config=config)
            except ImportError:
                error_msg = "❌ LabelEncoder 不可用"
                logger.error(error_msg)
                raise ImportError(error_msg)
        else:
            self.encoder = encoder
            logger.info("✅ 使用已有的 LabelEncoder")
        
        # 初始化或使用提供的 crawler
        if crawler is None:
            logger.info("📚 初始化 RasterCrawler...")
            try:
                from raster_crawler import RasterCrawler
                filename_pattern = config.get('data_specs.raster_crawler.filename_pattern')
                self.crawler = RasterCrawler(config=config, filename_pattern=filename_pattern)
            except ImportError:
                logger.warning("⚠️  RasterCrawler 不可用，将不能关联时间序列栅格")
                self.crawler = None
            except Exception as e:
                logger.warning(f"⚠️  RasterCrawler 初始化失败: {e}")
                self.crawler = None
        else:
            self.crawler = crawler
            logger.info("✅ 使用已有的 RasterCrawler")
        
        # 获取标签数据
        logger.info("📥 加载标签数据...")
        self.labels_gdf = self.encoder.get_geodataframe()
        self.sample_count = len(self.labels_gdf)
        logger.info(f"✅ 加载了 {self.sample_count} 个样本")
        
        # 关联栅格
        if self.crawler:
            logger.info("🔗 关联时间序列栅格...")
            self._associate_rasters()
            logger.info("✅ 栅格关联完成")
        
        # 生成训练索引
        self.train_indices = None
        self.val_indices = None
        
        logger.info("✅ DataLoader 初始化完成")
    
    @staticmethod
    def _setup_logging():
        """配置日志系统"""
        if not logging.getLogger(__name__).handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logging.getLogger(__name__).addHandler(handler)
            logging.getLogger(__name__).setLevel(logging.INFO)
    
    def _associate_rasters(self):
        """
        为每个标签点关联时间序列栅格
        
        使用 RasterCrawler 的批量索引功能（高效）
        """
        logger = logging.getLogger(__name__)
        
        # 为每个点查找包含的栅格
        logger.info(f"🔍 为 {len(self.labels_gdf)} 个点查询栅格...")
        
        raster_files_list = []
        for idx, row in self.labels_gdf.iterrows():
            x, y = row['x'], row['y']
            rasters = self.crawler.find_rasters_by_point(x, y)
            raster_files = [m.filepath for m in rasters]
            raster_files_list.append(raster_files)
        
        self.labels_gdf['raster_files'] = raster_files_list
        
        # 统计关联结果
        raster_counts = [len(rf) for rf in raster_files_list]
        logger.info(f"✅ 栅格关联统计:")
        logger.info(f"   平均每点 {np.mean(raster_counts):.1f} 个栅格")
        logger.info(f"   最多 {max(raster_counts)} 个栅格")
        logger.info(f"   最少 {min(raster_counts)} 个栅格")
        
        # 统计覆盖情况
        covered = sum(1 for rc in raster_counts if rc > 0)
        logger.info(f"   {covered}/{len(raster_counts)} 个点有覆盖栅格 ({covered/len(raster_counts)*100:.1f}%)")
    
    def get_sample(self, sample_id: int) -> Dict:
        """
        获取单个样本的完整信息
        
        Args:
            sample_id: 样本 ID（0-based 索引）
        
        Returns:
            Dict: 包含标签、坐标、栅格列表的字典
        
        Example:
            >>> sample = loader.get_sample(0)
            >>> print(sample['detail_label'])
            >>> print(sample['raster_files'])
        """
        if sample_id < 0 or sample_id >= len(self.labels_gdf):
            raise ValueError(f"样本 ID {sample_id} 超出范围 [0, {len(self.labels_gdf)-1}]")
        
        row = self.labels_gdf.iloc[sample_id]
        
        sample = {
            'sample_id': sample_id,
            'x': row['x'],
            'y': row['y'],
            'detail_class': row.get('detail_class', 'unknown'),
            'major_class': row.get('major_class', 'unknown'),
            'detail_label': int(row['detail_label']),
            'major_label': int(row['major_label']),
        }
        
        # 添加栅格信息
        if 'raster_files' in row:
            raster_files = row['raster_files']
            sample['raster_count'] = len(raster_files)
            sample['raster_files'] = [str(f) for f in raster_files]
        else:
            sample['raster_count'] = 0
            sample['raster_files'] = []
        
        return sample
    
    def get_samples_batch(self, sample_ids: List[int]) -> List[Dict]:
        """
        批量获取样本信息
        
        Args:
            sample_ids: 样本 ID 列表
        
        Returns:
            List[Dict]: 样本列表
        """
        return [self.get_sample(sid) for sid in sample_ids]
    
    def create_train_val_split(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[List[int], List[int]]:
        """
        创建训练/验证集分割
        
        Args:
            test_size: 验证集比例 (0-1)
            random_state: 随机种子
        
        Returns:
            Tuple: (train_indices, val_indices)
        
        Example:
            >>> train_idx, val_idx = loader.create_train_val_split(test_size=0.2)
            >>> print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
        """
        logger = logging.getLogger(__name__)
        
        np.random.seed(random_state)
        
        total = len(self.labels_gdf)
        indices = np.arange(total)
        np.random.shuffle(indices)
        
        split_point = int(total * (1 - test_size))
        
        self.train_indices = indices[:split_point].tolist()
        self.val_indices = indices[split_point:].tolist()
        
        logger.info(f"✅ 数据分割完成:")
        logger.info(f"   训练集: {len(self.train_indices)} ({len(self.train_indices)/total*100:.1f}%)")
        logger.info(f"   验证集: {len(self.val_indices)} ({len(self.val_indices)/total*100:.1f}%)")
        
        return self.train_indices, self.val_indices
    
    def create_class_balanced_split(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[List[int], List[int]]:
        """
        创建类别平衡的训练/验证集分割
        
        确保每个类别在训练集和验证集中的比例一致
        
        Args:
            test_size: 验证集比例
            random_state: 随机种子
        
        Returns:
            Tuple: (train_indices, val_indices)
        """
        logger = logging.getLogger(__name__)
        
        np.random.seed(random_state)
        
        train_indices = []
        val_indices = []
        
        # 按类别分割
        for major_label in self.labels_gdf['major_label'].unique():
            mask = self.labels_gdf['major_label'] == major_label
            class_indices = self.labels_gdf[mask].index.tolist()
            
            np.random.shuffle(class_indices)
            split_point = int(len(class_indices) * (1 - test_size))
            
            train_indices.extend(class_indices[:split_point])
            val_indices.extend(class_indices[split_point:])
        
        self.train_indices = train_indices
        self.val_indices = val_indices
        
        logger.info(f"✅ 类别平衡分割完成:")
        logger.info(f"   训练集: {len(self.train_indices)}")
        logger.info(f"   验证集: {len(self.val_indices)}")
        
        return self.train_indices, self.val_indices
    
    def get_class_distribution(self, indices: Optional[List[int]] = None) -> Dict:
        """
        获取类别分布统计
        
        Args:
            indices: 要统计的样本索引。如果为 None，则统计全部
        
        Returns:
            Dict: 类别分布信息
        
        Example:
            >>> dist = loader.get_class_distribution(loader.train_indices)
            >>> print(dist)
        """
        if indices is None:
            subset = self.labels_gdf
        else:
            subset = self.labels_gdf.iloc[indices]
        
        detailed_dist = subset['detail_label'].value_counts().sort_index().to_dict()
        major_dist = subset['major_label'].value_counts().sort_index().to_dict()
        
        # 转换标签为类别名称
        detailed_dist_named = {
            self.encoder.label_to_category(k, 'detailed'): v
            for k, v in detailed_dist.items()
        }
        major_dist_named = {
            self.encoder.label_to_category(k, 'major'): v
            for k, v in major_dist.items()
        }
        
        return {
            'total_samples': len(subset),
            'detailed_classes': detailed_dist_named,
            'major_classes': major_dist_named,
        }
    
    def get_coverage_statistics(self) -> Dict:
        """
        获取栅格覆盖统计信息（仅当有 crawler 时）
        
        Returns:
            Dict: 覆盖统计
        """
        if not self.crawler or 'raster_files' not in self.labels_gdf.columns:
            return {'message': '栅格关联不可用'}
        
        raster_counts = [len(rf) for rf in self.labels_gdf['raster_files']]
        
        return {
            'total_samples': len(self.labels_gdf),
            'covered_samples': sum(1 for rc in raster_counts if rc > 0),
            'coverage_rate': sum(1 for rc in raster_counts if rc > 0) / len(raster_counts),
            'avg_rasters_per_point': np.mean(raster_counts),
            'max_rasters': max(raster_counts),
            'min_rasters': min(raster_counts),
            'raster_distribution': {
                f'{i}_rasters': sum(1 for rc in raster_counts if rc == i)
                for i in range(max(raster_counts) + 1)
            }
        }
    
    def save_index(self, output_path: Optional[Path] = None):
        """
        保存数据索引到 CSV 文件
        
        Args:
            output_path: 输出路径。如果为 None，则保存到实验目录
        
        Example:
            >>> loader.save_index()
            # 保存到 {experiment_output_dir}/data_index.csv
        """
        logger = logging.getLogger(__name__)
        
        if output_path is None:
            output_path = self.output_dir / 'data_index.csv'
        else:
            output_path = Path(output_path)
        
        # 准备输出数据
        export_data = []
        for idx, row in self.labels_gdf.iterrows():
            data = {
                'sample_id': idx,
                'x': row['x'],
                'y': row['y'],
                'detail_class': row.get('detail_class', 'unknown'),
                'major_class': row.get('major_class', 'unknown'),
                'detail_label': int(row['detail_label']),
                'major_label': int(row['major_label']),
            }
            
            if 'raster_files' in row:
                raster_files = row['raster_files']
                data['raster_count'] = len(raster_files)
                data['raster_files'] = '|'.join([str(f) for f in raster_files])
            else:
                data['raster_count'] = 0
                data['raster_files'] = ''
            
            # 添加集合标记
            if self.train_indices and idx in self.train_indices:
                data['split'] = 'train'
            elif self.val_indices and idx in self.val_indices:
                data['split'] = 'val'
            else:
                data['split'] = 'unknown'
            
            export_data.append(data)
        
        # 保存为 CSV
        export_df = pd.DataFrame(export_data)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        export_df.to_csv(output_path, index=False, encoding='utf-8')
        
        logger.info(f"💾 数据索引已保存: {output_path}")
    
    def get_statistics(self) -> Dict:
        """
        获取数据加载器统计信息
        
        Returns:
            Dict: 统计信息
        """
        stats = {
            'total_samples': len(self.labels_gdf),
            'detailed_classes': len(self.encoder.detailed_labels_map),
            'major_classes': len(self.encoder.major_labels_map),
            'class_distribution': self.get_class_distribution(),
        }
        
        if self.crawler:
            stats['coverage_statistics'] = self.get_coverage_statistics()
        
        if self.train_indices is not None:
            stats['train_size'] = len(self.train_indices)
            stats['val_size'] = len(self.val_indices)
        
        return stats
    
    def __repr__(self) -> str:
        """字符串表示"""
        crawler_status = "✅" if self.crawler else "❌"
        return (
            f"DataLoader(\n"
            f"  total_samples={len(self.labels_gdf)},\n"
            f"  detailed_classes={len(self.encoder.detailed_labels_map)},\n"
            f"  major_classes={len(self.encoder.major_labels_map)},\n"
            f"  raster_crawler={crawler_status},\n"
            f"  train_size={len(self.train_indices) if self.train_indices else 'None'},\n"
            f"  val_size={len(self.val_indices) if self.val_indices else 'None'}\n"
            f")"
        )


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    try:
        from config_manager import ConfigManager
        from label_encoder import LabelEncoder
        
        print("=" * 80)
        print("DataLoader 使用示例")
        print("=" * 80)
        
        # 初始化
        print("\n1️⃣  初始化 DataLoader...")
        config = ConfigManager('./config.yaml')
        encoder = LabelEncoder(config=config)
        
        # 尝试初始化 RasterCrawler
        try:
            from raster_crawler import RasterCrawler
            filename_pattern = config.get('data_specs.raster_crawler.filename_pattern')
            crawler = RasterCrawler(config=config, filename_pattern=filename_pattern)
            print("   ✅ RasterCrawler 已初始化")
        except Exception as e:
            print(f"   ⚠️  RasterCrawler 初始化失败: {e}")
            crawler = None
        
        # 创建 DataLoader
        loader = DataLoader(config=config, encoder=encoder, crawler=crawler)
        print(f"\n{loader}\n")
        
        # 获取样本
        print("2️⃣  获取样本信息...")
        sample = loader.get_sample(0)
        print(f"   样本 0:")
        for key, value in sample.items():
            if key != 'raster_files':
                print(f"      {key}: {value}")
            else:
                print(f"      {key}: {len(value)} 个文件")
        
        # 创建数据分割
        print("\n3️⃣  创建训练/验证分割...")
        train_idx, val_idx = loader.create_train_val_split(test_size=0.2)
        
        # 获取类别分布
        print("\n4️⃣  获取类别分布...")
        dist = loader.get_class_distribution()
        print(f"   全部: {dist['total_samples']} 个样本")
        print(f"   详细类别: {dist['detailed_classes']}")
        print(f"   大类: {dist['major_classes']}")
        
        # 获取统计信息
        print("\n5️⃣  获取统计信息...")
        stats = loader.get_statistics()
        print(f"   ✅ 统计完成:")
        for key, value in stats.items():
            if key != 'class_distribution' and key != 'coverage_statistics':
                print(f"      {key}: {value}")
        
        # 保存索引
        print("\n6️⃣  保存数据索引...")
        loader.save_index()
        print("   ✅ 索引已保存")
        
        print("\n" + "=" * 80)
        print("✅ DataLoader 示例完成!")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
