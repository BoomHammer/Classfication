"""
LabelEncoder: 标签编码与类别映射模块

功能：
1. 读取 CSV 标签文件
2. 生成类别映射（详细类别和大类）
3. 空间投影转换（经纬度 → Target CRS）
4. 层级标签处理（大类 + 详细类别）
5. 保存映射到 JSON 文件
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


class LabelEncoder:
    """
    标签编码类
    
    功能：
    1. 加载 CSV 标签文件并验证
    2. 生成类别映射（大类和详细类别）
    3. 进行空间投影转换
    4. 生成层级标签
    5. 保存映射到实验目录
    
    使用示例：
        encoder = LabelEncoder(
            config=config,
            csv_path=csv_path,
            output_dir=output_dir
        )
        detailed_map = encoder.get_detailed_labels_map()
        major_map = encoder.get_major_labels_map()
        gdf = encoder.get_geodataframe()
    """
    
    def __init__(
        self,
        config: 'ConfigManager',
        csv_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        target_crs: Optional[str] = None,
    ):
        """
        初始化 LabelEncoder
        
        Args:
            config: ConfigManager 对象
            csv_path: CSV 标签文件路径（如果为 None，则从 config 读取）
            output_dir: 输出目录（如果为 None，则从 config 读取）
            target_crs: 目标投影系统（如果为 None，则从 config 读取）
        
        Raises:
            FileNotFoundError: CSV 文件不存在
            ValueError: 配置或数据格式错误
        """
        self._setup_logging()
        logger = logging.getLogger(__name__)
        
        # 保存配置
        self.config = config
        
        # 解析参数
        self.csv_path = Path(csv_path) if csv_path else config.get_resolved_path('csv_labels')
        self.output_dir = Path(output_dir) if output_dir else config.get_experiment_output_dir()
        self.target_crs = target_crs or config.get('data_specs.spatial.target_crs')
        
        logger.info(f"📂 CSV 路径: {self.csv_path}")
        logger.info(f"📂 输出目录: {self.output_dir}")
        logger.info(f"🗺️  目标投影: {self.target_crs}")
        
        # 验证文件存在
        if not self.csv_path.exists():
            error_msg = f"❌ CSV 文件不存在: {self.csv_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # 读取配置中的列名映射
        csv_cols_config = config.get('data_specs.csv_columns')
        self.id_col = csv_cols_config.get('id')
        self.lon_col = csv_cols_config.get('longitude')
        self.lat_col = csv_cols_config.get('latitude')
        self.major_class_col = csv_cols_config.get('major_class')
        self.detail_class_col = csv_cols_config.get('detail_class')
        
        logger.info(f"📋 CSV 列配置:")
        logger.info(f"   ID 列: {self.id_col}")
        logger.info(f"   经度列: {self.lon_col}")
        logger.info(f"   纬度列: {self.lat_col}")
        logger.info(f"   大类列: {self.major_class_col}")
        logger.info(f"   详细类别列: {self.detail_class_col}")
        
        # 初始化数据存储
        self.df = None
        self.gdf = None
        self.detailed_labels_map = None
        self.major_labels_map = None
        self.inverse_detailed_map = None
        self.inverse_major_map = None
        
        # 加载和处理数据
        logger.info("🔍 开始处理标签...")
        self._load_csv()
        self._validate_columns()
        self._generate_labels_maps()
        self._transform_crs()
        logger.info("✅ 标签处理完成")
        
        # 保存映射
        self._save_maps()
    
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
    
    def _load_csv(self):
        """
        加载 CSV 文件
        
        Raises:
            ValueError: CSV 格式错误
        """
        logger = logging.getLogger(__name__)
        
        try:
            self.df = pd.read_csv(self.csv_path, encoding='utf-8')
            logger.info(f"✅ CSV 加载成功: {len(self.df)} 行")
        except UnicodeDecodeError:
            # 尝试其他编码
            self.df = pd.read_csv(self.csv_path, encoding='gbk')
            logger.info(f"✅ CSV 加载成功（使用 GBK 编码）: {len(self.df)} 行")
        except Exception as e:
            error_msg = f"❌ CSV 加载失败: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _validate_columns(self):
        """
        验证必要的列是否存在
        
        Raises:
            ValueError: 缺少必要列
        """
        logger = logging.getLogger(__name__)
        
        required_columns = [
            self.id_col,
            self.lon_col,
            self.lat_col,
            self.major_class_col,
            self.detail_class_col,
        ]
        
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            error_msg = f"❌ CSV 中缺少列: {missing_columns}"
            logger.error(error_msg)
            logger.error(f"   现有列: {list(self.df.columns)}")
            raise ValueError(error_msg)
        
        # 检查数据完整性
        null_counts = {
            col: self.df[col].isnull().sum()
            for col in required_columns
        }
        
        for col, null_count in null_counts.items():
            if null_count > 0:
                logger.warning(f"⚠️  列 '{col}' 包含 {null_count} 个空值")
        
        logger.info(f"✅ 所有必要列都存在")
    
    def _generate_labels_maps(self):
        """
        生成类别映射
        
        创建两个映射：
        1. detailed_labels_map: 详细类别 → 数字标签
        2. major_labels_map: 大类 → 数字标签
        """
        logger = logging.getLogger(__name__)
        
        # 生成详细类别映射
        detailed_categories = sorted(self.df[self.detail_class_col].unique())
        self.detailed_labels_map = {
            cat: idx for idx, cat in enumerate(detailed_categories)
        }
        self.inverse_detailed_map = {v: k for k, v in self.detailed_labels_map.items()}
        
        logger.info(f"📊 详细类别映射 ({len(self.detailed_labels_map)} 类):")
        for cat, idx in sorted(self.detailed_labels_map.items(), key=lambda x: x[1]):
            logger.info(f"   {idx}: {cat}")
        
        # 生成大类映射
        major_categories = sorted(self.df[self.major_class_col].unique())
        self.major_labels_map = {
            cat: idx for idx, cat in enumerate(major_categories)
        }
        self.inverse_major_map = {v: k for k, v in self.major_labels_map.items()}
        
        logger.info(f"📊 大类映射 ({len(self.major_labels_map)} 类):")
        for cat, idx in sorted(self.major_labels_map.items(), key=lambda x: x[1]):
            logger.info(f"   {idx}: {cat}")
        
        # 生成层级映射（大类 → 详细类别列表）
        self.hierarchical_map = {}
        for major_class in self.df[self.major_class_col].unique():
            mask = self.df[self.major_class_col] == major_class
            detail_classes = sorted(
                self.df[mask][self.detail_class_col].unique()
            )
            self.hierarchical_map[major_class] = {
                'major_id': self.major_labels_map[major_class],
                'detail_classes': {
                    det_cat: self.detailed_labels_map[det_cat]
                    for det_cat in detail_classes
                }
            }
        
        logger.info(f"📊 层级映射:")
        for major_class, info in sorted(self.hierarchical_map.items()):
            logger.info(f"   {info['major_id']}: {major_class}")
            for det_cat, det_id in sorted(info['detail_classes'].items(), key=lambda x: x[1]):
                logger.info(f"      └─ {det_id}: {det_cat}")
        
        # 添加标签列到数据框
        self.df['detail_label'] = self.df[self.detail_class_col].map(
            self.detailed_labels_map
        )
        self.df['major_label'] = self.df[self.major_class_col].map(
            self.major_labels_map
        )
        
        logger.info(f"✅ 类别映射生成完成")
    
    def _transform_crs(self):
        """
        进行空间投影转换
        
        功能：
        1. 自动检测 CSV 坐标系（如果配置为 'auto'）
        2. 将坐标转换为目标投影系统
        3. 支持任意坐标系（不仅限于 WGS84）
        """
        logger = logging.getLogger(__name__)
        
        # 步骤 1: 自动检测 CSV 坐标系
        csv_crs_config = self.config.get('data_specs.spatial.csv_crs', 'auto')
        
        if csv_crs_config == 'auto':
            # 自动检测
            from crs_manager import CRSManager
            crs_manager = CRSManager(self.config)
            detected_crs = crs_manager.auto_detect_csv_crs(
                self.csv_path,
                lon_col=self.lon_col,
                lat_col=self.lat_col
            )
            
            if detected_crs:
                csv_crs = detected_crs
                logger.info(f"✅ CSV 坐标系自动检测: {csv_crs}")
            else:
                # 回退到默认值
                csv_crs = 'EPSG:4326'
                logger.warning(f"⚠️  无法自动检测，使用默认坐标系: {csv_crs}")
        else:
            csv_crs = csv_crs_config
            logger.info(f"📍 使用配置的 CSV 坐标系: {csv_crs}")
        
        # 步骤 2: 创建 GeoDataFrame
        geometry = [
            Point(xy) for xy in zip(self.df[self.lon_col], self.df[self.lat_col])
        ]
        self.gdf = gpd.GeoDataFrame(
            self.df,
            geometry=geometry,
            crs=csv_crs
        )
        
        logger.info(f"✅ GeoDataFrame 创建完成 (初始投影: {csv_crs})")
        
        # 步骤 3: 转换到目标投影
        if csv_crs != self.target_crs:
            try:
                self.gdf = self.gdf.to_crs(self.target_crs)
                logger.info(f"✅ 投影转换完成: {csv_crs} → {self.target_crs}")
            except Exception as e:
                error_msg = f"❌ 投影转换失败: {e}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            logger.info(f"✅ CSV 坐标系与目标坐标系一致，无需转换")
        
        # 步骤 4: 提取转换后的坐标
        self.gdf['x'] = self.gdf.geometry.x
        self.gdf['y'] = self.gdf.geometry.y
        
        logger.info(f"✅ 空间坐标提取完成")
    
    def _save_maps(self):
        """
        保存映射到 JSON 文件
        
        生成以下文件：
        1. detailed_labels_map.json - 详细类别映射
        2. major_labels_map.json - 大类映射
        3. hierarchical_labels_map.json - 层级映射
        4. labels_summary.json - 汇总信息
        """
        logger = logging.getLogger(__name__)
        
        # 创建输出目录（如果不存在）
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存详细类别映射
        detailed_map_file = self.output_dir / 'detailed_labels_map.json'
        with open(detailed_map_file, 'w', encoding='utf-8') as f:
            json.dump(self.detailed_labels_map, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 详细类别映射已保存: {detailed_map_file}")
        
        # 保存大类映射
        major_map_file = self.output_dir / 'major_labels_map.json'
        with open(major_map_file, 'w', encoding='utf-8') as f:
            json.dump(self.major_labels_map, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 大类映射已保存: {major_map_file}")
        
        # 保存层级映射
        hierarchical_map_file = self.output_dir / 'hierarchical_labels_map.json'
        with open(hierarchical_map_file, 'w', encoding='utf-8') as f:
            json.dump(self.hierarchical_map, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 层级映射已保存: {hierarchical_map_file}")
        
        # 保存汇总信息
        summary = {
            'csv_file': str(self.csv_path),
            'total_samples': len(self.df),
            'detailed_classes': {
                'count': len(self.detailed_labels_map),
                'map': self.detailed_labels_map,
            },
            'major_classes': {
                'count': len(self.major_labels_map),
                'map': self.major_labels_map,
            },
            'target_crs': self.target_crs,
            'hierarchy': {
                major: {
                    'major_id': info['major_id'],
                    'detail_count': len(info['detail_classes']),
                    'details': info['detail_classes']
                }
                for major, info in self.hierarchical_map.items()
            }
        }
        
        summary_file = self.output_dir / 'labels_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 汇总信息已保存: {summary_file}")
        
        logger.info(f"✅ 所有映射文件已保存到 {self.output_dir}")
    
    # =========================================================================
    # 公共接口方法
    # =========================================================================
    
    def get_detailed_labels_map(self) -> Dict[str, int]:
        """
        获取详细类别映射
        
        Returns:
            Dict[str, int]: 详细类别 → 数字标签的映射
        
        Example:
            >>> encoder.get_detailed_labels_map()
            {'水体': 0, '建筑': 1, '农业': 2, ...}
        """
        return self.detailed_labels_map.copy()
    
    def get_major_labels_map(self) -> Dict[str, int]:
        """
        获取大类映射
        
        Returns:
            Dict[str, int]: 大类 → 数字标签的映射
        
        Example:
            >>> encoder.get_major_labels_map()
            {'水体': 0, '建筑': 1, '其他': 2}
        """
        return self.major_labels_map.copy()
    
    def get_hierarchical_map(self) -> Dict[str, Dict]:
        """
        获取层级映射
        
        Returns:
            Dict[str, Dict]: 大类 → 详细类别映射的层级结构
        
        Example:
            >>> encoder.get_hierarchical_map()
            {
                '水体': {
                    'major_id': 0,
                    'detail_classes': {'河流': 0, '湖泊': 1, '海洋': 2}
                },
                ...
            }
        """
        return self.hierarchical_map.copy()
    
    def get_geodataframe(self) -> gpd.GeoDataFrame:
        """
        获取 GeoDataFrame（包含坐标、标签等）
        
        Returns:
            gpd.GeoDataFrame: 包含几何和标签的地理数据框
        
        Columns:
            - geometry: 点几何
            - x, y: 转换后的坐标
            - detail_label: 详细类别标签（数字）
            - major_label: 大类标签（数字）
            - 其他原始列...
        """
        return self.gdf.copy()
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        获取数据框（不包含几何）
        
        Returns:
            pd.DataFrame: 包含标签但不包含几何的数据框
        """
        return self.df.copy()
    
    def label_to_category(self, label: int, label_type: str = 'detailed') -> str:
        """
        将数字标签转换回类别名称
        
        Args:
            label: 数字标签
            label_type: 标签类型 ('detailed' 或 'major')
        
        Returns:
            str: 类别名称
        
        Raises:
            ValueError: 标签不存在
        
        Example:
            >>> encoder.label_to_category(0, 'detailed')
            '水体'
        """
        if label_type == 'detailed':
            if label not in self.inverse_detailed_map:
                raise ValueError(f"详细标签 {label} 不存在")
            return self.inverse_detailed_map[label]
        elif label_type == 'major':
            if label not in self.inverse_major_map:
                raise ValueError(f"大类标签 {label} 不存在")
            return self.inverse_major_map[label]
        else:
            raise ValueError(f"未知标签类型: {label_type}")
    
    def category_to_label(self, category: str, category_type: str = 'detailed') -> int:
        """
        将类别名称转换为数字标签
        
        Args:
            category: 类别名称
            category_type: 类别类型 ('detailed' 或 'major')
        
        Returns:
            int: 数字标签
        
        Raises:
            ValueError: 类别不存在
        
        Example:
            >>> encoder.category_to_label('水体', 'detailed')
            0
        """
        if category_type == 'detailed':
            if category not in self.detailed_labels_map:
                raise ValueError(f"详细类别 '{category}' 不存在")
            return self.detailed_labels_map[category]
        elif category_type == 'major':
            if category not in self.major_labels_map:
                raise ValueError(f"大类 '{category}' 不存在")
            return self.major_labels_map[category]
        else:
            raise ValueError(f"未知类别类型: {category_type}")
    
    def get_sample_info(self, sample_id: int) -> Dict:
        """
        获取单个样本的信息
        
        Args:
            sample_id: 样本 ID
        
        Returns:
            Dict: 包含坐标、标签等信息的字典
        
        Raises:
            ValueError: 样本不存在
        
        Example:
            >>> encoder.get_sample_info(0)
            {
                'id': 0,
                'longitude': 120.5,
                'latitude': 35.2,
                'x': 621234.5,
                'y': 3896234.1,
                'major_class': '农业',
                'detail_class': '水稻',
                'major_label': 1,
                'detail_label': 5
            }
        """
        row = self.gdf[self.gdf[self.id_col] == sample_id]
        
        if len(row) == 0:
            raise ValueError(f"样本 ID {sample_id} 不存在")
        
        row = row.iloc[0]
        
        return {
            'id': sample_id,
            'longitude': row[self.lon_col],
            'latitude': row[self.lat_col],
            'x': row['x'],
            'y': row['y'],
            'major_class': row[self.major_class_col],
            'detail_class': row[self.detail_class_col],
            'major_label': int(row['major_label']),
            'detail_label': int(row['detail_label']),
        }
    
    def get_statistics(self) -> Dict:
        """
        获取数据集统计信息
        
        Returns:
            Dict: 包含样本数、类别分布等信息
        
        Example:
            >>> encoder.get_statistics()
            {
                'total_samples': 100,
                'detailed_class_distribution': {'水体': 20, '建筑': 30, ...},
                'major_class_distribution': {'水体': 20, '建筑': 80},
                'coordinates_stats': {
                    'x': {'min': 620000, 'max': 630000, 'mean': 625000},
                    'y': {'min': 3890000, 'max': 3900000, 'mean': 3895000}
                }
            }
        """
        detailed_dist = self.df[self.detail_class_col].value_counts().to_dict()
        major_dist = self.df[self.major_class_col].value_counts().to_dict()
        
        coords_stats = {
            'x': {
                'min': float(self.gdf['x'].min()),
                'max': float(self.gdf['x'].max()),
                'mean': float(self.gdf['x'].mean()),
                'std': float(self.gdf['x'].std()),
            },
            'y': {
                'min': float(self.gdf['y'].min()),
                'max': float(self.gdf['y'].max()),
                'mean': float(self.gdf['y'].mean()),
                'std': float(self.gdf['y'].std()),
            }
        }
        
        return {
            'total_samples': len(self.df),
            'detailed_class_distribution': detailed_dist,
            'major_class_distribution': major_dist,
            'coordinates_stats': coords_stats,
        }
    
    def save_geodataframe(self, filepath: Path = None, format: str = 'geojson'):
        """
        保存 GeoDataFrame 到文件
        
        Args:
            filepath: 输出文件路径。如果为 None，则保存到实验目录
            format: 输出格式 ('geojson', 'shapefile', 'geopackage')
        
        Example:
            >>> encoder.save_geodataframe(format='geojson')
            # 保存到 {output_dir}/labels_geodata.geojson
        """
        logger = logging.getLogger(__name__)
        
        if filepath is None:
            if format == 'geojson':
                filepath = self.output_dir / 'labels_geodata.geojson'
            elif format == 'shapefile':
                filepath = self.output_dir / 'labels_geodata.shp'
            elif format == 'geopackage':
                filepath = self.output_dir / 'labels_geodata.gpkg'
            else:
                raise ValueError(f"未知格式: {format}")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format == 'geojson':
                self.gdf.to_file(filepath, driver='GeoJSON', encoding='utf-8')
            elif format == 'shapefile':
                self.gdf.to_file(filepath, driver='ESRI Shapefile', encoding='utf-8')
            elif format == 'geopackage':
                self.gdf.to_file(filepath, driver='GPKG')
            
            logger.info(f"💾 GeoDataFrame 已保存: {filepath}")
        except Exception as e:
            error_msg = f"❌ 保存 GeoDataFrame 失败: {e}"
            logger.error(error_msg)
            raise IOError(error_msg)
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"LabelEncoder(\n"
            f"  csv_path={self.csv_path},\n"
            f"  output_dir={self.output_dir},\n"
            f"  target_crs={self.target_crs},\n"
            f"  total_samples={len(self.df)},\n"
            f"  detailed_classes={len(self.detailed_labels_map)},\n"
            f"  major_classes={len(self.major_labels_map)}\n"
            f")"
        )


# ============================================================================
# 使用示例和测试
# ============================================================================

if __name__ == "__main__":
    try:
        # 初始化配置管理器
        from config_manager import ConfigManager
        
        print("=" * 80)
        print("LabelEncoder 使用示例")
        print("=" * 80)
        
        config = ConfigManager('./config.yaml')
        
        # 初始化标签编码器
        print("\n1️⃣  初始化 LabelEncoder...")
        encoder = LabelEncoder(config=config)
        print(f"\n{encoder}\n")
        
        # 获取映射
        print("2️⃣  获取类别映射...")
        detailed_map = encoder.get_detailed_labels_map()
        major_map = encoder.get_major_labels_map()
        print(f"\n详细类别映射 ({len(detailed_map)} 类):")
        for cat, label in sorted(detailed_map.items(), key=lambda x: x[1]):
            print(f"   {label}: {cat}")
        
        print(f"\n大类映射 ({len(major_map)} 类):")
        for cat, label in sorted(major_map.items(), key=lambda x: x[1]):
            print(f"   {label}: {cat}")
        
        # 获取层级映射
        print("\n3️⃣  获取层级映射...")
        hierarchical_map = encoder.get_hierarchical_map()
        print(f"\n层级映射结构:")
        for major_class, info in sorted(hierarchical_map.items()):
            print(f"   {info['major_id']}: {major_class}")
            for det_cat, det_id in sorted(info['detail_classes'].items(), key=lambda x: x[1]):
                print(f"      └─ {det_id}: {det_cat}")
        
        # 获取 GeoDataFrame
        print("\n4️⃣  获取 GeoDataFrame...")
        gdf = encoder.get_geodataframe()
        print(f"\nGeoDataFrame 信息:")
        print(f"   行数: {len(gdf)}")
        print(f"   列数: {len(gdf.columns)}")
        print(f"   投影: {gdf.crs}")
        print(f"\n前 3 行:")
        print(gdf[['x', 'y', 'detail_label', 'major_label']].head(3))
        
        # 获取样本信息
        print("\n5️⃣  获取样本信息...")
        sample_info = encoder.get_sample_info(1)
        print(f"\n样本 1 信息:")
        for key, value in sample_info.items():
            print(f"   {key}: {value}")
        
        # 标签转换
        print("\n6️⃣  标签转换...")
        print(f"   标签 0 (详细) → {encoder.label_to_category(0, 'detailed')}")
        print(f"   '水体' (详细) → {encoder.category_to_label('水体', 'detailed')}")
        
        # 获取统计信息
        print("\n7️⃣  获取统计信息...")
        stats = encoder.get_statistics()
        print(f"\n样本统计:")
        print(f"   总样本数: {stats['total_samples']}")
        print(f"   详细类别分布: {stats['detailed_class_distribution']}")
        print(f"   大类分布: {stats['major_class_distribution']}")
        
        # 保存 GeoDataFrame
        print("\n8️⃣  保存 GeoDataFrame...")
        encoder.save_geodataframe(format='geojson')
        print(f"✅ GeoDataFrame 已保存")
        
        print("\n" + "=" * 80)
        print("✅ 所有示例完成!")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
