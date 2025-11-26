"""
RasterCrawler: 遥感影像文件爬虫模块

功能：
1. 递归扫描影像文件夹
2. 正则解析文件名提取时间信息
3. 懒加载提取空间元数据（边界框、投影）
4. 构建 R-树索引以加速空间查询
5. 快速查找点所在的影像文件
"""

import re
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Pattern
from dataclasses import dataclass, asdict
from rtree import index

import rasterio
from rasterio.io import MemoryFile
import numpy as np
import pandas as pd


@dataclass
class RasterMetadata:
    """栅格元数据"""
    filepath: Path
    filename: str
    bounds: Tuple[float, float, float, float]  # (left, bottom, right, top)
    crs: str
    width: int
    height: int
    resolution: Tuple[float, float]  # (x_res, y_res)
    date: Optional[datetime] = None
    year: Optional[int] = None
    month: Optional[int] = None
    extra_fields: Optional[Dict] = None
    
    def __post_init__(self):
        """数据验证"""
        if len(self.bounds) != 4:
            raise ValueError(f"bounds 必须是 4 元组，得到: {self.bounds}")
        if not isinstance(self.filepath, Path):
            self.filepath = Path(self.filepath)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        d = asdict(self)
        d['filepath'] = str(self.filepath)
        d['bounds'] = list(self.bounds)
        d['resolution'] = list(self.resolution)
        d['date'] = self.date.isoformat() if self.date else None
        return d
    
    def contains_point(self, x: float, y: float) -> bool:
        """检查点是否在栅格边界内"""
        left, bottom, right, top = self.bounds
        return left <= x <= right and bottom <= y <= top


class RasterCrawler:
    """
    遥感影像爬虫类
    
    功能：
    1. 递归扫描影像文件夹
    2. 使用正则表达式解析文件名
    3. 懒加载提取空间元数据
    4. 构建 R-树索引
    5. 快速空间查询
    
    使用示例：
        crawler = RasterCrawler(
            config=config,
            raster_dir='./data/raster/dynamic/',
            filename_pattern=r'S2_(?P<year>\d{4})_(?P<month>\d{2})_.*'
        )
        
        # 获取所有栅格
        rasters = crawler.get_all_rasters()
        
        # 查找包含点的栅格
        point_rasters = crawler.find_rasters_by_point(120.5, 35.2)
    """
    
    def __init__(
        self,
        config: 'ConfigManager',
        raster_dir: Optional[Path] = None,
        filename_pattern: Optional[str] = None,
        file_extensions: Tuple[str, ...] = ('.tif', '.tiff', '.jp2'),
        date_format: Optional[str] = None,
    ):
        """
        初始化 RasterCrawler
        
        Args:
            config: ConfigManager 对象
            raster_dir: 影像目录路径。如果为 None，则从 config 读取
            filename_pattern: 文件名正则表达式。包含命名组如 (?P<year>...), (?P<month>...)
            file_extensions: 要扫描的文件扩展名
            date_format: 日期格式字符串（如 '%Y-%m-%d'）
        
        Raises:
            FileNotFoundError: 影像目录不存在
            ValueError: 文件名正则表达式无效
        """
        self._setup_logging()
        logger = logging.getLogger(__name__)
        
        # 保存配置
        self.config = config
        self.raster_dir = Path(raster_dir) if raster_dir else config.get_resolved_path('dynamic_images_dir')
        self.output_dir = config.get_experiment_output_dir()
        self.file_extensions = file_extensions
        self.date_format = date_format
        
        logger.info(f"📂 影像目录: {self.raster_dir}")
        logger.info(f"📂 输出目录: {self.output_dir}")
        
        # 验证目录存在
        if not self.raster_dir.exists():
            error_msg = f"❌ 影像目录不存在: {self.raster_dir}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # 编译正则表达式
        if filename_pattern:
            try:
                self.filename_pattern = re.compile(filename_pattern)
                logger.info(f"✅ 正则表达式已编译: {filename_pattern}")
            except re.error as e:
                error_msg = f"❌ 正则表达式无效: {e}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            self.filename_pattern = None
            logger.info("⚠️  未指定文件名正则表达式，将不进行时间解析")
        
        # 初始化数据存储
        self.rasters_metadata: Dict[str, RasterMetadata] = {}  # filepath → metadata
        self.rtree_index: Optional[index.Index] = None
        self.raster_list: List[RasterMetadata] = []
        
        # 扫描并索引
        logger.info("🔍 开始扫描影像文件...")
        self._scan_rasters()
        logger.info(f"✅ 发现 {len(self.rasters_metadata)} 个影像文件")
        
        # 构建 R-树索引
        logger.info("🌳 开始构建 R-树索引...")
        self._build_rtree_index()
        logger.info(f"✅ R-树索引构建完成")
        
        # 保存元数据
        logger.info("💾 保存元数据...")
        self._save_metadata()
        logger.info("✅ 元数据已保存")
    
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
    
    def _scan_rasters(self):
        """
        递归扫描影像文件
        
        懒加载策略：仅读取文件元数据，不读取像素数据
        """
        logger = logging.getLogger(__name__)
        
        raster_files = []
        
        # 递归搜索
        for ext in self.file_extensions:
            raster_files.extend(self.raster_dir.rglob(f'*{ext}'))
        
        if not raster_files:
            logger.warning(f"⚠️  未找到影像文件")
            return
        
        logger.info(f"📁 找到 {len(raster_files)} 个影像文件")
        
        # 处理每个文件
        for filepath in raster_files:
            try:
                metadata = self._extract_metadata(filepath)
                self.rasters_metadata[str(filepath)] = metadata
                self.raster_list.append(metadata)
            except Exception as e:
                logger.warning(f"⚠️  跳过无效文件 {filepath}: {str(e)[:100]}")
    
    def _extract_metadata(self, filepath: Path) -> RasterMetadata:
        """
        提取单个栅格的元数据
        
        使用懒加载（Lazy Loading）策略，仅读取元数据，不读取像素数据
        
        Args:
            filepath: 栅格文件路径
        
        Returns:
            RasterMetadata: 栅格元数据对象
        """
        logger = logging.getLogger(__name__)
        
        # 使用 rasterio 读取元数据
        try:
            with rasterio.open(filepath) as src:
                bounds = src.bounds
                crs = src.crs
                width = src.width
                height = src.height
                transform = src.transform
                
                # 计算分辨率（从 transform 中提取）
                x_res = abs(transform.a)
                y_res = abs(transform.e)
                resolution = (x_res, y_res)
        except Exception as e:
            logger.warning(f"⚠️  无法读取栅格文件元数据: {filepath}")
            logger.warning(f"   错误: {e}")
            raise ValueError(f"无法读取栅格元数据: {e}") from e
        
        # 解析文件名
        filename = filepath.name
        date = None
        year = None
        month = None
        extra_fields = {}
        
        # 使用智能时间解析（支持可变长度前缀）
        try:
            from time_parser import extract_time_from_filename
            date, year, month, data_type = extract_time_from_filename(filename)
            if data_type:
                extra_fields['data_type'] = data_type
        except ImportError:
            logger.debug("⚠️  time_parser 模块不可用，使用备用解析方法")
            # 备用方法：使用正则表达式（如果配置了的话）
            if self.filename_pattern:
                match = self.filename_pattern.match(filename)
                if match:
                    groups = match.groupdict()
                    
                    # 提取年月信息
                    if 'year' in groups:
                        year = int(groups.pop('year'))
                    if 'month' in groups:
                        month = int(groups.pop('month'))
                    
                    # 构造日期对象
                    if year is not None:
                        try:
                            day = int(groups.pop('day', 1))
                            date = datetime(year, month or 1, day)
                        except (ValueError, TypeError):
                            date = None
                    
                    # 保存其他字段
                    extra_fields = {k: v for k, v in groups.items() if v is not None}
        except Exception as e:
            logger.debug(f"⚠️  时间解析失败: {e}")
        
        logger.debug(f"✓ {filename}: bounds={bounds}, crs={crs}, date={date}, year={year}, month={month}")
        
        return RasterMetadata(
            filepath=filepath,
            filename=filename,
            bounds=bounds,
            crs=str(crs) if crs else 'UNKNOWN',
            width=width,
            height=height,
            resolution=resolution,
            date=date,
            year=year,
            month=month,
            extra_fields=extra_fields if extra_fields else None
        )
    
    def _build_rtree_index(self):
        """
        构建 R-树索引
        
        用于加速空间查询。复杂度从 O(N) 降至 O(log N)。
        """
        logger = logging.getLogger(__name__)
        
        # 创建 R-树索引（使用 interleaved=True）
        # interleaved=True 表示坐标格式为 (minx, miny, maxx, maxy)
        # interleaved=False 需要 (minx, maxx, miny, maxy) 格式，容易出错
        self.rtree_index = index.Index(interleaved=True)
        
        # 为每个栅格添加边界框到索引
        valid_count = 0
        invalid_count = 0
        
        for idx, metadata in enumerate(self.raster_list):
            try:
                left, bottom, right, top = metadata.bounds
                
                # 验证边界框的有效性
                if left >= right or bottom >= top:
                    logger.warning(f"⚠️  无效的边界框: {metadata.filename}")
                    logger.warning(f"   bounds: ({left}, {bottom}, {right}, {top})")
                    invalid_count += 1
                    continue
                
                # R-树 insert 格式 (interleaved=True): (id, (minx, miny, maxx, maxy), object)
                self.rtree_index.insert(
                    valid_count,
                    (left, bottom, right, top),
                    obj=metadata
                )
                valid_count += 1
            except Exception as e:
                logger.warning(f"⚠️  无法添加栅格到索引: {metadata.filename}")
                logger.warning(f"   错误: {e}")
                invalid_count += 1
        
        logger.info(f"✅ R-树索引已构建 ({valid_count} 个条目)")
        if invalid_count > 0:
            logger.warning(f"⚠️  跳过了 {invalid_count} 个无效栅格")
    
    def _save_metadata(self):
        """
        保存栅格元数据到 JSON 文件
        
        生成以下文件：
        1. rasters_metadata.json - 所有栅格的详细元数据
        2. rasters_summary.json - 汇总统计信息
        """
        logger = logging.getLogger(__name__)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存详细元数据
        metadata_file = self.output_dir / 'rasters_metadata.json'
        metadata_list = [m.to_dict() for m in self.raster_list]
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"💾 栅格元数据已保存: {metadata_file}")
        
        # 保存汇总信息
        summary_file = self.output_dir / 'rasters_summary.json'
        
        # 统计时间覆盖范围
        dates = [m.date for m in self.raster_list if m.date]
        date_range = None
        if dates:
            dates_sorted = sorted(dates)
            date_range = {
                'start_date': dates_sorted[0].isoformat(),
                'end_date': dates_sorted[-1].isoformat(),
                'date_count': len(set([d.date() for d in dates]))
            }
        
        # 统计空间范围
        if self.raster_list:
            all_bounds = [m.bounds for m in self.raster_list]
            all_lefts = [b[0] for b in all_bounds]
            all_bottoms = [b[1] for b in all_bounds]
            all_rights = [b[2] for b in all_bounds]
            all_tops = [b[3] for b in all_bounds]
            
            spatial_range = {
                'min_x': min(all_lefts),
                'min_y': min(all_bottoms),
                'max_x': max(all_rights),
                'max_y': max(all_tops),
            }
        else:
            spatial_range = None
        
        # 统计投影
        crs_counts = {}
        for m in self.raster_list:
            crs_counts[m.crs] = crs_counts.get(m.crs, 0) + 1
        
        summary = {
            'total_rasters': len(self.raster_list),
            'raster_dir': str(self.raster_dir),
            'date_range': date_range,
            'spatial_range': spatial_range,
            'crs_distribution': crs_counts,
            'file_extensions': self.file_extensions,
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 汇总信息已保存: {summary_file}")
    
    # =========================================================================
    # 公共接口方法
    # =========================================================================
    
    def get_all_rasters(self) -> List[RasterMetadata]:
        """
        获取所有栅格元数据
        
        Returns:
            List[RasterMetadata]: 所有栅格的元数据列表
        """
        return [m for m in self.raster_list]
    
    def get_raster_count(self) -> int:
        """
        获取栅格总数
        
        Returns:
            int: 栅格数量
        """
        return len(self.raster_list)
    
    def find_rasters_by_point(
        self,
        x: float,
        y: float,
        return_count: bool = False
    ) -> List[RasterMetadata]:
        """
        使用 R-树索引快速查找包含指定点的栅格
        
        时间复杂度：O(log N)，其中 N 为栅格总数
        
        Args:
            x: 点的 X 坐标
            y: 点的 Y 坐标
            return_count: 是否返回计数而不是对象列表
        
        Returns:
            List[RasterMetadata]: 包含该点的栅格列表（已排序按时间）
        
        Example:
            >>> crawlers = crawler.find_rasters_by_point(120.5, 35.2)
            >>> print(f"找到 {len(rasters)} 个包含该点的栅格")
        """
        logger = logging.getLogger(__name__)
        
        if self.rtree_index is None:
            logger.error("❌ R-树索引未初始化")
            return []
        
        # 查询 R-树：找到边界框包含该点的所有栅格
        # 查询点为 (x, y, x, y) - 一个点的边界框
        hits = list(self.rtree_index.intersection((x, y, x, y), objects=True))
        
        # 获取命中的元数据
        rasters = []
        for hit in hits:
            metadata = hit.object
            # 精确检查：确认点确实在栅格内
            if metadata.contains_point(x, y):
                rasters.append(metadata)
        
        # 按日期排序
        rasters.sort(key=lambda m: m.date if m.date else datetime.min)
        
        if return_count:
            return len(rasters)
        
        logger.debug(f"✓ 点 ({x}, {y}) 包含在 {len(rasters)} 个栅格中")
        return rasters
    
    def find_rasters_by_bounds(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
    ) -> List[RasterMetadata]:
        """
        使用 R-树索引查找与指定边界框相交的栅格
        
        Args:
            min_x, min_y, max_x, max_y: 边界框坐标
        
        Returns:
            List[RasterMetadata]: 与边界框相交的栅格列表
        """
        logger = logging.getLogger(__name__)
        
        if self.rtree_index is None:
            logger.error("❌ R-树索引未初始化")
            return []
        
        # 查询 R-树
        hits = list(self.rtree_index.intersection(
            (min_x, min_y, max_x, max_y),
            objects=True
        ))
        
        rasters = [hit.object for hit in hits]
        rasters.sort(key=lambda m: m.date if m.date else datetime.min)
        
        logger.debug(f"✓ 边界框 ({min_x}, {min_y}, {max_x}, {max_y}) 包含 {len(rasters)} 个栅格")
        return rasters
    
    def find_rasters_by_date(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        year: Optional[int] = None,
        month: Optional[int] = None,
    ) -> List[RasterMetadata]:
        """
        按时间条件查找栅格
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            year: 特定年份
            month: 特定月份
        
        Returns:
            List[RasterMetadata]: 符合条件的栅格列表
        
        Example:
            >>> # 查找 2023 年的所有栅格
            >>> rasters = crawler.find_rasters_by_date(year=2023)
            >>> # 查找 2023 年 5 月的栅格
            >>> rasters = crawler.find_rasters_by_date(year=2023, month=5)
        """
        results = []
        
        for metadata in self.raster_list:
            # 检查年份
            if year is not None:
                if metadata.year != year:
                    continue
            
            # 检查月份
            if month is not None:
                if metadata.month != month:
                    continue
            
            # 检查日期范围
            if metadata.date:
                if start_date and metadata.date < start_date:
                    continue
                if end_date and metadata.date > end_date:
                    continue
            
            results.append(metadata)
        
        results.sort(key=lambda m: m.date if m.date else datetime.min)
        return results
    
    def find_rasters_by_filename_pattern(self, pattern: str) -> List[RasterMetadata]:
        """
        按文件名模式查找栅格
        
        Args:
            pattern: 文件名正则表达式模式
        
        Returns:
            List[RasterMetadata]: 符合条件的栅格列表
        """
        compiled_pattern = re.compile(pattern)
        results = [m for m in self.raster_list if compiled_pattern.match(m.filename)]
        return results
    
    def create_point_index(self, points_df: pd.DataFrame) -> pd.DataFrame:
        """
        为点数据集批量建立与栅格的关联
        
        这是 find_rasters_by_point 的向量化版本，适合处理大量点
        
        Args:
            points_df: 包含 'x' 和 'y' 列的 DataFrame
        
        Returns:
            pd.DataFrame: 原 DataFrame 加上 'raster_files' 列（列表）
        
        Example:
            >>> points_df['raster_files'] = crawler.create_point_index(points_df)
        """
        logger = logging.getLogger(__name__)
        
        logger.info(f"🔍 为 {len(points_df)} 个点建立栅格关联...")
        
        raster_files_list = []
        for _, row in points_df.iterrows():
            x, y = row['x'], row['y']
            rasters = self.find_rasters_by_point(x, y)
            raster_paths = [str(m.filepath) for m in rasters]
            raster_files_list.append(raster_paths)
        
        logger.info(f"✅ 点-栅格关联完成")
        
        return raster_files_list
    
    def get_time_series_for_point(
        self,
        x: float,
        y: float,
    ) -> List[RasterMetadata]:
        """
        获取某个点的时间序列栅格
        
        Returns:
            List[RasterMetadata]: 按时间排序的栅格列表
        """
        return self.find_rasters_by_point(x, y)
    
    def get_statistics(self) -> Dict:
        """
        获取爬虫统计信息
        
        Returns:
            Dict: 统计信息
        """
        if not self.raster_list:
            return {
                'total_rasters': 0,
                'message': '未找到栅格文件'
            }
        
        dates = [m.date for m in self.raster_list if m.date]
        all_bounds = [m.bounds for m in self.raster_list]
        
        all_lefts = [b[0] for b in all_bounds]
        all_bottoms = [b[1] for b in all_bounds]
        all_rights = [b[2] for b in all_bounds]
        all_tops = [b[3] for b in all_bounds]
        
        crs_counts = {}
        for m in self.raster_list:
            crs_counts[m.crs] = crs_counts.get(m.crs, 0) + 1
        
        year_counts = {}
        for m in self.raster_list:
            if m.year:
                year_counts[m.year] = year_counts.get(m.year, 0) + 1
        
        return {
            'total_rasters': len(self.raster_list),
            'time_coverage': {
                'date_range': (
                    min(dates).isoformat() if dates else None,
                    max(dates).isoformat() if dates else None
                ),
                'unique_dates': len(set([d.date() for d in dates])) if dates else 0,
                'year_distribution': year_counts,
            },
            'spatial_coverage': {
                'bounds': {
                    'min_x': min(all_lefts),
                    'min_y': min(all_bottoms),
                    'max_x': max(all_rights),
                    'max_y': max(all_tops),
                },
                'area': (max(all_rights) - min(all_lefts)) * (max(all_tops) - min(all_bottoms)),
            },
            'crs_distribution': crs_counts,
            'resolution_stats': {
                'min_x_res': min([m.resolution[0] for m in self.raster_list]),
                'max_x_res': max([m.resolution[0] for m in self.raster_list]),
                'min_y_res': min([m.resolution[1] for m in self.raster_list]),
                'max_y_res': max([m.resolution[1] for m in self.raster_list]),
            }
        }
    
    def detect_num_channels(self, sample_size: int = 5) -> Dict[str, int]:
        """
        检测影像的波段数
        
        功能：采样几个文件并检测其波段数，返回统计结果
        
        Args:
            sample_size: 采样的文件数量
        
        Returns:
            Dict: {
                'most_common': int,  # 最常见的波段数
                'all_channels': {num_channels: count, ...},  # 波段数的分布
                'files_checked': int,  # 检查的文件数
                'warning': str (如果波段数不一致)
            }
        
        Example:
            >>> crawler = RasterCrawler(config)
            >>> result = crawler.detect_num_channels()
            >>> print(f"最常见波段数: {result['most_common']}")
            最常见波段数: 1
        """
        import logging
        logger = logging.getLogger(__name__)
        
        if not self.raster_list:
            logger.warning("⚠️  无可用的栅格文件")
            return {
                'most_common': 0,
                'all_channels': {},
                'files_checked': 0,
                'warning': '没有栅格文件'
            }
        
        # 采样文件
        sample_files = self.raster_list[:min(sample_size, len(self.raster_list))]
        channel_counts = {}
        
        logger.info(f"🔍 检测波段数（采样 {len(sample_files)}/{len(self.raster_list)} 个文件）...")
        
        for metadata in sample_files:
            try:
                with rasterio.open(metadata.filepath) as src:
                    num_channels = src.count
                    channel_counts[num_channels] = channel_counts.get(num_channels, 0) + 1
                    logger.debug(f"   ✓ {metadata.filename}: {num_channels} 个波段")
            except Exception as e:
                logger.warning(f"   ⚠️  无法读取 {metadata.filename}: {e}")
                continue
        
        if not channel_counts:
            logger.error("❌ 无法检测任何文件的波段数")
            return {
                'most_common': 0,
                'all_channels': {},
                'files_checked': len(sample_files),
                'warning': '无法检测波段数'
            }
        
        # 找到最常见的波段数
        most_common = max(channel_counts, key=channel_counts.get)
        
        # 检查是否一致
        warning = None
        if len(channel_counts) > 1:
            warning = f"检测到不同的波段数: {dict(sorted(channel_counts.items()))}"
            logger.warning(f"⚠️  {warning}")
            logger.info(f"   将使用最常见的波段数: {most_common}")
        else:
            logger.info(f"✅ 所有采样文件都有 {most_common} 个波段")
        
        result = {
            'most_common': most_common,
            'all_channels': dict(sorted(channel_counts.items())),
            'files_checked': len(sample_files),
        }
        
        if warning:
            result['warning'] = warning
        
        return result
    
    def detect_crs(self) -> Dict:
        """
        检测所有影像的坐标参考系统
        
        功能：
        1. 扫描所有栅格的 CRS
        2. 检查是否一致
        3. 生成 CRS 检测报告
        
        Returns:
            Dict: {
                'is_consistent': bool,
                'detected_crs': {crs_code: count, ...},
                'most_common_crs': str,
                'crs_details': {filepath: crs_code},
                'warning': str (如果不一致),
                'recommendation': str
            }
        """
        logger = logging.getLogger(__name__)
        
        if not self.raster_list:
            logger.warning("⚠️  无可用的栅格文件")
            return {
                'is_consistent': True,
                'detected_crs': {},
                'most_common_crs': None,
                'crs_details': {},
                'warning': '没有栅格文件'
            }
        
        logger.info(f"\n🔍 检测坐标参考系统（CRS）...")
        logger.info(f"   扫描 {len(self.raster_list)} 个栅格文件...")
        
        crs_counts = {}
        crs_details = {}
        
        for metadata in self.raster_list:
            crs = metadata.crs
            crs_counts[crs] = crs_counts.get(crs, 0) + 1
            crs_details[str(metadata.filepath)] = crs
        
        # 判断一致性
        is_consistent = len(crs_counts) <= 1
        most_common_crs = max(crs_counts, key=crs_counts.get) if crs_counts else None
        
        # 生成警告和建议
        warning = None
        recommendation = None
        
        if len(crs_counts) > 1:
            warning = f"检测到 {len(crs_counts)} 个不同的坐标系"
            inconsistent_count = sum(count for crs, count in crs_counts.items() if crs != most_common_crs)
            recommendation = (
                f"将使用最常见的坐标系 {most_common_crs} "
                f"({crs_counts.get(most_common_crs, 0)} 个文件)。"
                f"其他 {inconsistent_count} 个文件将被标记为不一致。"
            )
            
            logger.warning(f"⚠️  {warning}")
            logger.info(f"✅ 坐标系分布:")
            for crs, count in sorted(crs_counts.items(), key=lambda x: -x[1]):
                logger.info(f"     - {crs}: {count} 个文件")
        
        elif is_consistent and most_common_crs:
            logger.info(f"✅ 所有文件使用相同的坐标系: {most_common_crs}")
        
        # 检查是否是常见投影
        if most_common_crs:
            from crs_manager import CRSManager
            crs_manager = CRSManager(self.config)
            crs_info = crs_manager.get_crs_info(most_common_crs)
            
            if crs_info:
                logger.info(f"   - 名称: {crs_info.crs_name}")
                logger.info(f"   - 类型: {'地理坐标' if crs_info.is_geographic else '投影坐标'}")
                logger.info(f"   - 单位: {crs_info.units}")
        
        result = {
            'is_consistent': is_consistent,
            'detected_crs': dict(sorted(crs_counts.items())),
            'most_common_crs': most_common_crs,
            'crs_details': crs_details,
        }
        
        if warning:
            result['warning'] = warning
        if recommendation:
            result['recommendation'] = recommendation
        
        return result
    
    def validate_crs_consistency(self, target_crs: Optional[str] = None) -> Dict:
        """
        验证 CRS 一致性并生成详细报告
        
        Args:
            target_crs: 期望的目标坐标系（可选）
        
        Returns:
            Dict: 验证结果
        """
        logger = logging.getLogger(__name__)
        
        crs_detection = self.detect_crs()
        
        validation_result = {
            'detection': crs_detection,
            'validation': {
                'is_valid': crs_detection['is_consistent'],
                'issues': [],
                'recommendations': []
            }
        }
        
        # 检查是否与目标 CRS 匹配
        if target_crs and crs_detection['most_common_crs']:
            if crs_detection['most_common_crs'] != target_crs:
                validation_result['validation']['is_valid'] = False
                issue = (
                    f"实际坐标系 ({crs_detection['most_common_crs']}) "
                    f"与目标坐标系 ({target_crs}) 不匹配"
                )
                validation_result['validation']['issues'].append(issue)
                logger.warning(f"⚠️  {issue}")
                
                # 建议自动重投影
                validation_result['validation']['recommendations'].append(
                    "可以配置 auto_reproject: true 来自动重投影文件"
                )
            else:
                logger.info(f"✅ 坐标系与目标一致: {target_crs}")
        
        return validation_result
    
    def save_crs_report(self, output_file: Optional[Path] = None) -> Path:
        """
        保存 CRS 检测报告到文件
        
        Args:
            output_file: 输出文件路径（如果为 None，则使用默认位置）
        
        Returns:
            Path: 保存的文件路径
        """
        logger = logging.getLogger(__name__)
        
        if output_file is None:
            output_file = self.output_dir / 'crs_detection_report.json'
        else:
            output_file = Path(output_file)
        
        # 生成报告
        report = {
            'timestamp': datetime.now().isoformat(),
            'raster_dir': str(self.raster_dir),
            'total_rasters': len(self.raster_list),
            'crs_detection': self.detect_crs(),
        }
        
        # 保存
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"💾 CRS 报告已保存: {output_file}")
        return output_file
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"RasterCrawler(\n"
            f"  raster_dir={self.raster_dir},\n"
            f"  total_rasters={len(self.raster_list)},\n"
            f"  output_dir={self.output_dir},\n"
            f"  rtree_indexed=True\n"
            f")"
        )


# ============================================================================
# 使用示例和测试
# ============================================================================

if __name__ == "__main__":
    try:
        from config_manager import ConfigManager
        
        print("=" * 80)
        print("RasterCrawler 使用示例")
        print("=" * 80)
        
        # 初始化配置 - 自动定位 config.yaml
        config_path = Path(__file__).parent / 'config.yaml'
        config = ConfigManager(str(config_path))
        
        # 定义文件名正则表达式
        # 例如文件名: GPP230101.tif → 提取 year=2023, month=01, day=01
        filename_pattern = r'GPP(?P<year>\d{2})(?P<month>\d{2})(?P<day>\d{2})'
        
        # 初始化爬虫
        print("\n1️⃣  初始化 RasterCrawler...")
        crawler = RasterCrawler(
            config=config,
            filename_pattern=filename_pattern
        )
        print(f"\n{crawler}\n")
        
        # 获取统计信息
        print("\n2️⃣  获取统计信息...")
        stats = crawler.get_statistics()
        print(f"✅ 栅格统计:")
        print(f"   总栅格数: {stats['total_rasters']}")
        if stats.get('time_coverage'):
            print(f"   时间覆盖: {stats['time_coverage']['date_range']}")
            print(f"   独立日期数: {stats['time_coverage']['unique_dates']}")
        if stats.get('spatial_coverage'):
            bounds = stats['spatial_coverage']['bounds']
            print(f"   空间范围: ({bounds['min_x']}, {bounds['min_y']}) - ({bounds['max_x']}, {bounds['max_y']})")
        
        # 查询示例
        print("\n3️⃣  栅格查询示例...")
        if stats['total_rasters'] > 0:
            # 获取所有栅格
            all_rasters = crawler.get_all_rasters()
            print(f"   总栅格: {len(all_rasters)}")
            print(f"   首个栅格: {all_rasters[0].filename}")
            print(f"   末个栅格: {all_rasters[-1].filename}")

            # 按日期查询
            rasters_2023 = crawler.find_rasters_by_date(year=2023)
            print(f"   2023 年栅格: {len(rasters_2023)}")
        
        print("\n" + "=" * 80)
        print("✅ 示例完成!")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
