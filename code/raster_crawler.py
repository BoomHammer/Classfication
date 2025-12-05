"""
raster_crawler.py: 遥感影像爬虫模块 (修复版)
功能：支持自定义目录扫描，兼容静态与动态数据解析
"""

import re
import json
import logging
import sys
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional, Union, Iterable
from dataclasses import dataclass, asdict
from rtree import index
import rasterio
from collections import defaultdict

@dataclass
class RasterMetadata:
    """栅格元数据"""
    filepath: Path
    filename: str
    bounds: Tuple[float, float, float, float]
    crs: str
    width: int
    height: int
    date: Optional[date] = None
    year: Optional[int] = None    # 显式增加年份字段
    month: Optional[int] = None   # 显式增加月份字段
    variable: Optional[str] = None 
    is_monthly: bool = False       

    def contains_point(self, x: float, y: float) -> bool:
        left, bottom, right, top = self.bounds
        return left <= x <= right and bottom <= y <= top

    def to_dict(self):
        d = asdict(self)
        d['filepath'] = str(self.filepath)
        d['date'] = self.date.isoformat() if self.date else None
        return d

class RasterCrawler:
    def __init__(self, config, raster_dir=None, filename_pattern=None, file_extensions=None):
        """
        Args:
            config: 配置对象
            raster_dir: (可选) 覆盖配置文件中的目录，用于静态数据扫描
            filename_pattern: (可选) 正则表达式模式
            file_extensions: (可选) 文件后缀列表，如 ['.tif', '.tiff']
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 1. 确定扫描目录
        if raster_dir:
            self.raster_dir = Path(raster_dir)
        else:
            self.raster_dir = Path(config.get_resolved_path('dynamic_images_dir'))
            
        # 2. 确定文件后缀
        if file_extensions:
            self.extensions = file_extensions
        else:
            # 默认只扫 tif
            self.extensions = ['.tif', '.tiff']

        # 结果存储
        self.raster_list: List[RasterMetadata] = []
        self.rtree = index.Index(interleaved=True)
        
        # 执行扫描
        self._scan_and_index()

    def _parse_variable_info(self, filename: str) -> Tuple[Optional[str], Optional[date], bool]:
        """
        核心逻辑：从文件名解析 (变量名, 日期, 是否月度)
        """
        name = Path(filename).stem
        
        # 1. 光谱反射率 SR (e.g., SR230117B4 -> Var:SR_B4)
        m_sr = re.match(r"SR(\d{6})(B\d+)", name)
        if m_sr:
            d_str, b_str = m_sr.groups()
            try:
                d_obj = datetime.strptime(d_str, "%y%m%d").date()
                return f"SR_{b_str}", d_obj, False
            except: pass

        # 2. 月度数据 (e.g., PR2011 -> Var:PR, Date:2020-11)
        m_mon = re.match(r"(PR|SOIL)([0-9]{4})", name)
        if m_mon:
            var, d_str = m_mon.groups()
            try:
                d_obj = datetime.strptime(d_str, "%y%m").date()
                return var, d_obj, True
            except: pass

        # 3. 通用日/8天数据 (e.g., GPP230210 -> Var:GPP)
        m_daily = re.match(r"([A-Z]+)(\d{6})", name)
        if m_daily:
            var, d_str = m_daily.groups()
            try:
                d_obj = datetime.strptime(d_str, "%y%m%d").date()
                return var, d_obj, False
            except: pass
            
        # 4. 无法解析 (静态数据通常走这里)
        return None, None, False

    def _scan_and_index(self):
        self.logger.info(f"🔍 扫描目录: {self.raster_dir}")
        if not self.raster_dir.exists():
            self.logger.warning(f"目录不存在: {self.raster_dir}")
            return

        files = []
        for ext in self.extensions:
            # 递归搜索
            files.extend(list(self.raster_dir.rglob(f"*{ext}")))
        
        files = sorted(list(set(files))) # 去重并排序
        
        for i, fpath in enumerate(files):
            try:
                # 解析变量信息
                var_name, d_obj, is_monthly = self._parse_variable_info(fpath.name)
                
                # 如果没解析出变量名 (比如静态数据 DEM.tif)，使用文件名作为变量名
                if var_name is None:
                    var_name = fpath.stem

                # 读取空间信息 (Lazy)
                with rasterio.open(fpath) as src:
                    bounds = src.bounds
                    crs = str(src.crs)
                    w, h = src.width, src.height
                
                meta = RasterMetadata(
                    filepath=fpath, 
                    filename=fpath.name, 
                    bounds=bounds,
                    crs=crs, 
                    width=w, 
                    height=h,
                    date=d_obj, 
                    year=d_obj.year if d_obj else None,
                    month=d_obj.month if d_obj else None,
                    variable=var_name, 
                    is_monthly=is_monthly
                )
                
                self.raster_list.append(meta)
                # 建立空间索引 (id, bounds, obj)
                self.rtree.insert(i, bounds, obj=meta)
                
            except Exception as e:
                self.logger.warning(f"跳过文件 {fpath.name}: {e}")

        self.logger.info(f"✅ 已索引 {len(self.raster_list)} 个影像文件")

    def find_rasters_by_point(self, x: float, y: float) -> List[RasterMetadata]:
        """空间查询"""
        hits = list(self.rtree.intersection((x, y, x, y), objects=True))
        results = [h.object for h in hits if h.object.contains_point(x, y)]
        return results

    def get_all_rasters(self) -> List[RasterMetadata]:
        return self.raster_list

    def get_super_channel_definition(self) -> Dict:
        """
        为 Dataset 生成超级通道定义
        """
        all_vars = set()
        all_dates = set()
        
        for r in self.raster_list:
            # 只有带日期的数据才算入动态时间轴
            if r.variable and r.date:
                all_vars.add(r.variable)
                if not r.is_monthly:
                    all_dates.add(r.date)
        
        return {
            'channel_map': {v: i for i, v in enumerate(sorted(list(all_vars)))},
            'timeline': sorted(list(all_dates))
        }

    def save_crs_report(self):
        """生成简易的 CRS 报告 (兼容 DataPreprocessor)"""
        pass