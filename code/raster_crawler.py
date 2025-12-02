"""
raster_crawler.py: 遥感影像爬虫模块 (增强版)
功能：支持多源异构数据的智能解析与元数据管理
"""

import re
import json
import logging
import sys
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from rtree import index
import rasterio
from collections import defaultdict

@dataclass
class RasterMetadata:
    """栅格元数据 (保留以修复 ImportError)"""
    filepath: Path
    filename: str
    bounds: Tuple[float, float, float, float]
    crs: str
    width: int
    height: int
    date: Optional[date] = None
    variable: Optional[str] = None # 新增：变量名 (e.g., 'GPP', 'NDVI')
    is_monthly: bool = False       # 新增：是否为月度数据

    def contains_point(self, x: float, y: float) -> bool:
        left, bottom, right, top = self.bounds
        return left <= x <= right and bottom <= y <= top

    def to_dict(self):
        d = asdict(self)
        d['filepath'] = str(self.filepath)
        d['date'] = self.date.isoformat() if self.date else None
        return d

class RasterCrawler:
    def __init__(self, config, filename_pattern=None):
        self.config = config
        self.raster_dir = Path(config.get_resolved_path('dynamic_images_dir'))
        self.logger = logging.getLogger(__name__)
        
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
            d_obj = datetime.strptime(d_str, "%y%m%d").date()
            return f"SR_{b_str}", d_obj, False

        # 2. 月度数据 (e.g., PR2011 -> Var:PR, Date:2020-11)
        m_mon = re.match(r"(PR|SOIL)([0-9]{4})", name)
        if m_mon:
            var, d_str = m_mon.groups()
            d_obj = datetime.strptime(d_str, "%y%m").date()
            return var, d_obj, True

        # 3. 通用日/8天数据 (e.g., GPP230210 -> Var:GPP)
        m_daily = re.match(r"([A-Z]+)(\d{6})", name)
        if m_daily:
            var, d_str = m_daily.groups()
            d_obj = datetime.strptime(d_str, "%y%m%d").date()
            return var, d_obj, False
            
        return None, None, False

    def _scan_and_index(self):
        self.logger.info(f"🔍 扫描目录: {self.raster_dir}")
        tifs = sorted(list(self.raster_dir.rglob("*.tif")))
        
        for i, fpath in enumerate(tifs):
            try:
                # 解析变量信息
                var_name, d_obj, is_monthly = self._parse_variable_info(fpath.name)
                
                # 读取空间信息 (Lazy)
                with rasterio.open(fpath) as src:
                    bounds = src.bounds
                    crs = str(src.crs)
                    w, h = src.width, src.height
                
                meta = RasterMetadata(
                    filepath=fpath, filename=fpath.name, bounds=bounds,
                    crs=crs, width=w, height=h,
                    date=d_obj, variable=var_name, is_monthly=is_monthly
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
            if r.variable:
                all_vars.add(r.variable)
                if not r.is_monthly and r.date:
                    all_dates.add(r.date)
        
        return {
            'channel_map': {v: i for i, v in enumerate(sorted(list(all_vars)))},
            'timeline': sorted(list(all_dates))
        }