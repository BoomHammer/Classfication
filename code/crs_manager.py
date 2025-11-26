"""
CRS Manager: 坐标参考系统（CRS）统一管理模块

功能：
1. 自动检测 GeoTIFF 文件的坐标系
2. 验证多个数据源的坐标系一致性
3. 执行坐标系转换和重投影
4. 管理全局坐标系配置

支持的坐标系：
- EPSG:4326 (WGS84) - 经纬度坐标
- EPSG:3857 (Web Mercator)
- UTM 系列 (EPSG:32630-32660, 等)
- MODIS Sinusoidal (EPSG:6974)
- 其他所有 GDAL 支持的投影

使用示例：
    # 检测单个文件的坐标系
    crs_manager = CRSManager(config)
    file_crs = crs_manager.detect_tif_crs('path/to/file.tif')
    print(f"File CRS: {file_crs}")
    
    # 验证多个文件的坐标系一致性
    raster_files = ['file1.tif', 'file2.tif', 'file3.tif']
    crs_info = crs_manager.validate_multiple_crs(raster_files)
    print(crs_info['is_consistent'])
    print(crs_info['detected_crs'])
    
    # 转换坐标
    from_crs = 'EPSG:4326'
    to_crs = 'EPSG:3857'
    transformed_coords = crs_manager.transform_coordinates(
        [(120.5, 35.2), (121.0, 35.5)],
        from_crs, to_crs
    )
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import Counter
from dataclasses import dataclass

import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import Point

try:
    from pyproj import CRS, Transformer, exceptions
except ImportError:
    CRS = None
    Transformer = None
    exceptions = None


@dataclass
class CRSInfo:
    """坐标参考系统信息"""
    crs_code: str  # e.g., "EPSG:4326", "EPSG:3857"
    crs_name: str  # e.g., "WGS 84", "Web Mercator"
    is_geographic: bool  # True if geographic (lat/lon), False if projected
    units: str  # e.g., "metre", "degree"
    bounds: Optional[Dict] = None  # 有效范围
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'crs_code': self.crs_code,
            'crs_name': self.crs_name,
            'is_geographic': self.is_geographic,
            'units': self.units,
            'bounds': self.bounds,
        }


class CRSManager:
    """
    坐标参考系统管理类
    
    核心功能：
    1. 检测 GeoTIFF 文件的坐标系
    2. 验证坐标系一致性
    3. 执行坐标转换
    4. 提供坐标系信息和建议
    """
    
    # 常见坐标系映射
    COMMON_CRS = {
        'EPSG:4326': 'WGS 84 (Geographic, 经纬度)',
        'EPSG:3857': 'Web Mercator (Projected)',
        'EPSG:3395': 'World Mercator (Projected)',
        'EPSG:6974': 'MODIS Sinusoidal (Projected)',
        'EPSG:32630': 'UTM Zone 30N',
        'EPSG:32631': 'UTM Zone 31N',
        'EPSG:32632': 'UTM Zone 32N',
        'EPSG:32633': 'UTM Zone 33N',
    }
    
    def __init__(self, config: Optional['ConfigManager'] = None):
        """
        初始化 CRS 管理器
        
        Args:
            config: ConfigManager 对象（可选）
        """
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # 检查依赖
        if CRS is None or Transformer is None:
            self.logger.warning(
                "⚠️  pyproj 未安装，某些功能将不可用。"
                "请运行: pip install pyproj"
            )
    
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
    
    def detect_tif_crs(self, filepath: Union[str, Path]) -> Optional[str]:
        """
        检测 GeoTIFF 文件的坐标系
        
        Args:
            filepath: GeoTIFF 文件路径
        
        Returns:
            坐标系代码 (e.g., 'EPSG:4326') 或 None
        """
        filepath = Path(filepath)
        
        try:
            with rasterio.open(filepath) as src:
                crs = src.crs
                
                if crs is None:
                    self.logger.warning(
                        f"⚠️  文件未定义坐标系: {filepath.name}"
                    )
                    return None
                
                # rasterio 返回 CRS 对象，转换为字符串
                crs_str = str(crs)
                
                self.logger.debug(
                    f"✓ {filepath.name}: CRS = {crs_str}"
                )
                
                return crs_str
        
        except Exception as e:
            self.logger.error(
                f"❌ 无法读取文件坐标系: {filepath.name}"
            )
            self.logger.error(f"   错误: {e}")
            return None
    
    def validate_multiple_crs(
        self,
        filepaths: List[Union[str, Path]],
        verbose: bool = True
    ) -> Dict:
        """
        验证多个文件的坐标系一致性
        
        Args:
            filepaths: 文件路径列表
            verbose: 是否打印详细信息
        
        Returns:
            包含验证结果的字典：
            {
                'is_consistent': bool,
                'detected_crs': {crs_code: count},
                'most_common_crs': str,
                'crs_details': {filepath: crs_code},
                'inconsistent_files': [filepath],
            }
        """
        detected_crs = {}
        crs_details = {}
        
        for filepath in filepaths:
            crs = self.detect_tif_crs(filepath)
            if crs:
                detected_crs[crs] = detected_crs.get(crs, 0) + 1
                crs_details[str(filepath)] = crs
            else:
                crs_details[str(filepath)] = 'UNKNOWN'
        
        # 判断一致性
        is_consistent = len(detected_crs) <= 1
        most_common_crs = max(detected_crs, key=detected_crs.get) if detected_crs else None
        
        # 找出不一致的文件
        inconsistent_files = [
            str(fp) for fp, crs in crs_details.items()
            if crs != most_common_crs and crs != 'UNKNOWN'
        ]
        
        if verbose:
            self.logger.info(f"\n📊 坐标系一致性检查:")
            self.logger.info(f"  总文件数: {len(filepaths)}")
            self.logger.info(f"  检测到的坐标系数: {len(detected_crs)}")
            self.logger.info(f"  一致性: {'✅ 一致' if is_consistent else '⚠️  不一致'}")
            
            for crs, count in detected_crs.items():
                self.logger.info(f"    - {crs}: {count} 个文件")
            
            if inconsistent_files:
                self.logger.warning(f"\n⚠️  发现不一致的坐标系:")
                for fp in inconsistent_files[:5]:  # 只显示前 5 个
                    self.logger.warning(f"    - {Path(fp).name}: {crs_details[fp]}")
                if len(inconsistent_files) > 5:
                    self.logger.warning(f"    ... 等 {len(inconsistent_files) - 5} 个文件")
        
        return {
            'is_consistent': is_consistent,
            'detected_crs': detected_crs,
            'most_common_crs': most_common_crs,
            'crs_details': crs_details,
            'inconsistent_files': inconsistent_files,
        }
    
    def get_crs_info(self, crs_code: str) -> Optional[CRSInfo]:
        """
        获取坐标系的详细信息
        
        Args:
            crs_code: 坐标系代码 (e.g., 'EPSG:4326')
        
        Returns:
            CRSInfo 对象
        """
        if CRS is None:
            self.logger.warning("⚠️  pyproj 未安装，无法获取 CRS 详情")
            return None
        
        try:
            crs_obj = CRS.from_string(crs_code)
            
            info = CRSInfo(
                crs_code=crs_code,
                crs_name=crs_obj.name,
                is_geographic=crs_obj.is_geographic,
                units=str(crs_obj.axis_info[0].unit_name) if crs_obj.axis_info else 'unknown',
            )
            
            return info
        
        except Exception as e:
            self.logger.warning(f"⚠️  无法获取 CRS 信息: {crs_code}")
            self.logger.warning(f"   错误: {e}")
            return None
    
    def transform_coordinates(
        self,
        coords: List[Tuple[float, float]],
        from_crs: str,
        to_crs: str
    ) -> List[Tuple[float, float]]:
        """
        执行坐标转换
        
        Args:
            coords: 坐标列表 [(x1, y1), (x2, y2), ...]
            from_crs: 源坐标系 (e.g., 'EPSG:4326')
            to_crs: 目标坐标系 (e.g., 'EPSG:3857')
        
        Returns:
            转换后的坐标列表
        """
        if Transformer is None:
            self.logger.error("❌ pyproj 未安装，无法执行坐标转换")
            return coords
        
        if from_crs == to_crs:
            return coords
        
        try:
            transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
            transformed = [
                transformer.transform(x, y) for x, y in coords
            ]
            return transformed
        
        except Exception as e:
            self.logger.error(f"❌ 坐标转换失败: {e}")
            return coords
    
    def transform_geodataframe(
        self,
        gdf: gpd.GeoDataFrame,
        to_crs: str
    ) -> gpd.GeoDataFrame:
        """
        转换 GeoDataFrame 的坐标系
        
        Args:
            gdf: GeoDataFrame 对象
            to_crs: 目标坐标系
        
        Returns:
            转换后的 GeoDataFrame
        """
        try:
            if gdf.crs == to_crs:
                return gdf
            
            gdf_transformed = gdf.to_crs(to_crs)
            self.logger.info(
                f"✅ GeoDataFrame 坐标系转换: {gdf.crs} → {to_crs}"
            )
            return gdf_transformed
        
        except Exception as e:
            self.logger.error(f"❌ GeoDataFrame 转换失败: {e}")
            return gdf
    
    def auto_detect_csv_crs(
        self,
        csv_path: Union[str, Path],
        lon_col: str = 'X',
        lat_col: str = 'Y'
    ) -> Optional[str]:
        """
        自动检测 CSV 文件的坐标系
        
        基于坐标范围推断可能的坐标系
        
        Args:
            csv_path: CSV 文件路径
            lon_col: 经度列名
            lat_col: 纬度列名
        
        Returns:
            推荐的坐标系代码或 None
        """
        try:
            import pandas as pd
            
            df = pd.read_csv(csv_path, nrows=100)  # 只读前 100 行
            
            if lon_col not in df.columns or lat_col not in df.columns:
                self.logger.warning(
                    f"⚠️  CSV 中未找到坐标列: {lon_col}, {lat_col}"
                )
                return None
            
            lons = df[lon_col].dropna()
            lats = df[lat_col].dropna()
            
            lon_min, lon_max = lons.min(), lons.max()
            lat_min, lat_max = lats.min(), lats.max()
            
            self.logger.info(
                f"📊 CSV 坐标范围: "
                f"Lon [{lon_min:.2f}, {lon_max:.2f}], "
                f"Lat [{lat_min:.2f}, {lat_max:.2f}]"
            )
            
            # 判断坐标系
            # 经纬度范围通常在 [-180, 180] × [-90, 90]
            if (-180 <= lon_min and lon_max <= 180 and
                -90 <= lat_min and lat_max <= 90):
                
                self.logger.info(
                    "✅ 推断坐标系: EPSG:4326 (WGS84 地理坐标)"
                )
                return 'EPSG:4326'
            
            # Web Mercator 范围大约在 [-20037508, 20037508]
            elif (-20037508 <= lon_min and lon_max <= 20037508 and
                  -20037508 <= lat_min and lat_max <= 20037508):
                
                self.logger.info(
                    "✅ 推断坐标系: EPSG:3857 (Web Mercator 投影坐标)"
                )
                return 'EPSG:3857'
            
            # 其他投影坐标范围通常较小
            else:
                # 尝试基于范围大小推断
                if abs(lon_max - lon_min) > 1000 and abs(lat_max - lat_min) > 1000:
                    self.logger.warning(
                        "⚠️  无法推断坐标系，请手动指定"
                    )
                    return None
                else:
                    self.logger.info(
                        "✅ 推断坐标系: EPSG:4326 (WGS84 地理坐标，基于范围大小)"
                    )
                    return 'EPSG:4326'
        
        except Exception as e:
            self.logger.error(f"❌ CSV 坐标系检测失败: {e}")
            return None
    
    def suggest_compatible_crs(self, reference_crs: str) -> Dict[str, str]:
        """
        建议与参考坐标系兼容的其他坐标系
        
        Args:
            reference_crs: 参考坐标系
        
        Returns:
            推荐的坐标系列表
        """
        recommendations = {
            'EPSG:4326': {
                'EPSG:3857': 'Web Mercator (全球覆盖)',
                'EPSG:3395': 'World Mercator',
            },
            'EPSG:3857': {
                'EPSG:4326': 'WGS84 (经纬度)',
                'EPSG:3395': 'World Mercator',
            },
            'EPSG:6974': {
                'EPSG:4326': 'WGS84 (经纬度)',
                'EPSG:3857': 'Web Mercator',
            },
        }
        
        return recommendations.get(reference_crs, {})
    
    def save_crs_report(self, report_data: Dict, output_file: Union[str, Path]):
        """
        保存坐标系检测报告
        
        Args:
            report_data: 报告数据
            output_file: 输出文件路径
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"💾 坐标系报告已保存: {output_file}")
        
        except Exception as e:
            self.logger.error(f"❌ 保存坐标系报告失败: {e}")


def demo_crs_manager():
    """演示 CRS 管理功能"""
    print("\n" + "=" * 80)
    print("🎯 CRS Manager 演示")
    print("=" * 80)
    
    manager = CRSManager()
    
    # 示例 1: 获取坐标系信息
    print("\n📋 示例 1: 坐标系信息")
    for crs_code in ['EPSG:4326', 'EPSG:3857', 'EPSG:6974']:
        info = manager.get_crs_info(crs_code)
        if info:
            print(f"  {crs_code}: {info.crs_name}")
            print(f"    - 地理坐标系: {info.is_geographic}")
            print(f"    - 单位: {info.units}")
    
    # 示例 2: 坐标转换
    print("\n📋 示例 2: 坐标转换")
    coords = [(120.5, 35.2), (121.0, 35.5)]
    print(f"  源坐标 (EPSG:4326): {coords}")
    
    transformed = manager.transform_coordinates(
        coords, 'EPSG:4326', 'EPSG:3857'
    )
    print(f"  转换后 (EPSG:3857): {transformed}")
    
    # 示例 3: 兼容坐标系建议
    print("\n📋 示例 3: 兼容坐标系建议")
    print(f"  参考坐标系: EPSG:4326")
    suggestions = manager.suggest_compatible_crs('EPSG:4326')
    for crs, desc in suggestions.items():
        print(f"    - {crs}: {desc}")


if __name__ == '__main__':
    demo_crs_manager()
