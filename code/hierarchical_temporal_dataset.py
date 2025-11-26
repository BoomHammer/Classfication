# ============================================================================
# 分层时间数据集 (Hierarchical Temporal Dataset)
# 
# 处理多种时间尺度的遥感数据：
# - 年度数据（标签）
# - 月度数据（中级特征）
# - 周度数据（细粒度特征）
#
# 关键概念：
# 1. 锚点数据集 (Anchor Dataset): 年度数据 - 定义空间采样点
# 2. 时间扩展: 从年份 (2020) 扩展为全年时间范围 (2020-01-01 ~ 2020-12-31)
# 3. 时间包含关系: 一年包含12个月，52周
# 4. 自定义采样: 不依赖 IntersectionDataset 的严格同时存在条件
# ============================================================================

import torch
from torch.utils.data import Dataset
from torchgeo.datasets import RasterDataset
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import copy


# ============================================================================
# Part 1: 自定义 RasterDataset 子类 - 处理不同时间分辨率
# ============================================================================

class YearlyCDL(RasterDataset):
    """
    年度遥感数据集（如 CDL - 作物数据层）
    
    文件命名: CDL_2020.tif, CDL_2021.tif, ...
    时间语义: 整个日历年 (2020-01-01 ~ 2020-12-31)
    """
    
    filename_glob = "CDL_*.tif"
    is_image = False  # 标签数据，不是影像
    
    def __init__(self, root: str, crs=None, res=None, **kwargs):
        """
        初始化年度数据集
        
        Args:
            root: 数据根目录
            crs: 坐标系
            res: 分辨率
        """
        super().__init__(root, crs=crs, res=res, **kwargs)
        # 父类会自动构建索引
    
    def _parse_date(self, path: str) -> datetime:
        """
        从文件名解析年份
        例如: CDL_2020.tif -> 2020-01-01 (年初)
        """
        try:
            # 提取文件名中的年份
            filename = path.split('/')[-1]  # CDL_2020.tif
            year = int(filename.split('_')[1].split('.')[0])
            return datetime(year, 1, 1)
        except:
            return datetime(1970, 1, 1)


class MonthlyNDVI(RasterDataset):
    """
    月度遥感数据集（如 NDVI）
    
    文件命名: NDVI_202001.tif, NDVI_202002.tif, ...
    时间语义: 特定月份 (2020-01-15 作为代表时间)
    """
    
    filename_glob = "NDVI_*.tif"
    is_image = True
    
    def __init__(self, root: str, crs=None, res=None, **kwargs):
        super().__init__(root, crs=crs, res=res, **kwargs)
    
    def _parse_date(self, path: str) -> datetime:
        """
        从文件名解析年月
        例如: NDVI_202001.tif -> 2020-01-15 (月中)
        """
        try:
            filename = path.split('/')[-1]
            date_str = filename.split('_')[1].split('.')[0]  # 202001
            year = int(date_str[:4])
            month = int(date_str[4:6])
            return datetime(year, month, 15)  # 用月中作为代表时间
        except:
            return datetime(1970, 1, 1)


class WeeklyLST(RasterDataset):
    """
    周度遥感数据集（如地表温度 LST）
    
    文件命名: LST_2020_W01.tif, LST_2020_W02.tif, ...
    时间语义: 特定周 (每周三作为代表时间)
    """
    
    filename_glob = "LST_*_W*.tif"
    is_image = True
    
    def __init__(self, root: str, crs=None, res=None, **kwargs):
        super().__init__(root, crs=crs, res=res, **kwargs)
    
    def _parse_date(self, path: str) -> datetime:
        """
        从文件名解析年份和周数
        例如: LST_2020_W01.tif -> 2020-01-08 (第1周的周三)
        
        ISO 8601 周数定义:
        - Week 1: 第一个包含周四的周
        - Week 01-53: 一年最多53周
        """
        try:
            filename = path.split('/')[-1]
            parts = filename.replace('_', '-').split('-')
            year = int(parts[1])
            week = int(parts[3].replace('W', '').replace('.tif', ''))
            
            # 计算该周的周三日期
            # ISO 8601: Week 1 的周一 = Jan 4 的周一
            jan4 = datetime(year, 1, 4)
            week_1_monday = jan4 - timedelta(days=jan4.weekday())
            target_date = week_1_monday + timedelta(weeks=week-1, days=2)  # +2 得到周三
            
            return target_date
        except:
            return datetime(1970, 1, 1)


# ============================================================================
# Part 2: 分层时间数据集 - 处理多尺度采样
# ============================================================================

class HierarchicalTemporalDataset(Dataset):
    """
    分层时间数据集
    
    架构:
    ┌─────────────────────────────────────────┐
    │  HierarchicalTemporalDataset            │
    │  (采样器，数据获取协调器)                │
    ├─────────────────────────────────────────┤
    │  Anchor: YearlyCDL (标签，定义位置)     │
    │  Monthly: MonthlyNDVI (中级特征)        │
    │  Weekly: WeeklyLST (细粒度特征)         │
    └─────────────────────────────────────────┘
    
    采样流程:
    1. 从年度数据中采样空间位置 (bbox) 和时间 (year=2020)
    2. 获取该位置的年度标签
    3. 将年份扩展为完整的时间范围 (2020-01-01 ~ 2020-12-31)
    4. 获取该位置在该时间范围内的所有月度数据
    5. 获取该位置在该时间范围内的所有周度数据
    6. 返回标签 + 月度特征序列 + 周度特征序列
    """
    
    def __init__(
        self,
        yearly_ds: RasterDataset,
        monthly_ds: Optional[RasterDataset] = None,
        weekly_ds: Optional[RasterDataset] = None,
        spatial_patch_size: int = 64,
        crs: str = "EPSG:4326",
    ):
        """
        初始化分层时间数据集
        
        Args:
            yearly_ds: 年度数据集（如 YearlyCDL）- 作为锚点数据集
            monthly_ds: 月度数据集（如 MonthlyNDVI）- 可选
            weekly_ds: 周度数据集（如 WeeklyLST）- 可选
            spatial_patch_size: 空间补丁大小（像素）
            crs: 坐标参考系统
        """
        self.yearly = yearly_ds
        self.monthly = monthly_ds
        self.weekly = weekly_ds
        self.patch_size = spatial_patch_size
        self.crs = crs
        
        # 提取年度数据中的所有年份
        self.available_years = self._extract_years_from_yearly_ds()
    
    def _extract_years_from_yearly_ds(self) -> List[int]:
        """从年度数据集的索引中提取所有可用年份"""
        years = set()
        for idx in self.yearly.index.data:
            # idx.time 应该是年初时间
            year = idx.time.year
            years.add(year)
        return sorted(list(years))
    
    def __len__(self) -> int:
        """数据集大小 = 年度数据的索引条数"""
        return len(self.yearly.index)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取样本
        
        Args:
            idx: 索引号
        
        Returns:
            字典包含:
                - 'label': 年度标签 (shape: [1, H, W])
                - 'monthly_features': 月度特征序列 (shape: [12, C, H, W]) 或 None
                - 'weekly_features': 周度特征序列 (shape: [52, C, H, W]) 或 None
                - 'metadata': 采样信息 (年份、位置等)
        """
        # Step 1: 获取年度数据的索引条目
        yearly_idx = self.yearly.index[idx]
        
        # 从索引中提取 bbox 和时间
        bounds = yearly_idx.bounds  # (left, bottom, right, top)
        year = yearly_idx.time.year
        
        # Step 2: 获取年度标签
        label_sample = self._get_yearly_label(bounds, year)
        
        # Step 3: 构造全年的时间范围查询
        year_start = datetime(year, 1, 1)
        year_end = datetime(year, 12, 31)
        
        # Step 4: 获取月度特征（如果存在）
        monthly_features = None
        if self.monthly is not None:
            monthly_features = self._get_monthly_features(bounds, year_start, year_end)
        
        # Step 5: 获取周度特征（如果存在）
        weekly_features = None
        if self.weekly is not None:
            weekly_features = self._get_weekly_features(bounds, year_start, year_end)
        
        # Step 6: 组织返回数据
        return {
            'label': label_sample,
            'monthly_features': monthly_features,
            'weekly_features': weekly_features,
            'metadata': {
                'year': year,
                'bounds': bounds,
                'patch_size': self.patch_size,
            }
        }
    
    def _get_yearly_label(self, bounds: Tuple[float, float, float, float], year: int) -> torch.Tensor:
        """
        从年度数据集获取标签
        
        Args:
            bounds: 空间范围 (left, bottom, right, top)
            year: 年份
        
        Returns:
            标签张量 (shape: [1, H, W])
        """
        # 创建查询对象
        # query 需要包含 bbox 和 time 信息
        from torchgeo.datasets.utils import BoundingBox
        
        query = BoundingBox(
            minx=bounds[0],
            miny=bounds[1],
            maxx=bounds[2],
            maxy=bounds[3],
            mint=datetime(year, 1, 1),
            maxt=datetime(year, 12, 31)
        )
        
        try:
            label_data = self.yearly[query]
            # 提取 'image' 或 'mask' 字段
            if isinstance(label_data, dict):
                if 'image' in label_data:
                    return label_data['image']
                elif 'mask' in label_data:
                    return label_data['mask']
            return label_data
        except Exception as e:
            print(f"Warning: Failed to get yearly label: {e}")
            return torch.zeros((1, self.patch_size, self.patch_size), dtype=torch.float32)
    
    def _get_monthly_features(
        self,
        bounds: Tuple[float, float, float, float],
        year_start: datetime,
        year_end: datetime
    ) -> Optional[torch.Tensor]:
        """
        从月度数据集获取全年的月度特征序列
        
        Args:
            bounds: 空间范围
            year_start: 年初日期
            year_end: 年末日期
        
        Returns:
            月度特征张量 (shape: [12, C, H, W]) 或 None
        """
        if self.monthly is None:
            return None
        
        from torchgeo.datasets.utils import BoundingBox
        
        monthly_list = []
        
        for month in range(1, 13):
            # 构造该月的时间范围
            month_date = datetime(year_start.year, month, 1)
            
            # 计算该月的最后一天
            if month == 12:
                month_end = datetime(year_start.year + 1, 1, 1) - timedelta(days=1)
            else:
                month_end = datetime(year_start.year, month + 1, 1) - timedelta(days=1)
            
            query = BoundingBox(
                minx=bounds[0],
                miny=bounds[1],
                maxx=bounds[2],
                maxy=bounds[3],
                mint=month_date,
                maxt=month_end
            )
            
            try:
                # 获取该月的数据
                monthly_data = self.monthly[query]
                
                if isinstance(monthly_data, dict) and 'image' in monthly_data:
                    monthly_list.append(monthly_data['image'])
                else:
                    # 如果获取失败，添加零张量
                    monthly_list.append(torch.zeros_like(monthly_list[0] if monthly_list else torch.zeros((1, self.patch_size, self.patch_size))))
            except Exception as e:
                print(f"Warning: Failed to get monthly data for {month}: {e}")
                # 添加零张量作为占位符
                if monthly_list:
                    monthly_list.append(torch.zeros_like(monthly_list[0]))
                else:
                    monthly_list.append(torch.zeros((1, self.patch_size, self.patch_size)))
        
        # 堆叠成 (12, C, H, W)
        if monthly_list:
            return torch.stack(monthly_list, dim=0)
        return None
    
    def _get_weekly_features(
        self,
        bounds: Tuple[float, float, float, float],
        year_start: datetime,
        year_end: datetime
    ) -> Optional[torch.Tensor]:
        """
        从周度数据集获取全年的周度特征序列
        
        Args:
            bounds: 空间范围
            year_start: 年初日期
            year_end: 年末日期
        
        Returns:
            周度特征张量 (shape: [52-53, C, H, W]) 或 None
        """
        if self.weekly is None:
            return None
        
        from torchgeo.datasets.utils import BoundingBox
        
        weekly_list = []
        year = year_start.year
        
        # 计算该年有多少周（通常是52-53周）
        dec31 = datetime(year, 12, 31)
        iso_year, iso_week, iso_day = dec31.isocalendar()
        num_weeks = iso_week if iso_year == year else 52
        
        for week in range(1, num_weeks + 1):
            # 计算该周的起止日期（周一-周日）
            jan4 = datetime(year, 1, 4)
            week_1_monday = jan4 - timedelta(days=jan4.weekday())
            week_start = week_1_monday + timedelta(weeks=week-1)
            week_end = week_start + timedelta(days=6)
            
            # 限制在该年范围内
            if week_end > year_end:
                week_end = year_end
            
            query = BoundingBox(
                minx=bounds[0],
                miny=bounds[1],
                maxx=bounds[2],
                maxy=bounds[3],
                mint=week_start,
                maxt=week_end
            )
            
            try:
                weekly_data = self.weekly[query]
                
                if isinstance(weekly_data, dict) and 'image' in weekly_data:
                    weekly_list.append(weekly_data['image'])
                else:
                    if weekly_list:
                        weekly_list.append(torch.zeros_like(weekly_list[0]))
                    else:
                        weekly_list.append(torch.zeros((1, self.patch_size, self.patch_size)))
            except Exception as e:
                print(f"Warning: Failed to get weekly data for week {week}: {e}")
                if weekly_list:
                    weekly_list.append(torch.zeros_like(weekly_list[0]))
                else:
                    weekly_list.append(torch.zeros((1, self.patch_size, self.patch_size)))
        
        # 堆叠成 (num_weeks, C, H, W)
        if weekly_list:
            return torch.stack(weekly_list, dim=0)
        return None


# ============================================================================
# Part 3: 时间范围查询工具
# ============================================================================

class TimeRange:
    """
    时间范围表示
    
    用途: 用于多数据源采样时，表达"从 start_time 到 end_time"的范围
    
    示例:
        range = TimeRange("2020-01-01", "2020-12-31")
        # 表示整个 2020 年
    """
    
    def __init__(self, start_str: str, end_str: str):
        """
        初始化时间范围
        
        Args:
            start_str: 开始时间字符串，格式 "YYYY-MM-DD"
            end_str: 结束时间字符串，格式 "YYYY-MM-DD"
        """
        self.start = datetime.strptime(start_str, "%Y-%m-%d")
        self.end = datetime.strptime(end_str, "%Y-%m-%d")
    
    def contains(self, dt: datetime) -> bool:
        """检查时间点是否在范围内"""
        return self.start <= dt <= self.end
    
    def __repr__(self):
        return f"TimeRange({self.start.date()}, {self.end.date()})"


# ============================================================================
# Part 4: 使用示例
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("分层时间数据集示例")
    print("=" * 70)
    
    # 假设数据目录结构
    data_root = "data/raster"
    yearly_root = f"{data_root}/yearly"      # CDL_2020.tif, CDL_2021.tif
    monthly_root = f"{data_root}/monthly"    # NDVI_202001.tif, NDVI_202002.tif, ...
    weekly_root = f"{data_root}/weekly"      # LST_2020_W01.tif, LST_2020_W02.tif, ...
    
    print("\n1️⃣  初始化单个数据集")
    print("-" * 70)
    
    try:
        yearly_ds = YearlyCDL(yearly_root)
        print(f"✓ 年度数据集: {len(yearly_ds)} 个样本")
        print(f"  索引大小: {len(yearly_ds.index)}")
    except Exception as e:
        print(f"⚠ 年度数据集初始化: {e}")
        yearly_ds = None
    
    try:
        monthly_ds = MonthlyNDVI(monthly_root)
        print(f"✓ 月度数据集: {len(monthly_ds)} 个样本")
    except Exception as e:
        print(f"⚠ 月度数据集初始化: {e}")
        monthly_ds = None
    
    try:
        weekly_ds = WeeklyLST(weekly_root)
        print(f"✓ 周度数据集: {len(weekly_ds)} 个样本")
    except Exception as e:
        print(f"⚠ 周度数据集初始化: {e}")
        weekly_ds = None
    
    print("\n2️⃣  创建分层时间数据集")
    print("-" * 70)
    
    if yearly_ds is not None:
        hierarchical_ds = HierarchicalTemporalDataset(
            yearly_ds=yearly_ds,
            monthly_ds=monthly_ds,
            weekly_ds=weekly_ds,
            spatial_patch_size=64
        )
        print(f"✓ 分层数据集创建成功")
        print(f"  数据集大小: {len(hierarchical_ds)} 个样本")
        print(f"  可用年份: {hierarchical_ds.available_years}")
        
        print("\n3️⃣  采样示例")
        print("-" * 70)
        
        try:
            sample = hierarchical_ds[0]
            print(f"✓ 成功获取样本 0")
            print(f"\n  标签 (年度):")
            print(f"    形状: {sample['label'].shape}")
            print(f"    数据类型: {sample['label'].dtype}")
            
            if sample['monthly_features'] is not None:
                print(f"\n  月度特征:")
                print(f"    形状: {sample['monthly_features'].shape}")
                print(f"    说明: (12个月, 通道数, 高, 宽)")
            
            if sample['weekly_features'] is not None:
                print(f"\n  周度特征:")
                print(f"    形状: {sample['weekly_features'].shape}")
                print(f"    说明: (周数, 通道数, 高, 宽)")
            
            print(f"\n  元数据:")
            for key, value in sample['metadata'].items():
                print(f"    {key}: {value}")
        
        except Exception as e:
            print(f"✗ 采样失败: {e}")
    
    print("\n4️⃣  架构说明")
    print("-" * 70)
    print("""
    分层时间数据集架构:
    
    ┌──────────────────────────────────────────────────┐
    │         HierarchicalTemporalDataset              │
    │  (协调多个时间尺度的数据集)                      │
    └──────────────────────────────────────────────────┘
              ↓
    采样流程:
    
    1️⃣  从年度数据 (YearlyCDL) 采样:
        - 空间: bbox (左, 下, 右, 上)
        - 时间: 年份 (2020)
        → 结果: 标签 (1, H, W)
    
    2️⃣  时间扩展:
        - 输入: 年份 = 2020
        - 输出: 时间范围 = 2020-01-01 ~ 2020-12-31
    
    3️⃣  获取月度数据 (MonthlyNDVI):
        - 对每个月份构造查询
        - 查询条件: bbox + 月份时间范围
        → 结果: 12 个月的序列 (12, C, H, W)
    
    4️⃣  获取周度数据 (WeeklyLST):
        - 对每周构造查询
        - 查询条件: bbox + 周时间范围
        → 结果: 52-53 周的序列 (52-53, C, H, W)
    
    5️⃣  返回组合样本:
        {
            'label': (1, H, W),           # 年度标签
            'monthly_features': (12, C, H, W),  # 月度序列
            'weekly_features': (52-53, C, H, W), # 周度序列
            'metadata': {...}              # 采样信息
        }
    """)
    
    print("\n5️⃣  关键创新点")
    print("-" * 70)
    print("""
    1. 锚点机制:
       - 年度数据作为"锚"定义空间采样点
       - 月度和周度数据通过共同的 bbox 关联
    
    2. 时间分层:
       - 年份 → 完整年份范围 (2020 → Jan1-Dec31)
       - 月份 → 月初到月末 (Month 1 → Jan1-Jan31)
       - 周数 → 周一到周日
    
    3. 动态查询:
       - 使用 BoundingBox(minx, miny, maxx, maxy, mint, maxt)
       - 支持复杂的时空查询
    
    4. 容错机制:
       - 缺失数据用零张量填充
       - 不会因某个月/周缺失而中断
    
    5. 灵活组合:
       - 可选择只用年度+月度 (不要周度)
       - 可选择只用年度+周度 (不要月度)
       - 完全模块化
    """)
    
    print("\n✅ 分层时间数据集演示完成！\n")
