"""
StatsCalculator: 流式统计计算器模块

功能：
1. 实现 Welford 增量算法，支持流式处理
2. 随机采样数据集的 5%-10%
3. 分批读取影像块进行统计
4. 分别计算动态和静态影像的均值和方差
5. 保存归一化参数到 JSON 文件

理论基础：Welford 在线算法
- 优点：数值稳定性高，内存占用极低
- 允许通过流式处理每个批次的数据来逐步更新均值和方差
- 不需要一次性加载所有数据

算法原理：
对于流式数据，维护以下变量：
- count: 处理的样本数量
- mean: 当前均值
- M2: 平方差聚合量（用于计算方差）

每处理一个新值 x：
  delta = x - mean
  mean = mean + delta / count
  delta2 = x - mean
  M2 = M2 + delta * delta2

最终得到方差: variance = M2 / count
"""

import json
import logging
import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
import rasterio
from tqdm import tqdm

from config_manager import ConfigManager
from raster_crawler import RasterCrawler, RasterMetadata


@dataclass
class ChannelStats:
    """单个通道的统计数据"""
    channel_name: str
    count: int = 0
    mean: float = 0.0
    M2: float = 0.0  # 平方差聚合量
    
    @property
    def variance(self) -> float:
        """计算方差"""
        if self.count < 2:
            return 0.0
        return self.M2 / self.count
    
    @property
    def std(self) -> float:
        """计算标准差"""
        return np.sqrt(self.variance)
    
    def update(self, value: float):
        """
        使用 Welford 算法更新统计量
        
        Args:
            value: 新的数据点
        """
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.M2 += delta * delta2
    
    def update_batch(self, values: np.ndarray):
        """
        批量更新统计量
        
        Args:
            values: 一维数组的数据点
        """
        for value in values.flat:
            self.update(float(value))
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'channel_name': self.channel_name,
            'count': self.count,
            'mean': float(self.mean),
            'std': float(self.std),
            'variance': float(self.variance),
        }


@dataclass
class ImageTypeStats:
    """单类影像（动态/静态）的统计数据"""
    image_type: str  # 'dynamic' 或 'static'
    channels: Dict[int, ChannelStats] = None  # channel_id -> ChannelStats
    num_channels: int = 0
    total_samples: int = 0
    
    def __post_init__(self):
        """初始化后处理"""
        if self.channels is None:
            self.channels = {}
    
    def initialize_channels(self, num_channels: int, channel_names: Optional[List[str]] = None):
        """初始化通道"""
        self.num_channels = num_channels
        self.channels = {}
        
        for i in range(num_channels):
            channel_name = channel_names[i] if channel_names and i < len(channel_names) else f"Band_{i}"
            self.channels[i] = ChannelStats(channel_name=channel_name)
    
    def update(self, data: np.ndarray):
        """
        更新统计量
        
        Args:
            data: 形状为 (num_channels, height, width) 的数据
        """
        if data.ndim != 3:
            raise ValueError(f"期望 3D 数据 (channels, height, width)，得到 {data.ndim}D")
        
        num_channels = data.shape[0]
        if num_channels != self.num_channels:
            raise ValueError(
                f"通道数不匹配：期望 {self.num_channels}，得到 {num_channels}"
            )
        
        # 获取单个通道的像素数（H × W）
        pixels_per_channel = data[0, :, :].size
        
        # 更新每个通道的统计量
        for ch in range(num_channels):
            channel_data = data[ch, :, :]
            self.channels[ch].update_batch(channel_data)
        
        # 只计算一次 total_samples
        # (而不是每个通道都加，这样避免了 channels 倍数的问题)
        self.total_samples += pixels_per_channel
    
    def get_means(self) -> List[float]:
        """获取所有通道的均值列表"""
        return [self.channels[i].mean for i in range(self.num_channels)]
    
    def get_stds(self) -> List[float]:
        """获取所有通道的标准差列表"""
        return [self.channels[i].std for i in range(self.num_channels)]
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'image_type': self.image_type,
            'num_channels': self.num_channels,
            'total_samples': self.total_samples,
            'channels': {
                i: self.channels[i].to_dict() for i in range(self.num_channels)
            },
            'mean': self.get_means(),
            'std': self.get_stds(),
        }


class StatsCalculator:
    """
    流式统计计算器
    
    功能：
    1. 随机采样数据集的 5%-10%
    2. 使用 Welford 增量算法计算统计量
    3. 分别处理动态和静态影像
    4. 保存归一化参数到 JSON 文件
    
    使用示例：
        calculator = StatsCalculator(config)
        calculator.compute_global_stats(
            dynamic_rasters=dynamic_metadata_list,
            static_rasters=static_metadata_list,
            sampling_rate=0.1  # 采样 10%
        )
        calculator.save_stats()
    """
    
    def __init__(
        self,
        config: ConfigManager,
        dynamic_channel_names: Optional[List[str]] = None,
        static_channel_names: Optional[List[str]] = None,
    ):
        """
        初始化统计计算器
        
        Args:
            config: ConfigManager 对象
            dynamic_channel_names: 动态影像通道名称（可选，如果为None将自动检测）
            static_channel_names: 静态影像通道名称（可选，如果为None将自动检测）
        """
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        self.config = config
        self.output_dir = config.get_experiment_output_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 通道名称（不设置默认值，由实际数据决定）
        self.dynamic_channel_names = dynamic_channel_names
        self.static_channel_names = static_channel_names
        
        # 统计数据
        self.dynamic_stats: Optional[ImageTypeStats] = None
        self.static_stats: Optional[ImageTypeStats] = None
        
        self.logger.info(f"📊 统计计算器已初始化")
        self.logger.info(f"   ℹ️  通道名称将从实际数据自动检测")
    
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
    
    def _read_raster_data(self, filepath: Path) -> Tuple[np.ndarray, int]:
        """
        读取栅格数据
        
        Args:
            filepath: 栅格文件路径
        
        Returns:
            Tuple[np.ndarray, int]: (数据数组, 通道数)
        
        Raises:
            Exception: 如果无法读取文件
        """
        try:
            with rasterio.open(filepath) as src:
                # 读取所有波段
                data = src.read()  # 形状: (num_bands, height, width)
                num_channels = src.count
                return data, num_channels
        except Exception as e:
            self.logger.error(f"❌ 无法读取文件 {filepath}: {e}")
            raise
    
    def compute_global_stats(
        self,
        dynamic_rasters: Optional[List[RasterMetadata]] = None,
        static_rasters: Optional[List[RasterMetadata]] = None,
        sampling_rate: float = 0.1,
        chunk_size: int = 1000,
    ):
        """
        计算全局统计量
        
        Args:
            dynamic_rasters: 动态影像元数据列表
            static_rasters: 静态影像元数据列表
            sampling_rate: 采样率 (0.0-1.0)
            chunk_size: 每个批次的像素样本数
        
        Raises:
            ValueError: 如果采样率无效或输入为空
        """
        if not (0.0 < sampling_rate <= 1.0):
            raise ValueError(f"采样率必须在 (0, 1] 范围内，得到 {sampling_rate}")
        
        print("=" * 80)
        print("[INFO] 开始计算全局统计量")
        print(f"       采样率: {sampling_rate * 100:.1f}%")
        print("=" * 80)
        
        # 处理动态影像
        if dynamic_rasters:
            print(f"\n🌐 处理动态影像...")
            print(f"   总文件数: {len(dynamic_rasters)}")
            self._compute_stats_for_type(
                rasters=dynamic_rasters,
                image_type='dynamic',
                sampling_rate=sampling_rate,
                num_channels=len(self.dynamic_channel_names) if self.dynamic_channel_names else None,
                channel_names=self.dynamic_channel_names,
            )
        
        # 处理静态影像
        if static_rasters:
            print(f"\n🌐 处理静态影像...")
            print(f"   总文件数: {len(static_rasters)}")
            self._compute_stats_for_type(
                rasters=static_rasters,
                image_type='static',
                sampling_rate=sampling_rate,
                num_channels=len(self.static_channel_names) if self.static_channel_names else None,
                channel_names=self.static_channel_names,
            )
        
        print("\n" + "=" * 80)
        print("[CHECK] ✅ 全局统计量计算完成")
        print("=" * 80 + "\n")
    
    def _compute_stats_for_type(
        self,
        rasters: List[RasterMetadata],
        image_type: str,
        sampling_rate: float,
        num_channels: int,
        channel_names: List[str],
    ):
        """
        计算特定类型影像的统计量
        
        Args:
            rasters: 栅格元数据列表
            image_type: 影像类型 ('dynamic' 或 'static')
            sampling_rate: 采样率
            num_channels: 期望通道数 (用于校验)
            channel_names: 通道名称列表
        """
        # 随机采样文件
        sample_count = max(1, int(len(rasters) * sampling_rate))
        sampled_rasters = random.sample(rasters, sample_count)
        
        print(f"   📊 采样 {sample_count}/{len(rasters)} 个文件 (采样率 {sampling_rate*100:.1f}%)")
        
        # 从第一个有效文件检测实际通道数
        detected_channels = None
        for metadata in sampled_rasters:
            try:
                data, file_num_channels = self._read_raster_data(metadata.filepath)
                detected_channels = file_num_channels
                print(f"   ✓ 检测到通道数: {detected_channels} (从 {metadata.filename})")
                break
            except Exception as e:
                print(f"   ⚠️  无法读取 {metadata.filename}: {e}")
                continue
        
        if detected_channels is None:
            print(f"   ❌ 无法检测 {image_type} 影像的通道数")
            return
        
        # 初始化统计对象（使用检测到的通道数）
        stats = ImageTypeStats(image_type=image_type)
        # 不使用预设的通道名称，让 initialize_channels 生成通用的 Band_0, Band_1 等名称
        stats.initialize_channels(detected_channels, channel_names=None)
        
        # 处理每个采样的栅格
        pbar = tqdm(
            sampled_rasters,
            desc=f"处理进度",
            unit="文件",
            ncols=80,
            position=0,
            leave=True
        )
        
        for metadata in pbar:
            try:
                # 读取数据
                data, file_num_channels = self._read_raster_data(metadata.filepath)
                
                # 如果通道数不匹配，尝试调整
                if file_num_channels != detected_channels:
                    print(f"   ⚠️  {metadata.filename} 有 {file_num_channels} 个通道，预期 {detected_channels} 个，跳过")
                    continue
                
                # 更新统计量
                stats.update(data)
                
                pbar.update(1)
                pbar.set_description(
                    f"处理进度 ({stats.total_samples:,} 样本处理)"
                )
                
            except Exception as e:
                print(f"   ⚠️  处理文件失败 {metadata.filename}: {str(e)[:100]}")
                continue
        
        pbar.close()
        
        # 保存统计结果
        if image_type == 'dynamic':
            self.dynamic_stats = stats
        else:
            self.static_stats = stats
        
        # 输出统计结果
        self._print_stats(stats)
    
    def _print_stats(self, stats: ImageTypeStats):
        """
        格式化输出统计结果
        
        Args:
            stats: 影像统计数据
        """
        print(f"\n{'=' * 80}")
        print(
            f"📈 {stats.image_type.upper()} 影像统计量 "
            f"({stats.num_channels} 波段: {', '.join(stats.channels[i].channel_name for i in range(stats.num_channels))}):"
        )
        print(f"{'=' * 80}")
        
        # 均值
        means = stats.get_means()
        print(f"\n🔹 Mean:")
        means_str = ", ".join(f"{m:.6f}" for m in means)
        print(f"   [{means_str}]")
        
        # 标准差
        stds = stats.get_stds()
        print(f"\n🔹 Std:")
        stds_str = ", ".join(f"{s:.6f}" for s in stds)
        print(f"   [{stds_str}]")
        
        # 详细的通道统计
        print(f"\n🔹 通道详情:")
        for ch_id in range(stats.num_channels):
            ch_stats = stats.channels[ch_id]
            print(f"   {ch_stats.channel_name}: mean={ch_stats.mean:.6f}, std={ch_stats.std:.6f}")
        
        # 样本统计
        print(f"\n🔹 总样本数: {stats.total_samples:,}")
        print(f"{'=' * 80}\n")
    
    def save_stats(self, filename: str = 'normalization_stats.json'):
        """
        保存统计量到 JSON 文件
        
        Args:
            filename: 输出文件名
        
        Raises:
            ValueError: 如果未计算统计量
        """
        if self.dynamic_stats is None and self.static_stats is None:
            raise ValueError("❌ 未计算任何统计量，请先调用 compute_global_stats")
        
        # 构建输出数据
        output_data = {}
        
        if self.dynamic_stats:
            output_data['dynamic'] = {
                'mean': self.dynamic_stats.get_means(),
                'std': self.dynamic_stats.get_stds(),
                'num_channels': self.dynamic_stats.num_channels,
                'channel_names': [self.dynamic_stats.channels[i].channel_name for i in range(self.dynamic_stats.num_channels)],
                'total_samples': self.dynamic_stats.total_samples,
            }
        
        if self.static_stats:
            output_data['static'] = {
                'mean': self.static_stats.get_means(),
                'std': self.static_stats.get_stds(),
                'num_channels': self.static_stats.num_channels,
                'channel_names': [self.static_stats.channels[i].channel_name for i in range(self.static_stats.num_channels)],
                'total_samples': self.static_stats.total_samples,
            }
        
        # 保存到文件
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print("=" * 80)
        print(f"[CHECK] ✅ 统计量已保存至 {output_path}")
        print("=" * 80 + "\n")
    
    def load_stats(self, filepath: Path) -> Dict:
        """
        从 JSON 文件加载统计量
        
        Args:
            filepath: 统计量文件路径
        
        Returns:
            Dict: 统计量字典
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        
        self.logger.info(f"📂 已加载统计量: {filepath}")
        return stats
    
    def get_normalization_params(self) -> Dict:
        """
        获取归一化参数
        
        返回格式与保存的 JSON 一致
        
        Returns:
            Dict: 包含动态和静态影像的均值和标准差
        """
        if self.dynamic_stats is None and self.static_stats is None:
            raise ValueError("❌ 未计算统计量")
        
        params = {}
        
        if self.dynamic_stats:
            params['dynamic'] = {
                'mean': self.dynamic_stats.get_means(),
                'std': self.dynamic_stats.get_stds(),
            }
        
        if self.static_stats:
            params['static'] = {
                'mean': self.static_stats.get_means(),
                'std': self.static_stats.get_stds(),
            }
        
        return params
    
    def __repr__(self) -> str:
        """字符串表示"""
        dynamic_info = (
            f"Dynamic: {self.dynamic_stats.num_channels} channels, "
            f"{self.dynamic_stats.total_samples:,} samples"
            if self.dynamic_stats else "Dynamic: Not computed"
        )
        
        static_info = (
            f"Static: {self.static_stats.num_channels} channels, "
            f"{self.static_stats.total_samples:,} samples"
            if self.static_stats else "Static: Not computed"
        )
        
        return (
            f"StatsCalculator(\n"
            f"  {dynamic_info},\n"
            f"  {static_info}\n"
            f")"
        )


# ============================================================================
# 使用示例和主程序
# ============================================================================

if __name__ == "__main__":
    try:
        from raster_crawler import RasterCrawler
        
        print("=" * 80)
        print("StatsCalculator 使用示例")
        print("=" * 80)
        
        # 初始化配置
        config_path = Path(__file__).parent / 'config.yaml'
        config = ConfigManager(str(config_path))
        
        # 初始化爬虫
        print("\n1️⃣  初始化 RasterCrawler...")
        filename_pattern = r'GPP(?P<year>\d{2})(?P<month>\d{2})(?P<day>\d{2})'
        
        try:
            dynamic_crawler = RasterCrawler(
                config=config,
                raster_dir=config.get_resolved_path('dynamic_images_dir'),
                filename_pattern=filename_pattern
            )
            dynamic_rasters = dynamic_crawler.get_all_rasters()
            print(f"✅ 发现 {len(dynamic_rasters)} 个动态影像文件")
        except Exception as e:
            print(f"⚠️  动态影像爬虫初始化失败: {e}")
            dynamic_rasters = []
        
        # 初始化统计计算器
        print("\n2️⃣  初始化 StatsCalculator...")
        calculator = StatsCalculator(
            config=config,
            dynamic_channel_names=['R', 'G', 'B', 'NIR'],
            static_channel_names=['DEM'],
        )
        print(f"✅ 统计计算器已初始化")
        
        # 计算全局统计量
        if dynamic_rasters:
            print("\n3️⃣  计算全局统计量...")
            calculator.compute_global_stats(
                dynamic_rasters=dynamic_rasters,
                sampling_rate=0.1,  # 采样 10%
            )
            
            # 保存统计量
            print("\n4️⃣  保存统计量...")
            calculator.save_stats()
            
            # 获取参数
            print("\n5️⃣  获取归一化参数...")
            params = calculator.get_normalization_params()
            print(f"✅ 参数获取成功:")
            print(f"   {params}")
        
        print("\n" + "=" * 80)
        print("✅ 示例完成!")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
