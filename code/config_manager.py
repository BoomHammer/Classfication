"""
ConfigManager: 配置管理模块
实现快速失败机制、路径验证、自动目录创建和参数冻结
"""

import os
import sys
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from copy import deepcopy


class FrozenDict(dict):
    """不可修改的字典实现"""
    
    def __init__(self, *args, **kwargs):
        """初始化时允许设置值"""
        object.__setattr__(self, '_frozen', False)
        super().__init__(*args, **kwargs)
        object.__setattr__(self, '_frozen', True)
    
    def __setitem__(self, key, value):
        if object.__getattribute__(self, '_frozen'):
            raise TypeError("配置已冻结，不允许修改")
        super().__setitem__(key, value)
    
    def __delitem__(self, key):
        if object.__getattribute__(self, '_frozen'):
            raise TypeError("配置已冻结，不允许删除")
        super().__delitem__(key)
    
    def clear(self):
        if object.__getattribute__(self, '_frozen'):
            raise TypeError("配置已冻结，不允许清空")
        super().clear()
    
    def pop(self, *args):
        if object.__getattribute__(self, '_frozen'):
            raise TypeError("配置已冻结，不允许弹出")
        return super().pop(*args)
    
    def popitem(self):
        if object.__getattribute__(self, '_frozen'):
            raise TypeError("配置已冻结，不允许弹出")
        return super().popitem()
    
    def setdefault(self, key, default=None):
        if object.__getattribute__(self, '_frozen'):
            raise TypeError("配置已冻结，不允许设置默认值")
        return super().setdefault(key, default)
    
    def update(self, *args, **kwargs):
        if object.__getattribute__(self, '_frozen'):
            raise TypeError("配置已冻结，不允许更新")
        return super().update(*args, **kwargs)
    
    def __reduce__(self):
        """支持 pickle 序列化"""
        # 返回一个元组：(可调用对象，参数元组)
        # 这样可以在反序列化时正确恢复对象
        return (FrozenDict, (dict(self),))
    
    def __getstate__(self):
        """获取序列化状态"""
        return dict(self)
    
    def __setstate__(self, state):
        """恢复序列化状态"""
        object.__setattr__(self, '_frozen', False)
        self.update(state)
        object.__setattr__(self, '_frozen', True)


def freeze_dict(d: dict) -> FrozenDict:
    """
    递归地冻结字典及其嵌套的字典
    
    Args:
        d: 待冻结的字典
    
    Returns:
        FrozenDict: 冻结后的字典
    """
    frozen_content = {}
    for key, value in d.items():
        if isinstance(value, dict):
            frozen_content[key] = freeze_dict(value)
        else:
            frozen_content[key] = value
    
    # 创建 FrozenDict 并一次性初始化所有内容
    return FrozenDict(frozen_content)


class ConfigManager:
    """
    配置管理器类
    
    功能：
    1. 从YAML文件读取配置
    2. 实现快速失败机制（路径验证）
    3. 自动创建时间戳子文件夹用于实验管理
    4. 参数冻结（只读保护）
    
    使用示例：
        config = ConfigManager('./code/config.yaml')
        print(config.get('model', {}).get('name'))
        output_path = config.get_experiment_output_dir()
    """
    
    def __init__(self, config_path: str, config_root: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: YAML配置文件路径
            config_root: 配置文件所在目录（用于相对路径计算），默认为None则使用config_path的父目录
        
        Raises:
            FileNotFoundError: 配置文件不存在
            yaml.YAMLError: YAML解析错误
            ValueError: 路径验证失败
        """
        # 初始化日志
        self._setup_logging()
        
        logger = logging.getLogger(__name__)
        
        # 转换路径为Path对象
        config_path = Path(config_path).resolve()
        
        # 验证配置文件存在
        if not config_path.exists():
            error_msg = f"❌ 配置文件不存在: {config_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        if not config_path.is_file():
            error_msg = f"❌ 配置路径不是文件: {config_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # 设置配置根目录
        self._config_root = Path(config_root).resolve() if config_root else config_path.parent
        self._config_path = config_path
        
        logger.info(f"📂 配置根目录: {self._config_root}")
        logger.info(f"📄 配置文件: {config_path}")
        
        # 读取YAML文件
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            logger.info(f"✅ 成功加载配置文件")
        except yaml.YAMLError as e:
            error_msg = f"❌ YAML解析错误: {e}"
            logger.error(error_msg)
            raise yaml.YAMLError(error_msg) from e
        except Exception as e:
            error_msg = f"❌ 读取配置文件失败: {e}"
            logger.error(error_msg)
            raise Exception(error_msg) from e
        
        if not config_data:
            error_msg = "❌ 配置文件为空"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # 保存原始配置（用于后续冻结）
        self._raw_config = config_data
        
        # 快速失败：验证所有路径
        logger.info("🔍 开始路径验证...")
        self._validate_paths(config_data)
        logger.info("✅ 路径验证完成")
        
        # 创建时间戳输出目录
        logger.info("📁 创建时间戳输出目录...")
        self._experiment_output_dir = self._create_timestamped_output_dir(config_data)
        logger.info(f"✅ 实验输出目录创建成功: {self._experiment_output_dir}")
        
        # 保存配置文件副本
        self._save_config_copy(config_data)
        
        # 冻结配置
        self._frozen_config = freeze_dict(config_data)
        
        logger.info("✅ 配置对象已冻结（只读保护）")
    
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
    
    def _validate_paths(self, config_data: Dict[str, Any]):
        """
        快速失败：验证所有关键路径
        
        Args:
            config_data: 配置字典
        
        Raises:
            ValueError: 路径验证失败
        """
        logger = logging.getLogger(__name__)
        
        if 'paths' not in config_data:
            error_msg = "❌ 配置中缺少'paths'字段"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        paths = config_data['paths']
        
        # 需要验证存在的路径列表（不包括output_dir，因为会自动创建）
        required_paths = {
            'csv_labels': '标签CSV文件',
            'static_images_dir': '静态影像目录',
            'dynamic_images_dir': '动态影像目录',
            'temp_dir': '临时目录'
        }
        
        for path_key, path_desc in required_paths.items():
            if path_key not in paths:
                error_msg = f"❌ 配置中缺少'{path_key}'路径"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            rel_path = paths[path_key]
            abs_path = self._resolve_path(rel_path)
            
            # 特殊处理：CSV文件需要检查文件，目录需要检查目录
            if path_key == 'csv_labels':
                if not abs_path.exists():
                    error_msg = f"❌ {path_desc} 不存在: {abs_path}"
                    logger.error(error_msg)
                    raise FileNotFoundError(error_msg)
                if not abs_path.is_file():
                    error_msg = f"❌ {path_desc} 不是文件: {abs_path}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            else:
                if not abs_path.exists():
                    error_msg = f"❌ {path_desc} 不存在: {abs_path}"
                    logger.error(error_msg)
                    raise FileNotFoundError(error_msg)
                if not abs_path.is_dir():
                    error_msg = f"❌ {path_desc} 不是目录: {abs_path}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            
            logger.debug(f"✓ {path_desc}: {abs_path}")
    
    def _resolve_path(self, path_str: str) -> Path:
        """
        解析相对或绝对路径
        
        Args:
            path_str: 路径字符串
        
        Returns:
            Path: 绝对路径对象
        """
        path = Path(path_str)
        if path.is_absolute():
            return path.resolve()
        else:
            return (self._config_root / path).resolve()
    
    def _create_timestamped_output_dir(self, config_data: Dict[str, Any]) -> Path:
        """
        为本次实验创建带时间戳的输出目录
        
        格式: {output_dir}/{YYYYMMDD_HHMM_EXP_ID}/
        例如: ./experiments/outputs/20231027_1430_EXP_2023_001/
        
        Args:
            config_data: 配置字典
        
        Returns:
            Path: 创建的实验输出目录路径
        """
        logger = logging.getLogger(__name__)
        
        # 获取基础输出目录和实验ID
        base_output_dir = self._resolve_path(config_data['paths']['output_dir'])
        experiment_id = config_data.get('experiment_id', 'DEFAULT')
        
        # 创建时间戳格式: YYYYMMDD_HHMM
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        timestamped_dir_name = f"{timestamp}_{experiment_id}"
        experiment_output_dir = base_output_dir / timestamped_dir_name
        
        # 创建目录（包括所有父目录）
        try:
            experiment_output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 实验输出目录: {experiment_output_dir}")
        except Exception as e:
            error_msg = f"❌ 无法创建实验输出目录 {experiment_output_dir}: {e}"
            logger.error(error_msg)
            raise IOError(error_msg) from e
        
        return experiment_output_dir
    
    def _save_config_copy(self, config_data: Dict[str, Any]):
        """
        将配置文件副本保存到实验输出目录
        
        Args:
            config_data: 配置字典
        """
        logger = logging.getLogger(__name__)
        
        config_copy_path = self._experiment_output_dir / 'config_used.yaml'
        
        try:
            with open(config_copy_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, 
                         allow_unicode=True, sort_keys=False)
            logger.info(f"💾 配置副本已保存: {config_copy_path}")
        except Exception as e:
            error_msg = f"❌ 无法保存配置副本: {e}"
            logger.error(error_msg)
            raise IOError(error_msg) from e
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值（支持点符号访问嵌套配置）
        
        Args:
            key: 配置键，支持点符号如'model.name'
            default: 默认值
        
        Returns:
            配置值或默认值
        
        Examples:
            config.get('model.name')  # 返回 "ResNet18_LTAE"
            config.get('model.dropout')  # 返回 0.2
        """
        keys = key.split('.')
        current = self._frozen_config
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        
        return current
    
    def get_all(self) -> FrozenDict:
        """
        获取所有配置（冻结的）
        
        Returns:
            FrozenDict: 完整的冻结配置字典
        """
        return self._frozen_config
    
    def get_experiment_output_dir(self) -> Path:
        """
        获取本次实验的输出目录
        
        Returns:
            Path: 实验输出目录路径
        
        Examples:
            output_dir = config.get_experiment_output_dir()
            model_path = output_dir / 'best_model.pth'
        """
        return self._experiment_output_dir
    
    def get_resolved_path(self, path_key: str) -> Path:
        """
        获取resolved的绝对路径
        
        Args:
            path_key: paths字典中的键，如'csv_labels', 'output_dir'等
        
        Returns:
            Path: 绝对路径
        
        Raises:
            KeyError: 路径键不存在
        """
        if 'paths' not in self._raw_config:
            raise KeyError("配置中缺少'paths'字段")
        
        if path_key not in self._raw_config['paths']:
            raise KeyError(f"路径键'{path_key}'不存在")
        
        path_str = self._raw_config['paths'][path_key]
        return self._resolve_path(path_str)
    
    def get_paths(self) -> Dict[str, Path]:
        """
        获取所有已解析的路径
        
        Returns:
            Dict[str, Path]: 路径键到绝对路径的映射
        """
        paths = {}
        if 'paths' in self._raw_config:
            for key, path_str in self._raw_config['paths'].items():
                paths[key] = self._resolve_path(path_str)
        return paths
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"ConfigManager(\n"
            f"  config_path={self._config_path},\n"
            f"  config_root={self._config_root},\n"
            f"  experiment_output_dir={self._experiment_output_dir},\n"
            f"  frozen=True\n"
            f")"
        )
    
    def __getitem__(self, key: str) -> Any:
        """支持字典式访问"""
        return self._frozen_config[key]


# ============================================================================
# 使用示例和测试
# ============================================================================

if __name__ == "__main__":
    try:
        # 初始化配置管理器
        print("=" * 70)
        print("ConfigManager 使用示例")
        print("=" * 70)
        
        config = ConfigManager('./config.yaml')
        
        print("\n✅ 配置加载成功！\n")
        
        # 1. 获取嵌套配置值
        print("1️⃣  获取模型配置:")
        print(f"   模型名称: {config.get('model.name')}")
        print(f"   分类数: {config.get('model.num_classes')}")
        print(f"   Dropout: {config.get('model.dropout')}")
        
        # 2. 获取训练超参数
        print("\n2️⃣  获取训练超参数:")
        print(f"   批次大小: {config.get('train.batch_size')}")
        print(f"   学习率: {config.get('train.learning_rate')}")
        print(f"   训练轮数: {config.get('train.epochs')}")
        
        # 3. 获取数据规范
        print("\n3️⃣  获取数据规范:")
        print(f"   切片大小: {config.get('data_specs.spatial.patch_size')}")
        print(f"   目标分辨率: {config.get('data_specs.spatial.resolution')} m")
        print(f"   时间序列长度: {config.get('data_specs.temporal.max_sequence_length')}")
        
        # 4. 获取路径
        print("\n4️⃣  获取路径:")
        paths = config.get_paths()
        for key, path in paths.items():
            print(f"   {key}: {path}")
        
        # 5. 获取实验输出目录
        print("\n5️⃣  实验输出目录:")
        exp_dir = config.get_experiment_output_dir()
        print(f"   路径: {exp_dir}")
        print(f"   存在: {exp_dir.exists()}")
        
        # 6. 测试参数冻结
        print("\n6️⃣  测试参数冻结:")
        try:
            config._frozen_config['model']['name'] = 'ResNet50'
            print("   ❌ 参数冻结失败！")
        except TypeError as e:
            print(f"   ✅ 参数冻结成功！错误信息: {e}")
        
        # 7. 显示完整配置
        print("\n7️⃣  完整配置结构:")
        print(f"   顶级键: {list(config.get_all().keys())}")
        
        print("\n" + "=" * 70)
        print("✅ 所有测试完成！")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
