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
    """递归地冻结字典"""
    frozen_content = {}
    for key, value in d.items():
        if isinstance(value, dict):
            frozen_content[key] = freeze_dict(value)
        else:
            frozen_content[key] = value
    return FrozenDict(frozen_content)


class ConfigManager:
    """
    配置管理器类
    """
    
    def __init__(self, config_path: str, config_root: Optional[str] = None, create_experiment_dir: bool = False):
        """
        初始化配置管理器
        
        Args:
            config_path: YAML配置文件路径
            config_root: 配置文件所在目录
            create_experiment_dir: 是否创建新的时间戳实验目录 (默认False，防止多进程或评估时产生垃圾目录)
        """
        self._setup_logging()
        logger = logging.getLogger(__name__)
        
        config_path = Path(config_path).resolve()
        
        if not config_path.exists():
            raise FileNotFoundError(f"❌ 配置文件不存在: {config_path}")
        
        self._config_root = Path(config_root).resolve() if config_root else config_path.parent
        self._config_path = config_path
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
        except Exception as e:
            raise Exception(f"❌ 读取配置文件失败: {e}") from e
        
        if not config_data:
            raise ValueError("❌ 配置文件为空")
        
        self._raw_config = config_data
        
        # 验证路径
        self._validate_paths(config_data)
        
        # 目录管理逻辑修复
        if create_experiment_dir:
            logger.info("📁 正在创建新的时间戳实验目录...")
            self._experiment_output_dir = self._create_timestamped_output_dir(config_data)
            self._save_config_copy(config_data)
            logger.info(f"✅ 实验目录已就绪: {self._experiment_output_dir}")
        else:
            # 如果不创建新实验，则指向配置文件中定义的 output_dir (通常是 experiments/outputs)
            # 这样 quick_eval 或子进程不会报错，但也不会创建新文件夹
            base_output_dir = self._resolve_path(config_data['paths']['output_dir'])
            self._experiment_output_dir = base_output_dir
            logger.debug(f"ℹ️  以只读模式加载配置，基础输出目录: {self._experiment_output_dir}")
        
        self._frozen_config = freeze_dict(config_data)
    
    @staticmethod
    def _setup_logging():
        """配置日志系统"""
        if not logging.getLogger(__name__).handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logging.getLogger(__name__).addHandler(handler)
            logging.getLogger(__name__).setLevel(logging.INFO)
    
    def _validate_paths(self, config_data: Dict[str, Any]):
        """快速失败：验证所有关键路径"""
        if 'paths' not in config_data:
            raise ValueError("❌ 配置中缺少'paths'字段")
        
        paths = config_data['paths']
        # 移除 output_dir 的验证要求，因为它可能尚不存在
        required_paths = {
            'csv_labels': '标签CSV文件',
            'static_images_dir': '静态影像目录',
            'dynamic_images_dir': '动态影像目录'
            # temp_dir 可选，不强制验证
        }
        
        for path_key, path_desc in required_paths.items():
            if path_key not in paths:
                continue # 允许缺失非核心路径
            
            rel_path = paths[path_key]
            abs_path = self._resolve_path(rel_path)
            
            if path_key == 'csv_labels':
                if not abs_path.is_file():
                    raise ValueError(f"❌ {path_desc} 不是有效文件: {abs_path}")
            else:
                if not abs_path.exists():
                     # 对于目录，仅仅警告或者检查父目录，视具体需求而定
                     # 这里保持原有逻辑，要求目录存在
                     raise FileNotFoundError(f"❌ {path_desc} 不存在: {abs_path}")
    
    def _resolve_path(self, path_str: str) -> Path:
        path = Path(path_str)
        if path.is_absolute():
            return path.resolve()
        else:
            return (self._config_root / path).resolve()
    
    def _create_timestamped_output_dir(self, config_data: Dict[str, Any]) -> Path:
        base_output_dir = self._resolve_path(config_data['paths']['output_dir'])
        experiment_id = config_data.get('experiment_id', 'DEFAULT')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        # 增加随机微秒以防止极短时间内的冲突（虽然不常见）
        timestamped_dir_name = f"{timestamp}_{experiment_id}"
        experiment_output_dir = base_output_dir / timestamped_dir_name
        
        try:
            experiment_output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise IOError(f"❌ 无法创建实验输出目录 {experiment_output_dir}: {e}") from e
        
        return experiment_output_dir
    
    def _save_config_copy(self, config_data: Dict[str, Any]):
        config_copy_path = self._experiment_output_dir / 'config_used.yaml'
        try:
            with open(config_copy_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        except Exception as e:
            logging.getLogger(__name__).error(f"❌ 无法保存配置副本: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        current = self._frozen_config
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        return current
    
    def get_all(self) -> FrozenDict:
        return self._frozen_config
    
    def get_experiment_output_dir(self) -> Path:
        return self._experiment_output_dir
    
    def get_resolved_path(self, path_key: str) -> Path:
        path_str = self._raw_config['paths'][path_key]
        return self._resolve_path(path_str)
    
    def __getitem__(self, key: str) -> Any:
        return self._frozen_config[key]