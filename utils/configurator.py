import os
import yaml
import functools
from pathlib import Path
from typing import Any, Dict, Optional, Union


class SingletonMeta(type):
    _instances = {}
    _lock = None

    def __call__(cls, *args, **kwargs):
        if cls._lock is None:
            import threading
            cls._lock = threading.Lock()

        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
            return cls._instances[cls]


class NestedDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__()
        for arg in args:
            if isinstance(arg, dict):
                self._update_from_dict(arg)
        self._update_from_dict(kwargs)

    def _update_from_dict(self, d: Dict):
        for k, v in d.items():
            if isinstance(v, dict):
                self[k] = NestedDict(v)
            else:
                self[k] = v

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'NestedDict' has no attribute '{item}'")

    def __setattr__(self, key: str, value: Any):
        self[key] = value

    def __delattr__(self, item: str):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"'NestedDict' has no attribute '{item}'")

    def get(self, key: str, default: Any = None, sep: str = '.') -> Any:
        keys = key.split(sep)
        value = self
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any, sep: str = '.') -> None:
        keys = key.split(sep)
        target = self
        for k in keys[:-1]:
            if k not in target:
                target[k] = NestedDict()
            target = target[k]
        target[keys[-1]] = value


class ConfigurationManager(metaclass=SingletonMeta):
    _config: Optional[NestedDict] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        if not self._initialized and config_path is not None:
            self._load_config(config_path)
            self._initialized = True

    def _load_config(self, config_path: Union[str, Path]) -> None:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)

        self._config = NestedDict(raw_config)
        self._resolve_env_vars()
        self._create_directories()

    def _resolve_env_vars(self) -> None:
        def resolve(obj):
            if isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
                env_var = obj[2:-1]
                default = None
                if ':' in env_var:
                    env_var, default = env_var.split(':', 1)
                return os.getenv(env_var, default)
            elif isinstance(obj, dict):
                return {k: resolve(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [resolve(item) for item in obj]
            return obj

        self._config = NestedDict(resolve(self._config))

    def _create_directories(self) -> None:
        output_dirs = [
            self._config.get('output.checkpoint_dir'),
            self._config.get('output.weight_dir'),
            self._config.get('output.log_dir'),
            self._config.get('output.result_dir'),
            self._config.get('output.viz_dir'),
        ]
        for dir_path in output_dirs:
            if dir_path:
                Path(dir_path).mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_instance(cls, config_path: Optional[Union[str, Path]] = None) -> 'ConfigurationManager':
        if cls._instances.get(cls) is None and config_path is not None:
            cls(config_path)
        return cls._instances.get(cls)

    @classmethod
    def get_config(cls, key: Optional[str] = None, default: Any = None) -> Any:
        instance = cls.get_instance()
        if instance is None or instance._config is None:
            raise RuntimeError("Configuration not initialized. Call initialize() first.")

        if key is None:
            return instance._config
        return instance._config.get(key, default)

    @classmethod
    def initialize(cls, config_path: Union[str, Path]) -> None:
        cls.get_instance(config_path)

    def update(self, updates: Dict[str, Any]) -> None:
        for key, value in updates.items():
            self._config.set(key, value)


def configurable(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cfg = ConfigurationManager.get_config()
        sig = func.__code__.co_varnames[:func.__code__.co_argcount]

        for i, arg_name in enumerate(sig):
            if arg_name == 'cfg' or arg_name == 'config':
                if i < len(args):
                    break
                kwargs[arg_name] = cfg
                break

        return func(*args, **kwargs)
    return wrapper
