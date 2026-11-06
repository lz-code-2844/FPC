import inspect
from typing import Any, Dict, Optional, Type, Union


class Registry:
    def __init__(self, name: str, scope: Optional[str] = None):
        self._name = name
        self._scope = scope
        self._module_dict: Dict[str, Type] = {}
        self._children: Dict[str, 'Registry'] = {}

    def __len__(self) -> int:
        return len(self._module_dict)

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def __repr__(self) -> str:
        format_str = f'{self.__class__.__name__}(name={self._name}, items={list(self._module_dict.keys())})'
        return format_str

    @property
    def name(self) -> str:
        return self._name

    @property
    def scope(self) -> Optional[str]:
        return self._scope

    @property
    def module_dict(self) -> Dict[str, Type]:
        return self._module_dict

    def get(self, key: str) -> Optional[Type]:
        scope = None
        if ':' in key:
            scope, key = key.split(':', 1)

        if scope is not None and scope != self._scope:
            if scope in self._children:
                return self._children[scope].get(key)
            else:
                parent = self.parent
                if parent is not None:
                    return parent.get(f'{scope}:{key}')
                return None

        obj = self._module_dict.get(key, None)
        if obj is None and scope is None:
            for child in self._children.values():
                obj = child.get(key)
                if obj is not None:
                    break

        return obj

    def _register_module(self, module_class: Type, module_name: Optional[str] = None, force: bool = False) -> None:
        if not inspect.isclass(module_class):
            raise TypeError(f'module must be a class, but got {type(module_class)}')

        if module_name is None:
            module_name = module_class.__name__

        if isinstance(module_name, str):
            module_name = [module_name]

        for name in module_name:
            if not force and name in self._module_dict:
                raise KeyError(f'{name} is already registered in {self.name}')
            self._module_dict[name] = module_class

    def register_module(self, name: Optional[str] = None, force: bool = False, module: Optional[Type] = None):
        if not isinstance(force, bool):
            raise TypeError(f'force must be a boolean, but got {type(force)}')

        if module is not None:
            self._register_module(module_class=module, module_name=name, force=force)
            return module

        def _register(module_class):
            self._register_module(module_class=module_class, module_name=name, force=force)
            return module_class

        return _register

    def register_child(self, registry: 'Registry') -> None:
        assert isinstance(registry, Registry)
        assert registry.scope is not None
        assert registry.scope not in self._children
        self._children[registry.scope] = registry

    def __call__(self, *args, **kwargs):
        return self.register_module(*args, **kwargs)


MODELS = Registry('model')
DATASETS = Registry('dataset')
TRANSFORMS = Registry('transform')
OPTIMIZERS = Registry('optimizer')
SCHEDULERS = Registry('scheduler')
LOSSES = Registry('loss')


def build_from_cfg(cfg: Union[Dict, Any], registry: Registry, default_args: Optional[Dict] = None) -> Any:
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
        if default_args is None or 'type' not in default_args:
            raise KeyError('cfg or default_args must contain the key "type"')

    cfg = cfg.copy()
    obj_type = cfg.pop('type')

    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f'{obj_type} is not in the {registry.name} registry')
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f'type must be a str or valid type, but got {type(obj_type)}')

    if default_args is not None:
        for key, value in default_args.items():
            cfg.setdefault(key, value)

    return obj_cls(**cfg)
