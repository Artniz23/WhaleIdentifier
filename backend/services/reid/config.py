import yaml
from typing import Optional

class Config(dict):
    """
    Конфиг с доступом к параметрам через точку.

    Вместо:
        cfg["model_name"]

    можно писать:
        cfg.model_name
    """
    def __getattr__(self, key):
        try:
            val = self[key]
        except KeyError:
            return super().__getattr__(key)
        # Вложенные словари также превращаем в Config,
        # чтобы поддерживать доступ через точку на любом уровне.
        if isinstance(val, dict):
            return Config(val)
        return val


def load_config(path: str, default_path: Optional[str]) -> Config:
    """
    Загружает основной конфиг и при необходимости
    дополняет его значениями из default-конфига.

    Приоритет:
        1. path
        2. default_path

    Если параметр отсутствует в основном конфиге,
    используется значение из default-конфига.
    """

    # Загружаем пользовательский конфиг.
    with open(path) as f:
        cfg = Config(yaml.full_load(f))
    if default_path is not None:
        # Подгружаем конфиг со значениями по умолчанию.
        with open(default_path) as f:
            default_cfg = Config(yaml.full_load(f))
        # Добавляем только отсутствующие параметры,
        # не перезаписывая значения из основного конфига.
        for key, val in default_cfg.items():
            if key not in cfg:
                print(f"used default config {key}: {val}")
                cfg[key] = val
    return cfg