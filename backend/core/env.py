import os


def get_required_env(name: str) -> str:
    """
    Возвращает значение обязательной переменной окружения.

    Raises:
        ValueError: если переменная не задана.
    """
    value = os.getenv(name)

    if not value:
        raise ValueError(
            f"Environment variable '{name}' is required"
        )

    return value
