from core.env import get_required_env


def build_db_config() -> dict:
    """
    Формирует конфигурацию подключения к PostgreSQL.
    """
    return {
        "host": get_required_env("DB_HOST"),
        "port": int(get_required_env("DB_PORT")),
        "dbname": get_required_env("DB_NAME"),
        "user": get_required_env("DB_USER"),
        "password": get_required_env("DB_PASSWORD"),
    }
