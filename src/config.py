import os
from dotenv import load_dotenv
from pathlib import Path

class Config:
    """
    Класс для загрузки и хранения переменных окружения.
    """

    def __init__(self, env_file: str = "../.env"):
        # ищем файл .env рядом с проектом
        env_path = Path(env_file)
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
        else:
            print(f"⚠️ Файл {env_file} не найден, используем только системные env")

        # грузим все переменные окружения
        self._vars = dict(os.environ)

    def get(self, key: str, default=None):
        """Безопасное получение значения"""
        return self._vars.get(key, default)

    def __getattr__(self, item):
        """Позволяет обращаться к переменным как к атрибутам"""
        if item in self._vars:
            return self._vars[item]
        raise AttributeError(f"Переменная '{item}' не найдена")

    def all(self):
        """Вернуть словарь со всеми переменными"""
        return self._vars
