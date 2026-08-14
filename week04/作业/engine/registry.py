"""工厂模式 + 单例懒加载：分类器注册表。

- 通过 ``@ClassifierRegistry.register("bert")`` 装饰器把引擎类登记进来；
- ``get(name)`` 时惰性实例化并缓存，保证重资源（BERT 权重 / LLM client / TFIDF 模型）
  在进程内只加载一次。
"""
from __future__ import annotations

from typing import Callable

from engine.base import BaseIntentClassifier


class ClassifierRegistry:
    _builders: dict[str, type[BaseIntentClassifier]] = {}
    _instances: dict[str, BaseIntentClassifier] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[BaseIntentClassifier]], type[BaseIntentClassifier]]:
        def decorator(builder: type[BaseIntentClassifier]) -> type[BaseIntentClassifier]:
            cls._builders[name] = builder
            return builder

        return decorator

    @classmethod
    def get(cls, name: str) -> BaseIntentClassifier:
        if name not in cls._builders:
            raise KeyError(f"未知分类器: {name}，可用: {list(cls._builders)}")
        if name not in cls._instances:
            cls._instances[name] = cls._builders[name]()  # 惰性实例化（单例）
        return cls._instances[name]

    @classmethod
    def names(cls) -> list[str]:
        return list(cls._builders)
