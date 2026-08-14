"""大模型引擎（策略实现）：OpenAI 兼容接口 + 静态 few-shot。

不依赖任何本地权重，通过 ``llm_config.py`` 提供的 key 调用大模型；
用类别说明 + 每类一个示例作为 system prompt，让模型输出类别名。
"""
from __future__ import annotations

import certifi
import httpx
import openai

from core.config import LLM
from core.constants import CLASS_NAMES
from engine.base import BaseIntentClassifier, Prediction
from engine.registry import ClassifierRegistry

# 每个类别一个静态示例（few-shot）
_FEW_SHOT: dict[str, str] = {
    "Travel-Query": "导航到最近的加油站",
    "Music-Play": "播放周杰伦的歌曲",
    "FilmTele-Play": "播放一部古装悬疑剧",
    "Video-Play": "找个魔兽世界的比赛视频",
    "Radio-Listen": "打开武汉交通广播",
    "HomeAppliance-Control": "把空调调到26度",
    "Weather-Query": "今天会下雨吗",
    "Alarm-Update": "设置早上七点的闹钟",
    "Calendar-Query": "今天星期几",
    "TVProgram-Play": "切换到湖南卫视",
    "Audio-Play": "播放一本有声小说",
    "Other": "讲个笑话",
}

_SYSTEM_PROMPT = (
    "你是一个车载语音助手的意图识别模块。请从下列类别中选择一个最能表达用户意图的类别，"
    "只输出类别名，不要输出任何解释或标点。\n\n类别及示例：\n"
) + "\n".join(f"- {c}：{_FEW_SHOT[c]}" for c in CLASS_NAMES)


@ClassifierRegistry.register("llm")
class LlmClassifier(BaseIntentClassifier):
    name = "llm"

    def __init__(self) -> None:
        if not LLM.api_key:
            raise RuntimeError("LLM api_key 为空，请检查仓库根目录 .env")
        # 显式指定 CA 证书，规避系统残留 SSL_CERT_FILE 指向失效文件的问题
        self.client = openai.Client(
            base_url=LLM.base_url,
            api_key=LLM.api_key,
            http_client=httpx.Client(verify=certifi.where()),
        )

    def classify(self, texts: list[str]) -> list[Prediction]:
        results: list[Prediction] = []
        for text in texts:
            resp = self.client.chat.completions.create(
                model=LLM.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": f"用户输入：{text}"},
                ],
                temperature=LLM.temperature,
                max_tokens=LLM.max_tokens,
            )
            raw = (resp.choices[0].message.content or "").strip()
            results.append(Prediction(label=self._normalize(raw), score=1.0))
        return results

    @staticmethod
    def _normalize(raw: str) -> str:
        """把模型输出归一化到类别表内，避免返回未知类别。"""
        if raw in CLASS_NAMES:
            return raw
        for c in CLASS_NAMES:
            if c in raw:
                return c
        return "Other"
