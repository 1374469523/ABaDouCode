"""正则规则引擎（策略实现）：基于关键词规则 + 命中打分。

规则按类别维护一组关键词，扫描命中数越多的类别得分越高，
返回得分最高的类别，命中不足时回退到 ``Other``。
"""
from __future__ import annotations

import re

from engine.base import BaseIntentClassifier, Prediction
from engine.registry import ClassifierRegistry

# 每个意图一组区分度较强的关键词
_REGEX_RULES: dict[str, list[str]] = {
    "Travel-Query": ["导航", "路线", "车票", "机票", "航班", "自驾", "途经", "怎么去"],
    "Music-Play": ["歌", "音乐", "单曲循环", "专辑", "钢琴曲", "演唱"],
    "FilmTele-Play": ["电视剧", "电影", "花絮", "推理剧", "悬疑剧"],
    "Video-Play": ["视频", "直播", "录像", "比赛视频"],
    "Radio-Listen": ["广播", "电台", "收音", "收听"],
    "HomeAppliance-Control": ["空调", "冰箱", "洗衣机", "灯光", "扫地", "窗帘"],
    "Weather-Query": ["天气", "气温", "下雨", "穿衣", "预报", "晴天"],
    "Alarm-Update": ["闹钟", "提醒", "叫醒", "定时"],
    "Calendar-Query": ["日历", "日程", "星期", "几号", "节日"],
    "TVProgram-Play": ["电视", "节目", "频道", "卫视", "中央台"],
    "Audio-Play": ["有声", "听书", "播客", "评书", "相声", "脱口秀"],
}


@ClassifierRegistry.register("regex")
class RegexClassifier(BaseIntentClassifier):
    name = "regex"

    def __init__(self) -> None:
        self._compiled: dict[str, re.Pattern[str]] = {
            cat: re.compile("|".join(map(re.escape, words)))
            for cat, words in _REGEX_RULES.items()
        }

    def classify(self, texts: list[str]) -> list[Prediction]:
        return [self._classify_one(text) for text in texts]

    def _classify_one(self, text: str) -> Prediction:
        best_label = "Other"
        best_score = 0.0
        for category, pattern in self._compiled.items():
            hits = pattern.findall(text)
            if not hits:
                continue
            score = min(1.0, len(hits) / len(_REGEX_RULES[category]))
            if score > best_score:
                best_score = score
                best_label = category
        return Prediction(label=best_label, score=best_score)
