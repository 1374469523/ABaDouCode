"""意图类别常量：全项目唯一权威来源。

训练与推理共用同一份类别表，避免 ``LabelEncoder`` 自动发现顺序
与推理端类别表不一致导致的标签错乱。
"""
from __future__ import annotations

# 12 个意图类别（字母序固定，训练/推理保持一致）
CLASS_NAMES: list[str] = [
    "Alarm-Update",
    "Audio-Play",
    "Calendar-Query",
    "FilmTele-Play",
    "HomeAppliance-Control",
    "Music-Play",
    "Other",
    "Radio-Listen",
    "TVProgram-Play",
    "Travel-Query",
    "Video-Play",
    "Weather-Query",
]

LABEL2ID: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}
ID2LABEL: dict[int, str] = {i: name for name, i in LABEL2ID.items()}
NUM_CLASSES: int = len(CLASS_NAMES)
