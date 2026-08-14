"""全局配置：路径、LLM 连接信息。

路径基于文件位置推导，避免运行目录不同导致的相对路径失效；
LLM 密钥从仓库根目录 ``llm_config.py``（读取 .env）获取。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录 = week04/作业
BASE_DIR = Path(__file__).resolve().parent.parent
# 仓库根目录 = ABaDouCode
REPO_ROOT = BASE_DIR.parent.parent

# 先加载仓库根的 .env，再导入 llm_config（大模型 key 来源）
load_dotenv(REPO_ROOT / ".env")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_config import DEEPSEEK_CONFIG  # noqa: E402


@dataclass(frozen=True)
class Paths:
    data_dir: Path = BASE_DIR / "data"
    dataset: Path = BASE_DIR / "data" / "dataset.csv"
    stopwords: Path = BASE_DIR / "data" / "baidu_stopwords.txt"
    weights_dir: Path = BASE_DIR / "weights"
    tfidf_weights: Path = BASE_DIR / "weights" / "tfidf_svm.pkl"
    bert_weights: Path = BASE_DIR / "weights" / "bert.pt"
    # 用户指定的 BERT 预训练模型路径
    bert_pretrained: Path = REPO_ROOT / "moudel" / "google-bert" / "bert-base-chinese"


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "deepseek"  # 可切换 deepseek / qwen
    model: str = "deepseek-chat"
    base_url: str = DEEPSEEK_CONFIG["base_url"]
    api_key: str = DEEPSEEK_CONFIG["api_key"]
    temperature: float = 0.0
    max_tokens: int = 64


PATHS = Paths()
LLM = LLMConfig()
