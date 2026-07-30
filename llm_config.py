# 模型厂商配置：api_key 和 base_url
# 使用方式：from llm_config import DEEPSEEK_CONFIG, OPENAI_CONFIG

import os
from dotenv import load_dotenv

load_dotenv()

# ==================== DeepSeek ====================
DEEPSEEK_CONFIG = {
    "api_key": os.getenv("DEEPSEEK_API_KEY"),
    "base_url": "https://api.deepseek.com",
}

# ==================== OpenAI ====================
OPENAI_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY"),
    "base_url": "https://api.openai.com/v1",
}

# ==================== 阿里百炼 ====================
DASHSCOPE_CONFIG = {
    "api_key": os.getenv("DASHSCOPE_API_KEY"),
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}

# ==================== 百度千帆 ====================
QIANFAN_CONFIG = {
    "api_key": os.getenv("QIANFAN_API_KEY"),
    "base_url": "https://qianfan.baidubce.com/v2",
}

# ==================== 讯飞星火 ====================
SPARK_CONFIG = {
    "api_key": os.getenv("SPARK_API_KEY"),
    "base_url": "https://spark-api-open.xf-yun.com/v1",
}
