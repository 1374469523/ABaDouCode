import os

# pip install openai
from openai import OpenAI
from llm_config import DEEPSEEK_CONFIG

client = OpenAI(
    api_key=DEEPSEEK_CONFIG["api_key"],
    base_url=DEEPSEEK_CONFIG["base_url"],
)

completion = client.chat.completions.create(
    # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    model="deepseek-v4-flash", # 模型的代号

    # 对话列表
    messages=[
        {"role": "system", "content": "You are a helpful assistant."}, # 给大模型的命令，角色的定义
        {"role": "user", "content": "你是谁？"},  # 用户的提问
        {"role": "user", "content": "你是谁？"},  # 用户的提问
        {"role": "user", "content": "你是谁？"},  # 用户的提问
    ]
)
print(completion.choices[0].message.content)