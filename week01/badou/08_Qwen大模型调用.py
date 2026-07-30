import os

# pip install openai
from openai import OpenAI

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    # https://bailian.console.aliyun.com/?tab=model#/api-key
    api_key="sk-026bc9ed889740aa948ebf5b06ebbe99", # 账号绑定，用来计费的

    # 大模型厂商的地址，阿里云
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    model="qwen-flash", # 模型的代号

    # 对话列表
    messages=[
        {"role": "system", "content": "You are a helpful assistant."}, # 给大模型的命令，角色的定义
        {"role": "user", "content": "你是谁？"},  # 用户的提问
        {"role": "user", "content": "你是谁？"},  # 用户的提问
        {"role": "user", "content": "你是谁？"},  # 用户的提问
    ]
)
print(completion.choices[0].message.content)