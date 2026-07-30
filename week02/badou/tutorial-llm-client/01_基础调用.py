import os
from openai import OpenAI # 结合openai sdk 调用deepseek 模型
from openai.types.chat import ChatCompletion
from llm_config import DEEPSEEK_CONFIG

# llm client
client = OpenAI(
    api_key=DEEPSEEK_CONFIG["api_key"], # 大模型厂商后台页面 创建api key ，计费  / 并发 / 账单
    base_url=DEEPSEEK_CONFIG["base_url"], # 服务地址，云端地址，运算大模型
)

# 创建了一个和大模型对话的请求
response = client.chat.completions.create(
    model="deepseek-v4-flash", # 模型名称

    # 历史对话， 基于 历史对话，生成后序的回答
    # 历史对话 -》 大模型 -》 下一轮回答
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "你好，帮我介绍机器学习"},
        # {"role": "assistant", "content": "机器学习是一种基于计算机算法的机器学习方法，它可以从数据中自动学习并预测结果。"},
    ],
    stream=False, # 非流式，
    reasoning_effort="high", # 思考能力。 高中低， 和 token 消耗，和 费用相关
    extra_body={"thinking": {"type": "enabled"}} # 是否打开思考
)

# 注：stream=False 时返回的实际上是 ChatCompletion，不是 Stream
# 但 openai SDK 的 create() 返回类型是 ChatCompletion | Stream 联合类型
# 这里用 ChatCompletion 类型声明结果变量，类型检查器就能认出 .choices
completion: ChatCompletion = response  # type: ignore[assignment]
print(response) # 完整的大大模型的回复： 包含回答 + token消耗 + 回答过程相关的信息
print(completion.choices[0].message.content)

"""
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "你好，帮我介绍机器学习"}, <- 输入的提问， 对话列表长度2
        
        {"role": "assistant", "content": "机器学习是一种基于计算机算法的机器学习方法，它可以从数据中自动学习并预测结果。"},

        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "你好，帮我介绍机器学习"},
        {"role": "assistant", "content": "机器学习是一种基于计算机算法的机器学习方法，它可以从数据中自动学习并预测结果。"},
        {"role": "user", "content": "你好，帮我介绍深度学习"}, <- 新输入的提问，对话长度4


"""