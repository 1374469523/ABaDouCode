"""
流式输出 — 逐 chunk 展示模型回复（含 thinking 过程）
"""

from openai import OpenAI
from llm_config import DEEPSEEK_CONFIG

client = OpenAI(
    api_key=DEEPSEEK_CONFIG["api_key"],
    base_url=DEEPSEEK_CONFIG["base_url"],
)

stream = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "你好，帮我介绍机器学习"},
    ],
    stream=True,  # 大模型预测了一个token，按照协议返回回来；
    # 大模型结束条件： [EOS] 特殊token，大模型认为结束应该结束； max output tokens 限制大模型输出的最多token个数
    # 训练过程中，【eos】也是训练文本的一部分。

    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
)

full_content = ""
reasoning_content = ""

# 迭代读取 stream
for chunk in stream:
    choice = chunk.choices[0] if chunk.choices else None
    if choice is None:
        # 最后一个 chunk 可能只包含 usage 信息
        if hasattr(chunk, "usage") and chunk.usage:
            print(f"\n--- Token 用量 ---")
            print(f"  prompt_tokens:     {chunk.usage.prompt_tokens}")
            print(f"  completion_tokens: {chunk.usage.completion_tokens}")
            print(f"  total_tokens:      {chunk.usage.total_tokens}")
        continue

    delta = choice.delta

    # reasoning 内容（若模型返回）
    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
        reasoning_content += delta.reasoning_content
        print(delta.reasoning_content, end="", flush=True)

    # 常规内容
    if delta.content:
        full_content += delta.content
        print(delta.content, end="", flush=True)

print("\n" + "=" * 50)
print("完整回复：")
print(full_content)
