# 车载意图识别服务

用户通过 FastAPI 访问，后端对用户询问做意图识别。四条技术路线，用统一接口对外服务：

| 路线 | 端点 | 说明 |
|------|------|------|
| 正则表达式 | `POST /v1/intent/regex` | 关键词规则 + 命中打分，无需训练 |
| TF-IDF + SVM | `POST /v1/intent/tfidf` | 轻量统计分类，需训练 |
| BERT | `POST /v1/intent/bert` | 深度语义分类，需微调 |
| 大模型 | `POST /v1/intent/llm` | 静态 few-shot 调用，key 在 `llm_config.py` |

## 设计模式

- **策略模式**：`engine/base.py` 定义 `BaseIntentClassifier` 抽象接口，四条路线各自实现 `classify`。
- **工厂模式 + 单例懒加载**：`engine/registry.py` 用装饰器注册引擎，`get(name)` 惰性实例化并缓存，保证 BERT/LLM 等重资源进程内只加载一次。
- **类别唯一来源**：`core/constants.py` 固定 12 类顺序，训练与推理共用，避免标签错乱。

## 目录结构

```
作业/
├── app/                 # FastAPI 入口与数据模型
│   ├── main.py
│   └── schemas.py
├── core/                # 配置 / 常量 / 日志
│   ├── config.py
│   ├── constants.py
│   └── logging_config.py
├── engine/              # 四条路线的策略实现
│   ├── base.py          # 抽象策略 + Prediction
│   ├── registry.py      # 注册表工厂
│   ├── regex_engine.py
│   ├── tfidf_engine.py
│   ├── bert_engine.py
│   └── llm_engine.py
├── training/            # 训练脚本
│   ├── train_tfidf.py
│   └── train_bert.py
├── data/                # dataset.csv + 停用词
├── weights/             # 训练产物
└── run.py               # 启动入口
```

## 快速开始

```bash
# 1. 训练 TF-IDF + SVM（产出 weights/tfidf_svm.pkl）
python training/train_tfidf.py

# 2. 训练 BERT（默认 2000 条；--sample 0 全量）
python training/train_bert.py --sample 2000

# 3. 启动服务
python run.py
# 或 uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 调用示例

```bash
curl -X POST 'http://127.0.0.1:8000/v1/intent/bert' \
  -H 'Content-Type: application/json' \
  -d '{"request_id": "001", "text": "帮我播放周杰伦的歌曲"}'
```

响应：

```json
{
  "request_id": "001",
  "engine": "bert",
  "items": [{"text": "帮我播放周杰伦的歌曲", "label": "Music-Play", "score": 0.98}],
  "elapsed_ms": 42.1,
  "error": "ok"
}
```

`text` 支持字符串或字符串数组（批量）。
