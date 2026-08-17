# 客服工作台与智能问答系统 (CS-Workstation & AI-FAQ System)

本项目是一款集成了 **多渠道接入、双环境 FAQ 管理、智能语义搜索**综合服务平台。系统采用 Python 异步生态构建，旨在平衡机器人自动化效率与人工服务的深度协作。

## 核心特性

* **双环境隔离机制**：“测试环境配置 $\rightarrow$ 模拟验证 $\rightarrow$ 发布中心一键同步 $\rightarrow$ 正式环境生效”流程管理。
* **多维 FAQ 体系**：
    * 支持**类目树管理**（一/二级类目）。
    * **多视角答案**：同一问题可针对微信、App、网页等不同渠道返回差异化答案。
    * **多类型支持**：纯文本、富文本（图文/附件/视频）、交互式卡片。
* **高性能架构**：基于 FastAPI 异步 IO，支持高并发长连接（WebSocket）。


## 技术栈

| 领域 | 技术选型 | 备注 |
| --- | --- | --- |
| **核心框架** | **FastAPI** | 基于 ASGI 的高性能异步 Web 框架 |
| **数据库** | **MySQL + SQLAlchemy 2.0** | 采用异步驱动（aiomysql）处理业务逻辑 |
| **缓存** | **Redis** | 缓存热点数据，避免重复查询数据库 |
| **搜索引擎** | **Elasticsearch 8.x** | 全文检索 + 向量搜索 (dense_vector)，支持语义相似匹配 |
| **Embedding** | **Sentence-Transformers** | 本地模型生成向量，用于语义搜索 |

## 缓存策略

本系统使用 Redis 实现多级缓存策略，减少数据库查询压力：

### 缓存内容

| 缓存类型 | Key 模式 | TTL | 说明 |
|----------|----------|-----|------|
| **类目树** | `faq:category:tree:{env}` | 2小时 | 按环境缓存完整类目树 |
| **FAQ详情** | `faq:detail:{id}` | 30分钟 | FAQ基本信息和元数据 |
| **FAQ答案** | `faq:solutions:{id}` | 30分钟 | FAQ的所有答案视角 |
| **搜索结果** | `faq:search:{hash}` | 5分钟 | 语义搜索结果缓存 |
| **渠道配置** | `faq:channel:{code}` | 24小时 | 渠道配置信息 |
| **热门FAQ** | `faq:hot:faqs:{limit}` | 10分钟 | 热门FAQ列表 |
| **导航目录** | `faq:nav:tree:{env}` | 2小时 | 自助导航目录 |

### 缓存失效策略

- **主动失效**: 在创建、更新、删除 FAQ 或类目时，主动清除相关缓存
- **TTL过期**: 所有缓存设置最大过期时间
- **版本控制**: 可通过配置 version 强制刷新所有缓存

### 缓存使用示例

```python
from app.cache import cached, invalidate_cache, CacheKeys, CacheTTL, cache_manager

# 使用装饰器缓存
@cached(key_prefix="category:tree:", ttl=CacheTTL.CATEGORY_TREE)
async def get_category_tree(env: str):
    # 只有缓存不存在时才会执行
    return await database.query_categories(env)

# 失效缓存
@ invalidate_cache(key_pattern="category:*")
async def update_category(category_id: int, data):
    await database.update_category(category_id, data)
    # 自动清除所有 category 相关缓存
```

### 缓存配置

在 `config.yaml` 中配置：

```yaml
database:
  redis:
    enabled: true
    host: "localhost"
    port: 6379

  cache:
    enabled: true
    default_ttl: 3600
    category_tree_ttl: 7200
    faq_detail_ttl: 1800
    search_result_ttl: 300
```

## 数据库设计

1. 类目管理表 (categories)

| 字段名 | 类型 | 描述 |
| --- | --- | --- |
| **id** | Integer (PK) | 类目唯一标识 |
| **env** | Enum | **环境标识**：`TEST` (测试), `PROD` (正式) |
| **name** | String(64) | 类目名称 |
| **parent_id** | Integer | 父类目 ID (关联本表的 id) |
| **level** | Integer | 层级 (1-一级, 2-二级) |
| **original_id** | Integer | **溯源 ID**：正式环境记录指向其来源的测试环境 ID |
| **creator/modifier** | String(64) | 操作人审计 |
| **created/updated_at** | DateTime | 时间审计 |

2. FAQ 主表 (faqs)

| 字段名 | 类型 | 描述 |
| --- | --- | --- |
| **id** | Integer (PK) | FAQ 唯一标识 |
| **env** | Enum | **环境标识**：`TEST` (测试), `PROD` (正式) |
| **category_id** | Integer | 关联 `categories.id` (需对应相同 env) |
| **title** | String(255) | **标准问标题** |
| **similar_queries** | JSONB | 相似问列表 (List of Strings) |
| **related_ids** | JSONB | 关联问题 ID 列表 |
| **tags** | JSONB | 标签名称列表 |
| **status** | Enum | 生效状态：`ENABLE` (生效), `DISABLE` (失效) |
| **original_id** | Integer | **溯源 ID**：正式环境记录指向对应的测试环境 ID |
| **is_permanent** | Boolean | 是否永久生效 |
| **start/end_time** | DateTime | 生效时间区间 |
| **creator/modifier** | String(64) | 创建人和修改人 |
| **created/updated_at** | DateTime | 创建时间和更新时间 |

3. FAQ 答案视角表 (faq_solutions)

| 字段名 | 类型 | 描述 |
| --- | --- | --- |
| **id** | Integer (PK) | 答案唯一标识 |
| **env** | Enum | **环境标识**：`TEST` (测试), `PROD` (正式) |
| **faq_id** | Integer | 关联 `faqs.id` |
| **perspective** | String(50) | **视角** (如：默认、微信、App) |
| **answer_type** | Enum | 类型：`TEXT` (纯文本), `RICH` (富文本), `CARD` (卡片) |
| **content** | Text | 答案具体内容 |
| **is_default** | Boolean | 是否为该问题的默认答案 |
| **creator/modifier** | String(64) | 创建人和修改人 |
| **created/updated_at** | DateTime | 创建时间和更新时间 |

## 核心流程

1. 知识生产与管理模块

| 功能子模块 | 接口路径 (Endpoint) | 方法 | 描述 |
| --- | --- | --- | --- |
| **类目管理** | `/api/v1/admin/categories` | **POST** | **新建类目**：支持一、二级层级创建。 |
|  | `/api/v1/admin/categories/{id}` | **PUT** | **修改类目**：更新类目名称或排序。 |
|  | `/api/v1/admin/categories/{id}` | **DELETE** | **删除类目**：校验下属是否有FAQ，支持物理/软删除。 |
|  | `/api/v1/admin/categories/tree` | **GET** | **获取类目树**：按环境返回完整的层级结构。 |
| **FAQ 核心** | `/api/v1/admin/faqs` | **POST** | **新建FAQ**：录入标准问及相似问法。 |
|  | `/api/v1/admin/faqs/{id}` | **PUT** | **修改FAQ**：更新基础问法与关联问题设置。 |
|  | `/api/v1/admin/faqs/{id}` | **DELETE** | **删除FAQ**：仅限删除指定环境下的单条记录。 |
|  | `/api/v1/admin/faqs/status` | **PATCH** | **状态切换**：快速控制FAQ的启用或失效。 |
| **批量操作** | `/api/v1/admin/faqs/batch` | **POST** | **数据转换**：支持3000条导入/50000条导出限制。 |


2. 对外服务模块 (Bot Service)

| 功能子模块 | 接口路径 (Endpoint) | 方法 | 描述 |
| --- | --- | --- | --- |
| **智能问答** | `/api/v1/service/ask` | **POST** | **语义检索**：多路召回正式环境中最匹配的答案。 |
| **自助导航** | `/api/v1/service/nav` | **GET** | **目录浏览**：展示正式环境的类目引导供用户点击。 |
| **详情获取** | `/api/v1/service/faq/{id}` | **GET** | **直接获取**：根据FAQ ID获取正式环境详情。 |
| **关联推荐** | `/api/v1/service/recommend` | **GET** | **相关知识**：返回该问题关联的最多5条其他知识。 |

