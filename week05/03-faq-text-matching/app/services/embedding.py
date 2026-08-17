"""
Embedding Service
文本向量转换服务 - 同步版本
"""
from typing import List, Optional
import numpy as np

from app.config import settings


# ============================================
# Embedding Client
# ============================================

class EmbeddingClient:
    """Embedding 客户端"""

    _model = None

    @classmethod
    def get_model(cls):
        """获取 embedding 模型"""
        if cls._model is None:
            if settings.embedding.provider == "local":
                from sentence_transformers import SentenceTransformer
                cls._model = SentenceTransformer(
                    settings.embedding.local.model_name,
                    device=settings.embedding.local.device
                )
            elif settings.embedding.provider == "openai":
                import openai
                openai.api_key = settings.embedding.openai.api_key
                cls._model = "openai"
            else:
                raise ValueError(f"Unsupported embedding provider: {settings.embedding.provider}")
        return cls._model

    @classmethod
    def encode(cls, texts: List[str], batch_size: int = None) -> List[List[float]]:
        """
        将文本转换为向量

        Args:
            texts: 文本列表
            batch_size: 批处理大小

        Returns:
            向量列表
        """
        if not texts:
            return []

        if settings.embedding.provider == "local":
            model = cls.get_model()
            if batch_size is None:
                batch_size = settings.embedding.local.batch_size

            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=settings.embedding.local.normalize_embeddings,
                show_progress_bar=False
            )

            # 转换为列表
            return embeddings.tolist()

        elif settings.embedding.provider == "openai":
            import openai
            model = cls.get_model()

            embeddings = []
            for text in texts:
                response = openai.Embedding.create(
                    model=settings.embedding.openai.model,
                    input=text
                )
                embeddings.append(response["data"][0]["embedding"])

            return embeddings

        else:
            raise ValueError(f"Unsupported embedding provider: {settings.embedding.provider}")

    @classmethod
    def encode_one(cls, text: str) -> List[float]:
        """
        将单个文本转换为向量

        Args:
            text: 文本

        Returns:
            向量
        """
        vectors = cls.encode([text])
        return vectors[0] if vectors else []

    @classmethod
    def close(cls):
        """关闭模型"""
        cls._model = None


# ============================================
# FAQ Index Service
# ============================================

class FAQIndexService:
    """FAQ 索引服务 - 将 FAQ 数据索引到 ES"""

    @staticmethod
    def index_faq(
        faq_id: int,
        title: str,
        content: str,
        similar_queries: List[str] = None,
        tags: List[str] = None,
        category_id: int = None,
        env: str = "PROD",
        status: str = "ENABLE",
        **kwargs
    ):
        """
        索引单个 FAQ 到 ES

        Args:
            faq_id: FAQ ID
            title: 标题
            content: 内容
            similar_queries: 相似问法
            tags: 标签
            category_id: 类目ID
            env: 环境
            status: 状态
        """
        # 生成向量
        title_vector = EmbeddingClient.encode_one(title)
        content_vector = EmbeddingClient.encode_one(content)

        # 合并所有文本生成综合向量
        all_texts = [title]
        if similar_queries:
            all_texts.extend(similar_queries)
        if content:
            all_texts.append(content)
        combined_vector = EmbeddingClient.encode_one(" ".join(all_texts))

        # 索引到 ES
        from app.services.elasticsearch import ESDocument
        ESDocument.index_faq(
            faq_id=faq_id,
            title=title,
            content=content,
            similar_queries=similar_queries,
            tags=tags,
            category_id=category_id,
            env=env,
            status=status,
            title_vector=title_vector,
            content_vector=content_vector,
            **kwargs
        )

    @staticmethod
    def index_faq_from_db(faq_id: int, db):
        """
        从数据库加载 FAQ 并索引到 ES

        Args:
            faq_id: FAQ ID
            db: 数据库会话
        """
        from sqlalchemy import select
        from app.models import FAQ, FAQSolution

        # 获取 FAQ
        faq = db.get(FAQ, faq_id)
        if not faq:
            raise ValueError(f"FAQ {faq_id} not found")

        # 获取答案内容
        solutions_stmt = select(FAQSolution).where(FAQSolution.faq_id == faq_id)
        result = db.execute(solutions_stmt)
        solutions = result.scalars().all()

        # 合并所有答案内容
        content_parts = [s.content for s in solutions if s.content]
        content = "\n".join(content_parts)

        # 索引
        FAQIndexService.index_faq(
            faq_id=faq.id,
            title=faq.title,
            content=content,
            similar_queries=faq.similar_queries,
            tags=faq.tags,
            category_id=faq.category_id,
            env=faq.env,
            status=faq.status,
            is_permanent=faq.is_permanent,
            start_time=faq.start_time,
            end_time=faq.end_time,
        )

    @staticmethod
    def reindex_all(env: str = "PROD", db=None):
        """
        重建所有 FAQ 索引

        Args:
            env: 环境
            db: 数据库会话
        """
        from sqlalchemy import select
        from app.models import FAQ, FAQSolution
        from app.services.elasticsearch import ESDocument

        # 获取所有 FAQ
        stmt = select(FAQ).where(FAQ.env == env)
        result = db.execute(stmt)
        faqs = result.scalars().all()

        success_count = 0
        failed_count = 0

        for faq in faqs:
            try:
                # 获取答案
                solutions_stmt = select(FAQSolution).where(FAQSolution.faq_id == faq.id)
                sol_result = db.execute(solutions_stmt)
                solutions = sol_result.scalars().all()

                content_parts = [s.content for s in solutions if s.content]
                content = "\n".join(content_parts)

                # 生成向量
                title_vector = EmbeddingClient.encode_one(faq.title)
                content_vector = EmbeddingClient.encode_one(content)

                # 批量索引
                ESDocument.index_faq(
                    faq_id=faq.id,
                    title=faq.title,
                    content=content,
                    similar_queries=faq.similar_queries,
                    tags=faq.tags,
                    category_id=faq.category_id,
                    env=faq.env,
                    status=faq.status,
                    title_vector=title_vector,
                    content_vector=content_vector,
                    is_permanent=faq.is_permanent,
                    start_time=faq.start_time,
                    end_time=faq.end_time,
                )
                success_count += 1

            except Exception as e:
                failed_count += 1
                print(f"Failed to index FAQ {faq.id}: {e}")

        return {
            "success_count": success_count,
            "failed_count": failed_count,
            "total": len(faqs)
        }

    @staticmethod
    def delete_faq(faq_id: int):
        """从 ES 删除 FAQ"""
        from app.services.elasticsearch import ESDocument
        ESDocument.delete_faq(faq_id)
