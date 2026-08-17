"""
Service API Router
对外服务模块 - Bot Service
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, and_, func
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import FAQ, Category, FAQSolution
from app.datamodels import (
    AskRequest,
    SearchResult,
    RecommendItem,
    CategoryTreeNode,
)
from app.exceptions import success, not_found, bad_request
from app.services.elasticsearch import ESSearch
from app.services.embedding import EmbeddingClient, FAQIndexService

router = APIRouter(prefix="/service", tags=["对外服务"])


# ============================================
# Smart Q&A
# ============================================

@router.post("/ask")
def ask_question(
    request: AskRequest,
    db: Session = Depends(get_db) # 数据库会话管理
):
    """
    智能问答 - 语义检索

    多路召回策略:
    1. 向量搜索 - 语义相似度匹配 (ES dense_vector)
    2. 关键词搜索 - ES 全文检索 (IK 分词)
    3. 混合搜索 - 结合向量和关键词 (RRF 融合)

    Args:
        question: 用户问题
        channel: 渠道 (default, wechat, app, web)
        top_k: 返回数量
    """
    from datetime import datetime

    # 1. 生成问题的向量表示
    query_vector = EmbeddingClient.encode_one(request.question)

    # 2. 混合搜索 (向量 + 关键词)
    search_results = ESSearch.hybrid_search(
        query=request.question,
        query_vector=query_vector,
        env="PROD",
        status="ENABLE",
        top_k=request.top_k,
        keyword_weight=0.4,
        vector_weight=0.6
    )

    # 3. 如果混合搜索结果不够，补充向量搜索
    if len(search_results) < request.top_k:
        remaining = request.top_k - len(search_results)
        existing_ids = [r["faq_id"] for r in search_results]

        vector_results = ESSearch.search_by_vector(
            query_vector=query_vector,
            env="PROD",
            status="ENABLE",
            top_k=remaining,
        )

        for vr in vector_results:
            if vr["faq_id"] not in existing_ids:
                search_results.append(vr)
                existing_ids.append(vr["faq_id"])

    # 4. 如果还不够，补充关键词搜索
    if len(search_results) < request.top_k:
        remaining = request.top_k - len(search_results)

        keyword_results = ESSearch.search_by_keyword(
            query=request.question,
            env="PROD",
            status="ENABLE",
            top_k=remaining,
        )

        for kr in keyword_results:
            if kr["faq_id"] not in existing_ids:
                search_results.append(kr)

    # 5. 构建响应
    answers = []
    for result in search_results:
        faq_id = result["faq_id"]

        # 从数据库获取答案
        solution = _get_solution_by_channel(db, faq_id, request.channel)

        if solution:
            answers.append({
                "faq_id": faq_id,
                "title": result["title"],
                 "content": result.get("content", ""),
                "score": result.get("score", 0),
                "solution": {
                    "perspective": solution.perspective,
                    "answer_type": solution.answer_type,
                    "content": solution.content,
                },
                "highlight": result.get("highlight", {}),
            })

    return success(data={"answers": answers, "total": len(answers)})


def _get_solution_by_channel(db: Session, faq_id: int, channel: str):
    """获取指定渠道的答案"""
    # 优先查询对应渠道的答案，没有则使用默认答案
    stmt = select(FAQSolution).where(
        and_(
            FAQSolution.faq_id == faq_id,
            FAQSolution.perspective == channel
        )
    )
    result = db.execute(stmt)
    solution = result.scalar_one_or_none()

    if not solution:
        # 获取默认答案
        stmt = select(FAQSolution).where(
            and_(
                FAQSolution.faq_id == faq_id,
                FAQSolution.is_default == True
            )
        )
        result = db.execute(stmt)
        solution = result.scalar_one_or_none()

    return solution


@router.get("/search")
def search_faqs(
    q: str = Query(..., min_length=1, max_length=500, description="搜索关键词"),
    channel: str = Query("default", description="渠道"),
    limit: int = Query(10, ge=1, le=50, description="返回数量"),
    db: Session = Depends(get_db)
):
    """关键词搜索 FAQ (仅使用 ES 关键词检索)"""
    # 使用 ES 关键词搜索
    results = ESSearch.search_by_keyword(
        query=q,
        env="PROD",
        status="ENABLE",
        top_k=limit,
    )

    answers = []
    for result in results:
        solution = _get_solution_by_channel(db, result["faq_id"], channel)

        answers.append({
            "faq_id": result["faq_id"],
            "title": result["title"],
            "content": result.get("content", ""),
            "score": result.get("score", 0),
            "solution": {
                "perspective": solution.perspective if solution else None,
                "content": solution.content if solution else None,
            } if solution else None,
            "highlight": result.get("highlight", {}),
        })

    return success(data={"results": answers, "total": len(answers)})


# ============================================
# Navigation
# ============================================

@router.get("/nav")
def get_navigation(
    db: Session = Depends(get_db)
):
    """获取自助导航目录 - PROD环境"""
    env = "PROD"

    # 查询所有启用的类目
    stmt = select(Category).where(Category.env == env).order_by(Category.level, Category.id)
    result = db.execute(stmt)
    categories = result.scalars().all()

    # 构建树形结构
    category_map = {c.id: c for c in categories}
    tree_nodes = {}

    for cat in categories:
        node = CategoryTreeNode(
            id=cat.id,
            name=cat.name,
            level=cat.level,
            children=[]
        )

        if cat.parent_id:
            parent = category_map.get(cat.parent_id)
            if parent:
                parent_node = tree_nodes.get(parent.id)
                if parent_node:
                    parent_node.children.append(node)
        else:
            tree_nodes[cat.id] = node

    tree_data = [tree_nodes[c.id].model_dump() for c in categories if c.parent_id is None]
    return success(data=tree_data)


# ============================================
# FAQ Detail
# ============================================

@router.get("/faq/{faq_id}")
def get_faq_detail(
    faq_id: int,
    channel: str = Query("default", description="渠道"),
    db: Session = Depends(get_db)
):
    """获取 FAQ 详情"""
    faq = db.get(FAQ, faq_id)
    if not faq:
        return not_found(msg="FAQ不存在")

    if faq.env != "PROD":
        return bad_request(msg="只支持查询正式环境FAQ")

    # 获取答案
    solution = _get_solution_by_channel(db, faq_id, channel)

    data = {
        "id": faq.id,
        "title": faq.title,
        "similar_queries": faq.similar_queries,
        "tags": faq.tags,
        "solution": {
            "perspective": solution.perspective if solution else None,
            "answer_type": solution.answer_type if solution else None,
            "content": solution.content if solution else None,
        } if solution else None,
        "related_ids": faq.related_ids,
        "created_at": faq.created_at.isoformat() if faq.created_at else None,
        "updated_at": faq.updated_at.isoformat() if faq.updated_at else None,
    }

    return success(data=data)


# ============================================
# Recommendation
# ============================================

@router.get("/recommend")
def get_recommend(
    faq_id: int = Query(..., description="FAQ ID"),
    limit: int = Query(5, ge=1, le=10, description="返回数量"),
    db: Session = Depends(get_db)
):
    """获取关联推荐 FAQ"""
    # 获取当前 FAQ
    faq = db.get(FAQ, faq_id)
    if not faq:
        return not_found(msg="FAQ不存在")

    recommended_faqs = []

    # 1. 通过 related_ids 获取关联 FAQ
    if faq.related_ids:
        stmt = select(FAQ).where(
            and_(
                FAQ.id.in_(faq.related_ids),
                FAQ.env == "PROD",
                FAQ.status == "ENABLE"
            )
        ).limit(limit)
        result = db.execute(stmt)
        related_faqs = result.scalars().all()

        for f in related_faqs:
            recommended_faqs.append(RecommendItem(
                faq_id=f.id,
                title=f.title,
                score=1.0
            ).model_dump())

    # 2. 如果不够，通过同分类获取
    if len(recommended_faqs) < limit:
        remaining = limit - len(recommended_faqs)
        existing_ids = [r["faq_id"] for r in recommended_faqs] + [faq_id]

        stmt = select(FAQ).where(
            and_(
                FAQ.category_id == faq.category_id,
                FAQ.id.not_in(existing_ids),
                FAQ.env == "PROD",
                FAQ.status == "ENABLE"
            )
        ).limit(remaining)

        result = db.execute(stmt)
        same_category_faqs = result.scalars().all()

        for f in same_category_faqs:
            recommended_faqs.append(RecommendItem(
                faq_id=f.id,
                title=f.title,
                score=0.8
            ).model_dump())

    return success(data={"faqs": recommended_faqs[:limit], "total": len(recommended_faqs[:limit])})


# ============================================
# Statistics
# ============================================

@router.get("/stats")
def get_statistics(
    db: Session = Depends(get_db)
):
    """获取系统统计信息"""
    # FAQ 统计
    faq_count_stmt = select(func.count(FAQ.id)).where(FAQ.env == "PROD")
    result = db.execute(faq_count_stmt)
    total_faqs = result.scalar()

    enabled_count_stmt = select(func.count(FAQ.id)).where(
        and_(FAQ.env == "PROD", FAQ.status == "ENABLE")
    )
    result = db.execute(enabled_count_stmt)
    enabled_faqs = result.scalar()

    # 类目统计
    category_count_stmt = select(func.count(Category.id)).where(Category.env == "PROD")
    result = db.execute(category_count_stmt)
    total_categories = result.scalar()

    data = {
        "total_faqs": total_faqs,
        "enabled_faqs": enabled_faqs,
        "disabled_faqs": total_faqs - enabled_faqs,
        "total_categories": total_categories,
    }

    return success(data=data)
