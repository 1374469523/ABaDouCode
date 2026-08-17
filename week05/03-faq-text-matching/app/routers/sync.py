"""
Sync API Router
发布同步模块 - TEST → PROD
"""
from typing import List
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, and_
from sqlalchemy.orm import Session
from datetime import datetime

from app.database import get_db
from app.models import Category, FAQ, FAQSolution, SyncRecord
from app.datamodels import (
    SyncExecute,
    SyncRecord as SyncRecordModel,
    SyncPreview,
    SyncTypeEnum,
    SyncStatusEnum,
)
from app.exceptions import success, not_found, server_error, conflict

router = APIRouter(prefix="/admin/sync", tags=["发布同步"])


# ============================================
# Preview
# ============================================

@router.post("/preview")
def preview_sync(
    request: SyncExecute,
    db: Session = Depends(get_db)
):
    """预览同步差异 - TEST → PROD"""

    # 1. 获取 TEST 环境的所有数据
    test_categories_stmt = select(Category).where(Category.env == "TEST")
    test_categories_result = db.execute(test_categories_stmt)
    test_categories = test_categories_result.scalars().all()

    test_faqs_stmt = select(FAQ).where(FAQ.env == "TEST")
    test_faqs_result = db.execute(test_faqs_stmt)
    test_faqs = test_faqs_result.scalars().all()

    # 2. 获取 PROD 环境的数据
    prod_categories_stmt = select(Category).where(Category.env == "PROD")
    prod_categories_result = db.execute(prod_categories_stmt)
    prod_categories = prod_categories_result.scalars().all()

    prod_faqs_stmt = select(FAQ).where(FAQ.env == "PROD")
    prod_faqs_result = db.execute(prod_faqs_stmt)
    prod_faqs = prod_faqs_result.scalars().all()

    # 3. 构建 ID 映射
    prod_category_map = {c.original_id: c for c in prod_categories if c.original_id}
    prod_faq_map = {f.original_id: f for f in prod_faqs if f.original_id}

    # 4. 计算差异
    added_categories = []
    updated_categories = []
    added_faqs = []
    updated_faqs = []
    deleted_faq_ids = []

    # 类目差异
    for test_cat in test_categories:
        prod_cat = prod_category_map.get(test_cat.id)
        if prod_cat:
            if test_cat.name != prod_cat.name:
                updated_categories.append({
                    "id": test_cat.id,
                    "name": test_cat.name,
                    "level": test_cat.level,
                    "old_name": prod_cat.name,
                })
        else:
            added_categories.append({
                "id": test_cat.id,
                "name": test_cat.name,
                "level": test_cat.level,
            })

    # FAQ 差异
    test_faq_map = {f.id: f for f in test_faqs}
    for test_faq in test_faqs:
        prod_faq = prod_faq_map.get(test_faq.id)
        if prod_faq:
            if (test_faq.title != prod_faq.title or
                test_faq.similar_queries != prod_faq.similar_queries or
                test_faq.status != prod_faq.status):
                updated_faqs.append({
                    "id": test_faq.id,
                    "title": test_faq.title,
                    "old_title": prod_faq.title,
                    "status": test_faq.status,
                })
        else:
            added_faqs.append({
                "id": test_faq.id,
                "title": test_faq.title,
                "category_id": test_faq.category_id,
            })

    # 检查 PROD 有但 TEST 已删除的
    for prod_faq in prod_faqs:
        if prod_faq.original_id and prod_faq.original_id not in test_faq_map:
            deleted_faq_ids.append(prod_faq.id)

    total = len(added_categories) + len(updated_categories) + len(added_faqs) + len(updated_faqs) + len(deleted_faq_ids)

    data = {
        "added_categories": added_categories,
        "added_categories_count": len(added_categories),
        "updated_categories": updated_categories,
        "updated_categories_count": len(updated_categories),
        "added_faqs": added_faqs,
        "added_faqs_count": len(added_faqs),
        "updated_faqs": updated_faqs,
        "updated_faqs_count": len(updated_faqs),
        "deleted_faq_ids": deleted_faq_ids,
        "deleted_faqs_count": len(deleted_faq_ids),
        "total_count": total,
    }

    return success(data=data)


# ============================================
# Execute
# ============================================

@router.post("/execute")
def execute_sync(
    request: SyncExecute,
    db: Session = Depends(get_db)
):
    """执行同步 - TEST → PROD"""

    # 创建同步记录
    sync_record = SyncRecord(
        operator=request.operator,
        sync_type=request.sync_type.value,
        source_env="TEST",
        target_env="PROD",
        status=SyncStatusEnum.PENDING,
    )
    db.add(sync_record)
    db.flush()

    try:
        # 1. 同步类目
        _sync_categories(db)

        # 2. 同步 FAQ
        _sync_faqs(db)

        # 更新同步状态
        sync_record.status = SyncStatusEnum.SUCCESS
        sync_record.detail = {"message": "Sync completed successfully"}

    except Exception as e:
        sync_record.status = SyncStatusEnum.FAILED
        sync_record.detail = {"error": str(e)}
        return server_error(msg=f"同步失败: {str(e)}")

    db.flush()
    db.refresh(sync_record)

    return success(data={
        "id": sync_record.id,
        "status": sync_record.status.value,
        "operator": sync_record.operator,
        "created_at": sync_record.created_at.isoformat() if sync_record.created_at else None,
    }, msg="同步成功")


def _sync_categories(db: Session):
    """同步类目"""
    # 获取 TEST 环境类目
    test_stmt = select(Category).where(Category.env == "TEST")
    test_result = db.execute(test_stmt)
    test_categories = test_result.scalars().all()

    # 获取 PROD 环境类目
    prod_stmt = select(Category).where(Category.env == "PROD")
    prod_result = db.execute(prod_stmt)
    prod_categories = prod_result.scalars().all()

    # 构建映射
    prod_category_map = {c.original_id: c for c in prod_categories if c.original_id}

    for test_cat in test_categories:
        existing = prod_category_map.get(test_cat.id)

        if existing:
            existing.name = test_cat.name
            existing.modifier = test_cat.creator
        else:
            new_cat = Category(
                env="PROD",
                name=test_cat.name,
                parent_id=test_cat.parent_id,
                level=test_cat.level,
                original_id=test_cat.id,
                creator=test_cat.creator,
            )
            db.add(new_cat)


def _sync_faqs(db: Session):
    """同步 FAQ"""
    # 获取 TEST 环境 FAQ
    test_stmt = select(FAQ).where(FAQ.env == "TEST")
    test_result = db.execute(test_stmt)
    test_faqs = test_result.scalars().all()

    # 获取 PROD 环境 FAQ
    prod_stmt = select(FAQ).where(FAQ.env == "PROD")
    prod_result = db.execute(prod_stmt)
    prod_faqs = prod_result.scalars().all()

    # 构建映射
    prod_faq_map = {f.original_id: f for f in prod_faqs if f.original_id}

    for test_faq in test_faqs:
        existing = prod_faq_map.get(test_faq.id)

        if existing:
            existing.title = test_faq.title
            existing.similar_queries = test_faq.similar_queries
            existing.related_ids = test_faq.related_ids
            existing.tags = test_faq.tags
            existing.status = test_faq.status
            existing.modifier = test_faq.creator
            existing.updated_at = datetime.utcnow()

            _sync_solutions(db, test_faq.id, existing.id)
        else:
            new_faq = FAQ(
                env="PROD",
                category_id=test_faq.category_id,
                title=test_faq.title,
                similar_queries=test_faq.similar_queries,
                related_ids=[],
                tags=test_faq.tags,
                status=test_faq.status,
                original_id=test_faq.id,
                is_permanent=test_faq.is_permanent,
                start_time=test_faq.start_time,
                end_time=test_faq.end_time,
                creator=test_faq.creator,
            )
            db.add(new_faq)
            db.flush()

            _sync_solutions(db, test_faq.id, new_faq.id)


def _sync_solutions(db: Session, test_faq_id: int, prod_faq_id: int):
    """同步 FAQ 答案"""
    test_stmt = select(FAQSolution).where(FAQSolution.faq_id == test_faq_id)
    test_result = db.execute(test_stmt)
    test_solutions = test_result.scalars().all()

    prod_stmt = select(FAQSolution).where(FAQSolution.faq_id == prod_faq_id)
    prod_result = db.execute(prod_stmt)
    prod_solutions = prod_result.scalars().all()

    prod_solution_map = {s.perspective: s for s in prod_solutions}

    for test_sol in test_solutions:
        existing = prod_solution_map.get(test_sol.perspective)

        if existing:
            existing.content = test_sol.content
            existing.answer_type = test_sol.answer_type
            existing.is_default = test_sol.is_default
            existing.sort = test_sol.sort
            existing.modifier = test_sol.creator
        else:
            new_sol = FAQSolution(
                env="PROD",
                faq_id=prod_faq_id,
                perspective=test_sol.perspective,
                answer_type=test_sol.answer_type,
                content=test_sol.content,
                is_default=test_sol.is_default,
                sort=test_sol.sort,
                creator=test_sol.creator,
            )
            db.add(new_sol)


# ============================================
# History
# ============================================

@router.get("/history")
def get_sync_history(
    limit: int = Query(20, ge=1, le=100, description="返回数量"),
    db: Session = Depends(get_db)
):
    """获取同步历史记录"""
    from app.datamodels import SyncRecord as SyncRecordModel

    stmt = select(SyncRecord).order_by(SyncRecord.created_at.desc()).limit(limit)
    result = db.execute(stmt)
    records = result.scalars().all()

    items = [
        {
            "id": r.id,
            "operator": r.operator,
            "sync_type": r.sync_type.value if r.sync_type else None,
            "source_env": r.source_env.value if r.source_env else None,
            "target_env": r.target_env.value if r.target_env else None,
            "status": r.status.value if r.status else None,
            "detail": r.detail,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in records
    ]

    return success(data={"items": items, "total": len(items)})
