"""
FAQ API Router
FAQ 管理接口
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, func, and_
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import FAQ, Category, FAQSolution, FAQVersion
from app.datamodels import (
    FAQCreate,
    FAQUpdate,
    FAQ as FAQModel,
    FAQWithSolutions,
    FAQSolutionCreate,
    FAQSolution as FAQSolutionModel,
    FAQStatusUpdate,
)
from app.exceptions import success, created, no_content, bad_request, not_found, conflict
from app.services.embedding import FAQIndexService

router = APIRouter(prefix="/admin/faqs", tags=["FAQ管理"])


# ============================================
# Dependencies
# ============================================

def get_current_user_dep(current_user: Optional[CurrentUser] = Depends(get_current_user_optional)) -> Optional[CurrentUser]:
    """获取当前用户 (可选)"""
    return current_user


# ============================================
# FAQ CRUD
# ============================================

@router.post("")
def create_faq(
    data: FAQCreate,
    db: Session = Depends(get_db)
):
    """创建 FAQ"""
    # 验证类目存在
    category = db.get(Category, data.category_id)
    if not category:
        return not_found(msg="类目不存在")
    if category.env != data.env.value:
        return bad_request(msg="类目环境与FAQ环境不匹配")

    faq = FAQ(
        env=data.env.value,
        category_id=data.category_id,
        title=data.title,
        similar_queries=data.similar_queries or [],
        related_ids=data.related_ids or [],
        tags=data.tags or [],
        status=data.status.value,
        is_permanent=data.is_permanent,
        start_time=data.start_time,
        end_time=data.end_time,
        creator=data.creator,
    )
    db.add(faq)
    db.flush()
    db.refresh(faq)

    return created(data=FAQModel.model_validate(faq).model_dump())


@router.get("")
def list_faqs(
    env: str = Query("TEST", description="环境"),
    category_id: Optional[int] = Query(None, description="类目ID"),
    status: Optional[str] = Query(None, description="状态"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    db: Session = Depends(get_db)
):
    """获取 FAQ 列表"""
    conditions = [FAQ.env == env]

    if category_id:
        conditions.append(FAQ.category_id == category_id)
    if status:
        conditions.append(FAQ.status == status)

    # 查询总数
    count_stmt = select(func.count(FAQ.id)).where(and_(*conditions))
    total_result = db.execute(count_stmt)
    total = total_result.scalar()

    # 分页查询
    stmt = select(FAQ).where(and_(*conditions)).order_by(FAQ.id.desc())
    stmt = stmt.offset((page - 1) * page_size).limit(page_size)
    result = db.execute(stmt)
    faqs = result.scalars().all()

    items = [FAQModel.model_validate(f).model_dump() for f in faqs]

    return success(data={
        "items": items,
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": (total + page_size - 1) // page_size if page_size > 0 else 0
    })


@router.get("/{faq_id}")
def get_faq(
    faq_id: int,
    db: Session = Depends(get_db)
):
    """获取 FAQ 详情（含答案）"""
    faq = db.get(FAQ, faq_id)
    if not faq:
        return not_found(msg="FAQ不存在")

    # 加载答案
    stmt = select(FAQSolution).where(FAQSolution.faq_id == faq_id)
    result = db.execute(stmt)
    solutions = result.scalars().all()

    solutions_data = [
        FAQSolutionModel(
            id=s.id,
            env=s.env,
            faq_id=s.faq_id,
            perspective=s.perspective,
            answer_type=s.answer_type,
            content=s.content,
            is_default=s.is_default,
            sort=s.sort,
            creator=s.creator,
            modifier=s.modifier,
            created_at=s.created_at,
            updated_at=s.updated_at,
        ).model_dump()
        for s in solutions
    ]

    faq_data = FAQModel.model_validate(faq).model_dump()
    faq_data["solutions"] = solutions_data

    return success(data=faq_data)


@router.put("/{faq_id}")
def update_faq(
    faq_id: int,
    data: FAQUpdate,
    db: Session = Depends(get_db)
):
    """更新 FAQ"""
    faq = db.get(FAQ, faq_id)
    if not faq:
        return not_found(msg="FAQ不存在")

    # 记录版本历史
    version_stmt = select(func.max(FAQVersion.version)).where(FAQVersion.faq_id == faq_id)
    version_result = db.execute(version_stmt)
    current_version = version_result.scalar() or 0

    version = FAQVersion(
        faq_id=faq_id,
        version=current_version + 1,
        diff_content={
            "old_title": faq.title,
            "new_title": data.title,
        },
        operator=data.modifier or faq.creator,
    )
    db.add(version)

    # 更新字段
    if data.title is not None:
        faq.title = data.title
    if data.category_id is not None:
        faq.category_id = data.category_id
    if data.similar_queries is not None:
        faq.similar_queries = data.similar_queries
    if data.related_ids is not None:
        faq.related_ids = data.related_ids
    if data.tags is not None:
        faq.tags = data.tags
    if data.is_permanent is not None:
        faq.is_permanent = data.is_permanent
    if data.start_time is not None:
        faq.start_time = data.start_time
    if data.end_time is not None:
        faq.end_time = data.end_time
    if data.modifier is not None:
        faq.modifier = data.modifier

    db.flush()
    db.refresh(faq)

    return success(data=FAQModel.model_validate(faq).model_dump())


@router.patch("/status")
def update_faq_status(
    data: FAQStatusUpdate,
    db: Session = Depends(get_db)
):
    """批量更新 FAQ 状态"""
    stmt = select(FAQ).where(FAQ.id.in_(data.faq_ids))
    result = db.execute(stmt)
    faqs = result.scalars().all()

    for faq in faqs:
        faq.status = data.status.value
        faq.modifier = data.modifier

    db.flush()

    return success(data={"updated_count": len(faqs)}, msg=f"成功更新 {len(faqs)} 条记录")


@router.delete("/{faq_id}")
def delete_faq(
    faq_id: int,
    env: str = Query("TEST", description="环境"),
    db: Session = Depends(get_db)
):
    """删除 FAQ"""
    stmt = select(FAQ).where(and_(FAQ.id == faq_id, FAQ.env == env))
    result = db.execute(stmt)
    faq = result.scalar_one_or_none()

    if not faq:
        return not_found(msg="FAQ不存在")

    db.delete(faq)
    db.flush()

    return no_content(msg="删除成功")


# ============================================
# FAQ Solutions
# ============================================

@router.post("/{faq_id}/solutions")
def create_solution(
    faq_id: int,
    data: FAQSolutionCreate,
    db: Session = Depends(get_db)
):
    """创建 FAQ 答案"""
    faq = db.get(FAQ, faq_id)
    if not faq:
        return not_found(msg="FAQ不存在")

    # 如果设为默认答案，取消其他默认答案
    if data.is_default:
        stmt = select(FAQSolution).where(
            and_(FAQSolution.faq_id == faq_id, FAQSolution.is_default == True)
        )
        result = db.execute(stmt)
        for solution in result.scalars():
            solution.is_default = False

    solution = FAQSolution(
        env=data.env.value,
        faq_id=faq_id,
        perspective=data.perspective,
        answer_type=data.answer_type.value,
        content=data.content,
        is_default=data.is_default,
        sort=data.sort,
        creator=data.creator,
    )
    db.add(solution)
    db.flush()
    db.refresh(solution)

    return created(data=FAQSolutionModel.model_validate(solution).model_dump())


@router.get("/{faq_id}/solutions")
def list_solutions(
    faq_id: int,
    db: Session = Depends(get_db)
):
    """获取 FAQ 的所有答案"""
    faq = db.get(FAQ, faq_id)
    if not faq:
        return not_found(msg="FAQ不存在")

    stmt = select(FAQSolution).where(FAQSolution.faq_id == faq_id).order_by(FAQSolution.sort)
    result = db.execute(stmt)
    solutions = result.scalars().all()

    items = [FAQSolutionModel.model_validate(s).model_dump() for s in solutions]
    return success(data={"items": items, "total": len(items)})


@router.get("/{faq_id}/versions")
def list_faq_versions(
    faq_id: int,
    db: Session = Depends(get_db)
):
    """获取 FAQ 版本历史"""
    from app.datamodels import FAQVersion as FAQVersionModel

    faq = db.get(FAQ, faq_id)
    if not faq:
        return not_found(msg="FAQ不存在")

    stmt = select(FAQVersion).where(FAQVersion.faq_id == faq_id).order_by(FAQVersion.version.desc())
    result = db.execute(stmt)
    versions = result.scalars().all()

    items = [
        FAQVersionModel(
            id=v.id,
            faq_id=v.faq_id,
            version=v.version,
            diff_content=v.diff_content,
            operator=v.operator,
            created_at=v.created_at,
        ).model_dump()
        for v in versions
    ]

    return success(data={"items": items, "total": len(items)})


# ============================================
# Index Management
# ============================================

@router.post("/reindex")
def reindex_faqs(
    env: str = Query("PROD", description="环境"),
    db: Session = Depends(get_db)
):
    """
    重建 FAQ 索引

    将数据库中的 FAQ 数据重新索引到 Elasticsearch，
    包括生成向量并存储到 ES 的 dense_vector 字段中。
    """
    result = FAQIndexService.reindex_all(env=env, db=db)
    return success(data=result, msg=f"索引重建完成: 成功 {result['success_count']}, 失败 {result['failed_count']}")


@router.post("/{faq_id}/index")
def index_single_faq(
    faq_id: int,
    db: Session = Depends(get_db)
):
    """索引单个 FAQ 到 ES"""
    try:
        FAQIndexService.index_faq_from_db(faq_id, db)
        return success(msg=f"FAQ {faq_id} 索引成功")
    except ValueError as e:
        return not_found(msg=str(e))
    except Exception as e:
        return bad_request(msg=f"索引失败: {str(e)}")


@router.delete("/{faq_id}/index")
def delete_faq_index(
    faq_id: int,
):
    """从 ES 索引中删除 FAQ"""
    try:
        FAQIndexService.delete_faq(faq_id)
        return success(msg=f"FAQ {faq_id} 索引已删除")
    except Exception as e:
        return bad_request(msg=f"删除索引失败: {str(e)}")
