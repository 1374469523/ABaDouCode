"""
Database Models
数据库模型定义
"""
from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum as SQLEnum,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# ============================================
# Base
# ============================================

class Base(DeclarativeBase):
    """数据库基类"""
    pass


# ============================================
# Enums
# ============================================

class EnvEnum(str, Enum):
    """环境枚举"""
    TEST = "TEST"
    PROD = "PROD"


class FAQStatusEnum(str, Enum):
    """FAQ 状态枚举"""
    ENABLE = "ENABLE"
    DISABLE = "DISABLE"


class AnswerTypeEnum(str, Enum):
    """答案类型枚举"""
    TEXT = "TEXT"
    RICH = "RICH"
    CARD = "CARD"


class SyncTypeEnum(str, Enum):
    """同步类型枚举"""
    FULL = "FULL"
    INCREMENTAL = "INCREMENTAL"


class SyncStatusEnum(str, Enum):
    """同步状态枚举"""
    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


# ============================================
# Models
# ============================================

class Category(Base):
    """类目管理表"""
    __tablename__ = "categories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    env: Mapped[EnvEnum] = mapped_column(SQLEnum(EnvEnum), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    parent_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("categories.id"), nullable=True)
    level: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-一级, 2-二级
    original_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # 溯源 ID
    creator: Mapped[str] = mapped_column(String(64), nullable=False)
    modifier: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    parent: Mapped[Optional["Category"]] = relationship("Category", remote_side=[id], backref="children")
    faqs: Mapped[list["FAQ"]] = relationship("FAQ", back_populates="category")

    __table_args__ = (
        Index("idx_env_parent", "env", "parent_id"),
        UniqueConstraint("env", "name", "parent_id", name="uq_category_env_name_parent"),
    )


class FAQ(Base):
    """FAQ 主表"""
    __tablename__ = "faqs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    env: Mapped[EnvEnum] = mapped_column(SQLEnum(EnvEnum), nullable=False, index=True)
    category_id: Mapped[int] = mapped_column(Integer, ForeignKey("categories.id"), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    similar_queries: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    related_ids: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    status: Mapped[FAQStatusEnum] = mapped_column(SQLEnum(FAQStatusEnum), default=FAQStatusEnum.DISABLE)
    original_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    is_permanent: Mapped[bool] = mapped_column(Boolean, default=True)
    start_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    end_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    creator: Mapped[str] = mapped_column(String(64), nullable=False)
    modifier: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    category: Mapped["Category"] = relationship("Category", back_populates="faqs")
    solutions: Mapped[list["FAQSolution"]] = relationship("FAQSolution", back_populates="faq", cascade="all, delete-orphan")
    versions: Mapped[list["FAQVersion"]] = relationship("FAQVersion", back_populates="faq", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_env_category_status", "env", "category_id", "status"),
        Index("idx_env_original_id", "env", "original_id"),
    )


class FAQSolution(Base):
    """FAQ 答案视角表"""
    __tablename__ = "faq_solutions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    env: Mapped[EnvEnum] = mapped_column(SQLEnum(EnvEnum), nullable=False, index=True)
    faq_id: Mapped[int] = mapped_column(Integer, ForeignKey("faqs.id"), nullable=False)
    perspective: Mapped[str] = mapped_column(String(50), nullable=False)  # default, wechat, app, web
    answer_type: Mapped[AnswerTypeEnum] = mapped_column(SQLEnum(AnswerTypeEnum), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)
    sort: Mapped[int] = mapped_column(Integer, default=0)
    creator: Mapped[str] = mapped_column(String(64), nullable=False)
    modifier: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    faq: Mapped["FAQ"] = relationship("FAQ", back_populates="solutions")

    __table_args__ = (
        UniqueConstraint("env", "faq_id", "perspective", name="uq_solution_env_faq_perspective"),
    )


class Channel(Base):
    """渠道配置表"""
    __tablename__ = "channels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class SyncRecord(Base):
    """发布记录表"""
    __tablename__ = "sync_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    operator: Mapped[str] = mapped_column(String(64), nullable=False)
    sync_type: Mapped[SyncTypeEnum] = mapped_column(SQLEnum(SyncTypeEnum), nullable=False)
    source_env: Mapped[EnvEnum] = mapped_column(SQLEnum(EnvEnum), nullable=False)
    target_env: Mapped[EnvEnum] = mapped_column(SQLEnum(EnvEnum), nullable=False)
    status: Mapped[SyncStatusEnum] = mapped_column(SQLEnum(SyncStatusEnum), default=SyncStatusEnum.PENDING)
    detail: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class FAQVersion(Base):
    """FAQ 版本历史表"""
    __tablename__ = "faq_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    faq_id: Mapped[int] = mapped_column(Integer, ForeignKey("faqs.id"), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    diff_content: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    operator: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    faq: Mapped["FAQ"] = relationship("FAQ", back_populates="versions")

    __table_args__ = (
        UniqueConstraint("faq_id", "version", name="uq_faq_version"),
    )
