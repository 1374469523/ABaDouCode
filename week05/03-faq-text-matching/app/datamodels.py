"""
Data Models
数据模型 - 使用 Pydantic 定义项目核心数据结构
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict, field_validator


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
# Category Data Models
# ============================================

class CategoryBase(BaseModel):
    """类目基础模型"""
    name: str = Field(..., max_length=64, description="类目名称")
    level: int = Field(..., ge=1, le=2, description="层级：1-一级，2-二级")
    parent_id: Optional[int] = Field(None, description="父类目ID")


class CategoryCreate(CategoryBase):
    """创建类目"""
    env: EnvEnum = Field(..., description="环境标识")
    creator: str = Field(..., max_length=64, description="创建人")

    @field_validator('parent_id')
    @classmethod
    def validate_parent_id(cls, v, info):
        if v is not None and 'level' in info.data:
            if info.data['level'] == 1 and v is not None:
                raise ValueError("一级类目不能有父类目")
            if info.data['level'] == 2 and v is None:
                raise ValueError("二级类目必须有父类目")
        return v


class CategoryUpdate(BaseModel):
    """更新类目"""
    name: Optional[str] = Field(None, max_length=64)
    modifier: Optional[str] = Field(None, max_length=64)


class Category(CategoryBase):
    """类目完整模型"""
    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., description="类目ID")
    env: EnvEnum
    original_id: Optional[int] = Field(None, description="溯源ID")
    creator: str
    modifier: Optional[str]
    created_at: datetime
    updated_at: Optional[datetime]


class CategoryTreeNode(BaseModel):
    """类目树节点"""
    id: int
    name: str
    level: int
    children: List["CategoryTreeNode"] = Field(default_factory=list)


# ============================================
# FAQ Data Models
# ============================================

class FAQBase(BaseModel):
    """FAQ 基础模型"""
    title: str = Field(..., max_length=255, description="标准问标题")
    category_id: int = Field(..., description="关联类目ID")
    similar_queries: Optional[List[str]] = Field(default_factory=list, description="相似问列表")
    related_ids: Optional[List[int]] = Field(default_factory=list, description="关联问题ID列表")
    tags: Optional[List[str]] = Field(default_factory=list, description="标签列表")
    is_permanent: bool = Field(default=True, description="是否永久生效")


class FAQCreate(FAQBase):
    """创建 FAQ"""
    env: EnvEnum
    status: FAQStatusEnum = FAQStatusEnum.DISABLE
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    creator: str = Field(..., max_length=64)


class FAQUpdate(BaseModel):
    """更新 FAQ"""
    title: Optional[str] = Field(None, max_length=255)
    category_id: Optional[int] = None
    similar_queries: Optional[List[str]] = None
    related_ids: Optional[List[int]] = None
    tags: Optional[List[str]] = None
    is_permanent: Optional[bool] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    modifier: Optional[str] = Field(None, max_length=64)


class FAQ(FAQBase):
    """FAQ 完整模型"""
    model_config = ConfigDict(from_attributes=True)

    id: int
    env: EnvEnum
    status: FAQStatusEnum
    original_id: Optional[int]
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    creator: str
    modifier: Optional[str]
    created_at: datetime
    updated_at: Optional[datetime]


class FAQWithSolutions(FAQ):
    """FAQ 包含答案列表"""
    solutions: List["FAQSolution"] = Field(default_factory=list)


# ============================================
# FAQ Solution Data Models
# ============================================

class FAQSolutionBase(BaseModel):
    """FAQ 答案基础模型"""
    perspective: str = Field(..., max_length=50, description="视角：default, wechat, app, web")
    answer_type: AnswerTypeEnum
    content: str = Field(..., description="答案内容")
    is_default: bool = Field(default=False, description="是否为默认答案")
    sort: int = Field(default=0, description="排序权重")


class FAQSolutionCreate(FAQSolutionBase):
    """创建 FAQ 答案"""
    env: EnvEnum
    faq_id: int
    creator: str = Field(..., max_length=64)


class FAQSolutionUpdate(BaseModel):
    """更新 FAQ 答案"""
    content: Optional[str] = None
    is_default: Optional[bool] = None
    sort: Optional[int] = None
    modifier: Optional[str] = Field(None, max_length=64)


class FAQSolution(FAQSolutionBase):
    """FAQ 答案完整模型"""
    model_config = ConfigDict(from_attributes=True)

    id: int
    env: EnvEnum
    faq_id: int
    creator: str
    modifier: Optional[str]
    created_at: datetime
    updated_at: Optional[datetime]


# ============================================
# Channel Data Models
# ============================================

class Channel(BaseModel):
    """渠道模型"""
    model_config = ConfigDict(from_attributes=True)

    id: int
    code: str = Field(..., max_length=32, description="渠道编码")
    name: str = Field(..., max_length=64, description="渠道名称")
    is_active: bool = Field(default=True, description="是否启用")
    created_at: datetime
    updated_at: Optional[datetime]


class ChannelCreate(BaseModel):
    """创建渠道"""
    code: str = Field(..., max_length=32)
    name: str = Field(..., max_length=64)
    is_active: bool = True


# ============================================
# Sync Data Models
# ============================================

class SyncRecord(BaseModel):
    """同步记录模型"""
    model_config = ConfigDict(from_attributes=True)

    id: int
    operator: str
    sync_type: SyncTypeEnum
    source_env: EnvEnum
    target_env: EnvEnum
    status: SyncStatusEnum
    detail: Optional[Dict[str, Any]] = None
    created_at: datetime


class SyncPreview(BaseModel):
    """同步预览结果"""
    added_categories: List[Category] = Field(default_factory=list)
    updated_categories: List[Category] = Field(default_factory=list)
    added_faqs: List[FAQ] = Field(default_factory=list)
    updated_faqs: List[FAQ] = Field(default_factory=list)
    deleted_faq_ids: List[int] = Field(default_factory=list)
    total_count: int = 0


class SyncExecute(BaseModel):
    """执行同步请求"""
    sync_type: SyncTypeEnum
    operator: str = Field(..., max_length=64)


# ============================================
# Version Data Models
# ============================================

class FAQVersion(BaseModel):
    """FAQ 版本历史"""
    model_config = ConfigDict(from_attributes=True)

    id: int
    faq_id: int
    version: int
    diff_content: Optional[Dict[str, Any]] = None
    operator: str
    created_at: datetime


# ============================================
# Search Data Models
# ============================================

class SearchResult(BaseModel):
    """搜索结果"""
    faq_id: int
    title: str
    score: float
    solution: Optional[FAQSolution] = None
    highlight: Optional[str] = None


class SearchResponse(BaseModel):
    """搜索响应"""
    answers: List[SearchResult]
    total: int
    took_ms: int = 0


class AskRequest(BaseModel):
    """智能问答请求"""
    question: str = Field(..., min_length=1, max_length=500)
    channel: str = Field(default="default", max_length=32)
    top_k: int = Field(default=3, ge=1, le=10)


class AskResponse(BaseModel):
    """智能问答响应"""
    answers: List[SearchResult]
    session_id: Optional[str] = None


# ============================================
# Batch Data Models
# ============================================

class BatchImportItem(BaseModel):
    """批量导入项"""
    category_id: int
    title: str
    similar_queries: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    solutions: List[FAQSolutionBase] = Field(default_factory=list)


class BatchImportRequest(BaseModel):
    """批量导入请求"""
    env: EnvEnum
    data: List[BatchImportItem] = Field(..., max_length=3000)


class BatchExportRequest(BaseModel):
    """批量导出请求"""
    env: EnvEnum
    category_id: Optional[int] = None
    status: Optional[FAQStatusEnum] = None
    limit: int = Field(default=1000, le=50000)
    offset: int = Field(default=0)


class BatchResult(BaseModel):
    """批量操作结果"""
    success_count: int = 0
    failed_count: int = 0
    errors: List[str] = Field(default_factory=list)
    data: Optional[List[Dict[str, Any]]] = None


# ============================================
# Recommendation Data Models
# ============================================

class RecommendItem(BaseModel):
    """推荐项"""
    faq_id: int
    title: str
    score: float


class RecommendRequest(BaseModel):
    """推荐请求"""
    faq_id: int
    limit: int = Field(default=5, ge=1, le=10)


class RecommendResponse(BaseModel):
    """推荐响应"""
    faqs: List[RecommendItem]


# ============================================
# Pagination
# ============================================

class PaginationParams(BaseModel):
    """分页参数"""
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)


class PaginatedResponse(BaseModel):
    """分页响应"""
    items: List[Any]
    page: int
    page_size: int
    total: int
    total_pages: int


# ============================================
# Health Check
# ============================================

class HealthStatus(str, Enum):
    """健康状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseModel):
    """组件健康状态"""
    name: str
    status: HealthStatus
    message: Optional[str] = None
    latency_ms: Optional[float] = None


class HealthCheckResponse(BaseModel):
    """健康检查响应"""
    status: HealthStatus
    version: str
    timestamp: datetime
    components: List[ComponentHealth]


# ============================================
# Statistics
# ============================================

class FAQStatistics(BaseModel):
    """FAQ 统计"""
    total_faqs: int = 0
    enabled_faqs: int = 0
    disabled_faqs: int = 0
    total_categories: int = 0
    total_solutions: int = 0
    by_channel: Dict[str, int] = Field(default_factory=dict)
    by_category: Dict[str, int] = Field(default_factory=dict)


class SyncStatistics(BaseModel):
    """同步统计"""
    total_syncs: int = 0
    successful_syncs: int = 0
    failed_syncs: int = 0
    last_sync_time: Optional[datetime] = None


# ============================================
# Common Response
# ============================================

class ResponseCode(int, Enum):
    """响应状态码"""
    SUCCESS = 200           # 成功
    CREATED = 201           # 创建成功
    NO_CONTENT = 204       # 无内容
    BAD_REQUEST = 400      # 请求参数错误
    UNAUTHORIZED = 401     # 未授权
    FORBIDDEN = 403        # 禁止访问
    NOT_FOUND = 404        # 资源不存在
    CONFLICT = 409         # 资源冲突
    SERVER_ERROR = 500     # 服务器内部错误


class ResponseModel(BaseModel):
    """
    统一响应模型

    字段说明:
    - code: 状态码 (200=成功, 400=参数错误, 401=未授权, 404=不存在, 500=服务器错误)
    - msg: 提示信息
    - data: 响应数据
    - time: 时间戳
    """
    model_config = ConfigDict(from_attributes=True)

    code: int = Field(default=200, description="状态码")
    msg: str = Field(default="success", description="提示信息")
    data: Any = Field(default=None, description="响应数据")
    time: datetime = Field(default_factory=datetime.utcnow, description="时间戳")

    @classmethod
    def success(cls, data: Any = None, msg: str = "success") -> "ResponseModel":
        """成功响应"""
        return cls(code=ResponseCode.SUCCESS, msg=msg, data=data)

    @classmethod
    def created(cls, data: Any = None, msg: str = "创建成功") -> "ResponseModel":
        """创建成功响应"""
        return cls(code=ResponseCode.CREATED, msg=msg, data=data)

    @classmethod
    def no_content(cls, msg: str = "操作成功") -> "ResponseModel":
        """无内容响应 (用于删除操作)"""
        return cls(code=ResponseCode.NO_CONTENT, msg=msg)

    @classmethod
    def fail(cls, msg: str = "操作失败", code: int = ResponseCode.SERVER_ERROR) -> "ResponseModel":
        """失败响应"""
        return cls(code=code, msg=msg, data=None)

    @classmethod
    def bad_request(cls, msg: str = "请求参数错误") -> "ResponseModel":
        """参数错误响应"""
        return cls(code=ResponseCode.BAD_REQUEST, msg=msg)

    @classmethod
    def unauthorized(cls, msg: str = "未授权") -> "ResponseModel":
        """未授权响应"""
        return cls(code=ResponseCode.UNAUTHORIZED, msg=msg)

    @classmethod
    def forbidden(cls, msg: str = "禁止访问") -> "ResponseModel":
        """禁止访问响应"""
        return cls(code=ResponseCode.FORBIDDEN, msg=msg)

    @classmethod
    def not_found(cls, msg: str = "资源不存在") -> "ResponseModel":
        """资源不存在响应"""
        return cls(code=ResponseCode.NOT_FOUND, msg=msg)

    @classmethod
    def conflict(cls, msg: str = "资源冲突") -> "ResponseModel":
        """资源冲突响应"""
        return cls(code=ResponseCode.CONFLICT, msg=msg)

    @classmethod
    def server_error(cls, msg: str = "服务器内部错误") -> "ResponseModel":
        """服务器错误响应"""
        return cls(code=ResponseCode.SERVER_ERROR, msg=msg)


class PageResponse(BaseModel):
    """分页响应"""
    model_config = ConfigDict(from_attributes=True)

    items: List[Any] = Field(default_factory=list, description="数据列表")
    page: int = Field(default=1, description="当前页码")
    page_size: int = Field(default=20, description="每页数量")
    total: int = Field(default=0, description="总数量")
    total_pages: int = Field(default=0, description="总页数")


class ListResponse(BaseModel):
    """列表响应 (不带分页)"""
    items: List[Any] = Field(default_factory=list, description="数据列表")
    total: int = Field(default=0, description="总数量")


# ============================================
# Additional Request/Response Models (from schemas.py)
# ============================================

class FAQStatusUpdate(BaseModel):
    """更新 FAQ 状态请求"""
    faq_ids: List[int] = Field(..., min_length=1, description="FAQ ID列表")
    status: FAQStatusEnum
    modifier: str = Field(..., max_length=64, description="修改人")


class AskAnswer(BaseModel):
    """问答答案"""
    faq_id: int
    title: str
    score: float
    content: Optional[str] = None
    solution: Optional[dict] = None
    highlight: Optional[dict] = None


class AskAnswerResponse(BaseModel):
    """智能问答响应 (兼容旧版)"""
    answers: List[AskAnswer]
    total: int = 0


class RecommendRequest(BaseModel):
    """推荐请求"""
    faq_id: int = Field(..., description="FAQ ID")
    limit: int = Field(default=5, ge=1, le=10, description="返回数量")


class SyncPreviewRequest(BaseModel):
    """同步预览请求"""
    sync_type: SyncTypeEnum
    operator: str = Field(..., max_length=64, description="操作人")


class SyncExecuteRequest(BaseModel):
    """同步执行请求"""
    sync_type: SyncTypeEnum
    operator: str = Field(..., max_length=64, description="操作人")


class PageInfo(BaseModel):
    """分页信息"""
    page: int
    page_size: int
    total: int
    total_pages: int


class APIResponse(BaseModel):
    """通用 API 响应 (兼容旧版)"""
    code: int = 200
    message: str = "success"
    data: Optional[Any] = None


class ErrorResponse(BaseModel):
    """错误响应"""
    code: int
    message: str
    detail: Optional[str] = None
