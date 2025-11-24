import logging
from functools import lru_cache

from fastapi import APIRouter, Depends

from app.core.config import settings
from app.utils.logging_utils import log_execution_time_async
from app.utils.es_client import get_es_client

from .service import QueryRefineService
from .models import RefinedQueryRequest, RefinedQueryResponse
from .helpers import get_faiss_client#, get_es_client

router = APIRouter()

# API용 로거 불러오기
api_logger = logging.getLogger("api_logger")

#=================== Service 객체화 함수(의존성 주입에 활용) =====================
@lru_cache(maxsize=None, typed=False)   
def get_query_refine_service():
    es_client = get_es_client()
    faiss_client = get_faiss_client()
    return QueryRefineService(es_client, faiss_client, app_env=settings.APP_ENVIRONMENT)

#=================== API router =====================
@router.post("/query_refine", response_model=RefinedQueryResponse)
@log_execution_time_async # @log_api_call(endpoint='query_refine', index_type="api")
async def refine_query(request: RefinedQueryRequest,
                        service: QueryRefineService = Depends(get_query_refine_service)):
    # 서비스 호출
    refined_result = await service.refine(query=request.query, task=request.task)

    return {"refined_query": refined_result["refined_query"]}