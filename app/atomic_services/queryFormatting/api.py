# ------------------------------
# Python 기본 라이브러리
# ------------------------------
from functools import lru_cache
import logging
import asyncio
# ------------------------------
# 외부 라이브러리
# ------------------------------
from fastapi import APIRouter, Depends, HTTPException

# ------------------------------
# app의 다른 경로에서 import
# ------------------------------
from app.core.config import settings
from app.utils.logging_utils import log_api_call, log_execution_time_async, log_api_data
from app.utils.es_client import get_es_client
# from app.main import app

#from app.atomic_services.global_app_helpers import app

from .service import QueryFormattingService
from .models import QueryFormattingRequest, QueryFormattingResponse
from .helpers import get_formatting_config, get_tokenizer#, get_es_client


logger = logging.getLogger(__name__)

router = APIRouter()

@lru_cache(maxsize=None, typed=False)   
def get_query_formatting_service():
    logger.info("QueryFormattingService 초기화 시작")
    es_client = get_es_client()
    formatting_config = get_formatting_config(settings)
    # TASK_LLM_MAPPING을 인자로 넘겨서 각 태스크별 토크나이저 생성
    tokenizer_dict = get_tokenizer(settings.TASK_LLM_MAPPING, settings.HF_TOKEN)
    service = QueryFormattingService(formatting_config, 
                                     tokenizer_dict, 
                                     app_env=settings.APP_ENVIRONMENT, 
                                     es_client=es_client)
    logger.info("QueryFormattingService 초기화 완료: %s", id(service))
    return service

@router.post("/query_formatting", response_model=QueryFormattingResponse)
@log_execution_time_async
async def format_query(request:QueryFormattingRequest, service: QueryFormattingService = Depends(get_query_formatting_service)):
# async def format_query(request:QueryFormattingRequest, service: QueryFormattingService = Depends(app.state.query_formatting_service)):
# async def format_query(request:QueryFormattingRequest, service: QueryFormattingService = Depends(router.app.state.query_formatting_service)):
    query, task, list_doc_ids = request.query, request.task, request.list_doc_ids
    format_result = await service.format_query(query=query, task=task, list_doc_ids=list_doc_ids)


    asyncio.create_task(log_api_data("query_formatting", 
                        request.query, 
                         ', '.join(format_result["list_doc_ids"])))
    
    if 'list_doc_ids' in format_result:
        return {"formatted_query": format_result["formatted_query"],
               "list_doc_ids": format_result["list_doc_ids"]}
    return {"formatted_query": format_result["formatted_query"]}
