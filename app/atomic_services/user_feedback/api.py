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
from app.utils.logging_utils import log_execution_time_async, log_api_data

from app.utils.es_client import get_es_client

from .service import FeedbackService
from .models import SampleRequest, SampleResponse
from .helpers import get_service_config

logger = logging.getLogger(__name__)

router = APIRouter()

@lru_cache(maxsize=None, typed=False)   
def get_feedback_service():
    logger.info("SampleService 초기화 시작")
    es_client = get_es_client()
    config = get_service_config()

    service = FeedbackService(es_client=es_client, 
                              config=config,
                            app_env=settings.APP_ENVIRONMENT)

    logger.info("FeedbackService 초기화 완료: %s", id(service))
    return service

@router.post("/user_feedback", response_model=SampleResponse)
@log_execution_time_async
async def save_user_feedback(request: SampleRequest, service: FeedbackService = Depends(get_feedback_service)):
    """
    사용자가 입력한 질의를 처리하고 결과를 반환합니다.

    - **query**: 사용자가 입력한 질의 문자열
    - **sample_result**: 질의에 대한 처리 결과
    """
    try:
        input_data = request.model_dump() #.dict()
        result = await service.save_feedback(input_data=input_data)
        return {"api_result":result["service_result"]}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
