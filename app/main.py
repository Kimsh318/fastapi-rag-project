import logging
import time
import asyncio
from datetime import datetime, timezone

from fastapi import FastAPI, Request

from app.core.config import settings
from app.core.events import startup_event, shutdown_event

# 각 서비스의 라우터 임포트
from app.atomic_services.queryRefine.api import router as query_refine_router
from app.atomic_services.queryFormatting.api import router as query_formatting_router
from app.atomic_services.queryValidation.api import router as query_validation_router
from app.atomic_services.retrieval_parallel_v4.api import router as retrieval_parallel_router
from app.atomic_services.healthcheck.api import router as healthcheck_router
from app.atomic_services.user_feedback.api import router as user_feedback_router

# Prometheus 설정
from app.core.metrics import setup_metrics

# 로깅 유틸리티
from app.utils.logging_utils import (
    log_http_middleware_to_es, 
    calculate_iso_time_difference, 
    setup_logging, 
    log_execution_time_async
)


def create_app():
    """
    FastAPI 애플리케이션을 생성하고 설정하는 함수.
    """
    app = FastAPI(docs_url=None, redoc_url=None)

    # Service 라우터 추가
    app.include_router(query_refine_router, prefix="/queryRefine", tags=["QueryRefine"])
    app.include_router(query_formatting_router, prefix="/queryFormatting", tags=["QueryFormatting"])
    app.include_router(query_validation_router, prefix="/queryValidation", tags=["QueryValidation"])
    app.include_router(retrieval_parallel_router, prefix="/retrieval_parallel", tags=["RetrievalParallele"])
    app.include_router(healthcheck_router, prefix="/healthCheck", tags=["HealthCheck"])
    app.include_router(user_feedback_router, prefix="/userFeedback", tags=["userFeedback"])

    # Prometheus 메트릭 설정
    setup_metrics(app)
    return app


# 애플리케이션 인스턴스 생성
app = create_app()


@app.on_event("startup")
async def startup():
    """애플리케이션 시작 이벤트"""
    logger = setup_logging()
    await startup_event(app)
    logger.info('app start!!')


@app.get("/")
def read_root():
    """루트 엔드포인트"""
    logger = logging.getLogger(__name__)
    logger.debug("Root endpoint called.")
    return {"Hello": "World"}


# Gunicorn 로거 설정
gunicorn_logger = logging.getLogger("gunicorn.error")
app.gunicorn_logger = gunicorn_logger


@app.middleware("http")
@log_execution_time_async
async def middleware(request: Request, call_next):
    """HTTP 미들웨어 - 요청/응답 로깅"""
    start_time = time.time()
    start_dtime = datetime.now(timezone.utc).isoformat()
    
    try:
        response = await call_next(request)
        elapsed_time = calculate_iso_time_difference(
            datetime.now(timezone.utc).isoformat(), 
            start_dtime
        )
        asyncio.create_task(
            log_http_middleware_to_es(
                request.url.path, 
                timestamp=start_dtime, 
                elapsed_time=elapsed_time
            )
        )
        return response
    except Exception as e:
        process_time = time.time() - start_time
        app.gunicorn_logger.error(
            f"Error occured after {process_time}s : {request.url.path}\n\n{e}\n=========================\n"
        )
        raise e
