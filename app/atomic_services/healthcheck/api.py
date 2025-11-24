import asyncio
import logging
from fastapi import APIRouter
from datetime import datetime

from .models import HealthCheckResponse

router = APIRouter()

# 로깅 설정
#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("excution_time_logger")

@router.get("/health_check", response_model=HealthCheckResponse)
async def health_check():
    """
    Health Check Endpoint
    간단한 서버 상태 확인용 엔드포인트
    """
    # 현재 시각을 기반으로 6자리 ID 생성 (시, 분, 초)
    now = datetime.now()
    #unique_id = f"{now.hour:02}{now.minute:02}{now.second:02}-{random.randint(1,99)}"

    # 비동기 로그 함수 호출
    # asynci.create_task(log_iterations(unique_id))
    # await log_iterations(unique_id)

    
    return HealthCheckResponse(
        status="success",
        message="The server is running and ready to accept requests."
    )