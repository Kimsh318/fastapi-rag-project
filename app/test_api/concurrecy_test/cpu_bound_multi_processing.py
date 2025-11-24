from fastapi import APIRouter
import asyncio
from pydantic import BaseModel
from concurrent.futures import ProcessPoolExecutor
import random
import time

import os  # os 모듈을 임포트합니다.
    

router = APIRouter()

def cpu_bound_task(data, request_id):
    """
    CPU 집약적인 작업을 수행하며 중간 과정을 출력합니다.
    각 반복마다 0.3초 대기합니다.
    """
    result = 0
    for i in range(10**4):
        result += data
        # 100,000번마다 중간 과정 출력
        if i % (10**3) == 0:
            print(f"[Request ID: {request_id}] Iteration {i}, Current Result: {result}")
            time.sleep(round(random.uniform(1,5), 1))  # 0.3초 대기
    print(f"[Request ID: {request_id}] Task Completed! Final Result: {result}")
    return result

class DataModel(BaseModel):
    data: int

@router.post("/cpu-bound-multi-processing")
async def cpu_bound_multi_processing_endpoint(data_model: DataModel):
    """
    CPU 집약적인 작업을 멀티프로세싱으로 처리하는 엔드포인트.
    요청 ID를 생성하여 로그에 출력합니다.
    """
    # 현재 워커의 PID 가져오기
    worker_pid = os.getpid()
    # 랜덤한 세 자리 정수 ID 생성
    request_id = random.randint(100, 999)
    data = data_model.data
    print(f"[Worker PID: {worker_pid}]-[Request ID: {request_id}] Request received with data: {data}")

    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor() as pool:
        # ProcessPoolExecutor에 ID 전달
        #result = await loop.run_in_executor(pool, cpu_bound_task, data, request_id)

        # 작업을 비동기적으로 시작하고, 모든 작업이 완료될 때까지 기다립니다.
        future = loop.run_in_executor(pool, cpu_bound_task, data, request_id)
        # asyncio.gather를 사용하여 모든 future를 동시에 실행합니다.
        result = await asyncio.gather(future)

    print(f"[Request ID: {request_id}] Response sent")
    return {"request_id": request_id, "result": result}
