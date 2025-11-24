from fastapi import APIRouter
import asyncio
from pydantic import BaseModel
from concurrent.futures import ProcessPoolExecutor
import random
import time

import os  # os 모듈을 임포트합니다.
    

router = APIRouter()

def cpu_bound_task(data, pid, request_id):
    """
    CPU 집약적인 작업을 수행하며 중간 과정을 출력합니다.
    각 반복마다 0.3초 대기합니다.
    """
    print(f"[Worker PID: {pid}]-[Request ID: {request_id}] Starting CPU-bound task")
    result = 0
    for i in range(10**4):
        result += data
        # 100,000번마다 중간 과정 출력
        if i % (10**3) == 0:
            #print(f"[Request ID: {request_id}] Iteration {i}, Current Result: {result}")
            time.sleep(round(random.uniform(1,5), 1))  # 0.3초 대기
    
    print(f"[Worker PID: {pid}]-[Request ID: {request_id}] CPU-bound Task Completed! Final Result: {result}")
    return result


# CPU 바운드 작업을 위한 함수
async def run_cpu_bound_task(data, pid, request_id):
    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor() as pool:
        future = loop.run_in_executor(pool, cpu_bound_task, data, pid, request_id)
        result = await asyncio.gather(future)
    return result

# IO 바운드 작업을 위한 함수 (예시로 추가)
async def io_bound_task(pid, request_id):
    print(f"[Worker PID: {pid}]-[Request ID: {request_id}] Starting IO-bound task")
    
    # IO 바운드 작업을 수행하는 코드
    await asyncio.sleep(1)  # 예시로 1초 대기를 넣었습니다.
    print(f"[Worker PID: {pid}]-[Request ID: {request_id}] IO-bound Task Completed!")
    
    return "IO Task Completed"



class DataModel(BaseModel):
    data: int

@router.post("/cpu-io-bound-multi-processing")
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

    # 1. 첫 번째 CPU 바운드 작업 실행
    result1 = await run_cpu_bound_task(data, worker_pid, request_id)
    #print(f"[Request ID: {request_id}] First CPU-bound task completed with result: {result1}")

    # 2. IO 바운드 작업 실행
    io_result = await io_bound_task(worker_pid, request_id)
    #print(f"[Request ID: {request_id}] IO-bound task completed with result: {io_result}")

    # 3. 두 번째 CPU 바운드 작업 실행
    result2 = await run_cpu_bound_task(data, worker_pid, request_id)
    #print(f"[Request ID: {request_id}] Second CPU-bound task completed with result: {result2}")

    #print(f"[Request ID: {request_id}] Response sent")
    return {"request_id": request_id, "result1": result1, "io_result": io_result, "result2": result2}
