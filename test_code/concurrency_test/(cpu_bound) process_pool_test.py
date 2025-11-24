from concurrent.futures import ProcessPoolExecutor
import time
import asyncio
import os 

# CPU 바운드 작업 함수
def calculate_factorial(n):
    """큰 수의 팩토리얼 계산"""
    result = 1
    for i in range(1, n + 1):
        time.sleep(0.1)
        result *= i
        print(f"Process {os.getpid()}: {n}의 팩토리얼 계산, 현재 단계: {i}")  # 진행 상태 출력
    return result

async def process_cpu_bound_tasks(numbers):
    """
    ProcessPoolExecutor를 사용하여 CPU 바운드 작업 병렬 처리
    """
    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor() as executor:
        # 병렬로 실행할 작업 생성
        tasks = [loop.run_in_executor(executor, calculate_factorial, num) for num in numbers]
        # 모든 작업 실행 및 결과 반환
        results = await asyncio.gather(*tasks)
        return results

if __name__ == "__main__":
    numbers = [5, 7, 10]  # 팩토리얼을 계산할 입력 값
    print("ProcessPoolExecutor 시작")
    start_time = time.time()
    # asyncio 이벤트 루프 실행
    results = asyncio.run(process_cpu_bound_tasks(numbers))
    end_time = time.time()
    print(f"결과(일부 출력): {results[:2]}...")  # 일부 결과만 출력
    print(f"실행 시간: {end_time - start_time:.2f} 초")
