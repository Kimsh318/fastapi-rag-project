# 비동기 함수와 동시성 테스트
import asyncio

# 1. 비동기 함수 (Asynchronous Function)
async def async_function():
    print("비동기 함수 시작")
    await asyncio.sleep(2)  # 2초 대기
    print("비동기 함수 종료")

# 2. 동시성 처리 (Concurrency Handling)
async def task_1():
    print("작업 1 시작")
    await asyncio.sleep(2)
    print("작업 1 종료")

async def task_2():
    print("작업 2 시작")
    await asyncio.sleep(1)
    print("작업 2 종료")

async def concurrency_handling():
    await asyncio.gather(task_1(), task_2())

# 3. 비동기 함수 + 동시성 처리 (Asynchronous Function with Concurrency Handling)
async def fetch_data(source):
    print(f"{source}에서 데이터 가져오기 시작")
    await asyncio.sleep(2)  # 데이터 가져오는 데 2초 소요
    print(f"{source}에서 데이터 가져오기 완료")
    return f"{source}의 데이터"

async def async_with_concurrency():
    sources = ["소스 1", "소스 2", "소스 3"]
    tasks = [fetch_data(source) for source in sources]
    results = await asyncio.gather(*tasks)
    print("모든 데이터 가져오기 완료:", results)

# 메인 함수에서 순차적으로 실행
async def main():
    print("\n1. 비동기 함수 실행")
    await async_function()

    print("\n2. 동시성 처리 실행")
    await concurrency_handling()

    print("\n3. 비동기 함수 + 동시성 처리 실행")
    await async_with_concurrency()

# 이벤트 루프를 통해 메인 함수 실행
if __name__ == "__main__":
    asyncio.run(main())
