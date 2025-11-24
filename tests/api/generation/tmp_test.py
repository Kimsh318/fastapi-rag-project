# import httpx

# url = "http://localhost:8000/generation/"
# payload = {"context": "example context for text generation"}

# async def get_streaming_response():
#     async with httpx.AsyncClient() as client:
#         async with client.stream("GET", url, json=payload) as response:
#             # 상태 코드 확인
#             response.raise_for_status()
#             # 스트리밍된 텍스트를 처리
#             async for chunk in response.aiter_text():
#                 print(chunk)

# # 비동기 실행을 위한 이벤트 루프 실행
# import asyncio
# asyncio.run(get_streaming_response())


import httpx

async def fetch_streamed_data():
    async with httpx.AsyncClient() as client:
        # 서버의 엔드포인트에 GET 요청을 보내 스트림 데이터를 받음
        async with client.stream("GET", "http://localhost:8000/generation/", params={"context": "example"}) as response:
            response.raise_for_status()
            async for chunk in response.aiter_text():
                print(chunk)  # 각 단어를 출력

# 비동기 함수 호출을 위해 asyncio 실행
import asyncio
asyncio.run(fetch_streamed_data())
