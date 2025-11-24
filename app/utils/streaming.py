from typing import AsyncGenerator, Coroutine, Union

#async def stream_response(generator: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
async def stream_response(generator: Union[AsyncGenerator[str, None], Coroutine]) -> AsyncGenerator[str, None]:
    """
    비동기 제너레이터로부터 데이터를 받아 스트리밍 응답으로 변환합니다.

    Args:
        generator (AsyncGenerator[str, None]): 데이터 스트림을 생성하는 비동기 제너레이터.

    Yields:
        str: 스트리밍 응답으로 전송할 데이터 청크.
    """
    if isinstance(generator, Coroutine):
        generator = await generator # 코루틴을 비동기 제너레이터로 변환

    print(f"streaming.py stream_response {generator}")
    async for chunk in generator:
        yield chunk
