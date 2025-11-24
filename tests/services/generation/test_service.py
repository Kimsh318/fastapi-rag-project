# tests/services/generation/test_service.py

import pytest
import httpx
from app.atomic_services.generation.service import GenerationService

# 모의 응답 클래스 정의
class MockResponse:
    """
    httpx.AsyncClient.stream 메서드를 모의하기 위한 클래스.
    """
    def __init__(self, chunks, status_code=200):
        self.chunks = chunks
        self.status_code = status_code

    async def aiter_text(self):
        """
        비동기 제너레이터로 토큰 스트림을 반환합니다.
        """
        for chunk in self.chunks:
            yield chunk

    def raise_for_status(self):
        if self.status_code != 200:
            raise httpx.HTTPStatusError("Error", request=None, response=None)

@pytest.mark.asyncio
async def test_generate_stream(mocker):
    """
    GenerationService의 generate_stream 메서드에 대한 단위 테스트.
    외부 API 호출을 모의하고, 예상되는 텍스트 토큰을 스트리밍으로 반환하는지 확인합니다.
    """
    # 모의 토큰 스트림 설정
    mock_chunks = ["Generated ", "text ", "based ", "on ", "context."]
    mocker.patch("httpx.AsyncClient.stream", return_value=MockResponse(chunks=mock_chunks))

    # 서비스 인스턴스 생성
    service = GenerationService()

    # 테스트할 컨텍스트
    context = "Test context for generation."

    # generate_stream 메서드 호출
    token_generator = service.generate_stream(context)

    # 생성된 토큰 수집
    generated_tokens = []
    async for token in token_generator:
        generated_tokens.append(token)

    # 결과 검증
    assert generated_tokens == mock_chunks
