# tests/api/generation/test_api.py

import pytest
from httpx import AsyncClient
from app.main import app

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
async def test_generate_endpoint(mocker):
    """
    생성 엔드포인트에 대한 통합 테스트.
    클라이언트로부터의 요청이 올바르게 처리되어 스트리밍된 텍스트가 예상대로 반환되는지 확인합니다.
    """
    # 모의 토큰 스트림 설정
    mock_chunks = ["Generated ", "text ", "based ", "on ", "context."]
    mocker.patch("httpx.AsyncClient.stream", return_value=MockResponse(chunks=mock_chunks))

    # 테스트 클라이언트 사용
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # /api/generation/ 엔드포인트로 POST 요청
        response = await ac.post("/api/generation/", json={"context": "Test context for generation."})
        
        # 응답 상태 코드 검증
        assert response.status_code == 200

        # 스트리밍 응답 받기
        generated_text = ""
        async for chunk in response.aiter_text():
            generated_text += chunk

        # 생성된 텍스트 검증
        assert generated_text == "Generated text based on context."
