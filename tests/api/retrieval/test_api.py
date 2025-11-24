# tests/api/retrieval/test_api.py

import pytest
import httpx
from httpx import AsyncClient
from app.main import app

# 모의 응답 클래스 정의
class MockResponse:
    """
    httpx.AsyncClient.post 메서드를 모의하기 위한 클래스.
    """
    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code

    async def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code != 200:
            raise httpx.HTTPStatusError("Error", request=None, response=None)

@pytest.mark.asyncio
async def test_retrieve_endpoint(mocker):
    """
    검색 엔드포인트에 대한 통합 테스트.
    클라이언트로부터의 요청이 올바르게 처리되어 예상한 문서 리스트를 반환하는지 확인합니다.
    """
    # 모의 응답 설정
    mock_response = {"documents": ["Document 1", "Document 2"]}
    mocker.patch("httpx.AsyncClient.post", return_value=MockResponse(json_data=mock_response))

    # 테스트 클라이언트 사용
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # /api/retrieval/ 엔드포인트로 POST 요청
        response = await ac.post("/api/retrieval/", json={"query": "Test query"})
        
        # 응답 상태 코드 검증
        assert response.status_code == 200

        # 응답 내용 검증
        json_response = response.json()
        assert json_response["documents"] == ["Document 1", "Document 2"]
