# tests/services/retrieval/test_service.py

import pytest
import httpx
from app.atomic_services.retrieval_async_v2.service import RetrievalService

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
async def test_retrieve(mocker):
    """
    RetrievalService의 retrieve 메서드에 대한 단위 테스트.
    외부 API 호출을 모의하고, 예상되는 문서 리스트를 반환하는지 확인합니다.
    """
    # 모의 응답 설정
    mock_response = {"documents": ["Document 1", "Document 2"]}
    mocker.patch("httpx.AsyncClient.post", return_value=MockResponse(json_data=mock_response))

    # 서비스 인스턴스 생성
    service = RetrievalService()

    # 테스트할 쿼리
    query = "Test query"

    # retrieve 메서드 호출
    documents = await service.retrieve(query)

    # 결과 검증
    assert documents == ["Document 1", "Document 2"]
