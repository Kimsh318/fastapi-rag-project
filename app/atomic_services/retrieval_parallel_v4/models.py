from pydantic import BaseModel, Field
from typing import List

# REMOVE : Retrieval service를 세분화하면서 더이상 안쓰는 모델
class RetrievalRequest(BaseModel):
    """
    검색 요청에 사용되는 데이터 모델.
    """
    query: str  # 검색 쿼리 문자열


class SemanticSearchRequest(BaseModel):
    """
    벡터 검색 요청에 사용되는 데이터 모델.
    """
    query: str  # 검색 쿼리 문자열
    user_specified_doc_types: list[str] # 검색할 문서 유형
    field: str
    user_id: str
    session_id: str

class KeywordSearchRequest(BaseModel):
    """
    키워드 검색 요청에 사용되는 데이터 모델.
    """
    query: str  # 검색 쿼리 문자열
    user_specified_doc_types: list[str] # 검색할 문서 유형
    field: str
    user_id: str
    session_id: str


class HybridSearchRequest(BaseModel):
    """
    하이브리드 검색 요청에 사용되는 데이터 모델.
    """
    query: str  # 검색 쿼리 문자열
    user_specified_doc_types: list[str] # 검색할 문서 유형
    field: str
    user_id: str
    session_id: str


class HybridSearchwithRerankRequest(BaseModel):
    """
    하이브리드 검색 및 Rerank 요청에 사용되는 데이터 모델.
    """
    query: str  # 검색 쿼리 문자열
    user_specified_doc_types: list # 검색할 문서 유형
    excluded_docs: list # Rerank 결과에서 제외할 문서 유형 
    field: str  # 검색 범위 : 현재는 'corporate'만 입력받을 예정이며, 엔드포인트 수행로직에 영향을 주진 않음
    user_id: str
    session_id: str
    


#========== 임시로 구현. 다른 py파일들 함께 수정 필요
class Document(BaseModel):
    """
    검색된 문서의 구조를 정의하는 모델.
    """
    doc_id: str = Field(..., description="문서의 소분류", example="여신1권")
    doc_type: str = Field(..., description="문서 유형 (예: 행통(소), 여신지침(소) 등)", example="행통(소)")
    chunk_id: str = Field(..., description="DB에 저장된 Chunk의 ID(uuid.uuid4()로 생성)", example="068dca2e-1b4a-4317-a2ef-459f1d516584")
    highlight: list[str] = Field(..., description="Chunk에서 Query의 키워드가 포함된 텍스트들.", example="['텍스트1', '텍스트2']")
    #chunk_head: list[str] = Field(..., description="highlight가 없을 때를 대비해서, Chunk의 첫 N개 라인을 반환", example="['텍스트1', '텍스트2']")
    chunk_context: str = Field(..., description="Chunk 본문", example="Chunk 본문 텍스트입니다")


class RetrievalResponse(BaseModel):
    """
    검색 응답에 사용되는 데이터 모델.
    """
    # doc_type: str #행통 교재 등
    # list_doc_ids
    documents: List[Document]  # 검색된 문서들의 리스트


# class RetrievalResponse(BaseModel):
#     """
#     검색 응답에 사용되는 데이터 모델.
#     """
#     # documents: List[str]  # 이전 코드는 주석 처리
#     document_objects: List[DocumentObject]  # 검색된 문서 객체들의 리스트