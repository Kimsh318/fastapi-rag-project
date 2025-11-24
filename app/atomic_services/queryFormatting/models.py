from pydantic import BaseModel

class QueryFormattingRequest(BaseModel):
    query: str
    task: str # 'rag', 'free_talking' 등
    list_doc_ids: list  # 질의 관련 문서들의 id 리스트
    user_id: str
    session_id: str

class QueryFormattingResponse(BaseModel):
    formatted_query: str
    list_doc_ids: list[str]    # 쿼리 포맷팅에 사용된 문서들의 id 리스트
