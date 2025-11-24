from pydantic import BaseModel

class QueryValidationRequest(BaseModel):
    query: str
    user_id: str
    session_id: str

class QueryValidationResponse(BaseModel):
    is_valid: bool     # 검증 결과 : 유효하면 True, 유효하지 않으면 False
    num_tokens: int     # 쿼리의 토큰 수

