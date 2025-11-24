from pydantic import BaseModel, Field

# QueryRefine 모델 정의
class RefinedQueryRequest(BaseModel):
    query: str = Field(..., description="사용자가 입력한 질의", example="약식심사 대상 여신은? 답변 요약해줘")
    task: str = Field(..., description="queryRefine이 필요한 태스크", example="VectorSearch")
    user_id: str = Field(..., description="Client의 user id", example="k220038")
    session_id: str = Field(..., description="Client의 session id", example="1871295")

class RefinedQueryResponse(BaseModel):
    refined_query: str = Field(..., description="queryRefine된 결과", example="약식심사 대상 여신은?")