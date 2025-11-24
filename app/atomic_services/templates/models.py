from pydantic import BaseModel, Field

class SampleRequest(BaseModel):
    query: str = Field(..., description="사용자가 입력한 질의 문자열", example="프랑스의 수도는 어디인가요?")

class SampleResponse(BaseModel):
    sample_result: str = Field(..., description="질의에 대한 처리 결과", example="파리")