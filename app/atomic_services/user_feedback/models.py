from pydantic import BaseModel, Field

class SampleRequest(BaseModel):
    user_id: str = Field(..., description="사용자의 ID", example="test_user_id")
    feedback: str = Field(..., description="사용자가 작성한 피드백", example="이것은 샘플 피드백입니다.")

class SampleResponse(BaseModel):
    feedback: str = Field(..., description="DB에 저장된 피드백", example="이것은 샘플 피드백입니다.")