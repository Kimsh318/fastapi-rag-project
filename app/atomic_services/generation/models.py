from pydantic import BaseModel

class GenerationRequest(BaseModel):
    """
    생성 요청에 사용되는 데이터 모델.
    """
    prompt: str  # 텍스트 생성에 사용할 컨텍스트 문자열
    len_prompt: int # prompt의 길이
    field: str
    user_id: str
    session_id: str

class GenerationResponse(BaseModel):
    """
    생성 응답에 사용되는 데이터 모델.
    """
    result: str  # 생성된 텍스트 결과
