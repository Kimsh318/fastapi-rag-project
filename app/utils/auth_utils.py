from fastapi import HTTPException, Request, Depends, status

def get_user_type(user_id: str) -> str:
    """
    주어진 user_id(k행번, o행번 등)을 기반으로 사용자의 권한 수준을 결정합니다.
    """
    
    user_type = user_id[0] # k, o 행번 추출
    if user_type == "o":
        return "외주직원"
    return "전직원"

def get_user_id(request: Request) -> str:
    """
    요청 헤더에서 사용자 인증 토큰을 추출합니다.
    
    Args:
        request (Request): FastAPI 요청 객체.
    
    Returns:
        str: 요청 헤더에서 추출한 인증 토큰.
    
    Raises:
        HTTPException: User ID가 없을 경우, 유효하지 않은 입력이 주어졌으므로, 401 Unauthorized 오류를 발생시킵니다. 
    """

    user_id = request.user_id
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing token",
        )
    return user_id

def user_auth_required(required_user_type: str):
    """
    특정 권한 수준이 필요한 엔드포인트에 대해 접근 권한을 확인하는 데코레이터를 반환합니다.
    
    Args:
        required_user_type (str): 엔드포인트에 필요한 권한을 가지는 유저 타입.(외주직원, 전직원 등)
    
    Returns:
        Callable: 접근 권한을 확인하는 내부 함수.

    Raises:
        HTTPException: 사용자의 권한이 부족할 경우 403 Forbidden 오류를 발생시킵니다.
    """

    def auth_checker(user_id: str = Depends(get_user_id)):
        user_type = get_user_type(user_id)
        if user_type != required_user_type:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User role '{user_type}' does not have access to this resource",
            )
    return auth_checker


# =======================
# 사용 예제 (api.py)
# =======================

# from fastapi import APIRouter, Depends
# from app.utils.auth_utils import role_required, get_user_token

# router = APIRouter()

# @router.post("/query_refine")
# @role_required("user")  # 사용자 역할이 "user" 이상이어야 접근 가능
# async def refine_query(request: RefinedQueryRequest, token: str = Depends(get_user_token)):
#     # 서비스 호출
#     refined_result = await service.refine(query=request.query, task=request.task)
#     return {"refined_query": refined_result["refined_query"]}