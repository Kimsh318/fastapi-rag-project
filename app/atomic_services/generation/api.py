from functools import lru_cache
from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
import asyncio
from typing import AsyncGenerator

from app.atomic_services.generation.service import GenerationService
from app.atomic_services.generation.models import GenerationRequest
from .helpers import get_llm_config
from app.utils.streaming import stream_response
from app.core.config import settings
from app.utils.logging_utils import log_api_call

from app.utils.logging_utils import log_streaming_api_call

from app.utils.logging_utils import log_api_call, StandardLogger

router = APIRouter()

@lru_cache(maxsize=None, typed=False)   
def get_generation_service():
    print('init generation service')
    llm_config = get_llm_config(settings)
    print(f'llm config : {llm_config}')
    return GenerationService(llm_config=llm_config, app_env=settings.APP_ENVIRONMENT)

# @router.post("/generation")
# @log_streaming_api_call(endpoint="/generation", index_type="api")
# async def generation(request: GenerationRequest, 
#                    service: GenerationService = Depends(get_generation_service)):
#     """
#     생성 엔드포인트.
#     클라이언트로부터 컨텍스트를 받아 텍스트를 스트리밍 방식으로 생성하여 반환합니다.
#     """
#     print(f'Generation request arrived \n{request}')
#     prompt, len_prompt = request.prompt, request.len_prompt

#     token_generator = service.generate_stream(prompt, len_prompt)
#     return StreamingResponse(
#         stream_response(token_generator),
#         media_type="text/event-stream"
#     )


@router.post("/generation")
@log_streaming_api_call(endpoint="/generation", index_type="api")
async def generation(request: GenerationRequest, 
                     service: GenerationService = Depends(get_generation_service)):
    """
    생성 엔드포인트.
    클라이언트로부터 컨텍스트를 받아 텍스트를 스트리밍 방식으로 생성하여 반환합니다.
    """
    print(f'Generation request arrived \n{request}')
    prompt, len_prompt = request.prompt, request.len_prompt

    token_generator = await service.generate_stream(prompt, len_prompt)

    print(f"token_generator type : {token_generator}")
    print(f"stream_response(token_generator) type : {stream_response(token_generator)}")
    # # stream_response가 token을 스트리밍하면서 동시에 logger로 로그를 기록
    # return StreamingResponse(
    #     stream_response(token_generator),
    #     media_type="text/event-stream"
    # )
    # 비동기 생성기를 데코레이터에 넘기고 StreamingResponse로 감싸서 반환
    return token_generator











# @router.post("/generation")
# @log_streaming_api_call(endpoint='generation', index_type="api")
# async def generation(request: GenerationRequest, 
#                      service: GenerationService = Depends(get_generation_service)):
#     # 스트리밍 응답 엔드포인트
#     print(f'Generation request arrived \n{request}')
#     prompt, len_prompt = request.prompt, request.len_prompt

#     # generate_stream을 비동기 제너레이터로 반환
#     token_generator: AsyncGenerator = service.generate_stream(prompt, len_prompt)

#     async def stream_generator():
#         async for token in service.generate_stream(prompt, len_prompt):
#             # SSE 형식으로 데이터 전송
#             yield f"data: {token}\n\n"

#     # StreamingResponse 생성 시 비동기 제너레이터로 사용
#     return StreamingResponse(
#         stream_generator(),
#         media_type="text/event-stream"
#     )



# @router.post("/generation")
# @log_streaming_api_call(endpoint='generation', index_type="api")
# async def generation(
#     request: GenerationRequest, 
#     service: GenerationService = Depends(get_generation_service)
# ):
#     print(f'Generation request arrived \n{request}')
#     prompt, len_prompt = request.prompt, request.len_prompt

#     # 제너레이터 함수를 직접 반환
#     return StreamingResponse(
#         generate_stream(service, prompt, len_prompt),
#         media_type="text/event-stream"
#     )
# async def generate_stream(service, prompt, len_prompt):
#     generator = service.generate_stream(prompt, len_prompt)
#     async for token in await generator:
#         yield f"data: {token}\n\n"
