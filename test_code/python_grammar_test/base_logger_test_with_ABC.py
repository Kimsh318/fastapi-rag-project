# logging_utils.py
from typing import AsyncGenerator, Any, Optional
from fastapi import StreamingResponse
from functools import wraps
from copy import deepcopy
from abc import ABC, abstractmethod


# ======= Logger Classes =======
class BaseLogger(ABC):
    # 로깅 시스템의 기본 클래스
    def __init__(self, endpoint: Optional[str], index_type: str):
        self.endpoint = endpoint
        self.index_type = index_type
        self.api_data = None
        self.index_name = f"{index_type}-usage-logs"

    def initialize_logging(self, args: tuple, kwargs: dict) -> None:
        # 로깅 초기화 및 기본 설정
        self.api_data = initialize_log_data()
        kwargs_copy = deepcopy(kwargs)
        
        if self.index_type == "api":
            setup_api_log_data(self.api_data, self.endpoint, kwargs_copy, args)
            LoggingContext.set_request_id(self.api_data['request_id'])

    @abstractmethod
    async def log_response(self, response: Any) -> Any:
        # 응답 로깅을 위한 추상 메서드
        pass

class StandardLogger(BaseLogger):
    # 일반 응답을 위한 로거
    async def log_response(self, response: Any) -> Any:
        self.api_data["output_data"] = response
        finalize_log_data(self.api_data, response)
        await save_log_to_es(index_name=self.index_name, log_data=self.api_data)
        return response

class StreamingLogger(BaseLogger):
    # 스트리밍 응답을 위한 로거
    class DataCollector:
        # 스트리밍 데이터 수집기
        def __init__(self, original_response: StreamingResponse):
            self.original_response = original_response
            self.collected_data = []

        async def wrap_generator(self, original_generator: AsyncGenerator) -> AsyncGenerator:
            # 데이터를 수집하면서 스트리밍
            async for chunk in original_generator:
                decoded_chunk = chunk.decode() if isinstance(chunk, bytes) else chunk
                self.collected_data.append(decoded_chunk)
                yield chunk

        def get_collected_data(self) -> str:
            return "".join(self.collected_data)

        def create_wrapped_response(self) -> StreamingResponse:
            # 수집기가 포함된 새로운 StreamingResponse 생성
            return StreamingResponse(
                self.wrap_generator(self.original_response.body_iterator),
                media_type=self.original_response.media_type,
                status_code=self.original_response.status_code,
                headers=dict(self.original_response.headers)
            )

    async def log_response(self, response: Any) -> Any:
        if not isinstance(response, StreamingResponse):
            # 스트리밍이 아닌 응답은 StandardLogger로 처리
            return await StandardLogger(self.endpoint, self.index_type).log_response(response)

        # 스트리밍 응답 처리
        collector = self.DataCollector(response)
        wrapped_response = collector.create_wrapped_response()

        async def on_streaming_complete():
            self.api_data["output_data"] = collector.get_collected_data()
            finalize_log_data(self.api_data, response)
            await save_log_to_es(index_name=self.index_name, log_data=self.api_data)

        asyncio.create_task(on_streaming_complete())
        return wrapped_response

# ======= Decorators for Logging API Calls =======
def log_api_call(endpoint: Optional[str] = None, index_type: str = None):
    # 일반 응답을 위한 로깅 데코레이터
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            logger = StandardLogger(endpoint, index_type)
            logger.initialize_logging(args, kwargs)
            response = await func(*args, **kwargs)
            return await logger.log_response(response)
        return wrapper
    return decorator

def log_streaming_api_call(endpoint: Optional[str] = None, index_type: str = None):
    # 스트리밍 응답을 위한 로깅 데코레이터
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            logger = StreamingLogger(endpoint, index_type)
            logger.initialize_logging(args, kwargs)
            response = await func(*args, **kwargs)
            return await logger.log_response(response)
        return wrapper
    return decorator

# ======= Example Usage =======
@router.post("/standard-endpoint")
@log_api_call(endpoint='standard', index_type="api")
async def standard_endpoint(request: Request):
    # 일반 응답 엔드포인트
    return {"message": "success"}

@router.post("/generation")
@log_streaming_api_call(endpoint='generation', index_type="api")
async def generation(request: GenerationRequest, 
                   service: GenerationService = Depends(get_generation_service)):
    # 스트리밍 응답 엔드포인트
    print(f'Generation request arrived \n{request}')
    prompt, len_prompt = request.prompt, request.len_prompt

    token_generator = service.generate_stream(prompt, len_prompt)
    return StreamingResponse(
        stream_response(token_generator),
        media_type="text/event-stream"
    )