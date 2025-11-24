import httpx
from typing import AsyncGenerator

from .helpers import QueryPreprocessor, get_llm_config
from .processors import get_llm_processor
from app.core.config import settings
from app.utils.logging_utils import log_api_call

from app.utils.logging_utils import log_streaming_api_call  

class GenerationService:
    """
    텍스트 생성을 제공하는 서비스 클래스.
    외부 생성 API를 호출하고, 스트리밍 방식으로 결과를 반환합니다.
    """
    # 비동기 생성기 호출 횟수를 기록하는 클래스 변수
    call_count = 0

    def __init__(self, llm_config:dict, app_env:str):
        # self.query_preprocessor = QueryPreprocessor()
        self.llm_processor = get_llm_processor(llm_config, app_env)

    # @log_streaming_api_call(index_type="service")
    # async def generate_stream(self, prompt: str, len_prompt: int) -> AsyncGenerator[str, None]:
    #     # 주어진 프롬프트에 기반하여 외부 API를 통해 텍스트를 스트리밍 방식으로 생성합니다..

    #     # 1. 프롬프트 전처리
    #     # processed_prompt = self.query_preprocessor.preprocess(prompt)
        
    #     # 2. LLM 응답 생성 및 스트리밍
    #     async for token in self.llm_processor.get_llm_response(prompt, len_prompt):
    #         yield token



    # @log_streaming_api_call(index_type="service")
    # async def generate_stream(self, prompt: str, len_prompt: int) -> AsyncGenerator[str, None]:
    #     generator = self.llm_processor.get_llm_response(prompt, len_prompt)
    #     print(f"service generate stream : {generator}")
    #     async for token in generator:
    #         yield token

    async def generate_stream(self, prompt: str, len_prompt: int) -> AsyncGenerator[str, None]:
        generator = self.llm_processor.get_llm_response(prompt, len_prompt)
        
        GenerationService.call_count += 1
        print(f"\n\n======\ngenerate_stream called. Total calls: {GenerationService.call_count}\n\n=======\n")

        print(f"service generate stream : {generator}")
        return generator




    # @log_streaming_api_call(index_type="service")
    # def generate_stream(self, prompt: str, len_prompt: int) -> AsyncGenerator[str, None]:
    #     generator = self.llm_processor.get_llm_response(prompt, len_prompt)
    #     print(f"service generate stream : {generator}")
    #     return generator
