import asyncio
import json
from typing import AsyncGenerator
import httpx
import requests

from app.core.config import settings


def get_llm_processor(llm_config, app_env):
    """LLM Processor를 선택하는 함수"""
    if app_env == "prototype":
        return PrototypeLLMProcessor(llm_config=llm_config)
    elif app_env == "development":
        return DevelopmentLLMProcessor(llm_config=llm_config)
    raise ValueError("지원하지 않는 환경입니다.")


class BaseLLMProcessor:
    """Base LLM Processor"""
    
    def __init__(self, llm_config: dict):
        self.llm_host = llm_config['llm_host']
        self.llm_name = llm_config['llm_name']

    async def get_llm_response(self, prompt: str, len_prompt: int) -> AsyncGenerator[str, None]:
        """LLM 응답을 생성합니다."""
        raise NotImplementedError("이 메서드는 하위 클래스에서 구현되어야 합니다.")


class PrototypeLLMProcessor(BaseLLMProcessor):
    """Prototype 환경의 LLM Processor"""
    
    async def get_llm_response(self, prompt: str, len_prompt: int) -> AsyncGenerator[str, None]:
        """
        변환된 query를 LLM에 입력하고 프로토타입 환경에 맞게 생성된 응답을 반환하는 로직.
        """
        max_tokens_for_llm = settings.OLLAMA_MODEL_MAX_LEN - len_prompt
        llm_url = f"http://{self.llm_host}/api/chat"
        llm_headers = {"Content-type": "application/json"}
        
        data = {
            "model": settings.OLLAMA_MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens_for_llm,
            "temperature": 0,
            "top_k": 50,
            "best_of": 1,
            "seed": 1,
            "stream": True
        }

        async with httpx.AsyncClient(timeout=600.0) as client:
            try:
                async with client.stream("POST", llm_url, headers=llm_headers, json=data, timeout=600.0) as response:
                    async for line in response.aiter_lines():
                        if line.strip() and line.strip() != '[DONE]':
                            try:
                                token = json.loads(line.strip())['message']['content']
                                yield token
                            except KeyError as e:
                                yield f"오류가 발생했습니다. {e}"
                                break
            except Exception as e:
                yield f"Prototype 환경에서 오류가 발생했습니다: {e}"


class DevelopmentLLMProcessor(BaseLLMProcessor):
    """Development 환경의 LLM Processor"""
    
    async def get_llm_response(self, prompt: str, len_prompt: int) -> AsyncGenerator[str, None]:
        """
        변환된 query를 LLM에 입력하고 개발 환경에 맞게 생성된 응답을 반환하는 로직.
        """
        max_tokens_for_llm = settings.VLLM_MODEL_MAX_LEN - len_prompt
        llm_url = f"{self.llm_host}/v1/completions"
        llm_headers = {"Content-type": "application/json"}
        
        data = {
            "model": settings.VLLM_MODEL_PATH,
            "prompt": prompt,
            "max_tokens": max_tokens_for_llm,
            "temperature": 0,
            "top_k": 50,
            "best_of": 1,
            "seed": 1,
            "stream": True
        }
        
        try:
            completion = requests.post(llm_url, headers=llm_headers, json=data, stream=True)
            for line in completion.iter_lines(chunk_size=4096, delimiter=b"data:"):
                line_text = line.decode('utf-8')
                
                if isinstance(line_text, str) and line_text.strip() and line_text.strip() != '[DONE]':
                    try:
                        token = json.loads(line_text)['choices'][0]['text']
                        yield token
                    except KeyError:
                        yield "오류가 발생했습니다."
                        break
        except Exception as e:
            yield f"An error occurred in development environment: {e}"
