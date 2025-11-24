# Python 기본 라이브러리
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict

# 애플리케이션 내부 모듈
from app.utils.logging_utils import log_execution_time_async
from app.core.config import settings


# 비동기 동시처리를 위해 사용
thread_executor = ThreadPoolExecutor(max_workers=2)


def get_query_refine_processor(es_client, faiss_client, app_env):
    """QueryRefine Processor를 선택하는 함수"""
    if app_env == "prototype":
        return PrototypeQueryRefineProcessor(es_client, faiss_client)
    elif app_env == "development":
        return DevelopmentQueryRefineProcessor(es_client, faiss_client)
    elif app_env == "production":
        return ProductionQueryRefineProcessor(es_client, faiss_client)
    raise ValueError("지원하지 않는 환경입니다.")


class BaseQueryRefineProcessor:
    """Base QueryRefine Processor"""
    
    # 클래스 변수로 패턴 정의
    BAD_PATTERN_HANGUL = '[ㄱ-ㅎㅏ-ㅣ]'
    BAD_PATTERN_SYMBOLS = r'[!@#\$%\^&\*\(\)\-_=\+\[\]\{\};:\'",<>\./\?\\\|`~]+$'
    
    def __init__(self, es_client, faiss_client):
        self.es = es_client
        self.es_index = settings.REFINE_ES_INDEX_NAME
        self.faiss_client = faiss_client
        self.llm_host = settings.REFINE_LLM_HOST
        self.llm_name = settings.REFINE_LLM_NAME
        
        # Configuration
        self.N_GRAM_SETTING = [1, 2, 3, 4]
        self.MATCH_THRESHOLD = 0.85

    async def refine_general(self, query: str) -> dict:
        """일반적인 쿼리 정제"""
        refined_query = re.sub(self.BAD_PATTERN_HANGUL, '', query)
        if re.match(self.BAD_PATTERN_SYMBOLS, refined_query):
            refined_query = re.sub(self.BAD_PATTERN_SYMBOLS, '', refined_query).strip()
        return {'refined_query': refined_query}

    async def refine_for_keyword_search(self, query: str) -> dict:
        """키워드 검색용 쿼리 정제"""
        raise NotImplementedError("이 메서드는 하위 클래스에서 구현되어야 합니다.")

    async def refine_for_vector_search(self, query: str) -> dict:
        """벡터 검색용 쿼리 정제"""
        raise NotImplementedError("이 메서드는 하위 클래스에서 구현되어야 합니다.")


class PrototypeQueryRefineProcessor(BaseQueryRefineProcessor):
    """Prototype 환경의 QueryRefine Processor"""
             
    async def refine_for_keyword_search(self, query: str) -> dict:
        """
        Elasticsearch analyzer 기반 키워드 검색용 쿼리 정제
        1. 기본 전처리
        2. Elasticsearch analyzer를 통한 토큰화
        """
        res = await self.es.indices.analyze(
            index=self.es_index,
            body={
                "analyzer": "standard_analyzer",
                "text": query,
                "explain": False
            }
        )
        refined_query = ' '.join(token['token'] for token in res['tokens'])
        return {'refined_query': refined_query}

    async def refine_for_vector_search(self, query: str) -> dict:
        """
        Vector 검색용 쿼리 정제
        1. 기본 전처리
        2. N-gram 기반 유사 문구 식별
        3. 식별된 문구 교체
        """
        revised_query_tokens = await self.faiss_client.filter_query(query)
        revised_query = ' '.join(revised_query_tokens).replace('[REMOVED', '')
        refined_query = await self.faiss_client.grammar_error_correction(revised_query)
        return {'refined_query': refined_query}


class DevelopmentQueryRefineProcessor(BaseQueryRefineProcessor):
    """Development 환경의 QueryRefine Processor"""
    
    @log_execution_time_async
    async def refine_for_keyword_search(self, query: str) -> dict:
        """
        Elasticsearch analyzer 기반 키워드 검색용 쿼리 정제
        1. 기본 전처리
        2. Elasticsearch analyzer를 통한 토큰화
        """
        res = await self.es.indices.analyze(
            index=self.es_index,
            body={
                "analyzer": "kdb_nori_analyzer",
                "text": query,
                "explain": False
            }
        )
        return {'refined_query': ' '.join(token['token'] for token in res['tokens']).strip()}
        
    @log_execution_time_async
    async def refine_for_vector_search(self, query: str) -> dict:
        """
        Vector 검색용 쿼리 정제
        Development 환경에서는 간단한 정제만 수행
        """
        # 간단한 쿼리 반환 (복잡한 synonym 처리는 제거)
        return {'refined_query': [query]}


class ProductionQueryRefineProcessor(BaseQueryRefineProcessor):
    """Production 환경의 QueryRefine Processor"""
    
    @log_execution_time_async
    async def refine_for_keyword_search(self, query: str) -> dict:
        """
        Elasticsearch analyzer 기반 키워드 검색용 쿼리 정제
        1. 기본 전처리
        2. Elasticsearch analyzer를 통한 토큰화
        """
        res = await self.es.indices.analyze(
            index=self.es_index,
            body={
                "analyzer": "kdb_nori_analyzer",
                "text": query,
                "explain": False
            }
        )
        return {'refined_query': ' '.join(token['token'] for token in res['tokens']).strip()}
        
    @log_execution_time_async
    async def refine_for_vector_search(self, query: str) -> dict:
        """
        Vector 검색용 쿼리 정제
        Production 환경에서는 간단한 정제만 수행
        """
        return {'refined_query': [query]}
