from fastapi import FastAPI
import httpx

from app.core.config import settings

from elasticsearch import Elasticsearch, AsyncElasticsearch
from prometheus_client import start_http_server


from app.atomic_services.queryFormatting.api import get_query_formatting_service
from app.atomic_services.retrieval_parallel_v4.api import get_hybrid_search_with_rerank_service
from app.atomic_services.retrieval_parallel_v4.api import hybrid_search_with_rerank
from app.atomic_services.retrieval_parallel_v4.models import HybridSearchwithRerankRequest

from app.utils.es_client import get_es_client

# TODO : 전역변수로 정의하는게 더 좋은지, 의사결정 필요
## 1. ChromaDB
## 2. Elastic Search
## 3. 임베딩 모델

app = FastAPI()

async def startup_event(app: FastAPI):
    """
    애플리케이션 시작 시 실행되는 이벤트 핸들러.
    필요한 초기화 작업을 수행합니다.
    """
    # 예: 데이터베이스 연결, 캐시 초기화 등
    # 프로메테우스 서버 시작 등

    # TODO : 모델 로드
    # load_models()

    # TODO : ChromaDB 로드
    # app.state.chromadb_client = chromadb.HttpClient(host=settings.CHROMADB_API_HOST, port=settings.CHROMADB_API_PORT)

    # TODO : ES DB 로드
    print('log es client initialized')
    if settings.APP_ENVIRONMENT == 'prototype' or settings.APP_ENVIRONMENT == 'development':
        app.state.log_es_client = AsyncElasticsearch(settings.ES_API_HOST)
        app.state.highlight_es_client = AsyncElasticsearch(settings.ES_API_HOST)
    elif settings.APP_ENVIRONMENT == 'production' :
        # app.state.log_es_client = AsyncElasticsearch(settings.ES_API_HOST, http_auth=("kdb", "kdbAi1234!"))
        # app.state.highlight_es_client = AsyncElasticsearch(settings.ES_API_HOST, http_auth=("kdb", "kdbAi1234!"))
        app.state.log_es_client= get_es_client()
        app.state.highlight_es_client = get_es_client()

        
    # Service 초기화 함수
    app.state.query_formatting_service = get_query_formatting_service()
    app.state.hybrid_search_with_rerank_service = get_hybrid_search_with_rerank_service()


    # TODO :추후 다시 주석 해제 필요
    # get_query_formatting_service()
    # get_hybrid_search_with_rerank_service()

    # request = HybridSearchwithRerankRequest(
    #                 query='기한전상환수수료의 정의를 알려줘', 
    #                 user_specified_doc_types=['여신 내규/지침',  '행통(여신,여신심화)', '여신 FAQ'],
    #                 field="corporate",
    #                 excluded_docs=[],
    #                 user_id='1234',
    #                 session_id= '1234',
    #         )
    # await hybrid_search_with_rerank(request=request)

    # Prometheus 메트릭 서버 시작
    # start_http_server(8001)
    
    # return app

async def shutdown_event():
    """
    애플리케이션 종료 시 실행되는 이벤트 핸들러.
    리소스 해제 작업을 수행합니다.
    """
    # 예: 데이터베이스 연결 해제, 로그 종료 등
    # 프로메테우스 연동 해제
    pass
