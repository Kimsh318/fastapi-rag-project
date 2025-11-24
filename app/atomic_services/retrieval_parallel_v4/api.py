# ====================== Python 기본 라이브러리 ======================
import os
import time
import asyncio
import logging
from functools import lru_cache

# ====================== 외부 라이브러리 ======================
from fastapi import APIRouter, Depends, HTTPException

# ====================== app 모듈 ======================
from app.core.config import settings
from app.utils.logging_utils import (
    log_api_call, 
    extract_function_name_from_traceback, 
    log_error_to_es,
    log_execution_time_async,
    log_api_data,
)
from app.utils.es_client import get_es_client
# from app.main import app

# ====================== app.atomic_services 하위 모듈 ======================
from app.atomic_services.queryValidation.service import QueryValidationService
from app.atomic_services.queryValidation.helpers import (
    get_tokenizer_info, 
    get_validation_criteria
)

from app.atomic_services.queryRefine.service import QueryRefineService
from app.atomic_services.queryRefine.helpers import (
    #get_es_client, 
    get_faiss_client as get_refine_faiss_client
)

# ====================== 현재 모듈의 기능들 ======================
from .service import (
    SemanticSearchService, 
    KeywordSearchService, 
    HybridSearchService, 
    HybridSearchwithRerankService
)
from .models import (
    SemanticSearchRequest, 
    KeywordSearchRequest, 
    HybridSearchRequest, 
    HybridSearchwithRerankRequest, 
    RetrievalResponse
)
from .helpers import (
    get_faiss_client, 
    #get_es_client, 
    get_reranker, 
    format_search_results
)


# ====================== Service Initialization Functions ==========================
# 1번만 초기화 되도록하여, embedding 모델이 재로드되는 상황 방지
## TODO : maxsize=None으로 지정하면, 캐시의 크기에 제한이 없어짐. 적절한 캐시 크기 탐색 필요
## OPTIMIZE : 단일 프로세스에 대해서는 캐시가 유지되지만, 멀티프로세싱을 하게 되면 각 프로세스간 서로 독립된 인메모리에 저장하기에, 메모리 비효율이 발생할 수 있음
##            대안 1) events에서 global로 embedding 모델 로드
##            대안 2) embedding 모델 별도 서빙(vLLM 처럼)

logger = logging.getLogger(__name__)

@lru_cache(maxsize=None, typed=False)   
def get_query_validation_service():
    logger.info('-----------init query validation service')
    tokenizer_info = get_tokenizer_info(settings)
    validation_criteria = get_validation_criteria(settings)

    return QueryValidationService(tokenizer_info=tokenizer_info, validation_criteria=validation_criteria, app_env=settings.APP_ENVIRONMENT)

@lru_cache(maxsize=None, typed=False)   
def get_query_refine_service():
    logger.info('-----------init refine service')
    es_client = get_es_client()
    faiss_client = get_refine_faiss_client()
    return QueryRefineService(es_client, faiss_client, app_env=settings.APP_ENVIRONMENT)

@lru_cache(maxsize=None, typed=False)   
def get_semantic_search_service(do_refine=True):
    validation_service = get_query_validation_service()
    if do_refine:
        refine_service = get_query_refine_service()
    else:
        refine_service = None
    faiss_client = get_faiss_client()
    logger.info('-----------init semantic search service')
    return SemanticSearchService(validation_service=validation_service, \
                                refine_service=refine_service, \
                                faiss_client=faiss_client, \
                                top_k=settings.VS_TOP_K, \
                                app_env=settings.APP_ENVIRONMENT)

@lru_cache(maxsize=None, typed=False)   
def get_keyword_search_service(do_refine=True):
    validation_service = get_query_validation_service()
    #refine_service = get_query_refine_service()
    if do_refine:
        refine_service = get_query_refine_service()
    else:
        refine_service = None
    logger.info('-----------init keyword search service')
    es_client = get_es_client()
    es_info = {'es_api_host': settings.ES_API_HOST,
                'es_index': settings.ELASTICSEARCH_DOC_DB_NM}
    return KeywordSearchService(validation_service=validation_service,\
                                refine_service=refine_service, \
                                es_client=es_client, \
                                es_info=es_info, \
                                top_k=settings.ES_TOP_K, \
                                app_env=settings.APP_ENVIRONMENT)

@lru_cache(maxsize=None, typed=False)   
def get_hybrid_search_service():
    # 검색결과 Highlight에 사용할 es_client
    es_client = get_es_client()
    es_index = settings.ELASTICSEARCH_DOC_DB_NM

    # Semantic, keyword service 초기화
    validation_service = get_query_validation_service()
    refine_service = get_query_refine_service()

    semantic_service = get_semantic_search_service(do_refine=False)
    keyword_service = get_keyword_search_service(do_refine=False)
    
    logger.info('-----------init hybrid search search service')
    
    return HybridSearchService(refine_service=refine_service, 
                               validation_service=validation_service,
                               semantic_search_service=semantic_service,
                               keyword_search_service=keyword_service,
                               k_weight=settings.K_WEIGHT,
                               s_weight=1-settings.K_WEIGHT,
                               es_client=es_client, 
                               es_index=es_index)

@lru_cache(maxsize=None, typed=False)   
def get_hybrid_search_with_rerank_service():
    logger.info("HybridSearchwithRerankService 초기화 시작")
    reranker = get_reranker(model_path=settings.RERANKER_MODEL_PATH , device=settings.RERANKER_CUDA, app_env=settings.APP_ENVIRONMENT)
    
    validation_service = get_query_validation_service()
    #refine_service = get_query_refine_service()
    
    hybrid_search_service = get_hybrid_search_service()

    service = HybridSearchwithRerankService(validation_service, 
                                         hybrid_search_service, 
                                         reranker=reranker, 
                                         top_k=settings.RERANKER_TOP_K, 
                                         lower_priority_docs=settings.RERANKER_LOWER_PRIORITY_DOCS, 
                                         excluded_docs=settings.RERANKER_EXCLUDED_DOCS, 
                                         rerank_threshold= settings.RERANKER_SCORE_THRESHOLD,
                                         app_env=settings.APP_ENVIRONMENT)


    # await service.search(query='기한전상환수수료의 정의를 알려줘', 
    #                                     user_specified_doc_types=['여신 내규/지침',  '행통(여신,여신심화)', '여신 FAQ'], 
    #                                     user_excluded_docs=settings.RERANKER_EXCLUDED_DOCS)
    logger.info("HybridSearchwithRerankService 초기화 완료: %s", id(service))
    return service
    

#=================== API router =====================

router = APIRouter()  # API Router 초기화

@router.post("/semantic_search", response_model=RetrievalResponse)
@log_execution_time_async # @log_api_call(endpoint='semantic_search', index_type="api")
async def semantic_search(request: SemanticSearchRequest, service: SemanticSearchService = Depends(get_semantic_search_service)):
    """
    검색 엔드포인트.
    클라이언트로부터 쿼리를 받아 관련 문서를 반환합니다.
    """

    documents = await service.search(request.query, field = request.field, user_specified_doc_types = request.user_specified_doc_types)
    if not documents:
        # client에서는 status_code 500만 응답되어, 우선 주석처리해둠
        # status_code에 따라 서로 다른 처리를 할 수 있도록 처리해야할 것
        raise HTTPException(status_code=400, detail=f"유효하지 않은 질의입니다.")    
    #return {'documents': documents['vs_results']}
    response = await format_search_results(documents['vs_results'])


    return {
        'documents':response,
        'vs_query': request.query
        }


@router.post("/keyword_search", response_model=RetrievalResponse)
@log_execution_time_async # @log_api_call(endpoint='keyword_search', index_type="api")
async def keyword_search(request: KeywordSearchRequest, service: KeywordSearchService = Depends(get_keyword_search_service)):
    """
    검색 엔드포인트.
    클라이언트로부터 쿼리를 받아 관련 문서를 반환합니다.
    """
    documents = await service.search(request.query, field = request.field, request.user_specified_doc_types)
    if not documents:
        raise HTTPException(status_code=400, detail=f"유효하지 않은 질의입니다.")    
    #return documents['ks_results']

    response = await format_search_results(documents['ks_results'])
    return {
        'documents':response,
        'vs_query': request.query
        }


@router.post("/hybrid_search", response_model=RetrievalResponse)
@log_execution_time_async # @log_api_call(endpoint='hybrid_search', index_type="api")
async def hybrid_search(request: HybridSearchRequest, service: HybridSearchService = Depends(get_hybrid_search_service)):
    """
    하이브리드 검색 엔드포인트.
    클라이언트로부터 쿼리를 받아 의미 기반과 키워드 기반 결과를 결합하여 반환합니다.
    """
    documents = await service.search(request.query, field = request.feild, request.user_specified_doc_types)
    if not documents:
        raise HTTPException(status_code=400, detail=f"유효하지 않은 질의입니다.")    

    vs_query = documents['vs_query']
    response = await format_search_results(documents['hs_results'])
    return {
        'documents':response,
        'vs_query': request.query
        }

    

@router.post("/hybrid_search_with_rerank", response_model=RetrievalResponse)
@log_execution_time_async # @log_api_call(endpoint='hybrid_search_with_rerank', index_type="api")
async def hybrid_search_with_rerank(request: HybridSearchwithRerankRequest, service: HybridSearchwithRerankService = Depends(get_hybrid_search_with_rerank_service)):
#async def hybrid_search_with_rerank(request: HybridSearchwithRerankRequest, service: HybridSearchwithRerankService = Depends(router.app.state.hybrid_search_with_rerank_service)):
    """
    하이브리드 검색과 Rerank 엔드포인트.
    클라이언트로부터 쿼리를 받아 하이브리드 검색을 수행한 후, Rerank 모델로 문서 순위를 재정렬하여 반환합니다.
    """
    pid = os.getpid()
    
    s_time = time.time()
    try:
        documents = await service.search(query=request.query, 
                                        field = request.field,
                                        user_specified_doc_types=request.user_specified_doc_types)

        if not documents:
            raise HTTPException(status_code=400, detail=f"유효하지 않은 질의입니다.")    
    
        response = await format_search_results(documents['hs_w_rerank_results'])
        vs_query = documents['vs_query']

        asyncio.create_task(log_api_data("hybrid_search_with_rerank", 
                     request.query, 
                     ', '.join([doc.chunk_id for doc in response])))
    except Exception as e:
        error_message = str(e)
        function_name = extract_function_name_from_traceback()
        asyncio.create_task(log_error_to_es(function_name, error_message, request.query))
        raise  # 원래의 예외를 다시 발생 시킴
    
    return {
        "documents": response,
        "vs_query": vs_query
    }