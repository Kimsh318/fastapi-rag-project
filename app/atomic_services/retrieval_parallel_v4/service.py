# ================== Python 기본 라이브러리 ==================
import logging
import asyncio
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

# ================== app의 다른 경로에서 import ==================
from app.utils.logging_utils import log_simple_data_to_es, calculate_iso_time_difference, log_execution_time_async

# ================== 현재 모듈의 기능 import ==================
from .helpers import merge_search_results_with_rrf, get_highlighted_snippet, format_search_results
from .processors import get_keyword_search_processor, get_vector_search_processor, get_rerank_processor

# ================== Retrieval 클래스 정의 ==================

logger = logging.getLogger(__name__)

class SemanticSearchService:
    def __init__(self, validation_service=None, refine_service=None, faiss_client=None, top_k=2, app_env='prototype'):
        self.validation_service = validation_service
        self.refine_service = refine_service
        self.vs_processor = get_vector_search_processor(faiss_client, top_k, app_env) if faiss_client else None
        
    # @log_api_call(index_type="service")
    @log_execution_time_async # #
    async def search(self, query: str, field: str, user_specified_doc_types: list, do_refine=True, do_validate=True):
        
        vs_query = query
        if do_refine:
            # 질의 정제
            s_time = time.time()
            refined_result = await self.refine_service.refine(query, task="VectorSearch")
            refined_query = refined_result['refined_query']
            vs_query = refined_query
            
        if do_validate:
            # 질의 검증
            validation_results = await self.validation_service.validate(query=vs_query)
            if not validation_results['is_valid']: return None
        # 문서 검색
        vs_results = await self.vs_processor.search(vs_query, field, user_specified_doc_types)

        return {
            "vs_results": vs_results["vs_results"],
            "vs_query": vs_results["vs_query"]
        }

        
class KeywordSearchService:
    def __init__(self, validation_service=None, refine_service=None, es_client=None, es_info=None, top_k=2, app_env='prototype'):
        self.validation_service = validation_service
        self.refine_service = refine_service

        es_api_host = es_info['es_api_host']
        es_index = es_info['es_index']

        self.ks_processor = get_keyword_search_processor(es_client, es_index, top_k, app_env)
        
    # @log_api_call(index_type="service")
    @log_execution_time_async
    async def search(self, query: str, user_specified_doc_types:list[str], do_refine=True, do_validate=True):

        ks_query = query
        if do_refine:
            # 질의 정제
            s_time = time.time()
            refined_result = await self.refine_service.refine(query, task="KeywordSearch")
            refined_query = refined_result['refined_query']
            ks_query = refined_query

        if do_validate:
            # 질의 검증
            validation_results = await self.validation_service.validate(query=ks_query)
            if not validation_results['is_valid']: return None      # 검증결과가 유효하지 않을 경우, 검색 종료
        
        # 문서 검색
        ks_results = await self.ks_processor.search(ks_query, user_specified_doc_types)

        return {"ks_results": ks_results["ks_results"]}

class HybridSearchService:
    def __init__(self, refine_service, validation_service, semantic_search_service: SemanticSearchService, keyword_search_service: KeywordSearchService, k_weight=0.5, s_weight=0.5, es_client=None, es_index=None):
        self.refine_service = refine_service
        self.validation_service = validation_service
        
        self.semantic_search_service = semantic_search_service
        self.keyword_search_service = keyword_search_service
        
        self.k_weight = k_weight    # 키워드 검색 가중치
        self.s_weight = s_weight    # 의미 검색 가중치

        # 하이라이트 검색용 ES
        self.es_client = es_client 
        self.es_index = es_index


    # @log_api_call(index_type="service")
    # @log_execution_time_async
    async def search(self, query: str, field:str, user_specified_doc_types:list[str], do_refine=True, do_validate=True):
        vs_query, ks_query = query, query
        
        refine_tasks = []
        validate_tasks = []
        if do_refine:
            s_time = time.time()
            
            # 방법 2) 동기 방식 처리 : Semantic Search용 refined query를 Vector Searcy 용 query refine에 활용
            pre_refined_query = await self.refine_service.refine(query, task = "General")
            vs_refined_result = await self.refine_service.refine(pre_refined_query["refined_query"], task="VectorSearch")
            ks_refined_result = await self.refine_service.refine(pre_refined_query["refined_query"], task="KeywordSearch")
            
            vs_queries = vs_refined_result['refined_query']
            ks_query = ks_refined_result['refined_query'][0]
            
              
        if do_validate:
            s_time = time.time()
            for query in vs_queries:
                validate_tasks.append(self.validation_service.validate(query))
            validate_tasks.append(self.validation_service.validate(ks_query))
            
            results = await asyncio.gather(*validate_tasks)
            is_vs_valid = [result.get('is_valid',False) for result in results[:len(vs_queries)]]
            is_ks_valid = results[len(vs_queries)].get('is_valid', False)

            if not all(is_vs_valid) or not is_ks_valid: 
                return None
        
        # =====================비동기 검색 서비스 병렬 실행 : Semantic search, Keyword search ================

        # 방법 2) CPU 바운드 작업이지만, IO바운드 작업처럼 코루틴 객체를 만들어 쓰레드에서 처리
        ## Python GIL 이슈(한 번에 1개 쓰레드만 실행) 때문에 효율적으로 병렬처리 되지 않을 수 있음
        semantic_search_task = asyncio.create_task(
            self.semantic_search_service.search(vs_queries, field, user_specified_doc_types, do_refine=False, do_validate=False)
        )
        keyword_search_task = asyncio.create_task(
            self.keyword_search_service.search(ks_query, field, user_specified_doc_types, do_refine=False, do_validate=False)  # I/O 바운드 작업
        )
        # 두 태스크를 병렬로 실행하고 결과를 기다림
        semantic_results, keyword_results = await asyncio.gather(
            semantic_search_task, keyword_search_task
        )

        vs_query = semantic_results["vs_query"]

        if semantic_results == None or keyword_results == None:
            print("[Hybrid search] semantic results and keyword results All None")
            return None
        elif semantic_results == None:
            print("[Hybrid search] semantic results: None")
        elif keyword_results == None:
            print("[Hybrid search] keyword results: None")

        # =========================================================================================

        # 검색 결과 합치기
        merged_results = merge_search_results_with_rrf(ks_query, vs_query,es_results=keyword_results["ks_results"], vs_results=semantic_results["vs_results"],  k_weight=self.k_weight, v_weight=self.s_weight)
        merged_results = [r for r in merged_results if len(r['chunk_context']) >= 15]

        return {"hs_results" : highlighted_results,
               "vs_query": vs_query,
               "ks_query": ks_query,
               "refined_query": pre_refined_query["refined_query"]}

class HybridSearchwithRerankService:
    def __init__(self, validation_service, hybrid_search_service: HybridSearchService, reranker, rerank_cuda='cuda:0', top_k=2, rerank_threshold=0, app_env='prototype' ):
        self.validation_service = validation_service
        self.hybrid_search_service = hybrid_search_service
        self.rerank_processor = get_rerank_processor(reranker=reranker, 
                                                     rerank_cuda=rerank_cuda, 
                                                     top_k=top_k,
                                                     rerank_threshold=rerank_threshold, 
                                                     app_env=app_env)
        
    async def search(self, query: str, user_specified_doc_types:list[str], do_refine=True, do_validate=True):
        
        s_time = time.time()
        # 하이브리드 검색 수행
        hybrid_search_results = await self.hybrid_search_service.search(query=query, 
                                                                        field=field,
                                                                        user_specified_doc_types=user_specified_doc_types,
                                                                        do_refine=True, # True
                                                                        do_validate=True)
        
        
        if hybrid_search_results == None:
            return None
        
        if hybrid_search_results.get('vs_query'):
            reranker_query = hybrid_search_results['vs_query']
        else:
            reranker_query = hybrid_search_results['refined_query']

        print(f"\n===================Reranker 검색어: {reranker_query}\n\n")
        reranked_results = await self.rerank_processor.rerank(query=reranker_query, answers = hybrid_search_results['hs_results'])        
        # 재정렬 결과 유효성 검증(ex. 유사도 점수 < 0)
        validated_results = await self.rerank_processor.validate_results(hybrid_results=hybrid_search_results["hs_results"], 
                                                                                                rerank_results=excluded_results["excluded_rerank_docs"])
        
        
        # 포맷 변환 : Response 할 정보들만 선정
        #print("============Total Rerank Processor elapsed time : ", time.time()-s_time)
        return {
            "hs_w_rerank_results" : validated_results['validated_docs'],
            "best_biz_type": validated_results.get('best_biz_type',''),
            "vs_query": reranker_query
        }
