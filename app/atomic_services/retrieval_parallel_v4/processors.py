# ================= Python 기본 라이브러리 =================
import os
import math
import time
from typing import List, Dict, Tuple
import logging
# ================= 외부 라이브러리 ========================
import chromadb
import numpy as np
from copy import deepcopy

# ================= app 모듈 ==============================
from app.core.config import settings
from app.utils.logging_utils import log_execution_time_async

# ================= 현재 모듈 ==============================
from .helpers import get_target_doc_types, get_elasticsearch_body


logger = logging.getLogger(__name__)

# ================= Processor 초기화 ====================
def get_keyword_search_processor(es_client, es_index, top_k, app_env):
    if app_env == "prototype":
        return PrototypeKeywordSearchProcessor(es_client, es_index, top_k)
    elif app_env == "development":
        return DevelopmentKeywordSearchProcessor(es_client, es_index, top_k)
    elif app_env == "production":
        return ProductionKeywordSearchProcessor(es_client, es_index, top_k)
    else:
        raise ValueError("지원하지 않는 환경입니다.")

def get_vector_search_processor(chromadb_collection, top_k, app_env):
    if app_env == "prototype":
        return PrototypeVectorSearchProcessor(chromadb_collection, top_k)
    elif app_env == "development":
        return DevelopmentVectorSearchProcessor(chromadb_collection, top_k)
    elif app_env == "production":
        return ProductionVectorSearchProcessor(chromadb_collection, top_k)    
    else:
        raise ValueError("지원하지 않는 환경입니다.")

def get_rerank_processor(reranker, rerank_cuda, top_k, lower_priority_docs, excluded_docs, rerank_threshold, app_env):
    if app_env == "prototype":
        return PrototypeRerankProcessor(reranker, rerank_cuda, top_k, lower_priority_docs, excluded_docs,  threshold=rerank_threshold)
    elif app_env == "development":
        return DevelopmentRerankProcessor(reranker=reranker, 
                                          rerank_cuda=rerank_cuda, 
                                          top_k=top_k, 
                                          threshold=rerank_threshold)
    elif app_env == "production":
        return ProductionRerankProcessor(reranker=reranker, 
                                          rerank_cuda=rerank_cuda, 
                                          top_k=top_k, 
                                          threshold=rerank_threshold)
    else:
        raise ValueError("지원하지 않는 환경입니다.")


# ============ Keyword Search Processor ====================
class BaseKeywordSearchProcessor:
    def __init__(self, es_client, es_index, top_k):
        self.es_client = es_client
        self.es_index = es_index
        self.top_k = top_k      

    def refine_query(self, query:str):
        raise NotImplementedError("이 메서드는 하위 클래스에서 구현되어야 합니다.")

    def search(self, query: str, user_specified_doc_types: list):
        raise NotImplementedError("이 메서  드는 하위 클래스에서 구현되어야 합니다.")

    def _get_document_by_id(self, chunk_id: str):
        es_body = self.get_elasticsearch_body(chunk_id, search_type='doc_id')
        es_results = self.es.search(index=self.es_index, body=es_body)
        final_es_results = [{
                'chunk_id': hit['_id'],
                'chunk_context': hit['_source']['chunk_context'],
                'doc_type': hit['_source']['doc_type'],
                'biz_type': hit['_source']['biz_type'],
                'chunk_src': hit['_source']['chunk_src']
            } for hit in es_results['hits']['hits']]
        
        return final_es_results[0] 


class PrototypeKeywordSearchProcessor(BaseKeywordSearchProcessor):
    # @log_api_call(index_type="processor")
    async def refine_query(self, query:str)->str:
        return {'refined_query': query}
    # @log_api_call(index_type="processor")
    async def search(self, query:str, user_specified_doc_types:list[str]):
        # 프로토타입용 간단한 검색 구현
        final_ks_results = [{
            'query': query,
            'chunk_id': '1234',
            'chunk_context': f'이건 KS 결과 context #{i}',
            'score': 0.3,
            'doc_type': '행통교재',
            'doc_id':'여신1권',
            'chunk_src': '행통교재',
            'highlight': [f'이건 KS 결과 context의 highlight #{i}'],
        } for i in range(self.top_k)]


        return {'ks_results': final_ks_results}

class DevelopmentKeywordSearchProcessor(BaseKeywordSearchProcessor):
    # @log_api_call(index_type="processor")
    async def refine_query(self, query:str)->str:
        return {'refined_query': query}
        
    # @log_api_call(index_type="processor")
    @log_execution_time_async
    async def search(self, query, field, user_specified_doc_types):
        # 실제 개발용 검색 구현
        print(f"\n=======================최종 입력된 키워드 검색어: {query}\n\n")
        target_doc_types, target_biz_types  = get_target_doc_types(field, user_specified_doc_types)
        es_body = get_elasticsearch_body(query, target_biz_types=target_biz_types, target_doc_types=target_doc_types, top_k=self.top_k*2)
        ks_results = await self.es_client.search(index=self.es_index, body=es_body)

        final_ks_results = []
        for hit in ks_results['hits']['hits']:

            highlight = [''] # ES 검색결과가 없을 경우 highlight는 빈문자열
            if  'highlight' in hit: 
                highlight = hit['highlight']['chunk_context']
            
            final_ks_results.append({
                'query': query,
                'chunk_id': hit['_id'],
                'chunk_context': hit['_source']['chunk_context'],
                'score': hit['_score'],
                'doc_type': hit['_source']['doc_type'],
                'biz_type': hit['_source']['biz_type'],
                'chunk_src': hit['_source']['chunk_src'],
                'highlight': highlight,    # list[str]
            })
        
        return {'ks_results': final_ks_results}

class ProductionKeywordSearchProcessor(BaseKeywordSearchProcessor):
    # @log_api_call(index_type="processor")
    async def refine_query(self, query:str)->str:
        return {'refined_query': query}
        
    # @log_api_call(index_type="processor")
    # @log_execution_time_async
    async def search(self, query, user_specified_doc_types):
        # 실제 개발용 검색 구현
        target_doc_types = get_target_doc_types(user_specified_doc_types)
        es_body = get_elasticsearch_body(query, target_doc_types=target_doc_types, top_k=self.top_k)
        ks_results = await self.es_client.search(index=self.es_index, body=es_body)
 
        final_ks_results = []
        for hit in ks_results['hits']['hits']:

            highlight = [''] # ES 검색결과가 없을 경우 highlight는 빈문자열
            if  'highlight' in hit: 
                highlight = hit['highlight']['chunk_context']
            
            final_ks_results.append({
                'query': query,
                'chunk_id': hit['_id'],
                'chunk_context': hit['_source']['chunk_context'],
                'score': hit['_score'],
                'doc_type': hit['_source']['doc_type'],
                'biz_type': hit['_source']['biz_type'],
                'chunk_src': hit['_source']['chunk_src'],
                'highlight': highlight,    # list[str]
            })
        
        return {'ks_results': final_ks_results}


# ============ Vector Search Processor ====================

class BaseVectorSearchProcessor:
    def __init__(self, faiss_client, top_k):
        self.faiss_client = faiss_client
        self.top_k = top_k

    def refine_query(self, query:str):
        raise NotImplementedError("이 메서드는 하위 클래스에서 구현되어야 합니다.")

    def search(self, query: str):
        raise NotImplementedError("이 메서드는 하위 클래스에서 구현되어야 합니다.")

class PrototypeVectorSearchProcessor(BaseVectorSearchProcessor):
    # @log_api_call(index_type="processor")
    async def refine_query(self, query:str)->str:
        return {'refined_query': query}
    
    async def _fetch_search_results(self, query, distances, indices):
        final_vs_results = []
        for i in range(self.top_k * 2):
            idx = indices[0][i]
            if idx != -1:
                final_vs_results.append({
                    "query": query,
                    "score": 1 / distances[0][i],
                    "chunk_src": "은행법",
                    "chunk_id": "1234",
                    "chunk_context": '문서 index : '+str(idx),
                    "highlight": [""],
                    "doc_type": "법률"
                })
        return final_vs_results

    # @log_api_call(index_type="processor")
    async def search(self, query: str, target_service:str = "RETRIEVAL", user_specified_doc_type: str="내규/지침"):

        # 프로토타입용 간단한 벡터 검색 구현
        distances, indices = await self.faiss_client.query(query_text=query)
        final_vs_results = await self._fetch_search_results(query, distances, indices)
        return {'vs_results': final_vs_results}

class DevelopmentVectorSearchProcessor(BaseVectorSearchProcessor):
    @log_execution_time_async # @log_api_call(index_type="processor")
    async def refine_query(self, query:str)->str:
        return {'refined_query': query}

    async def _fetch_search_results(self, query, distances, indices, target_biz_types:list[str], target_doc_types:list[str]):
        final_vs_results = []
        SEARCH_K = self.faiss_client.search_k  # 가정: SEARCH_K는 top_k의 두 배로 설정
        FAISS_DOC_METADATA = self.faiss_client.metadata

        for i in range(SEARCH_K):
            idx = indices[i]
            if idx == -1: break  # 더 이상 검색결과가 없을 경우, fetch처리 종료

            doc_metadata = FAISS_DOC_METADATA[idx]
            # target_doc_types 리스트에 doc_type이 포함되어 있는 경우에만 결과에 추가
            if doc_metadata["biz_type"] in target_biz_types and doc_metadata["doc_type"] in target_doc_types:
                final_vs_results.append({
                    "query": query,
                    "chunk_id": doc_metadata["chunk_id"],
                    "chunk_src": doc_metadata["chunk_src"],
                    "doc_type": doc_metadata["doc_type"],
                    "biz_type": doc_metadata["biz_type"],
                    "score": distances[i],
                    "chunk_context": doc_metadata["chunk_context"],
                })
                # final_vs_results의 크기가 top_k * 2에 도달하면 루프를 중단합니다.
                # if len(final_vs_results) >= self.top_k * 2:
                #     break
        final_vs_results = sorted(final_vs_results, key=lambda x: float(x["score"]), reverse=True)
        final_vs_results = final_vs_results[:self.top_k*2]

        return final_vs_results

    # @log_api_call(index_type="processor")
    @log_execution_time_async
    async def search(self, queries: list[str], field: str, user_specified_doc_types: list=["내규/지침"]):
        # 실제 개발용 벡터 검색 구현
        target_doc_types, target_biz_types = get_target_doc_types(field, user_specified_doc_types)

        ## 비동기방식 검색(threadpool 활용)
        distances, indices = await self.faiss_client.query(query_texts = queries)
        results_with_scores = []

        for i, query in enumerate(queries):
            results = await self._fetch_search_results(query, distances[i], indices[i], target_biz_types = target_biz_types, target_doc_types = target_doc_types)
            if results:
                total_score = sum(result["score"] for result in results)
                results_with_scores.append((results, total_score))

        if not results_with_scores:
            print("f:\n===================최종 입력된 벡터 검색어: 벡터 검색 결과 없음\n")
            return {'vs_results: None, 'vs_query': None}

        best_results, best_score = max(results_with_scores, key = lambda item: item[1])
        best_query = best_results[0]["query"]       

        if best_results:
            print("f:\n===================최종 입력된 벡터 검색어: {best_query}\n")
        
        return {'vs_results': best_results, 'vs_query': best_query}



class ProductionVectorSearchProcessor(BaseVectorSearchProcessor):
    @log_execution_time_async # @log_api_call(index_type="processor")
    async def refine_query(self, query:str)->str:
        return {'refined_query': query}

    async def _fetch_search_results(self, query, distances, indices, target_biz_types:list[str], target_doc_types:list[str]):
        final_vs_results = []
        SEARCH_K = self.faiss_client.search_k  # 가정: SEARCH_K는 top_k의 두 배로 설정
        FAISS_DOC_METADATA = self.faiss_client.metadata

        for i in range(SEARCH_K):
            idx = indices[i]
            if idx == -1: break  # 더 이상 검색결과가 없을 경우, fetch처리 종료

            doc_metadata = FAISS_DOC_METADATA[idx]
            # target_doc_types 리스트에 doc_type이 포함되어 있는 경우에만 결과에 추가
            if doc_metadata["biz_type"] in target_biz_types and doc_metadata["doc_type"] in target_doc_types:
                final_vs_results.append({
                    "query": query,
                    "chunk_id": doc_metadata["chunk_id"],
                    "chunk_src": doc_metadata["chunk_src"],
                    "doc_type": doc_metadata["doc_type"],
                    "biz_type": doc_metadata["biz_type"],
                    "score": distances[i],
                    "chunk_context": doc_metadata["chunk_context"],
                })
                # final_vs_results의 크기가 top_k * 2에 도달하면 루프를 중단합니다.
                # if len(final_vs_results) >= self.top_k * 2:
                #     break
        final_vs_results = sorted(final_vs_results, key=lambda x: float(x["score"]), reverse=True)
        final_vs_results = final_vs_results[:self.top_k*2]

        return final_vs_results

    # @log_api_call(index_type="processor")
    @log_execution_time_async
    async def search(self, queries: list[str], field: str, user_specified_doc_types: list=["내규/지침"]):
        # 실제 개발용 벡터 검색 구현
        target_doc_types, target_biz_types = get_target_doc_types(field, user_specified_doc_types)

        ## 비동기방식 검색(threadpool 활용)
        distances, indices = await self.faiss_client.query(query_texts = queries)
        results_with_scores = []

        for i, query in enumerate(queries):
            results = await self._fetch_search_results(query, distances[i], indices[i], target_biz_types = target_biz_types, target_doc_types = target_doc_types)
            if results:
                total_score = sum(result["score"] for result in results)
                results_with_scores.append((results, total_score))

        if not results_with_scores:
            print("f:\n===================최종 입력된 벡터 검색어: 벡터 검색 결과 없음\n")
            return {'vs_results: None, 'vs_query': None}

        best_results, best_score = max(results_with_scores, key = lambda item: item[1])
        best_query = best_results[0]["query"]       

        if best_results:
            print("f:\n===================최종 입력된 벡터 검색어: {best_query}\n")
        
        return {'vs_results': best_results, 'vs_query': best_query}

# ================================
#  Rerank Processor
# ================================

class BaseRerankProcessor:
    def __init__(self, reranker, rerank_cuda, top_k: int = 10, threshold = 0):
        logger.info('Reranker Processor 초기화 시작')
        self.reranker = reranker
        self.rerank_cuda = rerank_cuda

        # rerank 결과 설정
        self.top_k = top_k 
        self.threshold = threshold
        
        # rerank 길이 설정
        self.len_reranker_max_support = 1024
        self.len_extra = 16
        self.len_overlap = 256

        logger.info('Reranker Processor 초기화 완료')
        
    def rerank(self, query: str, answers: list) -> list:
        raise NotImplementedError("이 메서드는 하위 클래스에서 구현되어야 합니다.")

    def validate_results(self, hybrid_results: list, rerank_results: list) -> list:
        raise NotImplementedError("이 메서드는 하위 클래스에서 구현되어야 합니다.")

    def _get_len_input_ids(self, _str: str) -> int:
        # FIXME : 임시로 구현한 토크나이저 사용했음
        # 주어진 문자열을 토큰화하여 토큰의 수를 반환하는 함수
        # len_str = self.reranker.tokenizer(_str, return_tensors='pt').to(self.rerank_cuda)['input_ids'].size()[1]
        
        
        len_str = len(self.reranker.tokenizer(_str, return_tensors='pt'))
        
        return len_str

    def _pre_rerank(self, query: str, answers: list):
        # Pre-ReRank를 수행하여 초기 재정렬 결과와 ID 리스트를 반환하는 함수
        pre_rerank_results = []
        ans_ranked_ids = []
        ans_reranked_ids = []

        for ans in answers:
            ans_ranked_ids.append(ans['chunk_id'])
            rerank_score = self.reranker.compute_score([query, ans['chunk_context']], max_length=self.len_reranker_max_support)
            
            #pre_rerank_results.append(deepcopy(ans).update({'score': rerank_score}))
            
            pre_rerank_results.append({
                'chunk_id': ans['chunk_id'],
                'chunk_context': ans['chunk_context'],
                'score': rerank_score,
                'doc_type': ans['doc_type'],
                'biz_type': ans['biz_type'],
                'chunk_src': ans['chunk_src'],
            })

            
        
            # # deepcopy를 사용하여 ans의 복사본을 만들고, 'score'를 추가합니다.
            # ans_copy = deepcopy(ans)
            # ans_copy.update({'score': rerank_score})
            # pre_rerank_results.append(ans_copy)  # 수정된 복사본을 리스트에 추가합니다.

        import json
        from copy import deepcopy
        tmp_r = [{'rerank_query': query,
                 'type': str(type(self.reranker))}]
        tmp_r.extend(deepcopy(pre_rerank_results))
        # with open(f"/workspace/dev/dev1_p/13_langchain_fastapi_backend_dev/v18/tmp_retrieve_results/{time.strftime('%H%M%S')}_(pre_rerank_before_sort){query[:5]}_result.json", 'w') as f:
            
        #     json.dump(tmp_r, f)

        
        # print("pre_rerank_results\n\n", pre_rerank_results)
        # 점수에 따라 내림차순으로 정렬
        pre_rerank_results.sort(key=lambda x: x['score'], reverse=True)
        for result in pre_rerank_results:
            ans_reranked_ids.append(result['chunk_id'])
        return pre_rerank_results, ans_ranked_ids, ans_reranked_ids

    def _compare_results(self, ans_ranked_ids, ans_reranked_ids, answers_length):
        # Hybrid 검색과 Rerank 결과를 비교하여 동일한지 확인하는 함수
        if answers_length >= self.top_k and ans_reranked_ids[:self.top_k] == ans_ranked_ids[:self.top_k]:
            return True
        elif answers_length < self.top_k and ans_reranked_ids == ans_ranked_ids:
            return True
        else:
            return False

    def _post_rerank(self, query: str, answers: list):
        # Chunk를 분할하여 Post-ReRank를 수행하는 함수
        LEN_QUERY = self._get_len_input_ids(query)  # 토큰화된 질의의 길이
        LEN_RERANKER_MAX_SUPPORT = self.len_reranker_max_support  # ReRanker의 최대 입력 길이
        LEN_EXTRA = self.len_extra  # 추가 토큰 길이
        BASE_LEN_CHUNK = LEN_RERANKER_MAX_SUPPORT - LEN_QUERY - LEN_EXTRA  # chunk의 최대 길이
        LEN_OVERLAP = self.len_overlap  # chunk 간 겹치는 길이

        RERANKER_TOKENIZER = self.reranker.tokenizer
        
        rerank_results = []

        for ans in answers:
            context_len = self._get_len_input_ids(ans['chunk_context'])
            if context_len > BASE_LEN_CHUNK:
                # chunk를 분할하여 처리
                num_rerank = math.ceil((context_len - BASE_LEN_CHUNK) / (BASE_LEN_CHUNK - LEN_OVERLAP)) + 1
                nl_fragments = []

                for i in range(num_rerank):
                    if i*(BASE_LEN_CHUNK - LEN_OVERLAP) + BASE_LEN_CHUNK <= get_len_input_ids(ans['chunk_context']):
                        nl_fragment = RERANKER_TOKENIZER.decode(RERANKER_TOKENIZER(ans['chunk_context'], return_tensors = 'pt').to(RERANKER_CUDA)['input_ids'][0][i*(BASE_LEN_CHUNK - LEN_OVERLAP):i*(BASE_LEN_CHUNK - LEN_OVERLAP)+BASE_LEN_CHUNK])
                        nl_fragments.append(nl_fragment)
                    else:
                        nl_fragment = RERANKER_TOKENIZER.decode(RERANKER_TOKENIZER(ans['chunk_context'], return_tensors = 'pt').to(RERANKER_CUDA)['input_ids'][0][i*(BASE_LEN_CHUNK - LEN_OVERLAP):get_len_input_ids(ans['chunk_context'])])
                        nl_fragments.append(nl_fragment)
                
                rerank_scores = []
                for fragment in nl_fragments:
                    rerank_score = self.reranker.compute_score([query, fragment], max_length=1024)
                    rerank_scores.append(rerank_score)
                try:
                    rerank_score = max(rerank_scores)
                except ValueError:
                    rerank_score = -100
            else:
                # chunk를 분할하지 않고 처리
                rerank_score = self.reranker.compute_score([query, ans['chunk_context']], max_length=1024)


            # deepcopy를 사용하여 ans의 복사본을 만들고, 'score'를 추가합니다.
            ans_copy = deepcopy(ans)
            ans_copy.update({'score': rerank_score})
            rerank_results.append(ans_copy)  # 수정된 복사본을 리스트에 추가합니다.

            # rerank_results.append({
            #     'chunk_id': ans['chunk_id'],
            #     'chunk_context': ans['chunk_context'],
            #     'score': rerank_score,
            #     'doc_type': ans['doc_type'],
            #     'chunk_src': ans['chunk_src'],
            # })

        # 점수에 따라 내림차순으로 정렬
        rerank_results.sort(key=lambda x: x['score'], reverse=True)
        
        # top_k 개수만큼 결과 반환
        return rerank_results[:self.top_k] if len(rerank_results) >= self.top_k else rerank_results

    def _format_output_from_rerank(self, rerank_results:list):
        """
        리랭크된 결과를 기반으로 outputs 리스트를 구성
        """
        return [
            {
                'chunk_id': reranked_item['chunk_id'], 
                'chunk_context': reranked_item['chunk_context'],
                'score': reranked_item['score'],
                'doc_type': reranked_item['doc_type'],
                'biz_type': reranked_item['biz_type'],
                'chunk_src': reranked_item['chunk_src'],
            }
            for reranked_item in rerank_results
        ]

    def _process_fallback_to_hybrid(self, answers):
        """
        리랭크 결과가 없을 경우, 하이브리드 검색 결과를 처리하여 반환
        """
        # 하이브리드 검색 결과에서 top_k 만큼의 결과를 가져옴
        if len(answers) >= self.top_k:
            answers = answers[:self.top_k]
        
        # 하이브리드 검색 결과가 유효한지 확인 (첫 번째 결과의 점수가 0보다 큰 경우)
        if answers and answers[0]['score'] > 0:
            # outputs = [
            #     {
            #         'chunk_id': ans['chunk_id'], 
            #         'chunk_context': ans['chunk_context'],
            #         'score': ans['score'],
            #         'doc_type': ans['doc_type'],
            #         'chunk_src': ans['chunk_src'],
            #     }
            #     for ans in answers
            # ]
            # return outputs, "ReRank 결과 존재하지 않으므로, 하이브리드 검색으로 전환"
            return answers, "ReRank 결과 존재하지 않으므로, 하이브리드 검색으로 전환"
        else:
            # 유효한 하이브리드 검색 결과도 없을 경우
            return [], "검색결과 없음"

    def _process_fallback_to_empty(self, answers):
        return [], "검색결과 없음"


class PrototypeRerankProcessor(BaseRerankProcessor):
    # @log_api_call(index_type="processor")
    async def rerank(self, query: str, answers: list) -> list:
        """
        주어진 query와 answers 리스트를 기반으로 Pre-Rerank를 수행하고,
        Hybrid 검색과의 결과 비교 후 Post-Rerank 수행 여부를 결정합니다.
        """
        # 1. Pre-ReRank 수행
        pre_rerank_results, ans_ranked_ids, ans_reranked_ids = self._pre_rerank(query, answers)
        
        # 2. Hybrid 검색과 Pre-Rerank 결과가 동일한지 비교
        is_same = self._compare_results(ans_ranked_ids, ans_reranked_ids, len(answers))
        
        # 3. 동일한 결과일 경우, Post-Rerank 생략 (연산 최적화)
        if is_same:
            return {"reranked_docs": \
                    pre_rerank_results[:self.top_k] if len(pre_rerank_results) >= self.top_k else pre_rerank_results}
        
        # 4. 다른 결과일 경우, 정확성을 높이기 위해 Post-Rerank 수행
        rerank_results = self._post_rerank(query, answers)
        return {"reranked_docs": rerank_results}

    # @log_api_call(index_type="processor")
    async def exclude_target_docs(self, rerank_results: list, user_excluded_docs: list=[]):
        return {"excluded_rerank_docs": rerank_results}


    # @log_api_call(index_type="processor")
    async def validate_results(self, hybrid_results:list, rerank_results: list) -> list:
        """
        리랭크가 실패할 경우 하이브리드 검색 결과로 대체하는 함수.
        리랭크 성공 여부 및 처리 결과 메시지를 함께 반환.
        """
        try:
            # 유효한 리랭크 결과가 존재하는지 확인
            if rerank_results and rerank_results[0]['score'] > 0:
                # outputs = self._format_output_from_rerank(rerank_results)
                outputs = rerank_results
                etc_info = "ReRank 정상적용"
                rerank_success_yn = True
            else:
                # 유효한 리랭크 결과가 없을 경우 하이브리드 검색으로 전환
                outputs, etc_info = self._process_fallback_to_hybrid(hybrid_results)
                rerank_success_yn = False

        except (IndexError, KeyError):
            # 예외 발생 시 리랭크 실패로 간주하고 하이브리드 검색으로 전환
            outputs, etc_info = self._process_fallback_to_hybrid(hybrid_results)
            rerank_success_yn = False

        return {"validated_docs" :outputs, 
                "is_rerank_success": rerank_success_yn, 
                "etc_info": etc_info}

    # @log_api_call(index_type="processor")
    async def adjust_docs(self, reranked_docs: List[Dict]) -> List[Dict]:
        # 문서 순위 보정 : 중요도가 낮은 문서들은 후순위로 배치
        adjusted_docs = sorted(reranked_docs, key=lambda x: x['chunk_src'] in self.lower_priority_docs, reverse=False)
        return {"adjusted_docs": adjusted_docs}

class DevelopmentRerankProcessor(BaseRerankProcessor):
    # @log_api_call(index_type="processor")
    @log_execution_time_async
    async def rerank(self, query: str, answers: list) -> list:
        """
        주어진 query와 answers 리스트를 기반으로 Pre-Rerank를 수행하고,
        Hybrid 검색과의 결과 비교 후 Post-Rerank 수행 여부를 결정합니다.
        """
        # 1. Pre-Rerank 수행 : Hybrid search와 비교하여 검색결과 변화가 발생하는지 확인하기 위함
        
        pre_rerank_results, ans_ranked_ids, ans_reranked_ids = self._pre_rerank(query, answers)
        rerank_results = pre_rerank_results
        
        # 2. Hybrid 검색과 Pre-Rerank 결과가 동일한지 비교
        is_same = self._compare_results(ans_ranked_ids, ans_reranked_ids, len(answers))
        
        # 3. 동일한 결과일 경우, Post-Rerank 생략 (연산 최적화)
        if is_same:
            return{"reranked_docs": \
                    pre_rerank_results}
        
        # 4. 다른 결과일 경우, 정확성을 높이기 위해 Post-Rerank 수행
        rerank_results = self._post_rerank(query, answers)
        
        return {"reranked_docs": rerank_results, "type": "dev results"}
    
    # @log_api_call(index_type="processor")
    @log_execution_time_async
    async def validate_results(self, hybrid_results:list, rerank_results: list) -> list:
        """
        리랭크가 실패할 경우 하이브리드 검색 결과로 대체하는 함수.
        리랭크 성공 여부 및 처리 결과 메시지를 함께 반환.
        """
        def calcaulate_average_score_by_biz_type(search_results:list) -> dict:
            biz_type_scores = {}

            for i, result in enumerate(search_results):
                biz_type = result['biz_type']
                socre = result['score']
                if score > self.threshold:
                    if biz_type not in biz_type_scores:
                        biz_type_scores[biz_type] = {'total_weighted_score': 0, 'count': 0, 'results': []}
                    if len(biz_type_scores[biz_type]['results']) <= self.top_k:
                        rank = i+1
                        weight = 1 / math.log2(rank + 1)
                        biz_type_scores[biz_type]['total_weighted_score'] += score * weight
                        biz_type_scores[biz_type]['count'] += 1
                        biz_type_scores[biz_type]['results'].append(result)
            average_scores = {}

            for biz_type, data in biz_type_scores.items():
                if data['count'] > 0:
                    average_scores[biz_type] = data['total_weighted_score']
                else:
                    average_scores[biz_type] = 0

            sorted_biz_types = dict(sorted(average_scores.items(), key = lambda item: item[1], reverse = True))
            return sorted_biz_types
            

        sorted_business_types_by_rerank_results = calcaulate_average_score_by_biz_type(rerank_results)
        print(f"\nReRank 결과 기준 Biztype별 평균 점수: {sorted_business_types_by_rerank_results}")
        
        try:
            # 유효한 리랭크 결과가 존재하는지 확인
            if rerank_results and rerank_results[0]['score'] > self.threshold:
                #outputs = self._format_output_from_rerank(rerank_results)
                _biz_type = next(iter(sorted_business_types_by_rerank_results))
                filtered_results = [result for result in rerank_results in result['biz_type'] == _biz_type]
                
                if filtered_results:
                    print(f"\n최종 선정 Biztype: ***{_biz_type}***")

                    outputs = filtered_results[:self.top_k]
                    etc_info = "ReRank 정상적용"
                    rerank_success_yn = True
                else:
                    print(f"\nRerank 결과 ")
                    outputs, etc_info = self._process_fallback_to_empty(hybrid_results)
                    rerank_success_yn = False
            else:
                if rerank_results:
                    print(f"\nRerank 결과는 존재하지만, 점수가 {rerank_results[0]['score']}로 {self.threshold}를 하회하여 검색 결과가 반환되지 않습니다.")
                else:
                    print(f"\nRerank 결과가 존재하지 않습니다.")
                outputs, etc_info = self._process_fallback_to_empty(hybrid_results)
                rerank_success_yn = False

        except (IndexError, KeyError):
            # 예외 발생 시 리랭크 실패로 간주하고 하이브리드 검색으로 전환
            # outputs, etc_info = self._process_fallback_to_hybrid(hybrid_results)
            outputs, etc_info = self._process_fallback_to_empty(hybrid_results)
            rerank_success_yn = False

        best_biz_type = None

        if outputs:
            try:
                best_biz_type = outputs[0]['biz_type']
            except KeyError:
                best_biz_type
        
        print(f"\n######Rerank 결과######")
        for ans in outputs:
            print(ans['doc_type'], ans['biz_type'], ans['chunk_src'], ans['score'], ans['chunk_id'])

        return {"validated_docs": outputs,
                "best_biz_type": best_biz_type, 
                "is_rerank_success": rerank_success_yn, 
                "etc_info": etc_info}


class ProductionRerankProcessor(BaseRerankProcessor):
    # @log_api_call(index_type="processor")
    @log_execution_time_async
    async def rerank(self, query: str, answers: list) -> list:
        """
        주어진 query와 answers 리스트를 기반으로 Pre-Rerank를 수행하고,
        Hybrid 검색과의 결과 비교 후 Post-Rerank 수행 여부를 결정합니다.
        """
        # 1. Pre-Rerank 수행 : Hybrid search와 비교하여 검색결과 변화가 발생하는지 확인하기 위함
        
        pre_rerank_results, ans_ranked_ids, ans_reranked_ids = self._pre_rerank(query, answers)
        rerank_results = pre_rerank_results
        
        # 2. Hybrid 검색과 Pre-Rerank 결과가 동일한지 비교
        is_same = self._compare_results(ans_ranked_ids, ans_reranked_ids, len(answers))
        
        # 3. 동일한 결과일 경우, Post-Rerank 생략 (연산 최적화)
        if is_same:
            return{"reranked_docs": \
                    pre_rerank_results}
        
        # 4. 다른 결과일 경우, 정확성을 높이기 위해 Post-Rerank 수행
        rerank_results = self._post_rerank(query, answers)
        
        return {"reranked_docs": rerank_results, "type": "dev results"}
    
    # @log_api_call(index_type="processor")
    @log_execution_time_async
    async def validate_results(self, hybrid_results:list, rerank_results: list) -> list:
        """
        리랭크가 실패할 경우 하이브리드 검색 결과로 대체하는 함수.
        리랭크 성공 여부 및 처리 결과 메시지를 함께 반환.
        """
        def calcaulate_average_score_by_biz_type(search_results:list) -> dict:
            biz_type_scores = {}

            for i, result in enumerate(search_results):
                biz_type = result['biz_type']
                socre = result['score']
                if score > self.threshold:
                    if biz_type not in biz_type_scores:
                        biz_type_scores[biz_type] = {'total_weighted_score': 0, 'count': 0, 'results': []}
                    if len(biz_type_scores[biz_type]['results']) <= self.top_k:
                        rank = i+1
                        weight = 1 / math.log2(rank + 1)
                        biz_type_scores[biz_type]['total_weighted_score'] += score * weight
                        biz_type_scores[biz_type]['count'] += 1
                        biz_type_scores[biz_type]['results'].append(result)
            average_scores = {}

            for biz_type, data in biz_type_scores.items():
                if data['count'] > 0:
                    average_scores[biz_type] = data['total_weighted_score']
                else:
                    average_scores[biz_type] = 0

            sorted_biz_types = dict(sorted(average_scores.items(), key = lambda item: item[1], reverse = True))
            return sorted_biz_types
            

        sorted_business_types_by_rerank_results = calcaulate_average_score_by_biz_type(rerank_results)
        print(f"\nReRank 결과 기준 Biztype별 평균 점수: {sorted_business_types_by_rerank_results}")
        
        try:
            # 유효한 리랭크 결과가 존재하는지 확인
            if rerank_results and rerank_results[0]['score'] > self.threshold:
                #outputs = self._format_output_from_rerank(rerank_results)
                _biz_type = next(iter(sorted_business_types_by_rerank_results))
                filtered_results = [result for result in rerank_results in result['biz_type'] == _biz_type]
                
                if filtered_results:
                    print(f"\n최종 선정 Biztype: ***{_biz_type}***")

                    outputs = filtered_results[:self.top_k]
                    etc_info = "ReRank 정상적용"
                    rerank_success_yn = True
                else:
                    print(f"\nRerank 결과 ")
                    outputs, etc_info = self._process_fallback_to_empty(hybrid_results)
                    rerank_success_yn = False
            else:
                if rerank_results:
                    print(f"\nRerank 결과는 존재하지만, 점수가 {rerank_results[0]['score']}로 {self.threshold}를 하회하여 검색 결과가 반환되지 않습니다.")
                else:
                    print(f"\nRerank 결과가 존재하지 않습니다.")
                outputs, etc_info = self._process_fallback_to_empty(hybrid_results)
                rerank_success_yn = False

        except (IndexError, KeyError):
            # 예외 발생 시 리랭크 실패로 간주하고 하이브리드 검색으로 전환
            # outputs, etc_info = self._process_fallback_to_hybrid(hybrid_results)
            outputs, etc_info = self._process_fallback_to_empty(hybrid_results)
            rerank_success_yn = False

        best_biz_type = None

        if outputs:
            try:
                best_biz_type = outputs[0]['biz_type']
            except KeyError:
                best_biz_type
        
        print(f"\n######Rerank 결과######")
        for ans in outputs:
            print(ans['doc_type'], ans['biz_type'], ans['chunk_src'], ans['score'], ans['chunk_id'])

        return {"validated_docs": outputs,
                "best_biz_type": best_biz_type, 
                "is_rerank_success": rerank_success_yn, 
                "etc_info": etc_info}

