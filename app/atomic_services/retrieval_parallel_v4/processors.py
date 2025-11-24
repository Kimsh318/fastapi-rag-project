import math
import logging
from typing import List, Dict
from copy import deepcopy

from app.core.config import settings
from app.utils.logging_utils import log_execution_time_async

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
        return PrototypeRerankProcessor(reranker, rerank_cuda, top_k, lower_priority_docs, excluded_docs, threshold=rerank_threshold)
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


class PrototypeKeywordSearchProcessor(BaseKeywordSearchProcessor):
    async def refine_query(self, query: str) -> dict:
        return {'refined_query': query}
    
    async def search(self, query: str, user_specified_doc_types: list[str]) -> dict:
        final_ks_results = [{
            'query': query,
            'chunk_id': '1234',
            'chunk_context': f'이건 KS 결과 context #{i}',
            'score': 0.3,
            'doc_type': '행통교재',
            'doc_id': '여신1권',
            'chunk_src': '행통교재',
            'highlight': [f'이건 KS 결과 context의 highlight #{i}'],
        } for i in range(self.top_k)]

        return {'ks_results': final_ks_results}


class DevelopmentKeywordSearchProcessor(BaseKeywordSearchProcessor):
    async def refine_query(self, query: str) -> dict:
        return {'refined_query': query}
        
    @log_execution_time_async
    async def search(self, query, field, user_specified_doc_types):
        target_doc_types, target_biz_types = get_target_doc_types(field, user_specified_doc_types)
        es_body = get_elasticsearch_body(query, target_biz_types=target_biz_types, target_doc_types=target_doc_types, top_k=self.top_k*2)
        ks_results = await self.es_client.search(index=self.es_index, body=es_body)

        final_ks_results = []
        for hit in ks_results['hits']['hits']:
            highlight = ['']
            if 'highlight' in hit: 
                highlight = hit['highlight']['chunk_context']
            
            final_ks_results.append({
                'query': query,
                'chunk_id': hit['_id'],
                'chunk_context': hit['_source']['chunk_context'],
                'score': hit['_score'],
                'doc_type': hit['_source']['doc_type'],
                'biz_type': hit['_source']['biz_type'],
                'chunk_src': hit['_source']['chunk_src'],
                'highlight': highlight,
            })
        
        return {'ks_results': final_ks_results}


class ProductionKeywordSearchProcessor(BaseKeywordSearchProcessor):
    async def refine_query(self, query: str) -> dict:
        return {'refined_query': query}
        
    async def search(self, query, user_specified_doc_types):
        target_doc_types = get_target_doc_types(user_specified_doc_types)
        es_body = get_elasticsearch_body(query, target_doc_types=target_doc_types, top_k=self.top_k)
        ks_results = await self.es_client.search(index=self.es_index, body=es_body)
 
        final_ks_results = []
        for hit in ks_results['hits']['hits']:
            highlight = ['']
            if 'highlight' in hit: 
                highlight = hit['highlight']['chunk_context']
            
            final_ks_results.append({
                'query': query,
                'chunk_id': hit['_id'],
                'chunk_context': hit['_source']['chunk_context'],
                'score': hit['_score'],
                'doc_type': hit['_source']['doc_type'],
                'biz_type': hit['_source']['biz_type'],
                'chunk_src': hit['_source']['chunk_src'],
                'highlight': highlight,
            })
        
        return {'ks_results': final_ks_results}


# ============ Vector Search Processor ====================
class BaseVectorSearchProcessor:
    def __init__(self, faiss_client, top_k):
        self.faiss_client = faiss_client
        self.top_k = top_k


class PrototypeVectorSearchProcessor(BaseVectorSearchProcessor):
    async def refine_query(self, query: str) -> dict:
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
                    "chunk_context": '문서 index : ' + str(idx),
                    "highlight": [""],
                    "doc_type": "법률"
                })
        return final_vs_results

    async def search(self, query: str, target_service: str = "RETRIEVAL", user_specified_doc_type: str = "내규/지침"):
        distances, indices = await self.faiss_client.query(query_text=query)
        final_vs_results = await self._fetch_search_results(query, distances, indices)
        return {'vs_results': final_vs_results}


class DevelopmentVectorSearchProcessor(BaseVectorSearchProcessor):
    @log_execution_time_async
    async def refine_query(self, query: str) -> dict:
        return {'refined_query': query}

    async def _fetch_search_results(self, query, distances, indices, target_biz_types: list[str], target_doc_types: list[str]):
        final_vs_results = []
        SEARCH_K = self.faiss_client.search_k
        FAISS_DOC_METADATA = self.faiss_client.metadata

        for i in range(SEARCH_K):
            idx = indices[i]
            if idx == -1:
                break

            doc_metadata = FAISS_DOC_METADATA[idx]
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

        final_vs_results = sorted(final_vs_results, key=lambda x: float(x["score"]), reverse=True)
        final_vs_results = final_vs_results[:self.top_k*2]
        return final_vs_results

    @log_execution_time_async
    async def search(self, queries: list[str], field: str, user_specified_doc_types: list = ["내규/지침"]):
        target_doc_types, target_biz_types = get_target_doc_types(field, user_specified_doc_types)

        distances, indices = await self.faiss_client.query(query_texts=queries)
        results_with_scores = []

        for i, query in enumerate(queries):
            results = await self._fetch_search_results(query, distances[i], indices[i], target_biz_types=target_biz_types, target_doc_types=target_doc_types)
            if results:
                total_score = sum(result["score"] for result in results)
                results_with_scores.append((results, total_score))

        if not results_with_scores:
            return {'vs_results': None, 'vs_query': None}

        best_results, best_score = max(results_with_scores, key=lambda item: item[1])
        best_query = best_results[0]["query"]
        
        return {'vs_results': best_results, 'vs_query': best_query}


class ProductionVectorSearchProcessor(BaseVectorSearchProcessor):
    @log_execution_time_async
    async def refine_query(self, query: str) -> dict:
        return {'refined_query': query}

    async def _fetch_search_results(self, query, distances, indices, target_biz_types: list[str], target_doc_types: list[str]):
        final_vs_results = []
        SEARCH_K = self.faiss_client.search_k
        FAISS_DOC_METADATA = self.faiss_client.metadata

        for i in range(SEARCH_K):
            idx = indices[i]
            if idx == -1:
                break

            doc_metadata = FAISS_DOC_METADATA[idx]
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

        final_vs_results = sorted(final_vs_results, key=lambda x: float(x["score"]), reverse=True)
        final_vs_results = final_vs_results[:self.top_k*2]
        return final_vs_results

    @log_execution_time_async
    async def search(self, queries: list[str], field: str, user_specified_doc_types: list = ["내규/지침"]):
        target_doc_types, target_biz_types = get_target_doc_types(field, user_specified_doc_types)

        distances, indices = await self.faiss_client.query(query_texts=queries)
        results_with_scores = []

        for i, query in enumerate(queries):
            results = await self._fetch_search_results(query, distances[i], indices[i], target_biz_types=target_biz_types, target_doc_types=target_doc_types)
            if results:
                total_score = sum(result["score"] for result in results)
                results_with_scores.append((results, total_score))

        if not results_with_scores:
            return {'vs_results': None, 'vs_query': None}

        best_results, best_score = max(results_with_scores, key=lambda item: item[1])
        best_query = best_results[0]["query"]
        
        return {'vs_results': best_results, 'vs_query': best_query}


# ================================
#  Rerank Processor
# ================================
class BaseRerankProcessor:
    def __init__(self, reranker, rerank_cuda, top_k: int = 10, threshold=0):
        logger.info('Reranker Processor 초기화 시작')
        self.reranker = reranker
        self.rerank_cuda = rerank_cuda
        self.top_k = top_k 
        self.threshold = threshold
        
        # rerank 길이 설정
        self.len_reranker_max_support = 1024
        self.len_extra = 16
        self.len_overlap = 256

        logger.info('Reranker Processor 초기화 완료')

    def _get_len_input_ids(self, _str: str) -> int:
        """주어진 문자열을 토큰화하여 토큰의 수를 반환"""
        return len(self.reranker.tokenizer(_str, return_tensors='pt'))

    def _pre_rerank(self, query: str, answers: list):
        """Pre-ReRank를 수행하여 초기 재정렬 결과와 ID 리스트를 반환"""
        pre_rerank_results = []
        ans_ranked_ids = []
        ans_reranked_ids = []

        for ans in answers:
            ans_ranked_ids.append(ans['chunk_id'])
            rerank_score = self.reranker.compute_score([query, ans['chunk_context']], max_length=self.len_reranker_max_support)
            
            pre_rerank_results.append({
                'chunk_id': ans['chunk_id'],
                'chunk_context': ans['chunk_context'],
                'score': rerank_score,
                'doc_type': ans['doc_type'],
                'biz_type': ans['biz_type'],
                'chunk_src': ans['chunk_src'],
            })

        pre_rerank_results.sort(key=lambda x: x['score'], reverse=True)
        for result in pre_rerank_results:
            ans_reranked_ids.append(result['chunk_id'])
        return pre_rerank_results, ans_ranked_ids, ans_reranked_ids

    def _compare_results(self, ans_ranked_ids, ans_reranked_ids, answers_length):
        """Hybrid 검색과 Rerank 결과를 비교하여 동일한지 확인"""
        if answers_length >= self.top_k and ans_reranked_ids[:self.top_k] == ans_ranked_ids[:self.top_k]:
            return True
        elif answers_length < self.top_k and ans_reranked_ids == ans_ranked_ids:
            return True
        else:
            return False

    def _post_rerank(self, query: str, answers: list):
        """Chunk를 분할하여 Post-ReRank를 수행"""
        LEN_QUERY = self._get_len_input_ids(query)
        LEN_RERANKER_MAX_SUPPORT = self.len_reranker_max_support
        LEN_EXTRA = self.len_extra
        BASE_LEN_CHUNK = LEN_RERANKER_MAX_SUPPORT - LEN_QUERY - LEN_EXTRA
        LEN_OVERLAP = self.len_overlap
        RERANKER_TOKENIZER = self.reranker.tokenizer
        
        rerank_results = []

        for ans in answers:
            context_len = self._get_len_input_ids(ans['chunk_context'])
            if context_len > BASE_LEN_CHUNK:
                num_rerank = math.ceil((context_len - BASE_LEN_CHUNK) / (BASE_LEN_CHUNK - LEN_OVERLAP)) + 1
                nl_fragments = []

                for i in range(num_rerank):
                    start_pos = i * (BASE_LEN_CHUNK - LEN_OVERLAP)
                    end_pos = start_pos + BASE_LEN_CHUNK
                    
                    tokenized = RERANKER_TOKENIZER(ans['chunk_context'], return_tensors='pt').to(self.rerank_cuda)
                    if end_pos <= len(tokenized['input_ids'][0]):
                        nl_fragment = RERANKER_TOKENIZER.decode(tokenized['input_ids'][0][start_pos:end_pos])
                    else:
                        nl_fragment = RERANKER_TOKENIZER.decode(tokenized['input_ids'][0][start_pos:])
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
                rerank_score = self.reranker.compute_score([query, ans['chunk_context']], max_length=1024)

            ans_copy = deepcopy(ans)
            ans_copy.update({'score': rerank_score})
            rerank_results.append(ans_copy)

        rerank_results.sort(key=lambda x: x['score'], reverse=True)
        return rerank_results[:self.top_k] if len(rerank_results) >= self.top_k else rerank_results

    def _process_fallback_to_hybrid(self, answers):
        """리랭크 결과가 없을 경우, 하이브리드 검색 결과를 처리하여 반환"""
        if len(answers) >= self.top_k:
            answers = answers[:self.top_k]
        
        if answers and answers[0]['score'] > 0:
            return answers, "ReRank 결과 존재하지 않으므로, 하이브리드 검색으로 전환"
        else:
            return [], "검색결과 없음"

    def _process_fallback_to_empty(self, answers):
        """검색 결과 없음 처리"""
        return [], "검색결과 없음"


class PrototypeRerankProcessor(BaseRerankProcessor):
    async def rerank(self, query: str, answers: list) -> dict:
        """Pre-Rerank를 수행하고, Hybrid 검색과의 결과 비교 후 Post-Rerank 수행 여부를 결정"""
        pre_rerank_results, ans_ranked_ids, ans_reranked_ids = self._pre_rerank(query, answers)
        is_same = self._compare_results(ans_ranked_ids, ans_reranked_ids, len(answers))
        
        if is_same:
            return {"reranked_docs": pre_rerank_results[:self.top_k] if len(pre_rerank_results) >= self.top_k else pre_rerank_results}
        
        rerank_results = self._post_rerank(query, answers)
        return {"reranked_docs": rerank_results}

    async def exclude_target_docs(self, rerank_results: list, user_excluded_docs: list = []):
        return {"excluded_rerank_docs": rerank_results}

    async def validate_results(self, hybrid_results: list, rerank_results: list) -> dict:
        """리랭크가 실패할 경우 하이브리드 검색 결과로 대체"""
        try:
            if rerank_results and rerank_results[0]['score'] > 0:
                outputs = rerank_results
                etc_info = "ReRank 정상적용"
                rerank_success_yn = True
            else:
                outputs, etc_info = self._process_fallback_to_hybrid(hybrid_results)
                rerank_success_yn = False
        except (IndexError, KeyError):
            outputs, etc_info = self._process_fallback_to_hybrid(hybrid_results)
            rerank_success_yn = False

        return {
            "validated_docs": outputs, 
            "is_rerank_success": rerank_success_yn, 
            "etc_info": etc_info
        }

    async def adjust_docs(self, reranked_docs: List[Dict]) -> dict:
        """문서 순위 보정: 중요도가 낮은 문서들은 후순위로 배치"""
        adjusted_docs = sorted(reranked_docs, key=lambda x: x['chunk_src'] in self.lower_priority_docs, reverse=False)
        return {"adjusted_docs": adjusted_docs}


class DevelopmentRerankProcessor(BaseRerankProcessor):
    @log_execution_time_async
    async def rerank(self, query: str, answers: list) -> dict:
        """Pre-Rerank를 수행하고, Hybrid 검색과의 결과 비교 후 Post-Rerank 수행 여부를 결정"""
        pre_rerank_results, ans_ranked_ids, ans_reranked_ids = self._pre_rerank(query, answers)
        is_same = self._compare_results(ans_ranked_ids, ans_reranked_ids, len(answers))
        
        if is_same:
            return {"reranked_docs": pre_rerank_results}
        
        rerank_results = self._post_rerank(query, answers)
        return {"reranked_docs": rerank_results}
    
    @log_execution_time_async
    async def validate_results(self, hybrid_results: list, rerank_results: list) -> dict:
        """리랭크가 실패할 경우 검색 결과 처리"""
        def calculate_average_score_by_biz_type(search_results: list) -> dict:
            biz_type_scores = {}

            for i, result in enumerate(search_results):
                biz_type = result['biz_type']
                score = result['score']
                if score > self.threshold:
                    if biz_type not in biz_type_scores:
                        biz_type_scores[biz_type] = {'total_weighted_score': 0, 'count': 0, 'results': []}
                    if len(biz_type_scores[biz_type]['results']) <= self.top_k:
                        rank = i + 1
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

            sorted_biz_types = dict(sorted(average_scores.items(), key=lambda item: item[1], reverse=True))
            return sorted_biz_types

        sorted_business_types_by_rerank_results = calculate_average_score_by_biz_type(rerank_results)
        
        try:
            if rerank_results and rerank_results[0]['score'] > self.threshold:
                _biz_type = next(iter(sorted_business_types_by_rerank_results))
                filtered_results = [result for result in rerank_results if result['biz_type'] == _biz_type]
                
                if filtered_results:
                    outputs = filtered_results[:self.top_k]
                    etc_info = "ReRank 정상적용"
                    rerank_success_yn = True
                else:
                    outputs, etc_info = self._process_fallback_to_empty(hybrid_results)
                    rerank_success_yn = False
            else:
                outputs, etc_info = self._process_fallback_to_empty(hybrid_results)
                rerank_success_yn = False
        except (IndexError, KeyError):
            outputs, etc_info = self._process_fallback_to_empty(hybrid_results)
            rerank_success_yn = False

        best_biz_type = None
        if outputs:
            try:
                best_biz_type = outputs[0]['biz_type']
            except KeyError:
                pass

        return {
            "validated_docs": outputs,
            "best_biz_type": best_biz_type, 
            "is_rerank_success": rerank_success_yn, 
            "etc_info": etc_info
        }


class ProductionRerankProcessor(BaseRerankProcessor):
    @log_execution_time_async
    async def rerank(self, query: str, answers: list) -> dict:
        """Pre-Rerank를 수행하고, Hybrid 검색과의 결과 비교 후 Post-Rerank 수행 여부를 결정"""
        pre_rerank_results, ans_ranked_ids, ans_reranked_ids = self._pre_rerank(query, answers)
        is_same = self._compare_results(ans_ranked_ids, ans_reranked_ids, len(answers))
        
        if is_same:
            return {"reranked_docs": pre_rerank_results}
        
        rerank_results = self._post_rerank(query, answers)
        return {"reranked_docs": rerank_results}
    
    @log_execution_time_async
    async def validate_results(self, hybrid_results: list, rerank_results: list) -> dict:
        """리랭크가 실패할 경우 검색 결과 처리"""
        def calculate_average_score_by_biz_type(search_results: list) -> dict:
            biz_type_scores = {}

            for i, result in enumerate(search_results):
                biz_type = result['biz_type']
                score = result['score']
                if score > self.threshold:
                    if biz_type not in biz_type_scores:
                        biz_type_scores[biz_type] = {'total_weighted_score': 0, 'count': 0, 'results': []}
                    if len(biz_type_scores[biz_type]['results']) <= self.top_k:
                        rank = i + 1
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

            sorted_biz_types = dict(sorted(average_scores.items(), key=lambda item: item[1], reverse=True))
            return sorted_biz_types

        sorted_business_types_by_rerank_results = calculate_average_score_by_biz_type(rerank_results)
        
        try:
            if rerank_results and rerank_results[0]['score'] > self.threshold:
                _biz_type = next(iter(sorted_business_types_by_rerank_results))
                filtered_results = [result for result in rerank_results if result['biz_type'] == _biz_type]
                
                if filtered_results:
                    outputs = filtered_results[:self.top_k]
                    etc_info = "ReRank 정상적용"
                    rerank_success_yn = True
                else:
                    outputs, etc_info = self._process_fallback_to_empty(hybrid_results)
                    rerank_success_yn = False
            else:
                outputs, etc_info = self._process_fallback_to_empty(hybrid_results)
                rerank_success_yn = False
        except (IndexError, KeyError):
            outputs, etc_info = self._process_fallback_to_empty(hybrid_results)
            rerank_success_yn = False

        best_biz_type = None
        if outputs:
            try:
                best_biz_type = outputs[0]['biz_type']
            except KeyError:
                pass

        return {
            "validated_docs": outputs,
            "best_biz_type": best_biz_type, 
            "is_rerank_success": rerank_success_yn, 
            "etc_info": etc_info
        }
