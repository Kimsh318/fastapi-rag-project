# ================= Python 기본 라이브러리 =================
import json
import re
import logging
from functools import lru_cache
# ================= 외부 라이브러리 =================
import requests
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from elasticsearch import Elasticsearch

# ================= App 내부 모듈 =================
from app.core.config import settings

from .models import Document

from app.utils.logging_utils import log_execution_time_async, log_execution_time
# ================= Reranker 관련 =================
from FlagEmbedding import FlagReranker

logger = logging.getLogger(__name__)
# ==================================================================
# Search 보조 함수들
# ==================================================================
def get_target_doc_types(field: str, user_specified_doc_types:str):
    # 검색대상 문서종류 선정
    ## user가 요청한 doc_type에 따라 target_doc_type(검색대상 문서종류) 설정
    target_biz_type = []
    target_doc_types = []
    
    if field == 'everything':
        target_biz_type = ['여신','수신','외국환']
        if '내규/지침' in user_specified_doc_types:
            target_doc_types.append('내규/지침')
        if '행통교재' in user_specified_doc_types:
            target_doc_types.append('행통교재')
        if 'FAQ' in user_specified_doc_types:
            target_doc_types.append('FAQ')
    
    elif field == 'loan':
        target_biz_type = ['여신']
        if '내규/지침' in user_specified_doc_types:
            target_doc_types.append('내규/지침')
        if '행통교재' in user_specified_doc_types:
            target_doc_types.append('행통교재')
        if 'FAQ' in user_specified_doc_types:
            target_doc_types.append('FAQ')
        
    elif field == 'deposit':
        target_biz_type = ['수신']
        if '행통교재' in user_specified_doc_types:
            target_doc_types.append('행통교재')
        if 'FAQ' in user_specified_doc_types:
            target_doc_types.append('FAQ')

    elif field == 'forex':
        target_biz_type = ['외국환']
        if '행통교재' in user_specified_doc_types:
            target_doc_types.append('행통교재')
        
    return target_doc_types, target_biz_type    


def get_elasticsearch_body(query='', chunk_id='', target_doc_types=None, top_k=2, search_type='query'):
    """
    search_type에 따라 Elasticsearch 검색 쿼리의 body를 생성하는 함수.
        search_type : ES client의 검색 유형을 지정하는 인자로, 검색 종류에 따라 쿼리를 다르게 생성 : 질의로 검색(query), 특정문서 ID로 검색(doc_id), 부모 문서 ID로 검색(parent_doc_id), 특정 문서내 쿼리에 해당하는 영역 찾기(하이라이트 영역) (query_and_chunk_id)
    Elasticsearch 쿼리 body를 담은 딕셔너리를 반환.
    
    쿼리 구조:
    ----------------
    - `filter`: doc_type 필드에서 특정 문서 유형을 필터링.
    - `should`: 문서의 관련성을 높이기 위한 여러 검색 조건.
        1. 일반 다중 필드 매치 (chunk_context 필드).
        2. 정확한 구문 일치 (boost 값을 사용해 우선순위 증가, 분석기 적용).
        3. AND 연산자를 사용하는 다중 필드 매치 (모든 검색어가 일치해야 함).
    - `size`: top_k 값에 따른 최대 검색 결과 수.
    """
    body = {}
    if search_type=='query':
        body = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "terms": {
                                "doc_type": target_doc_type
                            },
                            
                        },
                        {
                            "terms": {
                                "biz_type": target_biz_type
                            }
                        }
                    ],
                    # 검색 정확도를 높이기 위한 다양한 검색 조건 설정 (다중 매치 쿼리)
                    "should": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["chunk_context"],
                                "type": "best_fields",
                                "operator": "or",
                                "analyzer": "kdb_nori_analyzer",
                                "boost": 1.0
                            }
                        },
                        {
                           "multi_match": {
                                "query": query,
                                "fields": ["chunk_context"],
                                "type": "best_fields",
                                "operator": "and",
                                "analyzer": "kdb_nori_analyzer",
                                "boost": 3.0
                            }
                        },
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["chunk_context"],
                                "type": "phrase",
                                "analyzer": "kdb_nori_analyzer",
                                "boost": 5.0
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            },
            "size": top_k,
            # "highlight": {
            #     "fields": {
            #         "chunk_context" : {}
            #     }
            # }
        }
    elif search_type == 'doc_id':
        body = {
            "query": {
                "bool": {
                    "filter": {
                        "term": {
                            "_id": chunk_id
                        }
                            }
                }
            }
        }
    elif search_type == 'query_and_chunk_id':
        body = {
            "query": {
                "bool": {
                    "filter": {
                        "term": {
                            "_id": chunk_id
                        }
                            },
                    "should": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields" : ["chunk_context"],
                                #"zero_terms_query": "all"
                            }
                        },
                        {
                            "multi_match": { 
                                "query": query, 
                                "type": "phrase",
                                "boost" : 5, 
                                "analyzer":  "kdb_nori_analyzer", #"my_nori_analyzer",
                                "fields" : ["chunk_context"],
                            }
                        },
                        { 
                            "multi_match": {
                                "query": query, 
                                "fields" : ["chunk_context"], 
                                "operator" : "and", 
                                "analyzer":  "kdb_nori_analyzer", #무"my_nori_analyzer" ,
                                #"zero_terms_query": "all"
                            }
                        }
                    ]
                }
            },
            "highlight": {
                "fields": {
                    "chunk_context" : {}
                }
            }
        }
    return body

def split_title(list_snippets: str) -> str:
    """
    보통 청크는 "제목\n\n본문"로 구성되어 있는데, \n\n 기준으로 제목과 본문 텍스트를 분리하고자 함
    """
    splitted_chunk = []
    for snippet in list_snippets:
        # '\n\n'의 위치를 찾음
        double_newline_index = snippet.find('\n\n')
        
        if double_newline_index != -1:
            splitted_chunk.extend(snippet.split('\n\n',1))
            continue
        splitted_chunk.append(snippet)
        
    return splitted_chunk


# ==================================================================
# Semantic Search : FAISS 
# ==================================================================
class FaissClient:
    def __init__(self, index_file_path, metadata_file_path='', model_path='', model_device_id=None, index_device_id=None, search_k=30):
        self.embedding_model = self._load_embedding_model(model_path, model_device_id)
        self.index = self._load_index(index_file_path, index_device_id)
        self.metadata = self._load_metadata(metadata_file_path)
 
        self.index_device_id = index_device_id
        self.model_device_id = model_device_id

        self.search_k = search_k        # Semantic Search 결과 top_k와는 별개. top_k < search_k

    def _load_embedding_model(self, model_path, model_device_id):
        if model_device_id:
            # GPU 사용
            return SentenceTransformer(model_path, device=model_device_id)
        # CPU 사용
        return SentenceTransformer(model_path)
            
    def _load_index(self, index_file_path, index_device_id=None):
        cpu_index = faiss.read_index(index_file_path)
        if index_device_id:
            # GPU 사용
            res = faiss.StandardGpuResources() # Faiss GPU 자원 설정
            device_num = int(re.search(r'cuda[:\s]*(\d+)', index_device_id).group(1))
            gpu_index = faiss.index_cpu_to_gpu(res, device_num, cpu_index)
            return gpu_index
        return cpu_index

    def _load_metadata(self, metadata_file_path):
        metadata = None
        if metadata_file_path:
            with open(metadata_file_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        return metadata

    def _encode_query(self, query):
        if self.model_device_id:
            return self.embedding_model.encode(query, device=self.index_device_id)\
                                            .astype('float32')
        else:
            return self.embedding_model.encode(query).astype('float32')
   
    # @log_execution_time_async
    async def query(self, query_texts):
        query_vector = self._encode_query(query_texts)
        distances, indices = self.index.search(query_vector, self.search_k)
        return distances, indices
        
@lru_cache(maxsize=None, typed=False)  
def get_faiss_client():
    if settings.APP_ENVIRONMENT == 'prototype':
        return FaissClient(index_file_path=settings.FAISS_INDEX_FILE_PATH,
                            metadata_file_path=settings.FAISS_METADATA_FILE_PATH,
                            model_path=settings.EMBEDDING_MODEL_PATH,
                            model_device_id=settings.FAISS_MODEL_DEVICE_ID,
                            index_device_id=settings.FAISS_INDEX_DEVICE_ID,
                            search_k=settings.FAISS_SEARCH_K)

    elif settings.APP_ENVIRONMENT == 'development':
        return FaissClient(index_file_path=settings.FAISS_INDEX_FILE_PATH,
                            metadata_file_path=settings.FAISS_METADATA_FILE_PATH,
                            model_path=settings.EMBEDDING_MODEL_PATH,
                            model_device_id=settings.FAISS_MODEL_DEVICE_ID,
                            index_device_id=settings.FAISS_INDEX_DEVICE_ID,
                            search_k=settings.FAISS_SEARCH_K)
        
    elif settings.APP_ENVIRONMENT == 'production':
        return FaissClient(index_file_path=settings.FAISS_INDEX_FILE_PATH,
                            metadata_file_path=settings.FAISS_METADATA_FILE_PATH,
                            model_path=settings.EMBEDDING_MODEL_PATH,
                            model_device_id=settings.FAISS_MODEL_DEVICE_ID,
                            index_device_id=settings.FAISS_INDEX_DEVICE_ID,
                            search_k=settings.FAISS_SEARCH_K)
        

# ==================================================================
# Keyword Search : ElasticSearch 
# ==================================================================

# ES 관련 함수 테스트를 위해 임시로 만든 class
class ESClient:
    def __init__(self, es_api_host:str):
        self.es_api_host = es_api_host
    
    def search(self, index:str, body:dict):
        return {
            'hits': {
                'hits': [
                    {
                        '_id': f'1287571{i}',
                        '_score': 49.241 + i,
                        '_source': {
                            'chunk_src': f'여신{i}',
                            'doc_type': '행통',
                            'chunk_context': f'행통 여신1권 키워드검색 결과{i+1}'
                        }
                    } for i in range(3)  # 3개의 검색 결과 생성
                ]
            }
        }


# TODO : 의존성 주입으로 Elasticsearch 클라이언트 생성
# OPTIMIZE : AsysncElasticsearch를 이용해서 비동기 검색 지원
def get_es_client():
    if settings.APP_ENVIRONMENT == 'prototype':
        return ESClient(settings.ES_API_HOST)
    elif settings.APP_ENVIRONMENT == 'development':
        return Elasticsearch(settings.ES_API_HOST)
    elif settings.APP_ENVIRONMENT == 'production':
        return Elasticsearch(settings.ES_API_HOST, http_auth=("kdb", "kdbAi1234!"))
    

# ==================================================================
# Reranking 
# ==================================================================

# REMOVE : rerank 모델을 사용할 수 없어서, 임시로 rerank 모델 클래스 구현
class TmpRerankModel():
    def __init__(self, model_path):
        self.model_path = model_path

    def compute_score(self, pair, max_length=1024):
        # 1~10 사이의 random score를 반환
        return round(6.323, 1)
    
    def to(self, device:str):
        return self

# REMOVE : rerank 모델을 사용할 수 없어서, 임시로 토크나이저 클래스 구현
class TmpRerankTokenizer():
    def __call__(self, chunk, return_tensors='pt'):
        # 띄어쓰기 기준으로 토크나이징
        return chunk.split()

# TODO : 반입 후, reranker 모델 연동 필요
# OPTIMIZE : reranker 모델 자체를 FastAPI 서버에 붙이는게 맞나? 별도의 Reranker API 서버가 필요하지는 않을까?\
class Reranker():
    def __init__(self, model_path:str ='MockReranker', device:str ='cuda:0', app_env:str ='prototype'):
        logger.info('Reranker 초기화 시작')
        # 개발환경에 맞는 reranker, tokenizer 초기화
        if app_env == 'prototype':
            self.model = TmpRerankModel(model_path).to(device=device)
            self.tokenizer = TmpRerankTokenizer()           
        elif app_env == 'development':
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = FlagReranker(model_path, use_fp16=True, device=device) 
        elif app_env == 'production':
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = FlagReranker(model_path, use_fp16=True, device=device) 
        
        else:
            raise ValueError("지원하지 않는 환경입니다.")
        logger.info('Reranker 초기화 완료')

    #@log_execution_time
    def compute_score(self, pair, max_length=1024):
        return self.model.compute_score(pair, max_length=max_length, normalize=True)

    def tokenize(self, chunk, return_tensors='pt'):
        return self.tokenizer(chunk, return_tensors)

@lru_cache(maxsize=None, typed=False)  
def get_reranker(model_path, device, app_env):
    return Reranker(model_path, device, app_env)


# ==================================================================
# Hybrid 검색 보조 함수 : 검색결과 병합에 필요한 함수 등
# ==================================================================
# @log_execution_time
def calculate_rrf_scores(es_result_ids: list, vs_results_ids: list, k_weight: float, v_weight: float) -> dict:
    #각각의 문서에 대해 두 검색기에서 산출된 순위를 구함
    def get_ranking(es_result_ids: list, vs_results_ids: list) -> dict: 
        all_ids = list(set(es_result_ids + vs_results_ids))
        ranks_of_both_retievers = {}
        # 순위 산출
        for _id in all_ids:
            rank_es = es_result_ids.index(_id) + 1 if _id in es_result_ids else None
            rank_vs = vs_results_ids.index(_id) + 1 if _id in vs_results_ids else None
            ranks_of_both_retievers[_id] = [rank_es, rank_vs]
        return ranks_of_both_retievers # ranks_of_both_retievers[searched_context_id] = [rank in es_result_ids, rank in vs_results_ids]
    
    # 주어진 순위에 대한 Reciprocal Rank를 계산
    def reciprocal_rank(rank):
        return 1 / rank if isinstance(rank, (int)) else 0.0
    
    all_ranks = get_ranking(es_result_ids, vs_results_ids)
    sorted_dict_for_ids_to_scores = {}
    scores = []

    for _key in all_ranks.keys(): # Here, _key indicates searched_context_id
        # 각 검색 시스템의 순위
        es_rank: int = all_ranks[_key][0] # ES rank of searched_context_id(=_key)
        vs_rank: int = all_ranks[_key][1] # VS rank of searched_context_id(=_key)

        # 각 검색 시스템(Keyword search:ES, Vector search:VS)의 Reciprocal Rank 계산
        es_rr: float = reciprocal_rank(es_rank)
        vs_rr: float = reciprocal_rank(vs_rank)

        # 가중치가 적용된 Reciprocal Rank 계산
        rrf = (v_weight * vs_rr) + (k_weight * es_rr)
        # For all searched_context_ids, calculates rrf score
        scores.append(rrf)

    sorted_scores = sorted(scores, reverse=True)
    sorted_ids = [ _key for _, _key in sorted(zip(scores, all_ranks.keys()), reverse=True) ]
    # Score (내림차순) 정렬에 따른 id 정렬 
    
    for i in range(len(sorted_ids)):
        sorted_dict_for_ids_to_scores[sorted_ids[i]] = sorted_scores[i]

    return sorted_dict_for_ids_to_scores #검색 결과에 포함된 문서별(contexts) RRF 점수를 dict (key: id, value: score)형태로 반환

# @log_execution_time
def merge_search_results_with_rrf(ks_query: str, vs_query: str, es_results: list, vs_results: list, k_weight: float, v_weight: float) -> list:
    """
    키워드 검색 결과와 벡터 검색 결과를 병합하고, RRF(Reciprocal Rank Fusion)를 적용해 최종 결과를 반환하는 함수.
      반환값 : RRF 스코어에 따라 정렬된 병합된 검색 결과 리스트. 각 항목은 사전 형태로 문서 ID, 내용, 질문, 스코어 등을 포함함.

    함수 설명:
    ----------
    1. `es_results`와 `vs_results`에서 중복된 문서 ID를 처리하고, 각 문서의 세부 정보를 `ids_to_docs`에 저장.
    2. 두 검색 결과 리스트에서 문서 ID를 추출해 RRF 스코어를 계산.
    3. RRF 스코어에 따라 병합된 검색 결과 리스트를 생성하고, 최종적으로 문서 내용을 포함한 결과를 반환.
    """
    
    ids_to_docs = {}
    
    # Elasticsearch 결과와 벡터 검색 결과에서 중복되지 않도록 문서 정보를 ids_to_docs에 저장
    if es_results:
        for each_dict in es_results:
            chunk_id = each_dict['chunk_id']
            if chunk_id not in ids_to_docs:
                ids_to_docs[chunk_id] = {key: value for key, value in each_dict.items() if key not in ['chunk_id', 'query', 'score']}

    if vs_results:
        for each_dict in vs_results:
            chunk_id = each_dict['chunk_id']
            if chunk_id not in ids_to_docs:
                ids_to_docs[chunk_id] = {key: value for key, value in each_dict.items() if key not in ['chunk_id', 'query', 'score']}
    
    # 각 검색 결과에서 문서 ID를 추출
    es_result_ids = [each_dict['chunk_id'] for each_dict in es_results] if es_results else []
    vs_result_ids = [each_dict['chunk_id'] for each_dict in vs_results] if vs_results else []   

    # RRF 알고리즘을 통해 두 결과의 스코어를 계산
    # get_rrf_scores는 문서 ID를 키로 하고, RRF 스코어를 값으로 하는 딕셔너리 반환 (스코어 내림차순 정렬)
    rrf_results = calculate_rrf_scores(es_result_ids, vs_result_ids, k_weight, v_weight)
    
    # RRF 결과에 따라 병합된 문서 정보를 최종 결과 리스트에 저장
    answers = []
    for i in range(len(rrf_results.keys())):
        rrf_chunk_id = list(rrf_results.keys())[i]
        rrf_chunk_score = rrf_results[rrf_chunk_id]
        if rrf_chunk_id in ids_to_docs:
            answers.append({
                'chunk_id': rrf_chunk_id,
                'chunk_context': ids_to_docs[rrf_chunk_id]['chunk_context'],
                'score': rrf_chunk_score,
                'doc_type': ids_to_docs[rrf_chunk_id]['doc_type'],
                'biz_type': ids_to_docs[rrf_chunk_id]['biz_type'],
                'chunk_src': ids_to_docs[rrf_chunk_id]['chunk_src'],
            })
        else:
            print(f"warning: chunk_id {rrf_chunk_id} not found in ids_to_docs")

# 디버깅 출력을 조건부로 실행
    if es_results:
        print(f"\n#########키워드검색(RRF직전) 결과#########")
        for ans in es_results:
            print(ans['doc_type'], ans['biz_type'], ans['chunk_src'], ans['score'], ans['chunk_id'])
    else:
        print(f"\n#########키워드검색(RRF직전) 결과######### (결과 없음)")

    if vs_results:
        print(f"\n#########벡터검색(RRF직전) 결과#########")
        for ans in vs_results:
            print(ans['doc_type'], ans['biz_type'], ans['chunk_src'], ans['score'], ans['chunk_id'])
    else:
        print(f"\n#########벡터검색(RRF직전) 결과######### (결과 없음)")

    print(f"\n#########RRF(필터링 전) 결과 ({len(answers)})#########")
    for ans in answers:
        print(ans['doc_type'], ans['biz_type'], ans['chunk_src'], ans['score'], ans['chunk_id'])


    def identify_user_query_specify_non_critical_docs(ks_query, vs_query):
        if not settings.NON_CRITICAL_DOCS_KEYWORDS:
            return False

        ks_query = ks_query or ""
        vs_query = vs_query or ""

        for keyword in settings.NON_CRITICAL_DOCS_KEYWORDS:
            if keyword in ks_query or keyword in vs_query:
                return True

        return False

    final_answers = []

    if identify_user_query_specify_non_critical_docs(ks_query, vs_query):
        final_answers = answers
        print(f"\n#########RRF(필터링 후) 결과 ({len(final_answers)})#########")
        print("필터링 필요없음")
        return final_answers
    else:
        for _ans in answers:
            if _ans['chunk_src'] not in settings.NON_CRITICAL_DOCS:
                final_answers.append(ans)

        print(f"\n#########RRF(필터링 후) 결과 ({len(final_answers)})#########")
        for _ans in final_answers:
            print(ans['doc_type'], ans['biz_type'], ans['chunk_src'], ans['score'], ans['chunk_id'])
        return final_answers


# 검색 결과 포맷팅
async def format_search_results(results):
    formatted_results = []
    
    for result in results:
        #if 'chunk_head' not in result: result['chunk_head'] = ['']
        if 'highlight' not in result: result['highlight'] = ['']
        formatted_results.append(
            Document(
                doc_type= result['doc_type'],  # str: 행통
                biz_type = result['biz_type'],
                doc_id= result['chunk_src'],    # str: 여신1권
                chunk_id= result['chunk_id'],   # str: '912581025'
                highlight= result['highlight'],  # list[str]: [여신1권입니다, 여신1권입니다 1 
                chunk_context = result['chunk_context'], # str
            )
        )
    return formatted_results