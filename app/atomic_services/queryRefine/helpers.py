# 표준 라이브러리 모듈
import json
import re
from functools import lru_cache
import logging

# 서드파티 라이브러리 모듈
import numpy as np
import requests
import aiohttp
import faiss
from elasticsearch import AsyncElasticsearch
from sentence_transformers import SentenceTransformer

# 로컬 애플리케이션/라이브러리 모듈
from app.core.config import settings

from app.utils.logging_utils import log_api_call, log_execution_time, log_execution_time_async

logger = logging.getLogger("excution_time_logger")

# 보조적인 유틸리티 함수들 (e.g. query의 특정 문자나 불필요한 공백 제거)
def remove_special_characters(query: str) -> str:
    import re
    return re.sub(r'[^a-zA-Z0-9\s]', '', query)

def remove_hangul_jamo(text):
    # 텍스트 내 자음 혹은 모음만 있는 경우, 해당 자모음을 제거
    # ex) '안녕ㅏ하세요ㅇ' -> '안녕하세요'
    
    # 패턴 정의
    consonant_pattern = re.compile(r'[\u1100-\u1112\u11A8-\u11C2\u3131-\u314E]') # 자음 패턴
    vowel_pattern = re.compile(r'[\u1161-\u1175\u314F-\u3163]') # 모음 패턴

    # 자음, 모음 패턴 제거
    remove_consonant = consonant_pattern.sub('', text)
    remove_vowel = vowel_pattern.sub('',remove_consonant)
    
    return remove_vowel

def normalize_whitespace(query: str) -> str:
    return ' '.join(query.split())


def get_noun_tokens(analyzed_result: dict):
    ori_tokens, synonym_tokens = [], []
    for token_info in analyzed_result['tokens']:
        if token_info['type'].lower() == 'word':
            ori_tokens.append(deepcopy(token_info))
    return ori_tokens

def get_ori_and_synonym_tokens(analyzed_result: dict):
    ori_tokens, synonym_tokens = [], []
    for token_info in analyzed_result['tokens']:
        if token_info['type'].lower() == 'word':
            ori_tokens.append(deepcopy(token_info))
        elif token_info['type'].lower() == "synonym":
            synonym_tokens.append(deepcopy(token_info))
    return ori_tokens, synonym_tokens

def split_by_keywords(text, keywords):
    pattern = '(' + '|'.join(re.escape(keyword) for keyword in keywords) + ')'
    return [part for part in re.split(pattern, text) if part]

def split_by_keywords_w_greedy_method(text, keywords):
    if not text:
        return []
    if not keywords:
        return [text]
    sorted_keywords = sorted(set(keywords), key=len, reverse=True)
    result = []
    i = 0
    while i < len(text):
        matched = False
        for keyword in sorted_keywords:
            if text[i:i+len(keyword)] == keyword:
                result.append(keyword)
                i += len(keyword)
                matched = True
                break
        if not matched:
            non_match_start = i
            while i < len(text):
                found_match = False
                for keyword in sorted_keywords:
                    if text[i:i+len(keyword)] == keyword:
                        found_match = True
                        break
                if found_match:
                    break
                i += 1
            if i > non_match_start:
            result.append(text[non_match_start:i])
    return result

def generate_all_combinations(query: str, ori_tokens_info: List[Dict]) -> List[str]:
    keywords = [token['token'] for token in ori_tokens_info]
    ori_tokens = split_by_keywords_w_greedy_method(query, keywords)
    
    with open(settings.KEYWORD_SYNONYM_PATH, 'r') as file:
        data = file.readlines()
        all_syms = []
        for row in data:
            words = row.split(",")
            for i in range(len(words)):
                words[i] = words[i].replace("\n",'').strip()
            all_syms.append(words)

    synonym_info = []
    position_options = []

    for idx, token_info in enumerate(ori_tokens):
        position = idx
        position_options[position] = [token]

        for sym in all_syms:
            if token in syms:
                alt_tokens = [word for word in syms if word != token]
                synonym_info.append({'position': postion, 'token': alt_tokens})

    for synonym_data in synonym_info:
        position = synonym_data["position"]
        synonym =  synonym_data["token"]
        position_options[position].extend(synonym)

    valid_positions = sorted(position_options.keys())
    
    combinations = list(product(*position_options[pos] for pos in valid_positions))
    
    generated_queries = []
    for combination in combinations:
        if list(combination) != ori_tokens:
            query = ' '.join(combination)
            generated_queries.append(query)
    
    return generated_queries


# ================================ 
# ELASTIC SEARCH
# ================================ 
def get_es_client():
    if settings.APP_ENVIRONMENT == 'prototype':
        # 테스트용 ES DB 생성
        es_client = AsyncElasticsearch(settings.ES_API_HOST)
        index_settings = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "standard_analyzer": {
                            "type": "standard",
                            "max_token_length": 255,
                            "stopwords": "_none_"
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "content": {
                        "type": "text",
                        "analyzer": "standard_analyzer"
                    }
                }
            }
        }
        index = settings.REFINE_ES_INDEX_NAME
        
        # 테스트용 인덱스 생성
        if not es_client.indices.exists(index=index):
            es_client.indices.create(index=index, body=index_settings)
        # 샘플 데이터 추가
        sample_documents = [
            {"content": "주식투자 초보자 가이드"},
            {"content": "부동산 시장 분석"},
            {"content": "암호화폐 투자 전략"},
            {"content": "퇴직연금 운용 방법"},
            {"content": "글로벌 금융시장 동향"}
        ]
        
        for i, doc in enumerate(sample_documents):
            es_client.index(index=index, id=i+1, body=doc)
        
        return es_client
    elif settings.APP_ENVIRONMENT == 'development':
        return AsyncElasticsearch(settings.ES_API_HOST)
    elif settings.APP_ENVIRONMENT == 'production':
        return AsyncElasticsearch(settings.ES_API_HOST, http_auth=("kdb", "kdbAi1234!"))


# ================================ 
# FAISS
# ================================ 

#FAISS 전용 Search 객체 생성
class FaissClient:
    def __init__(self, index_file_path, metadata_file_path, model_path, llm_host, llm_name, model_device_id=None, index_device_id=None, high_priority_threshold=0.8):
        logger.info("======intialize QueryRefine Faiss Client=====")
        self.model = self._load_embedding_model(model_path, model_device_id)
        self.index_file_path = index_file_path
        self.metadata_file_path = metadata_file_path
        self.high_priority_threshold = high_priority_threshold
        self.dimension = self.model.get_sentence_embedding_dimension()
        # self.index = faiss.IndexFlatL2(self.dimension)
        
        self.index = self._load_index(index_file_path, index_device_id)
        self.metadata = self._load_metadata(metadata_file_path)
        
        # self.gec_model = GEMMA_MODEL_NM
        self.llm_host = llm_host
        self.llm_name = llm_name
        
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

    def _generate_ngrams(self, text, n=4):
        words = text.split()
        ngrams = [{"phrase": " ".join(words[i:i+j]), "start": i, "end": i+j}
                  for j in range(1, n+1) for i in range(len(words)-j+1)]
        return ngrams

    # @log_execution_time
    def filter_query(self, query):
        ngrams = self._generate_ngrams(query)
        phrases = [phrase['phrase'] for phrase in ngrams]
        embeddings = self.model.encode(phrases, convert_to_tensor=False, show_progress_bar=False)
        embeddings = np.array(embeddings).astype('float32')
       
        # FAISS 검색: 각 n-gram에 대해 가장 가까운 메타데이터와 유사도를 계산
        distances, indices = self.index.search(embeddings, 1)
        remove_candidates = []
        
        for idx, (ngram, distance) in enumerate(zip(ngrams, distances[:, 0])):
            if distance <= (1 - self.high_priority_threshold):
                remove_candidates.append({"word_count": len(ngram['phrase'].split()), "score":1-distance, "phrase" : ngram['phrase'], "start": ngram['start'], "end": ngram['end']})
        
        remove_candidates = remove_candidates[::-1]
        
        remove_intervals = []
        is_vaild = 0

        for item in remove_candidates:
            if not remove_intervals:
                remove_intervals.append({"phrase": item['phrase'], "start" : item['start'], "end" : item['end']})
            else:
                is_valid = 1
                for i in range(len(remove_intervals)):
                    if item['start'] >= remove_intervals[i]['end'] or item['end'] <= remove_intervals[i]['start']:
                        is_vaild = is_valid * 1
                    else:
                        is_valid = is_valid * 0
                if is_valid == 1:
                    remove_intervals.append({"phrase": item['phrase'], "start" : item['start'], "end" : item['end']})
        diff = 0
        query_tokens = query.split()

        remove_intervals = sorted(remove_intervals, key=lambda x:x["start"])

        for interval in remove_intervals:
            for i in range(interval['start'], interval['end']):
                query_tokens[i] = "[REMOVED]"

        return query_tokens

    # @log_execution_time_async
    async def grammar_error_correction(self, query: str) -> str:
        """
        LLM을 사용한 문법 오류 수정 및 자연어 완성
        """
        prompt = f"""다음의 사용자 요청에서 문법적 오류를 식별 및 교정하세요. 만약 사용자 요청 문자열이 동사(서술어) 누락으로 불완전하거나, 부자연스러운 표현으로 작성되어 있다면, 문맥상 올바른 표현을 사용하여 완성해주세요.
        예를 들어, 문자열 내 서술어가 누락되어 있다면 '알려줘', '무엇이지?', '뭐야?' 등 간단한 서술어를 활용하여 교정하세요. 교정 관련 설명은 제공하지 말고, 교정된 문자열만을 제공하세요.
        사용자 요청 문자열: {query}
        교정된 문자열:
        """
        if settings.APP_ENVIRONMENT == 'prototype':
            # LLM API에 보낼 데이터 정의
            data = {
                "model": self.llm_name,
                "prompt": prompt,
                "max_tokens": 256,
                "temperature": 0,
                "seed": 1,
                "top_k": 50,
                "best_of": 1,
                "stream": False,
            }
            
            # (동기방식) POST 요청 보내기
            response = requests.post( f"http://{self.llm_host}/api/generate", # 생성결과를 한번에 받아오기 위해, "chat"이 아닌 "generate" 엔드포인트에 요청 
                                    headers={"Content-type": "application/json"}, 
                                    data=json.dumps(data))
            # 응답 출력
            return response.json()['response']

        elif settings.APP_ENVIRONMENT == 'development' or settings.APP_ENVIRONMENT == 'production':
            data = {
                "model": self.llm_name,
                "prompt": prompt,
                "max_tokens": 256,
                "temperature": 0,
                "seed" : 1,
                "top_k" : 50,
                "best_of" : 1,
                "stream" : False,
            }
            # (비동기방식) POST 요청 보내기
            async with aiohttp.ClientSession() as session:
                async with session.post( f"{self.llm_host}/v1/completions", # 생성결과를 한번에 받아오기 위해, "chat"이 아닌 "generate" 엔드포인트에 요청 
                                    headers={"Content-type": "application/json"}, 
                                    # headers = {"Content-Type": "application/json", "Authorization": "Bearer kdbAi1234!"},
                                    data=json.dumps(data)) as response:
                                    #data=data) as response:
                    response_json = await response.json()
            try:
                return response_json['choices'][0]['text']
            except Exception as e:
                logger.info(f"{query}vLLM Grammar Correction 실패 \n {e}\n")
                return query
        return None

@lru_cache(maxsize=None, typed=False)   
def get_faiss_client()->FaissClient:
    faiss_client = FaissClient(index_file_path=settings.REFINE_FAISS_FAISS_INDEX_FILE_PATH,
                                metadata_file_path=settings.REFINE_FAISS_METADATA_FILE_PATH,
                                model_path=settings.REFINE_EMBEDDING_MODEL_PATH,
                                model_device_id=settings.REFINE_FAISS_MODEL_DEVICE_ID,
                                index_device_id=settings.REFINE_FAISS_INDEX_DEVICE_ID,
                                high_priority_threshold=settings.REFINE_QUERY_REFINEMENT_THRESHOLD,
                                                                
                                llm_host = settings.REFINE_LLM_HOST,
                                llm_name = settings.REFINE_LLM_NAME)
    return faiss_client
