from typing import List, Dict

class BaseConfig():
    """
    기본 설정 클래스.
    모든 환경에서 공통으로 사용되는 설정값을 정의합니다.
    """
    APP_ENVIRONMENT: str = 'development'    # 앱사용환경 지정 : prototype, development, production
    APP_SERVER: str = 'ML1'  #  서버이름 지정 : ML1, serving1, serving2 (production 환경이라면 필수)
    

    # REMOVE : 모델 로드 관련 설정 (필요 시 추가)
    MODEL_CONFIGS: List[Dict[str, str]] = [
        {"type": "embedding", "name": "all-MiniLM-L6-v2"},
    ]

class PrototypeConfig(BaseConfig):
    # 프로토타입 환경에 대한 설정.

    # ---------------------------------
    # 공통 설정
    # ---------------------------------
    SERVER_IP: str = 'None'
    
    # Ollama API 설정
    OLLAMA_API_HOST: str = "172.18.0.5:11434"  # Ollama HOST
    OLLAMA_MODEL_NAME: str = "gemma2:2b"      # Ollama 모델 이름

    HF_TOKEN: str = "" # HugginFace Gemma 사용을 위한 토큰
    
    # Query Embedding 관련 설정
    EMBEDDING_MODEL_PATH: str = "all-MiniLM-L6-v2"  # 임베딩 모델명
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # 임베딩 모델명
    EMBEDDING_CUDA: str = 'cpu'
    VECTOR_DOC_COLLECTION_NM: str = 'banking_law_collection'  # 임베딩 검색을 위한 문서 컬렉션 이름

    # ---------------------------------
    # Query Refine 관련 설정
    # ---------------------------------
    # 임시 : 검색에 불필요한 술어 정보가 저장된 Collection 이름
    VECTOR_USELESS_ACTION_COLLECTION:str = "vdb_useless_action_terms"
    REFINE_LLM_HOST = OLLAMA_API_HOST
    REFINE_LLM_NAME = OLLAMA_MODEL_NAME

    REFINE_ES_INDEX_NAME:str = 'query_refine_index'

    REFINE_FAISS_FAISS_INDEX_FILE_PATH: str = "/app/app/db_faiss(will_be_removed)/vector_index.faiss"
    REFINE_FAISS_METADATA_FILE_PATH: str = ''
    REFINE_EMBEDDING_MODEL_PATH: str = EMBEDDING_MODEL_PATH
    REFINE_FAISS_MODEL_DEVICE_ID: str = ''
    REFINE_FAISS_INDEX_DEVICE_ID: str = ''
    REFINE_QUERY_REFINEMENT_THRESHOLD:float = 0.8

    # ---------------------------------
    # Query Validation 관련 설정
    # ---------------------------------

    LLM_TOKENIZER_PATH: str = "google/gemma-2-2b-it" # 사용할 Tokenizer명
    # MAX_TOKEN_LEN: int = 50  # Validation Criteria : 최대 토큰 길이
    # MIN_TOKEN_LEN: int = 2   # Validation Criteria : 최소 토큰 길이

    MAX_CHRACTER_LEN: int = 256
    MIN_CHRACTER_LEN: int = 2
    # ---------------------------------
    # Query Formatting 관련 설정
    # ---------------------------------

    # Task별 사용할 토크나이저
    TASK_LLM_MAPPING: Dict[str, str] = {
        "rag": "google/gemma-2-2b-it",
        "free_talking": "google/gemma-2-2b-it"
    }
    
    # Query 최대 길이 관련 설정: LLM (KDB-GPT)의 입력에 사용되는 Prompt의 최대 길이
    MAX_INPUT_LEN: int = 400  
    LENGTH_MARGIN: int = 16

    # ---------------------------------
    # Retrieval 관련 설정
    # ---------------------------------
    
    VS_TOP_K: int = 2

    # ChromaDB 관련 설정
    CHROMADB_API_HOST: str = "0.0.0.0"
    CHROMADB_API_PORT: int = 9202
    CHROMADB_TOP_K: int = 2

    # FAISS 관련 설정
    FAISS_INDEX_FILE_PATH: str = "/app/app/db_faiss(will_be_removed)/vector_index.faiss"
    FAISS_METADATA_FILE_PATH: str = ''
    FAISS_MODEL_DEVICE_ID: str = ''
    FAISS_INDEX_DEVICE_ID: str = ''
    FAISS_SEARCH_K: int = 30 # semantic search 결과의 top_k와는 별개의 값. search_k > top_k

    # Elasticsearch 관련 설정
    ES_API_HOST: str = "http://172.18.0.4:9200"
    ES_TOP_K: int = 2
    K_WEIGHT: float = 0.7  # 키워드 검색 가중치 (하이브리드 검색)
    NON_CRITICAL_DOCS: List[str] = ['상공인대출지침', '선박금융 온렌딩대출 취급지침']  # 중요하지 않은 문서
    ELASTICSEARCH_DOC_DB_NM: str = 'None'       # 모든 문서가 저장된 Elasticsearch Index 이름
    # ELASTICSEARCH_LOG_DB_NM = 'logging'     # 사용자의 사용 이력이 저장되는 Elasticsearch Index 이름
    # ELASTICSEARCH_QUERY_ANAL_DB = 'global_index'    # 사용자 질의 中 검색에 불필요한 조사 및 술어를 필터링하기 위해 사용되는 Elasticsearch Index 이름 (실제 문서를 저장하지 않고, POS TAG Filter 및 Token Position 정보 추출하기 위해 사용됨)

    # RERANK 관련 설정
    RERANKER_CUDA: str = 'none'
    RERANKER_MODEL_PATH: str = 'none' # Rerank 모델 경로
    RERANKER_LOWER_PRIORITY_DOCS: List[str] = ['은행법']  # 낮은 우선순위의 문서 목록
    RERANKER_TOP_K: int = 2
    RERANKER_EXCLUDED_DOCS: List[str] = ['상공인대출지침', '선박금융 온렌딩대출 취급지침']  # Rerank 제외 문서 목록
    RERANKER_SCORE_THRESHOLD:float = 0.2
    # ---------------------------------
    # Generation 관련 설정
    # ---------------------------------

    # ---------------------------------
    # User Feedback 관련 설정
    # ---------------------------------
    USER_FEEDBACK_INDEX_NAME: str = 'user-feedback'


class DevelopmentConfig(BaseConfig):
    # 개발 환경에 대한 설정.
    
    GUNICORN_PID_INFO_PATH = './gunicorn_log/pid_file.txt'
    SERVER_IP: str = '10.6.40.76' # ML1 IP
    
    HF_TOKEN: str = ''
    
    AI_MODEL_PATH: str = '/workspace/ai_model/'
    
    EMBEDDING_MODEL_NM: str = 'bge-m3'
    EMBEDDING_MODEL:str = AI_MODEL_PATH+EMBEDDING_MODEL_NM
    EMBEDDING_CUDA: str = 'cuda:1'
    
    GEMMA_MODEL_NM: str = 'gemma2-9b-it'
    VLLM_API_HOST: str = "http://10.6.40.90:32022"


    #DOC_VIEWER_URL: str = "http://10.6.40.76:32050/search/"    # 문서 뷰어 관련 설정
    ES_INDEX_NAME: str = 'documents_v_latest'    # 문서 검색용 ES Index명

    # ---------------------------------
    # Query Refine 관련 설정
    # ---------------------------------
    # 검색에 불필요한 술어 정보가 저장된 Collection 이름
    #VECTOR_USELESS_ACTION_COLLECTION:str = "vdb_useless_action_terms"
    REFINE_ES_INDEX_NAME:str = ES_INDEX_NAME
    
    # LLM 정보
    REFINE_MODEL:str = AI_MODEL_PATH + GEMMA_MODEL_NM # 모델 파일 위치
    REFINE_MODEL_URL:str = f"{VLLM_API_HOST}/v1/completions" # 모델 서버 URL

    REFINE_LLM_HOST:str = VLLM_API_HOST
    REFINE_LLM_NAME:str = AI_MODEL_PATH + GEMMA_MODEL_NM

    # TODO : 아래 키워드가 질의에 포함되면, LLM으로 질의 정제 수행
    REFINE_KEYWORD_LIST:list = ['표', '정리', '요약', '영어', '일본어', '번역', '중국어', '부탁',\
                                '비교', '예시', '분석', '구체적', '표현', '작성', '써줘']


    REFINE_FAISS_FAISS_INDEX_FILE_PATH: str = "/workspace/dev/dev1_p/13_langchain_fastapi_backend_dev/v18/app/db_faiss/faiss_useless_terms.index"
    REFINE_FAISS_METADATA_FILE_PATH: str = "/workspace/dev/dev1_p/13_langchain_fastapi_backend_dev/v18/app/db_faiss/faiss_useless_terms_meta.json"
    REFINE_EMBEDDING_MODEL_PATH: str = EMBEDDING_MODEL
    REFINE_FAISS_MODEL_DEVICE_ID: str = 'cuda:1'
    REFINE_FAISS_INDEX_DEVICE_ID: str = 'cuda:1'
    REFINE_QUERY_REFINEMENT_THRESHOLD:float = 0.8
    
    KEYWORD_SYNONYM_PATH: str = '/workspace/1_product~/synonym_dict.txt'

    # ---------------------------------
    # Query Validation 관련 설정
    # ---------------------------------

    LLM_TOKENIZER_PATH: str = AI_MODEL_PATH + GEMMA_MODEL_NM     # 사용할 Tokenizer명


    # MAX_TOKEN_LEN: int = 100
    # MIN_TOKEN_LEN: int = 3

    MAX_CHRACTER_LEN: int = 256
    MIN_CHRACTER_LEN: int = 2
    # MAX_TOKEN_LEN: int = 3584  # Validation Criteria : 최대 토큰 길이
    # MIN_TOKEN_LEN: int = 2048  # Validation Criteria : 최소 토큰 길이

    # ---------------------------------
    # Query Formatting 관련 설정
    # ---------------------------------

    # Task별 사용할 토크나이저
    TASK_LLM_MAPPING: Dict[str, str] = {
        "rag": AI_MODEL_PATH + GEMMA_MODEL_NM,
        "free_talking":  AI_MODEL_PATH + GEMMA_MODEL_NM, 
    }
    # Query 최대 길이 관련 설정: LLM (KDB-GPT)의 입력에 사용되는 Prompt의 최대 길이
    MAX_INPUT_LEN: int = 8192-2048
    LENGTH_MARGIN: int = 16
    
    # ---------------------------------
    # Retrieval 관련 설정
    # ---------------------------------
    
    VS_TOP_K: int = 5


    # ChromaDB 관련 설정
    EMBEDDING_MODEL_PATH: str = AI_MODEL_PATH + EMBEDDING_MODEL_NM
    #VECTOR_DOC_COLLECTION_NM: str = 'vdb_docs' # 모든 문서가 저장된 Collection 이름
    #CHROMADB_API_HOST: str = "localhost" #"0.0.0.0"
    #CHROMADB_API_PORT: int = 9202
    #CHROMADB_TOP_K: int = 3

    # FAISS 관련 설정
    FAISS_INDEX_FILE_PATH: str = '/workspace/dev/dev1_p/13_langchain_fastapi_backend_dev/v18/app/db_faiss/faiss_documents.index'
    FAISS_METADATA_FILE_PATH: str = "/workspace/dev/dev1_p/13_langchain_fastapi_backend_dev/v18/app/db_faiss/faiss_documents_meta.json"
    FAISS_MODEL_DEVICE_ID: str = 'cuda:3'
    FAISS_INDEX_DEVICE_ID: str = 'cuda:3'
    FAISS_SEARCH_K: int = 100 # semantic search 결과의 top_k와는 별개의 값. search_k > top_k


    
    # Elasticsearch 관련 설정
    ES_API_HOST: str = "http://127.0.0.1:9200"
    ES_TOP_K: int = 5
    K_WEIGHT: float = 0.4  # 키워드 검색 가중치 (하이브리드 검색)
    
    ELASTICSEARCH_DOC_DB_NM: str = ES_INDEX_NAME #'esdb_0827'       # 모든 문서가 저장된 Elasticsearch Index 이름
    ELASTICSEARCH_LOG_DB_NM: str = 'logging'     # 사용자의 사용 이력이 저장되는 Elasticsearch Index 이름
    NON_CRITICAL_DOCS: List[str] = ['상공인대출지침', '선박금융 온렌딩대출 취급지침', '일반고객대출 사후관리지침', '선박금융 온렌딩대출 취급지침','온렌딩대출세칙','온렌딩대출 지침','기업외대출세칙','기업외대출 운영지침']  # 중요하지 않은 문서
    NON_CRITICAL_DOCS_KEYWORDS: List[str] = ['상공인', '일반고객', '선박금융', '온렌딩','기업외']  # 중요하지 않은 문서
    #ELASTICSEARCH_QUERY_ANAL_DB: str = 'global_index'    # 사용자 질의 中 검색에 불필요한 조사 및 술어를 필터링하기 위해 사용되는 Elasticsearch Index 이름 (실제 문서를 저장하지 않고, POS TAG Filter 및 Token Position 정보 추출하기 위해 사용됨)

    RRF_TOP_K: int = (ES_TOP_K + VS_TOP_K) * 2
    # ---------------------------------
    # Rerank 관련 설정
    # ---------------------------------
    RERANKER_CUDA: str = 'cuda:0'
    RERANKER_MODEL_NM: str = 'bge-reranker-m3-v2'
    RERANKER_MODEL_PATH: str = AI_MODEL_PATH + RERANKER_MODEL_NM  # Rerank 모델 경로
    #RERANKER_LOWER_PRIORITY_DOCS: List[str] =  ['상공인대출지침', '선박금융 온렌딩대출 취급지침']  # 낮은 우선순위의 문서 목록
    RERANKER_TOP_K: int = 5
    #RERANKER_EXCLUDED_DOCS: List[str] = ['상공인대출지침', '선박금융 온렌딩대출 취급지침']  # Rerank 제외 문서 목록
    RERANKER_SCORE_THRESHOLD = 0.1

    # ---------------------------------
    # Generation 관련 설정
    # ---------------------------------

    # vLLM API 설정
    # TODO : 시스템 구현 설계결과에 따라 추후 Litellm 경로로 변경 검토
    VLLM_MODEL_NAME: str = "gemma2-9b-it"      # vLLM 모델 이름
    VLLM_MODEL_MAX_LEN = 8192 # 4096
    
    # LITELLM_API_HOST: str = "http://external.api/litellm"


    # ---------------------------------
    # User Feedback 관련 설정
    # ---------------------------------
    USER_FEEDBACK_INDEX_NAME: str = 'user-feedback'


class ProductionConfig(BaseConfig):
    # 프로덕션 환경에 대한 설정.
    
    GUNICORN_PID_INFO_PATH = './gunicorn_log/pid_file.txt'
    if BaseConfig.APP_SERVER == 'serving1':
        SERVER_IP: str = '10.6.40.78:32117' # Log Parsing용
        VLLM_API_HOST: str = 'http://10.6.40.78:32130'
    elif BaseConfig.APP_SERVER == 'serving2':
        SERVER_IP: str = '10.6.40.79:32117'   # Log Parsing용
        VLLM_API_HOST: str = 'http://10.6.40.79:32130'
    #HF_TOKEN: str = ''
    
    AI_MODEL_PATH: str = '/workspace/ai_model/'
    
    EMBEDDING_MODEL_NM: str = 'bge-m3'
    EMBEDDING_MODEL:str = AI_MODEL_PATH+EMBEDDING_MODEL_NM
    EMBEDDING_MODEL_PATH: str = AI_MODEL_PATH + EMBEDDING_MODEL_NM
    EMBEDDING_CUDA: str = 'cuda:0'
    
    GEMMA_MODEL_NM: str = 'gemma2-9b-it'
    #VLLM_API_HOST: str = "http://10.6.40.78:32120"


    # DOC_VIEWER_URL: str = "http://10.6.40.76:32050/search/"    # 문서 뷰어 관련 설정
    ES_INDEX_NAME: str = 'documents_v_latest'    # 문서 검색용 ES Index명

    # ---------------------------------
    # Query Refine 관련 설정
    # ---------------------------------
    # 검색에 불필요한 술어 정보가 저장된 Collection 이름
    #VECTOR_USELESS_ACTION_COLLECTION:str = "vdb_useless_action_terms"
    REFINE_ES_INDEX_NAME:str = ES_INDEX_NAME
    
    # LLM 정보
    REFINE_MODEL:str = AI_MODEL_PATH + GEMMA_MODEL_NM # 모델 파일 위치
    REFINE_MODEL_URL:str = f"{VLLM_API_HOST}/v1/completions" # 모델 서버 URL

    REFINE_LLM_HOST:str = VLLM_API_HOST
    # REFINE_LLM_NAME:str = AI_MODEL_PATH + GEMMA_MODEL_NM
    if BaseConfig.APP_SERVER == 'serving1':
        REFINE_LLM_NAME:str = "lite_gemma2"
    elif BaseConfig.APP_SERVER == 'serving2':
        REFINE_LLM_NAME:str = "lite_gemma2_sv2"

    # TODO : 아래 키워드가 질의에 포함되면, LLM으로 질의 정제 수행
    REFINE_KEYWORD_LIST:list = ['표', '정리', '요약', '영어', '일본어', '번역', '중국어', '부탁',\
                                '비교', '예시', '분석', '구체적', '표현', '작성', '써줘']


    if BaseConfig.APP_SERVER == 'serving1':
        REFINE_FAISS_FAISS_INDEX_FILE_PATH: str = "/workspace/1_product/faiss/faiss_useless_terms.index"
        REFINE_FAISS_METADATA_FILE_PATH: str = "/workspace/1_product/faiss/faiss_useless_terms_meta.json"
    elif BaseConfig.APP_SERVER == 'serving2':
        REFINE_FAISS_FAISS_INDEX_FILE_PATH: str = "/workspace/2_product/faiss/faiss_useless_terms.index"
        REFINE_FAISS_METADATA_FILE_PATH: str = "/workspace/2_product/faiss/faiss_useless_terms_meta.json"
        
    REFINE_EMBEDDING_MODEL_PATH: str = EMBEDDING_MODEL
    REFINE_FAISS_MODEL_DEVICE_ID: str = 'cuda:0'
    REFINE_FAISS_INDEX_DEVICE_ID: str = 'cuda:0'
    REFINE_QUERY_REFINEMENT_THRESHOLD:float = 0.8
    
    KEYWORD_SYNONYM_PATH: str = '/workspace/1_product~/synonym_dict.txt'

    # ---------------------------------
    # Query Validation 관련 설정
    # ---------------------------------

    LLM_TOKENIZER_PATH: str = AI_MODEL_PATH + GEMMA_MODEL_NM     # 사용할 Tokenizer명

    MAX_CHRACTER_LEN: int = 256
    MIN_CHRACTER_LEN: int = 2
    
    # MAX_TOKEN_LEN: int = 3584  # Validation Criteria : 최대 토큰 길이
    # MIN_TOKEN_LEN: int = 2048  # Validation Criteria : 최소 토큰 길이

    # ---------------------------------
    # Query Formatting 관련 설정
    # ---------------------------------

    # Task별 사용할 토크나이저
    TASK_LLM_MAPPING: Dict[str, str] = {
        "rag": AI_MODEL_PATH + GEMMA_MODEL_NM,
        "free_talking":  AI_MODEL_PATH + GEMMA_MODEL_NM, 
    }
    # Query 최대 길이 관련 설정: LLM (KDB-GPT)의 입력에 사용되는 Prompt의 최대 길이
    MAX_INPUT_LEN: int = 8192-2048
    LENGTH_MARGIN: int = 16
    
    # ---------------------------------
    # Retrieval 관련 설정
    # ---------------------------------
    
    VS_TOP_K: int = 5

    # FAISS 관련 설정
    if BaseConfig.APP_SERVER == 'serving1':
        FAISS_INDEX_FILE_PATH: str = "/workspace/1_product/faiss/faiss_documents.index"
        FAISS_METADATA_FILE_PATH: str ="/workspace/1_product/faiss/faiss_documents_meta.json"
    elif BaseConfig.APP_SERVER == 'serving2':
        FAISS_INDEX_FILE_PATH: str = "/workspace/2_product/faiss/faiss_documents.index"
        FAISS_METADATA_FILE_PATH: str ="/workspace/2_product/faiss/faiss_documents_meta.json"
        
    FAISS_MODEL_DEVICE_ID: str = 'cuda:0'
    FAISS_INDEX_DEVICE_ID: str = 'cuda:0'
    FAISS_SEARCH_K: int = 30 # semantic search 결과의 top_k와는 별개의 값. search_k > top_k
    
    # Elasticsearch 관련 설정
    if BaseConfig.APP_SERVER == 'serving1':
        ES_API_HOST: str = "http://127.0.0.1:9200"
        ES_API_SUB_HOST: str = "http://10.6.40.79:32110"  # Serving 1번 ES 장애시, Serving 2번의 ES를 사용하도록 하기 위함
    elif BaseConfig.APP_SERVER == 'serving2':
        ES_API_HOST: str = "http://127.0.0.1:9200"
        ES_API_SUB_HOST: str = "http://10.6.40.78:32110"  # Serving 2번 ES 장애시, Serving 1번의 ES를 사용하도록 하기 위함
        
    ES_TOP_K: int = 3
    K_WEIGHT: float = 0.4  # 키워드 검색 가중치 (하이브리드 검색)
    
    ELASTICSEARCH_DOC_DB_NM: str = ES_INDEX_NAME #'esdb_0827'       # 모든 문서가 저장된 Elasticsearch Index 이름
    ELASTICSEARCH_LOG_DB_NM: str = 'logging'     # 사용자의 사용 이력이 저장되는 Elasticsearch Index 이름
    NON_CRITICAL_DOCS: List[str] = ['상공인대출지침', '선박금융 온렌딩대출 취급지침', '일반고객대출 사후관리지침', '선박금융 온렌딩대출 취급지침','온렌딩대출세칙','온렌딩대출 지침','기업외대출세칙','기업외대출 운영지침']  # 중요하지 않은 문서
    NON_CRITICAL_DOCS_KEYWORDS: List[str] = ['상공인', '일반고객', '선박금융', '온렌딩','기업외']  # 중요하지 않은 문서
    #ELASTICSEARCH_QUERY_ANAL_DB: str = 'global_index'    # 사용자 질의 中 검색에 불필요한 조사 및 술어를 필터링하기 위해 사용되는 Elasticsearch Index 이름 (실제 문서를 저장하지 않고, POS TAG Filter 및 Token Position 정보 추출하기 위해 사용됨)


    RRF_TOP_K: int = (ES_TOP_K + VS_TOP_K) * 2

    # ---------------------------------
    # Rerank 관련 설정
    # ---------------------------------
    RERANKER_CUDA: str = 'cuda:0'
    RERANKER_MODEL_NM: str = 'bge-reranker-m3-v2'
    RERANKER_MODEL_PATH: str = AI_MODEL_PATH + RERANKER_MODEL_NM  # Rerank 모델 경로
    RERANKER_TOP_K: int = 5
    RERANKER_SCORE_THRESHOLD = 0.2

    # ---------------------------------
    # Generation 관련 설정
    # ---------------------------------

    # vLLM API 설정
    # TODO : 시스템 구현 설계결과에 따라 추후 Litellm 경로로 변경 검토
    VLLM_MODEL_NAME: str = "gemma2-9b-it"      # vLLM 모델 이름
    VLLM_MODEL_MAX_LEN = 4000 # 4096


    # ---------------------------------
    # User Feedback 관련 설정
    # ---------------------------------
    USER_FEEDBACK_INDEX_NAME: str = 'user-feedback'


# 환경에 맞는 설정을 불러오는 함수
def get_settings():
    app_env = BaseConfig().APP_ENVIRONMENT  # 혹은 os.getenv("APP_ENVIRONMENT")
    
    # APP Envrionment에 적합한 Config 불러오기
    if app_env == 'prototype':
        return PrototypeConfig()
    elif app_env == 'development':
        return DevelopmentConfig()
    elif app_env == 'production':
        return ProductionConfig()
    
    # 환경 설정 오류 : APP Envrionment가 잘못 설정되어 있을 때, 에러 발생
    raise ValueError(f"Invalid APP_ENVIRONMENT: {app_env}. Please set it to 'prototype', 'development', or 'production'.")


# 설정 인스턴스 생성
settings = get_settings()
