from string import Template
from transformers import AutoTokenizer
from elasticsearch import AsyncElasticsearch #from elasticsearch import Elasticsearch

from app.core.config import settings

#===================서비스 초기화 관련 함수==================
def get_formatting_config(settings)->dict:
    # 설정값을 받아서 formatting에 필요한 설정 반환
    return {'MAX_INPUT_LEN': settings.MAX_INPUT_LEN,
            'LENGTH_MARGIN': settings.LENGTH_MARGIN}

def get_tokenizer(task_llm_mapping: dict, hf_token:str='') -> dict:
    """
    task_llm_mapping 참고해서 각 태스크별로 토크나이저를 객체화하고 딕셔너리로 반환
    """
    tokenizer_dict = {}
    for task, llm_model in task_llm_mapping.items():
        # 태스크에 맞는 토크나이저 생성 및 저장
        if hf_token:
            tokenizer_dict[task] = AutoTokenizer.from_pretrained(llm_model, use_auth_token=hf_token)
        tokenizer_dict[task] = AutoTokenizer.from_pretrained(llm_model)
    return tokenizer_dict

#===================프로세서 관련 함수==================
def clean_query(query: str) -> str:
    return query.strip()

def add_system_message(query: str) -> str:
    return f"System: Please assist the user with the following query.\nUser: {query}"

def add_rag_message(query: str) -> str:
    return f"RAG: Answer the following based on retrieval results.\nUser: {query}"

def add_free_talking_message(query: str) -> str:
    return f"Let's have a conversation! {query}"

def handle_no_data():
    """
    데이터가 없는 경우 처리
    """
    return {"merged_chunk_texts": "@No data@",
        "used_doc_ids_and_scores": []}

def search_child_document_using_parent_id(parent_id):
    """
    부모 문서 ID를 사용하여 자식 문서 정보를 검색하는 함수
    검색된 자식 문서의 ID와 내용을 반환
    """
    return {
        'chunk_id': f'child_{parent_id}',
        'doc_type': 'child_document',
        'chunk_src': 'child_source',
        'chunk_context': f'This is a child document of {parent_id}'
    }

def get_es_client():
    if settings.APP_ENVIRONMENT == 'prototype' or  settings.APP_ENVIRONMENT == 'development':
        return AsyncElasticsearch(settings.ES_API_HOST)
    elif settings.APP_ENVIRONMENT == 'production':
        return AsyncElasticsearch(settings.ES_API_HOST, http_auth=("kdb", "kdbAi1234!"))

#====================== TASK 샘플 ======================

def clean_query(query: str) -> str:
    # 사용자 질의를 간단히 전처리 (예: 불필요한 공백 제거 등)
    return query.strip()

def add_system_prompt(query: str) -> str:   
    system_prompt = "System: This is a refined query for an LLM."
    return f"{system_prompt}\n{query}"
    