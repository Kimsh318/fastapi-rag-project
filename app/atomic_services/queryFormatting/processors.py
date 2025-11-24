from string import Template

from app.core.config import settings
from app.utils.logging_utils import log_execution_time_async
from .helpers import clean_query, handle_no_data


def get_query_formatting_processor(formatting_config, tokenizer_dict, app_env, es_client=None):
    """QueryFormatting Processor를 선택하는 함수"""
    if app_env == "prototype":
        return PrototypeQueryFormattingProcessor(formatting_config, tokenizer_dict, es_client=None)
    elif app_env == "development":
        return DevelopmentQueryFormattingProcessor(formatting_config, tokenizer_dict, es_client=es_client)
    elif app_env == "production":
        return ProductionQueryFormattingProcessor(formatting_config, tokenizer_dict, es_client=es_client)
    raise ValueError("지원하지 않는 환경입니다.")


class BaseQueryFormattingProcessor:
    """Base Query Formatting Processor - 공통 헬퍼 메서드만 포함"""
    
    def __init__(self, formatting_config: dict, tokenizer_dict: dict, es_client=None):
        self.es_client = es_client
        self.MAX_INPUT_LEN = formatting_config['MAX_INPUT_LEN']
        self.LENGTH_MARGIN = formatting_config['LENGTH_MARGIN']
        self.tokenizer_dict = tokenizer_dict

    def _calculate_max_prompt_length(self, prompt_without_data, tokenizer):
        """프롬프트 길이를 계산하여 최대 허용 길이를 반환"""
        input_ids_of_prompt_template = tokenizer(prompt_without_data, return_tensors='pt')
        len_prompt_template = input_ids_of_prompt_template['input_ids'].size()[1]
        return self.MAX_INPUT_LEN - len_prompt_template - self.LENGTH_MARGIN

    async def _get_documents_by_ids(self, list_doc_ids: list):
        """주어진 문서 ID 리스트에 대한 문서 텍스트를 검색하여 반환"""
        es_results = await self.es_client.mget(
            body={"ids": list_doc_ids}, 
            index=settings.ELASTICSEARCH_DOC_DB_NM
        )

        documents = []
        for doc in es_results['docs']:
            if doc['found']:
                documents.append({
                    'doc_id': doc['_id'],
                    'doc_text': doc['_source']['chunk_context'],
                    'doc_type': doc['_source']['doc_type'],
                    'doc_src': doc['_source']['chunk_src']
                })
        return documents

    def _merge_chunks(self, list_docs, MAX_PROMPT_LEN, tokenizer):
        """문서 청크들을 순회하며 각 청크를 처리하고, 유효한 청크들을 병합"""
        merged_chunk_texts = ""
        sum_of_token_ids = 0
        num_of_doc = 1
        list_doc_ids = []
        
        for output in list_docs:
            chunk_text = f"""{output["doc_type"]}-{output["doc_src"]}\n{output["doc_text"]}"""
            chunk_seperator = f"\n#####\n참고문서[{num_of_doc}]\n#####\n"
            
            input_ids_of_chunk_text = tokenizer(chunk_seperator + chunk_text, return_tensors='pt')
            len_input_ids = len(input_ids_of_chunk_text['input_ids'][0])

            if sum_of_token_ids + len_input_ids > MAX_PROMPT_LEN:
                break

            merged_chunk_texts += f"{chunk_seperator + chunk_text}"
            num_of_doc += 1
            sum_of_token_ids += len_input_ids
            list_doc_ids.append(output['doc_id'])
            
        return merged_chunk_texts, list_doc_ids


class PrototypeQueryFormattingProcessor(BaseQueryFormattingProcessor):
    """Prototype 환경용 Query Formatting Processor"""
    
    async def get_prompt_template(self, task: str) -> dict:
        if task == 'rag':
            prompt_template_rag = """이건 RAG Task의 프롬프트 템플릿입니다."""
            query_template_rag = Template("*QUESTION* : $query\n*SEARCH_RESULTS*: $merged_chunk_texts\n")
            answer_template_rag = "\n*ANSWER* : "
            template = Template(prompt_template_rag + query_template_rag.template + answer_template_rag)
        else:
            template = Template("너와 자유대화 하고 싶어.\n$query")
        return {'template': template}
    
    async def process_rag_task(self, prompt: str, list_doc_ids: list) -> dict:
        if not list_doc_ids:
            return handle_no_data()
        
        list_chunk = [f"chunk context {i}입니다\n\n" for i in range(len(list_doc_ids))]
        merged_chunk_texts = " ".join(list_chunk)
        return {"formatted_query": merged_chunk_texts}

    async def process_free_talking_task(self, prompt: str) -> dict:
        cleaned_query = clean_query(prompt)
        return {"cleaned_query": cleaned_query}


class DevelopmentQueryFormattingProcessor(BaseQueryFormattingProcessor):
    """Development 환경용 Query Formatting Processor - 독립적으로 수정 가능"""
    
    @log_execution_time_async
    async def get_prompt_template(self, task: str) -> dict:
        if task == 'rag':
            prompt_template_rag = """You are an AI chatbot that provides detailed ANSWER to a user's QUESTION. The ANSWER must be written using only the SEARCH_RESULTS provided to you. SEARCH_RESULTS typically consist of multiple documents, each separated by a delimiter "\n#####\n참고문서[NUMBER]\n#####\n". If SEARCH_RESULTS do not exist (e.g., SEARCH_RESULTS : @No data@), you should respond with '현재 지원하는 문서범위에 찾는 내용이 없습니다. 검색되지 않는 문서에 대해서는 향후 문서범위 확장 예정입니다.' After providing the ANSWER, please provide the SOURCES used to write the ANSWER. The SOURCES should be only the 참고문서[NUMBER] and its DOCUMENT_TITLE (e.g., 참고문서[1] (내규/지침-여신지침(3.심사및승인))), not the content of the SEARCH_RESULTS. If multiple SOURCES have been used, please distinguish them through a separator of ','. Typically, DOCUMENT_TITLE exists the first or second line in each of SEARCH_RESULTS. All ANSWER must be in Korean.\n"""
            query_template_rag = Template("QUESTION : $query\n")
            search_result_template_rag = Template("SEARCH RESULTS : $merged_chunk_texts\n")
            answer_template_rag = "\nContinue to answer the QUESTION by using ONLY the SEARCH_RESULTS.\nANSWER : "
            template = Template(prompt_template_rag + query_template_rag.template + search_result_template_rag.template + answer_template_rag)
        else:
            template = Template("$query")
        return {'template': template}
    
    @log_execution_time_async
    async def process_rag_task(self, prompt: str, list_doc_ids: list) -> dict:
        if not list_doc_ids:
            return handle_no_data()

        tokenizer = self.tokenizer_dict['rag']
        MAX_PROMPT_LEN = self._calculate_max_prompt_length(prompt, tokenizer)
        list_docs = await self._get_documents_by_ids(list_doc_ids)
        merged_chunk_texts, list_doc_ids = self._merge_chunks(list_docs, MAX_PROMPT_LEN, tokenizer)

        return {
            "formatted_query": merged_chunk_texts,
            "list_doc_ids": list_doc_ids,
            "max_llm_prompt_len": MAX_PROMPT_LEN
        }
    
    @log_execution_time_async
    async def process_free_talking_task(self, prompt: str) -> dict:
        cleaned_query = clean_query(prompt)
        return {"formatted_query": cleaned_query}


class ProductionQueryFormattingProcessor(BaseQueryFormattingProcessor):
    """Production 환경용 Query Formatting Processor - 독립적으로 수정 가능"""
    
    @log_execution_time_async
    async def get_prompt_template(self, task: str) -> dict:
        if task == 'rag':
            prompt_template_rag = """You are an AI chatbot that provides detailed ANSWER to a user's QUESTION. The ANSWER must be written using only the SEARCH_RESULTS provided to you. SEARCH_RESULTS typically consist of multiple documents, each separated by a delimiter "\n#####\n참고문서[NUMBER]\n#####\n". If SEARCH_RESULTS do not exist (e.g., SEARCH_RESULTS : @No data@), you should respond with '질문과 관련된 정보를 찾을 수 없습니다. 질문을 다시 작성해 보세요.' After providing the ANSWER, please provide the SOURCES used to write the ANSWER. The SOURCES should be only the 참고문서[NUMBER] and its DOCUMENT_TITLE (e.g., 참고문서[1] (내규/지침-여신지침(3.심사및승인))), not the content of the SEARCH_RESULTS. If multiple SOURCES have been used, please distinguish them through a separator of ','. Typically, DOCUMENT_TITLE exists the first or second line in each of SEARCH_RESULTS. All ANSWER must be in Korean.\n"""
            query_template_rag = Template("QUESTION : $query\n")
            search_result_template_rag = Template("SEARCH RESULTS : $merged_chunk_texts\n")
            answer_template_rag = "\nContinue to answer the QUESTION by using ONLY the SEARCH_RESULTS.\nANSWER : "
            template = Template(prompt_template_rag + query_template_rag.template + search_result_template_rag.template + answer_template_rag)
        else:
            template = Template("$query")
        return {'template': template}
    
    @log_execution_time_async
    async def process_rag_task(self, prompt: str, list_doc_ids: list) -> dict:
        if not list_doc_ids:
            return handle_no_data()

        tokenizer = self.tokenizer_dict['rag']
        MAX_PROMPT_LEN = self._calculate_max_prompt_length(prompt, tokenizer)
        list_docs = await self._get_documents_by_ids(list_doc_ids)
        merged_chunk_texts, list_doc_ids = self._merge_chunks(list_docs, MAX_PROMPT_LEN, tokenizer)

        return {
            "formatted_query": merged_chunk_texts,
            "list_doc_ids": list_doc_ids,
            "max_llm_prompt_len": MAX_PROMPT_LEN
        }
    
    async def process_free_talking_task(self, prompt: str) -> dict:
        cleaned_query = clean_query(prompt)
        return {"formatted_query": cleaned_query}
