from app.utils.logging_utils import log_api_call, log_execution_time_async

from .processors import get_query_formatting_processor

class QueryFormattingService:
    def __init__(self, formatting_config:dict, tokenizer_dict:dict, app_env:str, es_client=None):
        # processor 초기화 시, 각 태스크별 tokenizer 및 prompt_without_data 전달
        self.processor = get_query_formatting_processor(formatting_config, tokenizer_dict, app_env=app_env, es_client=es_client)
        
    # @log_api_call(index_type="service")
    # @log_execution_time_async
    async def format_query(self, query: str, task: str, list_doc_ids: list = None) -> str:
        # task에 맞는 prompt를 생성하고 query를 대입
        prompt = query
        if task == "rag":
            # 프롬프트 템플릿 생성
            prompt_template = await self.processor.get_prompt_template(task=task)
            prompt_template = prompt_template['template']

            # 청크 내용 병합
            prompt = prompt_template.safe_substitute(query=query, merged_chunk_texts='')
            merge_result = await self.processor.process_rag_task(prompt=prompt, list_doc_ids=list_doc_ids)
            
            # 템플릿에 맞추어 프롬프트 생성
            prompt = prompt_template.safe_substitute(query=query, merged_chunk_texts=merge_result['formatted_query'])
            
            return {"formatted_query": prompt,
                   "list_doc_ids": merge_result['list_doc_ids']}
        
        elif task == "free_talking":
            # 프롬프트 템플릿 생성
            prompt_template = await self.processor.get_prompt_template(task=task)
            prompt_template = prompt_template['template']
            
            # 템플릿에 맞추어 프롬프트 생성
            cleaned_query = await self.processor.process_free_talking_task(prompt=prompt)
            prompt = prompt_template.safe_substitute(query=cleaned_query['cleaned_query'])

        return {"formatted_query": prompt}

    