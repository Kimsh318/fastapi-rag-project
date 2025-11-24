import time

from app.utils.logging_utils import log_execution_time_async

from .processors import get_query_refine_processor
from .helpers import remove_hangul_jamo

class QueryRefineService:
    def __init__(self, es_client, faiss_client, app_env):
        self.processor = get_query_refine_processor(es_client, faiss_client, app_env)

    @log_execution_time_async  # @log_api_call(index_type="service")
    async def refine(self, query: str, task: str) -> str:
        if task == 'General':
            s_time = time.time()
            refined_result = await self.processor.refine_general(query = query)
            {"refined_query": refined_result["refined_query"]}
        if task == 'VectorSearch':
            s_time = time.time()
            query = remove_hangul_jamo(query)
            refined_result = await self.processor.refine_for_vector_search(query=query)
            return {"refined_query": refined_result["refined_query"]}

        elif task == 'KeywordSearch':
            s_time = time.time()
            query = remove_hangul_jamo(query)
            refined_result = await self.processor.refine_for_keyword_search(query=query)
            return {"refined_query": refined_result["refined_query"]}
