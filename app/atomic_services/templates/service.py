from app.utils.logging_utils import log_execution_time_async, log_execution_time

from .processors import get_sample_processor
from .helpers import handle_no_data

import asyncio

class SampleService:
    def __init__(self, sample_client, config: dict, app_env:str):
        self.config = config
        self.processor = get_sample_processor(app_env=app_env, sample_client=sample_client, config=config)

    @log_execution_time_async    
    async def run_task_async(self, query: str) -> str:
        handle_no_data(query)
        result = await self.processor.process_async(query)
        return {"service_result": result["process_result"]}

    @log_execution_time_async    
    async def run_multiple_tasks_async(self, query: str) -> str:
        task_1 = asyncio.create_task(
            self.processor.process_async(query)
        )
        task_2 = asyncio.create_task(
            self.processor.process_async(query)
        )
        # 두 태스크를 병렬로 실행하고 결과를 기다림
        result_1, result_2 = await asyncio.gather(
            task_1, task_2
        )
        result = self.processor.merge_results(result_1, result_2)
        return {"service_result": result["merge_result"]}

    @log_execution_time
    def run_task(self, query: str) -> str:
        result = self.processor.process(query)
        return {"service_result": result["process_result"]}