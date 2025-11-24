from app.utils.logging_utils import log_execution_time_async, log_execution_time

from .processors import get_processor
from .helpers import handle_no_data

class FeedbackService:
    def __init__(self, sample_client, config: dict, app_env:str):
        self.config = config
        self.processor = get_processor(app_env=app_env, sample_client=sample_client, config=config)

    @log_execution_time_async    
    async def save_feedback(self, input_data: dict) -> str:
        handle_no_data(input_data['feedback'])
        result = await self.processor.save_feedback_async(input_data)
        return {"service_result": result["process_result"]}