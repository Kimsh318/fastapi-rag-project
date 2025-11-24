from app.atomic_services.queryValidation.helpers import get_tokenizer
from app.atomic_services.queryValidation.processors import get_query_processor
from app.utils.logging_utils import log_api_call, log_execution_time_async

class QueryValidationService:
    def __init__(self, tokenizer_info:dict, validation_criteria:dict, app_env:str):
        self.processor = get_query_processor(tokenizer_info, validation_criteria, app_env)

    @log_execution_time_async # @log_api_call(index_type="service")
    async def validate(self, query: str)-> list:
        # 1. 토큰 수 검증 : 토큰 수가 일정이하면, 유효하지 않은 질의로 판단
        cleansed_query = await self.processor.cleansing_query(query)
        cal_result = await self.processor.calculate_token_count(cleansed_query['cleansed_query'])
        valid_result = await self.processor.validate_length_bounds(cleansed_query['cleansed_query'])
        # valid_result = await self.processor.validate_length_bounds(cal_result['num_tokens'])

        return {"is_valid": valid_result['is_valid'], 
                "num_tokens": cal_result['num_tokens']}
