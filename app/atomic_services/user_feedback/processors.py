from datetime import datetime
import asyncio, time

from app.utils.logging_utils import log_execution_time_async, log_execution_time

# app 실행환경에 맞는 Processor를 선택하는 함수
def get_processor(app_env, sample_client=None, config={}):
    if app_env == "prototype":
        return PrototypeProcessor(sample_client, config)
    elif app_env == "development":
        return DevelopmentProcessor(sample_client, config)
    elif app_env == "production":
        return ProductionProcessor(sample_client, config)
    raise ValueError("지원하지 않는 환경입니다.")

class BaseSampleProcessor:
    def __init__(self, es_client, config):
        self.client = es_client
        self.index_name = config['index_name']

    # 퍼블릭 메소드 정의 : service.py에서 호출될 메소드들을 정의
    # 오버라이딩 필수 : PrototypeSampleProcessor, DevelopmentSampleProcessor, ProductionSampleProcessor에서 각 app환경에 필요한 동작들을 구현해주어야 함
    async def save_feedback_async(self, feedback:str) -> dict:
        raise NotImplementedError("이 메서드는 하위 클래스에서 구현되어야 합니다.")

class PrototypeProcessor(BaseSampleProcessor):
    # @log_execution_time_async
    async def save_feedback_async(self, input_data:dict) -> dict:
        response = await self.es_client.index(index=self.index_name, body={
            'user_id': input_data['user_id'],
            'feedback': input_data['feedback'],
            'timestamp': datetime.now()
        })
        return {'process_result': input_data['feedback']}
    

class DevelopmentProcessor(BaseSampleProcessor):
    # @log_execution_time_async
    async def save_feedback_async(self, input_data:dict) -> dict:
        response = await self.es_client.index(index=self.index_name, body={
            'user_id': input_data['user_id'],
            'feedback': input_data['feedback'],
            'timestamp': datetime.now()
        })
        return {'process_result': input_data['feedback']}


class ProductionProcessor(BaseSampleProcessor):
    # @log_execution_time_async
    async def save_feedback_async(self, input_data:dict) -> dict:
        response = await self.es_client.index(index=self.index_name, body={
            'user_id': input_data['user_id'],
            'feedback': input_data['feedback'],
            'timestamp': datetime.now()
        })
        return {'process_result': input_data['feedback']}