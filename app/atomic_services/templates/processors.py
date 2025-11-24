import asyncio, time

from app.utils.logging_utils import log_execution_time_async, log_execution_time

# app 실행환경에 맞는 Processor를 선택하는 함수
def get_sample_processor(app_env, sample_client=None, config={}):
    if app_env == "prototype":
        return PrototypeSampleProcessor(sample_client, config=config)
    elif app_env == "development":
        return DevelopmentSampleProcessor(sample_client, config=config)
    elif app_env == "production":
        return ProductionSampleProcessor(sample_client, config=config)
    raise ValueError("지원하지 않는 환경입니다.")

class BaseSampleProcessor:
    def __init__(self, sample_client, config):
        self.client = sample_client

    # 퍼블릭 메소드 정의 : service.py에서 호출될 메소드들을 정의
    # 오버라이딩 필수 : PrototypeSampleProcessor, DevelopmentSampleProcessor, ProductionSampleProcessor에서 각 app환경에 필요한 동작들을 구현해주어야 함
    async def process_async(self, query:str) -> dict:
        raise NotImplementedError("이 메서드는 하위 클래스에서 구현되어야 합니다.")

    def process(self, prompt: str, query:str) -> dict:
        raise NotImplementedError("이 메서드는 하위 클래스에서 구현되어야 합니다.")

    def merge_results(self, list_results: list[str]) -> dict:
        raise NotImplementedError("이 메서드는 하위 클래스에서 구현되어야 합니다.")

    # 프라이빗 메소드 정의 : Processor 클래스 내부에서 사용될 메소드들을 정의
    # 공통 사용 메소드 : PrototypeSampleProcessor, DevelopmentSampleProcessor, ProductionSampleProcessor에서 공통으로 사용됨
    ## 필요에 따라 오버라이딩 혹은 오버로딩해서 사용도 가능
    def _noop(self):
        print("_noop 메소드가 실행되었습니다.")

class PrototypeSampleProcessor(BaseSampleProcessor):
    # @log_execution_time_async
    async def process_async(self, query:str) -> dict:
        # 비동기 실행 메소드 예시
        print('process_async 메소드 시작')
        await asyncio.sleep(1)
        print('process_async 메소드 종료')
        return {'process_result': 'process_async result입니다.'}
    
    # @log_execution_time
    def process(self, query:str) -> dict:
        # 동기 실행 메소드 예시
        print('process 메소드 시작')
        time.sleep(1)
        print('process 메소드 종료')
        return {'process_result': 'process result입니다.'}

    # @log_execution_time
    def merge_results(self, list_results: list[str]) -> str:
        # 결과 병합 예시
        return {'merge_result' : '\n'.join(list_results)}

class DevelopmentSampleProcessor(BaseSampleProcessor):
    # @log_execution_time_async
    async def process_async(self, query:str) -> dict:
        # 비동기 실행 메소드 예시
        print('process_async 메소드 시작')
        await asyncio.sleep(1)
        print('process_async 메소드 종료')
        return {'process_result': 'process_async result입니다.'}
    
    # @log_execution_time
    def process(self, query:str) -> dict:
        # 동기 실행 메소드 예시
        print('process 메소드 시작')
        time.sleep(1)
        print('process 메소드 종료')
        return {'process_result': 'process result입니다.'}

    # @log_execution_time
    def merge_results(self, list_results: list[str]) -> str:
        # 결과 병합 예시
        return {'merge_result' : '\n'.join(list_results)}


class ProductionSampleProcessor(BaseSampleProcessor):
    # @log_execution_time_async
    async def process_async(self, query:str) -> dict:
        # 비동기 실행 메소드 예시
        print('process_async 메소드 시작')
        await asyncio.sleep(1)
        print('process_async 메소드 종료')
        return {'process_result': 'process_async result입니다.'}
    
    # @log_execution_time
    def process(self, query:str) -> dict:
        # 동기 실행 메소드 예시
        print('process 메소드 시작')
        time.sleep(1)
        print('process 메소드 종료')
        return {'process_result': 'process result입니다.'}

    # @log_execution_time
    def merge_results(self, list_results: list[str]) -> str:
        # 결과 병합 예시
        return {'merge_result' : '\n'.join(list_results)}