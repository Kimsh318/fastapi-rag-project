
#===================클래스 정의==================
class SampleClient:
    def __init__(self):
        self.class_name = 'SampleClient'
    
    def search(self, query:str)->list[str]:
        return ['search result 1', 'search result 2']

#===================서비스 초기화 관련 함수==================
def get_service_config(settings)->dict:
    # settings에서 service에 필요한 설정값들만 반환
    return {'app_env': settings.APP_ENVIRONMENT,
            'index_name': settings.USER_FEEDBACK_INDEX_NAME}

def get_sample_client():
    return SampleClient()

#===================프로세서 관련 함수==================
def handle_no_data(feedback:str):
    """
    데이터가 없는 경우 처리
    """
    if not feedback.strip():
        raise ValueError("입력된 데이터가 없습니다.")

