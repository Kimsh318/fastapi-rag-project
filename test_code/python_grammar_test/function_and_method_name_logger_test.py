"""
함수와 메서드의 이름을 확인하는 방법을 테스트하는 코드
* func.__name__은 '함수명'만 반환
* func.__qualname__은 '클래스명.메소드명' 형식으로 반환하여, 어떤 클래스 하위의 메소드인지 식별하기 쉬움
"""
import logging
from functools import wraps

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 데코레이터 정의
def log_function_details(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 함수 이름과 qualified name 로깅
        func_name = func.__name__
        func_qualname = func.__qualname__
        
        # 클래스 이름 분리
        class_name, _, method_name = func_qualname.rpartition('.')
        if class_name:
            logger.info(f"Class: {class_name}, Method: {method_name}")
        else:
            logger.info(f"Function: {func_name}")
        
        # 원래 함수 실행
        return func(*args, **kwargs)
    
    return wrapper

# 전역 함수 테스트
@log_function_details
def global_function():
    print("This is a global function.")

# 클래스 내부 메서드 테스트
class TestClass:
    @log_function_details
    def method_in_class(self):
        print("This is a method inside a class.")
    
    @staticmethod
    @log_function_details
    def static_method_in_class():
        print("This is a static method inside a class.")

# 테스트 실행
if __name__ == "__main__":
    global_function()
    obj = TestClass()
    obj.method_in_class()
    TestClass.static_method_in_class()
