from functools import wraps

# ============= decorator 기초 ============

# 기본적인 데코레이터 함수 정의
def my_decorator(func):
    def wrapper(*args, **kwargs):
        # 함수 실행 전 동작
        print("Function is about to run")
        result = func(*args, **kwargs)  # 원래 함수 호출 및 결과 저장
        # 함수 실행 후 동작
        print("Function has finished running")
        return result  # 원래 함수의 결과 반환
    return wrapper

# say_hello 함수에 my_decorator 데코레이터 적용
@my_decorator
def say_hello():
    print("Hello!")

# ============= args, kwargs 활용 =============

# args와 kwargs를 활용하여 다양한 인자를 받는 데코레이터
def my_decorator_args_kwargs(func):
    @wraps(func)  # 원래 함수의 메타데이터 유지
    def wrapper(*args, **kwargs):
        print(f"Wrapper function is running  with {args}, {kwargs}")
        return func(*args, **kwargs)  # 인자를 원래 함수에 전달
    return wrapper

# say_hello_args_kwargs 함수에 my_decorator_args_kwargs 데코레이터 적용
@my_decorator_args_kwargs
def say_hello_args_kwargs(name, greeting="Hello"):
    print(f"{greeting}, {name}!")

# ============= decorator의 wraps 데코레이터 활용 =============

# @wraps 데코레이터 사용으로 메타데이터 유지
def my_decorator_with_wraps(func):
    @wraps(func)  # 원래 함수의 이름과 docstring을 유지
    def wrapper(*args, **kwargs):
        print(f"Wrapper function is running")
        return func(*args, **kwargs)
    return wrapper

# say_hello_with_wraps 함수에 my_decorator_with_wraps 데코레이터 적용
@my_decorator_with_wraps
def say_hello_with_wraps():
    """This function says hello."""  # docstring 설정
    print("Hello!")

# ============== log api 구현 ===============

# API 호출을 로깅하는 데코레이터
def log_api_call():
    """
    API 호출을 로깅하는 데코레이터
    """
    def decorator(func):
        @wraps(func)  # 원래 함수의 메타데이터 유지
        def wrapper(*args, **kwargs):
            # 로깅 로직 실행
            print("in decorator - Logging API call")
            func(*args, **kwargs)  # 원래 함수 호출
        return wrapper
    return decorator

# my_api_function 함수에 log_api_call 데코레이터 적용
@log_api_call()
def my_api_function():
    print("my_api_function called")


# ============= 인자, func 테스트 =============
# log_api_call 데코레이터 정의
def log_api_call(index_type=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # index_type을 제대로 받아오는지 확인
            print(f"Index Type for {func.__name__}: {index_type}")  # index_type 출력하여 확인

            # 실제 함수 (func) 실행 및 반환
            result = func(*args, **kwargs)
            print(f"Function Result for {func.__name__}: {result}")  # func 호출 결과 확인
            return result
        return wrapper
    return decorator

# 테스트용 함수 정의 (index_type = "test_index1")
@log_api_call(index_type="test_index1")
def sample_function_index_type1():
    """Returns a greeting message with index_type test_index1."""
    return "Hello from index type 1!"

# 테스트용 함수 정의 (index_type = "test_index2")
@log_api_call(index_type="test_index2")
def sample_function_index_type2():
    """Returns a greeting message with index_type test_index2."""
    return "Hello from index type 2!"



if __name__ == "__main__":
    
    print("\n===== Basic Decorator Test =====")
    say_hello()  # 기본 데코레이터 테스트
    
    print("\n===== args and kwargs Decorator Test =====")
    say_hello_args_kwargs("Alice")  # "Hello, Alice!" 출력
    say_hello_args_kwargs("Bob", greeting="Hi")  # "Hi, Bob!" 출력

    print("\n===== Decorator with wraps Test =====")
    say_hello_with_wraps()  # 데코레이터에서 wraps 사용
    print("Function Name:", say_hello_with_wraps.__name__)  # "say_hello_with_wraps"
    print("Docstring:", say_hello_with_wraps.__doc__)   # "This function says hello."

    print("\n===== API Logging Decorator Test =====")
    my_api_function()  # API 호출 로깅 테스트

    print("\n===== Basic Decorator Test with index_type =====")
    sample_function_index_type1()  # sample_function_index_type1 실행: index_type과 함수 결과 출력 확인
    sample_function_index_type2()  # sample_function_index_type2 실행: 다른 index_type과 함수 결과 출력 확인

