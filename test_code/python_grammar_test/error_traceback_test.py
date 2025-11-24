# 이 스크립트의 목적:
# - Python에서 예외 발생 시 스택 트레이스를 분석하여 에러가 발생한 함수 이름을 추출하고 출력

# 방법:
# - function_c에서 의도적으로 ValueError 발생
# - main 함수에서 예외를 처리하고 스택 트레이스를 통해 함수 이름 추출
# - traceback 모듈을 사용하여 스택 트레이스를 포맷팅하고 분석
# - 스택 트레이스에서 "in " 이후의 함수 이름을 추출하여 출력

import traceback  # traceback 모듈을 가져와서 예외 발생 시 스택 트레이스를 포맷팅하는 데 사용

def function_a():
    function_b()  # function_b를 호출

def function_b():
    function_c()  # function_c를 호출

def function_c():
    # 의도적으로 에러 발생
    raise ValueError("This is a test error")  # ValueError 예외를 발생시킴

def extract_function_name_from_traceback():
    tb = traceback.format_exc()  # 스택 트레이스를 문자열로 포맷
    print(f"traceback {tb}")  # 포맷된 스택 트레이스를 출력
    list_func_name = []  # 함수 이름을 저장할 리스트
    for line in tb.splitlines():  # 스택 트레이스를 줄 단위로 분리
        if "File" in line and "in " in line:  # 함수 이름이 포함된 줄 찾기
            list_func_name.append(line.split("in ")[-1].strip())  # 함수 이름 추출
    if list_func_name: 
        return list_func_name[-1]  # 마지막 함수 이름 반환
    return "Unknown"  # 함수 이름을 찾지 못한 경우

if __name__ == '__main__':
    try:
        function_a()  # function_a를 호출
    except Exception as e:  # 예외 발생 시 처리
        print("An error occurred:", str(e))  # 예외 메시지를 출력
        function_name = extract_function_name_from_traceback()  # 스