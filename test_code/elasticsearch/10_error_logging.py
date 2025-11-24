from elasticsearch import Elasticsearch
import traceback
import time
from datetime import datetime

# Elasticsearch 클라이언트 초기화
# es = Elasticsearch("http://172.18.0.4:9200") # prototype
es = Elasticsearch("http://127.0.0.1:9200")

def create_index_if_not_exists(index_name, mapping):
    """Elasticsearch 인덱스를 생성합니다. 이미 존재하면 생성을 생략합니다."""
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=mapping)
        print(f"Index '{index_name}' created.")
    else:
        print(f"Index '{index_name}' already exists.")

def function_a():
    function_b()

def function_b():
    function_c()

def function_c():
    raise ValueError("This is a test error")

def extract_function_name_from_traceback():
    tb = traceback.format_exc()
    list_func_name = []
    for line in tb.splitlines():
        if "File" in line and "in " in line:
            list_func_name.append(line.split("in ")[-1].strip())
    if list_func_name:
        return list_func_name[-1]
    return "Unknown"

def log_error_to_elasticsearch(function_name, error_message, input_text):
    log_entry = {
        "function": function_name,
        "error_message": error_message,
        "timestamp": datetime.now(),
        "input": input_text,
    }
    es.index(index="error_logs", body=log_entry)
    print(f"Error logged to Elasticsearch: {log_entry}")

def read_errors_from_elasticsearch(index_name, size=10):
    """Elasticsearch에서 최근 에러 로그를 읽어옵니다."""
    try:
        response = es.search(
            index=index_name,
            body={
                "query": {
                    "match_all": {}
                },
                "sort": [
                    {"timestamp": {"order": "desc"}}
                ],
                "size": size
            }
        )
        return response['hits']['hits']
    except Exception as e:
        print(f"Error reading from Elasticsearch: {e}")
        return []

def delete_error_log_index(index_name):
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
        print(f"Index '{index_name}' deleted")
    else:
        print(f"Index '{index_name}' does not exist")


if __name__ == "__main__":
    # 1. 인덱스 이름과 매핑 정의
    index_name = "test_error_logs"
    mapping = {
        "mappings": {
            "properties": {
                "function": {"type": "text"},
                "error_message": {"type": "text"},
                "timestamp": {"type": "date"},
                "input": {"type": "text"},
            }
        }
    }
    
    # 2. 인덱스 생성
    create_index_if_not_exists(index_name, mapping)

    # 3. 함수 실행 및 에러 로깅
    try:
        function_a()
    except Exception as e:
        print("An error occurred:", str(e))
        function_name = extract_function_name_from_traceback()
        log_error_to_elasticsearch(function_name, str(e), input_text='tmp')

    # 4. 최근 에러 로그 읽기
    errors = read_errors_from_elasticsearch(index_name)
    for error in errors:
        print(f"Logged Error: {error['_source']}")


    # 5. 인덱스 삭제
    delete_error_log_index(index_name)