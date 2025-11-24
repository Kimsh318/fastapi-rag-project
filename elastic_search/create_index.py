from elasticsearch import Elasticsearch, NotFoundError
from datetime import datetime, timezone, timedelta
import time

# Elasticsearch 클라이언트 초기화
# es_client = Elasticsearch("http://172.18.0.5:9200") # prototype 환경
# es_client = Elasticsearch("http://127.0.0.1:9200") # development 환경
es_client = Elasticsearch("http://127.0.0.1:9200", http_auth=("kdb", "kdbAi1234!")) # production 환경

num_shards = 6
num_replicas = 1

def create_index(index_name, index_type):
    """Elasticsearch 인덱스를 생성하는 함수. index_type에 따라 서로 다른 구조로 설정."""
    print("\n===== create_index 함수 호출 =====")

    # index_type별로 필드 설정
    if index_type == "api":
        index_settings = {
            "settings": {
                "number_of_shards": num_shards,
                "number_of_replicas": num_replicas,
            },
            "mappings": {
                "properties": {
                    "request_id": {"type": "keyword"},
                    "user_id": {"type": "keyword"},
                    "endpoint": {"type": "keyword"},
                    "timestamp": {"type": "date"},
                    "input_data": {
                        "type": "object",  # 모든 데이터 타입을 수용
                       "dynamic": True
                    },
                    "output_data": {
                        "type": "object",  # 모든 데이터 타입을 수용
                       "dynamic": True
                    },
                    "status": {"type": "keyword"},
                    "elapsed_time": {"type": "float"},
                    "logging_elapsed_time":  {"type": "float"},
                }
            }
        }
    elif index_type == "service":
        index_settings = {
            "settings": {
                "number_of_shards": num_shards,
                "number_of_replicas": num_replicas,
            },
            "mappings": {
                "properties": {
                    "timestamp": {"type": "date"},
                    "service_id": {"type": "keyword"},
                    "request_id": {"type": "keyword"},
                    "service_name": {"type": "keyword"},
                    "input_data": {
                        "type": "object",  # 모든 데이터 타입을 수용
                       "dynamic": True
                    },
                    "output_data": {
                        "type": "object",  # 모든 데이터 타입을 수용
                       "dynamic": True
                    },
                    "status": {"type": "keyword"},
                    "elapsed_time": {"type": "float"},
                    "logging_elapsed_time":  {"type": "float"},
                }
            }
        }
    elif index_type == "processor":
        index_settings = {
            "settings": {
                "number_of_shards": num_shards,
                "number_of_replicas": num_replicas,
            },
            "mappings": {
                "properties": {
                    "timestamp": {"type": "date"},
                    "processor_id": {"type": "keyword"},
                    "service_id": {"type": "keyword"},
                    "processor_name": {"type": "keyword"},
                    "input_data": {
                        "type": "object",  # 모든 데이터 타입을 수용
                       "dynamic": True
                    },
                    "output_data": {
                        "type": "object",  # 모든 데이터 타입을 수용
                       "dynamic": True
                    },
                    "status": {"type": "keyword"},
                    "elapsed_time": {"type": "float"},
                    "logging_elapsed_time":  {"type": "float"},
                    
                    #,
                    #"execution_time": {"type": "float"}
                }
            }
        }
    elif index_type == 'error':
        index_settings = {
            "settings": {
                "number_of_shards": num_shards,
                "number_of_replicas": num_replicas,
            },
            "mappings": {
                "properties": {
                    "function": {"type": "text"},
                    "error_message": {"type": "text"},
                    "timestamp": {"type": "date"},
                    "input": {"type": "text"},
                }
            }
        }
    elif index_type == 'http_middleware':
        index_settings = {
            "settings": {
                "number_of_shards": num_shards,
                "number_of_replicas": num_replicas,
            },
            "mappings": {
                "properties": {
                    # "request": {
                    #     "type": "object",  # 모든 데이터 타입을 수용
                    #    "dynamic": True,
                    # },
                    "url": {"type": "keyword"},
                    "timestamp": {"type": "date"},
                    "elapsed_time": {"type": "float"},
                    "logging_elapsed_time":  {"type": "float"},
                }
            }
        }
    elif index_type == 'etc_func':
        index_settings = {
            "settings": {
                "number_of_shards": num_shards,
                "number_of_replicas": num_replicas,
            },
            "mappings": {
                "properties": {
                    # "request": {
                    #     "type": "object",  # 모든 데이터 타입을 수용
                    #    "dynamic": True,
                    # },
                    "func_name": {"type": "keyword"},
                    "timestamp": {"type": "date"},
                    "elapsed_time": {"type": "float"},
                    "logging_elapsed_time":  {"type": "float"},
                }
            }
        }
    elif index_type == "func_usage":
        # 엔드포인트, service의 메소드, processor의 메소드 입출력을 로깅하기 위함
        index_settings = {
            "settings": {
                "number_of_shards": num_shards,
                "number_of_replicas": num_replicas,
            },
            "mappings": {
                "properties": {
                    "server_ip": {"type": "keyword"},
                    "user_id": {"type": "keyword"},
                    "request_id": {"type": "keyword"},
                    "func_name": {"type": "keyword"},
                    "timestamp": {"type": "date"},
                    "input_data": {"type": "text"},
                    "output_data": {
                        "type": "object",  # 모든 데이터 타입을 수용
                        #"dynamic": True
                    },
                    "elapsed_time": {"type": "float"},
                }
            }
        }
    elif index_type == "elasped_time":
        # 엔드포인트, service의 메소드, processor의 메소드 입출력을 로깅하기 위함
        index_settings = {
            "settings": {
                "number_of_shards": num_shards,
                "number_of_replicas": num_replicas,
            },
            "mappings": {
                "properties": {
                    "server_ip": {"type": "keyword"},
                    "request_id": {"type": "keyword"},
                    "func_name": {"type": "keyword"},
                    "timestamp": {"type": "date"},
                    "elapsed_time": {"type": "float"},
                    "user_id": {"type": "keyword"},
                }
            }
        }
    elif index_type == "app_metric":
        # 엔드포인트, service의 메소드, processor의 메소드 입출력을 로깅하기 위함
        index_settings = {
            "settings": {
                "number_of_shards": num_shards,
                "number_of_replicas": num_replicas,
            },
            "mappings": {
                "properties": {
                    "server_ip" : {"type": "keyword"}, # 서빙 1, 2 구분하기 위함
                    "metric_type": {"type": "keyword"},
                    "metric_data": {"type": "object"},
                    "timestamp": {"type": "date"},
                }
            }
        }
    else:
        raise ValueError("지원하지 않는 인덱스 타입입니다.")

    
    # 인덱스 존재 여부 확인 후 생성
    if not es_client.indices.exists(index=index_name):
        es_client.indices.create(index=index_name, body=index_settings)
        print(f"인덱스 '{index_name}'가 생성되었습니다. (타입: {index_type})")
    else:
        print(f"인덱스 '{index_name}'가 이미 존재합니다.")
    print("===== create_index 함수 종료 =====")

def read_all_data(index_name):
    """
    인덱스의 모든 데이터를 조회하는 함수
    """
    try:
        # 인덱스의 전체 문서 수를 확인
        count = es_client.count(index=index_name)['count']
        
        # 모든 문서를 가져오기 위한 검색 쿼리
        response = es_client.search(
            index=index_name,
            body={
                "query": {"match_all": {}},
                "size": count  # 전체 문서 수만큼 size 설정
            }
        )
        print(f"\n=== 전체 데이터 수: {count} ===")
        if count == 0:
            print("데이터가 없습니다.")
        else:
            for hit in response['hits']['hits']:
                print(f"Document ID: {hit['_id']}")
                print(f"Source: {hit['_source']}")
                print("---")
    except Exception as e:
        print(f"데이터 조회 중 오류 발생: {e}")


def delete_all_data(index_name):
    try:
        response = es_client.delete_by_query(
            index=index_name,
            body={"query": {"match_all": {}}}
        )
        print(f"삭제된 문서 수: {response['deleted']}")
        
        es_client.indices.refresh(index=index_name)
    except Exception as e:
        print(f"데이터 삭제 중 오류 발생: {e}")

def save_sample_data(index_name, doc):
    """
    샘플 데이터를 저장하고 저장된 문서의 ID를 반환하는 함수
    """
    try:
        response = es_client.index(index=index_name, body=doc)
        doc_id = response['_id']
        print(f"샘플 데이터가 저장되었습니다. (Document ID: {doc_id})")
        
        # 인덱스 새로고침
        es_client.indices.refresh(index=index_name)
        return doc_id
    except Exception as e:
        print(f"데이터 저장 중 오류 발생: {e}")
        return None
    
def list_indices():
    """현재 Elasticsearch 클러스터에 존재하는 모든 인덱스를 나열하는 함수."""
    try:
        indices = es_client.indices.get_alias("*")
        print("\n=== 현재 인덱스 목록 ===")
        for index in indices:
            print(f"{index} => #data {es_client.count(index=index)['count']}")
    except Exception as e:
        print(f"인덱스 조회 중 오류 발생: {e}")
    
if __name__ == '__main__':
    # 1. API 로그 인덱스 생성 및 샘플 데이터 추가
    print("\n=== HTTP Middleware 로그 인덱스 생성 및 데이터 추가 ===")
    create_index("http-middleware-logs", "http_middleware")
    http_sample = {
        "timestamp":  datetime.now(timezone.utc).isoformat(), # datetime.now(timezone(timedelta(hours=9))),
        "request": "/api/v1/analyze",
        "elapsed_time": 0.532
    }
    save_sample_data("http-middleware-log", http_sample)
    read_all_data("http-middleware-log")

    
    # print("\n=== API 로그 인덱스 생성 및 데이터 추가 ===")
    # create_index("api-usage-logs", "api")
    # api_sample = {
    #     "timestamp":  datetime.now(timezone.utc).isoformat(), # datetime.now(timezone(timedelta(hours=9))),
    #     "request_id": "req-001",
    #     "user_id": "user-123",
    #     "endpoint": "/api/v1/analyze",
    #     "input_data": {
    #         "query": "API 입력 데이터 예시",
    #         },
    #     "output_data": {
    #         "documents": [{
    #             "result_data": "API 처리 결과 데이터",
    #             "highlight": ["h1", "h2"]
    #         }]
    #     },
    #     "status": "SUCCESS",
    #     "elapsed_time": 0.532
    # }
    # save_sample_data("api-usage-logs", api_sample)
    # read_all_data("api-usage-logs")

    # # 2. 서비스 로그 인덱스 생성 및 샘플 데이터 추가
    # print("\n=== 서비스 로그 인덱스 생성 및 데이터 추가 ===")
    # create_index("service-usage-logs", "service")
    # service_sample = {
    #     "timestamp": datetime.now(timezone.utc).isoformat(), #datetime.now(timezone(timedelta(hours=9))),
    #     "service_id": "svc-001",
    #     "request_id": "req-001",
    #     "service_name": "Image Analysis Service",
    #     "input_data": {
    #         "query": "API 입력 데이터 예시",
    #         },
    #     "output_data": {
    #         "documents": [{
    #             "result_data": "API 처리 결과 데이터",
    #             "highlight": ["h1", "h2"]
    #         }],
    #     },
    #     "status": "SUCCESS"
    # }
    # save_sample_data("service-usage-logs", service_sample)
    # read_all_data("service-usage-logs")

    # # 3. 프로세서 로그 인덱스 생성 및 샘플 데이터 추가
    # print("\n=== 프로세서 로그 인덱스 생성 및 데이터 추가 ===")
    # create_index("processor-usage-logs", "processor")
    # processor_sample = {
    #     "timestamp":  datetime.now(timezone.utc).isoformat(), # datetime.now(timezone(timedelta(hours=9))),
    #     "processor_id": "proc-001",
    #     "service_id": "svc-001",
    #     "processor_name": "Image Resize Processor",
    #     "input_data": {
    #         "query": "API 입력 데이터 예시",
    #         },
    #     "output_data": {
    #         "documents": [{
    #             "result_data": "API 처리 결과 데이터",
    #             "highlight": ["h1", "h2"]
    #         }]
    #     },
    #     "execution_time": 0.245
    # }
    # save_sample_data("processor-usage-logs", processor_sample)
    # read_all_data("processor-usage-logs")

    print("\n=== 기타 funcition 로그 인덱스 생성 및 데이터 추가 ===")
    create_index("etc-func-logs", "etc_func")
    http_sample = {
        "func_name": 'etc-func-sample-name',
        "timestamp":  datetime.now(timezone.utc).isoformat(), # datetime.now(timezone(timedelta(hours=9))),
        "elapsed_time": 0.532,
        "logging_elapsed_time": 0.882,
    }
    save_sample_data("etc-func-logs", http_sample)
    read_all_data("etc-func-logs")
    

    # 4. API Level 에러 로그 인덱스 생성 및 샘플 데이터 추가
    print("\n=== 에러 로그 인덱스 생성 및 데이터 추가 ===")
    create_index("api-error-logs", "error")
    error_sample = {'function': 'function_c', 
                    'error_message': 'This is a test error', 
                    'timestamp':  datetime.now(timezone.utc).isoformat(), #datetime.now(),
                   'input': 'tmp_input_text'}
    save_sample_data("api-error-logs", error_sample)
    read_all_data("api-error-logs")

    # 5. 임의의 함수 입출력 로깅
    # 엔드포인트, service, processor 등
    print("\n=== func usage 로그 인덱스 생성 및 데이터 추가 ===")
    create_index("func-usage-logs", "func_usage")
    func_usage_sample = {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "server_ip" : "10.6.40.76", # 서빙 1, 2 구분하기 위함
        "user_id": "k220038",
        "request_id": "req-001",
        "func_name": "/api/v1/analyze",
        "input_data": "API 입력 데이터 예시",
        "output_data": {'output':"API 처리 결과 데이터"},
        "elapsed_time": 0.532
    }
    save_sample_data("func-usage-logs", func_usage_sample)
    read_all_data("func-usage-logs")


    # 6. 임의의 함수 처리시간 로깅
    # 엔드포인트, service, processor 등
    print("\n=== elapsed time 로그 인덱스 생성 및 데이터 추가 ===")
    create_index("elapsed-time-logs", "elasped_time")
    func_usage_sample = {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "server_ip" : "10.6.40.76", # 서빙 1, 2 구분하기 위함
        "request_id": "req-001",
        "func_name": "/api/v1/analyze",
        "elapsed_time": 0.532
    }
    save_sample_data("elapsed-time-logs", func_usage_sample)
    read_all_data("elapsed-time-logs")

      # 7. FastAPI앱 Metric 로깅
    # 엔드포인트, service, processor 등
    print("\n=== Metric 로깅 인덱스 생성 및 데이터 추가 ===")
    create_index("app-metric-logs", "app_metric")
    metric_sample = {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "server_ip" : "10.6.40.76", # 서빙 1, 2 구분하기 위함
        "metric_type": "resource_usage",
        "metric_data": {"worker1": {"cpu_usage":10.0, "memory_usage":30.0}, 
                        "worker2":{"cpu_usage":10.0, "memory_usage":30.0}},
    }
    save_sample_data("app-metric-logs", metric_sample)
    read_all_data("app-metric-logs")
    
    # 모든 데이터 삭제하여 초기화하고 싶다면 아래 코드 실행
    print("\n=== 모든 인덱스 데이터 삭제 ===")
    for index in ["http-middleware-log", "api-usage-logs", "service-usage-logs", "processor-usage-logs", "api-error-logs", "etc-func-logs", "func-usage-logs", "elapsed-time-logs", "app-metric-logs"]:
        print(f"\n{index} 삭제 중...")
        delete_all_data(index)
        read_all_data(index)
    
    # 생성된 index 리스트 확인
    list_indices()

    