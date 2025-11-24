from elasticsearch import Elasticsearch, NotFoundError
from datetime import datetime, timezone, timedelta
import time

# Elasticsearch 클라이언트 초기화
#es_client = Elasticsearch("http://172.18.0.3:9200")

es_client = Elasticsearch("http://172.18.0.4:9200")

def create_index(index_name, index_type):
    """Elasticsearch 인덱스를 생성하는 함수. index_type에 따라 서로 다른 구조로 설정."""
    print("\n===== create_index 함수 호출 =====")

    index_name = f"test-{index_name}"

    # index_type별로 필드 설정
    if index_type == "api":
        index_settings = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 1
            },
            "mappings": {
                "properties": {
                    "request_id": {"type": "keyword"},
                    "user_id": {"type": "keyword"},
                    "endpoint": {"type": "keyword"},
                    "timestamp": {"type": "date"},
                    "input_data": {"type": "object"},
                    "output_data": {"type": "object"},
                    "status": {"type": "keyword"},
                    "elapsed_time": {"type": "float"}
                }
            }
        }
    elif index_type == "service":
        index_settings = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 1
            },
            "mappings": {
                "properties": {
                    "timestamp": {"type": "date"},
                    "service_id": {"type": "keyword"},
                    "request_id": {"type": "keyword"},
                    "service_name": {"type": "keyword"},
                    "input_data": {"type": "object"},
                    "output_data": {"type": "object"},
                    "status": {"type": "keyword"}
                }
            }
        }
    elif index_type == "processor":
        index_settings = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 1
            },
            "mappings": {
                "properties": {
                    "timestamp": {"type": "date"},
                    "processor_id": {"type": "keyword"},
                    "service_id": {"type": "keyword"},
                    "processor_name": {"type": "keyword"},
                    "input_data": {"type": "object"},
                    "output_data": {"type": "object"},
                    "execution_time": {"type": "float"}
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
    """
    인덱스의 모든 데이터를 삭제하는 함수
    """
    try:
        # 먼저 전체 문서 수를 확인
        count = es_client.count(index=index_name)['count']
        if count == 0:
            print(f"인덱스 '{index_name}'에 삭제할 데이터가 없습니다.")
            return

        # 모든 문서를 가져와서 삭제
        response = es_client.search(
            index=index_name,
            body={
                "query": {"match_all": {}},
                "size": count
            }
        )

        deleted_count = 0
        for hit in response['hits']['hits']:
            es_client.delete(index=index_name, id=hit['_id'])
            deleted_count += 1
            print(f"문서 ID '{hit['_id']}'가 삭제되었습니다.")

        # 인덱스 새로고침
        es_client.indices.refresh(index=index_name)
        
        # 삭제 확인
        remaining_count = es_client.count(index=index_name)['count']
        print(f"\n=== 삭제 결과 ===")
        print(f"삭제된 문서 수: {deleted_count}")
        print(f"남은 문서 수: {remaining_count}")
        
        if remaining_count == 0:
            print(f"인덱스 '{index_name}'의 모든 데이터가 성공적으로 삭제되었습니다.")
        else:
            print(f"주의: {remaining_count}개의 문서가 여전히 남아있습니다.")

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
    

if __name__ == '__main__':
    # 사용 예시
    create_index("service-usage-logs", "service")

    # 샘플 데이터 저장, 읽기
    sample_doc = {
        "timestamp": datetime.now(timezone(timedelta(hours=9))),
        "service_id": "service-123",
        "request_id": "request-456",
        "service_name": "Test Service",
        "input_data": {"my_input":"샘플 입력 데이터"},
        "output_data": {"my_output":"샘플 출력 데이터"},
        "status": "SUCCESS"
    }
    save_sample_data("service-usage-logs", sample_doc)
    read_all_data("service-usage-logs")

    es_client.update("service-usage-logs", id="DCR3jJQBZYL6MInvMrOs", \
                     body={"doc":{
                         "status": "Fail"
                     }

    })
    es_client.indices.refresh()

    # 샘플 데이터 삭제
    print('========delete all data===========')
    #delete_all_data("service-usage-logs")
    read_all_data("service-usage-logs")


