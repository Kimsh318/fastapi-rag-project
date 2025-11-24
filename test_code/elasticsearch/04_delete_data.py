from elasticsearch import Elasticsearch, NotFoundError
from datetime import datetime, timezone, timedelta
import time


# Elasticsearch 클라이언트 초기화
es_client = Elasticsearch("http://172.18.0.3:9200")


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

def delete_all_data_alternative(index_name):
    try:
        response = es_client.delete_by_query(
            index=index_name,
            body={"query": {"match_all": {}}}
        )
        print(f"삭제된 문서 수: {response['deleted']}")
        
        es_client.indices.refresh(index=index_name)
    except Exception as e:
        print(f"데이터 삭제 중 오류 발생: {e}")



if __name__ == '__main__':
    index = "test-service-usage-logs"
    
    # 샘플 데이터 추가
    print("\n=== 서비스 로그 인덱스 생성 및 데이터 추가 ===")
    service_sample = {
        "timestamp": datetime.now(timezone(timedelta(hours=9))),
        "service_id": "svc-001",
        "request_id": "req-001",
        "service_name": "Image Analysis Service",
        "input_data": "서비스 입력 데이터 예시",
        "output_data": "서비스 처리 결과 데이터",
        "status": "SUCCESS"
    }

    for index in ["api-usage-logs", "service-usage-logs", "processor-usage-logs"]:
        # 1. delete 구현 방법 1 : 모든 문서 검색 후 1개씩 결과 불러와서 삭제
        save_sample_data(index, service_sample)
        read_all_data(index)
        delete_all_data(index)
        read_all_data(index)

        # 2. delete 구현 방법 2 : 검색과 삭제를 한번에 수행
        save_sample_data(index, service_sample)
        read_all_data(index)
        delete_all_data_alternative(index)
        read_all_data(index)



