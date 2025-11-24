from elasticsearch import Elasticsearch, NotFoundError
from datetime import datetime, timezone, timedelta
import time

# Elasticsearch 클라이언트 초기화
es_client = Elasticsearch("http://172.18.0.5:9200")



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



if __name__ == '__main__':

    # index 내 모든 데이터 확인
    print('======== api-usage-data ==========')
    read_all_data("test-api-usage-logs")


    print('======== service-usage-data ==========')
    read_all_data("test-service-usage-logs")


    print('======== processor-usage-data ==========')
    read_all_data("test-processor-usage-logs")