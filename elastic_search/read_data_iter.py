from elasticsearch import Elasticsearch, NotFoundError
from datetime import datetime, timezone, timedelta
import time

# Elasticsearch 클라이언트 초기화
# es_client = Elasticsearch("http://172.18.0.5:9200") # prototype 환경
es_client = Elasticsearch("http://127.0.0.1:9200") # development 환경
def read_all_data(index_name):
    """
    인덱스의 모든 데이터를 조회하는 함수
    """
    try:
        # Scroll API를 사용하여 대량의 데이터를 가져오기
        page_size = 1000  # 한 번에 가져올 문서 수
        response = es_client.search(
            index=index_name,
            body={
                "query": {"match_all": {}}
            },
            scroll='2m',  # scroll 유지 시간
            size=page_size
        )
       
        scroll_id = response['_scroll_id']
        hits = response['hits']['hits']
       
        while hits:
            for hit in hits:
                print(f"Document ID: {hit['_id']}")
                print(f"Source: {hit['_source']}")
                print("---")
           
            # 다음 페이지의 데이터를 가져오기
            response = es_client.scroll(scroll_id=scroll_id, scroll='2m')
            scroll_id = response['_scroll_id']
            hits = response['hits']['hits']
       
        print("모든 데이터를 성공적으로 조회했습니다.")
    except Exception as e:
        print(f"데이터 조회 중 오류 발생: {e}")
 

if __name__ == '__main__':

    # index 내 모든 데이터 확인
    print('======== api-usage-data ==========')
    read_all_data("api-usage-logs")


    # print('======== service-usage-data ==========')
    # read_all_data("service-usage-logs")


    # print('======== processor-usage-data ==========')
    # read_all_data("processor-usage-logs")