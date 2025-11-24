from elasticsearch import Elasticsearch, NotFoundError
from datetime import datetime, timezone, timedelta
import time

# Elasticsearch 클라이언트 초기화
es_client = Elasticsearch("http://172.18.0.5:9200")

def read_all_data(index_name):
    """
    인덱스의 모든 데이터를 조회하고, 각 endpoint에 대한 평균 elapsed_time을 계산하는 함수
    """
    try:
        # 인덱스의 전체 문서 수를 확인
        count = es_client.count(index=index_name)['count']

        print(f"\n=== 전체 데이터 수: {count} ===")


        # 검색 쿼리 정의
        query = {
            "query": {
                "term": {
                    "endpoint": "hybrid_search"
                }
            },
            "_source": ["elapsed_time"],
            "size": 10000  # 최대 10,000개 결과 반환
        }

        # Elasticsearch에서 데이터 검색
        response = es_client.search(index=index_name, body=query)

        # elapsed_time 값들 리스트로 추출
        elapsed_times = [hit["_source"]["elapsed_time"] for hit in response["hits"]["hits"]]
        print(response)
        # 결과 출력
        print("총 문서 수:", response["hits"]["total"]["value"])
        print("평균 경과 시간:", sum(elapsed_times) / len(elapsed_times))
        print("elapsed_time 값들:", elapsed_times)


        # agg_query = {
        #     "query": {
        #         "term": {
        #         "endpoint": "hybrid_search_sync"
        #         }
        #     },
        #     "aggs": {
        #         "avg_elapsed_time": {
        #         "avg": {
        #             "field": "elapsed_time"
        #         }
        #         }
        #     }
        # }
        # # Elasticsearch에서 데이터 검색
        # response = es_client.search(index=index_name, body=agg_query)

        # # 결과 출력
        # print("총 문서 수:", response["hits"]["total"]["value"])
        # print("평균 경과 시간:", response["aggregations"]["avg_elapsed_time"]["value"])



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