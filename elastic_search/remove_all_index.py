# tmp_test/elasticsearch/06_delete_indices.py

from elasticsearch import Elasticsearch

# Elasticsearch 클라이언트 초기화
# es_client = Elasticsearch("http://172.18.0.5:9200") # prototype 환경
# es_client = Elasticsearch("http://127.0.0.1:9200") # development 환경
es_client = Elasticsearch("http://127.0.0.1:9200",  http_auth=("kdb", "kdbAi1234!")) # production 환경


def delete_index(index_name):
    """지정된 인덱스를 삭제하는 함수."""
    try:
        if es_client.indices.exists(index=index_name):
            es_client.indices.delete(index=index_name)
            print(f"인덱스 '{index_name}'가 삭제되었습니다.")
        else:
            print(f"인덱스 '{index_name}'가 존재하지 않습니다.")
    except Exception as e:
        print(f"인덱스 삭제 중 오류 발생: {e}")

if __name__ == '__main__':
    # 삭제할 인덱스 목록
    indices_to_delete = [
        "http-middleware-logs",
        "api-usage-logs",
        "service-usage-logs",
        "processor-usage-logs",
        "api-error-logs",
        "etc-func-logs",
        "func-usage-logs",
        "elapsed-time-logs",
        "app-metric-logs",  # "app-metric-logs"
    ]
    for index in indices_to_delete:
        delete_index(index)