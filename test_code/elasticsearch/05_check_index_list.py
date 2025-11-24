from elasticsearch import Elasticsearch

# Elasticsearch 클라이언트 초기화
es_client = Elasticsearch("http://172.18.0.3:9200")

def list_indices():
    """현재 Elasticsearch 클러스터에 존재하는 모든 인덱스를 나열하는 함수."""
    try:
        indices = es_client.indices.get_alias("*")
        print("\n=== 현재 인덱스 목록 ===")
        for index in indices:
            print(index)
    except Exception as e:
        print(f"인덱스 조회 중 오류 발생: {e}")

if __name__ == '__main__':
    list_indices()