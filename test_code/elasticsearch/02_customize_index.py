from elasticsearch import Elasticsearch

# Elasticsearch 클라이언트 생성
es = Elasticsearch("http://172.18.0.4:9200")

# 인덱스 이름
index_name = "my_index"

# 인덱스 설정 및 매핑
index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 1
    },
    "mappings": {
        "properties": {
            "title": {"type": "text"},
            "author": {"type": "keyword"},
            "content": {"type": "text"},
            "publish_date": {"type": "date"}
        }
    }
}

# 인덱스 생성
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name, body=index_settings)
    print(f"인덱스 '{index_name}'가 생성되었습니다.")
else:
    print(f"인덱스 '{index_name}'가 이미 존재합니다.")

# 생성한 인덱스의 정보 얻기
print("\n===== 생성된 인덱스 정보 =====")

# 인덱스 매핑 정보
mapping = es.indices.get_mapping(index=index_name)
print("매핑 정보:", mapping)

# 인덱스 설정 정보
settings = es.indices.get_settings(index=index_name)
print("설정 정보:", settings)

# 인덱스 상태 및 통계 정보
indices_status = es.cat.indices(index=index_name, v=True)
print("상태 및 통계 정보:\n", indices_status)

# 인덱스 삭제
if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)
    print(f"\n인덱스 '{index_name}'가 삭제되었습니다.")
else:
    print(f"\n인덱스 '{index_name}'가 존재하지 않아 삭제할 수 없습니다.")
