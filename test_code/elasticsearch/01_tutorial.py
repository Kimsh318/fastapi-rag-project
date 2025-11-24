from elasticsearch import Elasticsearch

# Elasticsearch 클라이언트 생성 (localhost:9200에 연결)
es = Elasticsearch("http://172.18.0.4:9200")

# 인덱스 이름 지정
index_name = "sample_index"

# 샘플 데이터 생성
sample_data = [
    {"title": "Elasticsearch Basics", "author": "Alice", "content": "Learn the basics of Elasticsearch.", "publish_date": "2023-01-01"},
    {"title": "Advanced Elasticsearch", "author": "Bob", "content": "Master advanced Elasticsearch techniques.", "publish_date": "2023-02-15"},
    {"title": "Python for Data Science", "author": "Charlie", "content": "An introduction to Python for data science.", "publish_date": "2023-03-10"},
]

# 1. 인덱스 생성
def create_index(index_name):
    print("\n===== create_index 함수 호출 =====")
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
    
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=index_settings)
        print(f"인덱스 '{index_name}'가 생성되었습니다.")
    else:
        print(f"인덱스 '{index_name}'가 이미 존재합니다.")
    print("===== create_index 함수 종료 =====")

# 2. 인덱스 생성 여부 확인
def check_index_exists(index_name):
    print("\n===== check_index_exists 함수 호출 =====")
    if es.indices.exists(index=index_name):
        print(f"인덱스 '{index_name}'가 존재합니다.")
    else:
        print(f"인덱스 '{index_name}'가 존재하지 않습니다.")
    print("===== check_index_exists 함수 종료 =====")

# 3. 데이터 저장
def store_data(index_name, data):
    print("\n===== store_data 함수 호출 =====")
    for i, doc in enumerate(data):
        es.index(index=index_name, id=i+1, body=doc)
        print(f"문서 {i+1}이(가) 인덱스 '{index_name}'에 저장되었습니다.")
    print("===== store_data 함수 종료 =====")

# 4. 데이터 조회 (모든 문서 조회)
def retrieve_all_data(index_name):
    print("\n===== retrieve_all_data 함수 호출 =====")
    try:
        response = es.search(index=index_name, body={"query": {"match_all": {}}})
        print("조회한 문서들:")
        for hit in response["hits"]["hits"]:
            print(hit["_source"])
    except NotFoundError:
        print(f"인덱스 '{index_name}'가 존재하지 않습니다.")
    print("===== retrieve_all_data 함수 종료 =====")

# 5. 데이터 수정
def update_data(index_name, doc_id, field, new_value):
    print("\n===== update_data 함수 호출 =====")
    try:
        es.update(index=index_name, id=doc_id, body={"doc": {field: new_value}})
        print(f"문서 ID {doc_id}의 '{field}' 필드가 '{new_value}'(으)로 수정되었습니다.")
    except NotFoundError:
        print(f"문서 ID {doc_id}를 찾을 수 없습니다.")
    print("===== update_data 함수 종료 =====")

# 6. 인덱스 삭제
def delete_index(index_name):
    print("\n===== delete_index 함수 호출 =====")
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
        print(f"인덱스 '{index_name}'가 삭제되었습니다.")
    else:
        print(f"인덱스 '{index_name}'가 존재하지 않아 삭제할 수 없습니다.")
    print("===== delete_index 함수 종료 =====")

# 각 함수 실행
create_index(index_name)                   # 인덱스 생성
check_index_exists(index_name)              # 인덱스 생성 여부 확인
store_data(index_name, sample_data)         # 샘플 데이터 저장
retrieve_all_data(index_name)               # 데이터 조회
update_data(index_name, doc_id=1, field="title", new_value="Elasticsearch for Beginners")  # 데이터 수정
retrieve_all_data(index_name)               # 데이터 수정 확인을 위해 다시 조회
delete_index(index_name)                    # 인덱스 삭제
check_index_exists(index_name)              # 인덱스 삭제 여부 확인