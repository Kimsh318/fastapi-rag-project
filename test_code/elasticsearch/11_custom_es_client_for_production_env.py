import asyncio
from elasticsearch import AsyncElasticsearch, ConnectionError


class CustomESClient:
    def __init__(self, main_host, sub_host):
        self.main_client = AsyncElasticsearch(main_host, http_auth=("kdb", "kdbAi1234!"))
        self.sub_client = AsyncElasticsearch(sub_host, http_auth=("kdb", "kdbAi1234!"))

    class Indices:
        def __init__(self, main_client, sub_client):
            self.main_client = main_client
            self.sub_client = sub_client

        async def analyze(self, index, body):
            try:
                return await self.main_client.indices.analyze(index=index, body=body)
            except ConnectionError:
                return await self.sub_client.indices.analyze(index=index, body=body)

    @property
    def indices(self):
        return self.Indices(self.main_client, self.sub_client)

    async def mget(self, body, index):
        try:
            return await self.main_client.mget(body=body, index=index)
        except ConnectionError:
            return await self.sub_client.mget(body=body, index=index)

    async def search(self, index, body):
        try:
            return await self.main_client.search(index=index, body=body)
        except ConnectionError:
            return await self.sub_client.search(index=index, body=body)

    async def index(self, index, body):
        try:
            return await self.main_client.index(index=index, body=body)
        except ConnectionError:
            return await self.sub_client.index(index=index, body=body)

    async def close(self):
        try:
            await self.main_client.close()
            await self.sub_client.close()
        except Exception as e:
            print(f"Error while closing es clients : {e}")
            

async def test_custom_es_client():
    # Elasticsearch 서버의 메인 및 서브 호스트 URL
    main_host = 'http://10.6.40.79:32110'  # 실제 메인 Elasticsearch 서버 URL : 서빙2번 지정해서 테스트
    sub_host =  'http://10.6.40.78:32110'   # 실제 서브 Elasticsearch 서버 URL : 서빙1번 지정해서 테스트

    # CustomESClient 인스턴스 생성
    es_client = CustomESClient(main_host, sub_host)

    # 테스트할 인덱스와 쿼리
    test_index = 'doc_db_index_for_lexical_search'
    test_body = {
        "analyzer": "kdb_nori_analyzer",
        "text": "테스트용 쿼리 입니다."
    }

    # analyze 메서드 호출
    try:
        response = await es_client.indices.analyze(index=test_index, body=test_body)
        print("Analyze Response:", response)
    except Exception as e:
        print("Error during analyze:", e)

    # search 메서드 호출
    query_text = "기한전상환수수료의 정의 궁금해"  # 검색할 쿼리 텍스트
    top_k = 5  # 상위 k개의 결과를 가져옴
    search_body = {
        "query": {
            "match": {
                # "query": query_text,
                # "fields": ["chunk_context"],
                "chunk_context": query_text  # 검색할 필드와 쿼리 텍스트
            }
        },
        "size": top_k  # 상위 k개의 결과
    }
    try:
        response = await es_client.search(index=test_index, body=search_body)
        hits_count = len(response['hits']['hits'])
        print("Search found: ", hits_count)
        #print(response)
    except Exception as e:
        print("Error during search:", e)

    # mget 호출
    doc_ids = ["70b43fa3-270f-4e7a-81d9-8d3a0003290e", "3c0c401f-6275-49ef-be50-6b8ab30731fc"]
    try:
        response = await es_client.mget(body={"ids":doc_ids}, index=test_index)
        docs = response['docs']
        for doc in docs:
            print(f"'--------\nDoc ID: {doc['_id']}, Found: {doc['_source']['chunk_context'][:20]}")
    except Exception as e:
        print("Error during mget:", e)    
    finally:
        await es_client.close()

if __name__ == '__main__':
    asyncio.run(test_custom_es_client())

