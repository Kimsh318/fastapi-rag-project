from elasticsearch import AsyncElasticsearch, ConnectionError

from app.core.config import settings

class CustomESClient:
    def __init__(self, main_host, sub_host):
        # main, sub es client 초기화
        self.main_client = AsyncElasticsearch(main_host, http_auth=("kdb", "kdbAi1234!"))
        self.sub_client = AsyncElasticsearch(sub_host, http_auth=("kdb", "kdbAi1234!")) # main_client 연결 실패시, 사용되는 client

    class Indices:
        # Elasticsearch의 analyze API를 호출하기 위한 클래스
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
        # Indices 객체를 반환하여 es_client.indices.analyze 메소드를 사용할 수 있도록함
        return self.Indices(self.main_client, self.sub_client)

    async def mget(self, body, index):
        # doc_id에 해당하는 문서 정보를 검색하는 메소드
        try:
            return await self.main_client.mget(body=body, index=index)
        except ConnectionError:
            return await self.sub_client.mget(body=body, index=index)

    async def search(self, index, body):
        # 질의의 키워드가 포함된 문서들을 검색하는 메소드
        try:
            return await self.main_client.search(index=index, body=body)
        except ConnectionError:
            return await self.sub_client.search(index=index, body=body)

    async def index(self, index, body):
        # ES index에 데이터를 저장하는 메소드
        try:
            return await self.main_client.index(index=index, body=body)
        except ConnectionError:
            return await self.sub_client.index(index=index, body=body)

    async def close(self):
        # ES Client들을 안전하게 연결종료하는 메소드
        try:
            await self.main_client.close()
            await self.sub_client.close()
        except Exception as e:
            print(f"Error while closing es clients : {e}")


def get_es_client():
    if settings.APP_ENVIRONMENT == 'prototype':
        return ESClient(settings.ES_API_HOST)
    elif settings.APP_ENVIRONMENT == 'development':
        return AsyncElasticsearch(settings.ES_API_HOST)
    elif settings.APP_ENVIRONMENT == 'production':
        return CustomESClient(main_host=settings.ES_API_HOST, sub_host=settings.ES_API_SUB_HOST)