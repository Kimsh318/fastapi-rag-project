import re

# Base QueryRefine Processor
class BaseQueryRefineProcessor:
    def __init__(self):
        # 제거하고 싶은 한글 자모음 패턴
        self.bad_pattern_hangul = '[ㄱ-ㅎㅏ-ㅣ]'

        # 제거하고 싶은 특수문자 패턴
        self.bad_pattern_symbols = r'[!@#\$%\^&\*\(\)\-_=\+\[\]\{\};:\'",<>\./\?\\\|`~]+$'

    def process_query(self, query: str) -> str:
        raise NotImplementedError("이 메서드는 하위 클래스에서 구현되어야 합니다.")


# Prototype 환경의 QueryRefine Processor
class PrototypeQueryRefineProcessor(BaseQueryRefineProcessor):
    def process_query(self, query: str) -> str:
        # 한글 자모음 제거
        query = re.sub(self.bad_pattern_hangul, '', query)

        # 특수문자만 있을 경우 특수문자 제거
        if re.match(self.bad_pattern_symbols, query):
            query = re.sub(self.bad_pattern_symbols, '', query)

        return query.strip()


# PrototypeQueryRefineProcessor로 쿼리 처리 테스트
processor = PrototypeQueryRefineProcessor()
query = "!!!!rksekkfkk!!!"
processed_query = processor.process_query(query)

print(f"Original Query: {query}")
print(f"Processed Query: {processed_query}")

query = "!안녕ㅇㅇㅇㅇ"
processed_query = processor.process_query(query)

print(f"Original Query: {query}")
print(f"Processed Query: {processed_query}")


query = "!안녕!!!"
processed_query = processor.process_query(query)

print(f"Original Query: {query}")
print(f"Processed Query: {processed_query}")


query = "!!!!"
processed_query = processor.process_query(query)

print(f"Original Query: {query}")
print(f"Processed Query: {processed_query}")