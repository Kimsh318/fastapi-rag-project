# 기존 gradio 코드의 query_refine 함수를 리팩토링

import re

def cleansing_query(query: str) -> str:
    bad_pattern_english = '^[A-Za-z\s]+$'  # 영문자와 띄어쓰기만 있는 경우를 확인하는 패턴으로 변경
    bad_pattern_hangul = '[ㄱ-ㅎㅏ-ㅣ\s]+$'  # 한글 자음과 모음, 띄어쓰기를 포함하여 제거
    bad_pattern_symbols = r'[!@#\$%\^&\*\(\)\-_=\+\[\]\{\};:\'",<>\./\?\\\|`~]+$' #만약 사용자 질의가 특수문자만으로 구성될 경우, 해당 특수문자를 제거

    query = re.sub(bad_pattern_hangul, '', query)
    query = re.sub(bad_pattern_symbols, '', query)
    if re.match(bad_pattern_hangul, query):
        query = ''  # 자음과 모음, 띄어쓰기만 있는 경우 쿼리를 비웁니다.
    if re.match(bad_pattern_english, query):
        query = ''  # 영문자만 있는 경우 쿼리를 비웁니다.

    return {'cleansed_query': query.strip()}



if __name__ == '__main__':
    # 정제되어 공백 ''이 출력되는 샘플 데이터
    empty_samples = [
        '   ',
        'abcde',
        'ㅋㅋㅋㅋㅋ',
        'ㅏㅣㅓㅜㅡ',
        '!@#$%^&*()',
        'asbas dsgasge',
        'ㄱㄴㄷ ㄴㄷㅎ',
    ]

    # 정제되지 않고 일부 내용이 남는 샘플 데이터
    remaining_samples = [
        '안녕하세요123',
        'hello123',
        '정상적인 문장입니다.',
        '12345',
        'mixed문자열123'
    ]
    print("=== 정제되어 공백 ''이 출력되는 샘플 테스트 ===")
    for sample in empty_samples:
        cleansed = cleansing_query(sample)['cleansed_query']
        print(f"원본: '{sample}'\n정제 후: '{cleansed}'\n")

    print("=== 정제되지 않고 일부 내용이 남는 샘플 테스트 ===")
    for sample in remaining_samples:
        cleansed = cleansing_query(sample)['cleansed_query']
        print(f"원본: '{sample}'\n정제 후: '{cleansed}'\n")
